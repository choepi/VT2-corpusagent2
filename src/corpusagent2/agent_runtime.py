from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid
from typing import Any, Callable

from .agent_backends import (
    HybridSearchBackend,
    InMemoryWorkingSetStore,
    LocalSearchBackend,
    OpenSearchBackend,
    OpenSearchConfig,
    PostgresWorkingSetStore,
    WorkingSetStore,
    save_agent_manifest,
)
from .agent_capabilities import AgentExecutionContext, _infer_market_ticker_from_text, build_agent_registry
from .agent_executor import AgentExecutionSnapshot, AsyncPlanExecutor
from .agent_models import (
    AgentFailure,
    AgentPlanDAG,
    AgentPlanNode,
    AgentRunManifest,
    AgentRunState,
    LiveRunStatus,
    PlannerAction,
)
from .agent_policy import normalize_question_text, rejection_reason_for_question
from .app_config import load_project_configuration
from .llm_provider import LLMClient, LLMProviderConfig, OpenAICompatibleLLMClient
from .retrieval_budgeting import infer_requested_output_limit, infer_retrieval_budget
from .retrieval import dense_retrieval_enabled, pg_dsn_from_env, pg_table_from_env
from .run_manifest import FinalAnswerPayload
from .runtime_context import CorpusRuntime
from .seed import runtime_device_report
from .tool_registry import ToolRegistry
from .python_runner_service import DockerPythonRunnerService
from . import recovery_advisor as _llm_recovery_advisor

TERMINAL_RUN_STATUSES = {"completed", "partial", "failed", "rejected", "needs_clarification", "aborted"}


def resolve_working_store(
    *,
    doc_lookup: dict[str, Any] | None = None,
    doc_lookup_factory: "Callable[[], dict[str, Any]] | None" = None,
    require_backend_services: bool = False,
) -> WorkingSetStore:
    """Pick a WorkingSetStore based on env vars.

    The ``doc_lookup`` table is only consulted when we actually build an
    ``InMemoryWorkingSetStore``. Pass ``doc_lookup_factory`` instead of
    ``doc_lookup`` if loading the table is expensive (e.g. reads parquet
    from disk) so the cost is paid only when needed.

    CORPUSAGENT2_WORKINGSET_BACKEND:
      - ``memory``   force InMemoryWorkingSetStore (archive mode)
      - ``postgres`` require a configured DSN; raise if missing
      - ``auto``     try Postgres if DSN present, else fall back to memory
      - unset (default) keep historical behaviour: use Postgres if DSN
        present, else memory (unless ``require_backend_services`` is set).
    """
    def _materialize_doc_lookup() -> dict[str, Any]:
        if doc_lookup is not None:
            return doc_lookup
        if doc_lookup_factory is not None:
            try:
                return doc_lookup_factory() or {}
            except Exception:
                return {}
        return {}

    backend = os.getenv("CORPUSAGENT2_WORKINGSET_BACKEND", "").strip().lower()
    if backend == "memory":
        store = InMemoryWorkingSetStore()
        materialized = _materialize_doc_lookup()
        if materialized:
            store.document_lookup.update(materialized)
        return store
    if backend not in {"", "postgres", "auto"}:
        raise ValueError(
            f"Unknown CORPUSAGENT2_WORKINGSET_BACKEND={backend!r}; expected 'postgres', 'memory', or 'auto'."
        )
    try:
        dsn = pg_dsn_from_env(required=False)
    except Exception:
        dsn = ""
    if dsn:
        return PostgresWorkingSetStore(dsn=dsn, documents_table=pg_table_from_env(default="article_corpus"))
    if backend == "postgres":
        raise RuntimeError(
            "CORPUSAGENT2_WORKINGSET_BACKEND=postgres but CORPUSAGENT2_PG_DSN is not configured."
        )
    if require_backend_services and backend != "auto":
        raise RuntimeError("Postgres working store is required but CORPUSAGENT2_PG_DSN is not configured.")
    store = InMemoryWorkingSetStore()
    materialized = _materialize_doc_lookup()
    if materialized:
        store.document_lookup.update(materialized)
    return store

TOOL_USAGE_CATEGORIES: tuple[tuple[str, set[str]], ...] = (
    ("Retrieval and working sets", {"db_search", "sql_query_search", "fetch_documents", "create_working_set", "filter_working_set"}),
    ("Temporal and structured series", {"time_series_aggregate", "change_point_detect", "burst_detect", "join_external_series"}),
    ("Language preprocessing", {"lang_id", "clean_normalize", "sentence_split", "tokenize", "lemmatize", "pos_morph", "mwt_expand"}),
    ("Entities and syntax", {"ner", "entity_link", "noun_chunks", "dependency_parse", "extract_svo_triples"}),
    ("Lexical topics and readability", {"extract_keyterms", "extract_ngrams", "topic_model", "lexical_diversity", "readability_stats", "extract_acronyms"}),
    ("Sentiment classification and claims", {"sentiment", "text_classify", "claim_span_extract", "claim_strength_score"}),
    ("Embeddings and similarity", {"doc_embeddings", "word_embeddings", "similarity_index", "similarity_pairwise"}),
    ("Evidence and quotes", {"build_evidence_table", "quote_extract", "quote_attribute"}),
    ("Sandbox and plotting", {"python_runner", "plot_artifact"}),
)

FRAMEWORK_BACKBONE_CAPABILITIES = {
    "db_search",
    "fetch_documents",
    "create_working_set",
    "filter_working_set",
    "plot_artifact",
    "python_runner",
    "time_series_aggregate",
}

OUT_OF_CORPUS_MODEL_ANSWER_FLAG = "MODEL-ONLY / OUT-OF-CORPUS ANSWER (NOT CORPUS-GROUNDED)"


def _tool_usage_category(capabilities: list[str]) -> str:
    capability_set = {str(item) for item in capabilities if str(item).strip()}
    for category, members in TOOL_USAGE_CATEGORIES:
        if capability_set & members:
            return category
    return "Other"


def _tool_usage_role(capabilities: list[str]) -> str:
    capability_set = {str(item) for item in capabilities if str(item).strip()}
    return "framework backbone" if capability_set & FRAMEWORK_BACKBONE_CAPABILITIES else "question-specific"


def _tool_usage_reason(
    *,
    capabilities: list[str],
    completed_node_count: int,
    planned_node_count: int,
    planned_unresolved_count: int,
) -> str:
    if completed_node_count > 0:
        return "Used by at least one historical run."
    if planned_node_count > 0:
        if planned_unresolved_count > 0:
            return (
                "Planned but not completed in at least one saved run. Usually this means the run was still active, "
                "was aborted, hit an upstream dependency failure, or came from an older manifest before pending nodes "
                "were explicitly marked skipped."
            )
        return "Planned historically, but node records show it did not complete."
    if _tool_usage_role(capabilities) == "framework backbone":
        return "Framework/backbone tool; absence means historical runs did not reach that stage or used an alternate backend path."
    return "Question-specific tool; selected only when the planner needs that analysis type."


SOURCE_CANDIDATE_LEADING_NOISE = {
    "a",
    "after",
    "an",
    "and",
    "before",
    "between",
    "compare",
    "compared",
    "comparison",
    "did",
    "difference",
    "differences",
    "do",
    "does",
    "how",
    "media",
    "newspaper",
    "newspapers",
    "outlet",
    "outlets",
    "press",
    "source",
    "sources",
    "the",
    "versus",
    "vs",
    "what",
    "which",
}
SOURCE_CANDIDATE_TRAILING_NOISE = {
    "about",
    "after",
    "and",
    "announcement",
    "announcements",
    "before",
    "compare",
    "compared",
    "comparison",
    "cover",
    "coverage",
    "covers",
    "differ",
    "different",
    "differently",
    "during",
    "event",
    "explain",
    "explained",
    "explains",
    "frame",
    "framed",
    "frames",
    "in",
    "media",
    "newspaper",
    "newspapers",
    "news",
    "on",
    "only",
    "outlet",
    "outlets",
    "over",
    "period",
    "phase",
    "press",
    "report",
    "reported",
    "reports",
    "source",
    "sources",
    "window",
    "write",
    "writes",
    "wrote",
}
SOURCE_CANDIDATE_GENERIC_VALUES = {
    "after announcement",
    "after the announcement",
    "announcement",
    "before announcement",
    "before the announcement",
    "event",
    "media",
    "news",
    "newspaper",
    "newspapers",
    "outlet",
    "outlets",
    "press",
    "the announcement",
    "source",
    "sources",
}
TOPIC_QUERY_EXPANSIONS = (
    {
        "triggers": ("football", "soccer", "fussball", "fußball"),
        "query": "football OR soccer OR fussball OR fußball",
    },
    {
        "triggers": ("climate", "klima", "climat"),
        "query": "climate OR klima OR climat OR klimawandel OR rechauffement OR réchauffement",
    },
    {
        "triggers": ("protest", "protests", "protester", "protesters", "demonstration", "demonstrators", "unrest"),
        "query": "protest OR protests OR protesters OR protestors OR demonstrators OR demonstrations OR unrest OR police OR \"law enforcement\"",
    },
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _tool_call_signature(tool_name: str, inputs: dict[str, Any]) -> str:
    if not tool_name:
        return ""
    if not inputs:
        return f"{tool_name}()"
    rendered: list[str] = []
    for key, value in inputs.items():
        compact = json.dumps(value, ensure_ascii=True, default=str)
        if len(compact) > 100:
            compact = compact[:97] + "..."
        rendered.append(f"{key}={compact}")
    return f"{tool_name}({', '.join(rendered)})"


def _candidate_payload_rows(snapshot: AgentExecutionSnapshot) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in snapshot.node_results.values():
        payload = result.payload if isinstance(result.payload, dict) else {}
        for key in ("documents", "results", "rows", "evidence_items"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        rows.append(dict(item))
                if rows:
                    return rows[:50]
    return [dict(item) for item in snapshot.selected_docs[:50] if isinstance(item, dict)]


def _diagnostic_text(value: Any, *, max_chars: int = 600) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _diagnostic_list(value: Any, *, max_items: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = _diagnostic_text(item, max_chars=240)
        if text:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def _diagnostic_node_payload(record: Any) -> dict[str, Any]:
    payload = {
        "node_id": str(getattr(record, "node_id", "") or ""),
        "capability": str(getattr(record, "capability", "") or ""),
        "status": str(getattr(record, "status", "") or ""),
        "tool_name": str(getattr(record, "tool_name", "") or ""),
        "provider": str(getattr(record, "provider", "") or ""),
        "attempts": int(getattr(record, "attempts", 0) or 0),
        "error": _diagnostic_text(getattr(record, "error", "")),
        "caveats": _diagnostic_list(getattr(record, "caveats", [])),
        "unsupported_parts": _diagnostic_list(getattr(record, "unsupported_parts", [])),
    }
    return {key: value for key, value in payload.items() if value not in ("", [], 0)}


def _diagnostic_failure_payload(failure: AgentFailure) -> dict[str, Any]:
    return {
        "node_id": failure.node_id,
        "capability": failure.capability,
        "error_type": failure.error_type,
        "message": _diagnostic_text(failure.message),
        "retriable": bool(failure.retriable),
        "details": failure.details,
    }


def _execution_issue_records(snapshot: AgentExecutionSnapshot) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in snapshot.node_records:
        status = str(record.status or "").strip().lower()
        error = str(record.error or "").strip()
        if status == "failed" or (status == "skipped" and error):
            records.append(_diagnostic_node_payload(record))
    return records[:20]


def _execution_needs_diagnostics(snapshot: AgentExecutionSnapshot) -> bool:
    if snapshot.status in {"failed", "partial", "aborted"}:
        return True
    if snapshot.failures:
        return True
    return any(str(record.status or "").strip().lower() == "failed" for record in snapshot.node_records)


def _coerce_diagnostic_string_list(value: Any, *, max_items: int = 6) -> list[str]:
    if isinstance(value, list):
        items = [_diagnostic_text(item, max_chars=300) for item in value]
    elif isinstance(value, str):
        items = [_diagnostic_text(value, max_chars=300)]
    else:
        items = []
    return [item for item in items if item][:max_items]


def _merge_execution_snapshots(
    primary: AgentExecutionSnapshot,
    fallback: AgentExecutionSnapshot,
) -> AgentExecutionSnapshot:
    combined_results = dict(primary.node_results)
    combined_results.update(fallback.node_results)
    combined_failures = list(primary.failures) + list(fallback.failures)
    selected_docs = list(primary.selected_docs) if primary.selected_docs else list(fallback.selected_docs)
    if primary.status == "aborted" or fallback.status == "aborted":
        status = "aborted"
    elif combined_failures:
        status = "partial"
    else:
        status = "completed"
    return AgentExecutionSnapshot(
        node_records=list(primary.node_records) + list(fallback.node_records),
        node_results=combined_results,
        failures=combined_failures,
        provenance_records=list(primary.provenance_records) + list(fallback.provenance_records),
        selected_docs=selected_docs,
        status=status,
    )


def _merge_tool_call_rows(existing_rows: list[dict[str, Any]], payload: dict[str, Any]) -> list[dict[str, Any]]:
    node_id = str(payload.get("node_id", "")).strip()
    event = str(payload.get("event", "")).strip()
    started_at = str(payload.get("started_at_utc", "")).strip()
    entry = None
    if node_id:
        if event == "node_started" and started_at:
            entry = next(
                (
                    row
                    for row in existing_rows
                    if str(row.get("node_id", "")).strip() == node_id
                    and str(row.get("started_at_utc", "")).strip() == started_at
                ),
                None,
            )
        else:
            entry = next((row for row in reversed(existing_rows) if str(row.get("node_id", "")).strip() == node_id), None)
    if entry is None:
        entry = {
            "node_id": node_id,
            "capability": str(payload.get("capability", "")).strip(),
        }
        existing_rows.append(entry)
    for key in (
        "capability",
        "status",
        "tool_name",
        "requested_tool_name",
        "provider",
        "tool_version",
        "model_id",
        "tool_reason",
        "started_at_utc",
        "finished_at_utc",
        "duration_ms",
        "documents_processed",
        "cache_key",
        "error",
    ):
        if payload.get(key) not in (None, ""):
            entry[key] = payload[key]
    if "inputs" in payload:
        entry["inputs"] = dict(payload.get("inputs", {}))
        entry["call_signature"] = _tool_call_signature(str(entry.get("tool_name", "")), entry["inputs"])
    if "dependency_nodes" in payload:
        entry["dependency_nodes"] = list(payload.get("dependency_nodes", []))
    if "summary" in payload:
        entry["summary"] = dict(payload.get("summary", {}))
    if "artifacts" in payload:
        entry["artifacts"] = list(payload.get("artifacts", []))
    if "cache_hit" in payload:
        entry["cache_hit"] = bool(payload.get("cache_hit"))
    return existing_rows


def _runtime_corpus_info(project_root: Path, retrieval_health: dict[str, Any]) -> dict[str, Any]:
    env_name = os.getenv("CORPUSAGENT2_CORPUS_NAME", "").strip()
    hf_dataset = os.getenv("CORPUSAGENT2_HF_DATASET", "").strip()
    hf_config = os.getenv("CORPUSAGENT2_HF_CONFIG", "").strip()
    hf_split = os.getenv("CORPUSAGENT2_HF_SPLIT", "").strip()
    pg_table = pg_table_from_env(default="article_corpus")
    local_source = ""
    source_sha256 = ""

    summary_path = project_root / "outputs" / "stage_ccnews_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        summary = {}
    files = summary.get("files") if isinstance(summary, dict) else None
    if isinstance(files, list) and files:
        first_file = files[0] if isinstance(files[0], dict) else {}
        local_source = str(first_file.get("source") or first_file.get("destination") or "").strip()
        source_sha256 = str(first_file.get("sha256") or "").strip()
        source_name = Path(local_source).name.lower() if local_source else ""
        if not hf_dataset and "cc_news" in source_name:
            hf_dataset = "vblagoje/cc_news"
            hf_split = hf_split or "train"

    name = env_name or hf_dataset or (Path(local_source).name if local_source else pg_table)
    display_parts = [name]
    if hf_config:
        display_parts.append(hf_config)
    if hf_split and hf_dataset:
        display_parts.append(hf_split)
    display_name = " / ".join(part for part in display_parts if part)
    return {
        "name": name,
        "display_name": display_name,
        "hf_dataset": hf_dataset,
        "hf_config": hf_config,
        "hf_split": hf_split,
        "pg_table": pg_table,
        "local_source": local_source,
        "source_sha256": source_sha256,
        "document_count": int(retrieval_health.get("document_count") or 0),
    }


@dataclass(slots=True)
class AgentRuntimeConfig:
    project_root: Path
    outputs_root: Path
    planner_calls_max: int = 6

    @classmethod
    def from_project_root(cls, project_root: Path) -> "AgentRuntimeConfig":
        project_root = project_root.resolve()
        return cls(
            project_root=project_root,
            outputs_root=(project_root / "outputs" / "agent_runtime").resolve(),
        )

class MagicBoxOrchestrator:
    _SEARCH_BACKBONE_CAPABILITIES = {"db_search", "sql_query_search"}
    _DOC_RETRIEVAL_BACKBONE_CAPABILITIES = {
        "create_working_set",
        "filter_working_set",
        "clean_normalize",
        "entity_link",
        "extract_keyterms",
        "topic_model",
        "text_classify",
        "sentiment",
        "ner",
        "tokenize",
        "sentence_split",
        "pos_morph",
        "lemmatize",
        "dependency_parse",
        "extract_svo_triples",
        "noun_chunks",
        "extract_acronyms",
        "quote_extract",
        "quote_attribute",
        "claim_span_extract",
        "claim_strength_score",
        "build_evidence_table",
        "doc_embeddings",
        "word_embeddings",
        "similarity_index",
        "similarity_pairwise",
    }

    def __init__(self, llm_client: LLMClient | None = None, llm_config: LLMProviderConfig | None = None) -> None:
        self.llm_client = llm_client
        self.llm_config = llm_config or LLMProviderConfig.from_env()

    def _record_llm_trace(
        self,
        state: AgentRunState,
        *,
        stage: str,
        trace: dict[str, Any] | None = None,
        used_fallback: bool = False,
        error: str = "",
        note: str = "",
    ) -> None:
        entry = {
            "stage": stage,
            "provider_name": self.llm_config.provider_name if self.llm_client is not None else "heuristic",
            "base_url": self.llm_config.base_url if self.llm_client is not None else "",
            "model": trace.get("model", "") if trace else "",
            "temperature": trace.get("temperature", 0.0) if trace else 0.0,
            "messages": trace.get("messages", []) if trace else [],
            "raw_text": trace.get("raw_text", "") if trace else "",
            "parsed_json": trace.get("parsed_json", {}) if trace else {},
            "used_fallback": used_fallback,
            "error": error,
            "note": note,
        }
        state.llm_traces.append(entry)

    def _planner_payload_is_actionable(self, payload: dict[str, Any]) -> bool:
        if not isinstance(payload, dict) or not payload:
            return False
        action = str(payload.get("action", "")).strip()
        if action:
            if action == "emit_plan_dag":
                dag_payload = payload.get("plan_dag")
                return isinstance(dag_payload, dict) and bool(dag_payload.get("nodes"))
            return True
        dag_payload = payload.get("plan_dag")
        if isinstance(dag_payload, dict) and bool(dag_payload.get("nodes")):
            return True
        if str(payload.get("clarification_question", "")).strip():
            return True
        if str(payload.get("rejection_reason", "")).strip():
            return True
        return False

    def _search_node_is_full_population(self, node: AgentPlanNode) -> bool:
        inputs = dict(node.inputs or {})
        try:
            top_k = int(inputs.get("top_k", 0) or 0)
        except (TypeError, ValueError):
            top_k = 0
        return bool(inputs.get("retrieve_all")) or top_k == 0

    def _search_reuse_terms(self, query: str) -> set[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "by",
            "candidate",
            "for",
            "from",
            "in",
            "not",
            "of",
            "on",
            "or",
            "president",
            "the",
            "to",
            "with",
        }
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", str(query or ""))
            if len(token) >= 3 and token.lower() not in stopwords
        }

    def _search_query_can_filter_from(self, broad_query: str, narrow_query: str) -> bool:
        broad = str(broad_query or "").strip()
        narrow = str(narrow_query or "").strip()
        if not broad or not narrow or broad == narrow:
            return False
        if not re.search(r"\bAND\b", narrow, flags=re.IGNORECASE):
            return False
        broad_terms = self._search_reuse_terms(broad)
        narrow_terms = self._search_reuse_terms(narrow)
        if len(broad_terms) < 1 or len(narrow_terms) <= len(broad_terms):
            return False
        overlap = len(broad_terms & narrow_terms) / max(1, min(len(broad_terms), len(narrow_terms)))
        return overlap >= 0.5

    def _same_search_constraints(self, first: AgentPlanNode, second: AgentPlanNode) -> bool:
        first_inputs = dict(first.inputs or {})
        second_inputs = dict(second.inputs or {})
        for key in ("date_from", "date_to", "year_balance"):
            first_value = str(first_inputs.get(key, "") or "").strip()
            second_value = str(second_inputs.get(key, "") or "").strip()
            if first_value and second_value and first_value != second_value:
                return False
        first_source = "source:" in str(first_inputs.get("query", "")).lower()
        second_source = "source:" in str(second_inputs.get("query", "")).lower()
        return first_source == second_source or not first_source

    def _reuse_subsumed_search_branches(self, nodes: list[AgentPlanNode]) -> list[AgentPlanNode]:
        search_nodes = [
            node
            for node in nodes
            if node.capability in self._SEARCH_BACKBONE_CAPABILITIES and self._search_node_is_full_population(node)
        ]
        if len(search_nodes) < 2:
            return nodes
        replacements: dict[str, str] = {}
        for narrow in search_nodes:
            narrow_query = str(narrow.inputs.get("query", "") if isinstance(narrow.inputs, dict) else "").strip()
            if not narrow_query:
                continue
            broad_candidates = [
                broad
                for broad in search_nodes
                if broad.node_id != narrow.node_id
                and self._same_search_constraints(broad, narrow)
                and self._search_query_can_filter_from(
                    str(broad.inputs.get("query", "") if isinstance(broad.inputs, dict) else ""),
                    narrow_query,
                )
            ]
            if not broad_candidates:
                continue
            broad = min(
                broad_candidates,
                key=lambda item: len(self._search_reuse_terms(str(item.inputs.get("query", "")))),
            )
            replacements[narrow.node_id] = broad.node_id

        if not replacements:
            return nodes
        rewritten: list[AgentPlanNode] = []
        for node in nodes:
            if node.node_id in replacements:
                broad_node_id = replacements[node.node_id]
                inputs = dict(node.inputs)
                filtered_inputs = {
                    "query": str(inputs.get("query", "")).strip(),
                    "source_node_id": broad_node_id,
                    "working_set_source_node_id": broad_node_id,
                }
                for key in ("date_from", "date_to", "batch_size"):
                    if inputs.get(key) not in (None, ""):
                        filtered_inputs[key] = inputs[key]
                rewritten.append(
                    AgentPlanNode(
                        node_id=node.node_id,
                        capability="filter_working_set",
                        tool_name="working_set_filter",
                        inputs=filtered_inputs,
                        depends_on=[broad_node_id],
                        optional=node.optional,
                        cacheable=node.cacheable,
                        description=(
                            node.description
                            or "Filter an existing broad retrieval working set instead of running a second full-corpus search."
                        ),
                    )
                )
                continue
            rewritten.append(node)
        return rewritten

    _SOURCE_REFERENCE_INPUT_KEYS = {
        "source",
        "source_node",
        "source_node_id",
        "table_source",
        "table_from",
        "documents_node",
        "document_source",
        "input_from",
        "series_source",
        "working_set",
        "working_set_from",
        "working_set_id",
        "working_set_source",
        "working_set_source_node_id",
    }
    _EXPENSIVE_LEAF_CAPABILITIES = {
        "build_evidence_table",
        "claim_span_extract",
        "claim_strength_score",
        "dependency_parse",
        "doc_embeddings",
        "entity_link",
        "extract_ngrams",
        "extract_svo_triples",
        "ner",
        "noun_chunks",
        "pos_morph",
        "quote_attribute",
        "quote_extract",
        "similarity_index",
        "similarity_pairwise",
        "word_embeddings",
    }
    _EXPENSIVE_OPTIONAL_CAPABILITIES = _EXPENSIVE_LEAF_CAPABILITIES | {
        "extract_keyterms",
        "plot_artifact",
        "sentiment",
        "topic_model",
    }
    _PYTHON_RUNNER_NATIVE_TASK_HINTS = {
        "actor prominence",
        "actor_prominence",
        "correlation",
        "entity frequency",
        "entity_frequency",
        "numeric correlation",
        "numeric_correlation",
    }

    @classmethod
    def _effective_node_inputs(cls, node: AgentPlanNode) -> dict[str, Any]:
        inputs = dict(node.inputs or {})
        payload = inputs.get("payload")
        if isinstance(payload, dict):
            return {**payload, **{key: value for key, value in inputs.items() if key != "payload"}}
        return inputs

    @classmethod
    def _input_source_references(cls, inputs: dict[str, Any], node_ids: set[str]) -> list[str]:
        references: list[str] = []
        for key, value in inputs.items():
            if key in cls._SOURCE_REFERENCE_INPUT_KEYS and isinstance(value, str) and value in node_ids:
                references.append(value)
            if isinstance(value, dict):
                references.extend(cls._input_source_references(value, node_ids))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        references.extend(cls._input_source_references(item, node_ids))
        return list(dict.fromkeys(references))

    @staticmethod
    def _language_filter_base_key(raw_key: str) -> str:
        key = str(raw_key or "").strip().lower()
        for suffix in ("_contains", "_in", "_equals", "_eq"):
            if key.endswith(suffix):
                return key[: -len(suffix)]
        return key

    @staticmethod
    def _filter_control_metadata_key(raw_key: str) -> bool:
        key = str(raw_key or "").strip().lower()
        if key in {
            "source_node",
            "source_node_id",
            "source_ref",
            "source_result",
            "working_set_node",
            "working_set_source_node",
            "working_set_source_node_id",
        }:
            return True
        return key.endswith(("_source", "_source_node", "_source_node_id", "_source_ref", "_source_result"))

    @classmethod
    def _has_plain_language_filter(cls, inputs: dict[str, Any]) -> bool:
        for container_key in ("filters", "filter", "predicate"):
            filters = inputs.get(container_key)
            if not isinstance(filters, dict):
                continue
            for raw_key, value in filters.items():
                if cls._language_filter_base_key(str(raw_key)) not in {"language", "lang", "lang_id", "language_id"}:
                    continue
                if isinstance(value, dict) and str(value.get("source", "") or "").strip():
                    continue
                return True
        return False

    @classmethod
    def _rewrite_plain_language_filter_inputs(cls, inputs: dict[str, Any], lang_node_id: str) -> tuple[dict[str, Any], bool]:
        changed = False
        rewritten_inputs = dict(inputs)
        payload = rewritten_inputs.get("payload")
        payload_is_dict = isinstance(payload, dict)
        target = dict(payload) if payload_is_dict else rewritten_inputs

        for container_key in ("filters", "filter", "predicate"):
            filters = target.get(container_key)
            if not isinstance(filters, dict):
                continue
            rewritten_filters = dict(filters)
            for raw_key, value in list(filters.items()):
                if cls._filter_control_metadata_key(str(raw_key)):
                    rewritten_filters.pop(raw_key, None)
                    changed = True
                    continue
                base_key = cls._language_filter_base_key(str(raw_key))
                if base_key not in {"language", "lang", "lang_id", "language_id"}:
                    continue
                if isinstance(value, dict) and str(value.get("source", "") or "").strip():
                    continue
                expected_values = value if isinstance(value, list) else [value]
                rewritten_filters[raw_key] = {
                    "source": lang_node_id,
                    "field": "language",
                    "in": expected_values,
                }
                changed = True
            if rewritten_filters != filters:
                target[container_key] = rewritten_filters

        if not changed:
            return inputs, False
        if payload_is_dict:
            rewritten_inputs["payload"] = target
        return rewritten_inputs, True

    @staticmethod
    def _has_executable_python_code(value: Any) -> bool:
        code = str(value or "").strip()
        if not code:
            return False
        try:
            parsed = ast.parse(code, filename="<planned_python_runner>", mode="exec")
        except SyntaxError:
            return False
        for statement in parsed.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                if isinstance(statement.value.value, str):
                    continue
            return True
        return False

    @classmethod
    def _python_runner_is_routable(cls, node: AgentPlanNode) -> bool:
        inputs = cls._effective_node_inputs(node)
        if cls._has_executable_python_code(inputs.get("code")):
            return True
        task_text = " ".join(
            str(inputs.get(key, ""))
            for key in ("task", "task_name", "analysis", "analysis_type")
            if inputs.get(key) is not None
        ).lower()
        return any(hint in task_text for hint in cls._PYTHON_RUNNER_NATIVE_TASK_HINTS)

    @staticmethod
    def _question_requires_capability(question_text: str, capability: str) -> bool:
        lowered = str(question_text or "").lower()
        capability_terms = {
            "build_evidence_table": (
                "evidence table",
                "evidence",
                "noun distribution",
                "noun frequency",
                "distribution of nouns",
                "acted upon",
                "who did what",
                "directly attributed",
                "attributed to",
                "explanation",
                "explanations",
                "recurring terms",
                "abbreviations",
                "connected descriptions",
            ),
            "claim_span_extract": ("claim", "claims", "span", "explanation", "explanations", "reason", "reasons", "why"),
            "claim_strength_score": ("claim", "claims", "verdict", "strength", "explanation", "explanations"),
            "dependency_parse": ("dependency", "syntax", "grammatical", "subject", "verb", "object", "acted upon"),
            "doc_embeddings": ("semantic", "semantically", "similar", "similarity", "related", "same exact wording", "different wording"),
            "entity_link": ("entity link", "entity linking", "actor", "actors", "who", "named people", "institutions", "abbreviations"),
            "extract_acronyms": ("acronym", "acronyms", "abbreviation", "abbreviations"),
            "extract_ngrams": ("ngram", "n-gram", "phrase distribution", "bigram", "trigram"),
            "extract_keyterms": ("keyterm", "key term", "keywords", "dominant terms", "recurring terms", "explanation", "explanations"),
            "extract_svo_triples": ("svo", "subject", "verb", "object", "who did what", "acted upon", "acting"),
            "join_external_series": ("stock", "drawdown", "market", "price", "oil", "equity", "ticker"),
            "ner": ("named entities", "named entity", "actors", "actor", "entities", "named people", "institutions"),
            "noun_chunks": ("noun chunk", "noun phrase", "noun phrases", "syntax", "grammatical", "subject", "verb", "object", "acted upon"),
            "plot_artifact": ("chart", "plot", "graph", "visualize", "visualise", "visualization", "visualisation"),
            "pos_morph": ("part of speech", "pos", "noun distribution", "nouns", "syntax", "grammatical", "subject", "verb", "object", "acted upon"),
            "quote_attribute": ("quote", "quoted", "attribution", "attributed", "directly attributed", "named people", "named institutions", "described as acting"),
            "quote_extract": ("quote", "quoted", "quotation", "said", "according to", "directly attributed", "described as acting"),
            "sentiment": ("sentiment", "tone", "portrayal", "positive", "negative"),
            "similarity_index": ("semantic", "semantically", "similar", "similarity", "related", "same exact wording", "different wording"),
            "similarity_pairwise": ("semantic", "semantically", "similar", "similarity", "related", "same exact wording", "different wording"),
            "topic_model": ("topic", "topics", "framing", "frame", "frames"),
            "word_embeddings": ("semantic", "semantically", "similar", "similarity", "related", "wording", "recurring terms", "terms"),
        }
        return any(term in lowered for term in capability_terms.get(capability, (capability.replace("_", " "),)))

    @staticmethod
    def _downstream_map(nodes: list[AgentPlanNode]) -> dict[str, set[str]]:
        downstream: dict[str, set[str]] = {node.node_id: set() for node in nodes}
        for node in nodes:
            for dep in node.depends_on:
                downstream.setdefault(dep, set()).add(node.node_id)
        return downstream

    def _compile_plan_dag(self, dag: AgentPlanDAG, question_text: str) -> AgentPlanDAG:
        node_ids = {node.node_id for node in dag.nodes}
        compiler_notes: list[str] = []
        nodes_with_inferred_deps: list[AgentPlanNode] = []
        for node in dag.nodes:
            inferred_deps = self._input_source_references(self._effective_node_inputs(node), node_ids)
            depends_on = list(dict.fromkeys([*node.depends_on, *[dep for dep in inferred_deps if dep != node.node_id]]))
            if len(depends_on) != len(node.depends_on):
                compiler_notes.append(
                    f"node {node.node_id}: inferred dependencies {sorted(set(depends_on) - set(node.depends_on))}"
                )
            nodes_with_inferred_deps.append(
                AgentPlanNode(
                    node_id=node.node_id,
                    capability=node.capability,
                    tool_name=node.tool_name,
                    inputs=dict(node.inputs),
                    depends_on=depends_on,
                    optional=node.optional,
                    cacheable=node.cacheable,
                    description=node.description,
                )
            )

        invalid_nodes = {
            node.node_id
            for node in nodes_with_inferred_deps
            if node.capability == "python_runner" and not self._python_runner_is_routable(node)
        }
        for node_id in sorted(invalid_nodes):
            compiler_notes.append(f"node {node_id}: removed invalid natural-language python_runner payload")
        changed = True
        while changed:
            changed = False
            for node in nodes_with_inferred_deps:
                if node.node_id in invalid_nodes:
                    continue
                if any(dep in invalid_nodes for dep in node.depends_on):
                    invalid_nodes.add(node.node_id)
                    compiler_notes.append(f"node {node.node_id}: removed because it depended on invalid node")
                    changed = True

        retained_nodes = [node for node in nodes_with_inferred_deps if node.node_id not in invalid_nodes]
        if not retained_nodes:
            metadata = dict(dag.metadata)
            metadata["compiler_notes"] = [*metadata.get("compiler_notes", []), *compiler_notes]
            return AgentPlanDAG(nodes=nodes_with_inferred_deps, metadata=metadata)

        downstream = self._downstream_map(retained_nodes)
        removed_leaf_ids: set[str] = set()
        for node in retained_nodes:
            if node.capability not in self._EXPENSIVE_LEAF_CAPABILITIES:
                continue
            if downstream.get(node.node_id):
                continue
            if self._question_requires_capability(question_text, node.capability):
                continue
            removed_leaf_ids.add(node.node_id)
            compiler_notes.append(f"node {node.node_id}: removed unrequested expensive leaf analysis ({node.capability})")

        retained_nodes = [node for node in retained_nodes if node.node_id not in removed_leaf_ids]
        retained_ids = {node.node_id for node in retained_nodes}

        def unique_retained_node_id(preferred: str) -> str:
            if preferred not in retained_ids:
                retained_ids.add(preferred)
                return preferred
            index = 2
            while f"{preferred}_{index}" in retained_ids:
                index += 1
            node_id = f"{preferred}_{index}"
            retained_ids.add(node_id)
            return node_id

        has_plain_language_filter = any(
            node.capability == "filter_working_set"
            and self._has_plain_language_filter(self._effective_node_inputs(node))
            for node in retained_nodes
        )
        language_node_id = next((node.node_id for node in retained_nodes if node.capability == "lang_id"), "")
        if has_plain_language_filter and not language_node_id:
            fetch_node_id = next((node.node_id for node in retained_nodes if node.capability == "fetch_documents"), "")
            if fetch_node_id:
                language_node_id = unique_retained_node_id("language_annotations")
                retained_nodes.append(
                    AgentPlanNode(
                        node_id=language_node_id,
                        capability="lang_id",
                        depends_on=[fetch_node_id],
                        description="Compiler-added language annotations for working-set language filters.",
                    )
                )
                compiler_notes.append(
                    f"node {language_node_id}: added language annotations for working-set language filter"
                )
        if language_node_id:
            rewritten_nodes: list[AgentPlanNode] = []
            for node in retained_nodes:
                if node.capability != "filter_working_set":
                    rewritten_nodes.append(node)
                    continue
                rewritten_inputs, rewritten = self._rewrite_plain_language_filter_inputs(node.inputs, language_node_id)
                if rewritten:
                    rewritten_nodes.append(
                        AgentPlanNode(
                            node_id=node.node_id,
                            capability=node.capability,
                            tool_name=node.tool_name,
                            inputs=rewritten_inputs,
                            depends_on=list(dict.fromkeys([*node.depends_on, language_node_id])),
                            optional=node.optional,
                            cacheable=node.cacheable,
                            description=node.description,
                        )
                    )
                    compiler_notes.append(
                        f"node {node.node_id}: rewrote plain language filter to annotation-backed dependency {language_node_id}"
                    )
                else:
                    rewritten_nodes.append(node)
            retained_nodes = rewritten_nodes

        market_ticker = self._infer_market_ticker(question_text)
        if (
            market_ticker
            and self._question_requires_capability(question_text, "join_external_series")
            and not any(node.capability == "join_external_series" for node in retained_nodes)
        ):
            aggregate_node = next(
                (
                    node
                    for node in retained_nodes
                    if node.capability == "time_series_aggregate"
                    and isinstance(node.inputs.get("series", node.inputs.get("series_definitions")), list)
                ),
                next((node for node in retained_nodes if node.capability == "time_series_aggregate"), None),
            )
            if aggregate_node is not None:
                date_window = self._extract_date_window(question_text)
                market_node_id = unique_retained_node_id("market_series")
                retained_nodes.append(
                    AgentPlanNode(
                        node_id=market_node_id,
                        capability="join_external_series",
                        inputs={
                            "ticker": market_ticker,
                            "date_from": date_window.get("date_from", ""),
                            "date_to": date_window.get("date_to", ""),
                            "interval": "1mo",
                            "left_key": "time_bin",
                            "right_key": "time_bin",
                            "how": "left",
                        },
                        depends_on=[aggregate_node.node_id],
                    )
                )
                retained_nodes.append(
                    AgentPlanNode(
                        node_id=unique_retained_node_id("plot_market_drawdown"),
                        capability="plot_artifact",
                        inputs={
                            "plot_name": f"{market_ticker.lower()}_framing_vs_drawdown",
                            "plot_type": "line",
                            "x": "time_bin",
                            "y": ["share_of_documents", "market_drawdown"],
                            "series": "series_name",
                            "title": f"{market_ticker} framing share and stock drawdowns",
                            "x_label": "Date",
                            "y_label": "Share / drawdown",
                        },
                        depends_on=[market_node_id],
                    )
                )
                compiler_notes.append(
                    f"node {market_node_id}: added aggregate-based external series join after invalid market branch repair"
                )

        retained_node_map = {node.node_id: node for node in retained_nodes}
        downstream = self._downstream_map(retained_nodes)
        compiled_nodes: list[AgentPlanNode] = []
        for node in retained_nodes:
            optional = node.optional
            description = node.description
            downstream_nodes = [retained_node_map[item] for item in downstream.get(node.node_id, set()) if item in retained_node_map]
            has_required_downstream = any(not item.optional for item in downstream_nodes)
            if (
                node.capability in self._EXPENSIVE_OPTIONAL_CAPABILITIES
                and not self._question_requires_capability(question_text, node.capability)
                and not has_required_downstream
            ):
                optional = True
                if not description:
                    description = "Supporting analysis; not required for the core answer."
                compiler_notes.append(f"node {node.node_id}: marked optional supporting analysis ({node.capability})")
            compiled_nodes.append(
                AgentPlanNode(
                    node_id=node.node_id,
                    capability=node.capability,
                    tool_name=node.tool_name,
                    inputs=dict(node.inputs),
                    depends_on=list(node.depends_on),
                    optional=optional,
                    cacheable=node.cacheable,
                    description=description,
                )
            )

        metadata = dict(dag.metadata)
        if compiler_notes:
            metadata["compiler_notes"] = [*metadata.get("compiler_notes", []), *compiler_notes]
        return AgentPlanDAG(nodes=compiled_nodes or retained_nodes, metadata=metadata)

    def _normalize_plan_dag(self, dag: AgentPlanDAG, question_text: str = "") -> AgentPlanDAG:
        existing_ids = {node.node_id for node in dag.nodes}

        def unique_node_id(preferred: str) -> str:
            if preferred not in existing_ids:
                existing_ids.add(preferred)
                return preferred
            index = 2
            while f"{preferred}_{index}" in existing_ids:
                index += 1
            node_id = f"{preferred}_{index}"
            existing_ids.add(node_id)
            return node_id

        dag_text = " ".join(
            part
            for part in [question_text, dag.metadata.get("question_family", "")]
            if part
        ).strip()
        needs_semantic_similarity = self._needs_semantic_similarity_analysis(dag_text)
        needs_syntax_roles = self._needs_syntax_role_analysis(dag_text)
        needs_attributed_explanations = self._needs_attributed_explanation_analysis(dag_text)
        needs_specialized_intent = needs_semantic_similarity or needs_syntax_roles or needs_attributed_explanations
        search_node_id = next((node.node_id for node in dag.nodes if node.capability in self._SEARCH_BACKBONE_CAPABILITIES), "")
        fetch_node_id = next((node.node_id for node in dag.nodes if node.capability == "fetch_documents"), "")
        requires_retrieval_backbone = any(
            node.capability in self._DOC_RETRIEVAL_BACKBONE_CAPABILITIES
            for node in dag.nodes
        ) or needs_semantic_similarity or needs_syntax_roles or needs_attributed_explanations
        if requires_retrieval_backbone and not search_node_id:
            search_node_id = unique_node_id("search")
        if requires_retrieval_backbone and not fetch_node_id:
            fetch_node_id = unique_node_id("fetch")

        query_text = ""
        if dag_text:
            query_text = self._compact_query_terms(dag_text, self._query_anchor_terms(dag_text)) or dag_text
        search_inputs = infer_retrieval_budget(
            dag_text,
            configured_mode=os.getenv("CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE", "hybrid"),
        ).to_inputs()
        if needs_semantic_similarity:
            search_inputs["retrieval_strategy"] = "semantic_exploratory"
            search_inputs["retrieval_mode"] = "dense"
        if needs_syntax_roles:
            syntax_role_overrides = (
                {"retrieval_strategy": "exhaustive_analytic", "retrieve_all": True, "top_k": 0}
                if self._syntax_role_needs_population_analysis(dag_text)
                else {"retrieval_strategy": "precision_ranked"}
            )
            precision_search_inputs = infer_retrieval_budget(
                dag_text,
                inputs=syntax_role_overrides,
                configured_mode=os.getenv("CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE", "hybrid"),
            ).to_inputs()
            search_inputs.update(precision_search_inputs)
        elif needs_attributed_explanations:
            precision_search_inputs = infer_retrieval_budget(
                dag_text,
                inputs={"retrieval_strategy": "precision_ranked"},
                configured_mode=os.getenv("CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE", "hybrid"),
            ).to_inputs()
            search_inputs.update(precision_search_inputs)
        if query_text:
            repaired_query = self._repair_search_query(str(search_inputs.get("query", "")), query_text)
            search_inputs["query"] = self._apply_source_scope_to_query(repaired_query, dag_text)
        if requires_retrieval_backbone:
            search_inputs.update(self._extract_date_window(dag_text))
        if self._needs_source_comparison_analysis(dag_text) and not needs_specialized_intent:
            search_inputs["retrieval_strategy"] = "exhaustive_analytic"
            search_inputs["retrieve_all"] = True
            search_inputs["top_k"] = 0

        normalized_nodes: list[AgentPlanNode] = []
        if requires_retrieval_backbone and not any(node.capability in self._SEARCH_BACKBONE_CAPABILITIES for node in dag.nodes):
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=search_node_id,
                    capability="db_search",
                    inputs=search_inputs,
                )
            )
        if requires_retrieval_backbone and not any(node.capability == "fetch_documents" for node in dag.nodes):
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=fetch_node_id,
                    capability="fetch_documents",
                    inputs={},
                    depends_on=[search_node_id],
                )
            )

        for node in dag.nodes:
            depends_on = list(node.depends_on)
            node_inputs = dict(node.inputs)
            if node.capability in self._SEARCH_BACKBONE_CAPABILITIES:
                payload_inputs = node_inputs.get("payload") if isinstance(node_inputs.get("payload"), dict) else {}
                merged_node_inputs = {**payload_inputs, **{key: value for key, value in node_inputs.items() if key != "payload"}}
                normalized_search_inputs = infer_retrieval_budget(
                    dag_text,
                    inputs=merged_node_inputs,
                    configured_mode=os.getenv("CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE", "hybrid"),
                ).to_inputs()
                for key in ("query", "date_from", "date_to", "year_balance", "retrieve_all", "retrieval_strategy"):
                    if key in merged_node_inputs and merged_node_inputs.get(key) not in (None, ""):
                        normalized_search_inputs[key] = merged_node_inputs[key]
                question_window = self._extract_date_window(dag_text)
                if "date_from" in question_window:
                    normalized_search_inputs["date_from"] = question_window["date_from"]
                if "date_to" in question_window:
                    normalized_search_inputs["date_to"] = question_window["date_to"]
                elif "date_from" in question_window:
                    inferred_year = str(question_window["date_from"])[:4]
                    planned_date_to = str(normalized_search_inputs.get("date_to", "") or "")
                    if planned_date_to.startswith(inferred_year):
                        normalized_search_inputs.pop("date_to", None)
                if query_text:
                    if needs_specialized_intent:
                        normalized_search_inputs["query"] = self._apply_source_scope_to_query(query_text, dag_text)
                    else:
                        repaired_query = self._repair_search_query(
                            str(normalized_search_inputs.get("query", "")),
                            query_text,
                        )
                        normalized_search_inputs["query"] = self._apply_source_scope_to_query(repaired_query, dag_text)
                if self._needs_source_comparison_analysis(dag_text) and not needs_specialized_intent:
                    normalized_search_inputs["retrieval_strategy"] = "exhaustive_analytic"
                    normalized_search_inputs["retrieve_all"] = True
                    normalized_search_inputs["top_k"] = 0
                if needs_semantic_similarity:
                    normalized_search_inputs["retrieval_strategy"] = "semantic_exploratory"
                    normalized_search_inputs["retrieval_mode"] = "dense"
                if needs_specialized_intent:
                    for key in (
                        "top_k",
                        "retrieval_strategy",
                        "retrieval_mode",
                        "lexical_top_k",
                        "dense_top_k",
                        "use_rerank",
                        "fusion_k",
                        "fallback_top_k",
                        "rerank_top_k",
                        "retrieve_all",
                    ):
                        if key in search_inputs:
                            normalized_search_inputs[key] = search_inputs[key]
                    if "retrieve_all" not in search_inputs:
                        normalized_search_inputs["retrieve_all"] = False
                    if needs_semantic_similarity:
                        normalized_search_inputs["retrieval_strategy"] = "semantic_exploratory"
                node_inputs = normalized_search_inputs
            if node.capability == "fetch_documents" and search_node_id and not depends_on:
                depends_on = [search_node_id]
            elif (
                fetch_node_id
                and node.capability in self._DOC_RETRIEVAL_BACKBONE_CAPABILITIES
                and node.capability != "fetch_documents"
                and node.node_id != fetch_node_id
                and not depends_on
            ):
                depends_on = [fetch_node_id]
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=node.node_id,
                    capability=node.capability,
                    tool_name=node.tool_name,
                    inputs=node_inputs,
                    depends_on=list(dict.fromkeys(depends_on)),
                    optional=node.optional,
                    cacheable=node.cacheable,
                    description=node.description,
                )
            )
        metadata = dict(dag.metadata)
        if needs_semantic_similarity:
            self._ensure_semantic_similarity_nodes(normalized_nodes, unique_node_id, dag_text)
            if metadata.get("question_family", "") in {"", "generic"}:
                metadata["question_family"] = "semantic_similarity_terms"
        elif needs_syntax_roles:
            self._ensure_syntax_role_nodes(normalized_nodes, unique_node_id, dag_text)
            if metadata.get("question_family", "") in {"", "generic"}:
                metadata["question_family"] = "syntax_role_patterns"
        elif needs_attributed_explanations:
            self._ensure_attributed_explanation_nodes(normalized_nodes, unique_node_id, dag_text)
            if metadata.get("question_family", "") in {"", "generic"}:
                metadata["question_family"] = "attributed_explanation_series"
        elif self._needs_temporal_portrayal_analysis(dag_text):
            self._ensure_temporal_portrayal_nodes(normalized_nodes, unique_node_id, dag_text)
            if metadata.get("question_family", "") in {"", "generic"}:
                metadata["question_family"] = "temporal_portrayal_shift"
        elif self._needs_entity_trend_analysis(dag_text):
            normalized_nodes = self._strip_unrequested_quote_nodes(normalized_nodes, dag_text)
            self._ensure_entity_trend_nodes(normalized_nodes, unique_node_id, dag_text)
            if metadata.get("question_family", "") in {"", "generic"}:
                metadata["question_family"] = "entity_trend"
        elif self._needs_source_comparison_analysis(dag_text):
            self._ensure_source_comparison_nodes(normalized_nodes, unique_node_id)
            if metadata.get("question_family", "") in {"", "generic"}:
                metadata["question_family"] = "source_comparison"
        elif self._needs_noun_distribution_analysis(dag_text):
            self._ensure_noun_distribution_nodes(normalized_nodes, unique_node_id, dag_text)
            if metadata.get("question_family", "") in {"", "generic"}:
                metadata["question_family"] = "noun_distribution"
        normalized_nodes = self._reuse_subsumed_search_branches(normalized_nodes)
        return self._compile_plan_dag(AgentPlanDAG(nodes=normalized_nodes, metadata=metadata), dag_text)

    @staticmethod
    def _needs_semantic_similarity_analysis(text: str) -> bool:
        lowered = str(text or "").lower()
        if "semantic_similarity_terms" in lowered or "similarity_analysis" in lowered:
            return True
        wording_signal = bool(
            re.search(
                r"\b(?:semantic(?:ally)?|similar(?:ity)?|related|paraphras(?:e|es|ed)|same exact wording|exact wording|different wording|same wording|wording mismatch|lexical(?:ly)?)\b",
                lowered,
            )
            or re.search(r"\bnot\s+use\s+(?:the\s+)?same\b.{0,40}\bwording\b", lowered)
        )
        term_bridge_signal = bool(
            re.search(
                r"\b(?:recurring terms?|abbreviations?|acronyms?|connected descriptions?|describe(?:d|s)?|descriptions?|word choices?)\b",
                lowered,
            )
        )
        return wording_signal and term_bridge_signal

    @staticmethod
    def _needs_syntax_role_analysis(text: str) -> bool:
        lowered = str(text or "").lower()
        if "syntax_role_patterns" in lowered:
            return True
        return bool(
            re.search(r"\bwho\s+(?:did|does)\s+what\s+to\s+whom\b", lowered)
            or re.search(r"\b(?:subject[- ]verb[- ]object|svo|subject|verb|object|dependency|grammatical role|grammatical roles)\b", lowered)
            or re.search(r"\b(?:acting|acted)\b.{0,80}\b(?:acted upon|upon|against|by)\b", lowered)
            or re.search(r"\bwho\b.{0,80}\b(?:acting|acted upon|acted on|action patterns?|patterns differ)\b", lowered)
        )

    @staticmethod
    def _syntax_role_needs_population_analysis(text: str) -> bool:
        lowered = str(text or "").lower()
        return bool(
            re.search(r"\b(?:most\s+(?:often|frequent(?:ly)?|common)|frequency|distribution|dominant|dominated)\b", lowered)
            or re.search(r"\bpatterns?\s+(?:differ|changed?|shifted?)\b", lowered)
            or re.search(r"\bdiffer(?:ed|s|ent)?\s+between\b", lowered)
            or re.search(r"\b(?:across|between)\b.{0,80}\b(?:groups?|categories|actors?|institutions?|outlets?|sources?)\b", lowered)
            or re.search(r"\b(?:coverage|corpus|reports?|articles?)\b.{0,80}\b(?:who|subject|object|acted upon|acting)\b", lowered)
        )

    @staticmethod
    def _needs_attributed_explanation_analysis(text: str) -> bool:
        lowered = str(text or "").lower()
        if "attributed_explanation_series" in lowered:
            return True
        explanation_signal = bool(
            re.search(r"\b(?:explanation|explanations|explained|explain|reason|reasons|why|gave for)\b", lowered)
        )
        movement_signal = bool(
            re.search(
                r"\b(?:price movements?|market movements?|largest movements?|crash|recovery|drawdown|rally|selloff|spike|plunge|price shock)\b",
                lowered,
            )
        )
        attribution_signal = bool(
            re.search(
                r"\b(?:directly attributed|attributed to|quote|quoted|said|according to|named people|named institutions|speaker|spokes(?:man|woman|person|people))\b",
                lowered,
            )
        )
        external_signal = bool(
            re.search(r"\b(?:stock|share price|market|price|prices|oil|crude|gas|commodity|equity|ticker)\b", lowered)
        )
        return explanation_signal and movement_signal and (attribution_signal or external_signal)

    def _ensure_semantic_similarity_nodes(
        self,
        normalized_nodes: list[AgentPlanNode],
        unique_node_id: Callable[[str], str],
        question_text: str,
    ) -> None:
        def first_node_id(capability: str) -> str:
            return next((node.node_id for node in normalized_nodes if node.capability == capability), "")

        def ensure_node(
            capability: str,
            preferred_id: str,
            depends_on: list[str],
            *,
            inputs: dict[str, Any] | None = None,
            optional: bool = False,
        ) -> str:
            existing = first_node_id(capability)
            if existing:
                for node in normalized_nodes:
                    if node.node_id == existing:
                        if inputs:
                            updated_inputs = dict(node.inputs)
                            payload_inputs = dict(updated_inputs.get("payload", {})) if isinstance(updated_inputs.get("payload"), dict) else {}
                            for key, value in inputs.items():
                                updated_inputs[key] = value
                                if payload_inputs:
                                    payload_inputs[key] = value
                            if payload_inputs:
                                updated_inputs["payload"] = payload_inputs
                            node.inputs = updated_inputs
                        if not node.depends_on:
                            node.depends_on = list(dict.fromkeys(depends_on))
                        break
                return existing
            node_id = unique_node_id(preferred_id)
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=node_id,
                    capability=capability,
                    inputs=inputs or {},
                    depends_on=list(dict.fromkeys(depends_on)),
                    optional=optional,
                )
            )
            return node_id

        fetch_node_id = first_node_id("fetch_documents")
        if not fetch_node_id:
            return
        search_node_id = first_node_id("db_search") or first_node_id("sql_query_search")
        ensure_node("create_working_set", "working_set", [fetch_node_id], optional=True)
        doc_embeddings_id = ensure_node("doc_embeddings", "doc_embeddings", [fetch_node_id])
        similarity_index_id = ensure_node("similarity_index", "similarity_index", [fetch_node_id])
        similarity_pairwise_id = ensure_node(
            "similarity_pairwise",
            "similarity_pairwise",
            [fetch_node_id],
            inputs={"query": question_text},
        )
        ensure_node("word_embeddings", "word_embeddings", [fetch_node_id])
        ensure_node("extract_acronyms", "acronyms", [fetch_node_id])
        keyterms_id = ensure_node("extract_keyterms", "keyterms", [fetch_node_id], inputs={"top_k": 40})
        ner_id = ensure_node("ner", "entities", [fetch_node_id], optional=True)
        entity_link_id = ensure_node("entity_link", "entity_link", [ner_id], optional=True)
        series_id = ensure_node(
            "time_series_aggregate",
            "series",
            [fetch_node_id],
            inputs={
                "documents_node": fetch_node_id,
                "time_field": "published_at",
                "bucket_granularity": "month",
                "metrics": ["document_count"],
            },
        )
        ensure_node("change_point_detect", "changes", [series_id], optional=True)
        if not any(node.capability == "plot_artifact" and series_id in node.depends_on for node in normalized_nodes):
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=unique_node_id("plot_semantic_terms"),
                    capability="plot_artifact",
                    inputs={
                        "plot_name": "semantic_term_coverage_over_time",
                        "plot_type": "line",
                        "x": "time_bin",
                        "y": "document_count",
                        "series": "series_name",
                        "title": "Semantic coverage over time",
                    },
                    depends_on=[series_id],
                    optional=True,
                )
            )
        evidence_deps = [
            dep
            for dep in [
                keyterms_id,
                doc_embeddings_id,
                similarity_index_id,
                similarity_pairwise_id,
                search_node_id,
                entity_link_id,
            ]
            if dep
        ]
        has_semantic_evidence = any(
            node.capability == "build_evidence_table" and entity_link_id and entity_link_id in node.depends_on
            for node in normalized_nodes
        )
        if not has_semantic_evidence:
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=unique_node_id("semantic_evidence"),
                    capability="build_evidence_table",
                    inputs={"task": "semantic_similarity_evidence"},
                    depends_on=list(dict.fromkeys(evidence_deps)),
                )
            )

    def _ensure_syntax_role_nodes(
        self,
        normalized_nodes: list[AgentPlanNode],
        unique_node_id: Callable[[str], str],
        question_text: str,
    ) -> None:
        def first_node_id(capability: str) -> str:
            return next((node.node_id for node in normalized_nodes if node.capability == capability), "")

        def ensure_node(
            capability: str,
            preferred_id: str,
            depends_on: list[str],
            *,
            inputs: dict[str, Any] | None = None,
            optional: bool = False,
        ) -> str:
            existing = first_node_id(capability)
            if existing:
                for node in normalized_nodes:
                    if node.node_id == existing:
                        if inputs:
                            updated_inputs = dict(node.inputs)
                            payload_inputs = dict(updated_inputs.get("payload", {})) if isinstance(updated_inputs.get("payload"), dict) else {}
                            for key, value in inputs.items():
                                updated_inputs[key] = value
                                if payload_inputs:
                                    payload_inputs[key] = value
                            if payload_inputs:
                                updated_inputs["payload"] = payload_inputs
                            node.inputs = updated_inputs
                        if not node.depends_on:
                            node.depends_on = list(dict.fromkeys(depends_on))
                        break
                return existing
            node_id = unique_node_id(preferred_id)
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=node_id,
                    capability=capability,
                    inputs=inputs or {},
                    depends_on=list(dict.fromkeys(depends_on)),
                    optional=optional,
                )
            )
            return node_id

        fetch_node_id = first_node_id("fetch_documents")
        if not fetch_node_id:
            return
        ensure_node("create_working_set", "working_set", [fetch_node_id], optional=True)
        sentence_id = ensure_node("sentence_split", "sentences", [fetch_node_id])
        token_id = ensure_node("tokenize", "tokens", [fetch_node_id])
        pos_id = ensure_node("pos_morph", "pos", [fetch_node_id])
        lemma_id = ensure_node("lemmatize", "lemmas", [fetch_node_id])
        dependency_id = ensure_node("dependency_parse", "dependencies", [fetch_node_id])
        svo_id = ensure_node("extract_svo_triples", "svo_triples", [fetch_node_id])
        noun_chunks_id = ensure_node("noun_chunks", "noun_chunks", [fetch_node_id])
        ner_id = ensure_node("ner", "entities", [fetch_node_id], optional=True)
        entity_link_id = ensure_node("entity_link", "entity_link", [ner_id], optional=True)
        quotes_id = ensure_node("quote_extract", "quotes", [fetch_node_id])
        attributed_quotes_id = ensure_node("quote_attribute", "quote_attribution", [quotes_id])
        date_window = self._extract_date_window(question_text)
        forced_dependencies = {
            sentence_id: [fetch_node_id],
            token_id: [fetch_node_id],
            pos_id: [fetch_node_id],
            lemma_id: [fetch_node_id],
            dependency_id: [fetch_node_id],
            svo_id: [fetch_node_id],
            noun_chunks_id: [fetch_node_id],
            quotes_id: [fetch_node_id],
            attributed_quotes_id: [quotes_id],
        }
        for node in normalized_nodes:
            if node.node_id in forced_dependencies:
                node.depends_on = forced_dependencies[node.node_id]
        role_counts_id = ensure_node(
            "time_series_aggregate",
            "syntax_role_counts",
            [svo_id],
            inputs={
                "group_by": "actor_group",
                "metrics": ["mention_count"],
                "bucket_granularity": "year",
                "fallback_time_bin": date_window.get("date_from", ""),
                "top_k": 8,
            },
            optional=True,
        )
        target_counts_id = next((node.node_id for node in normalized_nodes if node.node_id == "syntax_target_counts"), "")
        if not target_counts_id:
            target_counts_id = unique_node_id("syntax_target_counts")
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=target_counts_id,
                    capability="time_series_aggregate",
                    inputs={
                        "group_by": "target_group",
                        "metrics": ["mention_count"],
                        "bucket_granularity": "year",
                        "fallback_time_bin": date_window.get("date_from", ""),
                        "top_k": 8,
                    },
                    depends_on=[svo_id],
                    optional=True,
                )
            )
        if (
            self._question_requires_capability(question_text, "plot_artifact")
            and not any(node.capability == "plot_artifact" and role_counts_id in node.depends_on for node in normalized_nodes)
        ):
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=unique_node_id("plot_syntax_roles"),
                    capability="plot_artifact",
                    inputs={
                        "plot_name": "syntax_role_patterns",
                        "top_k": 8,
                        "title": "Syntax role pattern summary",
                    },
                    depends_on=[role_counts_id],
                    optional=True,
                )
            )
        role_evidence_deps = [
            sentence_id,
            token_id,
            pos_id,
            lemma_id,
            dependency_id,
            noun_chunks_id,
            attributed_quotes_id,
            svo_id,
        ]
        has_role_evidence = any(
            node.capability == "build_evidence_table" and any(dep in node.depends_on for dep in role_evidence_deps)
            for node in normalized_nodes
        )
        if not has_role_evidence:
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=unique_node_id("role_evidence"),
                    capability="build_evidence_table",
                    inputs={"task": "syntax_role_evidence"},
                    depends_on=role_evidence_deps,
                )
            )

    def _ensure_attributed_explanation_nodes(
        self,
        normalized_nodes: list[AgentPlanNode],
        unique_node_id: Callable[[str], str],
        question_text: str,
    ) -> None:
        def first_node_id(capability: str) -> str:
            return next((node.node_id for node in normalized_nodes if node.capability == capability), "")

        def ensure_node(
            capability: str,
            preferred_id: str,
            depends_on: list[str],
            *,
            inputs: dict[str, Any] | None = None,
            optional: bool = False,
        ) -> str:
            existing = first_node_id(capability)
            if existing:
                for node in normalized_nodes:
                    if node.node_id == existing:
                        if inputs:
                            updated_inputs = dict(node.inputs)
                            payload_inputs = dict(updated_inputs.get("payload", {})) if isinstance(updated_inputs.get("payload"), dict) else {}
                            for key, value in inputs.items():
                                updated_inputs[key] = value
                                if payload_inputs:
                                    payload_inputs[key] = value
                            if payload_inputs:
                                updated_inputs["payload"] = payload_inputs
                            node.inputs = updated_inputs
                        if not node.depends_on:
                            node.depends_on = list(dict.fromkeys(depends_on))
                        break
                return existing
            node_id = unique_node_id(preferred_id)
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=node_id,
                    capability=capability,
                    inputs=inputs or {},
                    depends_on=list(dict.fromkeys(depends_on)),
                    optional=optional,
                )
            )
            return node_id

        fetch_node_id = first_node_id("fetch_documents")
        if not fetch_node_id:
            return
        ensure_node("create_working_set", "working_set", [fetch_node_id], optional=True)
        keyterms_id = ensure_node("extract_keyterms", "keyterms", [fetch_node_id], inputs={"top_k": 40})
        quotes_id = ensure_node("quote_extract", "quotes", [fetch_node_id])
        quote_attribute_id = ensure_node("quote_attribute", "quote_attribution", [quotes_id])
        claim_spans_id = ensure_node("claim_span_extract", "claim_spans", [fetch_node_id], inputs={"query_focus": question_text})
        claim_scores_id = ensure_node("claim_strength_score", "claim_scores", [claim_spans_id])
        ner_id = ensure_node("ner", "entities", [fetch_node_id], optional=True)
        entity_link_id = ensure_node("entity_link", "entity_link", [ner_id], optional=True)
        series_id = ensure_node(
            "time_series_aggregate",
            "series",
            [fetch_node_id],
            inputs={
                "documents_node": fetch_node_id,
                "time_field": "published_at",
                "bucket_granularity": "month",
                "metrics": ["document_count"],
                "fallback_time_bin": self._extract_date_window(question_text).get("date_from", ""),
            },
        )
        ensure_node("change_point_detect", "changes", [series_id], optional=True)
        ensure_node("burst_detect", "bursts", [series_id], optional=True)

        ticker = self._infer_market_ticker(question_text)
        market_node_id = first_node_id("join_external_series")
        date_window = self._extract_date_window(question_text)
        market_inputs = {
            "ticker": ticker,
            "date_from": date_window.get("date_from", ""),
            "date_to": date_window.get("date_to", ""),
            "interval": "1d",
            "join_granularity": "month",
            "left_key": "time_bin",
            "right_key": "time_bin",
            "how": "left",
        }
        if ticker and market_node_id:
            for node in normalized_nodes:
                if node.node_id != market_node_id:
                    continue
                updated_inputs = dict(node.inputs)
                payload_inputs = dict(updated_inputs.get("payload", {})) if isinstance(updated_inputs.get("payload"), dict) else {}
                for key, value in market_inputs.items():
                    updated_inputs[key] = value
                    if payload_inputs:
                        payload_inputs[key] = value
                if payload_inputs:
                    payload_inputs.setdefault("date_range", {})
                    if isinstance(payload_inputs["date_range"], dict):
                        payload_inputs["date_range"]["start"] = market_inputs["date_from"]
                        payload_inputs["date_range"]["end"] = market_inputs["date_to"]
                    updated_inputs["payload"] = payload_inputs
                node.inputs = updated_inputs
                node.depends_on = [series_id]
                break
        if ticker and not market_node_id:
            market_node_id = unique_node_id("market_series")
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=market_node_id,
                    capability="join_external_series",
                    inputs=market_inputs,
                    depends_on=[series_id],
                )
            )

        plot_dep = market_node_id or series_id
        if not any(node.capability == "plot_artifact" and plot_dep in node.depends_on for node in normalized_nodes):
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=unique_node_id("plot_explanations"),
                    capability="plot_artifact",
                    inputs={
                        "plot_name": "coverage_vs_external_price_movements",
                        "plot_type": "line",
                        "x": "time_bin",
                        "y": ["document_count", "market_drawdown"] if market_node_id else "document_count",
                        "series": "series_name",
                        "title": "Coverage and price movement timeline",
                    },
                    depends_on=[plot_dep],
                    optional=True,
                )
            )
        if not any(node.capability == "build_evidence_table" for node in normalized_nodes):
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=unique_node_id("explanation_evidence"),
                    capability="build_evidence_table",
                    inputs={"task": "attributed_explanation_evidence"},
                    depends_on=[keyterms_id, quote_attribute_id, entity_link_id, claim_scores_id],
                )
            )

    @staticmethod
    def _needs_noun_distribution_analysis(text: str) -> bool:
        lowered = str(text or "").lower()
        return "noun" in lowered and any(term in lowered for term in ("distribution", "frequency", "frequencies", "top"))

    def _ensure_noun_distribution_nodes(
        self,
        normalized_nodes: list[AgentPlanNode],
        unique_node_id: Callable[[str], str],
        question_text: str,
    ) -> None:
        fetch_node_id = next((node.node_id for node in normalized_nodes if node.capability == "fetch_documents"), "")
        if not fetch_node_id:
            return
        if not any(node.capability == "build_evidence_table" for node in normalized_nodes):
            top_n = infer_requested_output_limit(question_text, default=100, minimum=20, maximum=500)
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=unique_node_id("noun_distribution"),
                    capability="build_evidence_table",
                    inputs={
                        "task": "noun_frequency_distribution",
                        "task_name": "noun_frequency_distribution",
                        "text_fields": ["title", "text"],
                        "filters": {"upos": ["NOUN", "PROPN"]},
                        "top_n": top_n,
                    },
                    depends_on=[fetch_node_id],
                )
            )
        noun_node_id = next((node.node_id for node in normalized_nodes if node.capability == "build_evidence_table"), "")
        if noun_node_id and not any(node.capability == "plot_artifact" and noun_node_id in node.depends_on for node in normalized_nodes):
            plot_top_k = min(infer_requested_output_limit(question_text, default=30, minimum=10, maximum=120), 120)
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=unique_node_id("plot_noun_distribution"),
                    capability="plot_artifact",
                    inputs={
                        "plot_name": "noun_distribution",
                        "plot_type": "bar",
                        "x": "lemma",
                        "y": "count",
                        "top_k": plot_top_k,
                        "title": "Noun distribution",
                    },
                    depends_on=[noun_node_id],
                )
            )

    def _needs_temporal_portrayal_analysis(self, text: str) -> bool:
        lowered = str(text or "").lower()
        entity_trend_question = any(
            phrase in lowered
            for phrase in ("named entities", "named entity", "which actors", "actors dominated", "entities dominate")
        )
        explicit_portrayal_or_value = any(
            term in lowered
            for term in (
                "sentiment", "tone", "framing", "portrayal", "portrayed",
                "perceived", "perception", "value", "valuation", "worth", "reputation",
            )
        )
        if entity_trend_question and not explicit_portrayal_or_value:
            return False
        temporal = any(
            term in lowered
            for term in (
                "shift", "shifted", "changed", "change", "trend", "over time",
                "evolve", "evolved", "evolution", "from ", "between ", "correspond",
            )
        )
        portrayal = explicit_portrayal_or_value or any(
            term in lowered
            for term in (
                "coverage",
                "explained", "explain", "media", "news", "topic", "topics", "keyterm", "keyterms",
            )
        )
        comparison = bool(re.search(r"\b(vs\.?|versus|compared? to|comparison|relative to)\b", lowered))
        return temporal and (portrayal or comparison)

    def _split_framing_terms(self, raw: str) -> list[str]:
        cleaned = re.sub(r"\([^)]*\)", " ", str(raw or ""))
        cleaned = re.sub(
            r"\b(?:framing|frame|frames|coverage|reporting|narrative|narratives|portrayal|tone|topic|topics)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\b(?:around|during|through|between|from|to|toward|towards|into|and how|correspond.*)$", " ", cleaned, flags=re.IGNORECASE)
        parts = re.split(r"\s*(?:/|,|;|\band\b|\bor\b|&|\+)\s*", cleaned)
        stop_terms = {
            "a",
            "an",
            "and",
            "as",
            "correspond",
            "corresponded",
            "did",
            "from",
            "how",
            "in",
            "not",
            "of",
            "shift",
            "shifted",
            "they",
            "the",
            "to",
            "with",
        }
        terms: list[str] = []
        for part in parts:
            normalized = re.sub(r"[^A-Za-z0-9'. -]+", " ", part).strip(" .'-")
            normalized = re.sub(r"\s+", " ", normalized)
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in stop_terms or re.fullmatch(r"(?:19|20)\d{2}(?:\s*-\s*(?:19|20)\d{2})?", lowered):
                continue
            if not re.search(r"[A-Za-z]", normalized):
                continue
            terms.append(lowered)
        return list(dict.fromkeys(terms))

    def _infer_framing_series_definitions(self, text: str) -> list[dict[str, Any]]:
        question = str(text or "")
        patterns = [
            re.compile(
                r"\bfrom\s+(?P<left>.+?)\s+framing\s+(?:to|toward|towards|into)\s+(?P<right>.+?)\s+framing\b",
                flags=re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"\bfrom\s+(?P<left>.+?)\s+(?:to|toward|towards|into)\s+(?P<right>.+?)(?:\s+(?:around|during|between|from\s+(?:19|20)\d{2}|and how)|[?.]|$)",
                flags=re.IGNORECASE | re.DOTALL,
            ),
        ]
        for pattern in patterns:
            match = pattern.search(question)
            if not match:
                continue
            definitions: list[dict[str, Any]] = []
            for side in ("left", "right"):
                terms = self._split_framing_terms(match.group(side))
                if not terms:
                    continue
                label = "/".join(terms[:3])
                definitions.append(
                    {
                        "series_name": label,
                        "label": label,
                        "aliases": terms,
                        "keyword_terms": terms,
                    }
                )
            if len(definitions) >= 2:
                return definitions
        return []

    def _needs_entity_trend_analysis(self, text: str) -> bool:
        lowered = str(text or "").lower()
        asks_entities = any(
            phrase in lowered
            for phrase in (
                "named entities",
                "named entity",
                "which actors",
                "main actors",
                "prominent actors",
                "actors dominated",
                "actor dominated",
                "entities dominate",
                "entities dominated",
            )
        )
        if not asks_entities and re.search(r"\bactors?\b", lowered):
            asks_entities = bool(
                re.search(
                    r"\b(?:main|most|prominent|prominence|dominant|dominate|dominated|leading|top|central|key)\b",
                    lowered,
                )
            )
        asks_change = any(term in lowered for term in ("over time", "change", "changed", "trend", "evolve", "evolved"))
        return asks_entities and asks_change

    def _has_explicit_quote_intent(self, text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:quote|quotes|quoted|quotation|speaker|speakers|said|says|told|according to|spokes(?:man|woman|person|people)|attribut(?:e|ed|ion))\b",
                str(text or ""),
                flags=re.IGNORECASE,
            )
        )

    def _strip_unrequested_quote_nodes(
        self,
        nodes: list[AgentPlanNode],
        question_text: str,
    ) -> list[AgentPlanNode]:
        if self._has_explicit_quote_intent(question_text):
            return nodes
        quote_capabilities = {"quote_extract", "quote_attribute"}
        removable_after_quote = {
            "build_evidence_table",
            "change_point_detect",
            "plot_artifact",
            "python_runner",
            "time_series_aggregate",
        }
        removed: set[str] = {node.node_id for node in nodes if node.capability in quote_capabilities}
        changed = True
        while changed:
            changed = False
            for node in nodes:
                if node.node_id in removed:
                    continue
                if not any(dep in removed for dep in node.depends_on):
                    continue
                if node.capability in removable_after_quote:
                    removed.add(node.node_id)
                    changed = True
        if not removed:
            return nodes
        return [
            AgentPlanNode(
                node_id=node.node_id,
                capability=node.capability,
                tool_name=node.tool_name,
                inputs=node.inputs,
                depends_on=[dep for dep in node.depends_on if dep not in removed],
                optional=node.optional,
                cacheable=node.cacheable,
                description=node.description,
            )
            for node in nodes
            if node.node_id not in removed
        ]

    def _needs_source_comparison_analysis(self, text: str) -> bool:
        matches = self._explicit_source_scope_matches(text)
        if not self._explicit_source_scope_is_active(text, matches) or len(matches) < 2:
            return False
        lowered = str(text or "").lower()
        return bool(
            re.search(r"\b(vs\.?|versus|compared? to|compare|differ(?:ent|ently)?|difference|between)\b", lowered)
            or re.search(r"\b(?:report|reports|reported|coverage|cover)\b", lowered)
        )

    def _ensure_source_comparison_nodes(
        self,
        normalized_nodes: list[AgentPlanNode],
        unique_node_id: Callable[[str], str],
    ) -> None:
        fetch_node_id = next((node.node_id for node in normalized_nodes if node.capability == "fetch_documents"), "")
        if not fetch_node_id:
            return
        keyterms_node_id = next((node.node_id for node in normalized_nodes if node.capability == "extract_keyterms"), "")
        if keyterms_node_id:
            for node in normalized_nodes:
                if node.node_id == keyterms_node_id:
                    updated_inputs = dict(node.inputs)
                    updated_inputs.setdefault("group_by", "outlet")
                    updated_inputs.setdefault("top_k", 25)
                    node.inputs = updated_inputs
                    if not node.depends_on:
                        node.depends_on = [fetch_node_id]
                    break
        else:
            keyterms_node_id = unique_node_id("keyterms_by_source")
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=keyterms_node_id,
                    capability="extract_keyterms",
                    inputs={"group_by": "outlet", "top_k": 25},
                    depends_on=[fetch_node_id],
                )
            )
        has_plot = False
        for node in normalized_nodes:
            if node.capability == "plot_artifact" and keyterms_node_id in node.depends_on:
                updated_inputs = dict(node.inputs)
                updated_inputs.setdefault("plot_name", "source_keyterm_comparison")
                updated_inputs.setdefault("plot_type", "bar")
                updated_inputs.setdefault("x", "term")
                updated_inputs.setdefault("y", "score")
                updated_inputs.setdefault("series", "outlet")
                updated_inputs.setdefault("top_k", 20)
                node.inputs = updated_inputs
                has_plot = True
        if not has_plot:
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=unique_node_id("plot_keyterms"),
                    capability="plot_artifact",
                    inputs={
                        "plot_name": "source_keyterm_comparison",
                        "plot_type": "bar",
                        "x": "term",
                        "y": "score",
                        "series": "outlet",
                        "top_k": 20,
                    },
                    depends_on=[keyterms_node_id],
                    optional=True,
                )
            )

    def _ensure_entity_trend_nodes(
        self,
        normalized_nodes: list[AgentPlanNode],
        unique_node_id: Callable[[str], str],
        question_text: str,
    ) -> None:
        def first_node_id(capability: str) -> str:
            return next((node.node_id for node in normalized_nodes if node.capability == capability), "")

        def merged_inputs(node: AgentPlanNode) -> dict[str, Any]:
            payload = node.inputs.get("payload")
            if not isinstance(payload, dict):
                return dict(node.inputs)
            return {**payload, **{key: value for key, value in node.inputs.items() if key != "payload"}}

        fetch_node_id = first_node_id("fetch_documents")
        if not fetch_node_id:
            return

        ner_node_id = first_node_id("ner")
        if not ner_node_id:
            ner_node_id = unique_node_id("ner")
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=ner_node_id,
                    capability="ner",
                    depends_on=[fetch_node_id],
                )
            )

        entity_table_id = next(
            (
                node.node_id
                for node in normalized_nodes
                if node.capability == "build_evidence_table"
                and str(merged_inputs(node).get("task", merged_inputs(node).get("task_name", ""))).lower()
                in {"named_entity_frequency", "entity_frequency", "entity_frequency_distribution", "entity_prominence", "actor_prominence"}
            ),
            "",
        )
        if not entity_table_id:
            entity_table_id = unique_node_id("entity_trend")
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=entity_table_id,
                    capability="build_evidence_table",
                    inputs={
                        "task": "named_entity_frequency",
                        "entity_field": "entity_text",
                        "entity_label_field": "label",
                        "entity_types": ["PERSON", "ORG", "GPE", "NORP", "EVENT"],
                        "group_by_time": True,
                        "time_field": "published_at",
                        "top_k": 100,
                    },
                    depends_on=[ner_node_id],
                )
            )
        else:
            for node in normalized_nodes:
                if node.node_id == entity_table_id:
                    updated_inputs = dict(node.inputs)
                    updated_inputs.setdefault("entity_field", "entity_text")
                    updated_inputs.setdefault("entity_label_field", "label")
                    updated_inputs.setdefault("entity_types", ["PERSON", "ORG", "GPE", "NORP", "EVENT"])
                    updated_inputs.setdefault("group_by_time", True)
                    updated_inputs.setdefault("time_field", "published_at")
                    updated_inputs.setdefault("top_k", 100)
                    node.inputs = updated_inputs
                    if not node.depends_on:
                        node.depends_on = [ner_node_id]
                    break

        plot_defaults = {
            "plot_name": "entity_trend",
            "plot_type": "line",
            "x": "time_bin",
            "y": "mention_count",
            "series": "entity",
            "top_k": 10,
        }
        has_plot = False
        for node in normalized_nodes:
            if node.capability == "plot_artifact" and (
                entity_table_id in node.depends_on or first_node_id("time_series_aggregate") in node.depends_on
            ):
                updated_inputs = dict(node.inputs)
                for key, value in plot_defaults.items():
                    updated_inputs.setdefault(key, value)
                node.inputs = updated_inputs
                if entity_table_id not in node.depends_on:
                    node.depends_on = [entity_table_id]
                has_plot = True
        if not has_plot:
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=unique_node_id("plot_entity_trend"),
                    capability="plot_artifact",
                    inputs=plot_defaults,
                    depends_on=[entity_table_id],
                    optional=True,
                )
            )

    def _ensure_temporal_portrayal_nodes(
        self,
        normalized_nodes: list[AgentPlanNode],
        unique_node_id: Callable[[str], str],
        question_text: str,
    ) -> None:
        def first_node_id(capability: str) -> str:
            return next((node.node_id for node in normalized_nodes if node.capability == capability), "")

        fetch_node_id = first_node_id("fetch_documents")
        if not fetch_node_id:
            return
        framing_series_definitions = self._infer_framing_series_definitions(question_text)

        def is_static_topic_plot(node: AgentPlanNode, topics_id: str) -> bool:
            if node.capability != "plot_artifact":
                return False
            inputs = dict(node.inputs)
            plot_name = str(inputs.get("plot_name", "") or "").strip().lower()
            x_field = str(inputs.get("x", inputs.get("x_field", "")) or "").strip().lower()
            has_temporal_axis = x_field in {"time", "time_bin", "date", "published_at", "month", "period", "time_period"}
            return (
                bool(topics_id and topics_id in node.depends_on)
                and not has_temporal_axis
                and plot_name in {"", "plot_topics", "portrayal_topics", "framing_topics"}
            )

        def ensure_node(
            capability: str,
            preferred_id: str,
            depends_on: list[str],
            *,
            inputs: dict[str, Any] | None = None,
            optional: bool = False,
        ) -> str:
            existing = first_node_id(capability)
            if existing:
                return existing
            node_id = unique_node_id(preferred_id)
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=node_id,
                    capability=capability,
                    inputs=inputs or {},
                    depends_on=list(dict.fromkeys(depends_on)),
                    optional=optional,
                )
            )
            return node_id

        ensure_node("ner", "entities", [fetch_node_id], optional=True)
        keyterms_node_id = ensure_node("extract_keyterms", "keyterms", [fetch_node_id], optional=True)
        topics_node_id = ensure_node("topic_model", "topics", [fetch_node_id], inputs={"num_topics": 6}, optional=True)
        if framing_series_definitions:
            normalized_nodes[:] = [node for node in normalized_nodes if not is_static_topic_plot(node, topics_node_id)]
        value_focus_terms = [
            "value", "valuation", "worth", "market value", "transfer fee", "transfer",
            "salary", "salaries", "wage", "wages", "earnings", "contract", "price",
            "priced", "expensive", "cost", "paid",
        ]
        lowered_question = str(question_text or "").lower()
        context_keywords = [term for term in value_focus_terms if term in lowered_question]
        sentiment_node_id = ensure_node("sentiment", "sentiment", [fetch_node_id])
        for node in normalized_nodes:
            if node.node_id == sentiment_node_id:
                updated_inputs = dict(node.inputs)
                updated_inputs.setdefault("window_strategy", "entity_local_context")
                updated_inputs.setdefault("query_focus", question_text)
                if context_keywords:
                    updated_inputs.setdefault("context_keywords", context_keywords)
                node.inputs = updated_inputs
                break
        series_node_id = ensure_node("time_series_aggregate", "series", [sentiment_node_id])
        for node in normalized_nodes:
            if node.node_id == series_node_id:
                updated_inputs = dict(node.inputs)
                updated_inputs.setdefault("documents_node", sentiment_node_id)
                updated_inputs.setdefault("time_field", "published_at")
                updated_inputs.setdefault("bucket_granularity", "month")
                updated_inputs.setdefault("group_by", "target_label")
                updated_inputs.setdefault("metrics", ["average_sentiment", "document_count"])
                node.inputs = updated_inputs
                break
        ensure_node("change_point_detect", "changes", [series_node_id], optional=True)

        sentiment_plot_defaults = {
            "x": "time_bin",
            "y": "average_sentiment",
            "series": "target_label",
            "plot_type": "line",
        }
        has_sentiment_plot = False
        for node in normalized_nodes:
            if node.capability == "plot_artifact" and series_node_id in node.depends_on:
                updated_inputs = dict(node.inputs)
                for key, value in sentiment_plot_defaults.items():
                    updated_inputs.setdefault(key, value)
                node.inputs = updated_inputs
                has_sentiment_plot = True
        if not has_sentiment_plot:
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=unique_node_id("plot_sentiment"),
                    capability="plot_artifact",
                    inputs={"plot_name": "portrayal_sentiment_series", **sentiment_plot_defaults},
                    depends_on=[series_node_id],
                    optional=True,
                )
            )
        if framing_series_definitions:
            framing_series_node_id = next(
                (
                    node.node_id
                    for node in normalized_nodes
                    if node.capability == "time_series_aggregate"
                    and isinstance(node.inputs.get("series", node.inputs.get("series_definitions")), list)
                ),
                "",
            )
            if not framing_series_node_id:
                framing_series_node_id = unique_node_id("framing_series")
                normalized_nodes.append(
                    AgentPlanNode(
                        node_id=framing_series_node_id,
                        capability="time_series_aggregate",
                        inputs={
                            "documents_node": fetch_node_id,
                            "time_field": "published_at",
                            "bucket_granularity": "month",
                            "series": framing_series_definitions,
                            "metrics": ["document_count"],
                        },
                        depends_on=[fetch_node_id],
                    )
                )
            if not any(node.capability == "plot_artifact" and framing_series_node_id in node.depends_on for node in normalized_nodes):
                normalized_nodes.append(
                    AgentPlanNode(
                        node_id=unique_node_id("plot_framing_shift"),
                        capability="plot_artifact",
                        inputs={
                            "plot_name": "framing_shift_over_time",
                            "plot_type": "line",
                            "x": "time_bin",
                            "y": "share_of_documents",
                            "series": "series_name",
                            "title": "Framing shift over time",
                            "x_label": "Month",
                            "y_label": "Share of matched documents",
                        },
                        depends_on=[framing_series_node_id],
                    )
                )
        elif not any(node.capability == "plot_artifact" and topics_node_id in node.depends_on for node in normalized_nodes):
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=unique_node_id("plot_topics"),
                    capability="plot_artifact",
                    inputs={"plot_name": "portrayal_topics"},
                    depends_on=[topics_node_id or keyterms_node_id],
                    optional=True,
                )
            )

        market_ticker = self._infer_market_ticker(question_text)
        if market_ticker and not any(node.capability == "join_external_series" for node in normalized_nodes):
            date_window = self._extract_date_window(question_text)
            aggregate_source_node_id = next(
                (
                    node.node_id
                    for node in normalized_nodes
                    if node.capability == "time_series_aggregate"
                    and isinstance(node.inputs.get("series", node.inputs.get("series_definitions")), list)
                ),
                series_node_id,
            )
            market_node_id = unique_node_id("market_series")
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=market_node_id,
                    capability="join_external_series",
                    inputs={
                        "ticker": market_ticker,
                        "date_from": date_window.get("date_from", ""),
                        "date_to": date_window.get("date_to", ""),
                        "interval": "1mo",
                        "left_key": "time_bin",
                        "right_key": "time_bin",
                        "how": "left",
                    },
                    depends_on=[aggregate_source_node_id],
                )
            )
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=unique_node_id("plot_market_drawdown"),
                    capability="plot_artifact",
                    inputs={
                        "plot_name": f"{market_ticker.lower()}_framing_vs_drawdown",
                        "plot_type": "line",
                        "x": "time_bin",
                        "y": ["share_of_documents", "market_drawdown"],
                        "series": "series_name",
                        "title": f"{market_ticker} framing share and stock drawdowns",
                        "x_label": "Date",
                        "y_label": "Share / drawdown",
                    },
                    depends_on=[market_node_id],
                )
            )

    def _question_with_clarifications(self, state: AgentRunState) -> str:
        history = [str(item).strip() for item in state.clarification_history if str(item).strip()]
        if not history:
            return state.question
        suffix = "\n".join(f"- {item}" for item in history)
        return f"{state.question}\n\nUser clarification history:\n{suffix}"

    def _planning_context_text(self, state: AgentRunState, primary: str = "") -> str:
        parts = [
            str(state.question or "").strip(),
            str(primary or "").strip(),
            str(state.rewritten_question or "").strip(),
        ]
        parts.extend(str(item).strip() for item in state.clarification_history if str(item).strip())
        return " ".join(dict.fromkeys(part for part in parts if part))

    def _extract_date_window(self, text: str) -> dict[str, str]:
        year_values = sorted({int(item) for item in re.findall(r"\b(?:19|20)\d{2}\b", text)})
        if not year_values:
            return {}
        if len(year_values) == 1:
            year = year_values[0]
            if re.search(r"\bduring\b", text, flags=re.IGNORECASE):
                return {"date_from": f"{year}-01-01", "date_to": f"{year}-12-31"}
            year_pattern = re.escape(str(year))
            has_open_start_marker = bool(
                re.search(
                    rf"\b(?:from|since|starting|started|beginning|began|after)\b(?:\W+\w+){{0,4}}\W+{year_pattern}\b",
                    text,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    rf"\b{year_pattern}\b.{0,40}\b(?:onward|onwards|forward|and\s+after|to\s+present|through\s+present)\b",
                    text,
                    flags=re.IGNORECASE,
                )
            )
            if has_open_start_marker:
                return {"date_from": f"{year}-01-01"}
            return {"date_from": f"{year}-01-01", "date_to": f"{year}-12-31"}
        return {
            "date_from": f"{year_values[0]}-01-01",
            "date_to": f"{year_values[-1]}-12-31",
        }

    def _compact_query_terms(self, text: str, preferred_terms: list[str]) -> str:
        found_terms: list[str] = []
        lowered = text.lower()
        blocked_source_tokens = self._source_scope_query_tokens_for_question(text)
        temporal_scope_terms: set[str] = set()
        if re.search(r"\b(?:19|20)\d{2}\b", text) and re.search(
            r"\b(?:from|since|starting|started|beginning|began|after|through|until|to)\b",
            text,
            flags=re.IGNORECASE,
        ):
            temporal_scope_terms.update({"campaign", "presidency", "administration"})
        preferred_stopwords = {
            "actor", "actors", "does", "perceived", "perception", "portrayal", "portrayed",
            "perceptions", "relative", "within", "sentiment", "tone", "tones", "attitude",
            "attitudes", "through", "until", "since", "presidency", "administration",
            "term", "terms", "period", "periods", "relationship", "relationships",
            "relation", "relations", "pattern", "patterns", "trend", "trends",
            "toward", "towards",
            "abbreviation", "abbreviations", "acronym", "acronyms", "connected",
            "description", "descriptions", "describe", "described", "describes",
            "exact", "not", "recurring", "same", "they", "wording",
        } | temporal_scope_terms
        coverage_entity = self._coverage_entity_query(text)
        if coverage_entity:
            return coverage_entity
        for term in preferred_terms:
            if term.lower() in blocked_source_tokens or term.lower() in preferred_stopwords:
                continue
            if term.lower() in lowered and term not in found_terms:
                found_terms.append(term)
        if found_terms:
            compact = " ".join(found_terms)
            return self._expand_topic_query(compact, text) or compact
        tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text) if len(token) > 2]
        stopwords = {
            "how", "did", "from", "into", "with", "what", "when", "which", "that", "this", "those",
            "they", "not",
            "coverage", "framing", "frame", "around", "during", "between", "across", "their", "there",
            "correspond", "corresponded", "media", "shift", "shifted",
            "does",
            "change", "changed", "changes", "changing", "evolve", "evolved", "evolution", "over", "time",
            "different", "differently", "difference", "differences", "compare", "compared", "comparison", "versus",
            "dominate", "dominates", "dominated", "dominant", "entity", "entities", "named", "actor", "actors",
            "public", "discourse", "newspaper", "newspapers", "report", "reports", "reported",
            "perceived", "perception", "portrayal", "portrayed",
            "perceptions", "relative", "within",
            "sentiment", "tone", "tones", "attitude", "attitudes", "through", "until", "since",
            "presidency", "administration", "term", "terms", "period", "periods",
            "relationship", "relationships", "relation", "relations", "pattern", "patterns",
            "trend", "trends", "heightened", "elevated", "specific", "subperiod", "subperiods",
            "his", "her", "its", "our", "your", "toward", "towards",
            "abbreviation", "abbreviations", "acronym", "acronyms", "connected",
            "description", "descriptions", "describe", "described", "describes",
            "exact", "recurring", "same", "wording",
        } | temporal_scope_terms
        filtered = [token for token in tokens if token.lower() not in stopwords and token.lower() not in blocked_source_tokens]
        compact = " ".join(filtered[:8]).strip()
        return self._expand_topic_query(compact, text) or compact

    def _coverage_entity_query(self, text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        patterns = (
            r"\bcoverage\s+of\s+(?P<entity>[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,3})\b",
            r"\b(?P<entity>[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,3})\s+coverage\b",
        )
        trailing_stop = re.compile(
            r"\b(?:change|changed|changes|changing|shift|shifted|evolve|evolved|between|from|with|and|compare|compared|peaks?|stock|performance)\b",
            flags=re.IGNORECASE,
        )
        blocked = {
            "How",
            "What",
            "Which",
            "When",
            "Where",
            "Why",
            "News",
            "Media",
            "Major",
        }
        for pattern in patterns:
            match = re.search(pattern, value)
            if not match:
                continue
            raw_entity = str(match.group("entity") or "").strip(" ,.;:!?()[]{}")
            if not raw_entity:
                continue
            stop_match = trailing_stop.search(raw_entity)
            if stop_match:
                raw_entity = raw_entity[: stop_match.start()].strip()
            tokens = [
                token.strip(" ,.;:!?()[]{}")
                for token in raw_entity.split()
                if token.strip(" ,.;:!?()[]{}")
            ]
            while tokens and tokens[0] in blocked:
                tokens = tokens[1:]
            while tokens and tokens[-1] in blocked:
                tokens = tokens[:-1]
            if not tokens:
                continue
            entity = " ".join(tokens).strip()
            if entity and any(char.isalpha() for char in entity):
                return entity
        return ""

    def _expand_topic_query(self, query_text: str, question_text: str) -> str:
        combined = f"{query_text} {question_text}".lower()
        for expansion in TOPIC_QUERY_EXPANSIONS:
            if any(re.search(rf"\b{re.escape(trigger)}\b", combined, flags=re.IGNORECASE) for trigger in expansion["triggers"]):
                return str(expansion["query"])
        return ""

    def _query_needs_topical_repair(self, planned_query: str, fallback_query: str) -> bool:
        planned_tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", planned_query)]
        fallback_tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", fallback_query)]
        if not planned_tokens or not fallback_tokens:
            return False
        broad_scope_tokens = {
            "america",
            "american",
            "usa",
            "united",
            "states",
            "state",
            "us",
            "swiss",
            "switzerland",
            "media",
            "news",
            "newspaper",
            "newspapers",
            "outlet",
            "outlets",
        }
        if len(planned_tokens) <= 2 and all(token in broad_scope_tokens for token in planned_tokens):
            topical_tokens = [token for token in fallback_tokens if token not in broad_scope_tokens]
            return len(topical_tokens) >= 2
        query_scaffold_tokens = {
            *broad_scope_tokens,
            "actor",
            "actors",
            "attitude",
            "attitudes",
            "campaign",
            "change",
            "changed",
            "changes",
            "changing",
            "coverage",
            "did",
            "does",
            "elevated",
            "evolve",
            "evolved",
            "evolution",
            "explain",
            "explained",
            "framing",
            "heightened",
            "her",
            "his",
            "its",
            "pattern",
            "patterns",
            "period",
            "periods",
            "portrayal",
            "portrayed",
            "presidency",
            "relation",
            "relations",
            "relationship",
            "relationships",
            "sentiment",
            "specific",
            "subperiod",
            "subperiods",
            "term",
            "terms",
            "through",
            "tone",
            "tones",
            "trend",
            "trends",
            "until",
        }
        planned_set = set(planned_tokens)
        fallback_set = set(fallback_tokens)
        if fallback_set and fallback_set.issubset(planned_set):
            extras = planned_set - fallback_set
            noisy_extras = extras & query_scaffold_tokens
            meaningful_extras = extras - query_scaffold_tokens
            if noisy_extras and (not meaningful_extras or len(noisy_extras) >= 2):
                return True
        if len(fallback_tokens) >= 2 and len(planned_tokens) >= len(fallback_tokens) + 3:
            overlap = len(planned_set & fallback_set)
            if overlap >= min(2, len(fallback_set)) and len(planned_set & query_scaffold_tokens) >= 2:
                return True
        return False

    def _query_is_hollow_after_filter_removal(self, planned_query: str, fallback_query: str) -> bool:
        planned_tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", planned_query)
        ]
        fallback_tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", fallback_query)
        ]
        if not fallback_tokens:
            return False
        hollow_tokens = {
            "and",
            "co",
            "company",
            "corp",
            "corporation",
            "gmbh",
            "inc",
            "limited",
            "llc",
            "ltd",
            "motors",
            "or",
            "plc",
            "sa",
            "se",
            "source",
            "the",
        }
        meaningful_tokens = [
            token for token in planned_tokens
            if token not in hollow_tokens and len(token) > 1
        ]
        if not meaningful_tokens:
            return bool(planned_tokens)
        if len(meaningful_tokens) <= 1 and len(fallback_tokens) >= 2:
            return not bool(set(meaningful_tokens) & set(fallback_tokens))
        return False

    def _query_has_unbalanced_parentheses(self, query_text: str) -> bool:
        depth = 0
        quote: str | None = None
        for char in str(query_text or ""):
            if quote:
                if char == quote:
                    quote = None
                continue
            if char in {"'", '"'}:
                quote = char
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return True
        return depth != 0 or quote is not None

    def _query_lost_meaning_after_source_filter_removal(self, candidate_query: str, fallback_query: str) -> bool:
        syntax_tokens = {"and", "or", "not"}
        candidate_tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", str(candidate_query or ""))
            if token.lower() not in syntax_tokens
        ]
        fallback_tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", str(fallback_query or ""))
            if token.lower() not in syntax_tokens
        ]
        if len(fallback_tokens) < 2 or not candidate_tokens:
            return False
        candidate_set = set(candidate_tokens)
        fallback_set = set(fallback_tokens)
        if not candidate_set or not candidate_set.issubset(fallback_set):
            return False
        if not fallback_set - candidate_set:
            return False
        has_fragment_syntax = bool(
            re.search(r"\bOR\b", str(candidate_query or ""), flags=re.IGNORECASE)
            or re.search(r'"[^"]*\s+"|\'[^\']*\s+\'', str(candidate_query or ""))
        )
        return has_fragment_syntax or len(candidate_set) < len(fallback_set)

    def _compact_comparison_entity_query(self, query_text: str) -> str:
        text = str(query_text or "").strip()
        lowered = text.lower()
        if not re.search(
            r"\b(?:evolve|evolved|value|valuation|worth|perceived|perception|perceptions|portrayal|portrayed|relative)\b",
            lowered,
        ):
            return ""
        capitalized = [
            token
            for token in re.findall(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Z]?[A-Za-z0-9]+)?\b", text)
            if token.lower()
            not in {
                "how",
                "what",
                "which",
                "when",
                "where",
                "why",
                "american",
                "swiss",
                "republican",
            }
        ]
        if len(capitalized) < 3:
            return ""
        if len(capitalized) >= 4:
            entity_terms = capitalized[1::2]
        else:
            entity_terms = capitalized[1:]
        topic_terms: list[str] = []
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", text):
            lowered_token = token.lower()
            if lowered_token in {"value", "valuation", "worth", "market", "contract", "salary", "transfer", "price"}:
                if lowered_token not in {item.lower() for item in topic_terms}:
                    topic_terms.append(lowered_token)
        compacted = [*entity_terms[:4], *topic_terms[:3]]
        return " ".join(dict.fromkeys(term for term in compacted if term)).strip()

    def _repair_search_query(self, planned_query: str, fallback_query: str) -> str:
        planned = str(planned_query or "").strip()
        fallback = str(fallback_query or "").strip()
        if not planned:
            return fallback
        had_source_filter = bool(re.search(r"\bsource\s*:", planned, flags=re.IGNORECASE))
        planned_without_source_filters = self._remove_source_field_filters(planned)
        repair_candidate = planned_without_source_filters or planned
        if had_source_filter and fallback and self._query_has_unbalanced_parentheses(repair_candidate):
            return fallback
        if (
            had_source_filter
            and fallback
            and self._query_lost_meaning_after_source_filter_removal(repair_candidate, fallback)
        ):
            return fallback
        compacted_comparison = self._compact_comparison_entity_query(repair_candidate)
        if compacted_comparison:
            return compacted_comparison
        if fallback and self._query_needs_topical_repair(repair_candidate, fallback):
            return fallback
        if fallback and self._query_is_hollow_after_filter_removal(repair_candidate, fallback):
            return fallback
        return repair_candidate

    def _source_candidate_tokens(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+(?:[._'-][A-Za-z0-9]+)*", str(text or ""))

    def _trim_source_candidate(self, text: str) -> str:
        cleaned = re.sub(r"\bsource\s*:\s*", " ", str(text or ""), flags=re.IGNORECASE)
        cleaned = re.sub(r"^[\s\"'`({\[]+|[\s\"'`)}\].,;:!?]+$", "", cleaned)
        cleaned = re.sub(
            r"^(?:how\s+(?:does|do|did)\s+|what\s+(?:does|do|did)\s+|which\s+|compare\s+|comparison\s+of\s+|difference\s+between\s+|between\s+)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s+\b(?:report|reports|reported|cover|covers|covered|coverage|write|writes|wrote|explain|explains|explained|frame|frames|framed|differ|differs|different|differently|compare|compared|comparison|on|about|over|during|from|only)\b.*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        tokens = self._source_candidate_tokens(cleaned)
        while tokens and tokens[0].lower().rstrip(".") in SOURCE_CANDIDATE_LEADING_NOISE:
            tokens = tokens[1:]
        while tokens and tokens[-1].lower().rstrip(".") in SOURCE_CANDIDATE_TRAILING_NOISE:
            tokens = tokens[:-1]
        topic_tail_tokens = {
            str(trigger).lower()
            for expansion in TOPIC_QUERY_EXPANSIONS
            for trigger in expansion.get("triggers", ())
        }
        while len(tokens) > 1 and tokens[-1].lower().rstrip(".") in topic_tail_tokens:
            tokens = tokens[:-1]
        if not tokens:
            return ""
        if len(tokens) > 5:
            return ""
        if tokens[0].isupper() and len(tokens[0]) <= 8 and any(token.islower() for token in tokens[1:]):
            tokens = tokens[:1]
        else:
            prefix: list[str] = []
            for token in tokens:
                token_clean = token.strip("._'")
                if not token_clean:
                    continue
                if token_clean.isupper() or token_clean[:1].isupper() or "-" in token_clean:
                    prefix.append(token)
                    continue
                break
            if prefix and len(prefix) < len(tokens) and (len(prefix) >= 2 or len(tokens) > 4):
                tokens = prefix
        candidate = " ".join(tokens).strip()
        if not candidate:
            return ""
        if candidate.lower() in SOURCE_CANDIDATE_GENERIC_VALUES:
            return ""
        if re.fullmatch(r"(?:18|19|20)\d{2}", candidate):
            return ""
        if not re.search(r"[A-Za-z]", candidate):
            return ""
        return candidate

    def _source_scope_candidate_names(self, text: str) -> list[str]:
        value = str(text or "").strip()
        if not value:
            return []
        source_context = re.search(
            r"\b(?:media|newspaper|newspapers|press|outlet|outlets|source|sources|report|reports|reported|cover|covers|coverage|write|writes|wrote)\b",
            value,
            flags=re.IGNORECASE,
        )
        if not source_context:
            return []
        candidates: list[str] = []
        patterns = (
            r"(.+?)\s+\bvs\.?\b\s+(.+?)(?=\s+\b(?:report|reports|reported|cover|covers|coverage|write|writes|wrote|differ|differs|differently|on|about|over|during)\b|[?.!,;]|$)",
            r"(.+?)\s+\bversus\b\s+(.+?)(?=\s+\b(?:report|reports|reported|cover|covers|coverage|write|writes|wrote|differ|differs|differently|on|about|over|during)\b|[?.!,;]|$)",
            r"\b(?:compare|comparison\s+of|difference\s+between|between)\b\s+(.+?)\s+\b(?:and|with|to)\b\s+(.+?)(?=\s+\b(?:report|reports|reported|cover|covers|coverage|write|writes|wrote|differ|differs|differently|on|about|over|during)\b|[?.!,;]|$)",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, value, flags=re.IGNORECASE):
                left = self._trim_source_candidate(match.group(1))
                right = self._trim_source_candidate(match.group(2))
                for candidate in (left, right):
                    if candidate and candidate.lower() not in {item.lower() for item in candidates}:
                        candidates.append(candidate)
        return candidates

    def _explicit_source_scope_matches(self, text: str) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for candidate in self._source_scope_candidate_names(text):
            tokens = tuple(token.lower().rstrip(".") for token in self._source_candidate_tokens(candidate))
            matches.append({"filters": (candidate,), "tokens": tokens, "candidate": candidate})
        return matches

    def _explicit_source_scope_is_active(self, text: str, matches: list[dict[str, Any]]) -> bool:
        return bool(matches)

    def _source_scope_filters_for_question(self, text: str) -> list[str]:
        if (
            self._needs_semantic_similarity_analysis(text)
            or self._needs_syntax_role_analysis(text)
            or self._needs_attributed_explanation_analysis(text)
        ):
            return []
        filters: list[str] = []
        explicit_matches = self._explicit_source_scope_matches(text)
        if self._explicit_source_scope_is_active(text, explicit_matches):
            for match in explicit_matches:
                filters.extend(str(item) for item in match["filters"])
        return list(dict.fromkeys(item for item in filters if item))

    def _source_scope_query_tokens_for_question(self, text: str) -> set[str]:
        if (
            self._needs_semantic_similarity_analysis(text)
            or self._needs_syntax_role_analysis(text)
            or self._needs_attributed_explanation_analysis(text)
        ):
            return set()
        explicit_matches = self._explicit_source_scope_matches(text)
        if not self._explicit_source_scope_is_active(text, explicit_matches):
            return set()
        blocked: set[str] = set()
        for match in explicit_matches:
            blocked.update(str(item).lower() for item in match["tokens"])
        return blocked

    def _described_source_scope_phrases(self, text: str) -> list[str]:
        value = str(text or "").strip()
        if not value:
            return []
        descriptors: list[str] = []
        source_noun = r"(?:media|newspapers?|press|outlets?|sources?)"
        descriptor = r"[A-Za-z0-9&.'-]+(?:\s+[A-Za-z0-9&.'-]+){0,4}"
        patterns = (
            rf"\b(?:in|from|by|among|across|within|of)\s+(?:the\s+)?(?P<descriptor>{descriptor})\s+{source_noun}\b",
            rf"\b(?P<descriptor>{descriptor})\s+{source_noun}\s+(?:reported|reports|report|covered|covers|cover|explained|explains|explain|portrayed|portrays|portray|framed|frames|frame|wrote|writes|write)\b",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, value, flags=re.IGNORECASE):
                phrase = " ".join(str(match.group("descriptor")).split()).strip(" ,.;:!?()[]{}")
                lowered = phrase.lower()
                if not phrase or lowered in SOURCE_CANDIDATE_GENERIC_VALUES:
                    continue
                if lowered in {"social", "news", "mainstream", "online", "legacy", "traditional"}:
                    continue
                if lowered not in {item.lower() for item in descriptors}:
                    descriptors.append(phrase)
        return descriptors

    def _question_requests_described_source_scope(self, text: str) -> bool:
        return bool(self._described_source_scope_phrases(text))

    def _remove_source_field_filters(self, query: str) -> str:
        cleaned = re.sub(
            r"\s*(?:AND|OR)?\s*\(?\s*source\s*:\s*\([^)]+\)\s*\)?",
            " ",
            str(query or ""),
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*(?:AND|OR)?\s*\(?\s*source\s*:\s*(?:\"[^\"]+\"|'[^']+'|[A-Za-z0-9_.-]+)\s*\)?",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        return " ".join(cleaned.split()).strip()

    def _remove_explicit_source_mentions_from_query(self, query: str, question_text: str) -> str:
        cleaned = str(query or "").strip()
        matches = self._explicit_source_scope_matches(question_text)
        if not self._explicit_source_scope_is_active(question_text, matches):
            return cleaned
        for source_match in matches:
            candidate = str(source_match.get("candidate", "")).strip()
            if candidate:
                phrase_pattern = r"[\s._'-]+".join(re.escape(token) for token in self._source_candidate_tokens(candidate))
                if phrase_pattern:
                    cleaned = re.sub(rf"\b{phrase_pattern}\b", " ", cleaned, flags=re.IGNORECASE)
            for token in source_match["tokens"]:
                cleaned = re.sub(rf"\b{re.escape(str(token))}\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:vs\.?|versus|compared\s+to|compare|between)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\(\s*\)", " ", cleaned)
        cleaned = re.sub(r"\b(?:AND|OR)\s*(?=\)|$)", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*(?:AND|OR)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\(\s*(?:AND|OR)\b", "(", cleaned, flags=re.IGNORECASE)
        return " ".join(cleaned.split()).strip(" ()")

    def _apply_source_scope_to_query(self, query: str, question_text: str) -> str:
        cleaned_query = str(query or "").strip()
        if not cleaned_query:
            return cleaned_query
        filters = self._source_scope_filters_for_question(question_text)
        if filters:
            cleaned_query = self._remove_source_field_filters(cleaned_query)
            cleaned_query = self._remove_explicit_source_mentions_from_query(cleaned_query, question_text)
            cleaned_query = self._expand_topic_query(cleaned_query, question_text) or cleaned_query
            if not cleaned_query:
                cleaned_query = self._compact_query_terms(question_text, self._query_anchor_terms(question_text)) or question_text
        if re.search(r"\bsource\s*:\s*\*", cleaned_query, flags=re.IGNORECASE):
            cleaned_query = re.sub(
                r"\s*(?:AND|OR)?\s*\(?\s*source\s*:\s*\*\s*\)?",
                " ",
                cleaned_query,
                flags=re.IGNORECASE,
            )
            cleaned_query = " ".join(cleaned_query.split()).strip()
        elif re.search(r"\bsource\s*:", cleaned_query, flags=re.IGNORECASE):
            if not filters:
                cleaned_query = self._remove_source_field_filters(cleaned_query)
                cleaned_query = " ".join(cleaned_query.split()).strip()
                if not cleaned_query:
                    cleaned_query = self._compact_query_terms(question_text, self._query_anchor_terms(question_text)) or question_text
                elif cleaned_query.lower().strip("() ") in SOURCE_CANDIDATE_GENERIC_VALUES:
                    cleaned_query = self._compact_query_terms(question_text, self._query_anchor_terms(question_text)) or cleaned_query
            else:
                return cleaned_query
        if not filters:
            return self._expand_topic_query(cleaned_query, question_text) or cleaned_query
        rendered = " OR ".join(f'"{item}"' for item in filters)
        return f"({cleaned_query}) AND source:({rendered})"

    def _query_anchor_terms(self, text: str) -> list[str]:
        anchors: list[str] = []
        seen: set[str] = set()
        blocked_terms: set[str] = set()
        blocked_source_tokens = self._source_scope_query_tokens_for_question(text)
        if re.search(r"\b(?:19|20)\d{2}\b", text) and re.search(
            r"\b(?:from|since|starting|started|beginning|began|after|through|until|to)\b",
            text,
            flags=re.IGNORECASE,
        ):
            blocked_terms.update({"campaign", "presidency", "administration"})
        stopwords = {
            "across",
            "aggregate",
            "aggregated",
            "all",
            "analysis",
            "analyze",
            "and",
            "association",
            "associations",
            "associated",
            "around",
            "available",
            "between",
            "breakdown",
            "collection",
            "common",
            "compare",
            "compared",
            "comparison",
            "corpus",
            "coverage",
            "dataset",
            "did",
            "different",
            "differently",
            "difference",
            "differences",
            "distribution",
            "document",
            "documents",
            "does",
            "during",
            "for",
            "from",
            "frequency",
            "how",
            "attitude",
            "attitudes",
            "explained",
            "explain",
            "her",
            "identified",
            "identifying",
            "identify",
            "include",
            "included",
            "including",
            "individual",
            "his",
            "it",
            "its",
            "lemma",
            "lemmas",
            "most",
            "noun",
            "nouns",
            "actor",
            "actors",
            "related",
            "relevant",
            "report",
            "reports",
            "reported",
            "result",
            "results",
            "row",
            "rows",
            "sentiment",
            "such",
            "that",
            "the",
            "their",
            "there",
            "this",
            "those",
            "through",
            "tone",
            "tones",
            "toward",
            "towards",
            "trend",
            "trends",
            "until",
            "used",
            "using",
            "what",
            "when",
            "where",
            "with",
            "which",
            "who",
            "why",
            "change",
            "changed",
            "changes",
            "changing",
            "evolve",
            "evolved",
            "evolution",
            "over",
            "time",
            "america",
            "american",
            "usa",
            "united",
            "states",
            "state",
            "swiss",
            "switzerland",
            "media",
            "news",
            "newspaper",
            "newspapers",
            "entity",
            "entities",
            "named",
            "dominate",
            "dominated",
            "public",
            "discourse",
            "presidency",
            "administration",
            "term",
            "terms",
            "period",
            "periods",
            "relationship",
            "relationships",
            "relation",
            "relations",
            "pattern",
            "patterns",
            "heightened",
            "elevated",
            "specific",
            "subperiod",
            "subperiods",
            "perceived",
            "perception",
            "perceptions",
            "portrayal",
            "portrayed",
            "relative",
            "versus",
            "within",
        }

        def _add_anchor(token: str) -> None:
            for part in re.findall(r"[A-Za-z][A-Za-z0-9]+", str(token or "")):
                lowered = part.lower()
                if (
                    len(lowered) < 3
                    or lowered in stopwords
                    or lowered in blocked_source_tokens
                    or lowered in seen
                    or lowered in blocked_terms
                ):
                    continue
                seen.add(lowered)
                anchors.append(part)

        for match in re.finditer(
            r"['\"]?([A-Za-z][A-Za-z0-9-]+)['\"]?\s+(?:means|refers\s+to|interpreted\s+as)\s+['\"]?([A-Za-z][A-Za-z0-9-]+)['\"]?",
            text,
            flags=re.IGNORECASE,
        ):
            blocked_terms.update(part.lower() for part in re.findall(r"[A-Za-z][A-Za-z0-9]+", match.group(1)))
            _add_anchor(match.group(2))
        if anchors:
            return anchors[:8]
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Z]?[A-Za-z0-9]+)?\b", text):
            _add_anchor(match.group(0).strip())
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]+", text):
            _add_anchor(token)
        return anchors[:8]

    def _infer_market_ticker(self, text: str) -> str:
        return _infer_market_ticker_from_text(text)

    def _search_inputs_for_question(
        self,
        question_text: str,
        *,
        overrides: dict[str, Any] | None = None,
        query_text: str = "",
        lightweight: bool = False,
    ) -> dict[str, Any]:
        merged_inputs = dict(overrides or {})
        budget = infer_retrieval_budget(
            question_text,
            inputs=merged_inputs,
            configured_mode=os.getenv("CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE", "hybrid"),
            lightweight=lightweight,
        )
        search_inputs = budget.to_inputs()
        fallback_query = self._compact_query_terms(question_text, self._query_anchor_terms(question_text)) or question_text
        if query_text:
            search_inputs["query"] = self._apply_source_scope_to_query(query_text, question_text)
        elif "query" in merged_inputs and str(merged_inputs.get("query", "")).strip():
            repaired_query = self._repair_search_query(str(merged_inputs["query"]), fallback_query)
            search_inputs["query"] = self._apply_source_scope_to_query(repaired_query, question_text)
        elif str(search_inputs.get("query", "")).strip():
            repaired_query = self._repair_search_query(str(search_inputs["query"]), fallback_query)
            search_inputs["query"] = self._apply_source_scope_to_query(repaired_query, question_text)
        for key in ("top_k", "lexical_top_k", "dense_top_k", "fallback_top_k", "rerank_top_k", "date_from", "date_to", "year_balance", "retrieve_all", "retrieval_strategy"):
            if key in merged_inputs and merged_inputs.get(key) not in (None, ""):
                search_inputs[key] = merged_inputs[key]
        if "date_from" not in search_inputs and "date_to" not in search_inputs:
            search_inputs.update(self._extract_date_window(question_text))
        return search_inputs

    def _broad_scope_clarification_is_sufficient(self, state: AgentRunState) -> tuple[bool, list[str]]:
        question_text = f"{state.question}\n" + "\n".join(state.clarification_history)
        lowered = question_text.lower()
        assumptions: list[str] = []
        sufficient = False

        has_broad_aggregation = any(
            pattern.search(lowered)
            for pattern in (
                re.compile(r"\boverall\b"),
                re.compile(r"\baggregate\b"),
                re.compile(r"\bbroadly\b"),
                re.compile(r"\bquantitative\b"),
                re.compile(r"\bquantitativ\b"),
            )
        )
        has_explicit_range = bool(re.search(r"\b20\d{2}\s*-\s*20\d{2}\b", lowered)) or (
            bool(re.search(r"\b(?:19|20)\d{2}\b", lowered))
            and any(token in lowered for token in ("from", "between", "to", "-", "through"))
        )
        has_time_granularity = any(
            pattern.search(lowered)
            for pattern in (
                re.compile(r"\bmonthly\b"),
                re.compile(r"\bmonth(?:ly)?\b"),
                re.compile(r"\bweekly\b"),
                re.compile(r"\bdaily\b"),
                re.compile(r"\bquarterly\b"),
            )
        )
        has_all_scope = any(
            pattern.search(lowered)
            for pattern in (
                re.compile(r"\ball phases\b"),
                re.compile(r"\beverything\b"),
                re.compile(r"\boverall\b"),
                re.compile(r"\ball\b"),
            )
        )
        asks_grouping = any(token in state.question.lower() for token in ("group", "groups", "region", "regions", "type", "types", "source", "sources", "outlet", "outlets"))
        asks_time_or_market_detail = any(
            token in state.question.lower()
            for token in ("stock", "drawdown", "share price", "market", "valuation", "granular", "monthly", "weekly", "daily")
        )

        if has_broad_aggregation and asks_grouping:
            sufficient = True
            assumptions.append("Interpret the requested group scope as an aggregate comparison over available metadata groups.")
        if has_broad_aggregation:
            sufficient = True
            assumptions.append("Prefer a quantitative aggregate summary where possible and note any metadata limits explicitly.")
        if asks_time_or_market_detail and has_explicit_range:
            sufficient = True
            assumptions.append("Use the explicitly provided time range for retrieval, comparison, and market alignment.")
        if asks_time_or_market_detail and has_time_granularity:
            sufficient = True
            assumptions.append("Use the requested time granularity for temporal aggregation and plots.")
        if has_all_scope:
            sufficient = True
            assumptions.append("Cover the full requested scope rather than a small illustrative subset.")
        if asks_grouping and sufficient:
            assumptions.append("Use available metadata fields as grouping proxies; if labels are sparse, report the limitation.")
        return sufficient, assumptions

    def rephrase_or_clarify(self, state: AgentRunState) -> PlannerAction:
        rejection_reason = rejection_reason_for_question(state.question)
        if rejection_reason:
            return PlannerAction(action="grounded_rejection", rejection_reason=rejection_reason)

        enriched_question = self._question_with_clarifications(state)
        rewritten, assumptions = normalize_question_text(enriched_question)
        if self.llm_client is None:
            self._record_llm_trace(
                state,
                stage="rephrase_or_clarify",
                used_fallback=True,
                note="No LLM client configured; heuristic clarification policy used.",
            )
            if state.clarification_history:
                return PlannerAction(
                    action="accept_with_assumptions",
                    rewritten_question=rewritten,
                    assumptions=assumptions,
                )
            if " between groups " in f" {state.question.lower()} " and not state.force_answer:
                return PlannerAction(
                    action="ask_clarification",
                    rewritten_question=rewritten,
                    clarification_question="Which exact groups, outlets, or entities should be compared?",
                    assumptions=assumptions,
                )
            return PlannerAction(
                action="accept_with_assumptions",
                rewritten_question=rewritten,
                assumptions=assumptions,
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the rephrasing and clarification module for a corpus agent operating over a user-provided corpus. "
                    "Return JSON with keys action, rewritten_question, clarification_question, assumptions, rejection_reason, message. "
                    "Allowed actions: ask_clarification, accept_with_assumptions, grounded_rejection. "
                    "Reject hidden-motive questions. Ask clarification only if workflow changes materially. "
                    "Treat clarification_history as authoritative user follow-up memory. "
                    "If prior follow-up answers resolve part of a multi-part clarification, ask only for the remaining unresolved detail instead of repeating the whole clarification prompt. "
                    "Do not assume the corpus is news, media, finance, or any specific domain unless the question or corpus schema indicates that."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": state.question,
                        "clarification_history": list(state.clarification_history),
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        try:
            trace = self.llm_client.complete_json_trace(
                messages,
                model=self.llm_config.planner_model,
                temperature=0.0,
            )
            payload = dict(trace["parsed_json"])
            self._record_llm_trace(state, stage="rephrase_or_clarify", trace=trace)
            action = PlannerAction.from_dict(payload)
            if action.action == "ask_clarification" and state.force_answer:
                forced_rewrite = rewritten or state.question
                forced_assumptions = list(dict.fromkeys(list(action.assumptions) + assumptions + [
                    "force_answer=true: proceeded with best-effort assumptions instead of waiting for clarification."
                ]))
                return PlannerAction(
                    action="accept_with_assumptions",
                    rewritten_question=forced_rewrite,
                    assumptions=forced_assumptions,
                    message=action.message,
                )
            sufficient, clarification_assumptions = self._broad_scope_clarification_is_sufficient(state)
            if action.action == "ask_clarification" and sufficient:
                return PlannerAction(
                    action="accept_with_assumptions",
                    rewritten_question=action.rewritten_question or rewritten or state.question,
                    assumptions=list(dict.fromkeys(list(action.assumptions) + assumptions + clarification_assumptions)),
                    message="Broad-scope clarification accepted; proceeding with explicit assumptions.",
                )
            if not action.rewritten_question:
                action.rewritten_question = rewritten
            if assumptions:
                action.assumptions = list(dict.fromkeys(list(action.assumptions) + assumptions))
            return action
        except Exception as exc:
            self._record_llm_trace(
                state,
                stage="rephrase_or_clarify",
                used_fallback=True,
                error=str(exc),
                note="LLM rephrase step failed; heuristic fallback used.",
            )
            return PlannerAction(
                action="accept_with_assumptions",
                rewritten_question=rewritten,
                assumptions=assumptions,
            )

    def _heuristic_plan(self, state: AgentRunState) -> PlannerAction:
        rewritten = state.rewritten_question or self._question_with_clarifications(state)
        text = rewritten.lower()
        anchor_terms = self._query_anchor_terms(rewritten)
        compact_query = self._compact_query_terms(rewritten, anchor_terms)
        query_text = compact_query or rewritten
        inferred_budget = infer_retrieval_budget(
            rewritten,
            configured_mode=os.getenv("CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE", "hybrid"),
        )
        heuristic_assumptions: list[str] = []
        if inferred_budget.retrieval_strategy == "exhaustive_analytic":
            heuristic_assumptions.append(
                "This question calls for exhaustive corpus analysis, so retrieval will materialize the full lexical match set before downstream analysis."
            )
        elif inferred_budget.retrieval_strategy == "semantic_exploratory":
            heuristic_assumptions.append(
                "This question is semantically open-ended, so retrieval will favor exploratory semantic recall over a tiny exact-match evidence set."
            )
        elif inferred_budget.scope in {"comparative", "broad", "exhaustive"}:
            heuristic_assumptions.append(
                "Heuristic fallback increased retrieval budget to improve recall for this broad analytical question."
            )
        if self._needs_semantic_similarity_analysis(rewritten):
            search_inputs = self._search_inputs_for_question(
                rewritten,
                query_text=query_text,
                overrides={"retrieval_strategy": "semantic_exploratory"},
            )
            dag = AgentPlanDAG(
                nodes=[
                    AgentPlanNode("search", "db_search", inputs=search_inputs),
                    AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                    AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                    AgentPlanNode("doc_embeddings", "doc_embeddings", depends_on=["fetch"]),
                    AgentPlanNode("similarity_index", "similarity_index", depends_on=["fetch"]),
                    AgentPlanNode("similarity_pairwise", "similarity_pairwise", inputs={"query": rewritten}, depends_on=["fetch"]),
                    AgentPlanNode("word_embeddings", "word_embeddings", depends_on=["fetch"]),
                    AgentPlanNode("acronyms", "extract_acronyms", depends_on=["fetch"]),
                    AgentPlanNode("keyterms", "extract_keyterms", inputs={"top_k": 40}, depends_on=["fetch"]),
                    AgentPlanNode("entities", "ner", depends_on=["fetch"], optional=True),
                    AgentPlanNode("entity_link", "entity_link", depends_on=["entities"], optional=True),
                    AgentPlanNode(
                        "series",
                        "time_series_aggregate",
                        inputs={
                            "documents_node": "fetch",
                            "time_field": "published_at",
                            "bucket_granularity": "month",
                            "metrics": ["document_count"],
                        },
                        depends_on=["fetch"],
                    ),
                    AgentPlanNode("changes", "change_point_detect", depends_on=["series"], optional=True),
                    AgentPlanNode(
                        "plot",
                        "plot_artifact",
                        inputs={
                            "plot_name": "semantic_term_coverage_over_time",
                            "plot_type": "line",
                            "x": "time_bin",
                            "y": "document_count",
                            "series": "series_name",
                        },
                        depends_on=["series"],
                        optional=True,
                    ),
                    AgentPlanNode(
                        "evidence",
                        "build_evidence_table",
                        inputs={"task": "semantic_similarity_evidence"},
                        depends_on=["keyterms", "doc_embeddings", "similarity_index", "similarity_pairwise", "entity_link"],
                    ),
                ],
                metadata={"question_family": "semantic_similarity_terms"},
            )
            return PlannerAction(
                action="emit_plan_dag",
                rewritten_question=state.rewritten_question or rewritten,
                assumptions=list(heuristic_assumptions),
                plan_dag=dag,
            )
        if self._needs_syntax_role_analysis(rewritten):
            syntax_overrides = (
                {"retrieval_strategy": "exhaustive_analytic", "retrieve_all": True, "top_k": 0}
                if self._syntax_role_needs_population_analysis(rewritten)
                else {"retrieval_strategy": "precision_ranked"}
            )
            search_inputs = self._search_inputs_for_question(
                rewritten,
                query_text=query_text,
                overrides=syntax_overrides,
            )
            date_window = self._extract_date_window(rewritten)
            nodes = [
                AgentPlanNode("search", "db_search", inputs=search_inputs),
                AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                AgentPlanNode("sentences", "sentence_split", depends_on=["fetch"]),
                AgentPlanNode("tokens", "tokenize", depends_on=["fetch"]),
                AgentPlanNode("pos", "pos_morph", depends_on=["fetch"]),
                AgentPlanNode("lemmas", "lemmatize", depends_on=["fetch"]),
                AgentPlanNode("dependencies", "dependency_parse", depends_on=["fetch"]),
                AgentPlanNode("svo_triples", "extract_svo_triples", depends_on=["fetch"]),
                AgentPlanNode("noun_chunks", "noun_chunks", depends_on=["fetch"]),
                AgentPlanNode("entities", "ner", depends_on=["fetch"], optional=True),
                AgentPlanNode("entity_link", "entity_link", depends_on=["entities"], optional=True),
                AgentPlanNode("quotes", "quote_extract", depends_on=["fetch"]),
                AgentPlanNode("quote_attribution", "quote_attribute", depends_on=["quotes"]),
                AgentPlanNode(
                    "syntax_role_counts",
                    "time_series_aggregate",
                    inputs={
                        "group_by": "actor_group",
                        "metrics": ["mention_count"],
                        "bucket_granularity": "year",
                        "fallback_time_bin": date_window.get("date_from", ""),
                        "top_k": 8,
                    },
                    depends_on=["svo_triples"],
                    optional=True,
                ),
                AgentPlanNode(
                    "syntax_target_counts",
                    "time_series_aggregate",
                    inputs={
                        "group_by": "target_group",
                        "metrics": ["mention_count"],
                        "bucket_granularity": "year",
                        "fallback_time_bin": date_window.get("date_from", ""),
                        "top_k": 8,
                    },
                    depends_on=["svo_triples"],
                    optional=True,
                ),
            ]
            if self._question_requires_capability(rewritten, "plot_artifact"):
                nodes.append(
                    AgentPlanNode(
                        "plot",
                        "plot_artifact",
                        inputs={
                            "plot_name": "syntax_role_patterns",
                            "top_k": 8,
                            "title": "Syntax role pattern summary",
                        },
                        depends_on=["syntax_role_counts"],
                        optional=True,
                    )
                )
            nodes.append(
                AgentPlanNode(
                    "evidence",
                    "build_evidence_table",
                    inputs={"task": "syntax_role_evidence"},
                    depends_on=[
                        "sentences",
                        "tokens",
                        "pos",
                        "lemmas",
                        "dependencies",
                        "noun_chunks",
                        "quote_attribution",
                        "svo_triples",
                    ],
                )
            )
            dag = AgentPlanDAG(
                nodes=nodes,
                metadata={"question_family": "syntax_role_patterns"},
            )
            return PlannerAction(
                action="emit_plan_dag",
                rewritten_question=state.rewritten_question or rewritten,
                assumptions=list(heuristic_assumptions),
                plan_dag=dag,
            )
        if self._needs_attributed_explanation_analysis(rewritten):
            search_inputs = self._search_inputs_for_question(rewritten, query_text=query_text)
            market_ticker = self._infer_market_ticker(rewritten)
            nodes = [
                AgentPlanNode("search", "db_search", inputs=search_inputs),
                AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                AgentPlanNode("keyterms", "extract_keyterms", inputs={"top_k": 40}, depends_on=["fetch"]),
                AgentPlanNode("quotes", "quote_extract", depends_on=["fetch"]),
                AgentPlanNode("quote_attribution", "quote_attribute", depends_on=["quotes"]),
                AgentPlanNode("claim_spans", "claim_span_extract", inputs={"query_focus": rewritten}, depends_on=["fetch"]),
                AgentPlanNode("claim_scores", "claim_strength_score", depends_on=["claim_spans"]),
                AgentPlanNode("entities", "ner", depends_on=["fetch"], optional=True),
                AgentPlanNode("entity_link", "entity_link", depends_on=["entities"], optional=True),
                AgentPlanNode(
                    "series",
                    "time_series_aggregate",
                    inputs={
                        "documents_node": "fetch",
                        "time_field": "published_at",
                        "bucket_granularity": "month",
                        "metrics": ["document_count"],
                        "fallback_time_bin": self._extract_date_window(rewritten).get("date_from", ""),
                    },
                    depends_on=["fetch"],
                ),
                AgentPlanNode("changes", "change_point_detect", depends_on=["series"], optional=True),
                AgentPlanNode("bursts", "burst_detect", depends_on=["series"], optional=True),
            ]
            assumptions: list[str] = []
            plot_dep = "series"
            if market_ticker:
                date_window = self._extract_date_window(rewritten)
                nodes.append(
                    AgentPlanNode(
                        "market_series",
                        "join_external_series",
                        inputs={
                            "ticker": market_ticker,
                            "date_from": date_window.get("date_from", ""),
                            "date_to": date_window.get("date_to", ""),
                            "interval": "1d",
                            "join_granularity": "month",
                            "left_key": "time_bin",
                            "right_key": "time_bin",
                            "how": "left",
                        },
                        depends_on=["series"],
                    )
                )
                plot_dep = "market_series"
            else:
                assumptions.append(
                    "The runtime could not infer a ticker for the external price series; the explanation analysis will still run over coverage and evidence."
                )
            nodes.extend(
                [
                    AgentPlanNode(
                        "plot",
                        "plot_artifact",
                        inputs={
                            "plot_name": "coverage_vs_external_price_movements",
                            "plot_type": "line",
                            "x": "time_bin",
                            "y": ["document_count", "market_drawdown"] if plot_dep == "market_series" else "document_count",
                            "series": "series_name",
                        },
                        depends_on=[plot_dep],
                        optional=True,
                    ),
                    AgentPlanNode(
                        "evidence",
                        "build_evidence_table",
                        inputs={"task": "attributed_explanation_evidence"},
                        depends_on=["keyterms", "quote_attribution", "entity_link", "claim_scores"],
                    ),
                ]
            )
            dag = AgentPlanDAG(
                nodes=nodes,
                metadata={"question_family": "attributed_explanation_series", "market_ticker": market_ticker},
            )
            return PlannerAction(
                action="emit_plan_dag",
                rewritten_question=state.rewritten_question or rewritten,
                assumptions=list(dict.fromkeys(heuristic_assumptions + assumptions)),
                plan_dag=dag,
            )
        if "distribution" in text and "noun" in text:
            noun_top_k = infer_requested_output_limit(rewritten, default=100, minimum=20, maximum=500)
            plot_top_k = min(noun_top_k, infer_requested_output_limit(rewritten, default=30, minimum=10, maximum=120))
            search_inputs = self._search_inputs_for_question(rewritten, query_text=query_text)
            dag = AgentPlanDAG(
                nodes=[
                    AgentPlanNode("search", "db_search", inputs=search_inputs),
                    AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                    AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                    AgentPlanNode("pos", "pos_morph", depends_on=["fetch"]),
                    AgentPlanNode("lemmas", "lemmatize", depends_on=["fetch"]),
                    AgentPlanNode(
                        "noun_distribution",
                        "build_evidence_table",
                        inputs={
                            "task": "noun_frequency_distribution",
                            "top_k": noun_top_k,
                            "filters": {"upos": ["NOUN", "PROPN"]},
                        },
                        depends_on=["fetch", "pos", "lemmas"],
                    ),
                    AgentPlanNode(
                        "plot",
                        "plot_artifact",
                        inputs={"plot_name": "noun_distribution", "x": "lemma", "y": "count", "top_k": plot_top_k},
                        depends_on=["noun_distribution"],
                        optional=True,
                    ),
                    AgentPlanNode(
                        "summary",
                        "build_evidence_table",
                        inputs={"task": "summary_stats"},
                        depends_on=["fetch", "noun_distribution"],
                    ),
                ],
                metadata={"question_family": "noun_distribution"},
            )
            return PlannerAction(
                action="emit_plan_dag",
                rewritten_question=state.rewritten_question or rewritten,
                assumptions=list(heuristic_assumptions),
                plan_dag=dag,
            )
        if "named entit" in text or ("entity" in text and any(term in text for term in ("dominate", "dominant", "trend", "over time", "across", "coverage"))):
            search_inputs = self._search_inputs_for_question(rewritten, query_text=query_text)
            dag = AgentPlanDAG(
                nodes=[
                    AgentPlanNode("search", "db_search", inputs=search_inputs),
                    AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                    AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                    AgentPlanNode("ner", "ner", depends_on=["fetch"]),
                    AgentPlanNode(
                        "entity_trend",
                        "build_evidence_table",
                        inputs={
                            "task": "named_entity_frequency",
                            "entity_field": "entity_text",
                            "entity_label_field": "label",
                            "entity_types": ["PERSON", "ORG", "GPE", "NORP", "EVENT"],
                            "group_by_time": True,
                            "time_field": "published_at",
                            "top_k": 100,
                        },
                        depends_on=["ner"],
                    ),
                    AgentPlanNode("changes", "change_point_detect", depends_on=["entity_trend"], optional=True),
                    AgentPlanNode(
                        "plot",
                        "plot_artifact",
                        inputs={
                            "plot_name": "entity_trend",
                            "plot_type": "line",
                            "x": "time_bin",
                            "y": "mention_count",
                            "series": "entity",
                            "top_k": 10,
                        },
                        depends_on=["entity_trend"],
                        optional=True,
                    ),
                ],
                metadata={"question_family": "entity_trend"},
            )
            return PlannerAction(
                action="emit_plan_dag",
                rewritten_question=state.rewritten_question or rewritten,
                assumptions=list(heuristic_assumptions),
                plan_dag=dag,
            )
        if self._needs_source_comparison_analysis(rewritten):
            search_inputs = self._search_inputs_for_question(
                rewritten,
                query_text=query_text,
                overrides={"retrieval_strategy": "exhaustive_analytic", "retrieve_all": True, "top_k": 0},
            )
            dag = AgentPlanDAG(
                nodes=[
                    AgentPlanNode("search", "db_search", inputs=search_inputs),
                    AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                    AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                    AgentPlanNode(
                        "keyterms_by_source",
                        "extract_keyterms",
                        inputs={"group_by": "outlet", "top_k": 25},
                        depends_on=["fetch"],
                    ),
                    AgentPlanNode(
                        "plot_keyterms",
                        "plot_artifact",
                        inputs={
                            "plot_name": "source_keyterm_comparison",
                            "plot_type": "bar",
                            "x": "term",
                            "y": "score",
                            "series": "outlet",
                            "top_k": 20,
                        },
                        depends_on=["keyterms_by_source"],
                        optional=True,
                    ),
                    AgentPlanNode(
                        "summary",
                        "build_evidence_table",
                        inputs={"task": "summary_stats"},
                        depends_on=["fetch", "keyterms_by_source"],
                    ),
                ],
                metadata={"question_family": "source_comparison"},
            )
            return PlannerAction(
                action="emit_plan_dag",
                rewritten_question=state.rewritten_question or rewritten,
                assumptions=list(heuristic_assumptions),
                plan_dag=dag,
            )
        if any(term in text for term in ("predict", "predicted", "prediction", "warn", "warning", "warned", "forecast", "anticipated", "foresaw")):
            search_inputs = self._search_inputs_for_question(rewritten, query_text=query_text)
            dag = AgentPlanDAG(
                nodes=[
                    AgentPlanNode("search", "db_search", inputs=search_inputs),
                    AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                    AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                    AgentPlanNode("claim_spans", "claim_span_extract", depends_on=["fetch"]),
                    AgentPlanNode("claim_scores", "claim_strength_score", depends_on=["claim_spans"]),
                    AgentPlanNode("evidence", "build_evidence_table", depends_on=["claim_scores"]),
                ],
                metadata={"question_family": "prediction_evidence"},
            )
            return PlannerAction(
                action="emit_plan_dag",
                rewritten_question=state.rewritten_question or rewritten,
                assumptions=list(heuristic_assumptions),
                plan_dag=dag,
            )
        if "similar" in text or "semantically" in text:
            search_inputs = self._search_inputs_for_question(rewritten, query_text=query_text)
            dag = AgentPlanDAG(
                nodes=[
                    AgentPlanNode("search", "db_search", inputs=search_inputs),
                    AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                    AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                    AgentPlanNode("doc_embeddings", "doc_embeddings", depends_on=["fetch"]),
                    AgentPlanNode("similarity", "similarity_pairwise", depends_on=["fetch"]),
                ],
                metadata={"question_family": "similarity_analysis"},
            )
            return PlannerAction(
                action="emit_plan_dag",
                rewritten_question=state.rewritten_question or rewritten,
                assumptions=list(heuristic_assumptions),
                plan_dag=dag,
            )
        asks_media_shift = self._needs_temporal_portrayal_analysis(rewritten)
        if asks_media_shift:
            search_inputs = self._search_inputs_for_question(rewritten, query_text=query_text)
            market_ticker = self._infer_market_ticker(rewritten)
            framing_series_definitions = self._infer_framing_series_definitions(rewritten)
            nodes = [
                AgentPlanNode("search", "db_search", inputs=search_inputs),
                AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                AgentPlanNode("entities", "ner", depends_on=["fetch"]),
                AgentPlanNode("keyterms", "extract_keyterms", depends_on=["fetch"]),
                AgentPlanNode("topics", "topic_model", inputs={"num_topics": 6}, depends_on=["fetch"]),
                AgentPlanNode("sentiment", "sentiment", depends_on=["fetch"]),
                AgentPlanNode("series", "time_series_aggregate", depends_on=["sentiment"]),
                AgentPlanNode("changes", "change_point_detect", depends_on=["series"], optional=True),
                AgentPlanNode("plot_sentiment", "plot_artifact", inputs={"plot_name": "framing_sentiment_series"}, depends_on=["series"], optional=True),
            ]
            if framing_series_definitions:
                nodes.extend(
                    [
                        AgentPlanNode(
                            "framing_series",
                            "time_series_aggregate",
                            inputs={
                                "documents_node": "fetch",
                                "time_field": "published_at",
                                "bucket_granularity": "month",
                                "series": framing_series_definitions,
                                "metrics": ["document_count"],
                            },
                            depends_on=["fetch"],
                            optional=True,
                        ),
                        AgentPlanNode(
                            "plot_framing_shift",
                            "plot_artifact",
                            inputs={
                                "plot_name": "framing_shift_over_time",
                                "plot_type": "line",
                                "x": "time_bin",
                                "y": "share_of_documents",
                                "series": "series_name",
                                "title": "Framing shift over time",
                                "x_label": "Month",
                                "y_label": "Share of matched documents",
                            },
                            depends_on=["framing_series"],
                            optional=True,
                        ),
                    ]
                )
            else:
                nodes.append(AgentPlanNode("plot_topics", "plot_artifact", inputs={"plot_name": "framing_topics"}, depends_on=["topics"], optional=True))
            assumptions: list[str] = []
            needs_external_series = bool(market_ticker) or any(
                term in text
                for term in ["stock", "drawdown", "share price", "market", "valuation", "oil price", "oil prices", "crude oil", "gas price", "gas prices"]
            )
            if needs_external_series:
                if market_ticker:
                    nodes.append(
                        AgentPlanNode(
                            "market_series",
                            "join_external_series",
                            inputs={
                                "ticker": market_ticker,
                                "date_from": search_inputs.get("date_from", ""),
                                "date_to": search_inputs.get("date_to", ""),
                                "left_key": "time_bin",
                                "right_key": "time_bin",
                                "how": "left",
                            },
                            depends_on=["fetch"],
                        )
                    )
                    nodes.append(
                        AgentPlanNode(
                            "plot_market_drawdown",
                            "plot_artifact",
                            inputs={
                                "plot_name": f"{market_ticker.lower()}_stock_drawdown",
                                "plot_type": "line",
                                "x": "time_bin",
                                "y": "market_drawdown",
                                "title": f"{market_ticker} stock drawdowns",
                                "x_label": "Date",
                                "y_label": "Drawdown",
                            },
                            depends_on=["market_series"],
                            optional=True,
                        )
                    )
                else:
                    assumptions.append(
                        "The runtime could not infer an external price-series ticker automatically, so market correspondence needs a supplied ticker or external series."
                    )
            dag = AgentPlanDAG(
                nodes=nodes,
                metadata={
                    "question_family": "framing_shift",
                    "requires_external_series": needs_external_series,
                    "market_ticker": market_ticker,
                },
            )
            return PlannerAction(
                action="emit_plan_dag",
                rewritten_question=state.rewritten_question or rewritten,
                assumptions=list(dict.fromkeys(heuristic_assumptions + assumptions)),
                plan_dag=dag,
            )
        generic_search_inputs = self._search_inputs_for_question(rewritten, query_text=query_text)
        dag = AgentPlanDAG(
            nodes=[
                AgentPlanNode("search", "db_search", inputs=generic_search_inputs),
                AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                AgentPlanNode("keyterms", "extract_keyterms", depends_on=["fetch"], optional=True),
            ],
            metadata={"question_family": "generic"},
        )
        return PlannerAction(
            action="emit_plan_dag",
            rewritten_question=state.rewritten_question or rewritten,
            assumptions=list(heuristic_assumptions),
            plan_dag=dag,
        )

    def plan(self, state: AgentRunState) -> PlannerAction:
        if self.llm_client is None:
            self._record_llm_trace(
                state,
                stage="plan",
                used_fallback=True,
                note="No LLM client configured; heuristic planning policy used.",
            )
            heuristic = self._heuristic_plan(state)
            if heuristic.action == "emit_plan_dag" and heuristic.plan_dag is not None:
                heuristic.plan_dag = self._normalize_plan_dag(
                    heuristic.plan_dag,
                    question_text=self._planning_context_text(state, heuristic.rewritten_question),
                )
            return heuristic
        messages = [
            {
                "role": "system",
                "content": (
                    """You are the planning module for a corpus agent operating over a user-provided corpus.

                    Return strict JSON with keys:
                    action, rewritten_question, assumptions, clarification_question, rejection_reason, message, plan_dag.

                    Allowed actions:
                    ask_clarification, emit_plan_dag, grounded_rejection.

                    The plan_dag is a compact directed acyclic graph. Each node must include:
                    node_id, capability, tool_name if a concrete tool is selected, inputs, depends_on.

                    Core planning order:
                    1. Rewrite the user question into a precise analytical task.
                    2. Decide whether the task is:
                    - precision evidence lookup,
                    - exhaustive corpus analysis,
                    - semantic/conceptual exploration,
                    - linguistic/statistical analysis,
                    - temporal/external-series alignment,
                    - unsupported or underspecified.
                    3. Choose the retrieval coverage policy.
                    4. Select tools by required capability.
                    5. Resolve each capability to a concrete typed repo tool from tool_catalog when available.
                    6. If no typed repo tool fits the required capability, use python_runner as a bounded fallback.
                    7. If a typed repo tool exists but cannot express the required operation or has failed according to prior context, use python_runner on the smallest already-materialized working set that is sufficient.

                    Do not assume the corpus is news unless the user question or corpus schema indicates that.

                    Retrieval coverage policy:
                    Use retrieval_strategy='precision_ranked' when the user needs a small set of best supporting documents, examples, quotes, or evidence snippets.

                    Use retrieval_strategy='exhaustive_analytic' when the user asks for corpus-wide aggregates, distributions, trends, comparisons, frequencies, rankings, "all relevant records", "all documents about X", "how often", "over time", "compare X and Y", or any analytical result where missing relevant records would bias the conclusion.

                    Use retrieval_strategy='semantic_exploratory' when the user asks about similarity, themes, recurring descriptions, vague concepts, paraphrases, abbreviations, or wording that may not match exact keywords.

                    Critical rule for exhaustive_analytic:
                    When retrieval_strategy='exhaustive_analytic', top_k must not define the analyzed population size. It may only be used as a safety fallback, preview limit, or batch/page size. The plan must materialize the candidate working set using retrieve_all=true, SQL/filter-based retrieval, pagination, or corpus metadata filters before running analysis.

                    For exhaustive_analytic tasks:
                    - set retrieve_all=true when supported;
                    - include explicit topical query strings for db_search or sql_query_search;
                    - use source/time/entity filters when supplied by the user or schema;
                    - avoid tiny default retrieval budgets;
                    - prefer one broad materialized working set plus filter_working_set over multiple overlapping retrieve_all searches;
                    - run aggregate/statistical tools only after the relevant working set is materialized;
                    - include summary/count diagnostics such as candidate_count, filtered_count, and coverage_notes when tools support them.

                    For precision_ranked tasks:
                    Use db_search plus fetch_documents. Set top_k, lexical_top_k, dense_top_k, rerank_top_k, and use_rerank according to the scope. Use reranking when the best evidence documents matter.

                    For semantic_exploratory tasks:
                    Use embeddings/similarity capabilities when available:
                    doc_embeddings, similarity_index, similarity_pairwise, word_embeddings, extract_acronyms.
                    Combine with keyterms/entities and temporal summaries when the question asks over time.

                    Tool selection policy:
                    Always select by capability first, not by tool name first.

                    For every required capability:
                    - If tool_catalog contains a typed repo tool that directly implements the capability, use it.
                    - If multiple typed tools fit, choose the most specific one.
                    - If no typed tool fits, use python_runner with bounded inputs, explicit expected output schema, and dependencies on already-materialized data.
                    - Do not invent tool names.
                    - Do not invent task names that are not documented in tool_catalog.
                    - Prefer typed repo tools before python_runner, but do not force an ill-fitting typed tool when python_runner is the only way to correctly compute the requested analysis.

                    Python fallback policy:
                    Use python_runner only when:
                    - no typed repo tool exists for the required operation;
                    - the typed tool is unavailable or failed in prior context;
                    - the user requests a custom statistic, transformation, or visualization not covered by typed tools;
                    - the required computation is simple and bounded over an already-materialized working set.

                    Every python_runner node must include:
                    - input_artifacts or upstream node references,
                    - operation_description,
                    - expected_output_schema,
                    - resource_bounds such as max_rows, max_runtime_seconds, or batch_size when applicable.

                    Query string rule:
                    Every db_search or sql_query_search node must include a non-empty query string derived from rewritten_question. Do not rely on implicit query context.

                    Use sql_query_search when:
                    - hybrid retrieval is sparse or off-target;
                    - exact metadata filtering is needed;
                    - entity/source/time coverage is poor;
                    - exhaustive_analytic requires full candidate materialization and SQL can express the slice better than ranked retrieval.

                    Use db_search when:
                    - natural-language or hybrid retrieval is appropriate;
                    - ranked evidence is needed;
                    - broad topical candidate generation is needed before filtering.

                    For outlet/source-scoped questions:
                    Add source filters only when the user names specific outlets or the corpus metadata clearly supplies source names. Do not invent backend alias lists for broad phrases such as "Swiss newspapers" or "American media". If outlet aliases may be needed, list them in assumptions and keep source terms separate from topical query terms.

                    For linguistic questions:
                    If the user asks about nouns, verbs, adjectives, grammatical roles, who acted, who was acted upon, or subject-verb-object patterns, include the relevant linguistic capabilities:
                    sentence_split, tokenize, pos_morph, lemmatize, dependency_parse, extract_svo_triples, noun_chunks, named_entities, entity_linking.
                    For noun frequency tables, use build_evidence_table with task_name='noun_frequency_distribution' exactly when supported.
                    noun_frequency_distribution rows contain lemma, count, relative_frequency, document_frequency, and rank.
                    Plot noun distributions with x='lemma' and y='count'.

                    For price/external-series explanation questions:
                    Combine external price/time series with:
                    time_series_aggregate, burst/change detection, keyterms, quote_extract, quote_attribute, claim_span_extract, claim_strength_score, NER/entity_linking, and an evidence table.
                    The plan must align corpus evidence windows with external time-series movement windows.

                    For plots:
                    plot_artifact must depend on an analytical table node, not raw document evidence rows.
                    Pass x, y, limit or top_k, and title.
                    When the user specifies an output limit such as "top 20", pass it through to the aggregation and plot nodes.

                    Clarification policy:
                    Ask clarification only when the missing information changes the required analysis materially.
                    If clarification_history resolves a term, use the resolved term as the retrieval anchor.
                    If prior clarification resolved only part of an ambiguity, ask only for the missing remainder.
                    When a clarification says one term means another, use the resolved term as the retrieval anchor and avoid broad synonym paraphrases that dilute retrieval.

                    Rejection policy:
                    Use grounded_rejection only when the question cannot be answered from the available corpus/tools, violates constraints, or requires unavailable external data.
                    Explain the missing requirement precisely in rejection_reason."""
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": state.question,
                        "rewritten_question": state.rewritten_question,
                        "clarification_history": list(state.clarification_history),
                        "available_capabilities": state.available_capabilities,
                        "tool_catalog": state.tool_catalog,
                        "corpus_schema": state.corpus_schema,
                        "failures": state.failures,
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        repair_system_message = (
            "You previously returned an invalid or empty plan for a corpus agent operating over a user-provided corpus. "
            "Return valid JSON with keys action, rewritten_question, assumptions, clarification_question, rejection_reason, message, plan_dag. "
            "Allowed actions: ask_clarification, emit_plan_dag, grounded_rejection. "
            "If action is emit_plan_dag, plan_dag.nodes must contain at least one executable node with node_id, capability, optional tool_name, inputs, and depends_on. "
            "Choose retrieval budgets and retrieval_strategy values that match question scope instead of relying on tiny generic defaults. "
            "Preserve specialized intent: semantic wording/abbreviation questions need embeddings and similarity; who-did-what questions need syntax/SVO tools; attributed price-movement explanation questions need external series, quotes, claim spans, and evidence. "
            "Avoid redundant parallel full-corpus searches; use filter_working_set for narrower slices that can be derived from an existing broad retrieve_all working set. "
            "Do not invent source/outlet aliases for broad geographic media phrases; use source filters only for explicit outlet names or metadata-backed source names. "
            "Use documented build_evidence_table task names, especially noun_frequency_distribution for noun/POS lemma counts, and do not plot document evidence rows as aggregate plots."
        )
        try:
            trace = self.llm_client.complete_json_trace(
                messages,
                model=self.llm_config.planner_model,
                temperature=0.0,
            )
            payload = dict(trace["parsed_json"])
            self._record_llm_trace(state, stage="plan", trace=trace)
            actionable_payload = self._planner_payload_is_actionable(payload)
            action = PlannerAction.from_dict(payload) if actionable_payload else None
            if (not actionable_payload) or (action is not None and action.action == "emit_plan_dag" and action.plan_dag is None):
                repair_messages = [
                    {"role": "system", "content": repair_system_message},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": state.question,
                                "rewritten_question": state.rewritten_question,
                                "available_capabilities": state.available_capabilities,
                                "tool_catalog": state.tool_catalog,
                                "corpus_schema": state.corpus_schema,
                                "previous_invalid_output": trace.get("raw_text", ""),
                            },
                            ensure_ascii=True,
                        ),
                    },
                ]
                repair_trace = self.llm_client.complete_json_trace(
                    repair_messages,
                    model=self.llm_config.planner_model,
                    temperature=0.0,
                )
                self._record_llm_trace(state, stage="plan_repair", trace=repair_trace)
                repair_payload = dict(repair_trace["parsed_json"])
                if not self._planner_payload_is_actionable(repair_payload):
                    raise ValueError("Planner returned no executable nodes.")
                action = PlannerAction.from_dict(repair_payload)
                if action.action == "emit_plan_dag" and action.plan_dag is None:
                    raise ValueError("Planner returned no executable nodes.")
            if action is None:
                raise ValueError("Planner returned no executable nodes.")
            if action.action == "emit_plan_dag" and action.plan_dag is not None:
                action.plan_dag = self._normalize_plan_dag(
                    action.plan_dag,
                    question_text=self._planning_context_text(state, action.rewritten_question),
                )
            if action.action == "ask_clarification" and state.force_answer:
                heuristic = self._heuristic_plan(state)
                if heuristic.action == "emit_plan_dag" and heuristic.plan_dag is not None:
                    heuristic.plan_dag = self._normalize_plan_dag(
                        heuristic.plan_dag,
                        question_text=self._planning_context_text(state, heuristic.rewritten_question),
                    )
                heuristic.assumptions = list(
                    dict.fromkeys(
                        list(action.assumptions)
                        + ["force_answer=true: planner clarification skipped and best-effort heuristic plan was used."]
                    )
                )
                return heuristic
            sufficient, clarification_assumptions = self._broad_scope_clarification_is_sufficient(state)
            if action.action == "ask_clarification" and sufficient:
                heuristic = self._heuristic_plan(state)
                if heuristic.action == "emit_plan_dag" and heuristic.plan_dag is not None:
                    heuristic.plan_dag = self._normalize_plan_dag(
                        heuristic.plan_dag,
                        question_text=self._planning_context_text(state, heuristic.rewritten_question),
                    )
                heuristic.rewritten_question = action.rewritten_question or heuristic.rewritten_question or state.rewritten_question
                heuristic.assumptions = list(
                    dict.fromkeys(
                        list(action.assumptions)
                        + list(heuristic.assumptions)
                        + clarification_assumptions
                        + ["Planner clarification was skipped because the provided clarification history was already sufficient."]
                    )
                )
                return heuristic
            return action
        except Exception as exc:
            error_text = str(exc)
            note = "LLM planning step failed; heuristic planning fallback used."
            if error_text == "Planner returned no executable nodes." or error_text.startswith("Unsupported planner action:"):
                error_text = ""
                note = "Planner returned no executable nodes or no actionable JSON; heuristic planning fallback used."
            self._record_llm_trace(
                state,
                stage="plan",
                used_fallback=True,
                error=error_text,
                note=note,
            )
            heuristic = self._heuristic_plan(state)
            if heuristic.action == "emit_plan_dag" and heuristic.plan_dag is not None:
                heuristic.plan_dag = self._normalize_plan_dag(
                    heuristic.plan_dag,
                    question_text=self._planning_context_text(state, heuristic.rewritten_question),
                )
            return heuristic

    def revise_after_failure(
        self,
        state: AgentRunState,
        failure: AgentFailure,
        snapshot: AgentExecutionSnapshot,
    ) -> PlannerAction | None:
        if failure.capability == "python_runner":
            return None
        candidate_rows = _candidate_payload_rows(snapshot)
        evidence_rows = self._extract_evidence(snapshot)
        summary = self._derive_summary(snapshot)
        code = (
            "from collections import Counter\n"
            "from pathlib import Path\n"
            "import json\n"
            "import re\n"
            "\n"
            "STOPWORDS = {\n"
            "    'the', 'and', 'for', 'that', 'with', 'from', 'this', 'have', 'were', 'been', 'will', 'into',\n"
            "    'their', 'about', 'which', 'what', 'when', 'where', 'would', 'could', 'should', 'using', 'within',\n"
            "    'document', 'documents', 'corpus', 'article', 'articles', 'report', 'reports', 'coverage', 'content'\n"
            "}\n"
            "\n"
            "def _text(row):\n"
            "    for key in ('text', 'body', 'content', 'snippet', 'excerpt', 'title'):\n"
            "        value = str(row.get(key, '')).strip()\n"
            "        if value:\n"
            "            return value\n"
            "    return ''\n"
            "\n"
            "def _outlet(row):\n"
            "    for key in ('outlet', 'source', 'publisher', 'source_domain'):\n"
            "        value = str(row.get(key, '')).strip()\n"
            "        if value:\n"
            "            return value\n"
            "    return ''\n"
            "\n"
            "def _date(row):\n"
            "    for key in ('date', 'published_at', 'time_bin', 'year'):\n"
            "        value = str(row.get(key, '')).strip()\n"
            "        if value:\n"
            "            return value\n"
            "    return ''\n"
            "\n"
            "payload = INPUTS_JSON\n"
            "rows = [dict(item) for item in payload.get('candidate_rows', []) if isinstance(item, dict)]\n"
            "question = str(payload.get('question', '')).strip()\n"
            "failed_capability = str(payload.get('failed_capability', '')).strip()\n"
            "tokens = [\n"
            "    token.lower()\n"
            "    for token in re.findall(r\"[A-Za-z][A-Za-z0-9\\-]+\", ' '.join(_text(row) for row in rows))\n"
            "    if len(token) >= 4 and token.lower() not in STOPWORDS\n"
            "]\n"
            "top_terms = Counter(tokens).most_common(12)\n"
            "source_counts = Counter(_outlet(row) for row in rows if _outlet(row)).most_common(8)\n"
            "time_counts = Counter(_date(row)[:7] for row in rows if _date(row)).most_common(8)\n"
            "evidence_rows = []\n"
            "for row in rows[:10]:\n"
            "    text = _text(row).replace('\\n', ' ').strip()\n"
            "    excerpt = text[:277].rstrip() + '...' if len(text) > 280 else text\n"
            "    evidence_rows.append(\n"
            "        {\n"
            "            'doc_id': str(row.get('doc_id', '')),\n"
            "            'outlet': _outlet(row),\n"
            "            'date': _date(row),\n"
            "            'excerpt': excerpt,\n"
            "            'score': float(row.get('score', 0.0) or 0.0),\n"
            "        }\n"
            "    )\n"
            "highlights = []\n"
            "if source_counts:\n"
            "    highlights.append('Most frequent sources in the available slice: ' + ', '.join(f'{name} ({count})' for name, count in source_counts[:5]))\n"
            "if top_terms:\n"
            "    highlights.append('Most repeated content terms: ' + ', '.join(f'{term} ({count})' for term, count in top_terms[:8]))\n"
            "if time_counts:\n"
            "    highlights.append('Most represented time bins: ' + ', '.join(f'{name} ({count})' for name, count in time_counts[:5]))\n"
            "if not highlights and payload.get('summary'):\n"
            "    highlights.append('Python fallback summarized the available intermediate outputs after a typed tool failure.')\n"
            "result = {\n"
            "    'rows': evidence_rows,\n"
            "    'highlights': highlights,\n"
            "    'analysis': {\n"
            "        'question': question,\n"
            "        'failed_capability': failed_capability,\n"
            "        'document_count': len(rows),\n"
            "        'top_sources': source_counts,\n"
            "        'top_terms': top_terms,\n"
            "        'time_bins': time_counts,\n"
            "        'received_summary_keys': sorted(str(key) for key in payload.get('summary', {}).keys()),\n"
            "    },\n"
            "    'caveats': [\n"
            "        f\"Primary capability '{failed_capability}' failed, so a bounded python fallback summarized the available intermediate corpus slice.\",\n"
            "        'The python fallback is descriptive and limited to the rows passed into the sandbox.',\n"
            "    ],\n"
            "}\n"
            "Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)\n"
            "Path(OUTPUT_DIR, 'result.json').write_text(json.dumps(result), encoding='utf-8')\n"
            "print(json.dumps({'highlights': highlights[:2], 'rows': len(evidence_rows)}))\n"
        )
        inputs_json = {
            "question": state.question,
            "rewritten_question": state.rewritten_question,
            "failed_capability": failure.capability,
            "failure_message": failure.message,
            "candidate_rows": candidate_rows,
            "evidence_rows": evidence_rows,
            "summary": summary,
            "corpus_schema": state.corpus_schema,
        }
        dag = AgentPlanDAG(
            nodes=[
                AgentPlanNode(
                    "python_fallback",
                    "python_runner",
                    tool_name="python_runner",
                    inputs={"code": code, "inputs_json": inputs_json},
                    description="Summarize available intermediate corpus outputs with a bounded python fallback.",
                )
            ],
            metadata={"revision_for": failure.capability},
        )
        return PlannerAction(
            action="revise_plan_after_failure",
            rewritten_question=state.rewritten_question,
            assumptions=[f"Used python_runner fallback after failure in capability '{failure.capability}'."],
            plan_dag=dag,
        )

    def synthesize(self, state: AgentRunState, snapshot: AgentExecutionSnapshot) -> FinalAnswerPayload:
        evidence_rows = self._extract_evidence(snapshot)
        summary = self._derive_summary(snapshot)
        strict_signals = self._derive_strict_signals(snapshot)
        if self.llm_client is None:
            self._record_llm_trace(
                state,
                stage="final_synthesis",
                used_fallback=True,
                note="No LLM client configured; fallback synthesis used.",
            )
            return self._apply_answer_guardrails(
                state,
                snapshot,
                self._fallback_synthesis(state, evidence_rows, summary, snapshot),
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the grounded synthesis module for a corpus agent operating over a user-provided corpus. "
                    "Return JSON with keys answer_text, evidence_items, artifacts_used, unsupported_parts, caveats, claim_verdicts. "
                    "Use only the provided summaries, tool outputs, and evidence. "
                    "Do not claim direct correspondence to stock-price moves or external market behavior unless an external series was explicitly attached. "
                    "When has_external_series is true and summary.external_series is present, use that compact external-series summary and do not say the external series is missing. "
                    "When summary contains entity_trend_time_series or time_series_summaries, treat those as valid temporal analysis rows; do not say a time breakdown is missing solely because evidence_rows are only examples. "
                    "Use summary.run_diagnostics to qualify answer scope: for aggregate questions, do not generalize beyond a small ranked slice, preview rows, or noisy/no-data analytical outputs. "
                    "STRICT SCIENTIFIC MODE: The user-message contains a `strict_signals` block with valid_signals, degraded_signals, unavailable_methods, null_metrics_by_node, and warnings. "
                    "Do NOT make claims that depend on metrics listed in null_metrics_by_node — those metrics are null because their source tool produced no rows. "
                    "Do NOT describe degraded_signals or unavailable_methods as if they had succeeded. "
                    "Structure answer_text into clearly labeled sections: 'Valid findings:' (only from valid_signals), 'Degraded analyses:' (from degraded_signals, with the reason), and 'Unavailable methods:' (from unavailable_methods, with the reason). "
                    "If a comparative question (e.g. A vs B) lacks data for one side, say so explicitly — do not paper over the gap by summarizing only the side that has data. "
                    "Put every entry from strict_signals.warnings into caveats. "
                    "Put every node in unavailable_methods into unsupported_parts. "
                    "Honest partial analysis beats fake complete analysis."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": state.question,
                        "rewritten_question": state.rewritten_question,
                        "assumptions": list(state.assumptions),
                        "summary": summary,
                        "evidence_rows": evidence_rows,
                        "strict_signals": strict_signals,
                        "tool_caveats": self._snapshot_caveats(snapshot),
                        "failures": [item.to_dict() for item in snapshot.failures],
                        "has_external_series": self._has_external_series(snapshot),
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        try:
            trace = self.llm_client.complete_json_trace(
                messages,
                model=self.llm_config.synthesis_model,
                temperature=0.1,
            )
            payload = dict(trace["parsed_json"])
            self._record_llm_trace(state, stage="final_synthesis", trace=trace)
            answer = FinalAnswerPayload.from_payload(payload)
            if evidence_rows:
                answer.evidence_items = evidence_rows
            return self._apply_answer_guardrails(state, snapshot, answer)
        except Exception as exc:
            self._record_llm_trace(
                state,
                stage="final_synthesis",
                used_fallback=True,
                error=str(exc),
                note="LLM synthesis failed; fallback synthesis used.",
            )
            return self._apply_answer_guardrails(
                state,
                snapshot,
                self._fallback_synthesis(state, evidence_rows, summary, snapshot),
            )

    def _extract_evidence(self, snapshot: AgentExecutionSnapshot) -> list[dict[str, Any]]:
        for node_id, result in snapshot.node_results.items():
            payload = result.payload
            if isinstance(payload, dict) and "rows" in payload and payload["rows"]:
                first = payload["rows"][0]
                if {"doc_id", "outlet", "date", "excerpt", "score"}.issubset(first.keys()):
                    rows = []
                    for row in payload["rows"]:
                        copied = dict(row)
                        score = copied.get("score", 0.0)
                        try:
                            numeric = float(score)
                        except (TypeError, ValueError):
                            numeric = 0.0
                        copied.setdefault("score_display", f"{numeric:.4f}".rstrip("0").rstrip(".") if abs(numeric) >= 0.01 else (f"{numeric:.2e}" if abs(numeric) > 0 else "0"))
                        rows.append(copied)
                    return rows
        fallback_rows: list[dict[str, Any]] = []
        for row in snapshot.selected_docs[:10]:
            doc_id = str(row.get("doc_id", "")).strip()
            if not doc_id:
                continue
            excerpt_source = (
                row.get("snippet")
                or row.get("excerpt")
                or row.get("text")
                or row.get("body")
                or row.get("title")
                or ""
            )
            excerpt = str(excerpt_source).strip().replace("\n", " ")
            if len(excerpt) > 280:
                excerpt = excerpt[:277].rstrip() + "..."
            fallback_rows.append(
                {
                    "doc_id": doc_id,
                    "outlet": str(
                        row.get("outlet")
                        or row.get("source")
                        or row.get("source_domain")
                        or ""
                    ),
                    "date": str(row.get("published_at") or row.get("date") or row.get("year") or ""),
                    "excerpt": excerpt,
                    "score": float(row.get("score", 0.0) or 0.0),
                    "score_display": str(row.get("score_display") or ""),
                }
            )
        return fallback_rows

    @staticmethod
    def _is_internal_tool_caveat(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return True
        internal_markers = (
            "applied date window filter",
            "applied minimum text length filter",
            "applied structured working-set filters",
            "dependency nodes",
            "exhaustive analytical retrieval",
            "fetched preview",
            "fetched 250 preview",
            "fetched 1000 preview",
            "inferred plot ",
            "input docs",
            "missing plot",
            "missing_working_set_ref",
            "no input documents were available",
            "no rows available for plotting",
            "no_input_documents",
            "no_data",
            "plot requested",
            "provider_unavailable",
            "preview/batch",
            "python_runner",
            "resolved annotation-backed filters",
            "resolved plot ",
            "streamed from working_set_ref",
            "upstream fetched documents were only a preview",
            "working set '",
            "working_set_ref",
            "year-balanced retrieval was applied",
        )
        return any(marker in lowered for marker in internal_markers)

    def _snapshot_caveats(self, snapshot: AgentExecutionSnapshot) -> list[str]:
        caveats: list[str] = []
        empty_supporting_outputs = 0
        for result in snapshot.node_results.values():
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            if metadata.get("no_data"):
                empty_supporting_outputs += 1

            for caveat in result.caveats:
                caveat_text = str(caveat).strip()
                if self._is_internal_tool_caveat(caveat_text):
                    continue
                if caveat_text:
                    caveats.append(caveat_text)

            for key in ("no_data_reason", "reason", "warning"):
                reason_text = str(metadata.get(key, "")).strip()
                if self._is_internal_tool_caveat(reason_text):
                    continue
                if reason_text:
                    caveats.append(reason_text)

            payload = result.payload if isinstance(result.payload, dict) else {}
            for key in ("no_data_reason", "reason", "warning"):
                reason_text = str(payload.get(key, "")).strip()
                if self._is_internal_tool_caveat(reason_text):
                    continue
                if reason_text:
                    caveats.append(reason_text)
        if empty_supporting_outputs:
            caveats.append(
                "Some supporting tool outputs were empty; affected plots or comparisons were omitted or marked unsupported."
            )
        return list(dict.fromkeys(caveats))

    def _derive_strict_signals(self, snapshot: AgentExecutionSnapshot) -> dict[str, Any]:
        """Partition node outputs into valid / degraded / unavailable.

        The synthesis prompt uses this to keep the LLM from generating claims
        about metrics whose source nodes produced 0 rows or were marked
        no_data — the exact "fake analysis from null metrics" pattern this
        whole strict-mode pass exists to prevent.
        """
        valid_signals: list[dict[str, Any]] = []
        degraded_signals: list[dict[str, Any]] = []
        unavailable_methods: list[dict[str, Any]] = []
        warnings: list[str] = []
        null_metrics_by_node: dict[str, list[str]] = {}

        for node_id, result in snapshot.node_results.items():
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            payload = result.payload if isinstance(result.payload, dict) else {}
            capability = ""
            for record in snapshot.node_records:
                if getattr(record, "node_id", "") == node_id:
                    capability = str(getattr(record, "capability", "") or getattr(record, "tool", "") or "")
                    break
            rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
            no_data = bool(metadata.get("no_data") or payload.get("no_data"))
            degraded = bool(metadata.get("degraded"))
            reason = str(metadata.get("no_data_reason") or metadata.get("reason") or payload.get("no_data_reason") or "").strip()

            entry = {
                "node_id": node_id,
                "capability": capability,
                "row_count": len(rows),
                "reason": reason,
            }

            metric_diag = metadata.get("metric_diagnostics") or payload.get("metric_diagnostics")
            if isinstance(metric_diag, dict):
                empty_metrics = [name for name, info in metric_diag.items() if isinstance(info, dict) and info.get("empty_source")]
                if empty_metrics:
                    null_metrics_by_node[node_id] = empty_metrics
                    for name in empty_metrics:
                        warnings.append(f"metric '{name}' (node {node_id}) is null — its source produced no rows")

            payload_warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
            for w in payload_warnings:
                warning_text = str(w).strip()
                if warning_text:
                    warnings.append(warning_text)

            if no_data and not rows:
                unavailable_methods.append(entry)
            elif degraded or no_data:
                degraded_signals.append(entry)
            elif rows:
                valid_signals.append(entry)

        return {
            "valid_signals": valid_signals,
            "degraded_signals": degraded_signals,
            "unavailable_methods": unavailable_methods,
            "null_metrics_by_node": null_metrics_by_node,
            "warnings": list(dict.fromkeys(warnings)),
        }

    @staticmethod
    def _summary_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            number = float(value)
            return number if math.isfinite(number) else None
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None

    def _external_series_rows(self, snapshot: AgentExecutionSnapshot) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for result in snapshot.node_results.values():
            payload = result.payload if isinstance(result.payload, dict) else {}
            payload_rows = payload.get("rows", [])
            if not isinstance(payload_rows, list):
                continue
            for row in payload_rows:
                if not isinstance(row, dict):
                    continue
                if not any(str(key).startswith("market_") for key in row.keys()):
                    continue
                if self._summary_float(row.get("market_close")) is None:
                    continue
                rows.append(dict(row))
        return rows

    def _external_series_summary(self, snapshot: AgentExecutionSnapshot) -> dict[str, Any]:
        rows = self._external_series_rows(snapshot)
        if not rows:
            return {}
        ordered = sorted(rows, key=lambda row: str(row.get("date") or row.get("time_bin") or ""))
        close_points: list[tuple[str, float]] = []
        drawdowns: list[float] = []
        for row in ordered:
            close = self._summary_float(row.get("market_close"))
            if close is None:
                continue
            period = str(row.get("time_bin") or row.get("date") or "").strip()
            close_points.append((period, close))
            drawdown = self._summary_float(row.get("market_drawdown"))
            if drawdown is not None:
                drawdowns.append(drawdown)
        if not close_points:
            return {}
        first_period, first_close = close_points[0]
        last_period, last_close = close_points[-1]
        change = last_close - first_close
        pct_change = None if first_close == 0 else change / first_close
        ticker = ""
        for row in ordered:
            ticker = str(row.get("ticker", "")).strip()
            if ticker:
                break
        if not ticker:
            for result in snapshot.node_results.values():
                metadata = result.metadata if isinstance(result.metadata, dict) else {}
                ticker = str(metadata.get("ticker", "")).strip()
                if ticker:
                    break
        return {
            "ticker": ticker,
            "row_count": len(close_points),
            "start_period": first_period,
            "end_period": last_period,
            "first_close": round(first_close, 4),
            "last_close": round(last_close, 4),
            "absolute_change": round(change, 4),
            "percent_change": round(pct_change, 6) if pct_change is not None else None,
            "min_drawdown": round(min(drawdowns), 6) if drawdowns else None,
            "sample": [
                {"period": period, "market_close": round(close, 4)}
                for period, close in close_points[:3]
            ],
        }

    @staticmethod
    def _summary_field_from_rows(rows: list[dict[str, Any]], candidates: tuple[str, ...]) -> str:
        observed: set[str] = set()
        for row in rows[:50]:
            if isinstance(row, dict):
                observed.update(str(key) for key in row.keys())
        for candidate in candidates:
            if candidate in observed:
                return candidate
        return ""

    @staticmethod
    def _summary_display_number(value: float) -> int | float:
        if value.is_integer():
            return int(value)
        return round(value, 4)

    def _grouped_time_series_summary(
        self,
        rows: list[dict[str, Any]],
        *,
        max_series: int = 8,
        max_points_per_series: int = 12,
    ) -> dict[str, Any]:
        time_field = self._summary_field_from_rows(
            rows,
            ("time_bin", "period", "month", "published_month", "date", "published_at", "year"),
        )
        series_field = self._summary_field_from_rows(
            rows,
            (
                "entity",
                "canonical_entity",
                "entity_text",
                "actor",
                "source",
                "outlet",
                "publisher",
                "target_label",
                "label",
                "term",
                "topic",
                "series",
            ),
        )
        value_field = self._summary_field_from_rows(
            rows,
            (
                "mention_count",
                "document_count",
                "doc_count",
                "count",
                "frequency",
                "value",
                "average_sentiment",
                "avg_sentiment",
                "score",
                "sentiment_score",
            ),
        )
        if not time_field or not series_field or not value_field:
            return {}

        totals: dict[str, float] = {}
        period_values: dict[str, dict[str, float]] = {}
        valid_rows: list[tuple[str, str, float]] = []
        skipped_rows = 0
        for row in rows:
            if not isinstance(row, dict):
                skipped_rows += 1
                continue
            period = str(row.get(time_field, "")).strip()
            series = str(row.get(series_field, "")).strip()
            value = self._summary_float(row.get(value_field))
            if (
                not period
                or period.lower() in {"unknown", "none", "null"}
                or not series
                or series.lower() in {"unknown", "none", "null"}
                or value is None
            ):
                skipped_rows += 1
                continue
            valid_rows.append((period, series, value))
            totals[series] = totals.get(series, 0.0) + value
            bucket = period_values.setdefault(period, {})
            bucket[series] = bucket.get(series, 0.0) + value

        if not valid_rows:
            return {}

        top_series = sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:max_series]
        top_names = {series for series, _ in top_series}
        points_by_series: dict[str, list[dict[str, Any]]] = {series: [] for series in top_names}
        for period, series, value in sorted(valid_rows, key=lambda item: (item[1], item[0])):
            if series not in top_names:
                continue
            points_by_series.setdefault(series, []).append(
                {
                    "period": period,
                    "value": self._summary_display_number(value),
                }
            )

        sampled_points: dict[str, list[dict[str, Any]]] = {}
        for series, points in points_by_series.items():
            if len(points) <= max_points_per_series:
                sampled_points[series] = points
            else:
                split = max_points_per_series // 2
                sampled_points[series] = points[:split] + points[-(max_points_per_series - split) :]

        period_leaders = []
        for period, values in sorted(period_values.items()):
            leader, value = max(values.items(), key=lambda item: (item[1], item[0]))
            period_leaders.append(
                {
                    "period": period,
                    "series": leader,
                    "value": self._summary_display_number(value),
                }
            )
        if len(period_leaders) > 12:
            period_leaders_sample = period_leaders[:6] + period_leaders[-6:]
        else:
            period_leaders_sample = period_leaders

        return {
            "time_field": time_field,
            "series_field": series_field,
            "value_field": value_field,
            "row_count": len(valid_rows),
            "skipped_row_count": skipped_rows,
            "period_count": len(period_values),
            "series_count": len(totals),
            "top_series": [
                {
                    "series": series,
                    "total": self._summary_display_number(total),
                }
                for series, total in top_series
            ],
            "sampled_points": sampled_points,
            "period_leaders_sample": period_leaders_sample,
        }

    def _derive_summary(self, snapshot: AgentExecutionSnapshot) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        diagnostics = self._run_quality_diagnostics(snapshot)
        if diagnostics:
            summary["run_diagnostics"] = diagnostics
        external_summary = self._external_series_summary(snapshot)
        if external_summary:
            summary["external_series"] = external_summary
        non_metric_rows_available = any(
            isinstance((payload := (result.payload if isinstance(result.payload, dict) else {})).get("rows"), list)
            and bool(payload.get("rows"))
            and not (
                isinstance(payload["rows"][0], dict)
                and "metric" in payload["rows"][0]
                and "value" in payload["rows"][0]
            )
            for result in snapshot.node_results.values()
        )
        noun_distribution_attempted = any(
            node_id == "noun_distribution"
            or str((result.metadata if isinstance(result.metadata, dict) else {}).get("task", "")).strip().lower()
            == "noun_frequency_distribution"
            for node_id, result in snapshot.node_results.items()
        )
        for node_id, result in snapshot.node_results.items():
            payload = result.payload if isinstance(result.payload, dict) else {}
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            rows = list(payload.get("rows", []))
            if not rows:
                continue
            first = rows[0]
            temporal_summary = self._grouped_time_series_summary(
                [dict(row) for row in rows if isinstance(row, dict)]
            )
            if temporal_summary:
                summary.setdefault("time_series_summaries", {})[node_id] = temporal_summary
            if (
                "lemma" in first
                and "count" in first
                and ("document_frequency" in first or "relative_frequency" in first)
            ):
                summary["noun_distribution"] = [
                    (str(row.get("lemma", "")).lower(), int(row.get("count", 0) or 0))
                    for row in rows
                    if str(row.get("lemma", "")).strip()
                ][:15]
                summary["noun_distribution_source"] = "aggregated_table"
            elif "metric" in first and "value" in first:
                summary_stats = {
                    str(row.get("metric", "")).strip(): row.get("value")
                    for row in rows
                    if str(row.get("metric", "")).strip()
                }
                if not (non_metric_rows_available and summary_stats.get("matched_document_count") == 0):
                    summary["summary_stats"] = summary_stats
            elif "pos" in first and "lemma" in first and "noun_distribution" not in summary and not noun_distribution_attempted:
                counts: dict[str, int] = {}
                for row in rows:
                    if str(row.get("pos", "")) not in {"NOUN", "PROPN"}:
                        continue
                    lemma = str(row.get("lemma", "")).lower()
                    if not lemma or len(lemma) < 3:
                        continue
                    counts[lemma] = counts.get(lemma, 0) + 1
                summary["noun_distribution"] = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:15]
                summary["noun_distribution_source"] = "raw_pos_fallback"
            elif "entity" in first and "time_bin" in first:
                if temporal_summary and temporal_summary.get("series_field") == "entity":
                    summary["entity_trend"] = [
                        (str(item.get("series", "")), item.get("total", 0))
                        for item in temporal_summary.get("top_series", [])[:15]
                    ]
                    summary["entity_trend_time_series"] = temporal_summary
                else:
                    grouped: dict[str, float] = {}
                    value_field = self._summary_field_from_rows(
                        [dict(row) for row in rows if isinstance(row, dict)],
                        ("mention_count", "document_count", "doc_count", "count", "frequency", "value"),
                    )
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        entity = str(row.get("entity", "")).strip()
                        value = self._summary_float(row.get(value_field)) if value_field else None
                        if entity and value is not None:
                            grouped[entity] = grouped.get(entity, 0.0) + value
                    summary["entity_trend"] = [
                        (entity, self._summary_display_number(value))
                        for entity, value in sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:15]
                    ]
            elif "term" in first and "score" in first:
                summary["keyterms"] = rows[:15]
            elif "topic_id" in first and "top_terms" in first:
                summary["topics"] = rows[:10]
            elif "label" in first and "time_bin" in first and "score" in first:
                by_time: dict[str, dict[str, float]] = {}
                for row in rows:
                    time_bin = str(row.get("time_bin", "unknown"))
                    bucket = by_time.setdefault(time_bin, {"count": 0.0, "score_total": 0.0})
                    bucket["count"] += 1.0
                    bucket["score_total"] += float(row.get("score", 0.0))
                summary["sentiment_trend"] = [
                    {
                        "time_bin": time_bin,
                        "average_score": round(values["score_total"] / max(values["count"], 1.0), 4),
                        "documents": int(values["count"]),
                    }
                    for time_bin, values in sorted(by_time.items())
                ]
            elif "delta" in first and "time_bin" in first:
                summary["change_points"] = rows[:10]
            elif {"doc_id", "outlet", "date", "excerpt", "score"}.issubset(first.keys()):
                summary["evidence_rows"] = rows[:10]
        return summary

    def _run_quality_diagnostics(self, snapshot: AgentExecutionSnapshot) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        search_nodes: list[dict[str, Any]] = []
        analysis_nodes: list[dict[str, Any]] = []
        no_data_nodes: list[dict[str, Any]] = []
        records_by_id = {record.node_id: record for record in snapshot.node_records}
        for node_id, result in snapshot.node_results.items():
            record = records_by_id.get(node_id)
            payload = result.payload if isinstance(result.payload, dict) else {}
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
            documents = payload.get("documents") if isinstance(payload.get("documents"), list) else []
            results = payload.get("results") if isinstance(payload.get("results"), list) else []
            if record and record.capability == "db_search":
                search_nodes.append(
                    {
                        "node_id": node_id,
                        "retrieval_strategy": payload.get("retrieval_strategy", ""),
                        "retrieve_all": bool(payload.get("retrieve_all", False)),
                        "document_count": payload.get("document_count", len(results)),
                        "returned_count": len(results),
                        "working_set_ref": payload.get("working_set_ref", ""),
                    }
                )
            analyzed_count = (
                metadata.get("analyzed_document_count")
                or payload.get("analyzed_document_count")
                or payload.get("returned_document_count")
                or len(documents)
            )
            working_set_count = metadata.get("working_set_document_count") or payload.get("document_count")
            if record and record.capability not in {"db_search", "fetch_documents", "create_working_set", "plot_artifact"}:
                analysis_nodes.append(
                    {
                        "node_id": node_id,
                        "capability": record.capability,
                        "row_count": len(rows),
                        "analyzed_document_count": analyzed_count,
                        "working_set_document_count": working_set_count,
                    }
                )
            no_data_reason = metadata.get("no_data_reason") or payload.get("no_data_reason") or ""
            if (metadata.get("no_data") or payload.get("no_data") or (record and record.status == "completed" and not rows and not documents and not results)) and no_data_reason:
                no_data_nodes.append(
                    {
                        "node_id": node_id,
                        "capability": record.capability if record else "",
                        "reason": no_data_reason,
                    }
                )
        if search_nodes:
            diagnostics["search_nodes"] = search_nodes
        if analysis_nodes:
            diagnostics["analysis_nodes"] = analysis_nodes[:20]
        if no_data_nodes:
            diagnostics["no_data_nodes"] = no_data_nodes[:20]
        return diagnostics

    def _has_external_series(self, snapshot: AgentExecutionSnapshot) -> bool:
        return bool(self._external_series_rows(snapshot))

    def _apply_answer_guardrails(
        self,
        state: AgentRunState,
        snapshot: AgentExecutionSnapshot,
        answer: FinalAnswerPayload,
    ) -> FinalAnswerPayload:
        raw_question_text = f"{state.question} {state.rewritten_question}"
        question_text = raw_question_text.lower()
        assumptions_text = " ".join(str(item) for item in state.assumptions).lower()
        unresolved_source_scope = any(
            marker in assumptions_text
            for marker in (
                "unclear which source values correspond",
                "needs either explicit source names",
                "requires explicit source names",
                "source scope could not be resolved",
                "explicit outlet names were provided",
                "explicit outlet names were not provided",
                "does not add a source filter",
                "without inventing outlet aliases",
                "either already scoped to",
            )
        )
        requested_described_source_scope = self._question_requests_described_source_scope(raw_question_text)
        has_source_filter = any(
            "source:" in str(result.payload.get("query", "") if isinstance(result.payload, dict) else "").lower()
            or str(result.metadata.get("filtered_from_working_set", "")).lower() == "true"
            for result in snapshot.node_results.values()
        )
        if (unresolved_source_scope or requested_described_source_scope) and not has_source_filter:
            note = (
                "The requested source scope could not be resolved from corpus metadata or explicit outlet names; "
                "the analysis rows are therefore not a supported source-scoped answer."
            )
            unsupported = "The requested source-scoped comparison/coverage subset is unsupported without explicit source names or usable source metadata."
            if note not in answer.caveats:
                answer.caveats.append(note)
            if unsupported not in answer.unsupported_parts:
                answer.unsupported_parts.append(unsupported)
            lowered_answer = answer.answer_text.lower()
            if "source scope could not be resolved" not in lowered_answer and "not a supported source-scoped answer" not in lowered_answer:
                answer.answer_text = (
                    "I cannot answer the requested source-scoped question as stated because the source scope could not be resolved. "
                    "The unscoped corpus analysis found: "
                    + answer.answer_text.lstrip()
                ).strip()
        needs_market_guardrail = any(
            term in question_text
            for term in ["stock", "drawdown", "share price", "valuation", "market", "oil price", "oil prices", "crude oil", "gas price", "gas prices"]
        )
        if needs_market_guardrail and not self._has_external_series(snapshot):
            note = (
                "This run did not attach an external market time series, so any direct correspondence to stock drawdowns remains unverified."
            )
            unsupported = "Direct stock-price or drawdown correspondence was not verified because no external market series was attached."
            if note not in answer.caveats:
                answer.caveats.append(note)
            if unsupported not in answer.unsupported_parts:
                answer.unsupported_parts.append(unsupported)
            lowered_answer = answer.answer_text.lower()
            if "unverified" not in lowered_answer and "not verified" not in lowered_answer and "not attach" not in lowered_answer:
                answer.answer_text = f"{answer.answer_text.rstrip()} {note}".strip()
        if answer.evidence_items:
            answer.evidence_items = list(answer.evidence_items)[:10]
        for caveat in self._snapshot_caveats(snapshot):
            if caveat not in answer.caveats:
                answer.caveats.append(caveat)
        # Strict-mode guardrail: force unavailable methods into unsupported_parts
        # and surface metric-source warnings as caveats so the final answer
        # cannot quietly omit them even when the LLM forgets the prompt section.
        strict_signals = self._derive_strict_signals(snapshot)
        for entry in strict_signals.get("unavailable_methods", []) or []:
            cap = str(entry.get("capability", "") or "").strip()
            reason = str(entry.get("reason", "") or "").strip()
            if not cap:
                continue
            sentence = f"{cap} produced no usable output ({reason})." if reason else f"{cap} produced no usable output."
            if sentence not in answer.unsupported_parts:
                answer.unsupported_parts.append(sentence)
        for entry in strict_signals.get("degraded_signals", []) or []:
            cap = str(entry.get("capability", "") or "").strip()
            reason = str(entry.get("reason", "") or "").strip()
            if cap and reason:
                note = f"{cap} ran in degraded mode: {reason}"
                if note not in answer.caveats:
                    answer.caveats.append(note)
        for warning in strict_signals.get("warnings", []) or []:
            text = str(warning).strip()
            if text and text not in answer.caveats:
                answer.caveats.append(text)
        return answer

    def _fallback_synthesis(
        self,
        state: AgentRunState,
        evidence_rows: list[dict[str, Any]],
        summary: dict[str, Any],
        snapshot: AgentExecutionSnapshot,
    ) -> FinalAnswerPayload:
        snapshot_caveats = self._snapshot_caveats(snapshot)
        caveats = [failure.message for failure in snapshot.failures] + snapshot_caveats
        unsupported_parts = [failure.message for failure in snapshot.failures if not failure.retriable]
        if "noun_distribution" in summary:
            top = ", ".join(f"{term} ({count})" for term, count in summary["noun_distribution"][:8])
            answer_text = f"Top noun lemmas in the retrieved slice are: {top}."
        elif "topics" in summary or "keyterms" in summary or "sentiment_trend" in summary:
            topic_text = ""
            keyterm_text = ""
            sentiment_text = ""
            if "topics" in summary:
                topic_bits = []
                for topic in summary["topics"][:4]:
                    top_terms = ", ".join(str(term) for term in topic.get("top_terms", [])[:5])
                    topic_bits.append(f"{topic.get('time_bin', 'all')}: {top_terms}")
                topic_text = "Topic slices: " + " | ".join(topic_bits) + "."
            if "keyterms" in summary:
                keyterm_text = "Key terms: " + ", ".join(
                    f"{row.get('term', '')} ({row.get('score', 0):.2f})"
                    for row in summary["keyterms"][:8]
                ) + "."
            if "sentiment_trend" in summary:
                sentiment_text = "Sentiment over time: " + " | ".join(
                    f"{row.get('time_bin', 'unknown')} avg={row.get('average_score', 0):.2f}"
                    for row in summary["sentiment_trend"][:6]
                ) + "."
            answer_text = " ".join(part for part in [topic_text, keyterm_text, sentiment_text] if part).strip()
            if any(term in state.question.lower() for term in ["stock", "drawdown", "share price", "valuation", "market"]):
                caveats.append(
                    "The current run did not attach an external market series, so stock-price correspondence cannot be measured directly yet."
                )
        elif "entity_trend" in summary:
            top = ", ".join(f"{entity} ({count})" for entity, count in summary["entity_trend"][:8])
            answer_text = f"Most prominent entities in the retrieved slice are: {top}."
        elif evidence_rows:
            answer_text = (
                "The strongest prediction or warning evidence comes from the returned documents. "
                "Higher-scoring rows are more explicit according to heuristic claim-strength markers."
            )
            caveats.append(
                "Warnings, scenarios, and explicit predictions are not identical; the evidence table is ranked by heuristic claim strength."
            )
        else:
            if snapshot_caveats:
                answer_text = (
                    "The run could not produce a supported answer from the corpus outputs. "
                    f"{snapshot_caveats[0]}"
                )
            else:
                answer_text = "The run could not produce a supported answer from the corpus outputs."
            unsupported_parts.append(
                "The requested answer is unsupported because no evidence rows, retrieved documents, or answer-bearing analysis rows were returned."
            )
        return FinalAnswerPayload(
            answer_text=answer_text,
            evidence_items=evidence_rows,
            artifacts_used=[
                artifact
                for record in snapshot.node_records
                for artifact in record.artifacts_used
            ],
            unsupported_parts=list(dict.fromkeys(item for item in unsupported_parts if item)),
            caveats=list(dict.fromkeys(item for item in caveats if item)),
            claim_verdicts=[],
        )


class AgentRuntime:
    def __init__(
        self,
        *,
        config: AgentRuntimeConfig,
        runtime: CorpusRuntime | None = None,
        llm_client: LLMClient | None = None,
        llm_config: LLMProviderConfig | None = None,
        registry: ToolRegistry | None = None,
        search_backend: Any | None = None,
        working_store: WorkingSetStore | None = None,
        python_runner: DockerPythonRunnerService | None = None,
    ) -> None:
        self.config = config
        self.app_config = load_project_configuration(config.project_root)
        self.runtime = runtime or CorpusRuntime.from_project_root(config.project_root)
        self._startup_llm_config = llm_config or LLMProviderConfig.from_env()
        self.llm_config = self._startup_llm_config
        self.llm_client = llm_client or OpenAICompatibleLLMClient(self.llm_config)
        self.registry = registry or build_agent_registry()
        self.require_backend_services = _env_flag("CORPUSAGENT2_REQUIRE_BACKEND_SERVICES", False)
        self.allow_local_fallback = _env_flag("CORPUSAGENT2_ALLOW_LOCAL_FALLBACK", True)
        if self.require_backend_services:
            self.allow_local_fallback = False
        self.search_backend = search_backend or self._build_search_backend()
        self.working_store = working_store or self._build_working_store()
        self.python_runner = python_runner or DockerPythonRunnerService()
        self.orchestrator = MagicBoxOrchestrator(self.llm_client, self.llm_config)
        self.executor = AsyncPlanExecutor(self.registry)
        # Wire the optional LLM recovery advisor. The executor checks
        # CORPUSAGENT2_USE_LLM_RECOVERY at runtime; when off, the factory
        # is never invoked. When on, each failed node consults the advisor
        # for a bounded action that operates only on that node — PlanDAG
        # topology stays the source of truth.
        def _build_recovery_advisor() -> _llm_recovery_advisor.LLMRecoveryAdvisor:
            return _llm_recovery_advisor.LLMRecoveryAdvisor(
                self.llm_client,
                model=self.llm_config.planner_model or self.llm_config.synthesis_model,
            )

        self.executor._recovery_advisor_factory = _build_recovery_advisor  # type: ignore[attr-defined]
        self._live_runs: dict[str, LiveRunStatus] = {}
        self._run_cancel_events: dict[str, threading.Event] = {}
        self._run_threads: dict[str, threading.Thread] = {}
        self._run_lock = threading.Lock()
        self._llm_override_active = False
        self._provider_modules_cache: dict[str, bool] | None = None
        self._device_report_cache: dict[str, Any] | None = None
        self._retrieval_health_cache: tuple[float, dict[str, Any]] | None = None
        self._corpus_date_bounds_cache: tuple[str, str] | None = None
        try:
            self._runtime_info_health_ttl_s = max(
                0.0,
                float(os.getenv("CORPUSAGENT2_RUNTIME_INFO_HEALTH_TTL_S", "30").strip() or "30"),
            )
        except ValueError:
            self._runtime_info_health_ttl_s = 30.0
        try:
            self._startup_repaired_runs = int(self.working_store.cleanup_interrupted_runs())
        except Exception:
            self._startup_repaired_runs = 0

        self._warmup_state: dict[str, Any] = {
            "complete": False,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "completed_at_utc": None,
            "duration_ms": None,
            "stages": {},
            "errors": [],
        }
        self._warmup_thread = threading.Thread(
            target=self._run_warmup, name="agent-warmup", daemon=True
        )
        self._warmup_thread.start()

    _EXPECTED_WARMUP_STAGES = ("torch_init", "spacy_en_load", "llm_endpoint_reachable")

    def _run_warmup(self) -> None:
        started = time.monotonic()
        self._warmup_stage("torch_init", self._warmup_probe_torch)
        self._warmup_stage("spacy_en_load", self._warmup_probe_spacy_en)
        self._warmup_stage("llm_endpoint_reachable", self._warmup_probe_llm)
        if dense_retrieval_enabled(default=False):
            self._warmup_stage("dense_embedder_load", self._warmup_probe_dense_embedder)
        self._warmup_state["complete"] = True
        self._warmup_state["completed_at_utc"] = datetime.now(UTC).isoformat()
        self._warmup_state["duration_ms"] = round((time.monotonic() - started) * 1000, 1)

    def _warmup_stage(self, name: str, fn: Callable[[], None]) -> None:
        started = time.monotonic()
        try:
            fn()
            self._warmup_state["stages"][name] = {
                "ok": True,
                "duration_ms": round((time.monotonic() - started) * 1000, 1),
            }
        except Exception as exc:
            self._warmup_state["stages"][name] = {
                "ok": False,
                "duration_ms": round((time.monotonic() - started) * 1000, 1),
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._warmup_state["errors"].append(f"{name}: {type(exc).__name__}: {exc}")

    def _warmup_probe_torch(self) -> None:
        import torch  # noqa: F401

        _ = torch.cuda.is_available()

    def _warmup_probe_spacy_en(self) -> None:
        import spacy

        _ = spacy.load("en_core_web_sm")

    def _warmup_probe_llm(self) -> None:
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(self.llm_config.base_url or "")
        host = parsed.hostname
        if not host:
            raise RuntimeError("LLM base_url has no host")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=5):
            pass

    def _warmup_probe_dense_embedder(self) -> None:
        dense_id = self.runtime.dense_model_id
        if not dense_id:
            return
        from sentence_transformers import SentenceTransformer

        _ = SentenceTransformer(dense_id, device="cpu")

    def warmup_info(self) -> dict[str, Any]:
        """Snapshot of background warmup progress for /health and /runtime-info."""
        state = {
            "complete": bool(self._warmup_state.get("complete")),
            "started_at_utc": self._warmup_state.get("started_at_utc"),
            "completed_at_utc": self._warmup_state.get("completed_at_utc"),
            "duration_ms": self._warmup_state.get("duration_ms"),
            "stages": dict(self._warmup_state.get("stages", {})),
            "errors": list(self._warmup_state.get("errors", [])),
        }
        expected = list(self._EXPECTED_WARMUP_STAGES)
        if dense_retrieval_enabled(default=False):
            expected.append("dense_embedder_load")
        state["pending_stages"] = [name for name in expected if name not in state["stages"]]
        return state

    def _build_search_backend(self):
        lexical_backend = None
        try:
            lexical_backend = OpenSearchBackend(OpenSearchConfig.from_env())
        except Exception as exc:
            if self.require_backend_services:
                raise RuntimeError(f"OpenSearch backend is required but unavailable: {exc}") from exc
        return HybridSearchBackend(
            self.runtime,
            lexical_backend=lexical_backend,
            allow_lexical_fallback=self.allow_local_fallback,
        )

    def _build_working_store(self) -> WorkingSetStore:
        return resolve_working_store(
            doc_lookup_factory=self.runtime.doc_lookup,
            require_backend_services=self.require_backend_services,
        )

    def _corpus_date_bounds(self) -> tuple[str, str] | None:
        if self._corpus_date_bounds_cache is not None:
            return self._corpus_date_bounds_cache
        try:
            metadata = self.runtime.load_metadata()
        except Exception:
            return None
        if metadata is None or getattr(metadata, "empty", True):
            return None
        date_column = next((column for column in ("published_at", "date") if column in metadata.columns), "")
        if not date_column:
            return None
        values = metadata[date_column].dropna().astype(str).str.slice(0, 10)
        valid = sorted(value for value in values if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))
        if not valid:
            return None
        self._corpus_date_bounds_cache = (valid[0], valid[-1])
        return self._corpus_date_bounds_cache

    def _date_window_unsupported_reason(self, question_text: str) -> str:
        requested = self.orchestrator._extract_date_window(question_text)
        if not requested:
            return ""
        bounds = self._corpus_date_bounds()
        if not bounds:
            return ""
        corpus_start, corpus_end = bounds
        requested_start = str(requested.get("date_from", "") or "")
        requested_end = str(requested.get("date_to", "") or "")
        if not requested_end and requested_start:
            requested_end = "9999-12-31"
        if not requested_start and requested_end:
            requested_start = "0001-01-01"
        if not requested_start or not requested_end:
            return ""
        if requested_start[:10] <= corpus_end and requested_end[:10] >= corpus_start:
            return ""
        requested_label = requested_start[:10] if requested_start == requested_end else f"{requested_start[:10]} to {requested_end[:10]}"
        return (
            f"The requested time window ({requested_label}) is outside the loaded corpus date range "
            f"({corpus_start} to {corpus_end}), so this corpus cannot support the requested temporal analysis."
        )

    def _out_of_corpus_model_answer(
        self,
        *,
        state: AgentRunState,
        unsupported_reason: str,
    ) -> FinalAnswerPayload:
        no_corpus_caveat = (
            "No corpus data was available for the requested scope. The answer below is generated by the configured "
            "planner model from prior knowledge only; it is not grounded in retrieved documents, evidence rows, "
            "artifacts, or tool outputs from this corpus."
        )
        model_answer = ""
        model_caveats: list[str] = []
        model_unsupported: list[str] = []
        if self.llm_client is not None:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are answering a user question only because the loaded CorpusAgent2 corpus cannot support "
                        "the requested scope. Return JSON with keys answer_text, caveats, unsupported_parts, "
                        "and claim_verdicts. The answer_text must be concise, must explicitly state that no corpus "
                        "data is available for this answer, and must not cite or imply retrieved corpus evidence. "
                        "Use general model knowledge only and qualify uncertain or time-sensitive claims."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": state.question,
                            "rewritten_question": state.rewritten_question,
                            "corpus_unsupported_reason": unsupported_reason,
                            "required_flag": OUT_OF_CORPUS_MODEL_ANSWER_FLAG,
                        },
                        ensure_ascii=True,
                    ),
                },
            ]
            try:
                trace = self.llm_client.complete_json_trace(
                    messages,
                    model=self.llm_config.planner_model,
                    temperature=0.1,
                )
                self.orchestrator._record_llm_trace(state, stage="out_of_corpus_model_answer", trace=trace)
                payload = FinalAnswerPayload.from_payload(dict(trace.get("parsed_json", {})))
                model_answer = payload.answer_text.strip()
                model_caveats = list(payload.caveats)
                model_unsupported = list(payload.unsupported_parts)
            except Exception as exc:
                self.orchestrator._record_llm_trace(
                    state,
                    stage="out_of_corpus_model_answer",
                    used_fallback=True,
                    error=str(exc),
                    note="Out-of-corpus one-shot model answer failed; returning explicit unsupported-corpus notice.",
                )
        else:
            self.orchestrator._record_llm_trace(
                state,
                stage="out_of_corpus_model_answer",
                used_fallback=True,
                note="No LLM client configured; returning explicit unsupported-corpus notice.",
            )

        if model_answer.startswith(OUT_OF_CORPUS_MODEL_ANSWER_FLAG):
            model_answer = model_answer[len(OUT_OF_CORPUS_MODEL_ANSWER_FLAG) :].strip(" \n:-")

        if not model_answer:
            model_answer = (
                "A one-shot model answer could not be generated. Rerun with a corpus covering the requested scope "
                "to obtain a grounded agent answer."
            )

        answer_text = "\n\n".join(
            [
                OUT_OF_CORPUS_MODEL_ANSWER_FLAG,
                f"Corpus status: {unsupported_reason}",
                model_answer,
            ]
        )
        unsupported_parts = list(
            dict.fromkeys(
                [
                    unsupported_reason,
                    "The model-only answer is unsupported by this corpus because no matching corpus data was available for the requested scope.",
                    *model_unsupported,
                ]
            )
        )
        caveats = list(dict.fromkeys([no_corpus_caveat, *model_caveats]))
        return FinalAnswerPayload(
            answer_text=answer_text,
            unsupported_parts=unsupported_parts,
            caveats=caveats,
            claim_verdicts=[
                {
                    "claim": "The corpus-grounded agent cannot answer the requested scope.",
                    "verdict": "supported",
                    "evidence": unsupported_reason,
                },
                {
                    "claim": "Any substantive answer content is generated from model prior knowledge only.",
                    "verdict": "unsupported_by_corpus",
                    "evidence": no_corpus_caveat,
                },
            ],
        )

    def capability_catalog(self) -> list[dict[str, Any]]:
        specs = sorted(
            self.registry.list_tools(),
            key=lambda spec: (spec.capabilities[0] if spec.capabilities else "", spec.tool_name),
        )
        return [spec.to_dict() for spec in specs]

    def _saved_run_manifests(self) -> list[dict[str, Any]]:
        root = self.config.outputs_root
        if not root.exists():
            return []
        manifests: list[tuple[float, dict[str, Any]]] = []
        for path in root.glob("agent_*/run_manifest.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                payload.setdefault("run_id", path.parent.name)
                try:
                    modified = path.stat().st_mtime
                except OSError:
                    modified = 0.0
                manifests.append((modified, payload))
        return [payload for _, payload in sorted(manifests, key=lambda item: item[0], reverse=True)]

    def tool_usage_summary(self) -> dict[str, Any]:
        catalog = self.capability_catalog()
        by_tool = {str(item.get("tool_name", "")): item for item in catalog if str(item.get("tool_name", "")).strip()}
        capability_to_tool: dict[str, str] = {}
        for item in catalog:
            tool_name = str(item.get("tool_name", "")).strip()
            capabilities = item.get("capabilities", [])
            if not isinstance(capabilities, list):
                capabilities = []
            for capability in capabilities:
                capability_to_tool.setdefault(str(capability), tool_name)

        def new_entry(tool_name: str, spec: dict[str, Any] | None = None) -> dict[str, Any]:
            spec_capabilities = (spec or {}).get("capabilities", [])
            if not isinstance(spec_capabilities, list):
                spec_capabilities = []
            capabilities = [
                str(item)
                for item in spec_capabilities
                if str(item).strip()
            ]
            return {
                "tool_name": tool_name,
                "capabilities": capabilities,
                "category": _tool_usage_category(capabilities or [tool_name]),
                "role": _tool_usage_role(capabilities or [tool_name]),
                "registered": bool(spec),
                "event_count": 0,
                "completed_event_count": 0,
                "completed_node_count": 0,
                "planned_node_count": 0,
                "planned_unresolved_count": 0,
                "run_count": 0,
                "planned_run_count": 0,
                "status_counts": {},
                "node_status_counts": {},
                "last_run_id": "",
                "last_question": "",
                "_run_ids": set(),
                "_planned_run_ids": set(),
                "_event_keys": set(),
                "_call_node_keys": set(),
                "_completed_node_keys": set(),
                "_planned_node_keys": set(),
                "_recorded_node_keys": set(),
            }

        entries = {
            tool_name: new_entry(tool_name, spec)
            for tool_name, spec in by_tool.items()
        }

        def entry_for(tool_name: str = "", capability: str = "") -> dict[str, Any]:
            resolved_tool = tool_name.strip() or capability_to_tool.get(capability.strip(), "") or capability.strip() or "unknown_tool"
            if resolved_tool not in entries:
                entries[resolved_tool] = new_entry(resolved_tool)
                if capability and capability not in entries[resolved_tool]["capabilities"]:
                    entries[resolved_tool]["capabilities"].append(capability)
                    entries[resolved_tool]["category"] = _tool_usage_category(entries[resolved_tool]["capabilities"])
                    entries[resolved_tool]["role"] = _tool_usage_role(entries[resolved_tool]["capabilities"])
            return entries[resolved_tool]

        manifests = self._saved_run_manifests()
        questions_by_run = {
            str(manifest.get("run_id", "")): str(manifest.get("question", ""))
            for manifest in manifests
            if str(manifest.get("run_id", "")).strip()
        }

        db_rows: list[dict[str, Any]] = []
        try:
            reader = getattr(self.working_store, "read_all_tool_calls", None)
            if callable(reader):
                db_rows = [dict(row) for row in reader()]
        except Exception:
            db_rows = []
        db_run_ids = {str(row.get("run_id", "")) for row in db_rows if str(row.get("run_id", "")).strip()}

        def ingest_call(row: dict[str, Any], *, source: str) -> None:
            payload = row.get("payload", {}) if isinstance(row.get("payload"), dict) else {}
            run_id = str(row.get("run_id", "")).strip()
            node_id = str(row.get("node_id", "")).strip()
            capability = str(row.get("capability", "")).strip()
            tool_name = str(row.get("tool_name", "")).strip()
            status = str(row.get("status", "") or "unknown").strip().lower()
            entry = entry_for(tool_name, capability)
            event_key = (
                source,
                run_id,
                node_id,
                capability,
                tool_name,
                status,
                json.dumps(payload, sort_keys=True, default=str)[:500],
            )
            if event_key in entry["_event_keys"]:
                return
            entry["_event_keys"].add(event_key)
            entry["event_count"] += 1
            entry["status_counts"][status] = int(entry["status_counts"].get(status, 0)) + 1
            if status == "completed":
                entry["completed_event_count"] += 1
            if run_id:
                entry["_run_ids"].add(run_id)
                entry["last_run_id"] = run_id
                entry["last_question"] = questions_by_run.get(run_id, entry["last_question"])
            if run_id and node_id:
                node_key = f"{run_id}:{node_id}"
                entry["_call_node_keys"].add(node_key)
                if status == "completed":
                    entry["_completed_node_keys"].add(node_key)

        for row in db_rows:
            ingest_call(row, source="db")

        for manifest in manifests:
            run_id = str(manifest.get("run_id", "")).strip()
            if not run_id:
                continue
            plan_dags = manifest.get("plan_dags", [])
            if not isinstance(plan_dags, list):
                plan_dags = []
            for dag in plan_dags:
                dag_nodes = dag.get("nodes", []) if isinstance(dag, dict) else []
                if not isinstance(dag_nodes, list):
                    dag_nodes = []
                for node in dag_nodes:
                    if not isinstance(node, dict):
                        continue
                    node_id = str(node.get("node_id", node.get("id", ""))).strip()
                    capability = str(node.get("capability", "")).strip()
                    tool_name = str(node.get("tool_name", "")).strip()
                    entry = entry_for(tool_name, capability)
                    if run_id and node_id:
                        node_key = f"{run_id}:{node_id}"
                        entry["_planned_node_keys"].add(node_key)
                        entry["_planned_run_ids"].add(run_id)
            node_records = manifest.get("node_records", [])
            if not isinstance(node_records, list):
                node_records = []
            for record in node_records:
                if not isinstance(record, dict):
                    continue
                node_id = str(record.get("node_id", "")).strip()
                capability = str(record.get("capability", "")).strip()
                tool_name = str(record.get("tool_name", "")).strip()
                status = str(record.get("status", "") or "unknown").strip().lower()
                entry = entry_for(tool_name, capability)
                entry["node_status_counts"][status] = int(entry["node_status_counts"].get(status, 0)) + 1
                if run_id and node_id:
                    node_key = f"{run_id}:{node_id}"
                    entry["_recorded_node_keys"].add(node_key)
                    if status == "completed":
                        entry["_completed_node_keys"].add(node_key)
            if run_id not in db_run_ids:
                tool_calls = manifest.get("tool_calls", [])
                if not isinstance(tool_calls, list):
                    tool_calls = []
                for row in tool_calls:
                    if isinstance(row, dict):
                        ingest_call({"run_id": run_id, **row}, source="manifest")

        tool_rows: list[dict[str, Any]] = []
        for entry in entries.values():
            planned_node_count = len(entry["_planned_node_keys"])
            completed_node_count = len(entry["_completed_node_keys"])
            unresolved = entry["_planned_node_keys"] - entry["_recorded_node_keys"] - entry["_call_node_keys"]
            entry["planned_node_count"] = planned_node_count
            entry["completed_node_count"] = completed_node_count
            entry["planned_unresolved_count"] = len(unresolved)
            entry["run_count"] = len(entry["_run_ids"])
            entry["planned_run_count"] = len(entry["_planned_run_ids"])
            entry["never_used"] = completed_node_count == 0 and entry["completed_event_count"] == 0
            entry["reason"] = _tool_usage_reason(
                capabilities=list(entry["capabilities"]),
                completed_node_count=completed_node_count,
                planned_node_count=planned_node_count,
                planned_unresolved_count=len(unresolved),
            )
            tool_rows.append(
                {
                    key: value
                    for key, value in entry.items()
                    if not key.startswith("_")
                }
            )

        tool_rows.sort(
            key=lambda row: (
                -int(row.get("completed_node_count", 0)),
                -int(row.get("completed_event_count", 0)),
                str(row.get("tool_name", "")),
            )
        )
        category_rows: list[dict[str, Any]] = []
        for category, _members in TOOL_USAGE_CATEGORIES + (("Other", set()),):
            tools = [row for row in tool_rows if row.get("category") == category]
            if not tools:
                continue
            category_rows.append(
                {
                    "category": category,
                    "registered_tool_count": len(tools),
                    "used_tool_count": sum(1 for row in tools if not row.get("never_used")),
                    "completed_node_count": sum(int(row.get("completed_node_count", 0)) for row in tools),
                    "planned_node_count": sum(int(row.get("planned_node_count", 0)) for row in tools),
                }
            )

        return {
            "run_count": len({str(manifest.get("run_id", "")) for manifest in manifests if str(manifest.get("run_id", "")).strip()} | db_run_ids),
            "manifest_run_count": len(manifests),
            "db_run_count": len(db_run_ids),
            "registered_tool_count": len(catalog),
            "used_tool_count": sum(1 for row in tool_rows if not row.get("never_used")),
            "never_used_tool_count": sum(1 for row in tool_rows if row.get("registered") and row.get("never_used")),
            "tools": tool_rows,
            "categories": category_rows,
            "notes": [
                "Counts distinguish planned nodes, recorded tool-call events, and completed node executions.",
                "Tool-call events can include running and completed rows for one node; completed_node_count is the cleaner usage metric.",
                "Manifest files are used to recover planned nodes and local runs; database tool-call rows are preferred when present for a run.",
            ],
        }

    def _active_run_ids(self) -> list[str]:
        with self._run_lock:
            return [
                run_id
                for run_id, status in self._live_runs.items()
                if status.status not in TERMINAL_RUN_STATUSES
            ]

    def _provider_modules_installed(self) -> dict[str, bool]:
        if self._provider_modules_cache is None:
            self._provider_modules_cache = {
                module_name: importlib.util.find_spec(module_name) is not None
                for module_name in ["spacy", "textacy", "stanza", "nltk", "gensim", "flair", "textblob", "torch", "yfinance"]
            }
        return dict(self._provider_modules_cache)

    def _cached_device_report(self) -> dict[str, Any]:
        if self._device_report_cache is None:
            self._device_report_cache = dict(runtime_device_report())
        return dict(self._device_report_cache)

    def _cached_retrieval_health(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._retrieval_health_cache is not None and self._runtime_info_health_ttl_s > 0:
            cached_at, cached_payload = self._retrieval_health_cache
            if now - cached_at <= self._runtime_info_health_ttl_s:
                return dict(cached_payload)
        try:
            payload = dict(self.runtime.retrieval_health())
        except Exception as exc:
            payload = {
                "document_count": 0,
                "backend": getattr(self.runtime, "retrieval_backend", "unknown"),
                "local_lexical": {"ready": False, "path": ""},
                "local_dense": {"ready": False, "error": str(exc)},
                "pgvector": {
                    "configured": False,
                    "table": "",
                    "ready": False,
                    "total_rows": 0,
                    "dense_rows": 0,
                    "indices": [],
                    "error": "",
                },
                "dense_strategy": "unavailable",
                "full_corpus_dense_ready": False,
                "dense_candidate_fallback_ready": False,
                "metadata_error": f"{type(exc).__name__}: {exc}",
                "health_check_error": str(exc),
            }
        self._retrieval_health_cache = (now, payload)
        return dict(payload)

    def update_llm_runtime_settings(
        self,
        *,
        use_openai: bool,
        planner_model: str = "",
        synthesis_model: str = "",
    ) -> dict[str, Any]:
        active_run_ids = self._active_run_ids()
        if active_run_ids:
            raise RuntimeError(
                "LLM settings can only be changed when no query is running. Abort active runs first."
            )

        resolved = self._startup_llm_config.with_runtime_overrides(
            use_openai=use_openai,
            planner_model=planner_model or None,
            synthesis_model=synthesis_model or None,
        )
        self.llm_config = resolved
        self.llm_client = OpenAICompatibleLLMClient(resolved)
        self.orchestrator = MagicBoxOrchestrator(self.llm_client, self.llm_config)
        self._llm_override_active = (
            resolved.use_openai != self._startup_llm_config.use_openai
            or resolved.planner_model != self._startup_llm_config.planner_model
            or resolved.synthesis_model != self._startup_llm_config.synthesis_model
            or resolved.base_url != self._startup_llm_config.base_url
        )
        return self.runtime_info()

    def reset_llm_runtime_settings(self) -> dict[str, Any]:
        active_run_ids = self._active_run_ids()
        if active_run_ids:
            raise RuntimeError(
                "LLM settings can only be reset when no query is running. Abort active runs first."
            )
        self.llm_config = self._startup_llm_config
        self.llm_client = OpenAICompatibleLLMClient(self.llm_config)
        self.orchestrator = MagicBoxOrchestrator(self.llm_client, self.llm_config)
        self._llm_override_active = False
        return self.runtime_info()

    def runtime_info(self) -> dict[str, Any]:
        provider_modules = self._provider_modules_installed()
        device_report = self._cached_device_report()
        openai_defaults = self._startup_llm_config.with_runtime_overrides(use_openai=True)
        unclose_defaults = self._startup_llm_config.with_runtime_overrides(use_openai=False)
        llm_warnings: list[str] = []
        if self.llm_config.use_openai and "hermes.ai.unturf.com" in self.llm_config.base_url:
            llm_warnings.append("OpenAI mode is enabled but the resolved base URL still points to UncloseAI/Hermes.")
        if not self.llm_config.use_openai and "api.openai.com" in self.llm_config.base_url:
            llm_warnings.append("UncloseAI mode is enabled but the resolved base URL still points to api.openai.com.")
        gpu_warnings: list[str] = []
        if device_report.get("nvidia_smi_ok") and not device_report.get("cuda_available"):
            gpu_warnings.append(
                "An NVIDIA GPU is visible through nvidia-smi, but PyTorch is a CPU-only build. Reinstall a CUDA-enabled torch wheel to use the GPU."
            )

        provider_orders = {
            capability.lower().replace("CORPUSAGENT2_PROVIDER_ORDER_", ""): value
            for capability, value in os.environ.items()
            if capability.startswith("CORPUSAGENT2_PROVIDER_ORDER_")
        }
        retrieval_health = self._cached_retrieval_health()
        try:
            corpus_info = _runtime_corpus_info(self.config.project_root, retrieval_health)
        except Exception as exc:
            corpus_info = {
                "name": "unknown",
                "display_name": "unknown",
                "error": f"{type(exc).__name__}: {exc}",
            }
        configured_default_mode = os.getenv("CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE", "").strip().lower()
        if not configured_default_mode:
            configured_default_mode = (
                "hybrid" if dense_retrieval_enabled(default=self.runtime.retrieval_backend == "pgvector") else "lexical"
            )
        effective_default_mode = configured_default_mode
        if configured_default_mode in {"hybrid", "dense"} and not retrieval_health["full_corpus_dense_ready"]:
            effective_default_mode = "hybrid" if retrieval_health["dense_candidate_fallback_ready"] else "lexical"
        return {
            "llm": {
                "use_openai": self.llm_config.use_openai,
                "provider_name": self.llm_config.provider_name,
                "base_url": self.llm_config.base_url,
                "planner_model": self.llm_config.planner_model,
                "synthesis_model": self.llm_config.synthesis_model,
                "api_key_present": bool(self.llm_config.api_key),
                "override_active": self._llm_override_active,
                "startup_defaults": {
                    "use_openai": self._startup_llm_config.use_openai,
                    "provider_name": self._startup_llm_config.provider_name,
                    "base_url": self._startup_llm_config.base_url,
                    "planner_model": self._startup_llm_config.planner_model,
                    "synthesis_model": self._startup_llm_config.synthesis_model,
                },
                "available_defaults": {
                    "openai": {
                        "provider_name": openai_defaults.provider_name,
                        "base_url": openai_defaults.base_url,
                        "planner_model": openai_defaults.planner_model,
                        "synthesis_model": openai_defaults.synthesis_model,
                    },
                    "uncloseai": {
                        "provider_name": unclose_defaults.provider_name,
                        "base_url": unclose_defaults.base_url,
                        "planner_model": unclose_defaults.planner_model,
                        "synthesis_model": unclose_defaults.synthesis_model,
                    },
                },
                "warnings": llm_warnings,
            },
            "device": {
                **device_report,
                "warnings": gpu_warnings,
            },
            "providers_installed": provider_modules,
            "provider_order": provider_orders,
            "corpus": corpus_info,
            "capability_count": len(self.registry.list_tools()),
            "retrieval": {
                "backend": self.runtime.retrieval_backend,
                "configured_default_mode": configured_default_mode,
                "default_mode": effective_default_mode,
                "rerank_enabled": os.getenv("CORPUSAGENT2_RETRIEVAL_USE_RERANK", "true").strip().lower()
                not in {"0", "false", "no", "off"},
                "rerank_top_k": int(os.getenv("CORPUSAGENT2_RETRIEVAL_RERANK_TOP_K", "25").strip() or "25"),
                "fusion_k": int(os.getenv("CORPUSAGENT2_RETRIEVAL_FUSION_K", "60").strip() or "60"),
                "dense_model_id": self.runtime.dense_model_id,
                "rerank_model_id": self.runtime.rerank_model_id,
                "require_backend_services": self.require_backend_services,
                "allow_local_fallback": self.allow_local_fallback,
                "time_granularity": os.getenv("CORPUSAGENT2_TIME_GRANULARITY", "month").strip() or "month",
                "health": retrieval_health,
            },
            "analysis_notes": [
                "UI model/provider changes are process-local runtime overrides. They affect future runs only and reset when the backend restarts unless you also update .env.",
                "Classical spaCy/textacy/gensim analytics are usually CPU-bound even when CUDA is available.",
                "GPU is mainly relevant for torch- or Flair-backed models and only when those providers are selected.",
                "Per-node provider choice and artifacts are captured in the run manifest so you can verify what really ran.",
                "Some analytics remain heuristic by design in the prototype, especially claim scoring, quote attribution, and burst detection; check provenance and caveats per node.",
                "Provider chips are import checks only; per-node provenance shows which provider actually executed.",
                (
                    f"Recovered {self._startup_repaired_runs} interrupted run(s) from 'started' to 'failed' on startup."
                    if self._startup_repaired_runs
                    else "No interrupted runs needed cleanup on startup."
                ),
                (
                    f"Dense retrieval is using '{retrieval_health['dense_strategy']}' because full-corpus dense assets are not fully ready yet."
                    if not retrieval_health["full_corpus_dense_ready"] and retrieval_health["dense_candidate_fallback_ready"]
                    else "Full-corpus dense retrieval is ready."
                    if retrieval_health["full_corpus_dense_ready"]
                    else "Dense retrieval is unavailable until corpus metadata is loaded."
                ),
                *(
                    [f"doc_metadata.parquet is missing on this host: {retrieval_health['metadata_error']}. Local lexical/dense assets and corpus row counts will read as 0 until the file is restored or re-built."]
                    if retrieval_health.get("metadata_error")
                    else []
                ),
            ],
            "active_run_ids": self._active_run_ids(),
        }

    def _set_live_status(self, run_id: str, **updates: Any) -> LiveRunStatus:
        with self._run_lock:
            status = self._live_runs.get(run_id)
            if status is None:
                status = LiveRunStatus(
                    run_id=run_id,
                    question=str(updates.get("question", "")),
                    status=str(updates.get("status", "queued")),
                )
                self._live_runs[run_id] = status
            for key, value in updates.items():
                if hasattr(status, key):
                    setattr(status, key, value)
            status.updated_at_utc = datetime.now(UTC).isoformat()
            return status

    def _mark_abort_requested_live_status(self, run_id: str, *, detail: str) -> dict[str, Any]:
        with self._run_lock:
            status = self._live_runs.get(run_id)
            if status is None:
                raise FileNotFoundError(f"Run not found for run_id={run_id}")
            now = datetime.now(UTC).isoformat()
            status.status = "aborting"
            status.current_phase = "aborting"
            status.detail = detail
            status.active_steps = [
                {
                    **dict(item),
                    "status": "aborting",
                    "abort_requested": True,
                    "abort_requested_at_utc": now,
                }
                for item in status.active_steps
            ]
            status.updated_at_utc = now
            return status.to_dict()

    def _current_tool_calls(self, run_id: str) -> list[dict[str, Any]]:
        with self._run_lock:
            live = self._live_runs.get(run_id)
            if live is not None and live.tool_calls:
                return [dict(item) for item in live.tool_calls]
        try:
            return [dict(item) for item in self.working_store.read_tool_calls(run_id)]
        except Exception:
            return []

    def _register_cancel_event(self, run_id: str) -> threading.Event:
        with self._run_lock:
            event = self._run_cancel_events.get(run_id)
            if event is None:
                event = threading.Event()
                self._run_cancel_events[run_id] = event
            return event

    def _is_cancelled(self, run_id: str) -> bool:
        with self._run_lock:
            event = self._run_cancel_events.get(run_id)
            return bool(event is not None and event.is_set())

    def _build_aborted_manifest(
        self,
        *,
        run_id: str,
        question: str,
        rewritten_question: str,
        state: AgentRunState,
        artifacts_dir: Path,
        snapshot: AgentExecutionSnapshot | None = None,
        plan_dags: list[dict[str, Any]] | None = None,
        clarification_questions: list[str] | None = None,
    ) -> AgentRunManifest:
        empty_snapshot = AgentExecutionSnapshot(
            node_records=[],
            node_results={},
            failures=[],
            provenance_records=[],
            selected_docs=[],
            status="aborted",
        )
        resolved_snapshot = snapshot or empty_snapshot
        return AgentRunManifest(
            run_id=run_id,
            question=question,
            rewritten_question=rewritten_question or question,
            status="aborted",
            clarification_questions=list(clarification_questions or []),
            assumptions=list(state.assumptions),
            planner_actions=list(state.planner_actions),
            plan_dags=list(plan_dags or []),
            tool_calls=self._current_tool_calls(run_id),
            selected_docs=list(resolved_snapshot.selected_docs),
            node_records=list(resolved_snapshot.node_records),
            provenance_records=list(resolved_snapshot.provenance_records),
            evidence_table=list(self.orchestrator._extract_evidence(resolved_snapshot)),
            final_answer=FinalAnswerPayload(
                answer_text="Run aborted by user.",
                caveats=["Run was aborted before completion; any partial outputs may be incomplete."],
            ),
            artifacts_dir=str(artifacts_dir),
            failures=list(resolved_snapshot.failures),
            metadata={
                "force_answer": bool(state.force_answer),
                "no_cache": bool(state.no_cache),
                "clarification_history": list(state.clarification_history),
                "llm_traces": list(state.llm_traces),
                "runtime_info": self.runtime_info(),
                "aborted": True,
            },
        )

    def _maybe_abort(
        self,
        *,
        run_id: str,
        question: str,
        rewritten_question: str,
        state: AgentRunState,
        artifacts_dir: Path,
        snapshot: AgentExecutionSnapshot | None = None,
        plan_dags: list[dict[str, Any]] | None = None,
        clarification_questions: list[str] | None = None,
    ) -> AgentRunManifest | None:
        if not self._is_cancelled(run_id):
            return None
        self._set_live_status(
            run_id,
            status="aborted",
            current_phase="aborted",
            detail="Run aborted by user",
            active_steps=[],
            assumptions=list(state.assumptions),
            planner_actions=list(state.planner_actions),
            llm_traces=list(state.llm_traces),
            clarification_questions=list(clarification_questions or []),
        )
        manifest = self._build_aborted_manifest(
            run_id=run_id,
            question=question,
            rewritten_question=rewritten_question,
            state=state,
            artifacts_dir=artifacts_dir,
            snapshot=snapshot,
            plan_dags=plan_dags,
            clarification_questions=clarification_questions,
        )
        self._persist_manifest(manifest)
        return manifest

    def _record_step_event(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._run_lock:
            status = self._live_runs.get(run_id)
            if status is None:
                return
            event = str(payload.get("event", ""))
            entry = {
                "node_id": str(payload.get("node_id", "")),
                "capability": str(payload.get("capability", "")),
                "status": str(payload.get("status", "")),
            }
            if payload.get("started_at_utc"):
                entry["started_at_utc"] = str(payload["started_at_utc"])
            if payload.get("finished_at_utc"):
                entry["finished_at_utc"] = str(payload["finished_at_utc"])
            if payload.get("duration_ms") is not None:
                entry["duration_ms"] = float(payload["duration_ms"])
            if payload.get("error"):
                entry["error"] = str(payload["error"])
            if payload.get("short_message"):
                entry["short_message"] = str(payload["short_message"])
            if payload.get("degraded") is True:
                entry["degraded"] = True
            status.tool_calls = _merge_tool_call_rows(status.tool_calls, payload)
            tool_name = str(payload.get("tool_name", "")).strip()
            call_signature = _tool_call_signature(tool_name, dict(payload.get("inputs", {})))
            if event == "node_started":
                status.active_steps = [
                    item for item in status.active_steps if item.get("node_id") != entry["node_id"]
                ] + [entry]
                status.current_phase = "executing"
                status.detail = f"Running {tool_name or entry['capability']}"
            elif event == "node_completed":
                status.active_steps = [
                    item for item in status.active_steps if item.get("node_id") != entry["node_id"]
                ]
                status.completed_steps.append(entry)
                summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
                items_count = int(summary.get("items_count", 0) or 0)
                items_key = str(summary.get("items_key", "") or "items")
                suffix = f" -> {items_count} {items_key}" if items_count else ""
                status.detail = f"Completed {tool_name or entry['capability']}{suffix}"
            elif event == "node_failed":
                status.active_steps = [
                    item for item in status.active_steps if item.get("node_id") != entry["node_id"]
                ]
                status.failed_steps.append(entry)
                status.detail = f"Failed {tool_name or entry['capability']}"
            if call_signature:
                status.detail = f"{status.detail} | {call_signature}"
            status.updated_at_utc = datetime.now(UTC).isoformat()

    def corpus_schema(self) -> dict[str, Any]:
        metadata = self.runtime.load_metadata().head(1)
        return {
            "metadata_fields": sorted(str(column) for column in metadata.columns),
            "retrieval_backend": "opensearch+postgres",
            "document_count": int(self.runtime.load_metadata().shape[0]),
        }

    def _build_state(self, question: str, force_answer: bool, no_cache: bool) -> AgentRunState:
        tool_catalog = self.capability_catalog()
        return AgentRunState(
            question=question,
            force_answer=force_answer,
            available_capabilities=sorted(
                {capability for spec in self.registry.list_tools() for capability in spec.capabilities}
            ),
            tool_catalog=tool_catalog,
            corpus_schema=self.corpus_schema(),
            planner_calls_max=self.config.planner_calls_max,
            no_cache=no_cache,
        )

    def _needs_clarification_manifest(
        self,
        *,
        run_id: str,
        question: str,
        rewritten_question: str,
        clarification_question: str,
        assumptions: list[str],
        artifacts_dir: Path,
        state: AgentRunState,
    ) -> AgentRunManifest:
        return AgentRunManifest(
            run_id=run_id,
            question=question,
            rewritten_question=rewritten_question,
            status="needs_clarification",
            clarification_questions=[clarification_question],
            assumptions=assumptions,
            planner_actions=list(state.planner_actions),
            plan_dags=[],
            tool_calls=self._current_tool_calls(run_id),
            selected_docs=[],
            node_records=[],
            provenance_records=[],
            evidence_table=[],
            final_answer=FinalAnswerPayload(
                answer_text="",
                caveats=["Clarification is required before planning can continue."],
            ),
            artifacts_dir=str(artifacts_dir),
            metadata={
                "force_answer": bool(state.force_answer),
                "no_cache": bool(state.no_cache),
                "clarification_question": clarification_question,
                "clarification_history": list(state.clarification_history),
                "llm_traces": list(state.llm_traces),
                "runtime_info": self.runtime_info(),
            },
        )

    def _fallback_execution_diagnostics(
        self,
        *,
        question: str,
        snapshot: AgentExecutionSnapshot,
        plan_dags: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        issue_records = _execution_issue_records(snapshot)
        failures = [_diagnostic_failure_payload(item) for item in snapshot.failures[:10]]
        failed_nodes = [item for item in issue_records if item.get("status") == "failed"]
        skipped_nodes = [item for item in issue_records if item.get("status") == "skipped"]
        failure_types = {str(item.get("error_type", "")) for item in failures}
        failure_messages = [str(item.get("message", "")).strip() for item in failures if str(item.get("message", "")).strip()]
        node_messages = [str(item.get("error", "")).strip() for item in issue_records if str(item.get("error", "")).strip()]
        likely_root_causes: list[str] = []
        if "core_no_data" in failure_types:
            likely_root_causes.append(
                "A required corpus node returned no data, so downstream nodes could not safely run."
            )
        if "dependency_no_data" in failure_types:
            likely_root_causes.append(
                "A required dependency was skipped or empty, and a downstream required node was stopped."
            )
        if "execution_failed" in failure_types:
            likely_root_causes.append(
                "At least one selected tool raised an execution error instead of returning a structured no-data result."
            )
        if any("No tool available" in message or "unavailable" in message for message in failure_messages + node_messages):
            likely_root_causes.append(
                "The planner selected a capability whose implementation was unavailable in the current runtime."
            )
        if any("Plot input dependencies did not produce rows" in message for message in failure_messages + node_messages):
            likely_root_causes.append(
                "A plot node was requested, but its upstream analysis did not produce table rows to plot."
            )
        if not likely_root_causes:
            likely_root_causes.append(
                "Execution produced failed or skipped nodes; inspect the failed_nodes, skipped_nodes, and failures entries for the first concrete cause."
            )

        next_fix_trials = [
            "Check /runtime-info first, especially retrieval.health, dense model readiness, Postgres row counts, and OpenSearch count.",
            "Inspect the first failed node's capability, tool_name, error, inputs, and dependency nodes in the manifest/tool_calls.",
            "If the failure is no-data, rerun with a broader retrieval query or exhaustive retrieval; if it is an unavailable tool, enable/install that provider or force a supported fallback.",
        ]
        retrieval_no_data = any(
            str(item.get("error_type", "")) == "core_no_data"
            and str(item.get("capability", "")) in {"db_search", "sql_query_search"}
            for item in failures
        )
        if retrieval_no_data:
            user_message = (
                "No corpus documents matched the planner's retrieval query, so the run was stopped before any "
                "downstream analysis. Inspect the first node's `query` and `source` filters in the manifest "
                "and rerun with different keywords or a less restrictive filter."
            )
            likely_root_causes.insert(
                0,
                "The first retrieval node returned 0 documents, so every downstream node would be operating "
                "on an empty corpus. Common causes: keywords too narrow, source: filter referencing outlets "
                "that do not exist in this corpus, or malformed Lucene/SQL syntax in the planner output.",
            )
        elif snapshot.status == "partial":
            user_message = (
                "The run completed only partially. Some evidence may be usable, but at least one required or recovery path failed; "
                "use execution_diagnostics before trusting the answer scope."
            )
        elif snapshot.status == "failed":
            user_message = (
                "The run failed before a grounded answer could be completed. "
                "The first failed node and its upstream dependency chain are recorded in execution_diagnostics."
            )
        else:
            user_message = (
                "The run contains failed or skipped execution records. "
                "The manifest now includes structured diagnostics for the affected nodes."
            )
        return {
            "status": "generated",
            "consulted_llm": False,
            "question": question,
            "run_status": snapshot.status,
            "summary": (
                f"Execution status={snapshot.status}; "
                f"failed_nodes={len(failed_nodes)}; skipped_nodes={len(skipped_nodes)}; failures={len(failures)}."
            ),
            "user_facing_message": user_message,
            "likely_root_causes": list(dict.fromkeys(likely_root_causes)),
            "next_fix_trials": next_fix_trials,
            "failed_nodes": failed_nodes,
            "skipped_nodes": skipped_nodes,
            "failures": failures,
            "planned_dag_count": len(plan_dags or []),
            "llm_consult": {"status": "not_attempted"},
        }

    def _build_execution_diagnostics(
        self,
        *,
        state: AgentRunState | None,
        question: str,
        snapshot: AgentExecutionSnapshot,
        plan_dags: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if not _execution_needs_diagnostics(snapshot):
            return {}
        diagnostic = self._fallback_execution_diagnostics(
            question=question,
            snapshot=snapshot,
            plan_dags=plan_dags,
        )
        if not _env_flag("CORPUSAGENT2_LLM_CONSULT_ON_ERROR", True):
            diagnostic["llm_consult"] = {"status": "disabled_by_env"}
            return diagnostic
        if self.llm_client is None:
            diagnostic["llm_consult"] = {"status": "skipped", "reason": "llm_client_not_configured"}
            return diagnostic

        runtime_summary: dict[str, Any] = {}
        try:
            info = self.runtime_info()
            runtime_summary = {
                "retrieval": info.get("retrieval", {}),
                "device": info.get("device", {}),
                "providers_installed": info.get("providers_installed", {}),
            }
        except Exception as exc:
            runtime_summary = {"error": _diagnostic_text(exc)}

        messages = [
            {
                "role": "system",
                "content": (
                    "You diagnose failed or skipped CorpusAgent2 execution nodes. "
                    "Return JSON only with keys summary, likely_root_causes, next_fix_trials, user_facing_message. "
                    "Use only the supplied execution data. Do not invent corpus facts or claim the answer is supported. "
                    "Prefer concrete next checks over generic advice."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "rewritten_question": state.rewritten_question if state is not None else question,
                        "run_status": snapshot.status,
                        "deterministic_diagnostic": diagnostic,
                        "plan_dag_count": len(plan_dags or []),
                        "runtime_summary": runtime_summary,
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        try:
            trace = self.llm_client.complete_json_trace(
                messages,
                model=self.llm_config.planner_model,
                temperature=0.0,
            )
            if state is not None:
                self.orchestrator._record_llm_trace(state, stage="execution_diagnostics", trace=trace)
            payload = dict(trace.get("parsed_json", {}))
            diagnostic["consulted_llm"] = True
            diagnostic["llm_consult"] = {
                "status": "completed",
                "provider_name": trace.get("provider_name", self.llm_config.provider_name),
                "model": trace.get("model", self.llm_config.planner_model),
            }
            for key in ("summary", "user_facing_message"):
                value = _diagnostic_text(payload.get(key, ""))
                if value:
                    diagnostic[key] = value
            for key in ("likely_root_causes", "next_fix_trials"):
                value = _coerce_diagnostic_string_list(payload.get(key, []), max_items=8)
                if value:
                    diagnostic[key] = value
            return diagnostic
        except Exception as exc:
            if state is not None:
                self.orchestrator._record_llm_trace(
                    state,
                    stage="execution_diagnostics",
                    used_fallback=True,
                    error=str(exc),
                    note="LLM execution diagnostic failed; deterministic diagnostic retained.",
                )
            diagnostic["llm_consult"] = {
                "status": "failed",
                "error": _diagnostic_text(exc),
            }
            return diagnostic

    @staticmethod
    def _attach_execution_diagnostic_to_answer(
        final_answer: FinalAnswerPayload,
        diagnostic: dict[str, Any],
    ) -> None:
        if not diagnostic:
            return
        message = _diagnostic_text(diagnostic.get("user_facing_message") or diagnostic.get("summary"), max_chars=400)
        if not message:
            return
        caveat = f"Execution diagnostics: {message}"
        if caveat not in final_answer.caveats:
            final_answer.caveats.append(caveat)

    def _failed_manifest(
        self,
        *,
        run_id: str,
        question: str,
        rewritten_question: str,
        assumptions: list[str],
        artifacts_dir: Path,
        state: AgentRunState | None,
        error: Exception,
    ) -> AgentRunManifest:
        failure = AgentFailure(
            node_id="runtime",
            capability="runtime",
            error_type=type(error).__name__,
            message=str(error),
            retriable=False,
        )
        snapshot = AgentExecutionSnapshot(
            node_records=[],
            node_results={},
            failures=[failure],
            provenance_records=[],
            selected_docs=[],
            status="failed",
        )
        plan_dags = [state.last_plan] if state is not None and state.last_plan else []
        execution_diagnostics = self._build_execution_diagnostics(
            state=state,
            question=question,
            snapshot=snapshot,
            plan_dags=plan_dags,
        )
        llm_traces = list(state.llm_traces) if state is not None else []
        planner_actions = list(state.planner_actions) if state is not None else []
        clarification_history = list(state.clarification_history) if state is not None else []
        final_answer = FinalAnswerPayload(
            answer_text="The run failed before a grounded answer could be completed.",
            unsupported_parts=[str(error)],
            caveats=["Review the runtime failure details and LLM traces in the manifest."],
        )
        self._attach_execution_diagnostic_to_answer(final_answer, execution_diagnostics)
        return AgentRunManifest(
            run_id=run_id,
            question=question,
            rewritten_question=rewritten_question or question,
            status="failed",
            clarification_questions=[],
            assumptions=list(assumptions),
            planner_actions=planner_actions,
            plan_dags=plan_dags,
            tool_calls=self._current_tool_calls(run_id),
            selected_docs=[],
            node_records=[],
            provenance_records=[],
            evidence_table=[],
            final_answer=final_answer,
            artifacts_dir=str(artifacts_dir),
            failures=[failure],
            metadata={
                "force_answer": bool(state.force_answer) if state is not None else False,
                "no_cache": bool(state.no_cache) if state is not None else False,
                "clarification_history": clarification_history,
                "llm_traces": llm_traces,
                "runtime_info": self.runtime_info(),
                "runtime_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "execution_diagnostics": execution_diagnostics,
            },
        )

    def _run_query(
        self,
        run_id: str,
        question: str,
        *,
        force_answer: bool = False,
        no_cache: bool = False,
        clarification_history: list[str] | None = None,
    ) -> AgentRunManifest:
        force_answer = bool(force_answer)
        no_cache = bool(no_cache)
        artifacts_dir = (self.config.outputs_root / run_id).resolve()
        (artifacts_dir / "nodes").mkdir(parents=True, exist_ok=True)
        self._set_live_status(
            run_id,
            question=question,
            status="running",
            current_phase="initializing",
            detail="Preparing run state",
            force_answer=force_answer,
            no_cache=no_cache,
        )

        state = self._build_state(question=question, force_answer=force_answer, no_cache=no_cache)
        if clarification_history:
            state.clarification_history = list(clarification_history)
        maybe_aborted = self._maybe_abort(
            run_id=run_id,
            question=question,
            rewritten_question=question,
            state=state,
            artifacts_dir=artifacts_dir,
        )
        if maybe_aborted is not None:
            return maybe_aborted

        try:
            self.working_store.create_run(
                run_id=run_id,
                question=question,
                rewritten_question="",
                force_answer=force_answer,
                no_cache=no_cache,
            )
        except Exception:
            if self.require_backend_services:
                raise
            fallback_store = InMemoryWorkingSetStore()
            fallback_store.document_lookup.update(self.runtime.doc_lookup())
            self.working_store = fallback_store
            self.working_store.create_run(
                run_id=run_id,
                question=question,
                rewritten_question="",
                force_answer=force_answer,
                no_cache=no_cache,
            )

        self._set_live_status(run_id, current_phase="rephrase_or_clarify", detail="Rephrasing or clarifying question")
        rephrase_action = self.orchestrator.rephrase_or_clarify(state)
        state.planner_calls_used += 1
        state.planner_actions.append(rephrase_action.to_dict())
        self._set_live_status(run_id, planner_actions=list(state.planner_actions), llm_traces=list(state.llm_traces))
        if rephrase_action.assumptions:
            state.assumptions = list(dict.fromkeys(state.assumptions + rephrase_action.assumptions))
            self._set_live_status(run_id, assumptions=list(state.assumptions))
        maybe_aborted = self._maybe_abort(
            run_id=run_id,
            question=question,
            rewritten_question=rephrase_action.rewritten_question or question,
            state=state,
            artifacts_dir=artifacts_dir,
        )
        if maybe_aborted is not None:
            return maybe_aborted
        if rephrase_action.action == "grounded_rejection":
            manifest = AgentRunManifest(
                run_id=run_id,
                question=question,
                rewritten_question=rephrase_action.rewritten_question or question,
                status="rejected",
                clarification_questions=[],
                assumptions=list(state.assumptions),
                planner_actions=list(state.planner_actions),
                plan_dags=[],
                tool_calls=self._current_tool_calls(run_id),
                selected_docs=[],
                node_records=[],
                provenance_records=[],
                evidence_table=[],
                final_answer=FinalAnswerPayload(
                    answer_text=rephrase_action.rejection_reason,
                    unsupported_parts=[rephrase_action.rejection_reason],
                    caveats=[],
                ),
                artifacts_dir=str(artifacts_dir),
                metadata={
                    "force_answer": bool(state.force_answer),
                    "no_cache": bool(state.no_cache),
                    "llm_traces": list(state.llm_traces),
                    "runtime_info": self.runtime_info(),
                },
            )
            self._persist_manifest(manifest)
            return manifest

        state.rewritten_question = rephrase_action.rewritten_question or question

        if rephrase_action.action == "ask_clarification" and not state.force_answer:
            manifest = self._needs_clarification_manifest(
                run_id=run_id,
                question=question,
                rewritten_question=state.rewritten_question,
                clarification_question=rephrase_action.clarification_question,
                assumptions=list(state.assumptions),
                artifacts_dir=artifacts_dir,
                state=state,
            )
            self._set_live_status(
                run_id,
                status="needs_clarification",
                current_phase="waiting_for_user",
                clarification_questions=[rephrase_action.clarification_question],
                assumptions=list(state.assumptions),
                planner_actions=list(state.planner_actions),
                detail="Waiting for clarification before planning",
            )
            self._persist_manifest(manifest)
            return manifest

        unsupported_date_reason = self._date_window_unsupported_reason(
            self.orchestrator._planning_context_text(state, state.rewritten_question)
        )
        if unsupported_date_reason:
            final_answer = self._out_of_corpus_model_answer(
                state=state,
                unsupported_reason=unsupported_date_reason,
            )
            manifest = AgentRunManifest(
                run_id=run_id,
                question=question,
                rewritten_question=state.rewritten_question,
                status="rejected",
                clarification_questions=[],
                assumptions=list(state.assumptions),
                planner_actions=list(state.planner_actions),
                plan_dags=[],
                tool_calls=self._current_tool_calls(run_id),
                selected_docs=[],
                node_records=[],
                provenance_records=[],
                evidence_table=[],
                final_answer=final_answer,
                artifacts_dir=str(artifacts_dir),
                metadata={
                    "force_answer": bool(state.force_answer),
                    "no_cache": bool(state.no_cache),
                    "llm_traces": list(state.llm_traces),
                    "runtime_info": self.runtime_info(),
                    "answer_grounding": "model_only_out_of_corpus",
                    "out_of_corpus_model_answer": True,
                    "corpus_unsupported_reason": unsupported_date_reason,
                    "out_of_corpus_model": self.llm_config.planner_model,
                },
            )
            self._set_live_status(
                run_id,
                status="rejected",
                current_phase="out_of_corpus_model_answer",
                detail=unsupported_date_reason,
                planner_actions=list(state.planner_actions),
                llm_traces=list(state.llm_traces),
            )
            self._persist_manifest(manifest)
            return manifest

        self._set_live_status(run_id, current_phase="planning", detail="Building plan DAG")
        plan_action = self.orchestrator.plan(state)
        state.planner_calls_used += 1
        state.planner_actions.append(plan_action.to_dict())
        if plan_action.assumptions:
            state.assumptions = list(dict.fromkeys(state.assumptions + plan_action.assumptions))
        if plan_action.plan_dag is not None:
            state.last_plan = plan_action.plan_dag.to_dict()
        self._set_live_status(
            run_id,
            planner_actions=list(state.planner_actions),
            plan_dags=[state.last_plan] if state.last_plan else [],
            llm_traces=list(state.llm_traces),
        )
        maybe_aborted = self._maybe_abort(
            run_id=run_id,
            question=question,
            rewritten_question=state.rewritten_question,
            state=state,
            artifacts_dir=artifacts_dir,
            clarification_questions=[plan_action.clarification_question] if plan_action.clarification_question else [],
        )
        if maybe_aborted is not None:
            return maybe_aborted
        if plan_action.action == "ask_clarification" and state.force_answer:
            forced_plan = self.orchestrator._heuristic_plan(state)
            if forced_plan.plan_dag is not None:
                forced_plan.plan_dag = self.orchestrator._normalize_plan_dag(
                    forced_plan.plan_dag,
                    question_text=self.orchestrator._planning_context_text(state, forced_plan.rewritten_question),
                )
            forced_plan.assumptions = list(
                dict.fromkeys(
                    list(plan_action.assumptions)
                    + list(forced_plan.assumptions)
                    + ["force_answer=true: planner clarification skipped and best-effort heuristic plan was used."]
                )
            )
            plan_action = forced_plan
            state.planner_actions[-1] = forced_plan.to_dict()
            if forced_plan.plan_dag is not None:
                state.last_plan = forced_plan.plan_dag.to_dict()
            if forced_plan.assumptions:
                state.assumptions = list(dict.fromkeys(state.assumptions + forced_plan.assumptions))
            self._set_live_status(
                run_id,
                planner_actions=list(state.planner_actions),
                plan_dags=[state.last_plan] if state.last_plan else [],
                assumptions=list(state.assumptions),
                detail="Force answer skipped planner clarification",
            )
        if plan_action.action == "ask_clarification" and not state.force_answer:
            manifest = self._needs_clarification_manifest(
                run_id=run_id,
                question=question,
                rewritten_question=state.rewritten_question,
                clarification_question=plan_action.clarification_question,
                assumptions=list(state.assumptions),
                artifacts_dir=artifacts_dir,
                state=state,
            )
            self._set_live_status(
                run_id,
                status="needs_clarification",
                current_phase="waiting_for_user",
                clarification_questions=[plan_action.clarification_question],
                assumptions=list(state.assumptions),
                planner_actions=list(state.planner_actions),
                detail="Planner requested clarification",
            )
            self._persist_manifest(manifest)
            return manifest

        if plan_action.plan_dag is None:
            raise RuntimeError("Planner did not produce a PlanDAG.")

        state.last_plan = plan_action.plan_dag.to_dict()
        context = AgentExecutionContext(
            run_id=run_id,
            artifacts_dir=artifacts_dir,
            search_backend=self.search_backend,
            working_store=self.working_store,
            llm_client=self.llm_client,
            python_runner=self.python_runner,
            runtime=self.runtime,
            state=state,
            event_callback=lambda payload: self._record_step_event(run_id, payload),
            cancel_requested=lambda: self._is_cancelled(run_id),
        )

        self._set_live_status(run_id, current_phase="executing", detail="Executing plan DAG")
        snapshot = asyncio.run(self.executor.execute(plan_action.plan_dag, context))
        plan_dags = [plan_action.plan_dag.to_dict()]
        maybe_aborted = self._maybe_abort(
            run_id=run_id,
            question=question,
            rewritten_question=state.rewritten_question,
            state=state,
            artifacts_dir=artifacts_dir,
            snapshot=snapshot,
            plan_dags=plan_dags,
        )
        if maybe_aborted is not None:
            return maybe_aborted

        if snapshot.failures:
            state.failures = [item.to_dict() for item in snapshot.failures]
            self._set_live_status(run_id, current_phase="revising_after_failure", detail="Revising plan after failure")
            revised = self.orchestrator.revise_after_failure(state, snapshot.failures[0], snapshot)
            if revised is not None and revised.plan_dag is not None:
                state.planner_calls_used += 1
                state.planner_actions.append(revised.to_dict())
                if revised.assumptions:
                    state.assumptions = list(dict.fromkeys(state.assumptions + revised.assumptions))
                self._set_live_status(
                    run_id,
                    planner_actions=list(state.planner_actions),
                    plan_dags=plan_dags + [revised.plan_dag.to_dict()],
                    llm_traces=list(state.llm_traces),
                )
                revised_snapshot = asyncio.run(self.executor.execute(revised.plan_dag, context))
                plan_dags.append(revised.plan_dag.to_dict())
                snapshot = _merge_execution_snapshots(snapshot, revised_snapshot)
                maybe_aborted = self._maybe_abort(
                    run_id=run_id,
                    question=question,
                    rewritten_question=state.rewritten_question,
                    state=state,
                    artifacts_dir=artifacts_dir,
                    snapshot=snapshot,
                    plan_dags=plan_dags,
                )
                if maybe_aborted is not None:
                    return maybe_aborted

        self._set_live_status(run_id, current_phase="final_synthesis", detail="Synthesizing grounded answer")
        maybe_aborted = self._maybe_abort(
            run_id=run_id,
            question=question,
            rewritten_question=state.rewritten_question,
            state=state,
            artifacts_dir=artifacts_dir,
            snapshot=snapshot,
            plan_dags=plan_dags,
        )
        if maybe_aborted is not None:
            return maybe_aborted
        final_answer = self.orchestrator.synthesize(state, snapshot)
        execution_diagnostics = self._build_execution_diagnostics(
            state=state,
            question=question,
            snapshot=snapshot,
            plan_dags=plan_dags,
        )
        self._attach_execution_diagnostic_to_answer(final_answer, execution_diagnostics)
        manifest = AgentRunManifest(
            run_id=run_id,
            question=question,
            rewritten_question=state.rewritten_question,
            status=snapshot.status,
            clarification_questions=[],
            assumptions=list(state.assumptions),
            planner_actions=list(state.planner_actions),
            plan_dags=plan_dags,
            tool_calls=self._current_tool_calls(run_id),
            selected_docs=list(snapshot.selected_docs),
            node_records=list(snapshot.node_records),
            provenance_records=list(snapshot.provenance_records),
            evidence_table=list(self.orchestrator._extract_evidence(snapshot)),
            final_answer=final_answer,
            artifacts_dir=str(artifacts_dir),
            failures=list(snapshot.failures),
            metadata={
                "force_answer": bool(state.force_answer),
                "no_cache": bool(state.no_cache),
                "planner_calls_used": state.planner_calls_used,
                "clarification_history": list(state.clarification_history),
                "llm_traces": list(state.llm_traces),
                "runtime_info": self.runtime_info(),
                "execution_diagnostics": execution_diagnostics,
            },
        )
        self._persist_manifest(manifest)
        return manifest

    def handle_query(
        self,
        question: str,
        *,
        force_answer: bool = False,
        no_cache: bool = False,
        clarification_history: list[str] | None = None,
        run_id: str | None = None,
    ) -> AgentRunManifest:
        run_id = run_id or f"agent_{uuid.uuid4().hex[:12]}"
        self._register_cancel_event(run_id)
        try:
            return self._run_query(
                run_id,
                question,
                force_answer=force_answer,
                no_cache=no_cache,
                clarification_history=clarification_history,
            )
        except Exception as exc:
            artifacts_dir = (self.config.outputs_root / run_id).resolve()
            (artifacts_dir / "nodes").mkdir(parents=True, exist_ok=True)
            manifest = self._failed_manifest(
                run_id=run_id,
                question=question,
                rewritten_question=question,
                assumptions=[],
                artifacts_dir=artifacts_dir,
                state=None,
                error=exc,
            )
            self._persist_manifest(manifest)
            return manifest

    def submit_query(
        self,
        question: str,
        *,
        force_answer: bool = False,
        no_cache: bool = False,
        clarification_history: list[str] | None = None,
    ) -> LiveRunStatus:
        run_id = f"agent_{uuid.uuid4().hex[:12]}"
        status = self._set_live_status(
            run_id,
            question=question,
            status="queued",
            current_phase="queued",
            detail="Queued for execution",
            force_answer=bool(force_answer),
            no_cache=bool(no_cache),
        )
        self._register_cancel_event(run_id)

        def _runner() -> None:
            try:
                self._run_query(
                    run_id,
                    question,
                    force_answer=force_answer,
                    no_cache=no_cache,
                    clarification_history=clarification_history,
                )
            except Exception as exc:
                artifacts_dir = (self.config.outputs_root / run_id).resolve()
                (artifacts_dir / "nodes").mkdir(parents=True, exist_ok=True)
                with self._run_lock:
                    live = self._live_runs.get(run_id)
                state = AgentRunState(
                    question=question,
                    rewritten_question="",
                    clarification_history=list(clarification_history or []),
                    assumptions=list(live.assumptions) if live is not None else [],
                    planner_actions=list(live.planner_actions) if live is not None else [],
                    llm_traces=list(live.llm_traces) if live is not None else [],
                )
                manifest = self._failed_manifest(
                    run_id=run_id,
                    question=question,
                    rewritten_question=question,
                    assumptions=list(state.assumptions),
                    artifacts_dir=artifacts_dir,
                    state=state,
                    error=exc,
                )
                self._persist_manifest(manifest)
            finally:
                with self._run_lock:
                    self._run_threads.pop(run_id, None)

        thread = threading.Thread(target=_runner, daemon=True, name=f"corpusagent2-{run_id}")
        with self._run_lock:
            self._run_threads[run_id] = thread
        thread.start()
        return status

    def _persist_manifest(self, manifest: AgentRunManifest) -> None:
        manifest_path = Path(manifest.artifacts_dir) / "run_manifest.json"
        save_agent_manifest(manifest_path, manifest.to_dict())
        try:
            self.working_store.record_output(manifest.run_id, "final_answer", manifest.final_answer.to_dict())
            self.working_store.finalize_run(manifest.run_id, manifest.status)
        except Exception:
            pass
        self._set_live_status(
            manifest.run_id,
            question=manifest.question,
            status=manifest.status,
            current_phase="completed" if manifest.status in {"completed", "partial"} else manifest.status,
            detail="Run finished",
            active_steps=[],
            assumptions=list(manifest.assumptions),
            planner_actions=list(manifest.planner_actions),
            plan_dags=list(manifest.plan_dags),
            tool_calls=list(manifest.tool_calls),
            llm_traces=list(manifest.metadata.get("llm_traces", [])),
            clarification_questions=list(manifest.clarification_questions),
            force_answer=bool(manifest.metadata.get("force_answer", False)),
            no_cache=bool(manifest.metadata.get("no_cache", False)),
            final_manifest_path=str(manifest_path),
        )

    def get_run_status(self, run_id: str) -> dict[str, Any]:
        with self._run_lock:
            live = self._live_runs.get(run_id)
            if live is not None:
                return live.to_dict()
        return self.get_run(run_id)

    def abort_run(self, run_id: str) -> dict[str, Any]:
        with self._run_lock:
            status = self._live_runs.get(run_id)
            if status is None:
                raise FileNotFoundError(f"Run not found for run_id={run_id}")
            current_status = str(status.status)
            if current_status in TERMINAL_RUN_STATUSES:
                return status.to_dict()
            event = self._run_cancel_events.get(run_id)
            if event is None:
                event = threading.Event()
                self._run_cancel_events[run_id] = event
            event.set()
        return self._mark_abort_requested_live_status(
            run_id,
            detail="Abort requested; waiting for current step to stop",
        )

    def abort_all_runs(self) -> dict[str, Any]:
        aborted_run_ids: list[str] = []
        with self._run_lock:
            run_ids = list(self._live_runs.keys())
        for run_id in run_ids:
            with self._run_lock:
                status = self._live_runs.get(run_id)
                current_status = str(status.status) if status is not None else ""
            if current_status and current_status not in TERMINAL_RUN_STATUSES:
                self.abort_run(run_id)
                aborted_run_ids.append(run_id)
        return {"aborted_run_ids": aborted_run_ids, "count": len(aborted_run_ids)}

    def get_run(self, run_id: str) -> dict[str, Any]:
        manifest_path = self.config.outputs_root / run_id / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run manifest not found for run_id={run_id}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def resolve_artifact_path(self, run_id: str, artifact_path: str) -> Path:
        manifest = self.get_run(run_id)
        base_dir = Path(str(manifest.get("artifacts_dir", ""))).resolve()
        target = Path(artifact_path)
        resolved = target.resolve() if target.is_absolute() else (base_dir / target).resolve()
        try:
            resolved.relative_to(base_dir)
        except ValueError as exc:
            raise PermissionError("Artifact path escapes the run artifact directory.") from exc
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        return resolved
