from __future__ import annotations

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
from .agent_capabilities import AgentExecutionContext, build_agent_registry
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

TERMINAL_RUN_STATUSES = {"completed", "partial", "failed", "rejected", "needs_clarification", "aborted"}
SOURCE_SCOPE_ALIASES = {
    "swiss": (
        "swissinfoch",
        "nzzch",
        "tagesanzeigerch",
        "blickch",
        "letempsch",
        "20minch",
        "20minuten",
        "20minutench",
        "srfch",
        "rtsch",
        "watsonch",
    ),
}
EXPLICIT_SOURCE_SCOPE_ALIASES = (
    {
        "filters": ("nzz",),
        "tokens": ("nzz",),
        "patterns": (r"\bnzz(?:\.ch)?\b",),
    },
    {
        "filters": ("tagesanzeiger",),
        "tokens": ("tages", "anzeiger", "tagesanzeiger"),
        "patterns": (r"\btages[\s-]*anzeiger(?:\.ch)?\b",),
    },
)
TOPIC_QUERY_EXPANSIONS = (
    {
        "triggers": ("football", "soccer", "fussball", "fußball"),
        "query": "football OR soccer OR fussball OR fußball",
    },
    {
        "triggers": ("climate", "klima", "climat"),
        "query": "climate OR klima OR climat OR klimawandel OR rechauffement OR réchauffement",
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
        "clean_normalize",
        "entity_link",
        "extract_keyterms",
        "topic_model",
        "text_classify",
        "sentiment",
        "ner",
        "pos_morph",
        "lemmatize",
        "quote_extract",
        "quote_attribute",
        "claim_span_extract",
        "claim_strength_score",
        "build_evidence_table",
        "doc_embeddings",
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

        search_node_id = next((node.node_id for node in dag.nodes if node.capability in self._SEARCH_BACKBONE_CAPABILITIES), "")
        fetch_node_id = next((node.node_id for node in dag.nodes if node.capability == "fetch_documents"), "")
        requires_retrieval_backbone = any(
            node.capability in self._DOC_RETRIEVAL_BACKBONE_CAPABILITIES
            for node in dag.nodes
        )
        if requires_retrieval_backbone and not search_node_id:
            search_node_id = unique_node_id("search")
        if requires_retrieval_backbone and not fetch_node_id:
            fetch_node_id = unique_node_id("fetch")

        dag_text = " ".join(
            part
            for part in [question_text, dag.metadata.get("question_family", "")]
            if part
        ).strip()
        query_text = ""
        if dag_text:
            query_text = self._compact_query_terms(dag_text, self._query_anchor_terms(dag_text)) or dag_text
        search_inputs = infer_retrieval_budget(
            dag_text,
            configured_mode=os.getenv("CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE", "hybrid"),
        ).to_inputs()
        if query_text:
            repaired_query = self._repair_search_query(str(search_inputs.get("query", "")), query_text)
            search_inputs["query"] = self._apply_source_scope_to_query(repaired_query, dag_text)
        if requires_retrieval_backbone:
            search_inputs.update(self._extract_date_window(dag_text))
        if self._needs_source_comparison_analysis(dag_text):
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
                if query_text:
                    repaired_query = self._repair_search_query(
                        str(normalized_search_inputs.get("query", "")),
                        query_text,
                    )
                    normalized_search_inputs["query"] = self._apply_source_scope_to_query(repaired_query, dag_text)
                if self._needs_source_comparison_analysis(dag_text):
                    normalized_search_inputs["retrieval_strategy"] = "exhaustive_analytic"
                    normalized_search_inputs["retrieve_all"] = True
                    normalized_search_inputs["top_k"] = 0
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
        if self._needs_temporal_portrayal_analysis(dag_text):
            self._ensure_temporal_portrayal_nodes(normalized_nodes, unique_node_id, dag_text)
            if metadata.get("question_family", "") in {"", "generic"}:
                metadata["question_family"] = "temporal_portrayal_shift"
        elif self._needs_entity_trend_analysis(dag_text):
            self._ensure_entity_trend_nodes(normalized_nodes, unique_node_id, dag_text)
            if metadata.get("question_family", "") in {"", "generic"}:
                metadata["question_family"] = "entity_trend"
        elif self._needs_source_comparison_analysis(dag_text):
            self._ensure_source_comparison_nodes(normalized_nodes, unique_node_id)
            if metadata.get("question_family", "") in {"", "generic"}:
                metadata["question_family"] = "source_comparison"
        return AgentPlanDAG(nodes=normalized_nodes, metadata=metadata)

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

    def _needs_entity_trend_analysis(self, text: str) -> bool:
        lowered = str(text or "").lower()
        asks_entities = any(
            phrase in lowered
            for phrase in (
                "named entities",
                "named entity",
                "which actors",
                "actors dominated",
                "actor dominated",
                "entities dominate",
                "entities dominated",
            )
        )
        asks_change = any(term in lowered for term in ("over time", "change", "changed", "trend", "evolve", "evolved"))
        return asks_entities and asks_change

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
                and str(node.inputs.get("task", "")).lower() in {"named_entity_frequency", "entity_frequency", "entity_prominence", "actor_prominence"}
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
        if not any(node.capability == "plot_artifact" and topics_node_id in node.depends_on for node in normalized_nodes):
            normalized_nodes.append(
                AgentPlanNode(
                    node_id=unique_node_id("plot_topics"),
                    capability="plot_artifact",
                    inputs={"plot_name": "portrayal_topics"},
                    depends_on=[topics_node_id or keyterms_node_id],
                    optional=True,
                )
            )

    def _question_with_clarifications(self, state: AgentRunState) -> str:
        history = [str(item).strip() for item in state.clarification_history if str(item).strip()]
        if not history:
            return state.question
        suffix = "\n".join(f"- {item}" for item in history)
        return f"{state.question}\n\nUser clarification history:\n{suffix}"

    def _extract_date_window(self, text: str) -> dict[str, str]:
        year_values = sorted({int(item) for item in re.findall(r"\b(?:19|20)\d{2}\b", text)})
        if not year_values:
            return {}
        if len(year_values) == 1:
            year = year_values[0]
            return {"date_from": f"{year}-01-01", "date_to": f"{year}-12-31"}
        return {
            "date_from": f"{year_values[0]}-01-01",
            "date_to": f"{year_values[-1]}-12-31",
        }

    def _compact_query_terms(self, text: str, preferred_terms: list[str]) -> str:
        found_terms: list[str] = []
        lowered = text.lower()
        blocked_source_tokens = self._source_scope_query_tokens_for_question(text)
        preferred_stopwords = {
            "actor", "actors", "does", "perceived", "perception", "portrayal", "portrayed",
        }
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
            "coverage", "framing", "frame", "around", "during", "between", "across", "their", "there",
            "correspond", "corresponded", "media", "shift", "shifted",
            "does",
            "change", "changed", "changes", "changing", "evolve", "evolved", "evolution", "over", "time",
            "different", "differently", "difference", "differences", "compare", "compared", "comparison", "versus",
            "dominate", "dominates", "dominated", "dominant", "entity", "entities", "named", "actor", "actors",
            "public", "discourse", "newspaper", "newspapers", "report", "reports", "reported",
            "perceived", "perception", "portrayal", "portrayed",
        }
        filtered = [token for token in tokens if token.lower() not in stopwords and token.lower() not in blocked_source_tokens]
        compact = " ".join(filtered[:8]).strip()
        return self._expand_topic_query(compact, text) or compact

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
        return False

    def _repair_search_query(self, planned_query: str, fallback_query: str) -> str:
        planned = str(planned_query or "").strip()
        fallback = str(fallback_query or "").strip()
        if not planned:
            return fallback
        if fallback and self._query_needs_topical_repair(planned, fallback):
            return fallback
        return planned

    def _explicit_source_scope_matches(self, text: str) -> list[dict[str, Any]]:
        lowered = str(text or "").lower()
        matches: list[dict[str, Any]] = []
        for alias in EXPLICIT_SOURCE_SCOPE_ALIASES:
            if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in alias["patterns"]):
                matches.append(alias)
        return matches

    def _explicit_source_scope_is_active(self, text: str, matches: list[dict[str, Any]]) -> bool:
        if not matches:
            return False
        lowered = str(text or "").lower()
        if len(matches) >= 2 and re.search(r"\b(vs\.?|versus|compared? to|between|differ(?:ent|ently)|compare)\b", lowered):
            return True
        return bool(
            re.search(
                r"\b(?:newspapers?|media|press|outlets?|sources?|publisher|publishers|reports?|coverage)\b",
                lowered,
            )
        )

    def _source_scope_filters_for_question(self, text: str) -> list[str]:
        lowered = str(text or "").lower()
        filters: list[str] = []
        if re.search(r"\bswiss\s+(?:newspapers?|media|press|outlets?|sources?)\b", lowered) or re.search(
            r"\b(?:newspapers?|media|press|outlets?|sources?)\s+in\s+switzerland\b",
            lowered,
        ):
            filters.extend(SOURCE_SCOPE_ALIASES["swiss"])
        explicit_matches = self._explicit_source_scope_matches(text)
        if self._explicit_source_scope_is_active(text, explicit_matches):
            for match in explicit_matches:
                filters.extend(str(item) for item in match["filters"])
        return list(dict.fromkeys(item for item in filters if item))

    def _source_scope_query_tokens_for_question(self, text: str) -> set[str]:
        explicit_matches = self._explicit_source_scope_matches(text)
        if not self._explicit_source_scope_is_active(text, explicit_matches):
            return set()
        blocked: set[str] = set()
        for match in explicit_matches:
            blocked.update(str(item).lower() for item in match["tokens"])
        return blocked

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
        for alias in matches:
            for pattern in alias["patterns"]:
                cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
            for token in alias["tokens"]:
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
            return cleaned_query
        if not filters:
            return cleaned_query
        rendered = " OR ".join(f'"{item}"' for item in filters)
        return f"({cleaned_query}) AND source:({rendered})"

    def _query_anchor_terms(self, text: str) -> list[str]:
        anchors: list[str] = []
        seen: set[str] = set()
        blocked_terms: set[str] = set()
        blocked_source_tokens = self._source_scope_query_tokens_for_question(text)
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
            "explained",
            "explain",
            "identified",
            "identifying",
            "identify",
            "include",
            "included",
            "including",
            "individual",
            "it",
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
            "such",
            "that",
            "the",
            "their",
            "there",
            "this",
            "those",
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
            "perceived",
            "perception",
            "portrayal",
            "portrayed",
            "versus",
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
        lowered = str(text or "").lower()
        explicit_match = re.search(
            r"\b(?:ticker|symbol)\s*[:=]?\s*\$?([A-Z]{1,5}(?:\.[A-Z])?)\b",
            text,
        )
        if explicit_match:
            return explicit_match.group(1)
        cashtag_match = re.search(r"(?<![A-Za-z0-9])\$([A-Z]{1,5}(?:\.[A-Z])?)\b", text)
        if cashtag_match:
            return cashtag_match.group(1)
        if any(term in lowered for term in ("crude oil", "oil price", "oil prices", "wti", "brent")):
            return "CL=F"
        if any(term in lowered for term in ("gasoline price", "gas prices", "gas price")):
            return "RB=F"
        return ""

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
        if query_text:
            search_inputs["query"] = self._apply_source_scope_to_query(query_text, question_text)
        elif "query" in merged_inputs and str(merged_inputs.get("query", "")).strip():
            search_inputs["query"] = self._apply_source_scope_to_query(str(merged_inputs["query"]), question_text)
        elif str(search_inputs.get("query", "")).strip():
            search_inputs["query"] = self._apply_source_scope_to_query(str(search_inputs["query"]), question_text)
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
                AgentPlanNode("plot_topics", "plot_artifact", inputs={"plot_name": "framing_topics"}, depends_on=["topics"], optional=True),
            ]
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
                            depends_on=["series", "fetch"],
                        )
                    )
                    nodes.append(
                        AgentPlanNode(
                            "plot_market",
                            "plot_artifact",
                            inputs={
                                "plot_name": f"{market_ticker.lower()}_market_overlay",
                                "plot_type": "line",
                                "x": "time_bin",
                                "y": "market_close",
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
                    question_text=heuristic.rewritten_question or state.rewritten_question or state.question,
                )
            return heuristic
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the planning module for a corpus agent operating over a user-provided corpus. "
                    "Return JSON with keys action, rewritten_question, assumptions, clarification_question, rejection_reason, message, plan_dag. "
                    "Allowed actions: ask_clarification, emit_plan_dag, grounded_rejection. "
                    "Keep plans compact and parallel where possible. "
                    "Each plan node must include node_id, capability, optional tool_name, inputs, and depends_on. "
                    "Use the provided tool_catalog to choose concrete repo tools when that improves control or fallback behavior. "
                    "Prefer typed repo tools before python_runner. "
                    "Use python_runner only as a bounded last-resort fallback when no typed tool fits or a typed tool has already failed. "
                    "Prefer db_search plus fetch_documents for normal retrieval. "
                    "Every db_search or sql_query_search node must include a non-empty query string derived from the rewritten question; do not rely on implicit query context. "
                    "Use sql_query_search when hybrid retrieval is sparse, off-target, or entity coverage is poor. "
                    "Choose an explicit retrieval_strategy when planning retrieval-backed work. "
                    "Use retrieval_strategy='exhaustive_analytic' for corpus-wide aggregates, distributions, trends, comparisons, and questions that ask for all relevant records; in that mode the goal is to materialize the full candidate working set before analysis rather than relying on a tiny ranked slice. "
                    "Use retrieval_strategy='precision_ranked' for targeted evidence lookups where the best supporting documents matter more than full-population coverage. "
                    "Use retrieval_strategy='semantic_exploratory' for similarity, thematic, or concept-discovery questions where semantic closeness matters more than exact lexical overlap. "
                    "Size retrieval budgets to the question, and never use tiny default retrieval budgets for broad analytical questions. "
                    "When using db_search for ranked retrieval, set top_k and, when helpful, lexical_top_k, dense_top_k, rerank_top_k, and use_rerank based on the scope needed to answer the question faithfully. "
                    "When retrieve_all is true, top_k is only a fallback budget and must not be treated as the analyzed population size. "
                    "For analytical frequency tables, use build_evidence_table with supported task names exactly: noun_frequency_distribution for noun/POS lemma counts, and summary_stats for compact aggregate summaries. "
                    "noun_frequency_distribution rows contain lemma, count, relative_frequency, document_frequency, and rank; plot them with x='lemma' and y='count'. "
                    "Do not invent near-synonym task names such as aggregate_token_frequencies unless a tool catalog entry explicitly documents them. "
                    "For plot_artifact, depend on the analytical table node and pass x, y, limit or top_k, and title; do not plot document evidence rows as if they were aggregate rows. "
                    "When a clarification says one term means another, use the resolved term as the retrieval anchor; avoid hyphenated synonym paraphrases that introduce broad generic anchors. "
                    "When the user specifies an output limit such as 'top 20', pass that through to the relevant aggregation or plot node instead of hardcoding your own. "
                    "Treat clarification_history as authoritative user follow-up memory. If prior follow-up answers only resolve part of an ambiguity, ask only for the missing remainder instead of repeating the full earlier clarification prompt. "
                    "Do not assume the corpus is news unless the question or corpus schema indicates that."
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
                    question_text=action.rewritten_question or state.rewritten_question or state.question,
                )
            if action.action == "ask_clarification" and state.force_answer:
                heuristic = self._heuristic_plan(state)
                if heuristic.action == "emit_plan_dag" and heuristic.plan_dag is not None:
                    heuristic.plan_dag = self._normalize_plan_dag(
                        heuristic.plan_dag,
                        question_text=heuristic.rewritten_question or state.rewritten_question or state.question,
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
                        question_text=heuristic.rewritten_question or state.rewritten_question or state.question,
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
                    question_text=heuristic.rewritten_question or state.rewritten_question or state.question,
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
                    "When has_external_series is true and summary.external_series is present, use that compact external-series summary and do not say the external series is missing."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": state.question,
                        "rewritten_question": state.rewritten_question,
                        "summary": summary,
                        "evidence_rows": evidence_rows,
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

    def _snapshot_caveats(self, snapshot: AgentExecutionSnapshot) -> list[str]:
        caveats: list[str] = []
        for result in snapshot.node_results.values():
            for caveat in result.caveats:
                caveat_text = str(caveat).strip()
                if caveat_text:
                    caveats.append(caveat_text)

            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            for key in ("no_data_reason", "reason", "warning"):
                reason_text = str(metadata.get(key, "")).strip()
                if reason_text:
                    caveats.append(reason_text)

            payload = result.payload if isinstance(result.payload, dict) else {}
            for key in ("no_data_reason", "reason", "warning"):
                reason_text = str(payload.get(key, "")).strip()
                if reason_text:
                    caveats.append(reason_text)
        return list(dict.fromkeys(caveats))

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

    def _derive_summary(self, snapshot: AgentExecutionSnapshot) -> dict[str, Any]:
        summary: dict[str, Any] = {}
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
                grouped: dict[str, int] = {}
                for row in rows:
                    entity = str(row.get("entity", ""))
                    grouped[entity] = grouped.get(entity, 0) + int(row.get("count", 1))
                summary["entity_trend"] = sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:15]
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

    def _has_external_series(self, snapshot: AgentExecutionSnapshot) -> bool:
        return bool(self._external_series_rows(snapshot))

    def _apply_answer_guardrails(
        self,
        state: AgentRunState,
        snapshot: AgentExecutionSnapshot,
        answer: FinalAnswerPayload,
    ) -> FinalAnswerPayload:
        question_text = f"{state.question} {state.rewritten_question}".lower()
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
        self._live_runs: dict[str, LiveRunStatus] = {}
        self._run_cancel_events: dict[str, threading.Event] = {}
        self._run_threads: dict[str, threading.Thread] = {}
        self._run_lock = threading.Lock()
        self._llm_override_active = False
        try:
            self._startup_repaired_runs = int(self.working_store.cleanup_interrupted_runs())
        except Exception:
            self._startup_repaired_runs = 0

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
        try:
            dsn = pg_dsn_from_env(required=False)
        except Exception:
            dsn = ""
        if dsn:
            return PostgresWorkingSetStore(dsn=dsn, documents_table=pg_table_from_env(default="article_corpus"))
        if self.require_backend_services:
            raise RuntimeError("Postgres working store is required but CORPUSAGENT2_PG_DSN is not configured.")
        store = InMemoryWorkingSetStore()
        store.document_lookup.update(self.runtime.doc_lookup())
        return store

    def capability_catalog(self) -> list[dict[str, Any]]:
        specs = sorted(
            self.registry.list_tools(),
            key=lambda spec: (spec.capabilities[0] if spec.capabilities else "", spec.tool_name),
        )
        return [spec.to_dict() for spec in specs]

    def _active_run_ids(self) -> list[str]:
        with self._run_lock:
            return [
                run_id
                for run_id, status in self._live_runs.items()
                if status.status not in TERMINAL_RUN_STATUSES
            ]

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
        provider_modules = {}
        for module_name in ["spacy", "textacy", "stanza", "nltk", "gensim", "flair", "textblob", "torch", "yfinance"]:
            provider_modules[module_name] = importlib.util.find_spec(module_name) is not None
        device_report = runtime_device_report()
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
        retrieval_health = self.runtime.retrieval_health()
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
                "clarification_question": clarification_question,
                "clarification_history": list(state.clarification_history),
                "llm_traces": list(state.llm_traces),
                "runtime_info": self.runtime_info(),
            },
        )

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
        llm_traces = list(state.llm_traces) if state is not None else []
        planner_actions = list(state.planner_actions) if state is not None else []
        clarification_history = list(state.clarification_history) if state is not None else []
        return AgentRunManifest(
            run_id=run_id,
            question=question,
            rewritten_question=rewritten_question or question,
            status="failed",
            clarification_questions=[],
            assumptions=list(assumptions),
            planner_actions=planner_actions,
            plan_dags=[state.last_plan] if state is not None and state.last_plan else [],
            tool_calls=self._current_tool_calls(run_id),
            selected_docs=[],
            node_records=[],
            provenance_records=[],
            evidence_table=[],
            final_answer=FinalAnswerPayload(
                answer_text="The run failed before a grounded answer could be completed.",
                unsupported_parts=[str(error)],
                caveats=["Review the runtime failure details and LLM traces in the manifest."],
            ),
            artifacts_dir=str(artifacts_dir),
            failures=[failure],
            metadata={
                "clarification_history": clarification_history,
                "llm_traces": llm_traces,
                "runtime_info": self.runtime_info(),
                "runtime_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
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
        artifacts_dir = (self.config.outputs_root / run_id).resolve()
        (artifacts_dir / "nodes").mkdir(parents=True, exist_ok=True)
        self._set_live_status(
            run_id,
            question=question,
            status="running",
            current_phase="initializing",
            detail="Preparing run state",
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
                    "llm_traces": list(state.llm_traces),
                    "runtime_info": self.runtime_info(),
                },
            )
            self._persist_manifest(manifest)
            return manifest

        state.rewritten_question = rephrase_action.rewritten_question or question

        if rephrase_action.action == "ask_clarification" and not force_answer:
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
        if plan_action.action == "ask_clarification" and not force_answer:
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
                "planner_calls_used": state.planner_calls_used,
                "clarification_history": list(state.clarification_history),
                "llm_traces": list(state.llm_traces),
                "runtime_info": self.runtime_info(),
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
    ) -> AgentRunManifest:
        run_id = f"agent_{uuid.uuid4().hex[:12]}"
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
            assumptions=list(manifest.assumptions),
            planner_actions=list(manifest.planner_actions),
            plan_dags=list(manifest.plan_dags),
            tool_calls=list(manifest.tool_calls),
            llm_traces=list(manifest.metadata.get("llm_traces", [])),
            clarification_questions=list(manifest.clarification_questions),
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
        return self._set_live_status(
            run_id,
            status="aborting",
            current_phase="aborting",
            detail="Abort requested; waiting for current step to stop",
        ).to_dict()

    def abort_all_runs(self) -> dict[str, Any]:
        aborted_run_ids: list[str] = []
        with self._run_lock:
            run_ids = list(self._live_runs.keys())
        for run_id in run_ids:
            with self._run_lock:
                status = self._live_runs.get(run_id)
                current_status = str(status.status) if status is not None else ""
            if current_status and current_status not in TERMINAL_RUN_STATUSES and current_status != "aborting":
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
