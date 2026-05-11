from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import time
import traceback as _tb
from typing import Any

from .agent_capabilities import AgentExecutionContext
from .agent_models import (
    AgentFailure,
    AgentNodeExecutionRecord,
    AgentPlanDAG,
    AgentPlanNode,
    short_failure_message as _short_failure_message,
)
from .io_utils import write_json
from .provenance import make_provenance_record
from . import recovery_advisor as _recovery_advisor
from .retry import RetryPolicy
from .tool_registry import ToolExecutionResult, ToolRegistry


def _safe_json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


_TIMEOUT_HINTS = ("timeout", "timed out", "deadline")
_NETWORK_HINTS = ("connection", "connect", "dns", "resolve", "network", "ssl", "tls", "unreachable")
_LLM_HINTS = ("openai", "api key", "rate limit", "model", "completion", "chat", "llm")
_DATA_HINTS = ("empty", "no rows", "no data", "missing field", "key error", "no_data")


def _classify_exception(exc: BaseException) -> str:
    cls_name = exc.__class__.__name__.lower()
    text = (str(exc) or "").lower()
    blob = f"{cls_name} {text}"
    if any(hint in blob for hint in _TIMEOUT_HINTS):
        return "timeout"
    if "keyerror" in cls_name or "missing field" in blob:
        return "missing_input"
    if any(hint in blob for hint in _LLM_HINTS):
        return "llm_error"
    if any(hint in blob for hint in _NETWORK_HINTS):
        return "network_error"
    if any(hint in blob for hint in _DATA_HINTS):
        return "data_empty"
    return "tool_error"


def _truncate_for_snapshot(value: Any, *, max_chars: int = 2000) -> Any:
    try:
        safe = _safe_json(value)
    except Exception:
        safe = str(value)
    text = json.dumps(safe, default=str)
    if len(text) <= max_chars:
        return safe
    return {"_truncated": True, "_preview": text[:max_chars] + "..."}


# Capabilities where falling back to the "heuristic" provider means the
# intended NLP path failed and a token-counting shortcut was used. For
# these the node must NOT be reported as a clean "completed" because the
# thesis claim about real NLP would be invalid. Capabilities that are
# heuristic-by-design (entity_link, claim_strength_score, ...) stay out
# of this set: heuristic IS the intended provider there.
_HEURISTIC_FALLBACK_DEGRADES_CAPABILITY: frozenset[str] = frozenset(
    {
        "ner",
        "sentiment",
        "topic_model",
        "framing_series",
        "extract_svo_triples",
        "stance_detect",
        "discourse_parse",
        "coref_resolve",
        "keyphrase_extract",
        "readability_stats",
        "lemmatize",
        # Plumbing capabilities whose empty output silently breaks every
        # downstream thesis-core tool. Reporting them as "completed" with 0
        # rows is the exact "fake completion" pattern this list exists to
        # prevent.
        "sentence_split",
        "doc_embeddings",
        "similarity_index",
        "claim_span_extract",
    }
)


def _node_status_from_result(capability: str, result: ToolExecutionResult) -> str:
    metadata = result.metadata or {}
    if metadata.get("degraded") is True:
        return "degraded"
    provider = str(metadata.get("provider", "")).strip().lower()
    if capability in _HEURISTIC_FALLBACK_DEGRADES_CAPABILITY:
        if provider == "heuristic":
            return "degraded"
        if metadata.get("provider_fallback_reason"):
            return "degraded"
        if metadata.get("no_data") is True:
            # NLP tool ran on empty input — not real work, not a clean
            # completion. Avoids 'fake completion' on capabilities the
            # thesis claims about.
            return "degraded"
        payload = result.payload
        if isinstance(payload, dict):
            for key in ("rows", "documents", "results"):
                value = payload.get(key)
                if isinstance(value, list) and len(value) == 0:
                    return "degraded"
    return "completed"


def _build_degraded_recovery_result(
    *,
    reason: str,
    capability: str,
    upstream_rows: list[dict[str, Any]],
) -> ToolExecutionResult:
    return ToolExecutionResult(
        payload={
            "rows": upstream_rows,
            "degraded": True,
            "recovery_reason": reason,
        },
        caveats=[f"Recovery advisor marked {capability} degraded: {reason}"],
        metadata={
            "degraded": True,
            "recovery": "mark_degraded",
            "reason": reason,
            "provider": "recovery_advisor",
            "tool_name": "mark_degraded",
            "no_data": not upstream_rows,
        },
    )


def _result_items(payload: Any) -> tuple[str, list[Any]]:
    if not isinstance(payload, dict):
        return "", []
    for key in ("results", "documents", "rows", "evidence_items", "artifacts"):
        value = payload.get(key)
        if isinstance(value, list):
            return key, list(value)
    return "", []


def _dependency_document_count(dependency_results: dict[str, ToolExecutionResult]) -> int:
    doc_ids: set[str] = set()
    for result in dependency_results.values():
        payload = result.payload
        if not isinstance(payload, dict):
            continue
        working_set_doc_ids = payload.get("working_set_doc_ids")
        if isinstance(working_set_doc_ids, list):
            doc_ids.update(str(item) for item in working_set_doc_ids if str(item).strip())
        for key in ("documents", "results", "rows"):
            value = payload.get(key)
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict) and str(item.get("doc_id", "")).strip():
                    doc_ids.add(str(item["doc_id"]))
    return len(doc_ids)


def _result_document_count(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    document_count = payload.get("document_count")
    if isinstance(document_count, int) and document_count >= 0:
        return document_count
    doc_ids: set[str] = set()
    working_set_doc_ids = payload.get("working_set_doc_ids")
    if isinstance(working_set_doc_ids, list):
        doc_ids.update(str(item) for item in working_set_doc_ids if str(item).strip())
    for key in ("documents", "results", "rows"):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict) and str(item.get("doc_id", "")).strip():
                doc_ids.add(str(item["doc_id"]))
    return len(doc_ids)


def _payload_preview(payload: Any) -> Any:
    if isinstance(payload, dict):
        preview: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                preview[key] = _safe_json(value[:2])
            elif isinstance(value, dict):
                preview[key] = {nested_key: _safe_json(nested_value) for nested_key, nested_value in list(value.items())[:4]}
            elif isinstance(value, str):
                preview[key] = value[:280]
            else:
                preview[key] = _safe_json(value)
        return preview
    if isinstance(payload, list):
        return _safe_json(payload[:2])
    return _safe_json(payload)


def _storage_preview_limit() -> int:
    raw = os.getenv("CORPUSAGENT2_STORED_PAYLOAD_PREVIEW_ROWS", "50").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 50


def _compact_payload_for_storage(payload: Any) -> Any:
    preview_limit = _storage_preview_limit()
    if isinstance(payload, dict):
        compact: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, list) and key in {"results", "documents", "rows", "evidence_items", "artifacts", "working_set_doc_ids"}:
                compact[key] = _safe_json(value[:preview_limit])
                if len(value) > preview_limit:
                    compact[f"{key}_truncated"] = True
                    compact[f"{key}_count"] = len(value)
                continue
            compact[key] = _compact_payload_for_storage(value) if isinstance(value, (dict, list)) else _safe_json(value)
        return compact
    if isinstance(payload, list):
        compact_list = [_compact_payload_for_storage(item) for item in payload[:preview_limit]]
        if len(payload) > preview_limit:
            return {"items": compact_list, "items_truncated": True, "items_count": len(payload)}
        return compact_list
    return _safe_json(payload)


def _summarize_tool_result(result: ToolExecutionResult, *, dependency_results: dict[str, ToolExecutionResult]) -> dict[str, Any]:
    payload = result.payload
    items_key, items = _result_items(payload)
    input_documents_seen = _dependency_document_count(dependency_results)
    output_documents = _result_document_count(payload)
    no_data_reason = ""
    if result.metadata.get("no_data_reason") not in (None, ""):
        no_data_reason = str(result.metadata.get("no_data_reason", "")).strip()
    elif result.caveats:
        no_data_reason = str(result.caveats[0]).strip()
    no_data = bool(result.metadata.get("no_data")) or (
        isinstance(payload, dict) and items_key in {"documents", "results", "rows"} and len(items) == 0
    )
    summary = {
        "items_key": items_key,
        "items_count": len(items),
        "documents_processed": input_documents_seen,
        "input_documents_seen": input_documents_seen,
        "output_documents": output_documents,
        "evidence_count": len(result.evidence_items),
        "artifact_count": len(result.artifacts_used),
        "caveat_count": len(result.caveats),
        "unsupported_count": len(result.unsupported_parts),
        "no_data": no_data,
        "payload_preview": _payload_preview(payload),
    }
    if no_data_reason:
        summary["no_data_reason"] = no_data_reason
    if isinstance(payload, dict):
        stdout = str(payload.get("stdout", "")).strip()
        stderr = str(payload.get("stderr", "")).strip()
        if stdout:
            summary["stdout_preview"] = stdout[:280]
        if stderr:
            summary["stderr_preview"] = stderr[:280]
        if payload.get("exit_code") is not None:
            summary["exit_code"] = int(payload.get("exit_code", 0))
    return summary


_CORE_NO_DATA_FAILURE_CAPABILITIES = {
    "db_search",
    "join_external_series",
    "plot_artifact",
    "python_runner",
    "sql_query_search",
    "time_series_aggregate",
}

# Capabilities where an empty retrieval result means there is nothing for
# the entire downstream plan to work on. The orchestrator uses this to
# bail out early instead of grinding 20 NLP nodes against zero rows.
_RETRIEVAL_BAILOUT_CAPABILITIES = {"db_search", "sql_query_search"}


def _result_no_data_reason(result: ToolExecutionResult) -> str:
    if result.metadata.get("no_data_reason") not in (None, ""):
        return str(result.metadata.get("no_data_reason", "")).strip()
    if result.caveats:
        return str(result.caveats[0]).strip()
    payload = result.payload
    if isinstance(payload, dict):
        items_key, items = _result_items(payload)
        if items_key in {"documents", "results", "rows"} and not items:
            return f"empty {items_key}"
    return "no data returned"


def _result_has_no_data(result: ToolExecutionResult) -> bool:
    if bool(result.metadata.get("no_data")):
        return True
    payload = result.payload
    if not isinstance(payload, dict):
        return False
    items_key, items = _result_items(payload)
    return items_key in {"documents", "results", "rows"} and len(items) == 0


def _dependency_results_have_plot_rows(dependency_results: dict[str, ToolExecutionResult]) -> bool:
    if not dependency_results:
        return True
    for result in dependency_results.values():
        items_key, items = _result_items(result.payload)
        if items_key in {"rows", "results", "documents"} and items:
            return True
    return False


def _node_record_lookup(records: list[AgentNodeExecutionRecord]) -> dict[str, AgentNodeExecutionRecord]:
    return {record.node_id: record for record in records}


def _node_record_brief(record: AgentNodeExecutionRecord) -> str:
    tool = f" via {record.tool_name}" if record.tool_name else ""
    reason = str(record.error or "").strip()
    if not reason and record.caveats:
        reason = str(record.caveats[0]).strip()
    if not reason:
        reason = record.status
    return f"{record.node_id} ({record.capability}{tool}): {reason}"


def _transitive_dependents(failed_ids: set[str], node_map: dict[str, Any]) -> set[str]:
    """Return every node that transitively depends on any node in failed_ids.

    Used to skip only the unreachable subtree when a node fails, rather than
    halting every remaining pending node. Independent DAG branches must keep
    executing so a single localized failure (e.g. doc_embeddings hitting a
    PyTorch meta-tensor bug) doesn't take out sentiment/topic/keyterms paths
    that have no relationship to it.
    """
    blocked = set(failed_ids)
    changed = True
    while changed:
        changed = False
        for nid, node in node_map.items():
            if nid in blocked:
                continue
            if any(dep in blocked for dep in node.depends_on):
                blocked.add(nid)
                changed = True
    return blocked - set(failed_ids)


def _dependency_skip_reason(
    *,
    prefix: str,
    dependency_ids: list[str],
    node_records: list[AgentNodeExecutionRecord],
) -> str:
    records_by_id = _node_record_lookup(node_records)
    details = [
        _node_record_brief(records_by_id[dependency_id])
        for dependency_id in dependency_ids
        if dependency_id in records_by_id
    ]
    if details:
        return f"{prefix}: {'; '.join(details)}"
    return f"{prefix}: {', '.join(dependency_ids)}"


def _matches_schema_type(value: Any, schema_type: str) -> bool:
    normalized = str(schema_type or "").strip().lower()
    if normalized in {"", "any", "object"}:
        return True
    if normalized in {"dict", "mapping", "object"}:
        return isinstance(value, dict)
    if normalized in {"list", "array"}:
        return isinstance(value, list)
    if normalized in {"str", "string"}:
        return isinstance(value, str)
    if normalized in {"int", "integer"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if normalized in {"float", "number"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if normalized in {"bool", "boolean"}:
        return isinstance(value, bool)
    return True


def _validate_tool_input_schema(params: dict[str, Any], schema_fields: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for field_name, schema_type in schema_fields.items():
        normalized_type = str(schema_type or "").strip()
        required = normalized_type.endswith("!") or normalized_type.lower().startswith("required ")
        expected_type = normalized_type.rstrip("!")
        if expected_type.lower().startswith("required "):
            expected_type = expected_type[len("required "):]
        expected_type = expected_type.strip()
        if field_name not in params:
            if required:
                errors.append(f"missing required field '{field_name}'")
            continue
        value = params[field_name]
        if not _matches_schema_type(value, expected_type):
            errors.append(f"field '{field_name}' expected {expected_type or 'Any'}, got {type(value).__name__}")
    return errors


@dataclass(slots=True)
class AgentExecutionSnapshot:
    node_records: list[AgentNodeExecutionRecord]
    node_results: dict[str, ToolExecutionResult]
    failures: list[AgentFailure]
    provenance_records: list[dict[str, Any]]
    selected_docs: list[dict[str, Any]]
    status: str


class AsyncPlanExecutor:
    def __init__(self, registry: ToolRegistry, cache: dict[str, ToolExecutionResult] | None = None) -> None:
        self.registry = registry
        self._cache = cache if cache is not None else {}

    def _cache_key(
        self,
        node: AgentPlanNode,
        resolution_tool_name: str,
        dependency_results: dict[str, ToolExecutionResult],
        context: AgentExecutionContext,
    ) -> str:
        dependency_fingerprint = {
            key: sha256(json.dumps(_safe_json(value.payload), sort_keys=True).encode("utf-8")).hexdigest()
            for key, value in sorted(dependency_results.items())
        }
        implicit_query_context = ""
        node_inputs = dict(node.inputs)
        payload_inputs = node_inputs.get("payload") if isinstance(node_inputs.get("payload"), dict) else {}
        has_explicit_query = bool(
            str(node_inputs.get("query", "")).strip()
            or str(payload_inputs.get("query", "")).strip()
        )
        if node.capability in {"db_search", "sql_query_search"} and not has_explicit_query:
            implicit_query_context = str(
                getattr(context.state, "rewritten_question", "")
                or getattr(context.state, "question", "")
            ).strip()
        payload = {
            "capability": node.capability,
            "tool_name": resolution_tool_name,
            "inputs": _safe_json(node.inputs),
            "dependencies": dependency_fingerprint,
        }
        if implicit_query_context:
            payload["implicit_query_context"] = implicit_query_context
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return sha256(encoded).hexdigest()

    def _write_node_artifact(self, artifacts_dir: Path, node: AgentPlanNode, result: ToolExecutionResult) -> str:
        target = artifacts_dir / "nodes" / f"{node.node_id}.json"
        write_json(
            target,
            {
                "payload": _compact_payload_for_storage(result.payload),
                "metadata": _safe_json(result.metadata),
                "artifacts": list(result.artifacts_used),
                "caveats": list(result.caveats),
                "unsupported_parts": list(result.unsupported_parts),
            },
        )
        return str(target)

    def _write_failure_artifact(
        self,
        artifacts_dir: Path,
        node: AgentPlanNode,
        failure: AgentFailure,
        *,
        status: str,
        started_at_utc: str,
        finished_at_utc: str,
        duration_ms: float,
    ) -> str:
        target = artifacts_dir / "nodes" / f"{node.node_id}.json"
        write_json(
            target,
            {
                "status": status,
                "node_id": node.node_id,
                "capability": node.capability,
                "tool_name": node.tool_name,
                "started_at_utc": started_at_utc,
                "finished_at_utc": finished_at_utc,
                "duration_ms": duration_ms,
                "failure": failure.to_dict(),
            },
        )
        return str(target)

    def _write_run_state(
        self,
        artifacts_dir: Path,
        node_records: list[AgentNodeExecutionRecord],
        failures: list[AgentFailure],
        *,
        plan_dag: AgentPlanDAG | None = None,
    ) -> None:
        """Write a single-file progress snapshot to artifacts_dir/run_state.json.

        Updated after every node so a killed run leaves a self-describing
        breadcrumb: which nodes finished, which are still pending, which
        failed, and the most recent timestamp.
        """
        total = len(plan_dag.nodes) if plan_dag is not None else len({r.node_id for r in node_records})
        finished_ids = {r.node_id for r in node_records}
        by_status: dict[str, list[str]] = {}
        for record in node_records:
            by_status.setdefault(record.status, []).append(record.node_id)
        pending_ids: list[str] = []
        if plan_dag is not None:
            pending_ids = [n.node_id for n in plan_dag.nodes if n.node_id not in finished_ids]
        try:
            write_json(
                artifacts_dir / "run_state.json",
                {
                    "updated_at_utc": _utc_now(),
                    "total_nodes": total,
                    "finished_nodes": len(finished_ids),
                    "pending_node_ids": pending_ids,
                    "completed_node_ids": by_status.get("completed", []),
                    "skipped_node_ids": by_status.get("skipped", []),
                    "failed_node_ids": by_status.get("failed", []),
                    "failure_count": len(failures),
                },
            )
        except Exception:
            # Never let state writes break the run loop.
            pass

    def _emit_event(self, context: AgentExecutionContext, payload: dict[str, Any]) -> None:
        if context.event_callback is None:
            return
        context.event_callback(payload)

    def _is_cancelled(self, context: AgentExecutionContext) -> bool:
        return bool(context.cancel_requested is not None and context.cancel_requested())

    async def _execute_node(
        self,
        node: AgentPlanNode,
        dependency_results: dict[str, ToolExecutionResult],
        context: AgentExecutionContext,
    ) -> tuple[AgentNodeExecutionRecord, ToolExecutionResult | None, AgentFailure | None, dict[str, Any] | None]:
        started_at = _utc_now()
        started_perf = time.perf_counter()
        attempts = 0
        last_error = ""
        last_traceback = ""
        last_exception_type = ""
        last_category = "unknown"

        def cancelled_response(
            reason: str,
            *,
            tool_name: str | None = None,
            provider: str | None = None,
        ) -> tuple[AgentNodeExecutionRecord, None, None, None]:
            finished_at = _utc_now()
            duration_ms = (time.perf_counter() - started_perf) * 1000.0
            return (
                AgentNodeExecutionRecord(
                    node_id=node.node_id,
                    capability=node.capability,
                    status="skipped",
                    tool_name=tool_name or node.tool_name,
                    provider=provider or "",
                    started_at_utc=started_at,
                    finished_at_utc=finished_at,
                    duration_ms=duration_ms,
                    attempts=attempts,
                    error=reason,
                ),
                None,
                None,
                None,
            )

        if self._is_cancelled(context):
            return cancelled_response("Skipped because run abort was requested before this node started.")

        if node.capability == "plot_artifact" and not _dependency_results_have_plot_rows(dependency_results):
            finished_at = _utc_now()
            duration_ms = (time.perf_counter() - started_perf) * 1000.0
            message = "Plot input dependencies did not produce rows; plot was not executed."
            status = "skipped" if node.optional else "failed"
            context.working_store.record_tool_call(
                context.run_id,
                node.node_id,
                node.capability,
                node.tool_name or "plot_artifact",
                status,
                {
                    "error": message,
                    "started_at_utc": started_at,
                    "finished_at_utc": finished_at,
                    "inputs": _safe_json(node.inputs),
                    "dependency_nodes": sorted(dependency_results.keys()),
                },
            )
            self._emit_event(
                context,
                {
                    "event": "node_failed",
                    "node_id": node.node_id,
                    "capability": node.capability,
                    "status": status,
                    "error": message,
                    "short_message": _short_failure_message("data_empty", message),
                    "inputs": _safe_json(node.inputs),
                    "dependency_nodes": sorted(dependency_results.keys()),
                    "started_at_utc": started_at,
                    "finished_at_utc": finished_at,
                    "duration_ms": duration_ms,
                },
            )
            return (
                AgentNodeExecutionRecord(
                    node_id=node.node_id,
                    capability=node.capability,
                    status=status,
                    tool_name=node.tool_name or "plot_artifact",
                    started_at_utc=started_at,
                    finished_at_utc=finished_at,
                    duration_ms=duration_ms,
                    attempts=0,
                    error=message,
                ),
                None,
                None
                if node.optional
                else AgentFailure(
                    node_id=node.node_id,
                    capability=node.capability,
                    error_type="plot_input_no_data",
                    message=message,
                    retriable=False,
                ),
                None,
            )

        try:
            resolution = self.registry.resolve(
                capability=node.capability,
                context=context,
                params=node.inputs,
                requested_tool_name=node.tool_name,
            )
        except Exception as exc:
            finished_at = _utc_now()
            duration_ms = (time.perf_counter() - started_perf) * 1000.0
            tb_text = _tb.format_exc()
            category = _classify_exception(exc)
            self._emit_event(
                context,
                {
                    "event": "node_failed",
                    "node_id": node.node_id,
                    "capability": node.capability,
                    "status": "failed",
                    "error": str(exc),
                    "short_message": _short_failure_message(category, str(exc)),
                    "error_category": category,
                    "started_at_utc": started_at,
                    "finished_at_utc": finished_at,
                    "duration_ms": duration_ms,
                },
            )
            failure = AgentFailure(
                node_id=node.node_id,
                capability=node.capability,
                error_type=exc.__class__.__name__,
                message=str(exc),
                retriable=False,
                traceback=tb_text,
                input_snapshot=_truncate_for_snapshot(dict(node.inputs)),
                retry_count=0,
                category=category,
            )
            try:
                self._write_failure_artifact(
                    context.artifacts_dir,
                    node,
                    failure,
                    status="failed",
                    started_at_utc=started_at,
                    finished_at_utc=finished_at,
                    duration_ms=duration_ms,
                )
            except Exception:
                pass
            return (
                AgentNodeExecutionRecord(
                    node_id=node.node_id,
                    capability=node.capability,
                    status="failed",
                    tool_name=node.tool_name,
                    started_at_utc=started_at,
                    finished_at_utc=finished_at,
                    duration_ms=duration_ms,
                    attempts=1,
                    error=str(exc),
                ),
                None,
                failure,
                None,
            )

        if self._is_cancelled(context):
            return cancelled_response(
                "Skipped because run abort was requested after tool resolution.",
                tool_name=resolution.spec.tool_name,
                provider=resolution.spec.provider,
            )

        schema_errors = _validate_tool_input_schema(dict(node.inputs), dict(resolution.spec.input_schema.fields))
        if schema_errors:
            finished_at = _utc_now()
            duration_ms = (time.perf_counter() - started_perf) * 1000.0
            message = "; ".join(schema_errors)
            status = "skipped" if node.optional else "failed"
            context.working_store.record_tool_call(
                context.run_id,
                node.node_id,
                node.capability,
                resolution.spec.tool_name,
                status,
                {
                    "error": message,
                    "started_at_utc": started_at,
                    "finished_at_utc": finished_at,
                    "inputs": _safe_json(node.inputs),
                    "input_schema": _safe_json(resolution.spec.input_schema.to_dict()),
                },
            )
            self._emit_event(
                context,
                {
                    "event": "node_failed",
                    "node_id": node.node_id,
                    "capability": node.capability,
                    "status": status,
                    "error": message,
                    "short_message": _short_failure_message("missing_input", message),
                    "tool_name": resolution.spec.tool_name,
                    "provider": resolution.spec.provider,
                    "inputs": _safe_json(node.inputs),
                    "started_at_utc": started_at,
                    "finished_at_utc": finished_at,
                    "duration_ms": duration_ms,
                },
            )
            return (
                AgentNodeExecutionRecord(
                    node_id=node.node_id,
                    capability=node.capability,
                    status=status,
                    tool_name=node.tool_name or resolution.spec.tool_name,
                    provider=resolution.spec.provider,
                    started_at_utc=started_at,
                    finished_at_utc=finished_at,
                    duration_ms=duration_ms,
                    attempts=0,
                    error=message,
                ),
                None,
                AgentFailure(
                    node_id=node.node_id,
                    capability=node.capability,
                    error_type="input_schema_validation_failed",
                    message=message,
                    retriable=True,
                ),
                None,
            )

        if self._is_cancelled(context):
            return cancelled_response(
                "Skipped because run abort was requested before this tool started.",
                tool_name=resolution.spec.tool_name,
                provider=resolution.spec.provider,
            )

        cache_key = self._cache_key(node, resolution.spec.tool_name, dependency_results, context)
        start_event_payload = {
            "event": "node_started",
            "node_id": node.node_id,
            "capability": node.capability,
            "status": "running",
            "started_at_utc": started_at,
            "tool_name": resolution.spec.tool_name,
            "provider": resolution.spec.provider,
            "tool_version": resolution.spec.tool_version,
            "model_id": resolution.spec.model_id,
            "tool_reason": resolution.reason,
            "inputs": _safe_json(node.inputs),
            "dependency_nodes": sorted(dependency_results.keys()),
            "documents_processed": _dependency_document_count(dependency_results),
            "cache_key": cache_key,
        }
        if node.tool_name:
            start_event_payload["requested_tool_name"] = node.tool_name
        self._emit_event(context, start_event_payload)
        cache_hit = bool(not getattr(context.state, "no_cache", False) and node.cacheable and cache_key in self._cache)
        if cache_hit:
            context.working_store.record_tool_call(
                context.run_id,
                node.node_id,
                node.capability,
                resolution.spec.tool_name,
                "running",
                {
                    "cache_lookup": True,
                    "started_at_utc": started_at,
                    "inputs": _safe_json(node.inputs),
                    "dependency_nodes": sorted(dependency_results.keys()),
                    "cache_key": cache_key,
                },
            )
            result = self._cache[cache_key]
            resolved_tool_name = str(result.metadata.get("tool_name", resolution.spec.tool_name))
            resolved_provider = str(result.metadata.get("provider", resolution.spec.provider))
            resolved_tool_version = str(result.metadata.get("tool_version", resolution.spec.tool_version))
            resolved_model_id = str(result.metadata.get("model_id", resolution.spec.model_id))
            node_status_cached = _node_status_from_result(node.capability, result)
            artifact_path = self._write_node_artifact(context.artifacts_dir, node, result)
            context.working_store.record_tool_call(
                context.run_id,
                node.node_id,
                node.capability,
                resolved_tool_name,
                node_status_cached,
                {
                    "cache_hit": True,
                    "started_at_utc": started_at,
                    "finished_at_utc": _utc_now(),
                    "inputs": _safe_json(node.inputs),
                    "dependency_nodes": sorted(dependency_results.keys()),
                    "cache_key": cache_key,
                    "summary": _summarize_tool_result(result, dependency_results=dependency_results),
                    "payload": _compact_payload_for_storage(result.payload),
                },
            )
            context.working_store.record_artifact(
                context.run_id,
                node.node_id,
                artifact_path,
                {"capability": node.capability},
            )
            provenance = make_provenance_record(
                run_id=context.run_id,
                tool_name=resolved_tool_name,
                tool_version=resolved_tool_version,
                model_id=resolved_model_id or resolved_provider,
                params=dict(node.inputs),
                inputs_ref={"node_id": node.node_id, "dependency_nodes": sorted(dependency_results.keys())},
                outputs_ref={"artifact_path": artifact_path},
                evidence=list(result.evidence_items),
            ).to_dict()
            finished_at = _utc_now()
            duration_ms = (time.perf_counter() - started_perf) * 1000.0
            self._emit_event(
                context,
                {
                    "event": "node_completed",
                    "node_id": node.node_id,
                    "capability": node.capability,
                    "status": node_status_cached,
                    "degraded": node_status_cached == "degraded",
                    "cache_hit": True,
                    "tool_name": resolved_tool_name,
                    "provider": resolved_provider,
                    "tool_version": resolved_tool_version,
                    "model_id": resolved_model_id,
                    "tool_reason": resolution.reason,
                    "inputs": _safe_json(node.inputs),
                    "dependency_nodes": sorted(dependency_results.keys()),
                    "documents_processed": _dependency_document_count(dependency_results),
                    "cache_key": cache_key,
                    "summary": _summarize_tool_result(result, dependency_results=dependency_results),
                    "artifacts": list(result.artifacts_used) + [artifact_path],
                    "started_at_utc": started_at,
                    "finished_at_utc": finished_at,
                    "duration_ms": duration_ms,
                    "requested_tool_name": node.tool_name,
                },
            )
            return (
                AgentNodeExecutionRecord(
                    node_id=node.node_id,
                    capability=node.capability,
                    status=node_status_cached,
                    started_at_utc=started_at,
                    finished_at_utc=finished_at,
                    duration_ms=duration_ms,
                    attempts=1,
                    cache_key=cache_key,
                    cache_hit=True,
                    tool_name=resolved_tool_name,
                    provider=resolved_provider,
                    tool_version=resolved_tool_version,
                    model_id=resolved_model_id,
                    tool_reason=resolution.reason,
                    artifacts_used=list(result.artifacts_used) + [artifact_path],
                    evidence_count=len(result.evidence_items),
                    caveats=list(result.caveats),
                    unsupported_parts=list(result.unsupported_parts),
                ),
                result,
                None,
                provenance,
            )

        retry_policy = RetryPolicy.from_env()
        recovery_max_attempts = _recovery_advisor.max_attempts() if _recovery_advisor.is_enabled() else 0
        recovery_attempts_used = 0
        recovery_actions_log: list[dict[str, Any]] = []
        recovery_advisor_error: str = ""

        while True:
            # Reset retry state at the start of each recovery iteration so
            # the next attempt is a clean run with potentially modified
            # inputs from the advisor.
            attempts = 0
            last_error = ""
            last_traceback = ""
            last_exception_type = ""
            last_category = "unknown"

            while attempts < retry_policy.max_attempts:
                if self._is_cancelled(context):
                    return cancelled_response(
                        "Skipped because run abort was requested before this tool attempt started.",
                        tool_name=resolution.spec.tool_name,
                        provider=resolution.spec.provider,
                    )
                if attempts >= 1:
                    # Only retry transient errors. Permanent failures should
                    # not consume more attempts.
                    if not retry_policy.is_retriable(last_category):
                        break
                    delay_s = retry_policy.compute_delay_s(attempts + 1)
                    if delay_s > 0:
                        time.sleep(delay_s)
                attempts += 1
                try:
                    context.working_store.record_tool_call(
                        context.run_id,
                        node.node_id,
                        node.capability,
                        resolution.spec.tool_name,
                        "running",
                        {
                            "attempt": attempts,
                            "started_at_utc": started_at,
                            "inputs": _safe_json(node.inputs),
                            "dependency_nodes": sorted(dependency_results.keys()),
                            "cache_key": cache_key,
                        },
                    )
                    result = await asyncio.to_thread(
                        resolution.adapter.run,
                        dict(node.inputs),
                        dependency_results,
                        context,
                    )
                    if not getattr(context.state, "no_cache", False) and node.cacheable:
                        self._cache[cache_key] = result
                    artifact_path = self._write_node_artifact(context.artifacts_dir, node, result)
                    node_status_fresh = _node_status_from_result(node.capability, result)
                    context.working_store.record_tool_call(
                        context.run_id,
                        node.node_id,
                        node.capability,
                        str(result.metadata.get("tool_name", resolution.spec.tool_name)),
                        node_status_fresh,
                        {
                            "cache_hit": False,
                            "attempt": attempts,
                            "started_at_utc": started_at,
                            "finished_at_utc": _utc_now(),
                            "inputs": _safe_json(node.inputs),
                            "dependency_nodes": sorted(dependency_results.keys()),
                            "cache_key": cache_key,
                            "summary": _summarize_tool_result(result, dependency_results=dependency_results),
                            "payload": _compact_payload_for_storage(result.payload),
                        },
                    )
                    context.working_store.record_artifact(
                        context.run_id,
                        node.node_id,
                        artifact_path,
                        {"capability": node.capability},
                    )
                    resolved_tool_name = str(result.metadata.get("tool_name", resolution.spec.tool_name))
                    resolved_provider = str(result.metadata.get("provider", resolution.spec.provider))
                    resolved_tool_version = str(result.metadata.get("tool_version", resolution.spec.tool_version))
                    resolved_model_id = str(result.metadata.get("model_id", resolution.spec.model_id))
                    provenance = make_provenance_record(
                        run_id=context.run_id,
                        tool_name=resolved_tool_name,
                        tool_version=resolved_tool_version,
                        model_id=resolved_model_id or resolved_provider,
                        params=dict(node.inputs),
                        inputs_ref={"node_id": node.node_id, "dependency_nodes": sorted(dependency_results.keys())},
                        outputs_ref={"artifact_path": artifact_path},
                        evidence=list(result.evidence_items),
                    ).to_dict()
                    finished_at = _utc_now()
                    duration_ms = (time.perf_counter() - started_perf) * 1000.0
                    extra_event_fields: dict[str, Any] = {}
                    if recovery_actions_log:
                        extra_event_fields["recovery_actions"] = list(recovery_actions_log)
                    self._emit_event(
                        context,
                        {
                            "event": "node_completed",
                            "node_id": node.node_id,
                            "capability": node.capability,
                            "status": node_status_fresh,
                            "degraded": node_status_fresh == "degraded",
                            "cache_hit": False,
                            "tool_name": resolved_tool_name,
                            "provider": resolved_provider,
                            "tool_version": resolved_tool_version,
                            "model_id": resolved_model_id,
                            "tool_reason": resolution.reason,
                            "inputs": _safe_json(node.inputs),
                            "dependency_nodes": sorted(dependency_results.keys()),
                            "documents_processed": _dependency_document_count(dependency_results),
                            "cache_key": cache_key,
                            "summary": _summarize_tool_result(result, dependency_results=dependency_results),
                            "artifacts": list(result.artifacts_used) + [artifact_path],
                            "started_at_utc": started_at,
                            "finished_at_utc": finished_at,
                            "duration_ms": duration_ms,
                            "requested_tool_name": node.tool_name,
                            **extra_event_fields,
                        },
                    )
                    return (
                        AgentNodeExecutionRecord(
                            node_id=node.node_id,
                            capability=node.capability,
                            status=node_status_fresh,
                            started_at_utc=started_at,
                            finished_at_utc=finished_at,
                            duration_ms=duration_ms,
                            attempts=attempts,
                            cache_key=cache_key,
                            cache_hit=False,
                            tool_name=resolved_tool_name,
                            provider=resolved_provider,
                            tool_version=resolved_tool_version,
                            model_id=resolved_model_id,
                            tool_reason=resolution.reason,
                            artifacts_used=list(result.artifacts_used) + [artifact_path],
                            evidence_count=len(result.evidence_items),
                            caveats=list(result.caveats),
                            unsupported_parts=list(result.unsupported_parts),
                        ),
                        result,
                        None,
                        provenance,
                    )
                except Exception as exc:
                    last_error = str(exc)
                    last_traceback = _tb.format_exc()
                    last_exception_type = exc.__class__.__name__
                    last_category = _classify_exception(exc)
                    context.working_store.record_tool_call(
                        context.run_id,
                        node.node_id,
                        node.capability,
                        resolution.spec.tool_name,
                        "failed",
                        {
                            "error": last_error,
                            "error_category": last_category,
                            "exception_type": last_exception_type,
                            "attempt": attempts,
                            "started_at_utc": started_at,
                            "inputs": _safe_json(node.inputs),
                            "dependency_nodes": sorted(dependency_results.keys()),
                            "cache_key": cache_key,
                        },
                    )

            # Retry loop exhausted. Consult the recovery advisor if enabled
            # and the budget allows. The advisor can ask for a retry with
            # modified inputs (continue outer loop), accept a degraded
            # result, or fall back to fail. Plan topology is never touched.
            if (
                _recovery_advisor.is_enabled()
                and getattr(self, "_recovery_advisor_factory", None) is not None
                and recovery_attempts_used < recovery_max_attempts
            ):
                try:
                    advisor = self._recovery_advisor_factory()  # type: ignore[attr-defined]
                except Exception as exc:
                    advisor = None
                    recovery_advisor_error = f"factory error: {exc}"
                if advisor is not None:
                    try:
                        action = advisor.advise(
                            node_id=node.node_id,
                            capability=node.capability,
                            tool_name=resolution.spec.tool_name,
                            inputs=dict(node.inputs),
                            traceback=last_traceback,
                            failure_category=last_category,
                            upstream_summary={k: True for k in dependency_results},
                            candidate_capabilities=self.registry.list_capabilities_in_same_category(node.capability)
                            if hasattr(self.registry, "list_capabilities_in_same_category")
                            else [],
                        )
                    except Exception as exc:
                        recovery_advisor_error = f"advise error: {exc}"
                        action = None
                    if action is not None:
                        recovery_attempts_used += 1
                        recovery_actions_log.append(
                            {
                                "attempt": recovery_attempts_used,
                                "from_category": last_category,
                                **action.to_dict(),
                            }
                        )
                        if (
                            action.action == "retry_with_modified_inputs"
                            and isinstance(action.inputs, dict)
                            and action.inputs
                        ):
                            # Mutate in-place so the next retry uses the new
                            # inputs. PlanDAG topology is untouched.
                            node.inputs.update(action.inputs)
                            continue
                        if action.action == "mark_degraded":
                            upstream_rows: list[dict[str, Any]] = []
                            for dep_result in dependency_results.values():
                                payload = dep_result.payload
                                if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
                                    upstream_rows = [
                                        dict(item) for item in payload["rows"] if isinstance(item, dict)
                                    ]
                                    if upstream_rows:
                                        break
                            degraded_result = _build_degraded_recovery_result(
                                reason=action.reason or "advisor requested mark_degraded",
                                capability=node.capability,
                                upstream_rows=upstream_rows,
                            )
                            artifact_path = self._write_node_artifact(
                                context.artifacts_dir, node, degraded_result
                            )
                            context.working_store.record_tool_call(
                                context.run_id,
                                node.node_id,
                                node.capability,
                                "mark_degraded",
                                "degraded",
                                {
                                    "cache_hit": False,
                                    "attempt": attempts,
                                    "started_at_utc": started_at,
                                    "finished_at_utc": _utc_now(),
                                    "inputs": _safe_json(node.inputs),
                                    "dependency_nodes": sorted(dependency_results.keys()),
                                    "cache_key": cache_key,
                                    "recovery_action": "mark_degraded",
                                    "recovery_reason": action.reason,
                                },
                            )
                            context.working_store.record_artifact(
                                context.run_id,
                                node.node_id,
                                artifact_path,
                                {"capability": node.capability, "degraded": True},
                            )
                            finished_at = _utc_now()
                            duration_ms = (time.perf_counter() - started_perf) * 1000.0
                            self._emit_event(
                                context,
                                {
                                    "event": "node_completed",
                                    "node_id": node.node_id,
                                    "capability": node.capability,
                                    "status": "degraded",
                                    "degraded": True,
                                    "cache_hit": False,
                                    "tool_name": "mark_degraded",
                                    "provider": "recovery_advisor",
                                    "inputs": _safe_json(node.inputs),
                                    "dependency_nodes": sorted(dependency_results.keys()),
                                    "documents_processed": _dependency_document_count(dependency_results),
                                    "cache_key": cache_key,
                                    "artifacts": [artifact_path],
                                    "started_at_utc": started_at,
                                    "finished_at_utc": finished_at,
                                    "duration_ms": duration_ms,
                                    "recovery_actions": list(recovery_actions_log),
                                    "recovery_reason": action.reason,
                                },
                            )
                            return (
                                AgentNodeExecutionRecord(
                                    node_id=node.node_id,
                                    capability=node.capability,
                                    status="degraded",
                                    started_at_utc=started_at,
                                    finished_at_utc=finished_at,
                                    duration_ms=duration_ms,
                                    attempts=attempts,
                                    cache_key=cache_key,
                                    cache_hit=False,
                                    tool_name="mark_degraded",
                                    provider="recovery_advisor",
                                    tool_reason=action.reason,
                                    artifacts_used=[artifact_path],
                                    evidence_count=0,
                                    caveats=list(degraded_result.caveats),
                                    unsupported_parts=list(degraded_result.unsupported_parts),
                                ),
                                degraded_result,
                                None,
                                None,
                            )
                        # substitute_capability and fail: log and exit recovery.
                        break
            break

        finished_at = _utc_now()
        duration_ms = (time.perf_counter() - started_perf) * 1000.0
        short_message = _short_failure_message(last_category, last_error)
        self._emit_event(
            context,
            {
                "event": "node_failed",
                "node_id": node.node_id,
                "capability": node.capability,
                "status": "failed",
                "error": last_error,
                "short_message": short_message,
                "tool_name": resolution.spec.tool_name,
                "provider": resolution.spec.provider,
                "tool_version": resolution.spec.tool_version,
                "model_id": resolution.spec.model_id,
                "tool_reason": resolution.reason,
                "inputs": _safe_json(node.inputs),
                "dependency_nodes": sorted(dependency_results.keys()),
                "documents_processed": _dependency_document_count(dependency_results),
                "cache_key": cache_key,
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "duration_ms": duration_ms,
                "requested_tool_name": node.tool_name,
                "recovery_actions": list(recovery_actions_log),
            },
        )
        failure = AgentFailure(
            node_id=node.node_id,
            capability=node.capability,
            error_type=last_exception_type or "execution_failed",
            message=last_error,
            retriable=node.optional,
            traceback=last_traceback,
            input_snapshot=_truncate_for_snapshot(dict(node.inputs)),
            retry_count=attempts,
            category=last_category,
        )
        status = "skipped" if node.optional else "failed"
        if recovery_actions_log:
            failure.details["recovery_actions"] = list(recovery_actions_log)
        if recovery_advisor_error:
            failure.details["recovery_advisor_error"] = recovery_advisor_error
        try:
            self._write_failure_artifact(
                context.artifacts_dir,
                node,
                failure,
                status=status,
                started_at_utc=started_at,
                finished_at_utc=finished_at,
                duration_ms=duration_ms,
            )
        except Exception:
            pass
        return (
            AgentNodeExecutionRecord(
                node_id=node.node_id,
                capability=node.capability,
                status=status,
                tool_name=node.tool_name or resolution.spec.tool_name,
                started_at_utc=started_at,
                finished_at_utc=finished_at,
                duration_ms=duration_ms,
                attempts=attempts,
                error=last_error,
            ),
            None,
            failure,
            None,
        )

    async def execute(self, plan_dag: AgentPlanDAG, context: AgentExecutionContext) -> AgentExecutionSnapshot:
        node_map = plan_dag.node_map()
        pending = {node.node_id for node in plan_dag.nodes}
        completed: dict[str, ToolExecutionResult] = {}
        node_records: list[AgentNodeExecutionRecord] = []
        failures: list[AgentFailure] = []
        provenance_rows: list[dict[str, Any]] = []
        selected_docs: list[dict[str, Any]] = []

        def mark_pending_skipped(reason: str) -> None:
            now = _utc_now()
            for pending_id in sorted(pending):
                pending_node = node_map[pending_id]
                node_records.append(
                    AgentNodeExecutionRecord(
                        node_id=pending_node.node_id,
                        capability=pending_node.capability,
                        status="skipped",
                        tool_name=pending_node.tool_name,
                        started_at_utc=now,
                        finished_at_utc=now,
                        duration_ms=0.0,
                        attempts=0,
                        error=reason,
                    )
                )
            pending.clear()

        while pending:
            if self._is_cancelled(context):
                mark_pending_skipped("Skipped because run abort was requested.")
                return AgentExecutionSnapshot(
                    node_records=node_records,
                    node_results=completed,
                    failures=failures,
                    provenance_records=provenance_rows,
                    selected_docs=selected_docs,
                    status="aborted",
                )
            skipped_ids = {item.node_id for item in node_records if item.status == "skipped"}
            for node_id in list(pending):
                node = node_map[node_id]
                skipped_dependencies = [dep for dep in node.depends_on if dep in skipped_ids]
                if not skipped_dependencies:
                    continue
                pending.discard(node_id)
                message = _dependency_skip_reason(
                    prefix="Skipped because dependencies did not produce executable data",
                    dependency_ids=skipped_dependencies,
                    node_records=node_records,
                )
                now = _utc_now()
                if node.optional:
                    node_records.append(
                        AgentNodeExecutionRecord(
                            node_id=node.node_id,
                            capability=node.capability,
                            status="skipped",
                            tool_name=node.tool_name,
                            started_at_utc=now,
                            finished_at_utc=now,
                            duration_ms=0.0,
                            attempts=0,
                            error=message,
                        )
                    )
                    continue
                failure = AgentFailure(
                    node_id=node.node_id,
                    capability=node.capability,
                    error_type="dependency_no_data",
                    message=message,
                    retriable=True,
                )
                failures.append(failure)
                node_records.append(
                    AgentNodeExecutionRecord(
                        node_id=node.node_id,
                        capability=node.capability,
                        status="failed",
                        tool_name=node.tool_name,
                        started_at_utc=now,
                        finished_at_utc=now,
                        duration_ms=0.0,
                        attempts=0,
                        error=message,
                    )
                )
                mark_pending_skipped(
                    _dependency_skip_reason(
                        prefix="Skipped because the run stopped after an unmet required dependency",
                        dependency_ids=[node.node_id],
                        node_records=node_records,
                    )
                )
                return AgentExecutionSnapshot(
                    node_records=node_records,
                    node_results=completed,
                    failures=failures,
                    provenance_records=provenance_rows,
                    selected_docs=selected_docs,
                    status="failed",
                )

            ready_ids = []
            for node_id in list(pending):
                node = node_map[node_id]
                if all(dep in completed for dep in node.depends_on):
                    ready_ids.append(node_id)
            if not ready_ids:
                break
            required_ready_ids = [node_id for node_id in ready_ids if not node_map[node_id].optional]
            if required_ready_ids:
                ready_ids = required_ready_ids

            ready_nodes = [node_map[node_id] for node_id in ready_ids]
            tasks = []
            for node in ready_nodes:
                dependency_results = {dep: completed[dep] for dep in node.depends_on if dep in completed}
                tasks.append(self._execute_node(node, dependency_results, context))
            results = await asyncio.gather(*tasks)

            for node, (record, result, failure, provenance_row) in zip(ready_nodes, results, strict=False):
                pending.discard(node.node_id)
                effective_result = result
                effective_failure = failure
                if result is not None and _result_has_no_data(result):
                    reason = _result_no_data_reason(result)
                    if node.optional:
                        record.status = "skipped"
                        record.error = reason
                        effective_result = None
                        effective_failure = None
                        provenance_row = None
                    elif node.capability in _CORE_NO_DATA_FAILURE_CAPABILITIES:
                        record.status = "failed"
                        record.error = reason
                        effective_result = None
                        effective_failure = AgentFailure(
                            node_id=node.node_id,
                            capability=node.capability,
                            error_type="core_no_data",
                            message=reason,
                            retriable=True,
                            details={
                                "node_id": node.node_id,
                                "capability": node.capability,
                                "tool_name": record.tool_name,
                                "dependency_nodes": list(node.depends_on),
                                "inputs": _safe_json(node.inputs),
                            },
                        )
                node_records.append(record)
                if provenance_row is not None:
                    provenance_rows.append(provenance_row)
                if effective_result is not None:
                    completed[node.node_id] = effective_result
                    if node.capability == "fetch_documents":
                        payload = effective_result.payload if isinstance(effective_result.payload, dict) else {}
                        selected_docs = list(payload.get("documents", []))
                if effective_failure is not None:
                    failures.append(effective_failure)
                self._write_run_state(context.artifacts_dir, node_records, failures, plan_dag=plan_dag)
            failed_ready_ids = [
                item.node_id
                for item in node_records
                if item.node_id in ready_ids and item.status == "failed" and not node_map[item.node_id].optional
            ]
            if failed_ready_ids:
                # Two cases:
                #   1. A retrieval-bailout capability failed (db_search /
                #      sql_query_search produced no docs at all) — the entire
                #      run has nothing left to do, halt every remaining node.
                #   2. Any other capability failed — only that node's
                #      transitive children are unreachable. Independent DAG
                #      branches MUST keep executing. Previously a doc_embeddings
                #      failure would skip sentiment/topic/keyterms branches
                #      that had no dependency relationship to it; that was the
                #      observed "n18 failed -> 14 unrelated nodes skipped" bug.
                retrieval_bailout_failed = any(
                    node_map[nid].capability in _RETRIEVAL_BAILOUT_CAPABILITIES
                    for nid in failed_ready_ids
                )
                if retrieval_bailout_failed:
                    mark_pending_skipped(
                        _dependency_skip_reason(
                            prefix="Skipped because retrieval produced no documents; downstream analysis cannot run",
                            dependency_ids=failed_ready_ids,
                            node_records=node_records,
                        )
                    )
                    return AgentExecutionSnapshot(
                        node_records=node_records,
                        node_results=completed,
                        failures=failures,
                        provenance_records=provenance_rows,
                        selected_docs=selected_docs,
                        status="failed",
                    )
                # Partial-failure path: skip only descendants, continue with
                # the rest of the DAG.
                descendants = _transitive_dependents(set(failed_ready_ids), node_map) & pending
                if descendants:
                    skip_reason = _dependency_skip_reason(
                        prefix="Skipped because a required upstream node failed",
                        dependency_ids=failed_ready_ids,
                        node_records=node_records,
                    )
                    now = _utc_now()
                    for descendant_id in sorted(descendants):
                        descendant_node = node_map[descendant_id]
                        node_records.append(
                            AgentNodeExecutionRecord(
                                node_id=descendant_node.node_id,
                                capability=descendant_node.capability,
                                status="skipped",
                                tool_name=descendant_node.tool_name,
                                started_at_utc=now,
                                finished_at_utc=now,
                                duration_ms=0.0,
                                attempts=0,
                                error=skip_reason,
                            )
                        )
                    pending -= descendants
            if self._is_cancelled(context):
                mark_pending_skipped("Skipped because run abort was requested.")
                return AgentExecutionSnapshot(
                    node_records=node_records,
                    node_results=completed,
                    failures=failures,
                    provenance_records=provenance_rows,
                    selected_docs=selected_docs,
                    status="aborted",
                )

        # Final status:
        #   completed = no failures
        #   failed    = nothing succeeded (every executed node failed/skipped)
        #   partial   = at least one node succeeded but at least one failed
        if not failures:
            status = "completed"
        else:
            completed_count = sum(1 for record in node_records if record.status == "completed")
            status = "partial" if completed_count > 0 else "failed"
        return AgentExecutionSnapshot(
            node_records=node_records,
            node_results=completed,
            failures=failures,
            provenance_records=provenance_rows,
            selected_docs=selected_docs,
            status=status,
        )
