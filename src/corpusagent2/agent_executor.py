from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

from .agent_capabilities import AgentExecutionContext
from .agent_models import AgentFailure, AgentNodeExecutionRecord, AgentPlanDAG, AgentPlanNode
from .io_utils import write_json
from .provenance import make_provenance_record
from .tool_registry import ToolExecutionResult, ToolRegistry


def _safe_json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


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


def _summarize_tool_result(result: ToolExecutionResult, *, dependency_results: dict[str, ToolExecutionResult]) -> dict[str, Any]:
    payload = result.payload
    items_key, items = _result_items(payload)
    summary = {
        "items_key": items_key,
        "items_count": len(items),
        "documents_processed": _dependency_document_count(dependency_results),
        "evidence_count": len(result.evidence_items),
        "artifact_count": len(result.artifacts_used),
        "caveat_count": len(result.caveats),
        "unsupported_count": len(result.unsupported_parts),
        "payload_preview": _payload_preview(payload),
    }
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
    ) -> str:
        dependency_fingerprint = {
            key: sha256(json.dumps(_safe_json(value.payload), sort_keys=True).encode("utf-8")).hexdigest()
            for key, value in sorted(dependency_results.items())
        }
        payload = {
            "capability": node.capability,
            "tool_name": resolution_tool_name,
            "inputs": _safe_json(node.inputs),
            "dependencies": dependency_fingerprint,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return sha256(encoded).hexdigest()

    def _write_node_artifact(self, artifacts_dir: Path, node: AgentPlanNode, result: ToolExecutionResult) -> str:
        target = artifacts_dir / "nodes" / f"{node.node_id}.json"
        write_json(target, {"payload": _safe_json(result.payload), "metadata": _safe_json(result.metadata)})
        return str(target)

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

        try:
            resolution = self.registry.resolve(
                capability=node.capability,
                context=context,
                params=node.inputs,
            )
        except Exception as exc:
            finished_at = _utc_now()
            duration_ms = (time.perf_counter() - started_perf) * 1000.0
            self._emit_event(
                context,
                {
                    "event": "node_failed",
                    "node_id": node.node_id,
                    "capability": node.capability,
                    "status": "failed",
                    "error": str(exc),
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
            )
            return (
                AgentNodeExecutionRecord(
                    node_id=node.node_id,
                    capability=node.capability,
                    status="failed",
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

        cache_key = self._cache_key(node, resolution.spec.tool_name, dependency_results)
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
            artifact_path = self._write_node_artifact(context.artifacts_dir, node, result)
            context.working_store.record_tool_call(
                context.run_id,
                node.node_id,
                node.capability,
                resolved_tool_name,
                "completed",
                {
                    "cache_hit": True,
                    "started_at_utc": started_at,
                    "finished_at_utc": _utc_now(),
                    "inputs": _safe_json(node.inputs),
                    "dependency_nodes": sorted(dependency_results.keys()),
                    "cache_key": cache_key,
                    "summary": _summarize_tool_result(result, dependency_results=dependency_results),
                    "payload": _safe_json(result.payload),
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
                    "status": "completed",
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
                },
            )
            return (
                AgentNodeExecutionRecord(
                    node_id=node.node_id,
                    capability=node.capability,
                    status="completed",
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

        while attempts < 2:
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
                context.working_store.record_tool_call(
                    context.run_id,
                    node.node_id,
                    node.capability,
                    str(result.metadata.get("tool_name", resolution.spec.tool_name)),
                    "completed",
                    {
                        "cache_hit": False,
                        "attempt": attempts,
                        "started_at_utc": started_at,
                        "finished_at_utc": _utc_now(),
                        "inputs": _safe_json(node.inputs),
                        "dependency_nodes": sorted(dependency_results.keys()),
                        "cache_key": cache_key,
                        "summary": _summarize_tool_result(result, dependency_results=dependency_results),
                        "payload": _safe_json(result.payload),
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
                self._emit_event(
                    context,
                    {
                        "event": "node_completed",
                        "node_id": node.node_id,
                        "capability": node.capability,
                        "status": "completed",
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
                    },
                )
                return (
                    AgentNodeExecutionRecord(
                        node_id=node.node_id,
                        capability=node.capability,
                        status="completed",
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
                context.working_store.record_tool_call(
                    context.run_id,
                    node.node_id,
                    node.capability,
                    resolution.spec.tool_name,
                    "failed",
                    {
                        "error": last_error,
                        "attempt": attempts,
                        "started_at_utc": started_at,
                        "inputs": _safe_json(node.inputs),
                        "dependency_nodes": sorted(dependency_results.keys()),
                        "cache_key": cache_key,
                    },
                )

        finished_at = _utc_now()
        duration_ms = (time.perf_counter() - started_perf) * 1000.0
        self._emit_event(
            context,
            {
                "event": "node_failed",
                "node_id": node.node_id,
                "capability": node.capability,
                "status": "failed",
                "error": last_error,
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
            },
        )
        failure = AgentFailure(
            node_id=node.node_id,
            capability=node.capability,
            error_type="execution_failed",
            message=last_error,
            retriable=node.optional,
        )
        status = "skipped" if node.optional else "failed"
        return (
            AgentNodeExecutionRecord(
                node_id=node.node_id,
                capability=node.capability,
                status=status,
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

        while pending:
            if self._is_cancelled(context):
                return AgentExecutionSnapshot(
                    node_records=node_records,
                    node_results=completed,
                    failures=failures,
                    provenance_records=provenance_rows,
                    selected_docs=selected_docs,
                    status="aborted",
                )
            ready_ids = []
            for node_id in list(pending):
                node = node_map[node_id]
                if all(dep in completed or any(item.node_id == dep and item.status == "skipped" for item in node_records) for dep in node.depends_on):
                    ready_ids.append(node_id)
            if not ready_ids:
                break

            ready_nodes = [node_map[node_id] for node_id in ready_ids]
            tasks = []
            for node in ready_nodes:
                dependency_results = {dep: completed[dep] for dep in node.depends_on if dep in completed}
                tasks.append(self._execute_node(node, dependency_results, context))
            results = await asyncio.gather(*tasks)

            for node, (record, result, failure, provenance_row) in zip(ready_nodes, results, strict=False):
                pending.discard(node.node_id)
                node_records.append(record)
                if provenance_row is not None:
                    provenance_rows.append(provenance_row)
                if result is not None:
                    completed[node.node_id] = result
                    if node.capability == "fetch_documents":
                        payload = result.payload if isinstance(result.payload, dict) else {}
                        selected_docs = list(payload.get("documents", []))
                if failure is not None:
                    failures.append(failure)
                    if not node.optional:
                        return AgentExecutionSnapshot(
                            node_records=node_records,
                            node_results=completed,
                            failures=failures,
                            provenance_records=provenance_rows,
                            selected_docs=selected_docs,
                            status="failed",
                        )
            if self._is_cancelled(context):
                return AgentExecutionSnapshot(
                    node_records=node_records,
                    node_results=completed,
                    failures=failures,
                    provenance_records=provenance_rows,
                    selected_docs=selected_docs,
                    status="aborted",
                )

        status = "completed" if not failures else "partial"
        return AgentExecutionSnapshot(
            node_records=node_records,
            node_results=completed,
            failures=failures,
            provenance_records=provenance_rows,
            selected_docs=selected_docs,
            status=status,
        )
