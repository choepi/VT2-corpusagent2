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
        self._emit_event(
            context,
            {
                "event": "node_started",
                "node_id": node.node_id,
                "capability": node.capability,
                "status": "running",
            },
        )

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
        cache_hit = bool(not getattr(context.state, "no_cache", False) and node.cacheable and cache_key in self._cache)
        if cache_hit:
            context.working_store.record_tool_call(
                context.run_id,
                node.node_id,
                node.capability,
                resolution.spec.tool_name,
                "running",
                {"cache_lookup": True},
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
                {"cache_hit": True, "payload": _safe_json(result.payload)},
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
                    {"attempt": attempts},
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
                    {"cache_hit": False, "payload": _safe_json(result.payload)},
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
                    {"error": last_error, "attempt": attempts},
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
