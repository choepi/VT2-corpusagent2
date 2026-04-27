from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import time
import uuid
from typing import Any

from .io_utils import write_json
from .plan_graph import PlanGraph, PlanNode
from .provenance import make_provenance_record
from .question_spec import QuestionSpec
from .run_manifest import (
    FinalAnswerPayload,
    NodeExecutionRecord,
    RunManifest,
    StructuredFailure,
)
from .tool_registry import ToolExecutionResult, ToolRegistry


def _safe_json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


@dataclass(slots=True)
class ExecutionContext:
    run_id: str
    runtime: Any
    question_spec: QuestionSpec
    artifacts_dir: Path


class PlanExecutor:
    def __init__(self, registry: ToolRegistry, cache: dict[str, ToolExecutionResult] | None = None) -> None:
        self.registry = registry
        self._cache = cache if cache is not None else {}

    def _cache_key(
        self,
        node: PlanNode,
        resolution_tool_name: str,
        dependency_results: dict[str, ToolExecutionResult],
        context: ExecutionContext,
    ) -> str:
        dependency_fingerprint = {
            key: sha256(json.dumps(_safe_json(value.payload), sort_keys=True).encode("utf-8")).hexdigest()
            for key, value in sorted(dependency_results.items())
        }
        has_explicit_query = bool(str(node.params.get("query", "")).strip())
        payload = {
            "capability": node.capability,
            "tool_name": resolution_tool_name,
            "params": _safe_json(node.params),
            "dependencies": dependency_fingerprint,
        }
        if node.capability in {"db_search", "sql_query_search"} and not has_explicit_query:
            payload["implicit_query_context"] = str(getattr(context.question_spec, "raw_question", "")).strip()
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return sha256(encoded).hexdigest()

    def _write_node_artifact(self, artifacts_dir: Path, node: PlanNode, result: ToolExecutionResult) -> str:
        target = artifacts_dir / "nodes" / f"{node.node_id}.json"
        write_json(target, {"payload": _safe_json(result.payload), "metadata": _safe_json(result.metadata)})
        return str(target)

    def execute(
        self,
        plan_graph: PlanGraph,
        question_spec: QuestionSpec,
        runtime: Any,
        run_id: str | None = None,
        artifacts_root: Path | None = None,
    ) -> RunManifest:
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        artifacts_root = artifacts_root or Path.cwd() / "outputs" / "framework"
        artifacts_dir = (artifacts_root / run_id).resolve()
        (artifacts_dir / "nodes").mkdir(parents=True, exist_ok=True)

        exec_context = ExecutionContext(
            run_id=run_id,
            runtime=runtime,
            question_spec=question_spec,
            artifacts_dir=artifacts_dir,
        )

        node_map = plan_graph.node_map()
        node_status: dict[str, str] = {}
        node_records: list[NodeExecutionRecord] = []
        failures: list[StructuredFailure] = []
        provenance_rows: list[dict[str, Any]] = []
        output_results: dict[str, ToolExecutionResult] = {}

        for node in plan_graph.topological_order():
            started_at = _utc_now()
            started_perf = time.perf_counter()

            missing_dependencies: list[str] = []
            dependency_results: dict[str, ToolExecutionResult] = {}
            for dep_id in node.dependencies:
                dep_node = node_map[dep_id]
                dep_output_key = dep_node.output_key
                dep_result = output_results.get(dep_output_key)
                if dep_result is None:
                    if dep_node.optional:
                        continue
                    missing_dependencies.append(dep_id)
                    continue
                dependency_results[dep_output_key] = dep_result

            if missing_dependencies:
                finished_at = _utc_now()
                duration_ms = (time.perf_counter() - started_perf) * 1000.0
                message = f"Missing dependency outputs from nodes: {missing_dependencies}"
                status = "skipped" if node.optional else "failed"
                failures.append(
                    StructuredFailure(
                        node_id=node.node_id,
                        capability=node.capability,
                        error_type="dependency_missing",
                        message=message,
                        retriable=node.optional,
                        details={"missing_dependencies": missing_dependencies, "optional_node": node.optional},
                    )
                )
                node_status[node.node_id] = status
                node_records.append(
                    NodeExecutionRecord(
                        node_id=node.node_id,
                        node_type=node.node_type,
                        capability=node.capability,
                        output_key=node.output_key,
                        status=status,
                        started_at_utc=started_at,
                        finished_at_utc=finished_at,
                        duration_ms=duration_ms,
                        error=message,
                    )
                )
                continue

            try:
                resolution = self.registry.resolve(
                    capability=node.capability,
                    context=exec_context,
                    params=node.params,
                )
                cache_key = self._cache_key(node, resolution.spec.tool_name, dependency_results, exec_context)
                cache_hit = bool(node.cacheable and cache_key in self._cache)
                if cache_hit:
                    result = self._cache[cache_key]
                else:
                    result = resolution.adapter.run(
                        params=dict(node.params),
                        dependency_results=dependency_results,
                        context=exec_context,
                    )
                    if node.cacheable:
                        self._cache[cache_key] = result

                node_artifact_path = self._write_node_artifact(artifacts_dir, node, result)
                artifacts_used = _dedupe(list(result.artifacts_used) + [node_artifact_path])

                output_results[node.output_key] = ToolExecutionResult(
                    payload=result.payload,
                    evidence=list(result.evidence_items),
                    artifacts=artifacts_used,
                    caveats=list(result.caveats),
                    unsupported_parts=list(result.unsupported_parts),
                    metadata=dict(result.metadata),
                )

                provenance = make_provenance_record(
                    run_id=run_id,
                    tool_name=resolution.spec.tool_name,
                    tool_version=resolution.spec.tool_version,
                    model_id=resolution.spec.model_id or resolution.spec.provider,
                    params=dict(node.params),
                    inputs_ref={
                        "question_id": question_spec.question_id,
                        "node_id": node.node_id,
                        "dependency_outputs": sorted(dependency_results.keys()),
                    },
                    outputs_ref={"output_key": node.output_key, "artifact_path": node_artifact_path},
                    evidence=list(result.evidence_items),
                )
                provenance_rows.append(provenance.to_dict())

                finished_at = _utc_now()
                duration_ms = (time.perf_counter() - started_perf) * 1000.0
                node_status[node.node_id] = "completed"
                node_records.append(
                    NodeExecutionRecord(
                        node_id=node.node_id,
                        node_type=node.node_type,
                        capability=node.capability,
                        output_key=node.output_key,
                        status="completed",
                        started_at_utc=started_at,
                        finished_at_utc=finished_at,
                        duration_ms=duration_ms,
                        cache_key=cache_key,
                        cache_hit=cache_hit,
                        tool_name=resolution.spec.tool_name,
                        provider=resolution.spec.provider,
                        tool_version=resolution.spec.tool_version,
                        model_id=resolution.spec.model_id,
                        tool_reason=resolution.reason,
                        artifacts_used=artifacts_used,
                        evidence_count=len(result.evidence_items),
                        caveats=list(result.caveats),
                        unsupported_parts=list(result.unsupported_parts),
                    )
                )
            except Exception as exc:
                finished_at = _utc_now()
                duration_ms = (time.perf_counter() - started_perf) * 1000.0
                status = "skipped" if node.optional else "failed"
                message = str(exc)
                failures.append(
                    StructuredFailure(
                        node_id=node.node_id,
                        capability=node.capability,
                        error_type=exc.__class__.__name__,
                        message=message,
                        retriable=node.optional,
                        details={"optional_node": node.optional},
                    )
                )
                node_status[node.node_id] = status
                node_records.append(
                    NodeExecutionRecord(
                        node_id=node.node_id,
                        node_type=node.node_type,
                        capability=node.capability,
                        output_key=node.output_key,
                        status=status,
                        started_at_utc=started_at,
                        finished_at_utc=finished_at,
                        duration_ms=duration_ms,
                        error=message,
                    )
                )

        final_result = output_results.get(plan_graph.final_output_key)
        if final_result is None:
            final_payload = FinalAnswerPayload(
                answer_text="The run did not reach a final synthesis node.",
                unsupported_parts=_dedupe([item.message for item in failures]),
                caveats=["Partial completion: downstream answer synthesis did not execute."],
            )
        else:
            final_payload = FinalAnswerPayload.from_payload(
                final_result.payload if isinstance(final_result.payload, dict) else {}
            )
            final_payload.artifacts_used = _dedupe(
                list(final_payload.artifacts_used) + list(final_result.artifacts_used)
            )
            final_payload.caveats = _dedupe(list(final_payload.caveats) + list(final_result.caveats))
            final_payload.unsupported_parts = _dedupe(
                list(final_payload.unsupported_parts) + list(final_result.unsupported_parts)
            )

        status = "completed"
        if failures and final_result is not None:
            status = "partial"
        elif failures and final_result is None:
            status = "failed"

        return RunManifest(
            run_id=run_id,
            question_spec=question_spec,
            plan_graph=plan_graph,
            node_records=node_records,
            provenance_records=provenance_rows,
            final_answer=final_payload,
            artifacts_dir=str(artifacts_dir),
            status=status,
            failures=failures,
            cache_entries=len(self._cache),
        )
