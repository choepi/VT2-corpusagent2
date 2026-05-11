from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from corpusagent2.agent_backends import InMemoryWorkingSetStore
from corpusagent2.agent_capabilities import AgentExecutionContext
from corpusagent2.agent_executor import AsyncPlanExecutor, _summarize_tool_result
from corpusagent2.agent_models import AgentPlanDAG, AgentPlanNode
from corpusagent2.tool_registry import ToolExecutionResult, ToolRegistry

from .helpers import StaticAdapter


class _SearchBackend:
    def search(self, *, query: str, top_k: int, date_from: str = "", date_to: str = ""):
        return []


def test_async_executor_runs_independent_nodes_in_parallel(tmp_path: Path) -> None:
    registry = ToolRegistry()
    for name in ("a", "b"):
        registry.register(
            StaticAdapter(
                tool_name=f"{name}_tool",
                capability=name,
                run_fn=lambda p, d, c: (time.sleep(0.2), ToolExecutionResult(payload={"ok": True}))[1],
            )
        )
    registry.register(
        StaticAdapter(
            tool_name="join_tool",
            capability="join",
            run_fn=lambda p, d, c: ToolExecutionResult(payload={"deps": sorted(d.keys())}),
        )
    )

    plan = AgentPlanDAG(
        nodes=[
            AgentPlanNode("node_a", "a"),
            AgentPlanNode("node_b", "b"),
            AgentPlanNode("node_join", "join", depends_on=["node_a", "node_b"]),
        ]
    )
    context = AgentExecutionContext(
        run_id="run_parallel",
        artifacts_dir=tmp_path,
        search_backend=_SearchBackend(),
        working_store=InMemoryWorkingSetStore(),
    )

    started = time.perf_counter()
    snapshot = asyncio.run(AsyncPlanExecutor(registry).execute(plan, context))
    duration = time.perf_counter() - started

    assert snapshot.status == "completed"
    assert duration < 0.5


def test_async_executor_marks_pending_nodes_skipped_when_cancelled_before_execution(tmp_path: Path) -> None:
    plan = AgentPlanDAG(
        nodes=[
            AgentPlanNode("node_a", "a"),
            AgentPlanNode("node_b", "b", depends_on=["node_a"]),
        ]
    )
    context = AgentExecutionContext(
        run_id="run_cancelled",
        artifacts_dir=tmp_path,
        search_backend=_SearchBackend(),
        working_store=InMemoryWorkingSetStore(),
        cancel_requested=lambda: True,
    )

    snapshot = asyncio.run(AsyncPlanExecutor(ToolRegistry()).execute(plan, context))

    assert snapshot.status == "aborted"
    assert {record.node_id for record in snapshot.node_records} == {"node_a", "node_b"}
    assert all(record.status == "skipped" for record in snapshot.node_records)


def test_async_executor_stops_scheduling_after_cancel_requested(tmp_path: Path) -> None:
    registry = ToolRegistry()
    cancel_requested = False
    second_started = False

    def first_node(_params, _deps, _context):
        nonlocal cancel_requested
        cancel_requested = True
        return ToolExecutionResult(payload={"ok": True})

    def second_node(_params, _deps, _context):
        nonlocal second_started
        second_started = True
        return ToolExecutionResult(payload={"ok": True})

    registry.register(StaticAdapter(tool_name="first_tool", capability="first", run_fn=first_node))
    registry.register(StaticAdapter(tool_name="second_tool", capability="second", run_fn=second_node))
    plan = AgentPlanDAG(
        nodes=[
            AgentPlanNode("first", "first"),
            AgentPlanNode("second", "second", depends_on=["first"]),
        ]
    )
    context = AgentExecutionContext(
        run_id="run_cancel_after_first",
        artifacts_dir=tmp_path,
        search_backend=_SearchBackend(),
        working_store=InMemoryWorkingSetStore(),
        cancel_requested=lambda: cancel_requested,
    )

    snapshot = asyncio.run(AsyncPlanExecutor(registry).execute(plan, context))

    records = {record.node_id: record for record in snapshot.node_records}
    assert snapshot.status == "aborted"
    assert records["first"].status == "completed"
    assert records["second"].status == "skipped"
    assert second_started is False


def test_async_executor_skipped_nodes_include_failed_dependency_reason(tmp_path: Path) -> None:
    registry = ToolRegistry()
    downstream_started = False

    registry.register(
        StaticAdapter(
            tool_name="series_tool",
            capability="time_series_aggregate",
            run_fn=lambda _p, _d, _c: ToolExecutionResult(
                payload={"rows": []},
                metadata={"no_data": True, "no_data_reason": "Missing time field: quarter"},
            ),
        )
    )

    def downstream(_params, _deps, _context):
        nonlocal downstream_started
        downstream_started = True
        return ToolExecutionResult(payload={"ok": True})

    registry.register(StaticAdapter(tool_name="plot_tool", capability="plot_artifact", run_fn=downstream))
    plan = AgentPlanDAG(
        nodes=[
            AgentPlanNode("series", "time_series_aggregate"),
            AgentPlanNode("plot", "plot_artifact", depends_on=["series"]),
        ]
    )
    context = AgentExecutionContext(
        run_id="run_dependency_reason",
        artifacts_dir=tmp_path,
        search_backend=_SearchBackend(),
        working_store=InMemoryWorkingSetStore(),
    )

    snapshot = asyncio.run(AsyncPlanExecutor(registry).execute(plan, context))

    records = {record.node_id: record for record in snapshot.node_records}
    # Two-node plan where the only producer fails: nothing succeeded, so the
    # overall status is "failed". (Multi-branch plans with at least one
    # successful node would be "partial" under the new cascade semantics.)
    assert snapshot.status == "failed"
    assert records["series"].status == "failed"
    assert records["plot"].status == "skipped"
    assert "series (time_series_aggregate via series_tool): Missing time field: quarter" in records["plot"].error
    assert downstream_started is False
    assert snapshot.failures[0].details["dependency_nodes"] == []


def test_summarize_tool_result_distinguishes_input_docs_and_no_data() -> None:
    dependency_results = {
        "fetch": ToolExecutionResult(
            payload={
                "documents": [
                    {"doc_id": "doc-1", "text": "Alpha"},
                    {"doc_id": "doc-2", "text": "Beta"},
                ]
            }
        )
    }
    result = ToolExecutionResult(
        payload={"rows": []},
        caveats=["No rows available for plotting."],
        metadata={"no_data": True, "no_data_reason": "No rows available for plotting."},
    )

    summary = _summarize_tool_result(result, dependency_results=dependency_results)

    assert summary["input_documents_seen"] == 2
    assert summary["output_documents"] == 0
    assert summary["no_data"] is True
    assert summary["no_data_reason"] == "No rows available for plotting."


def test_node_artifact_preserves_artifacts_and_caveats(tmp_path: Path) -> None:
    executor = AsyncPlanExecutor(ToolRegistry())
    plot_path = tmp_path / "plots" / "chart.png"
    result = ToolExecutionResult(
        payload={"artifact_path": str(plot_path), "rows": []},
        artifacts=[str(plot_path)],
        caveats=["No rows available for plotting."],
        unsupported_parts=["Plot could not be generated."],
        metadata={"no_data": True},
    )

    artifact_path = executor._write_node_artifact(tmp_path, AgentPlanNode("plot", "plot_artifact"), result)
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))

    assert payload["artifacts"] == [str(plot_path)]
    assert payload["caveats"] == ["No rows available for plotting."]
    assert payload["unsupported_parts"] == ["Plot could not be generated."]
