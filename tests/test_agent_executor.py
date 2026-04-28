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
