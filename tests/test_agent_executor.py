from __future__ import annotations

import asyncio
import time
from pathlib import Path

from corpusagent2.agent_backends import InMemoryWorkingSetStore
from corpusagent2.agent_capabilities import AgentExecutionContext
from corpusagent2.agent_executor import AsyncPlanExecutor
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
