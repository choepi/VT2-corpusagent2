from __future__ import annotations

from pathlib import Path

from corpusagent2.execution_engine import PlanExecutor
from corpusagent2.plan_graph import NodeType, PlanGraph, PlanNode
from corpusagent2.question_spec import FeasibilityReport, QuestionSpec, TimeRange
from corpusagent2.tool_registry import ToolExecutionResult, ToolRegistry

from .helpers import StaticAdapter


def test_executor_captures_provenance(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(
        StaticAdapter(
            tool_name="synth_tool",
            capability="synthesize.answer",
            run_fn=lambda p, d, c: ToolExecutionResult(
                payload={
                    "answer_text": "ok",
                    "evidence_items": [],
                    "artifacts_used": [],
                    "unsupported_parts": [],
                    "caveats": [],
                    "claim_verdicts": [],
                }
            ),
        )
    )
    plan = PlanGraph(
        question_id="q_prov_1",
        template_name="prov_template",
        nodes=[
            PlanNode(
                node_id="synthesize",
                node_type=NodeType.SYNTHESIZE.value,
                capability="synthesize.answer",
                output_key="final_answer",
                params={"question_text": "x"},
                cacheable=False,
            )
        ],
    )
    spec = QuestionSpec(
        question_id="q_prov_1",
        raw_question="x",
        normalized_question="x",
        question_class="retrieval_qa",
        time_range=TimeRange(),
        feasibility=FeasibilityReport(),
    )
    manifest = PlanExecutor(registry=registry).execute(
        plan_graph=plan,
        question_spec=spec,
        runtime=object(),
        artifacts_root=tmp_path,
    )
    assert manifest.provenance_records
    first = manifest.provenance_records[0]
    assert first["run_id"] == manifest.run_id
    assert first["tool_name"] == "synth_tool"
    assert "params_hash" in first and first["params_hash"]
