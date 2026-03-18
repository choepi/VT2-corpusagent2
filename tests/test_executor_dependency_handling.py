from __future__ import annotations

from pathlib import Path

from corpusagent2.execution_engine import PlanExecutor
from corpusagent2.plan_graph import NodeType, PlanGraph, PlanNode
from corpusagent2.question_spec import FeasibilityReport, QuestionSpec, TimeRange
from corpusagent2.tool_registry import ToolExecutionResult, ToolRegistry

from .helpers import StaticAdapter


def test_executor_handles_optional_failure_and_continues(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(
        StaticAdapter(
            tool_name="retrieve_tool",
            capability="retrieve.documents",
            run_fn=lambda p, d, c: ToolExecutionResult(payload={"results": [{"doc_id": "d1", "title": "Doc", "snippet": "Snippet"}]}),
        )
    )
    registry.register(
        StaticAdapter(
            tool_name="filter_tool",
            capability="filter.documents",
            run_fn=lambda p, d, c: ToolExecutionResult(payload=d["retrieval_results"].payload),
        )
    )
    registry.register(
        StaticAdapter(
            tool_name="aggregate_tool",
            capability="aggregate.findings",
            run_fn=lambda p, d, c: ToolExecutionResult(
                payload={
                    "highlights": ["h1"],
                    "candidate_claims": ["c1"],
                    "artifacts_used": [],
                }
            ),
        )
    )
    registry.register(
        StaticAdapter(
            tool_name="verify_tool",
            capability="verify.claims",
            run_fn=lambda p, d, c: (_ for _ in ()).throw(RuntimeError("verification failed")),
        )
    )
    registry.register(
        StaticAdapter(
            tool_name="synth_tool",
            capability="synthesize.answer",
            run_fn=lambda p, d, c: ToolExecutionResult(
                payload={
                    "answer_text": "grounded answer",
                    "evidence_items": d["filtered_results"].payload.get("results", []),
                    "artifacts_used": [],
                    "unsupported_parts": [],
                    "caveats": [],
                    "claim_verdicts": [],
                }
            ),
        )
    )

    plan = PlanGraph(
        question_id="q_exec_1",
        template_name="test_template",
        nodes=[
            PlanNode("retrieve", NodeType.RETRIEVE.value, "retrieve.documents", "retrieval_results"),
            PlanNode("filter", NodeType.FILTER.value, "filter.documents", "filtered_results", dependencies=["retrieve"]),
            PlanNode("aggregate", NodeType.AGGREGATE.value, "aggregate.findings", "aggregate_summary", dependencies=["filter"]),
            PlanNode("verify", NodeType.VERIFY.value, "verify.claims", "claim_verification", dependencies=["aggregate"], optional=True),
            PlanNode(
                "synthesize",
                NodeType.SYNTHESIZE.value,
                "synthesize.answer",
                "final_answer",
                dependencies=["aggregate", "verify", "filter"],
                cacheable=False,
            ),
        ],
    )
    spec = QuestionSpec(
        question_id="q_exec_1",
        raw_question="test",
        normalized_question="test",
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

    assert manifest.status == "partial"
    assert manifest.final_answer.answer_text == "grounded answer"
    verify_rows = [row for row in manifest.node_records if row.node_id == "verify"]
    assert verify_rows and verify_rows[0].status == "skipped"
    assert any(item.node_id == "verify" for item in manifest.failures)
