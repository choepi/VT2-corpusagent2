from __future__ import annotations

from corpusagent2.planner import PlanningContext, QuestionPlanner
from corpusagent2.tool_registry import ToolExecutionResult, ToolRegistry

from .helpers import StaticAdapter


def _registry_with_probe_results(results: list[dict]) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        StaticAdapter(
            tool_name="retrieve_probe",
            capability="retrieve.documents",
            run_fn=lambda p, d, c: ToolExecutionResult(payload={"results": list(results)}),
        )
    )
    for capability in (
        "filter.documents",
        "analyze.entity_trend",
        "analyze.sentiment_series",
        "analyze.topics_over_time",
        "analyze.burst_events",
        "analyze.keyphrases",
        "aggregate.findings",
        "verify.claims",
        "synthesize.answer",
    ):
        registry.register(StaticAdapter(tool_name=f"{capability}_tool", capability=capability))
    return registry


def test_metadata_stage_blocks_when_schema_missing() -> None:
    planner = QuestionPlanner(registry=_registry_with_probe_results(results=[]))
    context = PlanningContext(
        total_documents=10,
        metadata_columns={"doc_id", "title"},
        available_artifacts=set(),
        retrieval_ready=True,
    )
    spec = planner.build_question_spec("What happened in inflation coverage?", planning_context=context, question_id="qf1")
    assert spec.feasibility_status == "not_feasible"
    assert any("Missing required metadata columns" in reason for reason in spec.unsupported_reasons)


def test_retrieval_stage_blocks_when_probe_returns_empty() -> None:
    planner = QuestionPlanner(registry=_registry_with_probe_results(results=[]))
    context = PlanningContext(
        total_documents=10,
        metadata_columns={"doc_id", "title", "text", "published_at"},
        available_artifacts=set(),
        retrieval_ready=True,
    )
    spec = planner.build_question_spec("What happened in inflation coverage?", planning_context=context, question_id="qf2")
    assert spec.feasibility_status == "not_feasible"
    assert any("Lightweight retrieval found no supporting documents" in reason for reason in spec.unsupported_reasons)


def test_retrieval_stage_can_be_partially_feasible() -> None:
    planner = QuestionPlanner(
        registry=_registry_with_probe_results(
            results=[{"doc_id": "d1", "title": "Inflation and ECB", "snippet": "No mention of Tesla here"}]
        )
    )
    context = PlanningContext(
        total_documents=10,
        metadata_columns={"doc_id", "title", "text", "published_at"},
        available_artifacts=set(),
        retrieval_ready=True,
    )
    spec = planner.build_question_spec(
        "Compare sentiment for 'Tesla' and 'BYD' coverage over time.",
        planning_context=context,
        question_id="qf3",
    )
    assert spec.feasibility_status == "partially_feasible"
