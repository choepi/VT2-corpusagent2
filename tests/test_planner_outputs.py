from __future__ import annotations

from corpusagent2.planner import PlanningContext, QuestionPlanner
from corpusagent2.tool_registry import ToolExecutionResult, ToolRegistry

from .helpers import StaticAdapter


def _registry_for_planner() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        StaticAdapter(
            tool_name="retrieve_probe",
            capability="retrieve.documents",
            run_fn=lambda params, deps, ctx: ToolExecutionResult(
                payload={
                    "results": [
                        {"doc_id": "d1", "title": "Inflation and ECB", "snippet": "Inflation coverage mentioned ECB."},
                        {"doc_id": "d2", "title": "Energy prices", "snippet": "Energy and inflation trends."},
                    ]
                }
            ),
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


def test_planner_builds_sentiment_template() -> None:
    planner = QuestionPlanner(registry=_registry_for_planner())
    planning_context = PlanningContext(
        total_documents=100,
        metadata_columns={"doc_id", "title", "text", "published_at"},
        available_artifacts={"sentiment_series"},
        retrieval_ready=True,
        dense_ready=True,
    )
    spec = planner.build_question_spec(
        raw_question="track sentiment in inflation coverage from 2020 to 2024.",
        planning_context=planning_context,
        question_id="q_sent_01",
    )
    assert spec.question_class == "sentiment_trend"
    assert "analyze.sentiment_series" in spec.required_capabilities
    assert spec.feasibility_status == "feasible"

    plan = planner.build_plan(spec)
    assert plan.template_name == "sentiment_trend_template"
    assert plan.final_output_key == "final_answer"
    assert any(node.capability == "analyze.sentiment_series" for node in plan.nodes)
    assert any(node.capability == "synthesize.answer" for node in plan.nodes)


def test_planner_requests_clarification_for_ambiguous_compare() -> None:
    planner = QuestionPlanner(registry=_registry_for_planner())
    planning_context = PlanningContext(
        total_documents=100,
        metadata_columns={"doc_id", "title", "text", "published_at"},
        available_artifacts=set(),
        retrieval_ready=True,
        dense_ready=True,
    )
    spec = planner.build_question_spec(
        raw_question="Compare sentiment between groups over time.",
        planning_context=planning_context,
        question_id="q_cmp_01",
    )
    assert spec.feasibility_status == "needs_clarification"
    assert spec.clarification_question is not None
