from __future__ import annotations

from corpusagent2.agent_models import PlannerAction


def test_planner_action_from_dict_accepts_empty_plan_without_crashing() -> None:
    action = PlannerAction.from_dict(
        {
            "action": "emit_plan_dag",
            "rewritten_question": "Test question",
            "plan_dag": {},
        }
    )

    assert action.action == "emit_plan_dag"
    assert action.plan_dag is None
