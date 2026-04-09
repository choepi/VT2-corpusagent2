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


def test_planner_action_from_dict_accepts_openai_style_id_and_input_dependencies() -> None:
    action = PlannerAction.from_dict(
        {
            "action": "emit_plan_dag",
            "rewritten_question": "Analyze Facebook framing over time.",
            "plan_dag": {
                "nodes": [
                    {
                        "id": "n1",
                        "capability": "db_search",
                        "inputs": {"top_k": 50},
                        "task": "Search the corpus.",
                    },
                    {
                        "id": "n2",
                        "capability": "fetch_documents",
                        "inputs": ["n1"],
                        "task": "Fetch matching documents.",
                    },
                ]
            },
        }
    )

    assert action.plan_dag is not None
    nodes = {node.node_id: node for node in action.plan_dag.nodes}
    assert nodes["n1"].description == "Search the corpus."
    assert nodes["n2"].depends_on == ["n1"]
