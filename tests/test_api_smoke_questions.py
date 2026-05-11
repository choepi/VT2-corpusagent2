from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_smoke_runner():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "34_run_api_smoke_questions.py"
    spec = importlib.util.spec_from_file_location("api_smoke_questions", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_question_set_contains_two_reliable_smoke_questions() -> None:
    module = _load_smoke_runner()
    questions = module._load_question_set(Path(__file__).resolve().parents[1] / "config" / "smoke_questions_10_rows.json")

    assert [item["id"] for item in questions] == ["ukraine_energy_articles", "markets_public_policy"]
    assert all(item["expected_terms"] for item in questions)


def test_smoke_runner_reports_completed_queries(monkeypatch, tmp_path: Path) -> None:
    module = _load_smoke_runner()
    module.SUMMARY_PATH = tmp_path / "summary.json"

    def fake_json_request(method: str, url: str, *, payload=None, timeout_s: float = 180.0):
        if url.endswith("/runtime-info"):
            return {
                "retrieval": {
                    "backend": "pgvector",
                    "health": {"document_count": 10},
                }
            }
        assert method == "POST"
        question = payload["question"]
        return {
            "run_id": "agent_test",
            "status": "completed",
            "final_answer": {
                "answer_text": f"Smoke answer for {question}. Ukraine energy markets public policy smoke.",
                "caveats": [],
            },
            "evidence_table": [
                {
                    "doc_id": "smoke-1",
                    "excerpt": "This smoke article mentions Ukraine, energy, markets, and public policy.",
                }
            ],
            "selected_docs": [],
            "metadata": {"execution_diagnostics": {}},
        }

    monkeypatch.setattr(module, "_json_request", fake_json_request)

    exit_code = module.main(["--api-base-url", "http://test.local", "--strict-expected-terms"])

    assert exit_code == 0
    assert module.SUMMARY_PATH.exists()
