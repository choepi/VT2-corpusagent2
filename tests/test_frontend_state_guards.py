from __future__ import annotations

from pathlib import Path


def test_frontend_does_not_submit_stale_clarification_history_for_new_question() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_js = (project_root / "web" / "app.js").read_text(encoding="utf-8")

    assert "let clarificationBaseQuestion" in app_js
    assert "function clarificationMatchesCurrentQuestion()" in app_js
    assert "if (!clarificationMatchesCurrentQuestion())" in app_js
    assert "clarification_history: preserveClarificationHistory ? clarificationHistory : []" in app_js


def test_frontend_layout_keeps_evidence_single_and_moves_usage_to_advanced() -> None:
    project_root = Path(__file__).resolve().parents[1]
    index_html = (project_root / "web" / "index.html").read_text(encoding="utf-8")

    evidence_index = index_html.index("<h2>Evidence</h2>")
    advanced_index = index_html.index('data-tab="advanced"')
    usage_index = index_html.index("<h2>Historical Tool Usage</h2>")

    assert "id=\"selectedDocs\"" not in index_html
    assert "<h3>Selected Documents</h3>" not in index_html
    assert evidence_index < advanced_index
    assert advanced_index < usage_index
    assert "class=\"claim-verdict-grid\"" in index_html


def test_frontend_suppresses_notifications_for_restored_or_instant_cached_runs() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_js = (project_root / "web" / "app.js").read_text(encoding="utf-8")

    assert "const notificationEligibleRunIds = new Set();" in app_js
    assert "const notificationObservedActiveRunIds = new Set();" in app_js
    assert "!notificationEligibleRunIds.has(runId) || !notificationObservedActiveRunIds.has(runId)" in app_js
    assert "notificationEligibleRunIds.clear();" in app_js
    assert "notificationObservedActiveRunIds.clear();" in app_js
    assert "notificationEligibleRunIds.add(currentRunId);" in app_js
    assert "notificationObservedActiveRunIds.add(runId);" in app_js


def test_frontend_prefers_local_runtime_api_base_for_static_launch() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_js = (project_root / "web" / "app.js").read_text(encoding="utf-8")

    assert "function initialApiBase(savedValue = \"\")" in app_js
    assert "runtimeConfig.preferRuntimeApiBase" in app_js
    assert "127.0.1" in app_js
    assert "localhost:5500" in app_js
