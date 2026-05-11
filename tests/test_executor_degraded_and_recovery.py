from __future__ import annotations

from corpusagent2.agent_executor import (
    _CORE_NO_DATA_FAILURE_CAPABILITIES,
    _HEURISTIC_FALLBACK_DEGRADES_CAPABILITY,
    _RETRIEVAL_BAILOUT_CAPABILITIES,
    _build_degraded_recovery_result,
    _node_status_from_result,
)
from corpusagent2.tool_registry import ToolExecutionResult


def _result(metadata: dict[str, object], payload: dict[str, object] | None = None) -> ToolExecutionResult:
    return ToolExecutionResult(payload=payload if payload is not None else {"rows": []}, metadata=metadata)


def test_metadata_degraded_flag_demotes_to_degraded() -> None:
    result = _result({"degraded": True, "provider": "matplotlib"})
    assert _node_status_from_result("plot_artifact", result) == "degraded"


def test_heuristic_fallback_demotes_thesis_core_capability() -> None:
    result = _result({"provider": "heuristic"})
    assert _node_status_from_result("topic_model", result) == "degraded"
    assert _node_status_from_result("ner", result) == "degraded"


def test_heuristic_provider_does_not_demote_heuristic_by_design_capability() -> None:
    # entity_link, claim_strength_score etc. are heuristic by design — those
    # should remain "completed" even when provider is "heuristic".
    result = _result({"provider": "heuristic"})
    assert _node_status_from_result("entity_link", result) == "completed"
    assert _node_status_from_result("claim_strength_score", result) == "completed"
    assert _node_status_from_result("burst_detect", result) == "completed"


def test_provider_fallback_reason_demotes_thesis_core() -> None:
    result = _result({"provider": "sql", "provider_fallback_reason": "textacy not installed"})
    assert _node_status_from_result("topic_model", result) == "degraded"


def test_provider_fallback_reason_keeps_neutral_for_convenience_capability() -> None:
    result = _result({"provider": "sql", "provider_fallback_reason": "alternative path"})
    assert _node_status_from_result("sql_query_search", result) == "completed"


def test_full_success_path_is_completed() -> None:
    result = _result(
        {"provider": "textacy"},
        payload={"rows": [{"topic_id": 1, "top_terms": ["soccer"]}]},
    )
    assert _node_status_from_result("topic_model", result) == "completed"


def test_no_data_metadata_demotes_to_degraded() -> None:
    result = _result({"provider": "spacy", "no_data": True})
    assert _node_status_from_result("ner", result) == "degraded"


def test_empty_rows_demotes_thesis_core_capability_to_degraded() -> None:
    result = _result({"provider": "spacy"}, payload={"rows": []})
    assert _node_status_from_result("ner", result) == "degraded"
    assert _node_status_from_result("sentiment", result) == "degraded"


def test_empty_rows_on_convenience_capability_stays_completed() -> None:
    result = _result({"provider": "heuristic"}, payload={"rows": []})
    assert _node_status_from_result("entity_link", result) == "completed"
    assert _node_status_from_result("burst_detect", result) == "completed"


def test_thesis_core_set_is_explicit() -> None:
    # Sanity check the curated set so accidental edits trigger this test.
    assert "ner" in _HEURISTIC_FALLBACK_DEGRADES_CAPABILITY
    assert "entity_link" not in _HEURISTIC_FALLBACK_DEGRADES_CAPABILITY


def test_db_search_short_circuits_when_empty() -> None:
    # When db_search returns 0 documents, the executor's orchestrator
    # treats it as a core no-data failure and stops the run. This is what
    # prevents the 'grind through 20 empty downstream nodes' pattern.
    assert "db_search" in _CORE_NO_DATA_FAILURE_CAPABILITIES
    assert "sql_query_search" in _CORE_NO_DATA_FAILURE_CAPABILITIES
    assert "db_search" in _RETRIEVAL_BAILOUT_CAPABILITIES
    assert "sql_query_search" in _RETRIEVAL_BAILOUT_CAPABILITIES


def test_build_degraded_recovery_result_includes_required_metadata() -> None:
    rows = [{"doc_id": "d1", "score": 0.5}]
    result = _build_degraded_recovery_result(
        reason="upstream produced 0 evidence",
        capability="topic_model",
        upstream_rows=rows,
    )
    assert result.metadata["degraded"] is True
    assert result.metadata["recovery"] == "mark_degraded"
    assert result.metadata["reason"] == "upstream produced 0 evidence"
    assert result.metadata["provider"] == "recovery_advisor"
    assert result.payload["rows"] == rows
    assert any("topic_model" in caveat for caveat in result.caveats)


def test_build_degraded_recovery_result_marks_no_data_when_rows_empty() -> None:
    result = _build_degraded_recovery_result(
        reason="no upstream",
        capability="sentiment",
        upstream_rows=[],
    )
    assert result.metadata["no_data"] is True
    assert result.payload["rows"] == []
