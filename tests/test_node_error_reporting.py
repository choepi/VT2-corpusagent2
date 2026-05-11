from __future__ import annotations

from corpusagent2.agent_executor import _classify_exception, _truncate_for_snapshot
from corpusagent2.agent_models import AgentFailure


def test_agent_failure_carries_diagnostic_fields() -> None:
    failure = AgentFailure(
        node_id="n1",
        capability="db_search",
        error_type="ValueError",
        message="boom",
        traceback="Traceback (most recent call last):\n  ...\nValueError: boom",
        input_snapshot={"query": "test"},
        retry_count=2,
        category="tool_error",
    )
    payload = failure.to_dict()
    assert payload["traceback"].startswith("Traceback")
    assert payload["input_snapshot"] == {"query": "test"}
    assert payload["retry_count"] == 2
    assert payload["category"] == "tool_error"


def test_classify_exception_categorizes_known_signals() -> None:
    assert _classify_exception(TimeoutError("request timed out")) == "timeout"
    assert _classify_exception(KeyError("doc_id")) == "missing_input"
    assert _classify_exception(RuntimeError("OpenAI rate limit exceeded")) == "llm_error"
    assert _classify_exception(ConnectionError("DNS resolve failed")) == "network_error"
    assert _classify_exception(ValueError("no_data after filter")) == "data_empty"
    assert _classify_exception(ValueError("unexpected internal state")) == "tool_error"


def test_truncate_for_snapshot_compacts_oversized_inputs() -> None:
    big = {"text": "x" * 10000}
    snapshot = _truncate_for_snapshot(big, max_chars=200)
    assert isinstance(snapshot, dict)
    assert snapshot.get("_truncated") is True
    assert "_preview" in snapshot


def test_truncate_for_snapshot_preserves_small_inputs() -> None:
    payload = {"a": 1, "b": [1, 2, 3]}
    snapshot = _truncate_for_snapshot(payload)
    assert snapshot == payload
