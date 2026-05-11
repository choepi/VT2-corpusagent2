from __future__ import annotations

from typing import Any

from corpusagent2.recovery_advisor import (
    DEFAULT_MAX_ATTEMPTS,
    HARD_CAP_ATTEMPTS,
    LLMRecoveryAdvisor,
    RecoveryAction,
    is_enabled,
    max_attempts,
)


class _StubClient:
    def __init__(self, payload: dict[str, Any] | Exception) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, messages, *, model, temperature=0.0):
        self.calls.append({"messages": messages, "model": model, "temperature": temperature})
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def _advise(client: _StubClient) -> RecoveryAction:
    advisor = LLMRecoveryAdvisor(client, model="stub-model")
    return advisor.advise(
        node_id="n1",
        capability="db_search",
        tool_name="opensearch",
        inputs={"query": "test"},
        traceback="Traceback...\nValueError: boom",
        failure_category="tool_error",
        upstream_summary={"prev": "ok"},
        candidate_capabilities=["sql_query_search"],
    )


def test_advisor_returns_retry_action() -> None:
    client = _StubClient({"action": "retry_with_modified_inputs", "inputs": {"query": "broader"}, "reason": "relax filter"})
    action = _advise(client)
    assert action.action == "retry_with_modified_inputs"
    assert action.inputs == {"query": "broader"}
    assert "relax" in action.reason


def test_advisor_returns_substitute_action() -> None:
    client = _StubClient({"action": "substitute_capability", "capability": "sql_query_search", "reason": "try sql backend"})
    action = _advise(client)
    assert action.action == "substitute_capability"
    assert action.capability == "sql_query_search"


def test_advisor_rejects_substitute_without_capability() -> None:
    client = _StubClient({"action": "substitute_capability", "reason": "no target given"})
    action = _advise(client)
    assert action.action == "fail"
    assert "substitute_capability" in action.reason


def test_advisor_rejects_unknown_action() -> None:
    client = _StubClient({"action": "spawn_new_plan", "reason": "out of spec"})
    action = _advise(client)
    assert action.action == "fail"
    assert "unknown action" in action.reason


def test_advisor_handles_llm_exception() -> None:
    client = _StubClient(RuntimeError("network down"))
    action = _advise(client)
    assert action.action == "fail"
    assert "network down" in action.reason


def test_advisor_returns_mark_degraded() -> None:
    client = _StubClient({"action": "mark_degraded", "reason": "accept empty result"})
    action = _advise(client)
    assert action.action == "mark_degraded"


def test_is_enabled_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("CORPUSAGENT2_USE_LLM_RECOVERY", raising=False)
    assert is_enabled() is False


def test_is_enabled_truthy_env(monkeypatch) -> None:
    for value in ("1", "true", "yes", "on", "TRUE"):
        monkeypatch.setenv("CORPUSAGENT2_USE_LLM_RECOVERY", value)
        assert is_enabled() is True


def test_max_attempts_default(monkeypatch) -> None:
    monkeypatch.delenv("CORPUSAGENT2_LLM_RECOVERY_MAX_ATTEMPTS", raising=False)
    assert max_attempts() == DEFAULT_MAX_ATTEMPTS


def test_max_attempts_capped(monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_LLM_RECOVERY_MAX_ATTEMPTS", "99")
    assert max_attempts() == HARD_CAP_ATTEMPTS


def test_max_attempts_invalid_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_LLM_RECOVERY_MAX_ATTEMPTS", "not-a-number")
    assert max_attempts() == DEFAULT_MAX_ATTEMPTS


def test_recovery_action_to_dict_has_canonical_keys() -> None:
    action = RecoveryAction(action="mark_degraded", reason="upstream returned 0 rows")
    payload = action.to_dict()
    assert set(payload.keys()) == {"action", "inputs", "capability", "reason"}
    assert payload["action"] == "mark_degraded"
