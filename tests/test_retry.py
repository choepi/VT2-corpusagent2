from __future__ import annotations

from corpusagent2.retry import (
    DEFAULT_BASE_DELAY_S,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_DELAY_S,
    RetryPolicy,
)


def test_retry_policy_defaults() -> None:
    policy = RetryPolicy()
    assert policy.max_attempts == DEFAULT_MAX_ATTEMPTS
    assert policy.base_delay_s == DEFAULT_BASE_DELAY_S
    assert policy.max_delay_s == DEFAULT_MAX_DELAY_S


def test_retry_policy_from_env_reads_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_NODE_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("CORPUSAGENT2_NODE_RETRY_BASE_DELAY_S", "2.5")
    monkeypatch.setenv("CORPUSAGENT2_NODE_RETRY_MAX_DELAY_S", "60")
    monkeypatch.setenv("CORPUSAGENT2_NODE_DEFAULT_TIMEOUT_S", "900")
    monkeypatch.setenv("CORPUSAGENT2_NODE_RETRY_JITTER", "0")

    policy = RetryPolicy.from_env()
    assert policy.max_attempts == 5
    assert policy.base_delay_s == 2.5
    assert policy.max_delay_s == 60.0
    assert policy.timeout_s == 900.0
    assert policy.jitter is False


def test_retry_policy_first_attempt_has_no_delay() -> None:
    policy = RetryPolicy(jitter=False)
    assert policy.compute_delay_s(1) == 0.0


def test_retry_policy_exponential_backoff_caps_at_max(monkeypatch) -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=10.0, jitter=False)
    assert policy.compute_delay_s(2) == 1.0
    assert policy.compute_delay_s(3) == 2.0
    assert policy.compute_delay_s(4) == 4.0
    assert policy.compute_delay_s(10) == 10.0


def test_retry_policy_classifies_retriable_categories() -> None:
    policy = RetryPolicy()
    assert policy.is_retriable("timeout") is True
    assert policy.is_retriable("network_error") is True
    assert policy.is_retriable("llm_error") is True
    assert policy.is_retriable("missing_input") is False
    assert policy.is_retriable("tool_error") is False
    assert policy.is_retriable("data_empty") is False


def test_retry_policy_custom_retriable_set() -> None:
    policy = RetryPolicy()
    assert policy.is_retriable("tool_error", retriable_categories={"tool_error"}) is True
