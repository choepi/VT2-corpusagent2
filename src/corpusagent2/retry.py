"""Retry policy helpers for long-running plan executions.

The agent executor previously hardcoded a 2-attempt retry loop with no
backoff. For multi-hour runs that need to ride out transient LLM/network
hiccups, that is both too aggressive (it retries non-retriable errors)
and too timid (it gives up after one extra attempt with zero delay).

This module centralizes:
- max-attempts and base-delay defaults (configurable via env vars)
- exponential backoff with optional jitter
- classification of which error categories are worth retrying

Tools that catch their own retriable errors should keep doing so. This
helper sits one level up, around tool invocation, so a network blip in
the middle of a 6-hour run does not abort the whole plan.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Iterable


DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY_S = 1.0
DEFAULT_MAX_DELAY_S = 30.0
DEFAULT_TIMEOUT_S = 300.0

# Error categories (matching agent_executor._classify_exception output)
# that are worth retrying. Permanent errors (missing_input, tool_error
# from a bug in the tool itself) should NOT be retried.
RETRIABLE_CATEGORIES: frozenset[str] = frozenset({"timeout", "network_error", "llm_error"})


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(slots=True, frozen=True)
class RetryPolicy:
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_delay_s: float = DEFAULT_BASE_DELAY_S
    max_delay_s: float = DEFAULT_MAX_DELAY_S
    timeout_s: float = DEFAULT_TIMEOUT_S
    jitter: bool = True

    @classmethod
    def from_env(cls) -> "RetryPolicy":
        return cls(
            max_attempts=max(1, _env_int("CORPUSAGENT2_NODE_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS)),
            base_delay_s=max(0.0, _env_float("CORPUSAGENT2_NODE_RETRY_BASE_DELAY_S", DEFAULT_BASE_DELAY_S)),
            max_delay_s=max(0.0, _env_float("CORPUSAGENT2_NODE_RETRY_MAX_DELAY_S", DEFAULT_MAX_DELAY_S)),
            timeout_s=max(1.0, _env_float("CORPUSAGENT2_NODE_DEFAULT_TIMEOUT_S", DEFAULT_TIMEOUT_S)),
            jitter=os.getenv("CORPUSAGENT2_NODE_RETRY_JITTER", "1").strip().lower() not in {"0", "false", "no", "off"},
        )

    def is_retriable(self, category: str, retriable_categories: Iterable[str] | None = None) -> bool:
        allowed = frozenset(retriable_categories) if retriable_categories is not None else RETRIABLE_CATEGORIES
        return str(category).strip().lower() in allowed

    def compute_delay_s(self, attempt: int) -> float:
        """Exponential backoff: base * 2**(attempt-1), capped at max_delay, optionally jittered.

        Attempts are 1-indexed. The delay returned is the wait BEFORE the
        next attempt; for attempt=1 (the first try) the caller should not
        sleep at all.
        """
        if attempt <= 1:
            return 0.0
        exponent = min(attempt - 1, 10)
        raw = self.base_delay_s * (2 ** (exponent - 1))
        capped = min(self.max_delay_s, raw)
        if not self.jitter:
            return capped
        # Equal jitter: half deterministic + half random within range.
        return capped / 2.0 + random.random() * (capped / 2.0)
