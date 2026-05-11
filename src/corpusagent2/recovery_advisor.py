"""Bounded LLM-assisted recovery for failed plan nodes.

When a node fails after all in-loop retries have been exhausted, the
optional ``LLMRecoveryAdvisor`` consults an LLM for a *bounded* recovery
action that operates ONLY on the current node. The advisor is forbidden
from:

- adding or removing nodes from the PlanDAG
- changing dependency edges
- spawning parallel plans

This is the Reflexion pattern, intentionally narrowed. The original
plan stays the source of truth; the advisor is a per-node recovery hook,
not a re-planner.

Feature flag (off by default): ``CORPUSAGENT2_USE_LLM_RECOVERY=true``.
Per-node cap: ``CORPUSAGENT2_LLM_RECOVERY_MAX_ATTEMPTS`` (default 2,
hard-capped to 5).

Returned actions are typed via ``RecoveryAction``:
- ``retry_with_modified_inputs(inputs)`` — try again with adjusted inputs
- ``substitute_capability(capability)`` — swap to a same-category tool
- ``mark_degraded(reason)`` — accept partial data, continue downstream
- ``fail(reason)`` — give up
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Literal, Protocol


HARD_CAP_ATTEMPTS = 5
DEFAULT_MAX_ATTEMPTS = 2

RecoveryActionType = Literal[
    "retry_with_modified_inputs",
    "substitute_capability",
    "mark_degraded",
    "fail",
]


@dataclass(slots=True)
class RecoveryAction:
    action: RecoveryActionType
    inputs: dict[str, Any] = field(default_factory=dict)
    capability: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "inputs": self.inputs,
            "capability": self.capability,
            "reason": self.reason,
        }


class _LLMClientLike(Protocol):
    def complete_json(self, messages: list[dict[str, str]], *, model: str, temperature: float = 0.0) -> dict[str, Any]:
        ...


def is_enabled() -> bool:
    return os.getenv("CORPUSAGENT2_USE_LLM_RECOVERY", "").strip().lower() in {"1", "true", "yes", "on"}


def max_attempts() -> int:
    raw = os.getenv("CORPUSAGENT2_LLM_RECOVERY_MAX_ATTEMPTS", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_MAX_ATTEMPTS
    except ValueError:
        value = DEFAULT_MAX_ATTEMPTS
    return max(0, min(HARD_CAP_ATTEMPTS, value))


def _system_prompt() -> str:
    return (
        "You are a recovery advisor for a deterministic data-analysis agent. "
        "A node in the existing plan has failed. You may suggest ONE of four "
        "bounded actions for THIS node only. You must NOT add nodes, remove "
        "nodes, change dependencies, or spawn parallel plans. The current "
        "plan stays unchanged.\n\n"
        "Respond with a single JSON object: "
        '{"action": "retry_with_modified_inputs"|"substitute_capability"|"mark_degraded"|"fail", '
        '"inputs": {<dict, optional>}, "capability": "<str, optional>", '
        '"reason": "<short reason, required>"}. '
        "Choose retry_with_modified_inputs only if you believe a small "
        "input change (e.g. relaxed filter, smaller k, retry without "
        "rerank) will fix it. Choose substitute_capability only if a "
        "same-category alternative exists in the candidates list. Choose "
        "mark_degraded if the node can return partial data and downstream "
        "nodes can still work. Choose fail when no recovery is sensible."
    )


def _user_prompt(
    *,
    node_id: str,
    capability: str,
    tool_name: str,
    inputs: dict[str, Any],
    traceback: str,
    failure_category: str,
    upstream_summary: dict[str, Any],
    candidate_capabilities: list[str],
) -> str:
    body = {
        "failed_node": {
            "node_id": node_id,
            "capability": capability,
            "tool_name": tool_name,
            "inputs": inputs,
            "failure_category": failure_category,
            "traceback_tail": "\n".join(traceback.strip().splitlines()[-20:]),
        },
        "upstream_summary": upstream_summary,
        "candidate_same_category_capabilities": candidate_capabilities,
    }
    return json.dumps(body, ensure_ascii=False, indent=2)


def _coerce_action(payload: dict[str, Any]) -> RecoveryAction:
    raw_action = str(payload.get("action", "")).strip().lower()
    if raw_action not in {"retry_with_modified_inputs", "substitute_capability", "mark_degraded", "fail"}:
        return RecoveryAction(action="fail", reason=f"advisor returned unknown action: {payload.get('action')!r}")
    inputs_raw = payload.get("inputs")
    inputs = inputs_raw if isinstance(inputs_raw, dict) else {}
    capability = str(payload.get("capability", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    if raw_action == "substitute_capability" and not capability:
        return RecoveryAction(action="fail", reason="advisor selected substitute_capability without a target capability")
    return RecoveryAction(action=raw_action, inputs=inputs, capability=capability, reason=reason)


class LLMRecoveryAdvisor:
    """Advisor that asks an LLM how to recover from a single-node failure.

    The advisor never mutates the plan. It returns a structured action
    that the executor applies in-place.
    """

    def __init__(
        self,
        client: _LLMClientLike,
        *,
        model: str,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    def advise(
        self,
        *,
        node_id: str,
        capability: str,
        tool_name: str,
        inputs: dict[str, Any],
        traceback: str,
        failure_category: str,
        upstream_summary: dict[str, Any] | None = None,
        candidate_capabilities: list[str] | None = None,
    ) -> RecoveryAction:
        messages = [
            {"role": "system", "content": _system_prompt()},
            {
                "role": "user",
                "content": _user_prompt(
                    node_id=node_id,
                    capability=capability,
                    tool_name=tool_name,
                    inputs=inputs,
                    traceback=traceback,
                    failure_category=failure_category,
                    upstream_summary=upstream_summary or {},
                    candidate_capabilities=candidate_capabilities or [],
                ),
            },
        ]
        try:
            payload = self._client.complete_json(messages, model=self._model, temperature=self._temperature)
        except Exception as exc:
            return RecoveryAction(action="fail", reason=f"advisor LLM call failed: {exc}")
        return _coerce_action(payload)
