from __future__ import annotations

from corpusagent2.tool_registry import ToolRegistry

from .helpers import StaticAdapter


def test_registry_prefers_deterministic_low_cost_high_priority() -> None:
    registry = ToolRegistry()
    registry.register(
        StaticAdapter(
            tool_name="candidate_high_cost",
            capability="analyze.entity_trend",
            deterministic=True,
            cost_class="high",
            priority=90,
        )
    )
    registry.register(
        StaticAdapter(
            tool_name="candidate_low_cost",
            capability="analyze.entity_trend",
            deterministic=True,
            cost_class="low",
            priority=10,
        )
    )
    registry.register(
        StaticAdapter(
            tool_name="candidate_non_det",
            capability="analyze.entity_trend",
            deterministic=False,
            cost_class="low",
            priority=100,
        )
    )
    resolution = registry.resolve("analyze.entity_trend", context={}, params={})
    assert resolution.spec.tool_name == "candidate_low_cost"
    assert "Other available implementations" in resolution.reason


def test_registry_reason_mentions_fallback_of() -> None:
    registry = ToolRegistry()
    registry.register(
        StaticAdapter(
            tool_name="fallback_tool",
            capability="nlp.tokenize",
            deterministic=True,
            cost_class="low",
            priority=1,
            fallback_of="primary_tool",
        )
    )
    resolution = registry.resolve("nlp.tokenize", context={}, params={})
    assert resolution.spec.fallback_of == "primary_tool"
    assert "fallback_of=primary_tool" in resolution.reason
