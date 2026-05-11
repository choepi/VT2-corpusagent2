"""Strict-mode tests for the final-answer signal partitioning.

These verify that the synthesis path:
- Partitions node outputs into valid_signals / degraded_signals / unavailable_methods.
- Forces unavailable methods into the final answer's `unsupported_parts`.
- Surfaces empty-metric-source warnings as caveats.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from corpusagent2.agent_executor import AgentExecutionSnapshot
from corpusagent2.agent_models import (
    AgentNodeExecutionRecord,
    AgentRunState,
    FinalAnswerPayload,
)
from corpusagent2.agent_runtime import MagicBoxOrchestrator
from corpusagent2.tool_registry import ToolExecutionResult


@pytest.fixture(autouse=True)
def _strict_mode_clean(monkeypatch):
    for var in (
        "CORPUSAGENT2_ANALYSIS_STRICT_MODE",
        "CORPUSAGENT2_ALLOW_SILENT_FALLBACKS",
        "CORPUSAGENT2_FAIL_ON_METRIC_SOURCE_EMPTY",
        "CORPUSAGENT2_PLOT_REQUIRE_VALID_Y",
        "CORPUSAGENT2_REQUIRE_SERIES_ASSIGNMENT",
    ):
        monkeypatch.delenv(var, raising=False)


def _record(node_id, capability, status="completed"):
    return AgentNodeExecutionRecord(node_id=node_id, capability=capability, status=status)


def _snapshot(results, records):
    return AgentExecutionSnapshot(
        node_records=records,
        node_results=results,
        failures=[],
        provenance_records=[],
        selected_docs=[],
        status="completed",
    )


def _synthesizer():
    syn = MagicBoxOrchestrator.__new__(MagicBoxOrchestrator)
    syn.llm_client = None
    return syn


def test_derive_strict_signals_partitions_correctly():
    results = {
        "n1": ToolExecutionResult(
            payload={"rows": [{"x": 1}, {"x": 2}]},
            metadata={"provider": "events"},
        ),
        "n2": ToolExecutionResult(
            payload={"rows": []},
            metadata={"no_data": True, "no_data_reason": "could_not_resolve_documents"},
        ),
        "n3": ToolExecutionResult(
            payload={"rows": [{"x": 3}]},
            metadata={"degraded": True, "reason": "fell back to heuristic"},
        ),
    }
    records = [
        _record("n1", "time_series_aggregate"),
        _record("n2", "sentiment", status="degraded"),
        _record("n3", "topic_model", status="degraded"),
    ]
    syn = _synthesizer()
    signals = syn._derive_strict_signals(_snapshot(results, records))

    valid_caps = {entry["capability"] for entry in signals["valid_signals"]}
    degraded_caps = {entry["capability"] for entry in signals["degraded_signals"]}
    unavailable_caps = {entry["capability"] for entry in signals["unavailable_methods"]}

    assert "time_series_aggregate" in valid_caps
    assert "topic_model" in degraded_caps
    assert "sentiment" in unavailable_caps


def test_derive_strict_signals_extracts_null_metric_warnings():
    results = {
        "n5": ToolExecutionResult(
            payload={"rows": [{"series_name": "Ronaldo", "event_count": 5, "avg_sentiment": None}]},
            metadata={
                "metric_diagnostics": {
                    "event_count": {"source": "events", "rows_seen": 1, "empty_source": False},
                    "avg_sentiment": {"source": "sentiment", "rows_seen": 0, "empty_source": True},
                },
            },
        ),
    }
    records = [_record("n5", "time_series_aggregate")]
    syn = _synthesizer()
    signals = syn._derive_strict_signals(_snapshot(results, records))

    assert "n5" in signals["null_metrics_by_node"]
    assert "avg_sentiment" in signals["null_metrics_by_node"]["n5"]
    assert any("avg_sentiment" in w for w in signals["warnings"])


def test_guardrail_forces_unavailable_into_unsupported_parts():
    """Even if the LLM ignores the prompt section, the guardrail must inject
    the unavailable methods into unsupported_parts."""
    results = {
        "n2": ToolExecutionResult(
            payload={"rows": []},
            metadata={"no_data": True, "no_data_reason": "could_not_resolve_documents"},
        ),
    }
    records = [_record("n2", "sentiment", status="degraded")]
    syn = _synthesizer()
    state = AgentRunState(question="Ronaldo vs Messi", rewritten_question="Ronaldo vs Messi")
    answer = FinalAnswerPayload(
        answer_text="Both players were covered extensively.",
        evidence_items=[],
        artifacts_used=[],
        unsupported_parts=[],
        caveats=[],
        claim_verdicts=[],
    )
    out = syn._apply_answer_guardrails(state, _snapshot(results, records), answer)
    joined = " ".join(out.unsupported_parts)
    assert "sentiment" in joined
    assert "could_not_resolve_documents" in joined


def test_guardrail_promotes_degraded_reason_to_caveat():
    results = {
        "n3": ToolExecutionResult(
            payload={"rows": [{"topic_id": 1, "top_terms": ["soccer"]}]},
            metadata={"degraded": True, "reason": "textacy unavailable; heuristic used"},
        ),
    }
    records = [_record("n3", "topic_model", status="degraded")]
    syn = _synthesizer()
    state = AgentRunState(question="topic trends", rewritten_question="topic trends")
    answer = FinalAnswerPayload(
        answer_text="Topics found.",
        evidence_items=[],
        artifacts_used=[],
        unsupported_parts=[],
        caveats=[],
        claim_verdicts=[],
    )
    out = syn._apply_answer_guardrails(state, _snapshot(results, records), answer)
    assert any("topic_model" in c and "degraded mode" in c for c in out.caveats)
