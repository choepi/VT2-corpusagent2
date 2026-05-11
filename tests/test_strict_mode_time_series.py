"""Scientific-integrity tests for time_series_aggregate metric-source isolation.

These tests pin down the contract that prevents the cross-metric copying bug
observed in the Ronaldo vs Messi run: a metric's value can only come from its
declared source node. If the source produced no rows, the metric is null on
the output rows — never silently borrowed from an unrelated dependency.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from corpusagent2.agent_capabilities import _time_series_aggregate
from corpusagent2.tool_registry import ToolExecutionResult


@pytest.fixture(autouse=True)
def _strict_mode_clean(monkeypatch):
    for var in (
        "CORPUSAGENT2_ANALYSIS_STRICT_MODE",
        "CORPUSAGENT2_ALLOW_SILENT_FALLBACKS",
        "CORPUSAGENT2_ALLOW_PROVIDER_FALLBACK",
        "CORPUSAGENT2_FAIL_ON_REQUIRED_NODE_EMPTY",
        "CORPUSAGENT2_FAIL_ON_METRIC_SOURCE_EMPTY",
        "CORPUSAGENT2_PLOT_REQUIRE_VALID_Y",
        "CORPUSAGENT2_REQUIRE_SERIES_ASSIGNMENT",
    ):
        monkeypatch.delenv(var, raising=False)


def _ctx():
    ctx = MagicMock()
    ctx.run_id = "run_test"
    ctx.state = MagicMock()
    return ctx


def _result(rows):
    return ToolExecutionResult(payload={"rows": rows})


def test_empty_metric_source_does_not_copy_from_another_node():
    """The reproducer for the Ronaldo vs Messi bug.

    sentiment_node returned 0 rows. claim_strength_node returned 0 rows.
    similarity_node returned 0 rows. But sentiment_event_count_node has rows
    with a `count` field. The old `_metric_value` and `_time_series_source_rows`
    would let avg_sentiment / avg_claim_strength / prestige_* all pick up the
    same `count` values. After the fix, those metrics must be null.
    """
    deps = {
        "events": _result([
            {"doc_id": "d1", "published_at": "2024-01-15", "series_name": "Ronaldo", "count": 5},
            {"doc_id": "d2", "published_at": "2024-01-20", "series_name": "Ronaldo", "count": 3},
            {"doc_id": "d3", "published_at": "2024-02-01", "series_name": "Messi", "count": 7},
        ]),
        "sentiment": _result([]),
        "claim_strength": _result([]),
        "similarity": _result([]),
    }
    params = {
        "frequency": "month",
        "series_definitions": [
            {"name": "Ronaldo", "aliases": ["Ronaldo", "Cristiano Ronaldo"]},
            {"name": "Messi", "aliases": ["Messi", "Lionel Messi"]},
        ],
        "metrics": [
            {"name": "event_count", "field": "count", "source": "events", "aggregation": "sum"},
            {"name": "avg_sentiment", "field": "sentiment_score", "source": "sentiment", "aggregation": "mean"},
            {"name": "avg_claim_strength", "field": "claim_strength", "source": "claim_strength", "aggregation": "mean"},
            {"name": "prestige_positive_similarity", "field": "similarity_score", "source": "similarity", "aggregation": "mean"},
        ],
    }
    result = _time_series_aggregate(params, deps, _ctx())
    rows = result.payload["rows"]
    assert rows, "expected at least one event_count row"
    for row in rows:
        # The metric that has data shows real values.
        assert isinstance(row.get("event_count"), (int, float))
        # The empty-source metrics are explicitly null, NEVER copied from event_count.
        assert row.get("avg_sentiment") is None, f"avg_sentiment leaked: row={row}"
        assert row.get("avg_claim_strength") is None, f"avg_claim_strength leaked: row={row}"
        assert row.get("prestige_positive_similarity") is None, f"prestige leaked: row={row}"
        # And those nulls aren't equal to the populated metric (the smoking-gun pattern).
        assert row.get("avg_sentiment") != row.get("event_count")


def test_metric_diagnostics_reports_empty_sources():
    deps = {
        "events": _result([
            {"doc_id": "d1", "published_at": "2024-01-15", "series_name": "Ronaldo", "count": 5},
        ]),
        "sentiment": _result([]),
    }
    params = {
        "frequency": "month",
        "series_definitions": [{"name": "Ronaldo", "aliases": ["Ronaldo"]}],
        "metrics": [
            {"name": "event_count", "field": "count", "source": "events"},
            {"name": "avg_sentiment", "field": "sentiment_score", "source": "sentiment", "aggregation": "mean"},
        ],
    }
    result = _time_series_aggregate(params, deps, _ctx())
    diag = result.metadata.get("metric_diagnostics", {})
    assert diag["avg_sentiment"]["empty_source"] is True
    assert diag["avg_sentiment"]["rows_seen"] == 0
    assert diag["event_count"]["empty_source"] is False
    assert diag["event_count"]["values_extracted"] == 1
    assert any("avg_sentiment" in w for w in result.payload.get("warnings", []))


def test_separate_working_sets_produce_separate_series(monkeypatch):
    """Ronaldo and Messi rows must NOT collapse into __all__."""
    monkeypatch.setenv("CORPUSAGENT2_REQUIRE_SERIES_ASSIGNMENT", "true")
    deps = {
        "events": _result([
            {"doc_id": "d1", "published_at": "2024-01-15", "series_name": "Ronaldo", "count": 5},
            {"doc_id": "d2", "published_at": "2024-01-20", "series_name": "Messi", "count": 3},
            {"doc_id": "d3", "published_at": "2024-02-01", "series_name": "Ronaldo", "count": 7},
        ]),
    }
    params = {
        "frequency": "month",
        "series_definitions": [
            {"name": "Ronaldo", "aliases": ["Ronaldo"]},
            {"name": "Messi", "aliases": ["Messi"]},
        ],
        "metrics": [{"name": "event_count", "field": "count", "source": "events"}],
    }
    result = _time_series_aggregate(params, deps, _ctx())
    rows = result.payload["rows"]
    series_names = {row.get("series_name") for row in rows}
    assert "Ronaldo" in series_names and "Messi" in series_names
    assert "__all__" not in series_names


def test_strict_fail_on_metric_source_empty_sets_no_data(monkeypatch):
    monkeypatch.setenv("CORPUSAGENT2_ANALYSIS_STRICT_MODE", "true")
    monkeypatch.setenv("CORPUSAGENT2_FAIL_ON_METRIC_SOURCE_EMPTY", "true")
    deps = {
        "events": _result([
            {"doc_id": "d1", "published_at": "2024-01-15", "series_name": "Ronaldo", "count": 5},
        ]),
        "sentiment": _result([]),
    }
    params = {
        "frequency": "month",
        "series_definitions": [{"name": "Ronaldo", "aliases": ["Ronaldo"]}],
        "metrics": [
            {"name": "event_count", "field": "count", "source": "events"},
            {"name": "avg_sentiment", "field": "sentiment_score", "source": "sentiment", "aggregation": "mean"},
        ],
    }
    result = _time_series_aggregate(params, deps, _ctx())
    assert result.metadata.get("no_data") is True
    assert "avg_sentiment" in result.metadata.get("no_data_reason", "")


def test_value_field_missing_is_skipped_not_fabricated():
    """Count-based path: when value_field is set and missing, row is skipped, not promoted to 1.0."""
    deps = {
        "events": _result([
            {"doc_id": "d1", "published_at": "2024-01-15", "entity": "Ronaldo"},  # no 'count' field
            {"doc_id": "d2", "published_at": "2024-01-20", "entity": "Ronaldo", "count": 3},
        ]),
    }
    params = {
        "frequency": "month",
        "value_field": "count",
        "group_by": "entity",
    }
    result = _time_series_aggregate(params, deps, _ctx())
    rows = result.payload["rows"]
    # Only d2 (which has count=3) contributes. d1 with no count is skipped, not fabricated to 1.
    total = sum(row.get("count", 0) for row in rows)
    assert total == 3, f"expected 3 (only d2 contributes), got {total}: rows={rows}"
