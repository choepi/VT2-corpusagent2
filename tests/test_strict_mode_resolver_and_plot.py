"""Strict-mode tests for the shared node resolver and plot guards.

These pin down:
  - working_set_ref dereferences to full documents (not preview rows).
  - empty upstream → sentiment / embeddings / similarity / claim tools degrade
    with explicit reason, not "completed with empty rows".
  - plot_artifact refuses to render when y is all-null in strict mode.
  - plot_artifact refuses comparative-series collapse to __all__.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from pathlib import Path

import pytest

from corpusagent2.tool_registry import ToolExecutionResult
from corpusagent2.node_resolution import (
    extract_actual_vs_preview_counts,
    make_tool_result,
    resolve_documents_from_node,
    resolve_rows_from_node,
    resolve_working_set_ref,
)


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


def _ctx(documents=None, artifacts_dir=None):
    ctx = MagicMock()
    ctx.run_id = "run_test"
    ctx.state = MagicMock()
    if artifacts_dir:
        ctx.artifacts_dir = Path(artifacts_dir)
    if documents is not None:

        def fetch(run_id, label, limit, offset):
            return documents[offset : offset + limit]

        store = MagicMock()
        store.fetch_working_set_documents = fetch
        ctx.working_store = store
    ctx.cancel_requested = None
    ctx.runtime = None
    return ctx


# ---------- resolver helpers -------------------------------------------------


def test_resolve_rows_from_node_returns_only_named_node():
    deps = {
        "a": ToolExecutionResult(payload={"rows": [{"x": 1}]}),
        "b": ToolExecutionResult(payload={"rows": [{"x": 2}]}),
    }
    assert resolve_rows_from_node(deps, "a") == [{"x": 1}]
    assert resolve_rows_from_node(deps, "b") == [{"x": 2}]
    assert resolve_rows_from_node(deps, "missing") == []


def test_resolve_working_set_ref_finds_from_payload():
    deps = {
        "fetch": ToolExecutionResult(payload={"rows": [], "working_set_ref": "ws_42"}),
    }
    assert resolve_working_set_ref(deps) == "ws_42"


def test_extract_counts_distinguishes_preview_from_actual():
    payload = {"rows": [{"doc_id": "a"}, {"doc_id": "b"}], "working_set_document_count": 68}
    counts = extract_actual_vs_preview_counts(payload)
    assert counts["previewed_input_count"] == 2
    assert counts["input_document_count"] == 68
    assert counts["results_truncated"] is True


def test_resolve_documents_dereferences_working_set_ref():
    """When upstream only has preview rows + a working_set_ref, the resolver
    must materialize full docs from the store — that's the fix for the
    'sentence_split: 50 input docs, rows: []' pattern."""
    full_docs = [
        {"doc_id": "d1", "title": "T1", "text": "Full text 1", "published_at": "2024-01-15"},
        {"doc_id": "d2", "title": "T2", "text": "Full text 2", "published_at": "2024-01-20"},
    ]
    ctx = _ctx(documents=full_docs)
    deps = {
        "fetch": ToolExecutionResult(
            payload={
                # preview rows have NO text — the bug case
                "rows": [{"doc_id": "d1", "title": "T1"}, {"doc_id": "d2", "title": "T2"}],
                "working_set_ref": "ws_test",
                "documents_truncated": True,
            }
        )
    }
    docs, diag = resolve_documents_from_node(deps, ctx, text_field="text")
    assert len(docs) == 2
    assert docs[0]["text"] == "Full text 1"
    assert diag["source"] == "working_set_ref"
    assert diag["working_set_ref"] == "ws_test"


def test_resolve_documents_uses_inline_rows_when_text_present():
    deps = {
        "fetch": ToolExecutionResult(
            payload={"rows": [{"doc_id": "d1", "text": "hi", "title": "T"}]}
        )
    }
    docs, diag = resolve_documents_from_node(deps, _ctx())
    assert docs == [{"doc_id": "d1", "text": "hi", "title": "T"}]
    assert diag["source"] == "inline_rows"


def test_make_tool_result_sets_no_data_metadata():
    result = make_tool_result(status="no_data", rows=[], reason="upstream_sentence_split_empty")
    assert result.metadata["no_data"] is True
    assert result.metadata["no_data_reason"] == "upstream_sentence_split_empty"
    assert result.payload["rows"] == []


def test_make_tool_result_marks_degraded():
    result = make_tool_result(
        status="degraded",
        rows=[{"x": 1}],
        reason="provider_fallback",
        provider="heuristic",
        warnings=["fell back to heuristic"],
    )
    assert result.metadata["degraded"] is True
    assert result.metadata["provider"] == "heuristic"
    assert "fell back to heuristic" in result.payload["warnings"]


# ---------- claim_strength_score / claim_span_extract degradation ----------


def test_claim_strength_score_returns_no_data_when_no_spans():
    """Claim strength must never fabricate values when upstream spans are empty."""
    from corpusagent2.agent_capabilities import _claim_strength_score

    result = _claim_strength_score({}, {"spans": ToolExecutionResult(payload={"rows": []})}, _ctx())
    assert result.metadata["no_data"] is True
    assert result.metadata["no_data_reason"] == "no_claim_spans"
    assert result.payload["rows"] == []


def test_claim_span_extract_degrades_when_no_documents():
    from corpusagent2.agent_capabilities import _claim_span_extract

    result = _claim_span_extract({}, {"docs": ToolExecutionResult(payload={"rows": []})}, _ctx())
    assert result.metadata["degraded"] is True
    assert result.metadata["no_data_reason"] == "no_sentence_rows"


# ---------- plot_artifact strict-mode guards --------------------------------


def test_plot_refuses_all_null_y_in_strict_mode(monkeypatch, tmp_path):
    """The killer guard: plot must not silently draw bars from another column
    when the requested y is null/missing for every row."""
    monkeypatch.setenv("CORPUSAGENT2_ANALYSIS_STRICT_MODE", "true")
    monkeypatch.setenv("CORPUSAGENT2_PLOT_REQUIRE_VALID_Y", "true")
    from corpusagent2.agent_capabilities import _plot_artifact

    deps = {
        "series": ToolExecutionResult(
            payload={
                "rows": [
                    {"time_bin": "2024-01", "series_name": "Ronaldo", "avg_sentiment": None, "count": 5},
                    {"time_bin": "2024-02", "series_name": "Ronaldo", "avg_sentiment": None, "count": 7},
                ]
            }
        )
    }
    ctx = _ctx(artifacts_dir=tmp_path)
    result = _plot_artifact(
        {"x": "time_bin", "y": "avg_sentiment", "series": "series_name"},
        deps,
        ctx,
    )
    assert result.metadata["no_data"] is True
    assert "plot_y_all_null" in result.metadata["no_data_reason"]
    assert result.metadata["resolved_y"] == "avg_sentiment"
    # No artifact should have been written.
    assert not list(tmp_path.glob("plots/*.png"))


def test_plot_refuses_collapsed_all_series_in_strict_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("CORPUSAGENT2_ANALYSIS_STRICT_MODE", "true")
    monkeypatch.setenv("CORPUSAGENT2_REQUIRE_SERIES_ASSIGNMENT", "true")
    from corpusagent2.agent_capabilities import _plot_artifact

    deps = {
        "series": ToolExecutionResult(
            payload={
                "rows": [
                    {"time_bin": "2024-01", "series_name": "__all__", "count": 5},
                    {"time_bin": "2024-02", "series_name": "__all__", "count": 7},
                ]
            }
        )
    }
    ctx = _ctx(artifacts_dir=tmp_path)
    result = _plot_artifact(
        {
            "x": "time_bin",
            "y": "count",
            "series": "series_name",
            "series_definitions": [
                {"name": "Ronaldo", "aliases": ["Ronaldo"]},
                {"name": "Messi", "aliases": ["Messi"]},
            ],
        },
        deps,
        ctx,
    )
    assert result.metadata["no_data"] is True
    assert result.metadata["no_data_reason"] == "comparative_series_collapsed"


def test_plot_non_strict_still_works_with_valid_data(monkeypatch, tmp_path):
    """Sanity: when y values are valid, strict mode does not block plotting."""
    monkeypatch.setenv("CORPUSAGENT2_ANALYSIS_STRICT_MODE", "true")
    from corpusagent2.agent_capabilities import _plot_artifact

    deps = {
        "series": ToolExecutionResult(
            payload={
                "rows": [
                    {"time_bin": "2024-01", "series_name": "Ronaldo", "count": 5},
                    {"time_bin": "2024-02", "series_name": "Ronaldo", "count": 7},
                ]
            }
        )
    }
    ctx = _ctx(artifacts_dir=tmp_path)
    ctx.cancel_requested = None
    result = _plot_artifact(
        {"x": "time_bin", "y": "count", "series": "series_name"},
        deps,
        ctx,
    )
    # Should not be refused with no_data on the strict checks.
    no_data_reason = (result.metadata or {}).get("no_data_reason", "")
    assert "plot_y_all_null" not in no_data_reason
    assert "comparative_series_collapsed" not in no_data_reason
