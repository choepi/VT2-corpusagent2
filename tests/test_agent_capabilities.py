from __future__ import annotations

import base64
import json
from pathlib import Path

import corpusagent2.retrieval as retrieval
import pandas as pd

from corpusagent2 import agent_capabilities
from corpusagent2.agent_capabilities import (
    AgentExecutionContext,
    _db_search,
    _fetch_documents,
    _join_external_series,
    _sql_query_search,
    _time_series_aggregate,
    _topic_model,
)
from corpusagent2.python_runner_service import PythonRunnerResult, SandboxArtifact
from corpusagent2.tool_registry import ToolExecutionResult


class _ExplodingStore:
    def fetch_documents(self, doc_ids):
        raise RuntimeError("store should not have been called")


class _FallbackStore:
    def fetch_documents(self, doc_ids):
        raise RuntimeError("transient postgres abort")


class _RuntimeWithDocs:
    def load_docs(self, doc_ids):
        return pd.DataFrame(
            [
                {
                    "doc_id": "doc-1",
                    "title": "Sample",
                    "text": "Document text",
                    "published_at": "2022-02-20",
                    "source": "example.com",
                }
            ]
        )


class _RuntimeWithMetadata:
    def __init__(self, rows):
        self._df = pd.DataFrame(rows)

    def load_metadata(self):
        return self._df.copy()


class _EmptySearchBackend:
    def search(self, **kwargs):
        return []


class _OffTopicSearchBackend:
    def search(self, **kwargs):
        return [
            {
                "doc_id": "fb-1",
                "title": "Facebook privacy pressure",
                "snippet": "Cambridge Analytica and Facebook privacy problems dominated the coverage.",
                "outlet": "Reuters",
                "date": "2018-03-20",
                "score": 0.8,
            }
        ]


class _YearAwareSearchBackend:
    def search(self, **kwargs):
        date_from = str(kwargs.get("date_from", ""))
        date_to = str(kwargs.get("date_to", ""))
        rows = [
            {"doc_id": "a-2016", "title": "Amazon growth 2016", "snippet": "Amazon growth story", "outlet": "Reuters", "date": "2016-03-01", "score": 9.0},
            {"doc_id": "dup-2018-1", "title": "Amazon privacy scandal", "snippet": "Same syndicated text", "outlet": "WireA", "date": "2018-03-20", "score": 8.5},
            {"doc_id": "dup-2018-2", "title": "Amazon privacy scandal", "snippet": "Same syndicated text", "outlet": "WireB", "date": "2018-03-20", "score": 8.4},
            {"doc_id": "a-2019", "title": "Amazon regulation 2019", "snippet": "Amazon regulation focus", "outlet": "FT", "date": "2019-04-11", "score": 7.2},
        ]
        filtered = []
        for row in rows:
            date = str(row["date"])
            if date_from and date < date_from:
                continue
            if date_to and date > date_to:
                continue
            filtered.append(dict(row))
        return filtered[: int(kwargs.get("top_k", 20))]


class _FakePythonRunner:
    def __init__(self, rows):
        self.rows = rows

    def run(self, code: str, inputs_json: dict):
        payload = {"results": self.rows}
        artifact = SandboxArtifact(
            name="sandbox_retrieval.json",
            mime="application/json",
            bytes_b64=base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii"),
        )
        return PythonRunnerResult(stdout="", stderr="", artifacts=[artifact], exit_code=0)


def test_sql_fallback_store_normalizes_windows_pg_dsn(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(retrieval.os, "name", "nt", raising=False)
    monkeypatch.setenv("CORPUSAGENT2_PG_DSN", "postgresql://corpus:corpus@localhost:5432/corpus_db")

    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=_ExplodingStore(),
        runtime=None,
    )

    store = agent_capabilities._sql_fallback_store(context)

    assert store is not None
    assert "localhost" not in store.dsn.lower()
    assert "127.0.0.1" in store.dsn


def test_fetch_documents_skips_store_when_no_doc_ids() -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=Path("."),
        search_backend=None,
        working_store=_ExplodingStore(),
        runtime=None,
    )
    deps = {"search": ToolExecutionResult(payload={"results": []})}

    result = _fetch_documents({}, deps, context)

    assert result.payload["documents"] == []
    assert result.caveats == []


def test_db_search_uses_sql_fallback_when_primary_search_is_empty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        agent_capabilities,
        "_sql_search_rows",
        lambda **kwargs: [
            {
                "doc_id": "btc-1",
                "title": "Bitcoin outlook",
                "snippet": "Bitcoin fell while Colgate-Palmolive was barely mentioned.",
                "outlet": "FT",
                "date": "2021-01-10",
                "score": 3.0,
                "score_display": "1",
                "retrieval_mode": "sql",
                "score_components": {"sql": 3.0},
            }
        ],
    )
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_EmptySearchBackend(),
        working_store=_ExplodingStore(),
        runtime=None,
    )

    result = _db_search({"query": "How did Bitcoin and Colgate-Palmolive co-move?", "top_k": 5}, {}, context)

    assert result.payload["results"]
    assert result.payload["results"][0]["retrieval_mode"] == "sql"


def test_db_search_discards_off_topic_rows_when_query_entities_do_not_match(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agent_capabilities, "_sql_search_rows", lambda **kwargs: [])
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_OffTopicSearchBackend(),
        working_store=_ExplodingStore(),
        runtime=None,
    )

    result = _db_search(
        {"query": "Did Bitcoin price movements correspond to media sentiment toward Colgate-Palmolive?", "top_k": 5},
        {},
        context,
    )

    assert result.payload["results"] == []
    assert any("off-topic" in caveat.lower() or "main query entities" in caveat.lower() for caveat in result.caveats)


def test_sql_query_search_reports_empty_results_cleanly(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agent_capabilities, "_sql_search_rows", lambda **kwargs: [])
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_EmptySearchBackend(),
        working_store=_ExplodingStore(),
        runtime=None,
    )

    result = _sql_query_search({"query": "Colgate-Palmolive and Bitcoin", "top_k": 5}, {}, context)

    assert result.payload["results"] == []
    assert result.caveats


def test_db_search_balances_years_and_suppresses_duplicates(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_YearAwareSearchBackend(),
        working_store=_ExplodingStore(),
        runtime=None,
    )

    result = _db_search(
        {
            "query": "How did Amazon coverage shift from growth to regulation from 2016 to 2019?",
            "top_k": 4,
            "date_from": "2016-01-01",
            "date_to": "2019-12-31",
        },
        {},
        context,
    )

    years = {str(row.get("date", ""))[:4] for row in result.payload["results"]}
    titles = [str(row.get("title", "")) for row in result.payload["results"]]
    assert "2016" in years
    assert "2019" in years
    assert titles.count("Amazon privacy scandal") == 1
    assert any("Year-balanced retrieval" in caveat for caveat in result.caveats)
    assert any("near-duplicate" in caveat for caveat in result.caveats)


def test_db_search_uses_sandbox_after_hybrid_and_sql_fail(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agent_capabilities, "_sql_search_rows", lambda **kwargs: [])
    runtime = _RuntimeWithMetadata(
        [
            {
                "doc_id": "c1",
                "title": "Colgate safety focus",
                "text": "Colgate safety regulation oversight and toothpaste recall coverage.",
                "published_at": "2020-06-01",
                "source": "Reuters",
            }
        ]
    )
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_EmptySearchBackend(),
        working_store=_ExplodingStore(),
        runtime=runtime,
        python_runner=_FakePythonRunner(
            [
                {
                    "doc_id": "c1",
                    "title": "Colgate safety focus",
                    "snippet": "Colgate safety regulation oversight and toothpaste recall coverage.",
                    "outlet": "Reuters",
                    "date": "2020-06-01",
                    "score": 5.0,
                    "retrieval_mode": "sandbox",
                    "score_components": {"sandbox": 5.0},
                }
            ]
        ),
    )

    result = _db_search(
        {"query": "How was Colgate-Palmolive framed in safety coverage from 2020 to 2021?", "top_k": 5},
        {},
        context,
    )

    assert result.payload["results"]
    assert result.payload["results"][0]["retrieval_mode"] == "sandbox"
    assert any("sandbox retrieval fallback" in caveat.lower() for caveat in result.caveats)


def test_fetch_documents_uses_runtime_fallback_when_store_errors() -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=Path("."),
        search_backend=None,
        working_store=_FallbackStore(),
        runtime=_RuntimeWithDocs(),
    )

    result = _fetch_documents({"doc_ids": ["doc-1"]}, {}, context)

    assert result.payload["documents"][0]["doc_id"] == "doc-1"
    assert result.payload["documents"][0]["outlet"] == "example.com"
    assert any("runtime fallback was used" in caveat for caveat in result.caveats)


def test_join_external_series_can_fetch_market_data(monkeypatch, tmp_path: Path) -> None:
    def _fake_series(**kwargs):
        assert kwargs["ticker"] == "META"
        return [
            {"ticker": "META", "date": "2018-03-01", "time_bin": "2018-03", "market_close": 180.5, "market_return": -0.01, "market_drawdown": -0.01},
            {"ticker": "META", "date": "2018-04-01", "time_bin": "2018-04", "market_close": 165.0, "market_return": -0.04, "market_drawdown": -0.04},
        ]

    monkeypatch.setattr(agent_capabilities, "_fetch_yfinance_series_rows", _fake_series)
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=_ExplodingStore(),
        runtime=None,
    )
    deps = {
        "series": ToolExecutionResult(
            payload={
                "rows": [
                    {"entity": "__all__", "time_bin": "2018-03", "count": 10},
                    {"entity": "__all__", "time_bin": "2018-04", "count": 12},
                ]
            }
        )
    }

    result = _join_external_series({"ticker": "META", "left_key": "time_bin", "right_key": "time_bin"}, deps, context)

    assert result.payload["rows"]
    assert result.payload["rows"][0]["market_close"] == 180.5
    assert result.metadata["provider"] == "yfinance"


def test_fetch_documents_strict_backend_disables_runtime_fallback(monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_REQUIRE_BACKEND_SERVICES", "true")
    monkeypatch.setenv("CORPUSAGENT2_ALLOW_LOCAL_FALLBACK", "false")
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=Path("."),
        search_backend=None,
        working_store=_FallbackStore(),
        runtime=_RuntimeWithDocs(),
    )

    try:
        _fetch_documents({"doc_ids": ["doc-1"]}, {}, context)
    except RuntimeError as exc:
        assert "transient postgres abort" in str(exc)
    else:
        raise AssertionError("Expected strict backend mode to propagate fetch failure.")


def test_time_series_aggregate_defaults_to_month_granularity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CORPUSAGENT2_TIME_GRANULARITY", "month")
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=_ExplodingStore(),
        runtime=None,
    )
    deps = {
        "sentiment": ToolExecutionResult(
            payload={
                "rows": [
                    {"label": "negative", "score": -0.8, "date": "2018-03-19"},
                    {"label": "negative", "score": -0.6, "date": "2018-03-20"},
                    {"label": "positive", "score": 0.4, "date": "2018-04-01"},
                ]
            }
        )
    }

    result = _time_series_aggregate({}, deps, context)

    assert {"entity": "negative", "time_bin": "2018-03", "count": -2} in result.payload["rows"]
    assert any(row["time_bin"] == "2018-04" for row in result.payload["rows"])


def test_topic_model_emits_time_slices_instead_of_all_bucket(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CORPUSAGENT2_TIME_GRANULARITY", "month")
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=_ExplodingStore(),
        runtime=None,
    )
    deps = {
        "fetch": ToolExecutionResult(
            payload={
                "documents": [
                    {"doc_id": "fb1", "text": "growth innovation product advertising expansion", "date": "2016-06-01"},
                    {"doc_id": "fb2", "text": "privacy regulation scandal data misuse oversight", "date": "2018-03-20"},
                ]
            }
        )
    }

    result = _topic_model({"topics_per_bin": 1, "granularity": "month"}, deps, context)

    assert result.payload["rows"]
    assert any(row["time_bin"] == "2016-06" for row in result.payload["rows"])
    assert any(row["time_bin"] == "2018-03" for row in result.payload["rows"])
