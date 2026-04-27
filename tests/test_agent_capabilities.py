from __future__ import annotations

import base64
import json
from pathlib import Path

import corpusagent2.retrieval as retrieval
import pandas as pd

from corpusagent2 import agent_capabilities
from corpusagent2.agent_capabilities import (
    AgentExecutionContext,
    _build_evidence_table,
    _clean_normalize,
    _create_working_set,
    _db_search,
    _fetch_documents,
    _join_external_series,
    _lang_id,
    _sql_query_search,
    _time_series_aggregate,
    _topic_model,
)
from corpusagent2.agent_backends import InMemoryWorkingSetStore
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


class _CapturingSearchBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(dict(kwargs))
        return []


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


def test_create_working_set_can_materialize_doc_ids_from_search_results() -> None:
    store = InMemoryWorkingSetStore()
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=Path("."),
        search_backend=None,
        working_store=store,
        runtime=None,
    )
    deps = {
        "search": ToolExecutionResult(
            payload={
                "results": [
                    {"doc_id": "doc-1", "title": "Football report", "snippet": "match report"},
                    {"doc_id": "doc-2", "title": "Football report two", "snippet": "another match report"},
                ]
            }
        )
    }

    result = _create_working_set({}, deps, context)

    assert result.payload["document_count"] == 2
    assert result.payload["working_set_doc_ids"] == ["doc-1", "doc-2"]


def test_fetch_documents_can_read_working_set_doc_ids_from_dependency_payload() -> None:
    store = InMemoryWorkingSetStore()
    store.document_lookup["doc-1"] = {
        "doc_id": "doc-1",
        "title": "Sample",
        "text": "Document text",
        "published_at": "2022-02-20",
        "source": "example.com",
    }
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=Path("."),
        search_backend=None,
        working_store=store,
        runtime=None,
    )
    deps = {"working_set": ToolExecutionResult(payload={"working_set_doc_ids": ["doc-1"]})}

    result = _fetch_documents({}, deps, context)

    assert result.payload["documents"][0]["doc_id"] == "doc-1"


def test_clean_normalize_lang_id_and_working_set_flow_preserves_documents() -> None:
    store = InMemoryWorkingSetStore()
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=Path("."),
        search_backend=None,
        working_store=store,
        runtime=None,
    )
    source_docs = {
        "fetch": ToolExecutionResult(
            payload={
                "documents": [
                    {
                        "doc_id": "doc-1",
                        "title": "Football report",
                        "text": " Football match report in English. ",
                        "published_at": "2022-02-20",
                        "source": "example.com",
                        "outlet": "example.com",
                    }
                ]
            }
        )
    }

    cleaned = _clean_normalize({}, source_docs, context)
    detected = _lang_id({}, {"cleaned": cleaned}, context)
    working_set = _create_working_set({"filter": {"language_in": ["en"]}}, {"lang": detected}, context)

    assert cleaned.payload["documents"][0]["text"] == "Football match report in English."
    assert detected.payload["rows"][0]["language"] == "en"
    assert working_set.payload["working_set_doc_ids"] == ["doc-1"]


def test_build_evidence_table_can_aggregate_noun_distribution() -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=Path("."),
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "fetch": ToolExecutionResult(
            payload={
                "documents": [
                    {"doc_id": "doc-1", "text": "Football club tactics player", "published_at": "2022-01-01", "outlet": "NZZ"},
                    {"doc_id": "doc-2", "text": "Player tactics and club strategy", "published_at": "2022-01-02", "outlet": "TA"},
                ]
            }
        ),
        "pos": ToolExecutionResult(
            payload={
                "rows": [
                    {"doc_id": "doc-1", "token": "Football", "lemma": "football", "pos": "NOUN"},
                    {"doc_id": "doc-1", "token": "club", "lemma": "club", "pos": "NOUN"},
                    {"doc_id": "doc-1", "token": "tactics", "lemma": "tactic", "pos": "NOUN"},
                    {"doc_id": "doc-1", "token": "player", "lemma": "player", "pos": "NOUN"},
                    {"doc_id": "doc-2", "token": "Player", "lemma": "player", "pos": "NOUN"},
                    {"doc_id": "doc-2", "token": "tactics", "lemma": "tactic", "pos": "NOUN"},
                    {"doc_id": "doc-2", "token": "club", "lemma": "club", "pos": "NOUN"},
                    {"doc_id": "doc-2", "token": "strategy", "lemma": "strategy", "pos": "NOUN"},
                ]
            }
        ),
    }

    result = _build_evidence_table({"task": "noun_frequency_distribution", "top_k": 5}, deps, context)
    summary = _build_evidence_table({"task": "summary_stats"}, {"fetch": deps["fetch"], "table": result}, context)

    assert result.payload["rows"]
    assert result.payload["rows"][0]["lemma"] in {"club", "player", "tactic"}
    assert any(row["metric"] == "matched_document_count" and row["value"] == 2 for row in summary.payload["rows"])


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
    monkeypatch.setattr(agent_capabilities, "_queryable_sql_store", lambda context: (object(), ""))
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


def test_query_anchor_terms_drop_analytic_scaffold_words_for_exhaustive_questions() -> None:
    anchors = [value.lower() for value in agent_capabilities._query_anchor_terms("What is the distribution of nouns across all football reports in the corpus?")]

    assert "football" in anchors
    assert "distribution" not in anchors
    assert "nouns" not in anchors
    assert "reports" not in anchors
    assert "corpus" not in anchors


def test_query_anchor_terms_split_hyphenated_topic_terms_and_drop_filler() -> None:
    query = (
        "What is the frequency distribution of individual noun lemmas across "
        "soccer-related reports in the corpus, such as the most common nouns?"
    )

    anchors = agent_capabilities._query_anchor_terms(query)

    assert anchors == ["soccer"]


def test_query_anchor_terms_keep_domain_topics_instead_of_example_frames() -> None:
    anchors = agent_capabilities._query_anchor_terms("privacy regulation stock drawdown reports")

    assert "privacy" in anchors
    assert "regulation" in anchors
    assert "stock" in anchors
    assert "drawdown" in anchors
    assert "reports" not in anchors


def test_query_anchor_terms_prefer_resolved_meaning_over_ambiguous_term() -> None:
    anchors = agent_capabilities._query_anchor_terms(
        "What is the distribution of noun lemmas in all football reports, where football means soccer?"
    )

    assert anchors == ["soccer"]


def test_db_search_materializes_full_sql_match_set_for_exhaustive_questions(monkeypatch, tmp_path: Path) -> None:
    search_backend = _CapturingSearchBackend()
    captured: dict[str, object] = {}

    def _fake_sql_search_rows(**kwargs):
        captured.update(kwargs)
        return [
            {
                "doc_id": "f-1",
                "title": "Football report one",
                "snippet": "Football tactics and club strategy.",
                "outlet": "NZZ",
                "date": "2022-05-01",
                "score": 3.5,
            },
            {
                "doc_id": "f-2",
                "title": "Football report two",
                "snippet": "Football match and league coverage.",
                "outlet": "TA",
                "date": "2022-05-02",
                "score": 3.0,
            },
        ]

    monkeypatch.setattr(agent_capabilities, "_queryable_sql_store", lambda context: (object(), ""))
    monkeypatch.setattr(agent_capabilities, "_sql_search_rows", _fake_sql_search_rows)
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=search_backend,
        working_store=_ExplodingStore(),
        runtime=None,
    )

    result = _db_search({"query": "What is the distribution of nouns across all football reports in the corpus?"}, {}, context)

    assert not search_backend.calls
    assert captured["top_k"] == 0
    assert result.payload["retrieval_mode"] == "sql"
    assert result.payload["retrieval_strategy"] == "exhaustive_analytic"
    assert len(result.payload["results"]) == 2
    assert any("full lexical postgres" in caveat.lower() for caveat in result.caveats)


def test_exhaustive_db_search_returns_preview_and_materialized_working_set(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CORPUSAGENT2_RESULT_PREVIEW_ROWS", "10")
    search_backend = _CapturingSearchBackend()

    def _fake_sql_search_rows(**kwargs):
        words = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
            "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
            "uniform", "victor", "whiskey", "xray", "yankee",
        ]
        return [
            {
                "doc_id": f"doc-{idx}",
                "title": f"Football {words[idx]} report",
                "snippet": f"{words[idx]} football tactics and club strategy.",
                "outlet": "NZZ",
                "date": f"2022-01-{idx + 1:02d}",
                "score": float(idx),
            }
            for idx in range(25)
        ]

    monkeypatch.setattr(agent_capabilities, "_queryable_sql_store", lambda context: (object(), ""))
    monkeypatch.setattr(agent_capabilities, "_sql_search_rows", _fake_sql_search_rows)
    store = InMemoryWorkingSetStore()
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=search_backend,
        working_store=store,
        runtime=None,
    )

    result = _db_search({"query": "What is the distribution of nouns across all football reports in the corpus?"}, {}, context)
    working_set = _create_working_set({}, {"search": result}, context)

    assert result.payload["result_count"] == 25
    assert len(result.payload["results"]) == 10
    assert result.payload["results_truncated"] is True
    assert result.payload["working_set_ref"]
    assert store.count_working_set("run", result.payload["working_set_ref"]) == 25
    assert working_set.payload["document_count"] == 25
    assert len(working_set.payload["working_set_doc_ids"]) == 10


def test_fetch_documents_prefers_working_set_ref_over_search_preview(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CORPUSAGENT2_WORKING_SET_FETCH_LIMIT", "2")
    store = InMemoryWorkingSetStore()
    rows = [
        {"doc_id": "doc-1", "rank": 1, "score": 3.0},
        {"doc_id": "doc-2", "rank": 2, "score": 2.0},
        {"doc_id": "doc-3", "rank": 3, "score": 1.0},
    ]
    store.record_working_set("run", "search_all", rows)
    for row in rows:
        store.document_lookup[row["doc_id"]] = {
            "doc_id": row["doc_id"],
            "title": f"Document {row['doc_id']}",
            "text": f"Full text for {row['doc_id']}",
            "published_at": "2022-01-01",
            "source": "NZZ",
        }
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_EmptySearchBackend(),
        working_store=store,
        runtime=None,
    )
    search = ToolExecutionResult(
        payload={
            "results": [{"doc_id": "doc-1", "title": "preview only"}],
            "working_set_ref": "search_all",
            "result_count": 3,
            "results_truncated": True,
        }
    )

    result = _fetch_documents({}, {"search": search}, context)

    assert [row["doc_id"] for row in result.payload["documents"]] == ["doc-1", "doc-2"]
    assert result.payload["document_count"] == 3
    assert result.payload["returned_document_count"] == 2
    assert result.payload["documents_truncated"] is True


def test_noun_distribution_streams_full_working_set_when_fetch_is_preview(tmp_path: Path) -> None:
    store = InMemoryWorkingSetStore()
    working_rows = [{"doc_id": f"doc-{idx}", "rank": idx, "score": float(idx)} for idx in range(1, 4)]
    store.record_working_set("run", "all_docs", working_rows)
    store.document_lookup.update(
        {
            "doc-1": {"doc_id": "doc-1", "title": "", "text": "apple banana", "published_at": "2022-01-01", "source": "NZZ"},
            "doc-2": {"doc_id": "doc-2", "title": "", "text": "banana banana", "published_at": "2022-01-02", "source": "NZZ"},
            "doc-3": {"doc_id": "doc-3", "title": "", "text": "carrot banana", "published_at": "2022-01-03", "source": "TA"},
        }
    )
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_EmptySearchBackend(),
        working_store=store,
        runtime=None,
    )
    fetch_preview = ToolExecutionResult(
        payload={
            "documents": [store.document_lookup["doc-1"]],
            "working_set_ref": "all_docs",
            "document_count": 3,
            "returned_document_count": 1,
            "documents_truncated": True,
        }
    )

    result = _build_evidence_table({"task": "noun_frequency_distribution", "top_k": 5}, {"fetch": fetch_preview}, context)

    rows_by_lemma = {row["lemma"]: row for row in result.payload["rows"]}
    assert rows_by_lemma["banana"]["count"] == 4
    assert rows_by_lemma["banana"]["document_frequency"] == 3
    assert result.payload["analyzed_document_count"] == 3
    assert result.metadata["full_working_set"] is True


def test_local_exhaustive_requires_multi_anchor_coverage(monkeypatch, tmp_path: Path) -> None:
    runtime = _RuntimeWithMetadata(
        [
            {
                "doc_id": "d1",
                "title": "Soccer schedule",
                "text": "Television times and sports listings.",
                "published_at": "2022-05-01",
                "source": "NZZ",
            },
            {
                "doc_id": "d2",
                "title": "Soccer league report",
                "text": "League tactics and match analysis.",
                "published_at": "2022-05-02",
                "source": "TA",
            },
        ]
    )
    monkeypatch.setattr(
        agent_capabilities,
        "_queryable_sql_store",
        lambda context: (None, "Postgres unavailable"),
    )
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_CapturingSearchBackend(),
        working_store=_ExplodingStore(),
        runtime=runtime,
    )

    result = _db_search({"query": "What is the distribution of nouns across all soccer league reports?"}, {}, context)

    assert [row["doc_id"] for row in result.payload["results"]] == ["d2"]


def test_db_search_uses_local_exhaustive_materialization_when_sql_store_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    search_backend = _CapturingSearchBackend()
    runtime = _RuntimeWithMetadata(
        [
            {
                "doc_id": "f-1",
                "title": "Football tactics report",
                "text": "Football tactics and league analysis.",
                "published_at": "2022-05-01",
                "source": "NZZ",
            },
            {
                "doc_id": "f-2",
                "title": "Soccer match report",
                "text": "Soccer match coverage and football strategy.",
                "published_at": "2022-05-02",
                "source": "TA",
            },
        ]
    )
    monkeypatch.setattr(
        agent_capabilities,
        "_queryable_sql_store",
        lambda context: (None, "Table 'article_corpus' was not found in the configured Postgres database."),
    )
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=search_backend,
        working_store=_ExplodingStore(),
        runtime=runtime,
    )

    result = _db_search({"query": "What is the distribution of nouns across all football reports in the corpus?"}, {}, context)

    assert not search_backend.calls
    assert result.payload["retrieval_mode"] == "local_exhaustive"
    assert result.payload["retrieval_strategy"] == "exhaustive_analytic"
    assert len(result.payload["results"]) == 2
    assert any("full local lexical materialization" in caveat.lower() for caveat in result.caveats)


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


def test_sql_query_search_materializes_full_sql_match_set_for_exhaustive_questions(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_sql_search_rows(**kwargs):
        captured.update(kwargs)
        return [
            {
                "doc_id": "f-1",
                "title": "Football report one",
                "snippet": "Football tactics and club strategy.",
                "outlet": "NZZ",
                "date": "2022-05-01",
                "score": 3.5,
            }
        ]

    monkeypatch.setattr(agent_capabilities, "_sql_search_rows", _fake_sql_search_rows)
    monkeypatch.setattr(agent_capabilities, "_queryable_sql_store", lambda context: (object(), ""))
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_EmptySearchBackend(),
        working_store=_ExplodingStore(),
        runtime=None,
    )

    result = _sql_query_search({"query": "What is the distribution of nouns across all football reports in the corpus?"}, {}, context)

    assert captured["top_k"] == 0
    assert result.payload["retrieval_mode"] == "sql"
    assert result.payload["retrieval_strategy"] == "exhaustive_analytic"
    assert len(result.payload["results"]) == 1
    assert any("full lexical postgres" in caveat.lower() for caveat in result.caveats)


def test_sql_query_search_uses_local_exhaustive_materialization_when_sql_store_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    runtime = _RuntimeWithMetadata(
        [
            {
                "doc_id": "f-1",
                "title": "Football tactics report",
                "text": "Football tactics and league analysis.",
                "published_at": "2022-05-01",
                "source": "NZZ",
            }
        ]
    )
    monkeypatch.setattr(
        agent_capabilities,
        "_queryable_sql_store",
        lambda context: (None, "Table 'article_corpus' was not found in the configured Postgres database."),
    )
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_EmptySearchBackend(),
        working_store=_ExplodingStore(),
        runtime=runtime,
    )

    result = _sql_query_search({"query": "What is the distribution of nouns across all football reports in the corpus?"}, {}, context)

    assert result.payload["retrieval_mode"] == "local_exhaustive"
    assert result.payload["retrieval_strategy"] == "exhaustive_analytic"
    assert len(result.payload["results"]) == 1
    assert any("full local lexical materialization" in caveat.lower() for caveat in result.caveats)


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
