from __future__ import annotations

import corpusagent2.retrieval as retrieval
import pandas as pd

from corpusagent2.retrieval import build_lexical_assets, dense_retrieval_enabled, retrieve_tfidf


def test_retrieve_tfidf_returns_empty_for_zero_match_query() -> None:
    df = pd.DataFrame(
        [
            {"doc_id": "a", "title": "Climate report", "text": "Climate regulation emissions policy"},
            {"doc_id": "b", "title": "Football report", "text": "Match tactics transfer injury player"},
        ]
    )
    vectorizer, matrix, doc_ids = build_lexical_assets(df, max_features=128)

    rows = retrieve_tfidf(
        query="quantum entanglement photons",
        vectorizer=vectorizer,
        matrix=matrix,
        doc_ids=doc_ids,
        top_k=5,
    )

    assert rows == []


def test_dense_retrieval_enabled_respects_explicit_toggle(monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_ENABLE_DENSE_RETRIEVAL", "false")
    monkeypatch.setenv("CORPUSAGENT2_BUILD_DENSE_ASSETS", "true")
    monkeypatch.setenv("CORPUSAGENT2_PG_INCLUDE_EMBEDDINGS", "true")

    assert dense_retrieval_enabled() is False


def test_dense_retrieval_enabled_falls_back_to_asset_flags(monkeypatch) -> None:
    monkeypatch.delenv("CORPUSAGENT2_ENABLE_DENSE_RETRIEVAL", raising=False)
    monkeypatch.setenv("CORPUSAGENT2_BUILD_DENSE_ASSETS", "false")
    monkeypatch.setenv("CORPUSAGENT2_PG_INCLUDE_EMBEDDINGS", "false")

    assert dense_retrieval_enabled(default=False) is False


def test_pg_dsn_from_env_normalizes_localhost_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(retrieval.os, "name", "nt", raising=False)
    monkeypatch.setenv("CORPUSAGENT2_PG_DSN", "postgresql://corpus:corpus@localhost:5432/corpus_db")

    dsn = retrieval.pg_dsn_from_env(required=False)

    assert "localhost" not in dsn.lower()
    assert "127.0.0.1" in dsn
