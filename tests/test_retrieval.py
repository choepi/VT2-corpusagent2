from __future__ import annotations

import pandas as pd

from corpusagent2.retrieval import build_lexical_assets, retrieve_tfidf


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
