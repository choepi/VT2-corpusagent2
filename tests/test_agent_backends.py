from __future__ import annotations

import pandas as pd

import corpusagent2.agent_backends as agent_backends
from corpusagent2.agent_backends import LocalSearchBackend, OpenSearchBackend, OpenSearchConfig
from corpusagent2.retrieval import RetrievalResult, build_lexical_assets


class _RuntimeWithBrokenDenseAssets:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._df = pd.DataFrame(rows)
        self.retrieval_backend = "local"
        self.dense_model_id = "intfloat/e5-base-v2"
        self.rerank_model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._lexical_assets = build_lexical_assets(self._df, max_features=128)

    def load_lexical_assets(self):
        return self._lexical_assets

    def load_dense_assets(self):
        raise RuntimeError("broken dense asset")

    def load_metadata(self):
        return self._df.copy()

    def doc_lookup(self):
        return {
            str(row.doc_id): {
                "doc_id": str(row.doc_id),
                "title": str(row.title),
                "text": str(row.text),
                "published_at": str(row.published_at),
                "source": str(row.source),
            }
            for row in self._df.itertuples(index=False)
        }

    def load_docs(self, doc_ids):
        wanted = {str(item) for item in doc_ids}
        return self._df[self._df["doc_id"].astype(str).isin(wanted)].reset_index(drop=True)

    def doc_text_by_id(self):
        return {
            str(row.doc_id): f"{str(row.title)} {str(row.text)}".strip()
            for row in self._df.itertuples(index=False)
        }


def test_local_search_backend_uses_dense_candidate_fallback(monkeypatch) -> None:
    runtime = _RuntimeWithBrokenDenseAssets(
        [
            {"doc_id": "a", "title": "Privacy pressure", "text": "Platform privacy regulation pressure platform", "published_at": "2018-03-20", "source": "Reuters"},
            {"doc_id": "b", "title": "Growth story", "text": "Platform growth impressed investors", "published_at": "2016-06-01", "source": "Reuters"},
        ]
    )

    def _fake_dense_from_texts(*, query, model_id, texts, doc_ids, top_k, **kwargs):
        assert query
        assert set(doc_ids) == {"a", "b"}
        return [
            RetrievalResult(doc_id="a", rank=1, score=0.91, score_components={"dense_candidate": 0.91}),
            RetrievalResult(doc_id="b", rank=2, score=0.12, score_components={"dense_candidate": 0.12}),
        ][:top_k]

    monkeypatch.setattr(agent_backends, "retrieve_dense_from_texts", _fake_dense_from_texts)
    backend = LocalSearchBackend(runtime)

    rows = backend.search(query="privacy regulation platform", top_k=2, retrieval_mode="dense")

    assert [row["doc_id"] for row in rows] == ["a", "b"]
    assert rows[0]["score_components"]["dense_candidate"] == 0.91


def test_local_search_backend_falls_back_to_lexical_rows_when_dense_mode_has_no_dense_hits(monkeypatch) -> None:
    runtime = _RuntimeWithBrokenDenseAssets(
        [
            {"doc_id": "a", "title": "Privacy pressure", "text": "Platform privacy regulation pressure platform", "published_at": "2018-03-20", "source": "Reuters"},
            {"doc_id": "b", "title": "Growth story", "text": "Platform growth impressed investors", "published_at": "2016-06-01", "source": "Reuters"},
        ]
    )

    monkeypatch.setattr(agent_backends, "retrieve_dense_from_texts", lambda **kwargs: [])
    backend = LocalSearchBackend(runtime)

    rows = backend.search(query="privacy regulation platform", top_k=2, retrieval_mode="dense")

    assert {row["doc_id"] for row in rows} == {"a", "b"}
    assert rows[0]["retrieval_mode"] == "dense"


def test_opensearch_backend_uses_query_string_for_structured_queries(monkeypatch) -> None:
    requests: list[dict] = []

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"hits": {"hits": []}}

    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url, auth=None, json=None):
            requests.append(dict(json or {}))
            return _FakeResponse()

    monkeypatch.setattr(agent_backends.httpx, "Client", _FakeClient)
    backend = OpenSearchBackend(OpenSearchConfig())

    backend.search(query='(soccer OR "association football") AND NOT NFL', top_k=5)

    assert requests
    clause = requests[0]["query"]["bool"]["must"][0]
    assert "query_string" in clause


def test_opensearch_backend_uses_multi_match_for_plain_queries(monkeypatch) -> None:
    requests: list[dict] = []

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"hits": {"hits": []}}

    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url, auth=None, json=None):
            requests.append(dict(json or {}))
            return _FakeResponse()

    monkeypatch.setattr(agent_backends.httpx, "Client", _FakeClient)
    backend = OpenSearchBackend(OpenSearchConfig())

    backend.search(query="football reports", top_k=5)

    assert requests
    clause = requests[0]["query"]["bool"]["must"][0]
    assert "multi_match" in clause
