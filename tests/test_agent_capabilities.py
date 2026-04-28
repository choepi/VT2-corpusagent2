from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

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
    _extract_keyterms,
    _join_external_series,
    _lang_id,
    _ner,
    _plot_artifact,
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


class _FailingSearchBackend:
    def search(self, **kwargs):
        raise RuntimeError("backend rejected query syntax")


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


def test_fetch_documents_resolves_node_id_working_set_ref_from_dependency_payload() -> None:
    store = InMemoryWorkingSetStore()
    store.document_lookup["doc-1"] = {
        "doc_id": "doc-1",
        "title": "Sample",
        "text": "Document text",
        "published_at": "2022-02-20",
        "source": "example.com",
    }
    store.record_working_set("run", "actual_ref", [{"doc_id": "doc-1", "rank": 1, "score": 1.0}])
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=Path("."),
        search_backend=None,
        working_store=store,
        runtime=None,
    )
    deps = {"fetch": ToolExecutionResult(payload={"working_set_ref": "actual_ref", "document_count": 1})}

    result = _fetch_documents({"working_set_ref": "n2", "limit": 1}, deps, context)

    assert result.payload["working_set_ref"] == "actual_ref"
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


def test_build_evidence_table_accepts_token_frequency_alias_without_fake_evidence(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "fetch": ToolExecutionResult(payload={"documents": [{"doc_id": "doc-1", "text": "Club player player"}]}),
        "pos": ToolExecutionResult(
            payload={
                "rows": [
                    {"doc_id": "doc-1", "token": "Club", "lemma": "club", "pos": "NOUN"},
                    {"doc_id": "doc-1", "token": "player", "lemma": "player", "pos": "NOUN"},
                    {"doc_id": "doc-1", "token": "player", "lemma": "player", "pos": "NOUN"},
                ]
            }
        ),
    }

    result = _build_evidence_table(
        {"task": "aggregate_token_frequencies", "filters": {"upos": ["NOUN"]}, "limit": 2},
        deps,
        context,
    )

    assert result.payload["rows"][0]["lemma"] == "player"
    assert result.evidence == []
    assert "doc_id" not in result.payload["rows"][0]


def test_build_evidence_table_uses_heuristic_noun_distribution_without_pos_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agent_capabilities, "_load_spacy_model", lambda: None)
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "fetch": ToolExecutionResult(
            payload={
                "documents": [
                    {"doc_id": "doc-1", "text": "Football club player player"},
                    {"doc_id": "doc-2", "text": "Soccer club strategy"},
                ],
                "document_count": 2,
                "returned_document_count": 2,
                "documents_truncated": False,
            }
        )
    }

    result = _build_evidence_table({"task": "noun_frequency_distribution", "top_k": 5}, deps, context)

    rows_by_lemma = {row["lemma"]: row for row in result.payload["rows"]}
    assert rows_by_lemma["player"]["count"] == 2
    assert rows_by_lemma["club"]["document_frequency"] == 2
    assert result.metadata["provider"] == "heuristic_batch"
    assert result.metadata["preview_only"] is False


def test_noun_distribution_ignores_machine_payload_terms(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agent_capabilities, "_load_spacy_model", lambda: None)
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "fetch": ToolExecutionResult(
            payload={
                "documents": [
                    {
                        "doc_id": "doc-1",
                        "title": "Ronaldo transfer value debate",
                        "text": (
                            '","duration":174,"analytics":{"video_id":"abc"},'
                            '"thumbnail_url":"https://example.invalid/t.jpg",'
                            '"article_type":"uber_article","content_type":"video"'
                        ),
                    }
                ],
                "document_count": 1,
                "returned_document_count": 1,
                "documents_truncated": False,
            }
        )
    }

    result = _build_evidence_table({"task": "noun_frequency_distribution", "top_k": 20}, deps, context)

    lemmas = {row["lemma"] for row in result.payload["rows"]}
    assert {"ronaldo", "transfer", "value", "debate"}.intersection(lemmas)
    assert "duration" not in lemmas
    assert "analytics" not in lemmas
    assert result.metadata["machine_payload_document_count"] == 1


def test_result_normalization_penalizes_machine_payload_snippets() -> None:
    rows = [
        {
            "doc_id": "json",
            "title": "Video shell",
            "snippet": '","duration":174,"analytics":{"video_id":"abc"},"thumbnail_url":"x","content_type":"video"',
            "score": 10.0,
        },
        {"doc_id": "article", "title": "Article", "snippet": "Ronaldo scored in the football match.", "score": 9.0},
    ]

    normalized = agent_capabilities._normalize_result_rows(rows, "sql")

    assert normalized[0]["doc_id"] == "article"
    machine_row = next(row for row in normalized if row["doc_id"] == "json")
    assert machine_row["snippet_is_machine_payload"] is True
    assert machine_row["score_quality_multiplier"] < 1.0


def test_plot_artifact_refuses_requested_missing_fields_instead_of_plotting_doc_ids(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "evidence": ToolExecutionResult(
            payload={"rows": [{"doc_id": "abc123", "score": 0.0, "rank": 1}]}
        )
    }

    result = _plot_artifact(
        {"x": "lemma", "y": "count", "limit": 10, "title": "Top nouns"},
        deps,
        context,
    )

    assert result.metadata["no_data"] is True
    assert result.artifacts == []
    assert "lemma" in result.caveats[0]


def test_plot_artifact_resolves_noun_lemma_alias_to_lemma(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "table": ToolExecutionResult(
            payload={"rows": [{"lemma": "team", "count": 3}, {"lemma": "game", "count": 2}]}
        )
    }

    result = _plot_artifact(
        {"x": "noun_lemma", "y": "count", "top_k": 2, "title": "Top nouns"},
        deps,
        context,
    )

    assert result.artifacts
    assert Path(result.payload["artifact_path"]).exists()
    assert result.payload["resolved_x"] == "lemma"
    assert result.metadata["resolved_x"] == "lemma"
    assert any("Resolved plot x field 'noun_lemma'" in caveat for caveat in result.caveats)


def test_plot_artifact_infers_fields_when_axis_names_are_omitted(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "table": ToolExecutionResult(
            payload={
                "rows": [
                    {"outlet": "NZZ", "mentions": "1,200"},
                    {"outlet": "TA", "mentions": "950"},
                    {"outlet": "SRF", "mentions": "not available"},
                    {"outlet": "WOZ", "mentions": "125"},
                ]
            }
        )
    }

    result = _plot_artifact({"top_k": 3, "title": "Mentions by outlet"}, deps, context)

    assert result.artifacts
    assert Path(result.payload["artifact_path"]).exists()
    assert result.payload["resolved_x"] == "outlet"
    assert result.payload["resolved_y"] == "mentions"
    assert result.payload["plotted_row_count"] == 3
    assert any("Inferred plot x field 'outlet'" in caveat for caveat in result.caveats)
    assert any("Inferred plot y field 'mentions'" in caveat for caveat in result.caveats)


def test_plot_artifact_resolves_normalized_axis_aliases(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "table": ToolExecutionResult(
            payload={"rows": [{"lemma": "team", "count": 3}, {"lemma": "game", "count": 2}]}
        )
    }

    result = _plot_artifact(
        {"x": "noun lemma", "y": "frequency", "top_k": 2, "title": "Top nouns"},
        deps,
        context,
    )

    assert result.artifacts
    assert result.payload["resolved_x"] == "lemma"
    assert result.payload["resolved_y"] == "count"


def test_plot_artifact_resolves_month_and_series_aliases(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "table": ToolExecutionResult(
            payload={
                "rows": [
                    {"time_bin": "2017-01", "linked_entity": "SPUR", "document_frequency": 4},
                    {"time_bin": "2017-02", "linked_entity": "SPUR", "document_frequency": 7},
                    {"time_bin": "2017-01", "linked_entity": "HUD", "document_frequency": 2},
                    {"time_bin": "2017-02", "linked_entity": "HUD", "document_frequency": 3},
                ]
            }
        )
    }

    result = _plot_artifact(
        {"plot_type": "line", "x": "month", "y": "document_frequency", "series": "actor", "title": "Actors"},
        deps,
        context,
    )

    assert result.artifacts
    assert result.payload["resolved_x"] == "time_bin"
    assert result.payload["resolved_y"] == "document_frequency"
    assert result.payload["resolved_series"] == "linked_entity"
    assert agent_capabilities._image_has_visual_content(Path(result.payload["artifact_path"]))


def test_plot_artifact_skips_placeholder_time_axis_labels(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "table": ToolExecutionResult(
            payload={
                "rows": [
                    {"time_bin": "2017-01", "entity": "__all__", "count": 4},
                    {"time_bin": "unkn", "entity": "__all__", "count": 999},
                    {"time_bin": "unknown", "entity": "__all__", "count": 888},
                    {"time_bin": "2017-02", "entity": "__all__", "count": 6},
                ]
            }
        )
    }

    result = _plot_artifact(
        {"plot_type": "line", "x": "time_bin", "y": "count", "series": "entity", "title": "Timeline"},
        deps,
        context,
    )

    assert result.artifacts
    assert [row["time_bin"] for row in result.payload["rows"]] == ["2017-01", "2017-02"]
    assert result.payload["plotted_row_count"] == 2
    assert result.payload["skipped_row_count"] == 2
    assert any("placeholder labels" in caveat for caveat in result.caveats)


def test_plot_artifact_outputs_nonblank_images_when_called_concurrently(tmp_path: Path) -> None:
    deps = {
        "table": ToolExecutionResult(
            payload={
                "rows": [
                    {"time_bin": "2017-01", "canonical_entity": "Switzerland", "count": 4},
                    {"time_bin": "2017-02", "canonical_entity": "Switzerland", "count": 6},
                    {"time_bin": "2017-01", "canonical_entity": "Paris Agreement", "count": 3},
                    {"time_bin": "2017-02", "canonical_entity": "Paris Agreement", "count": 2},
                ]
            }
        )
    }

    def render(index: int) -> ToolExecutionResult:
        context = AgentExecutionContext(
            run_id=f"run-{index}",
            artifacts_dir=tmp_path / f"run-{index}",
            search_backend=None,
            working_store=InMemoryWorkingSetStore(),
            runtime=None,
        )
        return _plot_artifact(
            {
                "plot_type": "line",
                "x": "time_bin",
                "y": "count",
                "series": "canonical_entity",
                "title": f"Concurrent Plot {index}",
            },
            deps,
            context,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(render, range(2)))

    for result in results:
        assert result.artifacts
        assert agent_capabilities._image_has_visual_content(Path(result.payload["artifact_path"]))


def test_plot_artifact_resolves_published_at_to_time_bin(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "series": ToolExecutionResult(
            payload={"rows": [{"entity": "Trump", "time_bin": "2017-01", "normalized_value": 4.0}]}
        )
    }

    result = _plot_artifact(
        {"plot_type": "line", "x": "published_at", "y": "normalized_value", "series": "entity"},
        deps,
        context,
    )

    assert result.artifacts
    assert result.payload["resolved_x"] == "time_bin"
    assert result.payload["resolved_y"] == "normalized_value"


def test_plot_artifact_resolves_published_at_month_to_time_bin(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "series": ToolExecutionResult(
            payload={"rows": [{"canonical_entity": "Switzerland", "time_bin": "2017-01", "mention_count": 4}]}
        )
    }

    result = _plot_artifact(
        {"plot_type": "line", "x": "published_at_month", "y": "mention_count", "series": "canonical_entity"},
        deps,
        context,
    )

    assert result.artifacts
    assert result.payload["resolved_x"] == "time_bin"


def test_plot_artifact_resolves_entity_canonical_series_alias(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "series": ToolExecutionResult(
            payload={
                "rows": [
                    {"canonical_entity": "NZZ", "month": "2017-01", "mention_share": 0.2},
                    {"canonical_entity": "NZZ", "month": "2017-02", "mention_share": 0.1},
                ]
            }
        )
    }

    result = _plot_artifact(
        {"plot_type": "line", "x": "month", "y": "mention_share", "series": "entity_canonical"},
        deps,
        context,
    )

    assert result.artifacts
    assert result.payload["resolved_series"] == "canonical_entity"


def test_plot_artifact_resolves_monthly_entity_share_alias(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "series": ToolExecutionResult(
            payload={
                "rows": [
                    {"canonical_entity": "NZZ", "month": "2017-01", "mention_share": 0.2},
                    {"canonical_entity": "NZZ", "month": "2017-02", "mention_share": 0.1},
                ]
            }
        )
    }

    result = _plot_artifact(
        {
            "plot_type": "line",
            "x": "month",
            "y": "share_of_monthly_entity_mentions",
            "series": "canonical_entity",
        },
        deps,
        context,
    )

    assert result.artifacts
    assert result.payload["resolved_y"] == "mention_share"


def test_plot_artifact_resolves_share_of_documents_alias(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "table": ToolExecutionResult(
            payload={"rows": [{"entity": "Trump", "time_bin": "2017-01", "share_of_docs": 0.25}]}
        )
    }

    result = _plot_artifact(
        {"x": "entity", "y": "share_of_documents", "title": "Entity share"},
        deps,
        context,
    )

    assert result.artifacts
    assert result.payload["resolved_y"] == "share_of_docs"


def test_plot_artifact_resolves_share_of_mentions_alias(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "table": ToolExecutionResult(
            payload={"rows": [{"entity": "Trump", "time_bin": "2017-01", "mention_share": 0.5}]}
        )
    }

    result = _plot_artifact(
        {"x": "entity", "y": "share_of_mentions", "title": "Mention share"},
        deps,
        context,
    )

    assert result.artifacts
    assert result.payload["resolved_y"] == "mention_share"


def test_plot_artifact_resolves_normalized_document_frequency_alias(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "series": ToolExecutionResult(
            payload={"rows": [{"entity": "Switzerland", "time_bin": "2017-01", "normalized_value": 3.0}]}
        )
    }

    result = _plot_artifact(
        {"plot_type": "line", "x": "time_bin", "y": "normalized_document_frequency", "series": "entity"},
        deps,
        context,
    )

    assert result.artifacts
    assert result.payload["resolved_y"] == "normalized_value"

    result = _plot_artifact(
        {"plot_type": "line", "x": "time_bin", "y": "document_frequency_normalized", "series": "entity"},
        deps,
        context,
    )

    assert result.artifacts
    assert result.payload["resolved_y"] == "normalized_value"


def test_plot_artifact_prefers_nonzero_alias_candidate(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "table": ToolExecutionResult(
            payload={
                "rows": [
                    {
                        "entity": "Switzerland",
                        "time_bin": "2017-01",
                        "share_of_documents": 0.0,
                        "mention_share": 0.25,
                    }
                ]
            }
        )
    }

    result = _plot_artifact(
        {"x": "entity", "y": "share_of_climate_docs", "title": "Climate document share"},
        deps,
        context,
    )

    assert result.artifacts
    assert result.payload["resolved_y"] == "mention_share"


def test_build_evidence_table_uses_task_name_for_entity_frequency(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "entities": ToolExecutionResult(
            payload={
                "rows": [
                    {"doc_id": "d1", "linked_entity": "SPUR", "label": "ORG", "time_bin": "2017-01"},
                    {"doc_id": "d1", "linked_entity": "SPUR", "label": "ORG", "time_bin": "2017-01"},
                    {"doc_id": "d2", "linked_entity": "HUD", "label": "ORG", "time_bin": "2017-02"},
                    {"doc_id": "d3", "linked_entity": "2017", "label": "DATE", "time_bin": "2017-02"},
                ]
            }
        )
    }

    result = _build_evidence_table(
        {
            "task_name": "named_entity_distribution",
            "entity_field": "linked_entity",
            "entity_types": ["ORG"],
            "group_by_time": "month",
        },
        deps,
        context,
    )

    rows = result.payload["rows"]
    spur = next(row for row in rows if row["entity"] == "SPUR")
    assert spur["month"] == "2017-01"
    assert spur["mention_count"] == 2
    assert spur["document_frequency"] == 1
    assert all(row["entity"] != "2017" for row in rows)


def test_python_runner_handles_native_entity_aggregation_without_sandbox(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
        python_runner=None,
    )
    deps = {
        "entities": ToolExecutionResult(
            payload={"rows": [{"doc_id": "d1", "entity": "Emmanuel Macron", "label": "PERSON", "time_bin": "2017-12"}]}
        )
    }

    result = agent_capabilities._python_runner(
        {"task": "aggregate_named_entity_prominence_overall", "entity_types": ["PERSON"]},
        deps,
        context,
    )

    assert result.payload["rows"][0]["entity"] == "Emmanuel Macron"
    assert result.payload["exit_code"] == 0


def test_python_runner_flattens_nested_entity_aggregation_params(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
        python_runner=None,
    )
    deps = {
        "entities": ToolExecutionResult(
            payload={
                "rows": [
                    {
                        "doc_id": "d1",
                        "entity_text": "Switzerland",
                        "entity_label": "GPE",
                        "published_at": "2017-01-15",
                    },
                    {
                        "doc_id": "d1",
                        "entity_text": "2017",
                        "entity_label": "DATE",
                        "published_at": "2017-01-15",
                    },
                ]
            }
        )
    }

    result = agent_capabilities._python_runner(
        {
            "task": "aggregate_named_entities_overall_and_monthly",
            "params": {
                "entity_text_field": "entity_text",
                "entity_label_field": "entity_label",
                "allowed_entity_labels": ["GPE"],
                "time_field": "published_at",
            },
        },
        deps,
        context,
    )

    rows = result.payload["rows"]
    assert rows[0]["entity"] == "Switzerland"
    assert rows[0]["month"] == "2017-01"
    assert all(row["entity"] != "2017" for row in rows)


def test_actor_prominence_merge_preserves_time_and_filters_junk_entities(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
        python_runner=None,
    )
    deps = {
        "actors": ToolExecutionResult(
            payload={
                "rows": [
                    {
                        "actor": "Councilman Ritchie Torres",
                        "month": "2017-01",
                        "mention_count": 2,
                        "document_frequency": 1,
                    },
                    {"actor": "he", "month": "2017-01", "mention_count": 2},
                    {"actor": "ICICI Bank today", "month": "2017-02", "mention_count": 1},
                ]
            }
        )
    }

    result = agent_capabilities._python_runner(
        {"task": "merge_actor_prominence_series"},
        deps,
        context,
    )

    actors = {row["actor"]: row for row in result.payload["rows"]}
    assert "he" not in actors
    assert actors["Councilman Ritchie Torres"]["month"] == "2017-01"
    assert "ICICI Bank" in actors
    assert "ICICI Bank today" not in actors


def test_entity_surface_filter_keeps_real_uppercase_entities() -> None:
    assert agent_capabilities._valid_entity_surface("US")
    assert agent_capabilities._valid_entity_surface("U.S.")
    assert agent_capabilities._valid_entity_surface("Will Smith")
    assert not agent_capabilities._valid_entity_surface("has")
    assert not agent_capabilities._valid_entity_surface("He")
    assert not agent_capabilities._valid_entity_surface("report")
    assert not agent_capabilities._valid_entity_surface("1cd6eb4bc602a32061d545e5bd3b39a7895b851ef3de7a82b1a82b39d630bc6f")


def test_sentence_split_rows_remain_document_inputs_for_downstream_tools(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "fetch": ToolExecutionResult(
            payload={
                "documents": [
                    {
                        "doc_id": "d1",
                        "text": "Cristiano Ronaldo is an elite star. Lionel Messi has great value.",
                        "published_at": "2018-01-02",
                        "source": "sports.example",
                    }
                ]
            }
        )
    }

    split = agent_capabilities._sentence_split_docs({}, deps, context)
    rows = agent_capabilities._doc_rows({"split": split})

    assert rows[0]["published_at"] == "2018-01-02"
    assert "Cristiano Ronaldo" in rows[0]["text"]


def test_claim_span_extract_honors_target_aliases_and_focus_terms(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "docs": ToolExecutionResult(
            payload={
                "documents": [
                    {
                        "doc_id": "d1",
                        "text": "Cristiano Ronaldo was called an elite star. A coach discussed unrelated tactics.",
                        "published_at": "2018-01-02",
                        "source": "sports.example",
                    },
                    {
                        "doc_id": "d2",
                        "text": "Lionel Messi retained great value for Barcelona.",
                        "published_at": "2018-02-03",
                        "source": "sports.example",
                    },
                ]
            }
        )
    }

    result = agent_capabilities._claim_span_extract(
        {
            "targets": [
                {"label": "Cristiano Ronaldo", "match_terms": ["Ronaldo"]},
                {"label": "Lionel Messi", "match_terms": ["Messi"]},
            ],
            "claim_focus_terms": ["elite", "value"],
        },
        deps,
        context,
    )

    rows = result.payload["rows"]
    assert {row["target_entity"] for row in rows} == {"Cristiano Ronaldo", "Lionel Messi"}
    assert all(row["claim_span"] for row in rows)
    assert all(row["entity_label"] for row in rows)
    assert all(row["published_at"] for row in rows)


def test_targeted_claim_and_sentiment_infer_comparison_entities_from_question(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_SENTIMENT", "heuristic")
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
        state=SimpleNamespace(question="How did the perceived value of Cristiano Ronaldo versus Lionel Messi evolve over time?"),
    )
    deps = {
        "docs": ToolExecutionResult(
            payload={
                "documents": [
                    {
                        "doc_id": "d1",
                        "text": "Cristiano Ronaldo signed a valuable contract. Lionel Messi retained great market value.",
                        "published_at": "2018-03-20",
                        "source": "sports.example",
                    }
                ]
            }
        )
    }

    claims = agent_capabilities._claim_span_extract(
        {"query_focus": "Cristiano Ronaldo Lionel Messi value valuation worth market value contract"},
        deps,
        context,
    )
    sentiment = agent_capabilities._sentiment(
        {
            "window_strategy": "entity_local_context",
            "context_keywords": ["value", "contract"],
            "query_focus": "Cristiano Ronaldo Lionel Messi value contract",
        },
        deps,
        context,
    )

    assert {row["target_entity"] for row in claims.payload["rows"]} == {"Cristiano Ronaldo", "Lionel Messi"}
    assert {row["entity_label"] for row in sentiment.payload["rows"]} == {"Cristiano Ronaldo", "Lionel Messi"}


def test_time_series_aggregate_merges_metric_sources_by_series_definitions(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "entities": ToolExecutionResult(
            payload={
                "rows": [
                    {"doc_id": "d1", "entity": "Ronaldo", "published_at": "2018-01-02"},
                    {"doc_id": "d2", "entity": "Messi", "published_at": "2018-01-03"},
                ]
            }
        ),
        "strength": ToolExecutionResult(
            payload={"rows": [{"doc_id": "d1", "target_entity": "Cristiano Ronaldo", "published_at": "2018-01-02", "score": 0.7}]}
        ),
        "sentiment": ToolExecutionResult(
            payload={"rows": [{"doc_id": "d2", "target_entity": "Lionel Messi", "published_at": "2018-01-03", "score": 0.25}]}
        ),
    }

    result = _time_series_aggregate(
        {
            "bucket": "month",
            "series_definitions": [
                {"series_name": "Cristiano Ronaldo", "aliases": ["Ronaldo"]},
                {"series_name": "Lionel Messi", "aliases": ["Messi"]},
            ],
            "metrics": [
                {"name": "mention_count", "source": "entities", "aggregation": "count"},
                {"name": "avg_claim_strength", "source": "strength", "aggregation": "mean"},
                {"name": "avg_claim_sentiment", "source": "sentiment", "aggregation": "mean"},
            ],
        },
        deps,
        context,
    )

    rows = {(row["series_name"], row["time_bin"]): row for row in result.payload["rows"]}
    assert rows[("Cristiano Ronaldo", "2018-01")]["mention_count"] == 1
    assert rows[("Cristiano Ronaldo", "2018-01")]["avg_claim_strength"] == 0.7
    assert rows[("Lionel Messi", "2018-01")]["avg_claim_sentiment"] == 0.25
    assert rows[("Lionel Messi", "2018-01")]["bucket"] == "2018-01"


def test_time_series_aggregate_uses_metrics_source_and_mean_alias_fields(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "claims": ToolExecutionResult(
            payload={
                "rows": [
                    {"doc_id": "d1", "entity_label": "Cristiano Ronaldo", "published_at": "2018-01-02"},
                ]
            }
        ),
        "scores": ToolExecutionResult(
            payload={
                "rows": [
                    {"doc_id": "d1", "entity_label": "Cristiano Ronaldo", "published_at": "2018-01-02", "claim_strength_score": 0.5},
                    {"doc_id": "d2", "entity_label": "Cristiano Ronaldo", "published_at": "2018-01-03", "claim_strength_score": 0.9},
                ]
            }
        ),
    }

    result = _time_series_aggregate(
        {
            "documents_source": "claims",
            "metrics_source": "scores",
            "time_field": "published_at",
            "series_key": "entity_label",
            "value_field": "claim_strength_score",
            "aggregation": "mean",
            "interval": "month",
        },
        deps,
        context,
    )

    rows = result.payload["rows"]
    assert rows[0]["entity_label"] == "Cristiano Ronaldo"
    assert rows[0]["claim_strength_score"] == 0.7
    assert rows[0]["mean_claim_strength_score"] == 0.7
    assert rows[0]["bucket"] == "2018-01"


def test_time_series_aggregate_handles_document_series_and_string_metrics(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "docs": ToolExecutionResult(
            payload={
                "documents": [
                    {
                        "doc_id": "d1",
                        "text": "Cristiano Ronaldo signed a valuable new contract.",
                        "published_at": "2018-03-20",
                    },
                    {
                        "doc_id": "d2",
                        "text": "Lionel Messi retained high market value.",
                        "published_at": "2018-03-21",
                    },
                ]
            }
        )
    }

    result = _time_series_aggregate(
        {
            "documents_node": "docs",
            "bucket_granularity": "month",
            "series": [
                {"name": "ronaldo_value_docs", "entity_terms": ["Cristiano Ronaldo", "Ronaldo"], "keyword_terms": ["contract", "value"]},
                {"name": "messi_value_docs", "entity_terms": ["Lionel Messi", "Messi"], "keyword_terms": ["market value"]},
            ],
            "metrics": ["document_count"],
        },
        deps,
        context,
    )

    rows = {(row["series_name"], row["time_bin"]): row for row in result.payload["rows"]}
    assert rows[("Cristiano Ronaldo", "2018-03")]["document_count"] == 1
    assert rows[("Lionel Messi", "2018-03")]["document_count"] == 1


def test_time_series_aggregate_handles_named_average_metrics(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "sentiment": ToolExecutionResult(
            payload={
                "rows": [
                    {"doc_id": "d1", "entity_label": "Cristiano Ronaldo", "published_at": "2018-03-20", "sentiment_score": 0.4},
                    {"doc_id": "d2", "entity_label": "Cristiano Ronaldo", "published_at": "2018-03-21", "sentiment_score": 0.8},
                ]
            }
        )
    }

    result = _time_series_aggregate(
        {
            "documents_node": "sentiment",
            "time_field": "published_at",
            "group_by": "target_label",
            "metrics": ["average_sentiment", "document_count"],
            "bucket_granularity": "month",
        },
        deps,
        context,
    )

    assert result.payload["rows"][0]["average_sentiment"] == 0.6
    assert result.payload["rows"][0]["document_count"] == 2
    assert result.payload["rows"][0]["target_label"] == "Cristiano Ronaldo"


def test_time_series_aggregate_falls_back_to_overall_series_for_ungrouped_sentiment(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "sentiment": ToolExecutionResult(
            payload={
                "rows": [
                    {"doc_id": "d1", "label": "negative", "published_at": "2018-03-19", "score": -0.8},
                    {"doc_id": "d2", "label": "negative", "published_at": "2018-03-20", "score": -0.6},
                    {"doc_id": "d3", "label": "positive", "published_at": "2018-04-01", "score": 0.4},
                ]
            }
        )
    }

    result = _time_series_aggregate(
        {
            "documents_node": "sentiment",
            "time_field": "published_at",
            "group_by": "target_label",
            "metrics": ["average_sentiment", "document_count"],
            "bucket_granularity": "month",
        },
        deps,
        context,
    )

    rows = {row["time_bin"]: row for row in result.payload["rows"]}
    assert rows["2018-03"]["target_label"] == "__all__"
    assert rows["2018-03"]["average_sentiment"] == -0.7
    assert rows["2018-03"]["document_count"] == 2
    assert rows["2018-04"]["average_sentiment"] == 0.4
    assert result.payload["skipped_row_count"] == 0


def test_time_series_aggregate_prefers_entity_source_over_document_source(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "docs": ToolExecutionResult(
            payload={"rows": [{"doc_id": "a" * 32, "text": "document text", "published_at": "2022-01-01"}]}
        ),
        "entities": ToolExecutionResult(
            payload={
                "rows": [
                    {"doc_id": "doc-1", "linked_entity": "City Council", "label": "ORG", "published_at": "2022-01-01"},
                    {"doc_id": "doc-2", "linked_entity": "City Council", "label": "ORG", "published_at": "2022-01-15"},
                    {"doc_id": "doc-3", "linked_entity": "Mayor Lee", "label": "PERSON", "published_at": "2022-01-20"},
                    {"doc_id": "doc-4", "linked_entity": "Zurich", "label": "GPE", "published_at": "2022-01-22"},
                ]
            }
        ),
    }

    result = _time_series_aggregate(
        {
            "documents_source": "docs",
            "entities_source": "entities",
            "time_field": "published_at",
            "entity_field": "linked_entity",
            "entity_types": ["PERSON", "ORG"],
            "metrics": ["mention_count", "document_frequency", "share_of_documents"],
            "granularity": "month",
        },
        deps,
        context,
    )

    rows = result.payload["rows"]
    council = next(row for row in rows if row["series_name"] == "City Council")
    assert council["mention_count"] == 2
    assert council["document_frequency"] == 2
    assert council["share_of_documents"] == 0.666667
    assert all(row["series_name"] != "Zurich" for row in rows)
    assert result.payload["skipped_row_count"] == 1


def test_python_runner_refuses_prose_instead_of_executing_invalid_code(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
        python_runner=None,
    )

    result = agent_capabilities._python_runner(
        {"code": "Merge monthly actor prominence tables from entity mentions and quote attributions."},
        {},
        context,
    )

    assert result.metadata["no_data"] is True
    assert result.payload["exit_code"] == 0
    assert "not valid Python" in result.caveats[0]


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


def test_db_search_returns_no_data_when_source_filtered_sql_fallback_is_empty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agent_capabilities, "_sql_search_rows", lambda **kwargs: [])
    monkeypatch.setattr(agent_capabilities, "_queryable_sql_store", lambda context: (object(), ""))
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_FailingSearchBackend(),
        working_store=_ExplodingStore(),
        runtime=None,
    )

    result = _db_search(
        {"query": '(football OR soccer) AND source:("NZZ" OR "Tages-Anzeiger")', "top_k": 5},
        {},
        context,
    )

    assert result.payload["results"] == []
    assert result.payload["result_count"] == 0
    assert result.metadata["no_data"] is True
    assert any("source filters" in caveat.lower() for caveat in result.caveats)


def test_query_anchor_terms_drop_analytic_scaffold_words_for_exhaustive_questions() -> None:
    anchors = [value.lower() for value in agent_capabilities._query_anchor_terms("What is the distribution of nouns across all football reports in the corpus?")]

    assert "football" in anchors
    assert "distribution" not in anchors
    assert "nouns" not in anchors
    assert "reports" not in anchors
    assert "corpus" not in anchors


def test_query_anchor_terms_drop_comparative_filler_words() -> None:
    anchors = agent_capabilities._query_anchor_terms(
        "Cristiano Ronaldo Lionel Messi perceptions relative value within"
    )

    assert "perceptions" not in anchors
    assert "relative" not in anchors
    assert "within" not in anchors
    assert "ronaldo" in anchors
    assert "messi" in anchors


def test_query_anchor_terms_split_hyphenated_topic_terms_and_drop_filler() -> None:
    query = (
        "What is the frequency distribution of individual noun lemmas across "
        "soccer-related reports in the corpus, such as the most common nouns?"
    )

    anchors = agent_capabilities._query_anchor_terms(query)

    assert anchors == ["soccer"]


def test_query_anchor_terms_drop_generic_hyphenated_modifiers() -> None:
    anchors = agent_capabilities._query_anchor_terms(
        "What is the frequency distribution of noun lemmas in association-soccer reports?"
    )

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


def test_sql_websearch_query_text_preserves_explicit_or_broadening() -> None:
    query = (
        "football OR soccer OR premier league OR champions league OR fa cup "
        "OR world cup OR match report OR fixture"
    )
    anchors = agent_capabilities._query_anchor_terms(query)

    query_text = agent_capabilities._sql_websearch_query_text(query, anchors)

    assert " OR " in query_text
    assert "football OR soccer" in query_text
    assert "premier league" in query_text
    assert not query_text.startswith("football soccer")
    assert agent_capabilities._min_required_anchor_hits(query, anchors, top_k=0) == 1


def test_sql_websearch_query_text_keeps_default_anchor_conjunction() -> None:
    query = "privacy regulation stock drawdown reports"
    anchors = agent_capabilities._query_anchor_terms(query)

    query_text = agent_capabilities._sql_websearch_query_text(query, anchors)

    assert " OR " not in query_text
    assert query_text == " ".join(anchors)
    assert agent_capabilities._min_required_anchor_hits(query, anchors, top_k=0) > 1


def test_sql_websearch_query_clauses_preserve_and_between_or_groups() -> None:
    query = '("Cristiano Ronaldo" OR Ronaldo OR CR7 OR "Lionel Messi" OR Messi) AND (value OR worth OR contract)'
    anchors = agent_capabilities._query_anchor_terms(query)

    clauses = agent_capabilities._sql_websearch_query_clauses(query, anchors)

    assert len(clauses) == 2
    assert "ronaldo" in clauses[0]
    assert " OR " in clauses[0]
    assert "value OR worth" in clauses[1]


def test_query_anchor_terms_strip_source_filters() -> None:
    query = '(football OR Fussball) AND source:("NZZ" OR "Tages-Anzeiger")'

    anchors = agent_capabilities._query_anchor_terms(query)
    sources = agent_capabilities._query_source_filters(query)

    assert "source" not in anchors
    assert "nzz" not in anchors
    assert "tages" not in anchors
    assert sources == ["nzz", "tagesanzeiger"]


def test_query_anchor_terms_drop_climate_scaffold_with_source_filter() -> None:
    query = '(climate coverage over time) AND source:("swissinfoch" OR "nzzch")'

    anchors = agent_capabilities._query_anchor_terms(query)
    sources = agent_capabilities._query_source_filters(query)

    assert anchors == ["climate"]
    assert sources == ["swissinfoch", "nzzch"]


def test_exhaustive_search_reports_absent_source_scope(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agent_capabilities, "_queryable_sql_store", lambda context: (object(), ""))
    monkeypatch.setattr(agent_capabilities, "_sql_search_rows", lambda **kwargs: [])
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_EmptySearchBackend(),
        working_store=InMemoryWorkingSetStore(),
        runtime=_RuntimeWithMetadata(
            [
                {"doc_id": "doc-1", "title": "Football", "text": "football", "published_at": "2020-01-01", "source": "www.swissinfo.ch"},
                {"doc_id": "doc-2", "title": "Football", "text": "football", "published_at": "2020-01-02", "source": "uk.reuters.com"},
            ]
        ),
    )

    result = _db_search(
        {
            "query": '(football OR soccer OR fussball) AND source:("nzz" OR "tagesanzeiger")',
            "retrieval_strategy": "exhaustive_analytic",
            "top_k": 0,
        },
        {},
        context,
    )

    assert result.payload["result_count"] == 0
    assert any("requested outlets may be absent" in caveat.lower() for caveat in result.caveats)


def test_local_exhaustive_applies_source_filters(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=InMemoryWorkingSetStore(),
        runtime=_RuntimeWithMetadata(
            [
                {"doc_id": "nzz-1", "title": "Football report", "text": "football match", "published_at": "2017-01-01", "source": "www.nzz.ch"},
                {"doc_id": "ta-1", "title": "Football report", "text": "football match", "published_at": "2017-01-01", "source": "www.tagesanzeiger.ch"},
            ]
        ),
    )

    rows = agent_capabilities._local_exhaustive_rows(
        query='football source:"NZZ"',
        top_k=0,
        date_from="",
        date_to="",
        context=context,
    )

    assert [row["doc_id"] for row in rows] == ["nzz-1"]


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


def test_entity_analysis_default_is_uncapped(monkeypatch) -> None:
    monkeypatch.delenv("CORPUSAGENT2_ENTITY_ANALYSIS_MAX_DOCS", raising=False)

    assert agent_capabilities._entity_analysis_max_documents() is None


def test_ner_streams_full_working_set_when_fetch_is_preview(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("CORPUSAGENT2_ENTITY_ANALYSIS_MAX_DOCS", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_PROVIDER_ORDER_NER", raising=False)
    monkeypatch.setenv("CORPUSAGENT2_ENTITY_PROVIDER_MAX_DOCS", "2")
    store = InMemoryWorkingSetStore()
    store.record_working_set("run", "all_docs", [{"doc_id": f"doc-{idx}"} for idx in range(1, 4)])
    store.document_lookup.update(
        {
            "doc-1": {"doc_id": "doc-1", "title": "", "text": "Alice met Bob.", "published_at": "2022-01-01", "source": "NZZ"},
            "doc-2": {"doc_id": "doc-2", "title": "", "text": "Carol visited Zurich.", "published_at": "2022-01-02", "source": "NZZ"},
            "doc-3": {"doc_id": "doc-3", "title": "", "text": "Daniel spoke in Bern.", "published_at": "2022-01-03", "source": "TA"},
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

    result = _ner({}, {"fetch": fetch_preview}, context)

    assert result.metadata["analyzed_document_count"] == 3
    assert result.metadata["analysis_document_limit"] is None
    assert result.metadata["provider"] == "regex"
    assert not any("capped" in caveat.lower() for caveat in result.caveats)
    assert any("provider ner was skipped" in caveat.lower() for caveat in result.caveats)


def test_extract_keyterms_groups_by_outlet(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_EmptySearchBackend(),
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )
    deps = {
        "fetch": ToolExecutionResult(
            payload={
                "documents": [
                    {
                        "doc_id": "nzz-1",
                        "title": "Football",
                        "text": "football club tactics player transfer match tactics club",
                        "published_at": "2022-01-01",
                        "source": "NZZ",
                        "outlet": "NZZ",
                    },
                    {
                        "doc_id": "nzz-2",
                        "title": "Football",
                        "text": "football club player tactics league match club",
                        "published_at": "2022-01-02",
                        "source": "NZZ",
                        "outlet": "NZZ",
                    },
                    {
                        "doc_id": "ta-1",
                        "title": "Football",
                        "text": "football fan stadium derby ticket supporter stadium",
                        "published_at": "2022-01-03",
                        "source": "Tages-Anzeiger",
                        "outlet": "Tages-Anzeiger",
                    },
                    {
                        "doc_id": "ta-2",
                        "title": "Football",
                        "text": "football supporter stadium crowd ticket derby crowd",
                        "published_at": "2022-01-04",
                        "source": "Tages-Anzeiger",
                        "outlet": "Tages-Anzeiger",
                    },
                ],
                "document_count": 4,
                "returned_document_count": 4,
                "documents_truncated": False,
            }
        )
    }

    result = _extract_keyterms({"group_by": "outlet", "top_k": 5}, deps, context)

    outlets = {row["outlet"] for row in result.payload["rows"]}
    assert {"NZZ", "Tages-Anzeiger"}.issubset(outlets)
    assert all(row["document_count"] == 2 for row in result.payload["rows"] if row["outlet"] in outlets)
    assert result.metadata["group_by"] == "outlet"


def test_extract_keyterms_streams_truncated_working_set(tmp_path: Path) -> None:
    store = InMemoryWorkingSetStore()
    store.record_working_set("run", "all_docs", [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}])
    store.document_lookup.update(
        {
            "doc-1": {
                "doc_id": "doc-1",
                "title": "",
                "text": "preview only climate policy",
                "published_at": "2022-01-01",
                "source": "NZZ",
            },
            "doc-2": {
                "doc_id": "doc-2",
                "title": "",
                "text": "emissions law referendum climate emissions law",
                "published_at": "2022-01-02",
                "source": "Tages-Anzeiger",
            },
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
            "document_count": 2,
            "returned_document_count": 1,
            "documents_truncated": True,
        }
    )

    result = _extract_keyterms({"top_k": 10}, {"fetch": fetch_preview}, context)

    terms = {row["term"] for row in result.payload["rows"]}
    assert {"emissions", "law"}.intersection(terms)
    assert result.metadata["documents_from"] == "working_set_ref"
    assert result.metadata["analyzed_document_count"] == 2
    assert any("preview" in caveat.lower() for caveat in result.caveats)


def test_join_external_series_returns_standalone_market_rows_when_internal_rows_empty(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    def fake_fetch_yfinance_series_rows(*, ticker: str, start: str, end: str, interval: str = "1d") -> list[dict]:
        captured.update({"ticker": ticker, "start": start, "end": end, "interval": interval})
        return [
            {"ticker": ticker, "date": "2018-01-01", "time_bin": "2018-01", "market_close": 60.0},
            {"ticker": ticker, "date": "2018-02-01", "time_bin": "2018-02", "market_close": 64.0},
        ]

    monkeypatch.setattr(agent_capabilities, "_fetch_yfinance_series_rows", fake_fetch_yfinance_series_rows)
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_EmptySearchBackend(),
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
        state=SimpleNamespace(question="How did the oil price change?", rewritten_question="How did the oil price change?"),
    )
    deps = {
        "series": ToolExecutionResult(payload={"rows": []}),
        "fetch": ToolExecutionResult(
            payload={
                "documents": [
                    {"doc_id": "doc-1", "published_at": "2018-01-15", "text": "oil price rose"},
                    {"doc_id": "doc-2", "published_at": "2018-02-20", "text": "oil price fell"},
                ]
            }
        ),
    }

    result = _join_external_series({"ticker": "CL=F", "left_key": "time_bin", "right_key": "time_bin"}, deps, context)

    assert [row["market_close"] for row in result.payload["rows"]] == [60.0, 64.0]
    assert captured["start"] == "2018-01-15"
    assert captured["end"] == "2018-02-21"
    assert any("standalone" in caveat.lower() for caveat in result.caveats)


def test_noun_distribution_streaming_filters_function_words(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agent_capabilities, "_load_spacy_model", lambda: None)
    store = InMemoryWorkingSetStore()
    working_rows = [{"doc_id": "doc-1", "rank": 1, "score": 1.0}]
    store.record_working_set("run", "all_docs", working_rows)
    store.document_lookup["doc-1"] = {
        "doc_id": "doc-1",
        "title": "",
        "text": "was his said but are has they will now how take right var try catch error team game soccer soccer player",
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
    fetch_preview = ToolExecutionResult(
        payload={
            "documents": [],
            "working_set_ref": "all_docs",
            "document_count": 1,
            "returned_document_count": 0,
            "documents_truncated": True,
        }
    )

    result = _build_evidence_table({"task": "noun_frequency_distribution", "top_k": 10}, {"fetch": fetch_preview}, context)

    lemmas = {row["lemma"] for row in result.payload["rows"]}
    assert {"team", "game", "soccer", "player"}.issubset(lemmas)
    assert not {"was", "his", "said", "but", "are", "has", "they", "will", "now", "how", "take", "right"}.intersection(lemmas)
    assert not {"var", "try", "catch", "error"}.intersection(lemmas)


def test_noun_distribution_streams_id_only_working_set(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agent_capabilities, "_load_spacy_model", lambda: None)
    store = InMemoryWorkingSetStore()
    store.record_working_set("run", "all_docs", [{"doc_id": "doc-1"}])
    store.document_lookup["doc-1"] = {
        "doc_id": "doc-1",
        "title": "Football report",
        "text": "Club player scored goal",
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
    working_set = ToolExecutionResult(
        payload={
            "working_set_ref": "all_docs",
            "working_set_doc_ids": ["doc-1"],
            "document_count": 1,
            "preview_count": 1,
            "working_set_truncated": False,
        }
    )

    result = _build_evidence_table({"task": "noun_frequency_distribution", "top_k": 10}, {"n2": working_set}, context)

    lemmas = {row["lemma"] for row in result.payload["rows"]}
    assert {"football", "club", "player"}.issubset(lemmas)
    assert result.metadata["full_working_set"] is True


def test_noun_distribution_resolves_node_id_working_set_ref(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agent_capabilities, "_load_spacy_model", lambda: None)
    store = InMemoryWorkingSetStore()
    store.record_working_set("run", "all_docs", [{"doc_id": "doc-1"}])
    store.document_lookup["doc-1"] = {
        "doc_id": "doc-1",
        "title": "Football report",
        "text": "Club player scored goal",
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
    fetch_preview = ToolExecutionResult(
        payload={
            "documents": [],
            "working_set_ref": "all_docs",
            "document_count": 1,
            "returned_document_count": 0,
            "documents_truncated": True,
        }
    )

    result = _build_evidence_table(
        {"task": "noun_frequency_distribution", "working_set_ref": "n2", "top_k": 10},
        {"fetch": fetch_preview},
        context,
    )

    assert result.payload["working_set_ref"] == "all_docs"
    assert result.payload["analyzed_document_count"] == 1
    assert {"club", "player"}.issubset({row["lemma"] for row in result.payload["rows"]})


def test_large_working_set_noun_distribution_can_skip_spacy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CORPUSAGENT2_NOUN_SPACY_MAX_DOCS", "0")
    monkeypatch.setattr(
        agent_capabilities,
        "_load_spacy_model",
        lambda: (_ for _ in ()).throw(AssertionError("spaCy should not load for large working-set fallback")),
    )
    store = InMemoryWorkingSetStore()
    store.record_working_set("run", "all_docs", [{"doc_id": "doc-1"}])
    store.document_lookup["doc-1"] = {
        "doc_id": "doc-1",
        "title": "Football report",
        "text": "Club player scored goal",
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

    rows, metadata = agent_capabilities._noun_frequency_rows_from_working_set(context, "all_docs", top_k=10)

    assert rows
    assert metadata["provider"] == "heuristic_batch"
    assert "exceeds CORPUSAGENT2_NOUN_SPACY_MAX_DOCS" in metadata["provider_fallback_reason"]


def test_large_working_set_noun_distribution_uses_uncapped_sql_when_enabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CORPUSAGENT2_WORKING_SET_ANALYSIS_MAX_DOCS", raising=False)
    monkeypatch.setenv("CORPUSAGENT2_USE_SQL_TOKEN_AGGREGATE", "true")
    monkeypatch.setattr(agent_capabilities, "_count_working_set", lambda context, label, fallback=0: 50000)
    captured: dict[str, object] = {}

    def fake_sql(context, working_set_ref, *, top_k, max_documents=None):
        captured["max_documents"] = max_documents
        return [{"lemma": "club", "count": 10}], {"provider": "postgres_token_aggregate", "analyzed_document_count": 50000}

    monkeypatch.setattr(agent_capabilities, "_sql_noun_frequency_rows_from_working_set", fake_sql)
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_EmptySearchBackend(),
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )

    rows, metadata = agent_capabilities._noun_frequency_rows_from_working_set(context, "all_docs", top_k=10)

    assert rows == [{"lemma": "club", "count": 10}]
    assert captured["max_documents"] is None
    assert metadata["provider"] == "postgres_token_aggregate"


def test_working_set_analysis_default_is_uncapped(monkeypatch) -> None:
    monkeypatch.delenv("CORPUSAGENT2_WORKING_SET_ANALYSIS_MAX_DOCS", raising=False)

    assert agent_capabilities._working_set_analysis_max_documents() is None


def test_large_working_set_noun_distribution_respects_explicit_sql_cap(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CORPUSAGENT2_WORKING_SET_ANALYSIS_MAX_DOCS", "123")
    monkeypatch.setenv("CORPUSAGENT2_USE_SQL_TOKEN_AGGREGATE", "true")
    monkeypatch.setattr(agent_capabilities, "_count_working_set", lambda context, label, fallback=0: 50000)
    captured: dict[str, object] = {}

    def fake_sql(context, working_set_ref, *, top_k, max_documents=None):
        captured["max_documents"] = max_documents
        return [{"lemma": "club", "count": 10}], {"provider": "postgres_token_aggregate", "analyzed_document_count": 123}

    monkeypatch.setattr(agent_capabilities, "_sql_noun_frequency_rows_from_working_set", fake_sql)
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=_EmptySearchBackend(),
        working_store=InMemoryWorkingSetStore(),
        runtime=None,
    )

    agent_capabilities._noun_frequency_rows_from_working_set(context, "all_docs", top_k=10)

    assert captured["max_documents"] == 123


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


def test_fetch_documents_prefers_working_set_ref_over_preview_ids(tmp_path: Path) -> None:
    store = InMemoryWorkingSetStore()
    store.record_documents(
        "run",
        [
            {"doc_id": "doc-1", "text": "one"},
            {"doc_id": "doc-2", "text": "two"},
            {"doc_id": "doc-3", "text": "three"},
        ],
    )
    store.record_working_set(
        "run",
        "full_set",
        [{"doc_id": "doc-1", "score": 0.9}, {"doc_id": "doc-2", "score": 0.7}, {"doc_id": "doc-3", "score": 0.5}],
    )
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=store,
        runtime=None,
    )
    deps = {
        "working_set": ToolExecutionResult(
            payload={
                "working_set_ref": "full_set",
                "working_set_doc_ids": ["doc-1"],
                "document_count": 3,
                "preview_count": 1,
                "working_set_truncated": True,
            }
        )
    }

    result = _fetch_documents({"batch_size": 10}, deps, context)

    assert [row["doc_id"] for row in result.payload["documents"]] == ["doc-1", "doc-2", "doc-3"]
    assert result.payload["document_count"] == 3
    assert result.payload["returned_document_count"] == 3
    assert result.payload["documents_truncated"] is False
    assert result.payload["documents"][0]["score"] == 0.9


def test_join_external_series_can_fetch_market_data(monkeypatch, tmp_path: Path) -> None:
    def _fake_series(**kwargs):
        assert kwargs["ticker"] == "META"
        assert kwargs["start"] == "2018-03-01"
        assert kwargs["end"] == "2018-05-01"
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


def test_join_external_series_aggregates_daily_rows_to_time_bin(monkeypatch, tmp_path: Path) -> None:
    def _fake_series(**kwargs):
        assert kwargs["ticker"] == "CL=F"
        assert kwargs["start"] == "2018-03-01"
        assert kwargs["end"] == "2018-04-01"
        return [
            {
                "ticker": "CL=F",
                "date": "2018-03-01",
                "time_bin": "2018-03",
                "market_open": 60.0,
                "market_high": 62.0,
                "market_low": 59.0,
                "market_close": 61.0,
                "market_volume": 10,
                "market_return": 0.01,
                "market_drawdown": -0.02,
            },
            {
                "ticker": "CL=F",
                "date": "2018-03-30",
                "time_bin": "2018-03",
                "market_open": 61.0,
                "market_high": 65.0,
                "market_low": 60.0,
                "market_close": 64.0,
                "market_volume": 15,
                "market_return": 0.02,
                "market_drawdown": -0.01,
            },
        ]

    monkeypatch.setattr(agent_capabilities, "_fetch_yfinance_series_rows", _fake_series)
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=_ExplodingStore(),
        runtime=None,
    )
    deps = {"series": ToolExecutionResult(payload={"rows": [{"entity": "__all__", "time_bin": "2018-03", "count": 10}]})}

    result = _join_external_series({"ticker": "CL=F", "left_key": "time_bin", "right_key": "time_bin"}, deps, context)

    assert len(result.payload["rows"]) == 1
    assert result.payload["rows"][0]["market_open"] == 60.0
    assert result.payload["rows"][0]["market_close"] == 64.0
    assert result.payload["rows"][0]["market_volume"] == 25
    assert any("Aggregated external series" in caveat for caveat in result.caveats)


def test_infer_market_ticker_maps_oil_price_to_crude_futures() -> None:
    assert agent_capabilities._infer_market_ticker_from_text("How did the oil price change in America?") == "CL=F"


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

    assert any(
        row["entity"] == "negative" and row["time_bin"] == "2018-03" and row["count"] == -1.4
        for row in result.payload["rows"]
    )
    assert any(row["time_bin"] == "2018-04" for row in result.payload["rows"])


def test_time_series_aggregate_filters_invalid_entities_and_adds_normalized_value(tmp_path: Path) -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=tmp_path,
        search_backend=None,
        working_store=_ExplodingStore(),
        runtime=None,
    )
    deps = {
        "entities": ToolExecutionResult(
            payload={
                "rows": [
                    {"entity": "Greta Thunberg", "label": "PERSON", "doc_id": "d1", "published_at": "2018-03-01"},
                    {"entity": "2018", "label": "DATE", "doc_id": "d1", "published_at": "2018-03-01"},
                    {"entity": "#", "label": "ORG", "doc_id": "d2", "published_at": "2018-03-02"},
                ]
            }
        )
    }

    result = _time_series_aggregate(
        {"group_by": "entity", "entity_types": ["PERSON", "ORG"], "normalize_by": "documents_per_period"},
        deps,
        context,
    )

    assert result.payload["rows"] == [
        {
            "entity": "Greta Thunberg",
            "canonical_entity": "Greta Thunberg",
            "actor": "Greta Thunberg",
            "entity_label": "Greta Thunberg",
            "series_name": "Greta Thunberg",
            "time_bin": "2018-03",
            "bucket": "2018-03",
            "month": "2018-03",
            "period": "2018-03",
            "time_period": "2018-03",
            "count": 1,
            "mention_count": 1,
            "mention_count_normalized": 1.0,
            "normalized_value": 1.0,
        }
    ]
    assert result.payload["skipped_row_count"] == 2


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
