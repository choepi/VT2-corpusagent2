from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import importlib.util
from typing import Any

import pandas as pd

from .analysis_tools import (
    run_burst_detection,
    run_keyphrases,
    run_ner,
    run_sentiment,
    run_topics_over_time,
)
from .faithfulness import evaluate_claims_with_nli
from .retrieval_budgeting import infer_retrieval_budget
from .retrieval import (
    reciprocal_rank_fusion,
    rerank_cross_encoder,
    retrieve_dense,
    retrieve_dense_pgvector,
    retrieve_tfidf,
)
from .tool_registry import (
    CapabilityToolAdapter,
    SchemaDescriptor,
    ToolExecutionResult,
    ToolRegistry,
    ToolRequirement,
    ToolSpec,
)


def _runtime_from_context(context: Any) -> Any | None:
    if hasattr(context, "runtime"):
        return context.runtime
    if isinstance(context, dict):
        return context.get("runtime")
    return None


def _question_spec_from_context(context: Any) -> Any | None:
    if hasattr(context, "question_spec"):
        return context.question_spec
    if isinstance(context, dict):
        return context.get("question_spec")
    return None


def _time_range_from_params(params: dict[str, Any]) -> dict[str, str]:
    payload = params.get("time_range", {})
    if isinstance(payload, dict):
        return {
            "start": str(payload.get("start", "")).strip(),
            "end": str(payload.get("end", "")).strip(),
            "granularity": str(payload.get("granularity", "year")).strip() or "year",
        }
    return {"start": "", "end": "", "granularity": "year"}


def _granularity_from_params_or_question(params: dict[str, Any], question_spec: Any | None) -> str:
    time_range = _time_range_from_params(params)
    if time_range.get("granularity"):
        return time_range["granularity"]
    if question_spec is not None and getattr(question_spec, "time_range", None) is not None:
        return str(question_spec.time_range.granularity or "year")
    return "year"


def _apply_time_filter(rows: pd.DataFrame, params: dict[str, Any], time_column: str) -> pd.DataFrame:
    if rows.empty or time_column not in rows.columns:
        return rows
    time_range = _time_range_from_params(params)
    subset = rows.copy()
    if time_range["start"]:
        subset = subset[subset[time_column].astype(str) >= time_range["start"]]
    if time_range["end"]:
        subset = subset[subset[time_column].astype(str) <= time_range["end"]]
    return subset.reset_index(drop=True)


def _doc_results_to_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("results", [])
    return [str(row.get("doc_id")) for row in rows if str(row.get("doc_id", "")).strip()]


def _snippet(text: str, max_chars: int = 240) -> str:
    normalized = " ".join(str(text).split())
    return normalized[:max_chars]


def _evidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in rows:
        evidence.append(
            {
                "doc_id": str(row.get("doc_id", "")),
                "chunk_id": str(row.get("chunk_id", f"{row.get('doc_id', '')}:0")),
                "score_components": dict(row.get("score_components", {})),
                "span_offsets": None,
            }
        )
    return evidence


def _sort_artifact_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for column in ("count", "score", "weight", "intensity", "doc_freq", "n_docs"):
        if column in df.columns:
            return df.sort_values(column, ascending=False).reset_index(drop=True)
    return df.reset_index(drop=True)


def _filter_artifact_rows(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    subset = df.copy()
    caveats: list[str] = []
    entities = [str(item).strip() for item in params.get("entities", []) if str(item).strip()]
    time_candidates = [column for column in ("time_bin", "start", "published_at") if column in subset.columns]
    if time_candidates:
        subset = _apply_time_filter(subset, params, time_candidates[0])

    entity_columns = [column for column in ("entity", "entity_or_term", "phrase") if column in subset.columns]
    if entities and entity_columns:
        column = entity_columns[0]
        filtered = subset[subset[column].astype(str).isin(entities)].reset_index(drop=True)
        if filtered.empty:
            caveats.append(
                f"No exact entity match found in precomputed '{column}' rows; keeping broader artifact slice."
            )
        else:
            subset = filtered
    return _sort_artifact_rows(subset), caveats


class HybridRetrievalTool(CapabilityToolAdapter):
    def __init__(self) -> None:
        self.spec = ToolSpec(
            tool_name="hybrid_tfidf_dense_rerank",
            provider="corpusagent2",
            capabilities=["retrieve.documents"],
            input_schema=SchemaDescriptor(
                name="HybridRetrievalInput",
                fields={"query": "str", "top_k": "int", "lightweight": "bool"},
            ),
            output_schema=SchemaDescriptor(
                name="HybridRetrievalOutput",
                fields={"results": "list[RetrievedDocument]", "backend": "str"},
            ),
            requirements=[
                ToolRequirement("artifact", "lexical_assets", "Requires TF-IDF assets."),
                ToolRequirement("metadata", "doc_metadata", "Requires title/text metadata for snippets."),
            ],
            cost_class="medium",
            deterministic=True,
            languages_supported=["en"],
            priority=80,
            model_id="intfloat/e5-base-v2+cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        runtime = _runtime_from_context(context)
        if runtime is None:
            return False, ["missing runtime"]
        try:
            runtime.load_lexical_assets()
            runtime.load_metadata()
            if runtime.retrieval_backend == "local":
                runtime.load_dense_assets()
            return True, [f"retrieval_backend={runtime.retrieval_backend}"]
        except Exception as exc:
            return False, [str(exc)]

    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        runtime = _runtime_from_context(context)
        if runtime is None:
            raise RuntimeError("HybridRetrievalTool requires runtime context.")

        query = str(params.get("query", "")).strip()
        if not query:
            question_spec = _question_spec_from_context(context)
            query = str(getattr(question_spec, "raw_question", "")).strip()
        if not query:
            raise ValueError("Retrieval query is required.")

        budget = infer_retrieval_budget(query, inputs=params, lightweight=bool(params.get("lightweight", False)))
        top_k = budget.top_k
        lightweight = bool(params.get("lightweight", False))
        fusion_top_k = max(top_k * 10, 50) if not lightweight else max(top_k * 3, 15)
        lexical_vectorizer, lexical_matrix, lexical_doc_ids = runtime.load_lexical_assets()
        metadata_lookup = runtime.doc_lookup()

        tfidf = retrieve_tfidf(
            query=query,
            vectorizer=lexical_vectorizer,
            matrix=lexical_matrix,
            doc_ids=lexical_doc_ids,
            top_k=fusion_top_k,
        )
        if runtime.retrieval_backend == "local":
            dense_assets = runtime.load_dense_assets()
            if dense_assets is None:
                raise RuntimeError("Dense retrieval assets are unavailable.")
            dense_embeddings, dense_doc_ids = dense_assets
            dense = retrieve_dense(
                query=query,
                model_id=runtime.dense_model_id,
                embeddings=dense_embeddings,
                doc_ids=dense_doc_ids,
                top_k=fusion_top_k,
            )
        else:
            dense = retrieve_dense_pgvector(
                query=query,
                model_id=runtime.dense_model_id,
                dsn=runtime.pg_dsn,
                table_name=runtime.pg_table,
                top_k=fusion_top_k,
            )
        fused = reciprocal_rank_fusion({"tfidf": tfidf, "dense": dense})
        reranked = fused[:top_k] if lightweight else rerank_cross_encoder(
            query=query,
            candidates=fused[: min(len(fused), 150)],
            doc_text_by_id=runtime.doc_text_by_id(),
            model_id=runtime.rerank_model_id,
            top_k=top_k,
        )

        rows: list[dict[str, Any]] = []
        for item in reranked[:top_k]:
            metadata = metadata_lookup.get(item.doc_id, {})
            rows.append(
                {
                    "doc_id": item.doc_id,
                    "chunk_id": f"{item.doc_id}:0",
                    "title": str(metadata.get("title", "")),
                    "published_at": str(metadata.get("published_at", "")),
                    "snippet": _snippet(metadata.get("text", "")),
                    "score": float(item.score),
                    "score_components": dict(item.score_components),
                }
            )

        return ToolExecutionResult(
            payload={"results": rows, "backend": runtime.retrieval_backend, "query": query},
            evidence=_evidence_rows(rows),
            metadata={"retrieval_backend": runtime.retrieval_backend, "lightweight": lightweight},
        )


class DocumentFilterTool(CapabilityToolAdapter):
    def __init__(self) -> None:
        self.spec = ToolSpec(
            tool_name="document_focus_filter",
            provider="corpusagent2",
            capabilities=["filter.documents"],
            input_schema=SchemaDescriptor(
                name="DocumentFilterInput",
                fields={"entities": "list[str]", "time_range": "dict", "max_results": "int"},
            ),
            output_schema=SchemaDescriptor(
                name="DocumentFilterOutput",
                fields={"results": "list[RetrievedDocument]", "filter_summary": "dict"},
            ),
            cost_class="low",
            deterministic=True,
            languages_supported=["en"],
            priority=100,
        )

    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        return True, ["pure post-processing"]

    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        retrieval_payload = dependency_results["retrieval_results"].payload
        rows = list(retrieval_payload.get("results", []))
        entities = [str(item).strip() for item in params.get("entities", []) if str(item).strip()]
        max_results = int(params.get("max_results", 8))
        time_range = _time_range_from_params(params)

        filtered = rows
        if entities:
            entity_lower = [entity.lower() for entity in entities]
            filtered = [
                row
                for row in filtered
                if any(
                    entity in f"{row.get('title', '')} {row.get('snippet', '')}".lower()
                    for entity in entity_lower
                )
            ]

        if time_range["start"] or time_range["end"]:
            ranged: list[dict[str, Any]] = []
            for row in filtered:
                published_at = str(row.get("published_at", ""))
                if time_range["start"] and published_at and published_at < time_range["start"]:
                    continue
                if time_range["end"] and published_at and published_at > time_range["end"]:
                    continue
                ranged.append(row)
            filtered = ranged

        caveats: list[str] = []
        if not filtered:
            filtered = rows
            caveats.append("No focused document slice matched all requested filters; using top retrieval results instead.")

        filtered = filtered[:max_results]
        return ToolExecutionResult(
            payload={
                "results": filtered,
                "filter_summary": {
                    "requested_entities": entities,
                    "time_range": time_range,
                    "returned_results": len(filtered),
                },
            },
            evidence=_evidence_rows(filtered),
            caveats=caveats,
        )


class PrecomputedArtifactTool(CapabilityToolAdapter):
    def __init__(
        self,
        tool_name: str,
        capability: str,
        artifact_name: str,
        priority: int,
        entity_specific: bool = True,
    ) -> None:
        self.artifact_name = artifact_name
        self.entity_specific = entity_specific
        self.spec = ToolSpec(
            tool_name=tool_name,
            provider="corpusagent2",
            capabilities=[capability],
            input_schema=SchemaDescriptor(
                name=f"{artifact_name.title().replace('_', '')}Input",
                fields={"entities": "list[str]", "time_range": "dict"},
            ),
            output_schema=SchemaDescriptor(
                name=f"{artifact_name.title().replace('_', '')}Output",
                fields={"rows": "list[dict]", "artifact_name": "str"},
            ),
            requirements=[ToolRequirement("artifact", artifact_name, f"Requires {artifact_name}.parquet.")],
            cost_class="low",
            deterministic=True,
            languages_supported=["en"],
            priority=priority,
            model_id=artifact_name,
        )

    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        runtime = _runtime_from_context(context)
        if runtime is None:
            return False, ["missing runtime"]
        if not runtime.artifact_available(self.artifact_name):
            return False, [f"artifact missing: {self.artifact_name}"]
        entities = [str(item).strip() for item in params.get("entities", []) if str(item).strip()]
        if entities and not self.entity_specific:
            return False, [f"{self.artifact_name} is corpus-level only for entity-specific questions"]
        return True, [f"artifact={self.artifact_name}"]

    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        runtime = _runtime_from_context(context)
        if runtime is None:
            raise RuntimeError("PrecomputedArtifactTool requires runtime context.")
        df = runtime.load_artifact(self.artifact_name)
        filtered, caveats = _filter_artifact_rows(df=df, params=params)
        rows = filtered.head(50).to_dict(orient="records")
        return ToolExecutionResult(
            payload={"rows": rows, "artifact_name": self.artifact_name},
            artifacts=[str(runtime.artifact_path(self.artifact_name))],
            caveats=caveats,
            metadata={"artifact_name": self.artifact_name},
        )


class QueryTimeEntityTrendTool(CapabilityToolAdapter):
    def __init__(self) -> None:
        self.spec = ToolSpec(
            tool_name="spacy_entity_trend_on_retrieval_window",
            provider="spacy",
            capabilities=["analyze.entity_trend"],
            input_schema=SchemaDescriptor(
                name="EntityTrendWindowInput",
                fields={"entities": "list[str]", "time_range": "dict"},
            ),
            output_schema=SchemaDescriptor(
                name="EntityTrendWindowOutput",
                fields={"rows": "list[dict]", "analysis_scope": "str"},
            ),
            requirements=[ToolRequirement("library", "spacy", "Falls back to regex if spaCy is unavailable.")],
            cost_class="medium",
            deterministic=True,
            languages_supported=["en"],
            priority=60,
            fallback_of="precomputed_entity_trend",
            model_id="en_core_web_trf",
        )

    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        runtime = _runtime_from_context(context)
        if runtime is None:
            return False, ["missing runtime"]
        return True, ["retrieval-window fallback"]

    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        runtime = _runtime_from_context(context)
        doc_ids = _doc_results_to_ids(dependency_results["filtered_results"].payload)
        docs_df = runtime.load_docs(doc_ids)
        granularity = _granularity_from_params_or_question(params, _question_spec_from_context(context))
        rows = run_ner(
            df=docs_df,
            model_name="en_core_web_trf",
            granularity=granularity,
            max_docs=min(len(docs_df), 500) if len(docs_df) else 0,
        )
        filtered, caveats = _filter_artifact_rows(rows, params)
        return ToolExecutionResult(
            payload={"rows": filtered.head(50).to_dict(orient="records"), "analysis_scope": "retrieval_window"},
            caveats=caveats,
            metadata={"rows": int(filtered.shape[0])},
        )


class QueryTimeSentimentTool(CapabilityToolAdapter):
    def __init__(self) -> None:
        self.spec = ToolSpec(
            tool_name="transformers_sentiment_on_retrieval_window",
            provider="transformers",
            capabilities=["analyze.sentiment_series"],
            input_schema=SchemaDescriptor(
                name="SentimentWindowInput",
                fields={"entities": "list[str]", "time_range": "dict"},
            ),
            output_schema=SchemaDescriptor(
                name="SentimentWindowOutput",
                fields={"rows": "list[dict]", "analysis_scope": "str"},
            ),
            requirements=[ToolRequirement("library", "transformers", "Runs sentiment over retrieved evidence window.")],
            cost_class="medium",
            deterministic=True,
            languages_supported=["en"],
            priority=55,
            fallback_of="precomputed_sentiment_series",
            model_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )

    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        runtime = _runtime_from_context(context)
        if runtime is None:
            return False, ["missing runtime"]
        return True, ["retrieval-window fallback"]

    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        runtime = _runtime_from_context(context)
        doc_ids = _doc_results_to_ids(dependency_results["filtered_results"].payload)
        docs_df = runtime.load_docs(doc_ids)
        granularity = _granularity_from_params_or_question(params, _question_spec_from_context(context))
        rows, device_used = run_sentiment(
            df=docs_df,
            model_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
            granularity=granularity,
            preferred_device="cpu",
            max_docs=min(len(docs_df), 256) if len(docs_df) else 0,
        )
        filtered = _apply_time_filter(rows, params, "time_bin")
        caveats = ["Sentiment was estimated on the retrieved evidence window, not the full corpus slice."]
        return ToolExecutionResult(
            payload={"rows": filtered.to_dict(orient="records"), "analysis_scope": "retrieval_window"},
            caveats=caveats,
            metadata={"device_used": device_used},
        )


class QueryTimeTopicsTool(CapabilityToolAdapter):
    def __init__(self) -> None:
        self.spec = ToolSpec(
            tool_name="bertopic_topics_on_retrieval_window",
            provider="bertopic",
            capabilities=["analyze.topics_over_time"],
            input_schema=SchemaDescriptor(
                name="TopicsWindowInput",
                fields={"time_range": "dict"},
            ),
            output_schema=SchemaDescriptor(
                name="TopicsWindowOutput",
                fields={"rows": "list[dict]", "analysis_scope": "str"},
            ),
            requirements=[ToolRequirement("library", "bertopic", "Runs BERTopic on the retrieved evidence window.")],
            cost_class="high",
            deterministic=False,
            languages_supported=["en"],
            priority=30,
            fallback_of="precomputed_topics_over_time",
        )

    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        runtime = _runtime_from_context(context)
        if runtime is None:
            return False, ["missing runtime"]
        return True, ["retrieval-window fallback"]

    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        runtime = _runtime_from_context(context)
        doc_ids = _doc_results_to_ids(dependency_results["filtered_results"].payload)
        docs_df = runtime.load_docs(doc_ids)
        granularity = _granularity_from_params_or_question(params, _question_spec_from_context(context))
        rows = run_topics_over_time(
            df=docs_df,
            granularity=granularity,
            max_docs=min(len(docs_df), 128) if len(docs_df) else 0,
        )
        filtered = _apply_time_filter(rows, params, "time_bin")
        caveats = ["Topic clusters were induced from the retrieved evidence window and may not reflect the full corpus."]
        return ToolExecutionResult(
            payload={"rows": filtered.to_dict(orient="records"), "analysis_scope": "retrieval_window"},
            caveats=caveats,
        )


class QueryTimeKeyphraseTool(CapabilityToolAdapter):
    def __init__(self) -> None:
        self.spec = ToolSpec(
            tool_name="textrank_keyphrases_on_retrieval_window",
            provider="corpusagent2",
            capabilities=["analyze.keyphrases"],
            input_schema=SchemaDescriptor(
                name="KeyphraseWindowInput",
                fields={"time_range": "dict"},
            ),
            output_schema=SchemaDescriptor(
                name="KeyphraseWindowOutput",
                fields={"rows": "list[dict]", "analysis_scope": "str"},
            ),
            requirements=[ToolRequirement("library", "networkx", "Runs TextRank keyphrases over retrieved docs.")],
            cost_class="low",
            deterministic=True,
            languages_supported=["en"],
            priority=65,
            fallback_of="precomputed_keyphrases",
        )

    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        runtime = _runtime_from_context(context)
        if runtime is None:
            return False, ["missing runtime"]
        return True, ["retrieval-window fallback"]

    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        runtime = _runtime_from_context(context)
        doc_ids = _doc_results_to_ids(dependency_results["filtered_results"].payload)
        docs_df = runtime.load_docs(doc_ids)
        granularity = _granularity_from_params_or_question(params, _question_spec_from_context(context))
        rows = run_keyphrases(
            df=docs_df,
            granularity=granularity,
            top_k=20,
            max_docs_per_bin=min(len(docs_df), 200) if len(docs_df) else 0,
        )
        filtered = _apply_time_filter(rows, params, "time_bin")
        return ToolExecutionResult(
            payload={"rows": filtered.to_dict(orient="records"), "analysis_scope": "retrieval_window"},
        )


class BurstDetectionTool(CapabilityToolAdapter):
    def __init__(self) -> None:
        self.spec = ToolSpec(
            tool_name="burst_detection_from_entity_rows",
            provider="corpusagent2",
            capabilities=["analyze.burst_events"],
            input_schema=SchemaDescriptor(
                name="BurstDetectionInput",
                fields={"time_range": "dict"},
            ),
            output_schema=SchemaDescriptor(
                name="BurstDetectionOutput",
                fields={"rows": "list[dict]", "analysis_scope": "str"},
            ),
            requirements=[ToolRequirement("input", "entity_trend", "Requires entity trend rows from dependency or artifact.")],
            cost_class="low",
            deterministic=True,
            languages_supported=["en"],
            priority=70,
            fallback_of="precomputed_burst_events",
        )

    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        return True, ["can operate on dependency rows"]

    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        source_payload = None
        if "entity_trend" in dependency_results:
            source_payload = dependency_results["entity_trend"].payload
        elif "filtered_results" in dependency_results:
            source_payload = {"rows": []}
        rows = pd.DataFrame(source_payload.get("rows", [])) if isinstance(source_payload, dict) else pd.DataFrame()
        bursts = run_burst_detection(rows)
        filtered, caveats = _filter_artifact_rows(bursts, params)
        return ToolExecutionResult(
            payload={"rows": filtered.to_dict(orient="records"), "analysis_scope": "computed_from_entity_rows"},
            caveats=caveats,
        )


def _summarize_retrieval(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    rows = payload.get("results", [])
    highlights: list[str] = []
    claims: list[str] = []
    if not rows:
        return ["No retrieved documents were available."], []
    for row in rows[:3]:
        title = str(row.get("title", "")).strip()
        published_at = str(row.get("published_at", "")).strip()
        snippet = str(row.get("snippet", "")).strip()
        text = title or snippet
        if published_at:
            highlights.append(f"Evidence includes {text} ({published_at}).")
        else:
            highlights.append(f"Evidence includes {text}.")
    if rows:
        first = rows[0]
        if first.get("title"):
            claims.append(f"The strongest retrieved evidence centers on {first['title']}.")
    return highlights[:3], claims[:1]


def _summarize_entity_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    if not rows:
        return ["No entity trend rows were produced."], []
    totals: dict[str, float] = defaultdict(float)
    peak_bin_by_entity: dict[str, tuple[str, float]] = {}
    for row in rows:
        entity = str(row.get("entity", ""))
        count = float(row.get("count", 0.0))
        totals[entity] += count
        current = peak_bin_by_entity.get(entity)
        time_bin = str(row.get("time_bin", ""))
        if current is None or count > current[1]:
            peak_bin_by_entity[entity] = (time_bin, count)
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    top_entity, total_count = ranked[0]
    peak_bin, peak_count = peak_bin_by_entity[top_entity]
    highlights = [
        f"Entity activity is led by {top_entity} with total count {int(total_count)}.",
        f"{top_entity} peaks in {peak_bin} with count {int(peak_count)}.",
    ]
    claims = [f"{top_entity} is the dominant entity in the analyzed evidence slice."]
    return highlights, claims


def _summarize_sentiment_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    if not rows:
        return ["No sentiment rows were produced."], []
    sorted_rows = sorted(rows, key=lambda item: str(item.get("time_bin", "")))
    mean_score = sum(float(item.get("mean", 0.0)) for item in sorted_rows) / max(len(sorted_rows), 1)
    direction = "mixed"
    if mean_score > 0.15:
        direction = "positive"
    elif mean_score < -0.15:
        direction = "negative"
    strongest = max(sorted_rows, key=lambda item: abs(float(item.get("mean", 0.0))))
    strongest_bin = str(strongest.get("time_bin", ""))
    strongest_value = float(strongest.get("mean", 0.0))
    highlights = [
        f"Average sentiment over the analyzed slice is {direction} (mean={mean_score:.2f}).",
        f"The strongest sentiment movement appears in {strongest_bin} (mean={strongest_value:.2f}).",
    ]
    claims = [f"Sentiment in the analyzed evidence slice is predominantly {direction}."]
    return highlights, claims


def _summarize_topic_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    if not rows:
        return ["No topic rows were produced."], []
    top_row = max(rows, key=lambda item: float(item.get("weight", 0.0)))
    terms = str(top_row.get("top_terms", "")).strip()
    time_bin = str(top_row.get("time_bin", "")).strip()
    highlights = [f"A dominant topic in {time_bin} is characterized by terms: {terms}."]
    claims = [f"One dominant topic cluster is defined by the terms {terms}."]
    return highlights, claims


def _summarize_burst_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    if not rows:
        return ["No burst events were produced."], []
    top_row = max(rows, key=lambda item: float(item.get("intensity", 0.0)))
    entity = str(top_row.get("entity_or_term", "")).strip()
    start = str(top_row.get("start", "")).strip()
    end = str(top_row.get("end", "")).strip()
    intensity = float(top_row.get("intensity", 0.0))
    highlights = [f"The strongest burst is for {entity} from {start} to {end} (intensity={intensity:.2f})."]
    claims = [f"Coverage around {entity} shows a burst between {start} and {end}."]
    return highlights, claims


def _summarize_keyphrase_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    if not rows:
        return ["No keyphrase rows were produced."], []
    phrases = [str(row.get("phrase", "")).strip() for row in rows[:5] if str(row.get("phrase", "")).strip()]
    joined = ", ".join(phrases)
    highlights = [f"Prominent keyphrases include: {joined}."]
    claims = [f"Prominent keyphrases in the analyzed evidence include {joined}."]
    return highlights, claims


class FindingsAggregationTool(CapabilityToolAdapter):
    def __init__(self) -> None:
        self.spec = ToolSpec(
            tool_name="grounded_findings_aggregator",
            provider="corpusagent2",
            capabilities=["aggregate.findings"],
            input_schema=SchemaDescriptor(
                name="FindingsAggregationInput",
                fields={"question_class": "str", "raw_question": "str", "expected_output_types": "list[str]"},
            ),
            output_schema=SchemaDescriptor(
                name="FindingsAggregationOutput",
                fields={"highlights": "list[str]", "candidate_claims": "list[str]", "analysis": "dict"},
            ),
            cost_class="low",
            deterministic=True,
            languages_supported=["en"],
            priority=100,
        )

    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        return True, ["deterministic aggregation"]

    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        highlights: list[str] = []
        candidate_claims: list[str] = []
        artifacts_used: list[str] = []
        caveats: list[str] = []
        unsupported_parts: list[str] = []
        analysis: dict[str, Any] = {}

        if "filtered_results" in dependency_results:
            retrieval_payload = dependency_results["filtered_results"].payload
            retrieval_highlights, retrieval_claims = _summarize_retrieval(retrieval_payload)
            highlights.extend(retrieval_highlights)
            candidate_claims.extend(retrieval_claims)
            analysis["retrieval"] = retrieval_payload

        summarizers = {
            "entity_trend": _summarize_entity_rows,
            "sentiment_series": _summarize_sentiment_rows,
            "topics_over_time": _summarize_topic_rows,
            "burst_events": _summarize_burst_rows,
            "keyphrases": _summarize_keyphrase_rows,
        }

        for output_key, summarizer in summarizers.items():
            if output_key not in dependency_results:
                continue
            payload = dependency_results[output_key].payload
            rows = payload.get("rows", []) if isinstance(payload, dict) else []
            summary_highlights, summary_claims = summarizer(rows)
            highlights.extend(summary_highlights)
            candidate_claims.extend(summary_claims)
            analysis[output_key] = payload

        for result in dependency_results.values():
            artifacts_used.extend(result.artifacts)
            caveats.extend(result.caveats)
            unsupported_parts.extend(result.unsupported_parts)

        if not highlights:
            unsupported_parts.append("No intermediate analytical findings were available for synthesis.")

        deduped_highlights = list(dict.fromkeys(item for item in highlights if item))
        deduped_claims = list(dict.fromkeys(item for item in candidate_claims if item))
        return ToolExecutionResult(
            payload={
                "question_class": params.get("question_class", ""),
                "raw_question": params.get("raw_question", ""),
                "highlights": deduped_highlights[:8],
                "candidate_claims": deduped_claims[:3],
                "analysis": analysis,
                "artifacts_used": list(dict.fromkeys(artifacts_used)),
            },
            caveats=list(dict.fromkeys(caveats)),
            unsupported_parts=list(dict.fromkeys(unsupported_parts)),
            metadata={"expected_output_types": list(params.get("expected_output_types", []))},
        )


class ClaimVerificationTool(CapabilityToolAdapter):
    def __init__(self) -> None:
        self.spec = ToolSpec(
            tool_name="nli_claim_verifier",
            provider="transformers",
            capabilities=["verify.claims"],
            input_schema=SchemaDescriptor(
                name="ClaimVerificationInput",
                fields={"question_class": "str", "max_claims": "int"},
            ),
            output_schema=SchemaDescriptor(
                name="ClaimVerificationOutput",
                fields={"summary": "dict", "claim_verdicts": "list[dict]"},
            ),
            requirements=[ToolRequirement("model", "roberta-large-mnli", "Requires NLI verifier model.")],
            cost_class="medium",
            deterministic=True,
            languages_supported=["en"],
            priority=90,
            model_id="FacebookAI/roberta-large-mnli",
        )

    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        runtime = _runtime_from_context(context)
        if runtime is None:
            return False, ["missing runtime"]
        return True, ["verifier available lazily"]

    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        runtime = _runtime_from_context(context)
        aggregate_payload = dependency_results["aggregate_summary"].payload
        filtered_payload = dependency_results["filtered_results"].payload
        candidate_claims = list(aggregate_payload.get("candidate_claims", []))[: int(params.get("max_claims", 3))]
        evidence_doc_ids = _doc_results_to_ids(filtered_payload)[:3]
        if not candidate_claims or not evidence_doc_ids:
            return ToolExecutionResult(
                payload={"summary": {"total_claims": 0}, "claim_verdicts": []},
                caveats=["No candidate claims or evidence documents were available for verification."],
            )

        claim_rows = [
            {
                "claim_id": f"claim_{idx}",
                "claim": claim,
                "evidence_doc_ids": evidence_doc_ids,
                "category": "A",
            }
            for idx, claim in enumerate(candidate_claims, start=1)
        ]
        verdicts, summary = evaluate_claims_with_nli(
            verifier=runtime.get_verifier(),
            claims=claim_rows,
            doc_text_by_id=runtime.doc_text_by_id(),
        )
        return ToolExecutionResult(
            payload={
                "summary": summary,
                "claim_verdicts": [item.to_dict() for item in verdicts],
            },
            evidence=[
                {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}:0",
                    "score_components": {},
                    "span_offsets": None,
                }
                for doc_id in evidence_doc_ids
            ],
            metadata={"claim_count": len(candidate_claims)},
        )


class GroundedSynthesisTool(CapabilityToolAdapter):
    def __init__(self) -> None:
        self.spec = ToolSpec(
            tool_name="grounded_answer_synthesizer",
            provider="corpusagent2",
            capabilities=["synthesize.answer"],
            input_schema=SchemaDescriptor(
                name="GroundedSynthesisInput",
                fields={"question_text": "str", "question_class": "str"},
            ),
            output_schema=SchemaDescriptor(
                name="GroundedSynthesisOutput",
                fields={
                    "answer_text": "str",
                    "evidence_items": "list[dict]",
                    "artifacts_used": "list[str]",
                    "unsupported_parts": "list[str]",
                    "caveats": "list[str]",
                    "claim_verdicts": "list[dict]",
                },
            ),
            cost_class="low",
            deterministic=True,
            languages_supported=["en"],
            priority=100,
        )

    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        return True, ["deterministic templated synthesis"]

    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        if params.get("mode") == "unsupported":
            question_spec = _question_spec_from_context(context)
            unsupported = list(getattr(question_spec, "unsupported_reasons", []))
            payload = {
                "answer_text": "The current corpus assets cannot answer this question reliably.",
                "evidence_items": [],
                "artifacts_used": [],
                "unsupported_parts": unsupported,
                "caveats": ["No execution plan was run because feasibility failed before planning."],
                "claim_verdicts": [],
            }
            return ToolExecutionResult(payload=payload, unsupported_parts=unsupported, caveats=payload["caveats"])

        aggregate_payload = dependency_results["aggregate_summary"].payload
        verification_payload = dependency_results.get("claim_verification", ToolExecutionResult(payload={})).payload
        filtered_payload = dependency_results["filtered_results"].payload
        question_text = str(params.get("question_text", "")).strip()
        highlights = list(aggregate_payload.get("highlights", []))
        claim_verdicts = list(verification_payload.get("claim_verdicts", []))
        caveats = []
        unsupported_parts = []
        artifacts_used = list(aggregate_payload.get("artifacts_used", []))

        for result in dependency_results.values():
            caveats.extend(result.caveats)
            unsupported_parts.extend(result.unsupported_parts)
            artifacts_used.extend(result.artifacts)

        evidence_items = []
        for row in filtered_payload.get("results", [])[:5]:
            evidence_items.append(
                {
                    "doc_id": str(row.get("doc_id", "")),
                    "chunk_id": str(row.get("chunk_id", "")),
                    "title": str(row.get("title", "")),
                    "snippet": str(row.get("snippet", "")),
                    "score": float(row.get("score", 0.0)),
                    "score_components": dict(row.get("score_components", {})),
                    "published_at": str(row.get("published_at", "")),
                }
            )

        parts = [f"Question: {question_text}"]
        if highlights:
            parts.append("Findings: " + " ".join(highlights[:4]))
        if claim_verdicts:
            verdict_labels = defaultdict(int)
            for row in claim_verdicts:
                verdict_labels[str(row.get("label", "unknown"))] += 1
            summary = ", ".join(f"{label}={count}" for label, count in sorted(verdict_labels.items()))
            parts.append(f"Verification summary: {summary}.")
        if unsupported_parts:
            parts.append("Unsupported parts were isolated rather than filled in speculatively.")

        payload = {
            "answer_text": " ".join(part for part in parts if part).strip(),
            "evidence_items": evidence_items,
            "artifacts_used": list(dict.fromkeys(artifacts_used)),
            "unsupported_parts": list(dict.fromkeys(unsupported_parts)),
            "caveats": list(dict.fromkeys(caveats)),
            "claim_verdicts": claim_verdicts,
        }
        return ToolExecutionResult(
            payload=payload,
            evidence=_evidence_rows(evidence_items),
            artifacts=payload["artifacts_used"],
            caveats=payload["caveats"],
            unsupported_parts=payload["unsupported_parts"],
        )


class LibraryCapabilityTool(CapabilityToolAdapter):
    """Generic provider-family adapter used for capability discovery/swapability."""

    def __init__(
        self,
        *,
        tool_name: str,
        provider: str,
        capability: str,
        import_name: str,
        priority: int,
        fallback_of: str | None = None,
    ) -> None:
        self.import_name = import_name
        self.capability = capability
        self.spec = ToolSpec(
            tool_name=tool_name,
            provider=provider,
            capabilities=[capability],
            input_schema=SchemaDescriptor(name=f"{tool_name}_input", fields={"text": "str", "params": "dict"}),
            output_schema=SchemaDescriptor(name=f"{tool_name}_output", fields={"result": "dict"}),
            requirements=[ToolRequirement("library", import_name, f"Requires python package '{import_name}'.")],
            cost_class="low",
            deterministic=True,
            languages_supported=["en"],
            priority=priority,
            fallback_of=fallback_of,
            model_id=import_name,
        )

    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        available = importlib.util.find_spec(self.import_name) is not None
        if not available:
            return False, [f"missing dependency: {self.import_name}"]
        return True, [f"library available: {self.import_name}"]

    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        text = str(params.get("text", "")).strip()
        tokens = [token for token in text.replace("\n", " ").split(" ") if token]
        payload = {
            "provider": self.spec.provider,
            "capability": self.capability,
            "text_len": len(text),
            "token_count": len(tokens),
            "result": {
                "tokens_preview": tokens[:32],
                "param_keys": sorted(str(key) for key in params.keys()),
            },
        }
        return ToolExecutionResult(payload=payload, metadata={"provider_family_adapter": True})


def _register_library_family_adapters(registry: ToolRegistry) -> None:
    rows: list[tuple[str, str, str, str, int, str | None]] = [
        # spaCy
        ("spacy_tokenize", "spacy", "nlp.tokenize", "spacy", 70, None),
        ("spacy_sentence_segment", "spacy", "nlp.sentence_segmentation", "spacy", 69, None),
        ("spacy_pos_tag", "spacy", "nlp.pos_tag", "spacy", 68, None),
        ("spacy_morphology", "spacy", "nlp.morphology", "spacy", 67, None),
        ("spacy_lemmatize", "spacy", "nlp.lemmatization", "spacy", 66, None),
        ("spacy_dependency_parse", "spacy", "nlp.dependency_parse", "spacy", 65, None),
        ("spacy_ner", "spacy", "nlp.ner", "spacy", 64, None),
        ("spacy_text_classification", "spacy", "nlp.text_classification", "spacy", 63, None),
        ("spacy_entity_linking", "spacy", "nlp.entity_linking", "spacy", 62, None),
        ("spacy_vector_similarity", "spacy", "nlp.vector_similarity", "spacy", 61, None),
        # textacy
        ("textacy_cleaning", "textacy", "text.cleaning", "textacy", 60, None),
        ("textacy_normalization", "textacy", "text.normalization", "textacy", 59, None),
        ("textacy_exploration", "textacy", "text.exploration", "textacy", 58, None),
        ("textacy_extraction", "textacy", "text.extraction", "textacy", 57, None),
        ("textacy_keyterms", "textacy", "text.keyterms", "textacy", 56, None),
        ("textacy_ngrams", "textacy", "text.ngrams", "textacy", 55, None),
        ("textacy_acronyms", "textacy", "text.acronyms", "textacy", 54, None),
        ("textacy_svo_triples", "textacy", "text.svo_triples", "textacy", 53, None),
        ("textacy_topic_workflow", "textacy", "text.topic_workflow", "textacy", 52, None),
        ("textacy_readability", "textacy", "text.readability", "textacy", 51, None),
        ("textacy_lexical_diversity", "textacy", "text.lexical_diversity", "textacy", 50, None),
        # Stanza
        ("stanza_tokenize", "stanza", "nlp.tokenize", "stanza", 49, "spacy_tokenize"),
        ("stanza_mwt", "stanza", "nlp.mwt", "stanza", 48, None),
        ("stanza_lemmatize", "stanza", "nlp.lemmatization", "stanza", 47, "spacy_lemmatize"),
        ("stanza_pos_tag", "stanza", "nlp.pos_tag", "stanza", 46, "spacy_pos_tag"),
        ("stanza_morphology", "stanza", "nlp.morphology", "stanza", 45, "spacy_morphology"),
        ("stanza_dependency_parse", "stanza", "nlp.dependency_parse", "stanza", 44, "spacy_dependency_parse"),
        ("stanza_ner", "stanza", "nlp.ner", "stanza", 43, "spacy_ner"),
        # NLTK
        ("nltk_corpora", "nltk", "corpora.access", "nltk", 42, None),
        ("nltk_tokenize", "nltk", "nlp.tokenize", "nltk", 41, "spacy_tokenize"),
        ("nltk_tagging", "nltk", "nlp.tagging", "nltk", 40, None),
        ("nltk_parsing", "nltk", "nlp.parsing", "nltk", 39, None),
        ("nltk_classical_classification", "nltk", "classification.classical", "nltk", 38, None),
        # gensim
        ("gensim_tfidf", "gensim", "topic.tfidf", "gensim", 37, None),
        ("gensim_lsi", "gensim", "topic.lsi", "gensim", 36, None),
        ("gensim_lda", "gensim", "topic.lda", "gensim", 35, None),
        ("gensim_hdp", "gensim", "topic.hdp", "gensim", 34, None),
        ("gensim_embeddings", "gensim", "embedding.training", "gensim", 33, None),
        ("gensim_phrase_detection", "gensim", "phrase_detection", "gensim", 32, None),
        ("gensim_similarity_indexing", "gensim", "similarity.indexing", "gensim", 31, None),
        # Flair
        ("flair_sequence_labeling", "flair", "sequence.labeling", "flair", 30, None),
        ("flair_text_classification", "flair", "text.classification", "flair", 29, None),
        ("flair_embeddings", "flair", "embedding.contextual", "flair", 28, None),
        # TextBlob
        ("textblob_pos_tag", "textblob", "nlp.pos_tag", "textblob", 27, "spacy_pos_tag"),
        ("textblob_noun_phrases", "textblob", "nlp.noun_phrase_extraction", "textblob", 26, None),
        ("textblob_sentiment", "textblob", "sentiment.basic", "textblob", 25, None),
        ("textblob_simple_classification", "textblob", "classification.simple", "textblob", 24, None),
    ]
    for tool_name, provider, capability, import_name, priority, fallback_of in rows:
        registry.register(
            LibraryCapabilityTool(
                tool_name=tool_name,
                provider=provider,
                capability=capability,
                import_name=import_name,
                priority=priority,
                fallback_of=fallback_of,
            )
        )


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(HybridRetrievalTool())
    registry.register(DocumentFilterTool())
    registry.register(PrecomputedArtifactTool("precomputed_entity_trend", "analyze.entity_trend", "entity_trend", priority=95))
    registry.register(QueryTimeEntityTrendTool())
    registry.register(
        PrecomputedArtifactTool(
            "precomputed_sentiment_series",
            "analyze.sentiment_series",
            "sentiment_series",
            priority=92,
            entity_specific=False,
        )
    )
    registry.register(QueryTimeSentimentTool())
    registry.register(
        PrecomputedArtifactTool(
            "precomputed_topics_over_time",
            "analyze.topics_over_time",
            "topics_over_time",
            priority=88,
            entity_specific=False,
        )
    )
    registry.register(QueryTimeTopicsTool())
    registry.register(
        PrecomputedArtifactTool(
            "precomputed_burst_events",
            "analyze.burst_events",
            "burst_events",
            priority=90,
        )
    )
    registry.register(BurstDetectionTool())
    registry.register(
        PrecomputedArtifactTool(
            "precomputed_keyphrases",
            "analyze.keyphrases",
            "keyphrases",
            priority=86,
            entity_specific=False,
        )
    )
    registry.register(QueryTimeKeyphraseTool())
    registry.register(FindingsAggregationTool())
    registry.register(ClaimVerificationTool())
    registry.register(GroundedSynthesisTool())
    _register_library_family_adapters(registry)
    return registry
