from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .seed import resolve_device

_ALLOWED_RETRIEVAL_BACKENDS = {"local", "pgvector"}
_SENTENCE_TRANSFORMER_CACHE: dict[tuple[str, str], Any] = {}
_CROSS_ENCODER_CACHE: dict[tuple[str, str], Any] = {}
_TORCH_DENSE_CACHE: dict[tuple[int, str], Any] = {}


@dataclass(slots=True)
class RetrievalResult:
    doc_id: str
    rank: int
    score: float
    score_components: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "rank": self.rank,
            "score": self.score,
            "score_components": self.score_components,
        }


def resolve_retrieval_backend(default: str = "local") -> str:
    requested = os.getenv("CORPUSAGENT2_RETRIEVAL_BACKEND", "").strip().lower()
    if requested in _ALLOWED_RETRIEVAL_BACKENDS:
        return requested
    normalized_default = (default or "local").strip().lower()
    if normalized_default in _ALLOWED_RETRIEVAL_BACKENDS:
        return normalized_default
    return "local"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def dense_retrieval_enabled(default: bool = True) -> bool:
    raw = os.getenv("CORPUSAGENT2_ENABLE_DENSE_RETRIEVAL", "").strip()
    if raw:
        return raw.lower() not in {"0", "false", "no", "off"}
    build_dense_assets = _env_flag("CORPUSAGENT2_BUILD_DENSE_ASSETS", default)
    include_pg_embeddings = _env_flag(
        "CORPUSAGENT2_PG_INCLUDE_EMBEDDINGS",
        build_dense_assets,
    )
    return build_dense_assets or include_pg_embeddings


def pg_dsn_from_env(required: bool = True) -> str:
    dsn = os.getenv("CORPUSAGENT2_PG_DSN", "").strip()
    if required and not dsn:
        raise RuntimeError(
            "CORPUSAGENT2_PG_DSN is required for Postgres/pgvector operations."
        )
    return dsn


def pg_table_from_env(default: str = "ca_documents") -> str:
    table_name = os.getenv("CORPUSAGENT2_PG_TABLE", "").strip()
    if not table_name:
        table_name = default
    if not table_name.replace("_", "").isalnum():
        raise ValueError(f"Invalid table name: {table_name}")
    return table_name


def _vector_literal(vector: np.ndarray) -> str:
    values = vector.astype(np.float32).tolist()
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _load_sentence_transformer(model_id: str, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    resolved_device = resolve_device(device)
    cache_key = (model_id, resolved_device)
    if cache_key in _SENTENCE_TRANSFORMER_CACHE:
        return _SENTENCE_TRANSFORMER_CACHE[cache_key], resolved_device
    try:
        model = SentenceTransformer(model_id, device=resolved_device)
        _SENTENCE_TRANSFORMER_CACHE[cache_key] = model
        return model, resolved_device
    except Exception:
        if resolved_device != "cpu":
            cpu_key = (model_id, "cpu")
            if cpu_key in _SENTENCE_TRANSFORMER_CACHE:
                return _SENTENCE_TRANSFORMER_CACHE[cpu_key], "cpu"
            model = SentenceTransformer(model_id, device="cpu")
            _SENTENCE_TRANSFORMER_CACHE[cpu_key] = model
            return model, "cpu"
        raise


def _load_cross_encoder(model_id: str, device: str | None = None):
    from sentence_transformers import CrossEncoder

    resolved_device = resolve_device(device)
    cache_key = (model_id, resolved_device)
    if cache_key in _CROSS_ENCODER_CACHE:
        return _CROSS_ENCODER_CACHE[cache_key], resolved_device
    try:
        model = CrossEncoder(model_id, device=resolved_device)
        _CROSS_ENCODER_CACHE[cache_key] = model
        return model, resolved_device
    except Exception:
        if resolved_device != "cpu":
            cpu_key = (model_id, "cpu")
            if cpu_key in _CROSS_ENCODER_CACHE:
                return _CROSS_ENCODER_CACHE[cpu_key], "cpu"
            model = CrossEncoder(model_id, device="cpu")
            _CROSS_ENCODER_CACHE[cpu_key] = model
            return model, "cpu"
        raise


def build_lexical_assets(df: pd.DataFrame, max_features: int = 250_000) -> tuple[TfidfVectorizer, object, list[str]]:
    corpus = (
        (df["title"].fillna("") + " " + df["text"].fillna(""))
        .str.replace("\n", " ", regex=False)
        .tolist()
    )
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(corpus)
    doc_ids = df["doc_id"].astype(str).tolist()
    return vectorizer, matrix, doc_ids


def save_lexical_assets(
    index_dir: Path,
    vectorizer: TfidfVectorizer,
    matrix: object,
    doc_ids: list[str],
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, index_dir / "tfidf_vectorizer.joblib")
    joblib.dump(matrix, index_dir / "tfidf_matrix.joblib")
    joblib.dump(doc_ids, index_dir / "tfidf_doc_ids.joblib")


def load_lexical_assets(index_dir: Path) -> tuple[TfidfVectorizer, object, list[str]]:
    vectorizer = joblib.load(index_dir / "tfidf_vectorizer.joblib")
    matrix = joblib.load(index_dir / "tfidf_matrix.joblib")
    doc_ids = joblib.load(index_dir / "tfidf_doc_ids.joblib")
    return vectorizer, matrix, doc_ids


def build_dense_embeddings(
    df: pd.DataFrame,
    model_id: str,
    batch_size: int = 128,
    device: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    model, _resolved_device = _load_sentence_transformer(model_id=model_id, device=device)

    corpus = (
        (df["title"].fillna("") + " " + df["text"].fillna(""))
        .str.replace("\n", " ", regex=False)
        .tolist()
    )
    embeddings = model.encode(
        corpus,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    doc_ids = df["doc_id"].astype(str).tolist()
    return embeddings, doc_ids


def save_dense_assets(index_dir: Path, embeddings: np.ndarray, doc_ids: list[str]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "dense_embeddings.npy", embeddings)
    joblib.dump(doc_ids, index_dir / "dense_doc_ids.joblib")


def load_dense_assets(index_dir: Path) -> tuple[np.ndarray, list[str]]:
    embeddings = np.load(index_dir / "dense_embeddings.npy", mmap_mode="r")
    doc_ids = joblib.load(index_dir / "dense_doc_ids.joblib")
    return embeddings, doc_ids


def retrieve_tfidf(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: object,
    doc_ids: list[str],
    top_k: int,
) -> list[RetrievalResult]:
    if top_k <= 0:
        return []
    query_vec = vectorizer.transform([query])
    if query_vec.nnz == 0:
        return []
    scores = cosine_similarity(query_vec, matrix).ravel()
    positive_idx = np.flatnonzero(scores > 0.0)
    if positive_idx.size == 0:
        return []
    limit = min(int(top_k), int(positive_idx.size))
    positive_scores = scores[positive_idx]
    best_relative = np.argpartition(positive_scores, -limit)[-limit:]
    ranked_relative = best_relative[np.argsort(positive_scores[best_relative])[::-1]]
    ranked_idx = positive_idx[ranked_relative]

    results: list[RetrievalResult] = []
    for rank, idx in enumerate(ranked_idx, start=1):
        results.append(
            RetrievalResult(
                doc_id=str(doc_ids[idx]),
                rank=rank,
                score=float(scores[idx]),
                score_components={"tfidf": float(scores[idx])},
            )
        )
    return results


def retrieve_dense(
    query: str,
    model_id: str,
    embeddings: np.ndarray,
    doc_ids: list[str],
    top_k: int,
    device: str | None = None,
) -> list[RetrievalResult]:
    if top_k <= 0:
        return []
    model, _resolved_device = _load_sentence_transformer(model_id=model_id, device=device)
    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    scores: np.ndarray
    if _resolved_device == "cuda":
        try:
            import torch

            cache_key = (id(embeddings), _resolved_device)
            if cache_key not in _TORCH_DENSE_CACHE:
                _TORCH_DENSE_CACHE[cache_key] = torch.as_tensor(
                    np.asarray(embeddings),
                    dtype=torch.float32,
                    device=_resolved_device,
                )
            dense_tensor = _TORCH_DENSE_CACHE[cache_key]
            query_tensor = torch.as_tensor(query_emb, dtype=torch.float32, device=_resolved_device)
            scores = torch.matmul(query_tensor, dense_tensor.T).detach().cpu().numpy().ravel()
        except Exception:
            scores = (query_emb @ np.asarray(embeddings).T).ravel()
    else:
        scores = (query_emb @ np.asarray(embeddings).T).ravel()
    limit = min(int(top_k), int(scores.shape[0]))
    if limit <= 0:
        return []
    best = np.argpartition(scores, -limit)[-limit:]
    ranked_idx = best[np.argsort(scores[best])[::-1]]

    results: list[RetrievalResult] = []
    for rank, idx in enumerate(ranked_idx, start=1):
        results.append(
            RetrievalResult(
                doc_id=str(doc_ids[idx]),
                rank=rank,
                score=float(scores[idx]),
                score_components={"dense": float(scores[idx])},
            )
        )
    return results


def retrieve_dense_pgvector(
    query: str,
    model_id: str,
    dsn: str,
    table_name: str,
    top_k: int,
    device: str | None = None,
) -> list[RetrievalResult]:
    if top_k <= 0:
        return []

    from psycopg import connect
    from psycopg.rows import tuple_row

    model, _resolved_device = _load_sentence_transformer(model_id=model_id, device=device)
    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)[0]
    query_vector_literal = _vector_literal(query_emb)

    sql = (
        f"SELECT doc_id, 1 - (dense_embedding <=> %s::vector) AS score "
        f"FROM {table_name} "
        "WHERE dense_embedding IS NOT NULL "
        "ORDER BY dense_embedding <=> %s::vector "
        "LIMIT %s"
    )

    rows: list[tuple] = []
    with connect(dsn) as conn:
        with conn.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(sql, (query_vector_literal, query_vector_literal, int(top_k)))
            rows = cursor.fetchall()

    results: list[RetrievalResult] = []
    for rank, row in enumerate(rows, start=1):
        doc_id, score = row
        score_value = float(score) if score is not None else 0.0
        results.append(
            RetrievalResult(
                doc_id=str(doc_id),
                rank=rank,
                score=score_value,
                score_components={"dense_pgvector": score_value},
            )
        )
    return results


def reciprocal_rank_fusion(
    rankings: dict[str, list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    score_map: dict[str, float] = defaultdict(float)
    component_map: dict[str, dict[str, float]] = defaultdict(dict)

    for method_name, results in rankings.items():
        for rank, result in enumerate(results, start=1):
            fused = 1.0 / float(k + rank)
            score_map[result.doc_id] += fused
            component_map[result.doc_id][method_name] = round(fused, 6)

    ranked = sorted(score_map.items(), key=lambda pair: pair[1], reverse=True)
    fused_results: list[RetrievalResult] = []
    for rank, (doc_id, score) in enumerate(ranked, start=1):
        fused_results.append(
            RetrievalResult(
                doc_id=doc_id,
                rank=rank,
                score=float(score),
                score_components=component_map[doc_id],
            )
        )
    return fused_results


def rerank_cross_encoder(
    query: str,
    candidates: list[RetrievalResult],
    doc_text_by_id: dict[str, str],
    model_id: str,
    top_k: int,
    device: str | None = None,
) -> list[RetrievalResult]:
    if not candidates:
        return []

    pairs: list[tuple[str, str]] = []
    candidate_doc_ids: list[str] = []
    for candidate in candidates:
        text = doc_text_by_id.get(candidate.doc_id, "")
        if not text:
            continue
        pairs.append((query, text[:2000]))
        candidate_doc_ids.append(candidate.doc_id)

    if not pairs:
        return candidates[:top_k]

    reranker, _resolved_device = _load_cross_encoder(model_id=model_id, device=device)
    scores = reranker.predict(pairs)

    score_map = {doc_id: float(score) for doc_id, score in zip(candidate_doc_ids, scores)}
    reranked = sorted(candidate_doc_ids, key=lambda doc_id: score_map[doc_id], reverse=True)

    results: list[RetrievalResult] = []
    for rank, doc_id in enumerate(reranked[:top_k], start=1):
        base = next((item for item in candidates if item.doc_id == doc_id), None)
        base_components = dict(base.score_components) if base else {}
        base_components["cross_encoder"] = score_map[doc_id]
        results.append(
            RetrievalResult(
                doc_id=doc_id,
                rank=rank,
                score=score_map[doc_id],
                score_components=base_components,
            )
        )
    return results
