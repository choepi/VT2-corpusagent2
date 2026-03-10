from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=device)
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
    embeddings = np.load(index_dir / "dense_embeddings.npy")
    doc_ids = joblib.load(index_dir / "dense_doc_ids.joblib")
    return embeddings, doc_ids


def retrieve_tfidf(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: object,
    doc_ids: list[str],
    top_k: int,
) -> list[RetrievalResult]:
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).ravel()
    best = np.argpartition(scores, -top_k)[-top_k:]
    ranked_idx = best[np.argsort(scores[best])[::-1]]

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
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=device)
    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores = (query_emb @ embeddings.T).ravel()
    best = np.argpartition(scores, -top_k)[-top_k:]
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
            component_map[result.doc_id][method_name] = fused

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
    from sentence_transformers import CrossEncoder

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

    reranker = CrossEncoder(model_id, device=device)
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
