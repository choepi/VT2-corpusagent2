from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.io_utils import (
    ensure_absolute,
    ensure_exists,
    read_jsonl,
    write_json,
    write_jsonl,
)
from corpusagent2.model_config import dense_model_id_from_env
from corpusagent2.metrics import (
    bootstrap_confidence_interval,
    evidence_completeness,
    mrr_at_k,
    ndcg_at_k,
    paired_t_test,
    recall_at_k,
)
from corpusagent2.retrieval import (
    load_dense_assets,
    load_lexical_assets,
    pg_dsn_from_env,
    pg_table_from_env,
    reciprocal_rank_fusion,
    resolve_retrieval_backend,
    rerank_cross_encoder,
    retrieve_dense,
    retrieve_dense_pgvector,
    retrieve_tfidf,
)
from corpusagent2.seed import resolve_run_mode, set_global_seed


def metric_row(system: str, query_id: str, predicted: list[str], relevant: set[str], evidence: set[str]) -> dict:
    return {
        "system": system,
        "query_id": query_id,
        "ndcg@10": ndcg_at_k(predicted, relevant, k=10),
        "mrr@10": mrr_at_k(predicted, relevant, k=10),
        "recall@100": recall_at_k(predicted, relevant, k=100),
        "evidence_completeness": evidence_completeness(predicted, evidence),
    }


if __name__ == "__main__":
    MODE = resolve_run_mode("full")
    SEED = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    INDEX_ROOT = (PROJECT_ROOT / "data" / "indices").resolve()
    QUERIES_PATH = (PROJECT_ROOT / "config" / "retrieval_queries.jsonl").resolve()

    OUTPUT_DIR = (PROJECT_ROOT / "outputs" / "retrieval_eval").resolve()
    PER_QUERY_PATH = (OUTPUT_DIR / "per_query_metrics.jsonl").resolve()
    SUMMARY_PATH = (OUTPUT_DIR / "summary.json").resolve()

    DENSE_MODEL_ID = dense_model_id_from_env()
    DENSE_DEVICE = None
    CROSS_ENCODER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RETRIEVAL_BACKEND = resolve_retrieval_backend("local")
    PG_DSN = pg_dsn_from_env(required=RETRIEVAL_BACKEND == "pgvector") if RETRIEVAL_BACKEND == "pgvector" else ""
    PG_TABLE = pg_table_from_env() if RETRIEVAL_BACKEND == "pgvector" else ""

    TOP_K = 100
    FUSION_INPUT_K = 200
    RERANK_TOP_K = 100
    USE_RERANK = True

    ensure_absolute(INDEX_ROOT, "INDEX_ROOT")
    ensure_absolute(QUERIES_PATH, "QUERIES_PATH")
    ensure_exists(QUERIES_PATH, "QUERIES_PATH")

    set_global_seed(SEED)

    lexical_vectorizer, lexical_matrix, lexical_doc_ids = load_lexical_assets(INDEX_ROOT / "lexical")
    dense_embeddings = None
    dense_doc_ids = None
    if RETRIEVAL_BACKEND == "local":
        dense_embeddings, dense_doc_ids = load_dense_assets(INDEX_ROOT / "dense")

    metadata_path = INDEX_ROOT / "doc_metadata.parquet"
    ensure_exists(metadata_path, "doc_metadata.parquet")
    metadata = pd.read_parquet(metadata_path)
    doc_text_by_id = {
        str(row.doc_id): f"{str(row.title)} {str(row.text)}".strip()
        for row in metadata.itertuples(index=False)
    }

    queries = read_jsonl(QUERIES_PATH)
    if MODE == "debug":
        queries = queries[: min(len(queries), 25)]

    if not queries:
        raise RuntimeError("No queries found in retrieval query file")

    metric_rows: list[dict] = []

    for query_row in queries:
        query_id = str(query_row["query_id"])
        query_text = str(query_row["query"])
        relevant_doc_ids = {str(doc_id) for doc_id in query_row.get("relevant_doc_ids", [])}
        if not relevant_doc_ids:
            relevant_doc_ids = {str(doc_id) for doc_id in query_row.get("gold_evidence_doc_ids", [])}
        evidence_doc_ids = {str(doc_id) for doc_id in query_row.get("gold_evidence_doc_ids", [])}

        tfidf = retrieve_tfidf(
            query=query_text,
            vectorizer=lexical_vectorizer,
            matrix=lexical_matrix,
            doc_ids=lexical_doc_ids,
            top_k=TOP_K,
        )
        dense = retrieve_dense(
            query=query_text,
            model_id=DENSE_MODEL_ID,
            embeddings=dense_embeddings,
            doc_ids=dense_doc_ids,
            top_k=TOP_K,
            device=DENSE_DEVICE,
        ) if RETRIEVAL_BACKEND == "local" else retrieve_dense_pgvector(
            query=query_text,
            model_id=DENSE_MODEL_ID,
            dsn=PG_DSN,
            table_name=PG_TABLE,
            top_k=TOP_K,
            device=DENSE_DEVICE,
        )

        fused = reciprocal_rank_fusion(
            {
                "tfidf": tfidf[:FUSION_INPUT_K],
                "dense": dense[:FUSION_INPUT_K],
            }
        )[:TOP_K]

        reranked = fused
        if USE_RERANK:
            reranked = rerank_cross_encoder(
                query=query_text,
                candidates=fused[:RERANK_TOP_K],
                doc_text_by_id=doc_text_by_id,
                model_id=CROSS_ENCODER_MODEL_ID,
                top_k=TOP_K,
            )

        tfidf_doc_ids = [item.doc_id for item in tfidf]
        dense_doc_ids_result = [item.doc_id for item in dense]
        fused_doc_ids = [item.doc_id for item in fused]
        rerank_doc_ids = [item.doc_id for item in reranked]

        metric_rows.append(metric_row("tfidf", query_id, tfidf_doc_ids, relevant_doc_ids, evidence_doc_ids))
        metric_rows.append(metric_row("dense", query_id, dense_doc_ids_result, relevant_doc_ids, evidence_doc_ids))
        metric_rows.append(metric_row("tfidf_dense_rrf", query_id, fused_doc_ids, relevant_doc_ids, evidence_doc_ids))
        metric_rows.append(metric_row("tfidf_dense_rrf_rerank", query_id, rerank_doc_ids, relevant_doc_ids, evidence_doc_ids))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(PER_QUERY_PATH, metric_rows)

    df = pd.DataFrame(metric_rows)
    systems = sorted(df["system"].unique().tolist())

    summary: dict = {
        "mode": MODE,
        "seed": SEED,
        "retrieval_backend": RETRIEVAL_BACKEND,
        "queries": len(set(df["query_id"].tolist())),
        "systems": {},
        "paired_tests_vs_tfidf": {},
    }

    for system in systems:
        subset = df[df["system"] == system]

        ndcg_scores = subset["ndcg@10"].tolist()
        mrr_scores = subset["mrr@10"].tolist()
        recall_scores = subset["recall@100"].tolist()
        evidence_scores = subset["evidence_completeness"].tolist()

        summary["systems"][system] = {
            "ndcg@10": asdict(bootstrap_confidence_interval(ndcg_scores)),
            "mrr@10": asdict(bootstrap_confidence_interval(mrr_scores)),
            "recall@100": asdict(bootstrap_confidence_interval(recall_scores)),
            "evidence_completeness": asdict(bootstrap_confidence_interval(evidence_scores)),
        }

    baseline = df[df["system"] == "tfidf"].sort_values("query_id")
    for system in systems:
        if system == "tfidf":
            continue
        test_subset = df[df["system"] == system].sort_values("query_id")
        test = paired_t_test(
            test_subset["ndcg@10"].tolist(),
            baseline["ndcg@10"].tolist(),
        )
        summary["paired_tests_vs_tfidf"][system] = {
            "metric": "ndcg@10",
            "mean_delta": test.mean_delta,
            "t_statistic": test.t_statistic,
            "p_value": test.p_value,
        }

    write_json(SUMMARY_PATH, summary)

    print(f"Wrote per-query metrics: {PER_QUERY_PATH}")
    print(f"Wrote summary: {SUMMARY_PATH}")
