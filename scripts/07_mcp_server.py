from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.faithfulness import NLIVerifier, evaluate_claims_with_nli
from corpusagent2.model_config import dense_model_id_from_env
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
from corpusagent2.seed import runtime_device_report
from corpusagent2.temporal import classify_time_bin_format, incompatible_time_bins, normalize_granularity

from mcp.server.fastmcp import FastMCP


INDEX_ROOT = (PROJECT_ROOT / "data" / "indices").resolve()
NLP_OUTPUT_DIR = (PROJECT_ROOT / "outputs" / "nlp_tools").resolve()
DENSE_MODEL_ID = dense_model_id_from_env()
RERANK_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL_ID = "FacebookAI/roberta-large-mnli"
RETRIEVAL_BACKEND = resolve_retrieval_backend("local")
PG_DSN = pg_dsn_from_env(required=RETRIEVAL_BACKEND == "pgvector") if RETRIEVAL_BACKEND == "pgvector" else ""
PG_TABLE = pg_table_from_env() if RETRIEVAL_BACKEND == "pgvector" else ""

mcp = FastMCP("CorpusAgent2")


@lru_cache(maxsize=1)
def load_runtime() -> dict:
    lexical_vectorizer, lexical_matrix, lexical_doc_ids = load_lexical_assets(INDEX_ROOT / "lexical")
    dense_embeddings = None
    dense_doc_ids = None
    if RETRIEVAL_BACKEND == "local":
        dense_embeddings, dense_doc_ids = load_dense_assets(INDEX_ROOT / "dense")
    metadata = pd.read_parquet(INDEX_ROOT / "doc_metadata.parquet")
    doc_text_by_id = {
        str(row.doc_id): f"{str(row.title)} {str(row.text)}".strip()
        for row in metadata.itertuples(index=False)
    }
    verifier = NLIVerifier(model_id=NLI_MODEL_ID, device=None)
    sentiment_path = NLP_OUTPUT_DIR / "sentiment_series.parquet"
    sentiment_summary_path = NLP_OUTPUT_DIR / "summary.json"
    sentiment_df = pd.DataFrame(columns=["entity", "time_bin", "mean", "std", "n_docs", "model_id"])
    sentiment_time_granularity = ""
    if sentiment_path.exists():
        sentiment_df = pd.read_parquet(sentiment_path)
    if sentiment_summary_path.exists():
        try:
            payload = json.loads(sentiment_summary_path.read_text(encoding="utf-8"))
            sentiment_time_granularity = str(payload.get("time_granularity", "")).strip().lower()
        except Exception:
            sentiment_time_granularity = ""
    if sentiment_time_granularity not in {"year", "month"}:
        inferred = ""
        for value in sentiment_df.get("time_bin", pd.Series([], dtype="object")).astype(str).tolist():
            bucket = classify_time_bin_format(value)
            if bucket in {"year", "month"}:
                inferred = bucket
                break
        sentiment_time_granularity = inferred
    return {
        "lexical_vectorizer": lexical_vectorizer,
        "lexical_matrix": lexical_matrix,
        "lexical_doc_ids": lexical_doc_ids,
        "dense_embeddings": dense_embeddings,
        "dense_doc_ids": dense_doc_ids,
        "doc_text_by_id": doc_text_by_id,
        "verifier": verifier,
        "device_report": runtime_device_report(),
        "retrieval_backend": RETRIEVAL_BACKEND,
        "pg_dsn": PG_DSN,
        "pg_table": PG_TABLE,
        "sentiment_df": sentiment_df,
        "sentiment_time_granularity": sentiment_time_granularity,
    }


@mcp.tool()
def retrieve(query: str, top_k: int = 20) -> list[dict]:
    runtime = load_runtime()

    tfidf = retrieve_tfidf(
        query=query,
        vectorizer=runtime["lexical_vectorizer"],
        matrix=runtime["lexical_matrix"],
        doc_ids=runtime["lexical_doc_ids"],
        top_k=max(100, top_k),
    )
    dense = retrieve_dense(
        query=query,
        model_id=DENSE_MODEL_ID,
        embeddings=runtime["dense_embeddings"],
        doc_ids=runtime["dense_doc_ids"],
        top_k=max(100, top_k),
    ) if runtime["retrieval_backend"] == "local" else retrieve_dense_pgvector(
        query=query,
        model_id=DENSE_MODEL_ID,
        dsn=runtime["pg_dsn"],
        table_name=runtime["pg_table"],
        top_k=max(100, top_k),
    )
    fused = reciprocal_rank_fusion({"tfidf": tfidf, "dense": dense})
    reranked = rerank_cross_encoder(
        query=query,
        candidates=fused[:150],
        doc_text_by_id=runtime["doc_text_by_id"],
        model_id=RERANK_MODEL_ID,
        top_k=top_k,
    )

    return [
        {
            "doc_id": row.doc_id,
            "chunk_id": f"{row.doc_id}:0",
            "score": row.score,
            "score_components": row.score_components,
        }
        for row in reranked
    ]


@mcp.tool()
def verify_claims(claims: list[str], evidence_doc_ids: list[str]) -> dict:
    runtime = load_runtime()

    claim_rows = [
        {
            "claim_id": f"claim_{idx}",
            "claim": claim,
            "evidence_doc_ids": evidence_doc_ids,
            "category": "A",
        }
        for idx, claim in enumerate(claims, start=1)
    ]

    verdicts, summary = evaluate_claims_with_nli(
        verifier=runtime["verifier"],
        claims=claim_rows,
        doc_text_by_id=runtime["doc_text_by_id"],
    )

    return {
        "summary": summary,
        "verdicts": [item.to_dict() for item in verdicts],
    }


@mcp.tool()
def sentiment_over_time(
    entity: str = "__all__",
    granularity: str = "year",
    start_time_bin: str = "",
    end_time_bin: str = "",
) -> dict:
    runtime = load_runtime()
    requested = normalize_granularity(granularity)
    available = str(runtime.get("sentiment_time_granularity", "")).strip().lower()
    sentiment_df = runtime["sentiment_df"].copy()

    if sentiment_df.empty:
        return {
            "status": "no_data",
            "message": f"Missing sentiment artifact at {NLP_OUTPUT_DIR / 'sentiment_series.parquet'}",
            "requested_granularity": requested,
        }

    if available in {"year", "month"} and available != requested:
        raise ValueError(
            "Granularity mismatch: "
            f"requested={requested}, available={available}. "
            f"Regenerate NLP tooling with CORPUSAGENT2_TIME_GRANULARITY={requested}."
        )

    bins = sentiment_df.get("time_bin", pd.Series([], dtype="object")).astype(str).tolist()
    incompatible = incompatible_time_bins(bins, requested)
    if incompatible:
        sample = incompatible[:10]
        raise ValueError(
            "Incompatible mixed time bins in sentiment data. "
            f"Requested '{requested}' but found incompatible bins, sample={sample}."
        )

    subset = sentiment_df.copy()
    if entity.strip():
        subset = subset[subset["entity"].astype(str) == entity]
    subset = subset[subset["time_bin"].astype(str) != "unknown"]

    if start_time_bin.strip():
        subset = subset[subset["time_bin"].astype(str) >= start_time_bin.strip()]
    if end_time_bin.strip():
        subset = subset[subset["time_bin"].astype(str) <= end_time_bin.strip()]

    subset = subset.sort_values("time_bin")

    rows = [
        {
            "entity": str(row.entity),
            "time_bin": str(row.time_bin),
            "mean": float(row.mean),
            "std": float(row.std),
            "n_docs": int(row.n_docs),
            "model_id": str(row.model_id),
        }
        for row in subset.itertuples(index=False)
    ]

    return {
        "status": "ok",
        "requested_granularity": requested,
        "available_granularity": available or requested,
        "entity": entity,
        "start_time_bin": start_time_bin,
        "end_time_bin": end_time_bin,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CorpusAgent2 MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="MCP transport to use (default: stdio).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Preload retrieval and NLI assets at startup.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run one local retrieval call and exit (no MCP server loop).",
    )
    parser.add_argument(
        "--self-test-query",
        default="test query",
        help="Query string used with --self-test.",
    )
    parser.add_argument(
        "--self-test-top-k",
        type=int,
        default=3,
        help="Number of results returned by --self-test.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.warmup:
        print("Preloading runtime assets...", file=sys.stderr, flush=True)
        runtime = load_runtime()
        print("Runtime assets loaded.", file=sys.stderr, flush=True)
        print(f"Device report: {runtime['device_report']}", file=sys.stderr, flush=True)

    if args.self_test:
        print("Running self-test retrieval...", file=sys.stderr, flush=True)
        runtime = load_runtime()
        results = retrieve(query=args.self_test_query, top_k=max(1, args.self_test_top_k))
        print(
            json.dumps(
                {
                    "query": args.self_test_query,
                    "top_k": max(1, args.self_test_top_k),
                    "result_count": len(results),
                    "device_report": runtime["device_report"],
                    "results": results,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        print("Self-test completed.", file=sys.stderr, flush=True)
        sys.exit(0)

    print(f"MCP server starting with transport={args.transport}", file=sys.stderr, flush=True)
    try:
        mcp.run(transport=args.transport)
    except KeyboardInterrupt:
        print("MCP server interrupted; shutting down.", file=sys.stderr, flush=True)
        # Avoid occasional stdin lock crashes during interpreter finalization on macOS.
        os._exit(130)
