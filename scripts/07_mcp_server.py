from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.faithfulness import NLIVerifier, evaluate_claims_with_nli
from corpusagent2.retrieval import (
    load_dense_assets,
    load_lexical_assets,
    reciprocal_rank_fusion,
    rerank_cross_encoder,
    retrieve_dense,
    retrieve_tfidf,
)

from mcp.server.fastmcp import FastMCP


INDEX_ROOT = (PROJECT_ROOT / "data" / "indices").resolve()
DENSE_MODEL_ID = "intfloat/e5-base-v2"
RERANK_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL_ID = "FacebookAI/roberta-large-mnli"

mcp = FastMCP("CorpusAgent2")


@lru_cache(maxsize=1)
def load_runtime() -> dict:
    lexical_vectorizer, lexical_matrix, lexical_doc_ids = load_lexical_assets(INDEX_ROOT / "lexical")
    dense_embeddings, dense_doc_ids = load_dense_assets(INDEX_ROOT / "dense")
    metadata = pd.read_parquet(INDEX_ROOT / "doc_metadata.parquet")
    doc_text_by_id = {
        str(row.doc_id): f"{str(row.title)} {str(row.text)}".strip()
        for row in metadata.itertuples(index=False)
    }
    return {
        "lexical_vectorizer": lexical_vectorizer,
        "lexical_matrix": lexical_matrix,
        "lexical_doc_ids": lexical_doc_ids,
        "dense_embeddings": dense_embeddings,
        "dense_doc_ids": dense_doc_ids,
        "doc_text_by_id": doc_text_by_id,
        "verifier": NLIVerifier(model_id=NLI_MODEL_ID, device=-1),
    }


@mcp.tool()
def retrieve(query: str, top_k: int = 20) -> list[dict]:
    runtime = load_runtime()

    bm25 = retrieve_tfidf(
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
    )
    fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense})
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
