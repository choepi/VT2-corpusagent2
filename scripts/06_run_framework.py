from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.faithfulness import NLIVerifier, evaluate_claims_with_nli
from corpusagent2.io_utils import ensure_absolute, ensure_exists, read_jsonl, write_json, write_jsonl
from corpusagent2.provenance import make_provenance_record
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
from corpusagent2.seed import resolve_run_mode, runtime_device_report, set_global_seed


if __name__ == "__main__":
    MODE = resolve_run_mode("full")
    SEED = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    INDEX_ROOT = (PROJECT_ROOT / "data" / "indices").resolve()
    WORKLOAD_PATH = (PROJECT_ROOT / "config" / "framework_workload.jsonl").resolve()

    RUN_ID = f"run_{uuid.uuid4().hex[:12]}"
    RUN_OUTPUT_DIR = (PROJECT_ROOT / "outputs" / "framework" / RUN_ID).resolve()

    TOP_K = 12
    DENSE_MODEL_ID = "intfloat/e5-base-v2"
    CROSS_ENCODER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    NLI_MODEL_ID = "FacebookAI/roberta-large-mnli"
    RETRIEVAL_BACKEND = resolve_retrieval_backend("local")
    PG_DSN = pg_dsn_from_env(required=RETRIEVAL_BACKEND == "pgvector") if RETRIEVAL_BACKEND == "pgvector" else ""
    PG_TABLE = pg_table_from_env() if RETRIEVAL_BACKEND == "pgvector" else ""

    ensure_absolute(INDEX_ROOT, "INDEX_ROOT")
    ensure_absolute(WORKLOAD_PATH, "WORKLOAD_PATH")
    ensure_exists(WORKLOAD_PATH, "WORKLOAD_PATH")

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

    workload = read_jsonl(WORKLOAD_PATH)
    if MODE == "debug":
        workload = workload[: min(len(workload), 5)]

    if not workload:
        raise RuntimeError("No workload entries found")

    verifier = NLIVerifier(model_id=NLI_MODEL_ID, device=None)

    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_reports: list[dict] = []
    provenance_rows: list[dict] = []

    for item in workload:
        query_id = str(item["query_id"])
        query = str(item["query"])

        bm25 = retrieve_tfidf(
            query=query,
            vectorizer=lexical_vectorizer,
            matrix=lexical_matrix,
            doc_ids=lexical_doc_ids,
            top_k=200,
        )
        dense = retrieve_dense(
            query=query,
            model_id=DENSE_MODEL_ID,
            embeddings=dense_embeddings,
            doc_ids=dense_doc_ids,
            top_k=200,
        ) if RETRIEVAL_BACKEND == "local" else retrieve_dense_pgvector(
            query=query,
            model_id=DENSE_MODEL_ID,
            dsn=PG_DSN,
            table_name=PG_TABLE,
            top_k=200,
        )
        fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense})
        reranked = rerank_cross_encoder(
            query=query,
            candidates=fused[:150],
            doc_text_by_id=doc_text_by_id,
            model_id=CROSS_ENCODER_MODEL_ID,
            top_k=TOP_K,
        )

        top_doc_ids = [row.doc_id for row in reranked]

        retrieval_record = make_provenance_record(
            run_id=RUN_ID,
            tool_name="hybrid_retriever",
            tool_version="1.0.0",
            model_id=f"{DENSE_MODEL_ID}+{CROSS_ENCODER_MODEL_ID}",
            params={"top_k": TOP_K, "rrf": True},
            inputs_ref={"query_id": query_id, "query": query},
            outputs_ref={"result_id": f"{query_id}_retrieval"},
            evidence=[
                {
                    "doc_id": row.doc_id,
                    "chunk_id": f"{row.doc_id}:0",
                    "score_components": row.score_components,
                    "span_offsets": None,
                }
                for row in reranked
            ],
        )
        provenance_rows.append(retrieval_record.to_dict())

        claims = item.get("claims", [])
        claim_rows = [
            {
                "claim_id": f"{query_id}_c{idx}",
                "claim": claim,
                "evidence_doc_ids": top_doc_ids[:3],
                "category": "A",
            }
            for idx, claim in enumerate(claims, start=1)
        ]

        verdict_rows = []
        faithfulness_summary = {
            "total_claims": 0,
            "faithfulness": 0.0,
            "contradiction_rate": 0.0,
            "unsupported_rate": 0.0,
        }
        if claim_rows:
            verdicts, faithfulness_summary = evaluate_claims_with_nli(
                verifier=verifier,
                claims=claim_rows,
                doc_text_by_id=doc_text_by_id,
            )
            verdict_rows = [row.to_dict() for row in verdicts]

            verify_record = make_provenance_record(
                run_id=RUN_ID,
                tool_name="nli_verifier",
                tool_version="1.0.0",
                model_id=NLI_MODEL_ID,
                params={"claims": len(claim_rows)},
                inputs_ref={"query_id": query_id, "claim_count": len(claim_rows)},
                outputs_ref={"result_id": f"{query_id}_verification"},
                evidence=[
                    {
                        "doc_id": evidence_doc_id,
                        "chunk_id": f"{evidence_doc_id}:0",
                        "score_components": {},
                        "span_offsets": None,
                    }
                    for evidence_doc_id in top_doc_ids[:3]
                ],
            )
            provenance_rows.append(verify_record.to_dict())

        final_reports.append(
            {
                "query_id": query_id,
                "query": query,
                "top_doc_ids": top_doc_ids,
                "retrieval": [row.to_dict() for row in reranked],
                "claims": claim_rows,
                "verdicts": verdict_rows,
                "faithfulness": faithfulness_summary,
            }
        )

    write_jsonl(RUN_OUTPUT_DIR / "reports.jsonl", final_reports)
    write_jsonl(RUN_OUTPUT_DIR / "provenance.jsonl", provenance_rows)
    write_json(
        RUN_OUTPUT_DIR / "run_summary.json",
        {
            "run_id": RUN_ID,
            "mode": MODE,
            "seed": SEED,
            "retrieval_backend": RETRIEVAL_BACKEND,
            "nli_device": verifier.device,
            "nli_fallback_reason": verifier.fallback_reason,
            "device_report": runtime_device_report(),
            "workload_size": len(final_reports),
            "output_dir": str(RUN_OUTPUT_DIR),
        },
    )

    print(f"Run completed: {RUN_ID}")
    print(f"Output directory: {RUN_OUTPUT_DIR}")
