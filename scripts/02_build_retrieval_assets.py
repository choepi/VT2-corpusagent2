from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.io_utils import ensure_absolute, ensure_exists, read_documents, write_json
from corpusagent2.retrieval import (
    build_dense_embeddings,
    build_lexical_assets,
    save_dense_assets,
    save_lexical_assets,
)
from corpusagent2.seed import resolve_run_mode, set_global_seed


if __name__ == "__main__":
    MODE = resolve_run_mode("full")
    SEED = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DOCUMENTS_PARQUET = (PROJECT_ROOT / "data" / "processed" / "documents.parquet").resolve()
    INDEX_ROOT = (PROJECT_ROOT / "data" / "indices").resolve()
    SUMMARY_PATH = (PROJECT_ROOT / "outputs" / "build_retrieval_assets_summary.json").resolve()

    DENSE_MODEL_ID = "intfloat/e5-base-v2"
    DENSE_BATCH_SIZE = 128
    DENSE_DEVICE = None
    TFIDF_MAX_FEATURES = 250_000
    DEBUG_MAX_DOCS = 30_000

    ensure_absolute(DOCUMENTS_PARQUET, "DOCUMENTS_PARQUET")
    ensure_absolute(INDEX_ROOT, "INDEX_ROOT")
    ensure_absolute(SUMMARY_PATH, "SUMMARY_PATH")
    ensure_exists(DOCUMENTS_PARQUET, "DOCUMENTS_PARQUET")

    set_global_seed(SEED)

    df = read_documents(DOCUMENTS_PARQUET)
    if MODE == "debug":
        df = df.head(DEBUG_MAX_DOCS).copy()

    if df.empty:
        raise RuntimeError("Input documents parquet is empty")

    lexical_dir = INDEX_ROOT / "lexical"
    dense_dir = INDEX_ROOT / "dense"

    vectorizer, tfidf_matrix, tfidf_doc_ids = build_lexical_assets(df=df, max_features=TFIDF_MAX_FEATURES)
    save_lexical_assets(index_dir=lexical_dir, vectorizer=vectorizer, matrix=tfidf_matrix, doc_ids=tfidf_doc_ids)

    dense_embeddings, dense_doc_ids = build_dense_embeddings(
        df=df,
        model_id=DENSE_MODEL_ID,
        batch_size=DENSE_BATCH_SIZE,
        device=DENSE_DEVICE,
    )
    save_dense_assets(index_dir=dense_dir, embeddings=dense_embeddings, doc_ids=dense_doc_ids)

    metadata = pd.DataFrame(
        {
            "doc_id": df["doc_id"].astype(str),
            "title": df["title"].fillna(""),
            "text": df["text"].fillna(""),
            "published_at": df["published_at"].fillna(""),
            "source": df.get("source", pd.Series([""] * df.shape[0])),
        }
    )
    metadata_path = INDEX_ROOT / "doc_metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)

    summary = {
        "mode": MODE,
        "seed": SEED,
        "documents_indexed": int(df.shape[0]),
        "lexical_index": str(lexical_dir),
        "dense_index": str(dense_dir),
        "doc_metadata": str(metadata_path),
        "dense_model_id": DENSE_MODEL_ID,
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Built lexical and dense retrieval assets for {df.shape[0]} documents")
    print(f"Summary: {SUMMARY_PATH}")

