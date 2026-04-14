from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.io_utils import ensure_absolute, ensure_exists, read_documents, write_json
from corpusagent2.retrieval import (
    _load_sentence_transformer,
    build_dense_embeddings,
    build_lexical_assets,
    save_dense_assets,
    save_lexical_assets,
)
from corpusagent2.seed import resolve_run_mode, set_global_seed


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return max(int(raw), 1)


def _build_dense_embeddings_streaming(
    df: pd.DataFrame,
    *,
    model_id: str,
    batch_size: int,
    device: str | None,
    chunk_size: int,
    index_dir: Path,
) -> tuple[Path, list[str]]:
    model, _resolved_device = _load_sentence_transformer(model_id=model_id, device=device)
    doc_ids = df["doc_id"].astype(str).tolist()
    embeddings_path = index_dir / "dense_embeddings.npy"
    index_dir.mkdir(parents=True, exist_ok=True)
    resume_existing = _env_flag("CORPUSAGENT2_RESUME_DENSE_ASSETS", True)

    total_rows = int(df.shape[0])
    if total_rows == 0:
        raise RuntimeError("Dense embedding build received an empty dataframe.")

    def _first_zero_row_index(memmap: np.ndarray, scan_chunk: int = 2048) -> int:
        for start in range(0, int(memmap.shape[0]), scan_chunk):
            stop = min(start + scan_chunk, int(memmap.shape[0]))
            chunk = np.asarray(memmap[start:stop], dtype=np.float32)
            zero_rows = np.flatnonzero(~chunk.any(axis=1))
            if zero_rows.size > 0:
                return int(start + zero_rows[0])
        return int(memmap.shape[0])

    first_stop = min(total_rows, chunk_size)
    first_texts = (
        (df.iloc[:first_stop]["title"].fillna("") + " " + df.iloc[:first_stop]["text"].fillna(""))
        .str.replace("\n", " ", regex=False)
        .tolist()
    )
    first_batch = model.encode(
        first_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    embedding_dim = int(first_batch.shape[1])
    start_index = first_stop
    if resume_existing and embeddings_path.exists():
        try:
            dense_memmap = np.load(embeddings_path, mmap_mode="r+")
            if tuple(dense_memmap.shape) != (total_rows, embedding_dim):
                raise RuntimeError(f"shape mismatch: existing={dense_memmap.shape}, expected={(total_rows, embedding_dim)}")
            start_index = _first_zero_row_index(dense_memmap)
            if start_index < first_stop:
                dense_memmap[start_index:first_stop] = first_batch[start_index:first_stop]
                start_index = first_stop
        except Exception:
            embeddings_path.unlink(missing_ok=True)
            dense_memmap = np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(total_rows, embedding_dim),
            )
            dense_memmap[:first_stop] = first_batch
            start_index = first_stop
    else:
        dense_memmap = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, embedding_dim),
        )
        dense_memmap[:first_stop] = first_batch
        start_index = first_stop

    for start in range(start_index, total_rows, chunk_size):
        stop = min(start + chunk_size, total_rows)
        texts = (
            (df.iloc[start:stop]["title"].fillna("") + " " + df.iloc[start:stop]["text"].fillna(""))
            .str.replace("\n", " ", regex=False)
            .tolist()
        )
        batch = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        dense_memmap[start:stop] = batch

    dense_memmap.flush()
    return embeddings_path, doc_ids


if __name__ == "__main__":
    MODE = resolve_run_mode("full")
    SEED = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DOCUMENTS_PARQUET = (PROJECT_ROOT / "data" / "processed" / "documents.parquet").resolve()
    INDEX_ROOT = (PROJECT_ROOT / "data" / "indices").resolve()
    SUMMARY_PATH = (PROJECT_ROOT / "outputs" / "build_retrieval_assets_summary.json").resolve()

    DENSE_MODEL_ID = "intfloat/e5-base-v2"
    DENSE_BATCH_SIZE = _env_int("CORPUSAGENT2_DENSE_BATCH_SIZE", 64)
    DENSE_CHUNK_SIZE = _env_int("CORPUSAGENT2_DENSE_CHUNK_SIZE", 2048)
    DENSE_DEVICE = None
    TFIDF_MAX_FEATURES = _env_int("CORPUSAGENT2_TFIDF_MAX_FEATURES", 250_000)
    DEBUG_MAX_DOCS = _env_int("CORPUSAGENT2_DEBUG_MAX_DOCS", 30_000)
    BUILD_LEXICAL = _env_flag("CORPUSAGENT2_BUILD_LEXICAL_ASSETS", True)
    BUILD_DENSE = _env_flag("CORPUSAGENT2_BUILD_DENSE_ASSETS", True)
    STREAM_DENSE = _env_flag("CORPUSAGENT2_STREAM_DENSE_ASSETS", True)

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

    if BUILD_LEXICAL:
        vectorizer, tfidf_matrix, tfidf_doc_ids = build_lexical_assets(df=df, max_features=TFIDF_MAX_FEATURES)
        save_lexical_assets(index_dir=lexical_dir, vectorizer=vectorizer, matrix=tfidf_matrix, doc_ids=tfidf_doc_ids)

    if BUILD_DENSE:
        if STREAM_DENSE:
            _build_dense_embeddings_streaming(
                df=df,
                model_id=DENSE_MODEL_ID,
                batch_size=DENSE_BATCH_SIZE,
                device=DENSE_DEVICE,
                chunk_size=DENSE_CHUNK_SIZE,
                index_dir=dense_dir,
            )
            dense_doc_ids = df["doc_id"].astype(str).tolist()
            joblib.dump(dense_doc_ids, dense_dir / "dense_doc_ids.joblib")
        else:
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
        "lexical_index": str(lexical_dir) if BUILD_LEXICAL else "",
        "dense_index": str(dense_dir) if BUILD_DENSE else "",
        "doc_metadata": str(metadata_path),
        "dense_model_id": DENSE_MODEL_ID,
        "build_lexical_assets": BUILD_LEXICAL,
        "build_dense_assets": BUILD_DENSE,
        "stream_dense_assets": STREAM_DENSE,
        "dense_batch_size": DENSE_BATCH_SIZE,
        "dense_chunk_size": DENSE_CHUNK_SIZE,
        "tfidf_max_features": TFIDF_MAX_FEATURES,
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Built retrieval assets for {df.shape[0]} documents")
    print(f"Summary: {SUMMARY_PATH}")
