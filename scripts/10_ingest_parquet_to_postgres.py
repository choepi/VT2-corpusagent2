from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.io_utils import ensure_absolute, ensure_exists, read_documents, write_json
from corpusagent2.app_config import load_project_configuration
from corpusagent2.retrieval import pg_dsn_from_env, pg_table_from_env
from corpusagent2.seed import resolve_run_mode, set_global_seed


def parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "no", "off"}


def vector_literal(vector: np.ndarray) -> str:
    return np.array2string(
        vector.astype(np.float32),
        separator=",",
        max_line_width=1_000_000,
        precision=8,
        suppress_small=False,
    ).replace("\n", "")


if __name__ == "__main__":
    MODE = resolve_run_mode("full")
    SEED = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DOCUMENTS_PARQUET = (PROJECT_ROOT / "data" / "processed" / "documents.parquet").resolve()
    DENSE_EMBEDDINGS_PATH = (PROJECT_ROOT / "data" / "indices" / "dense" / "dense_embeddings.npy").resolve()
    DENSE_DOC_IDS_PATH = (PROJECT_ROOT / "data" / "indices" / "dense" / "dense_doc_ids.joblib").resolve()
    SUMMARY_PATH = (PROJECT_ROOT / "outputs" / "postgres" / "ingest_summary.json").resolve()

    DEBUG_MAX_DOCS = 50_000
    BATCH_SIZE = 2_000
    INCLUDE_EMBEDDINGS = parse_bool_env("CORPUSAGENT2_PG_INCLUDE_EMBEDDINGS", True)

    load_project_configuration(PROJECT_ROOT)

    ensure_absolute(DOCUMENTS_PARQUET, "DOCUMENTS_PARQUET")
    ensure_absolute(DENSE_EMBEDDINGS_PATH, "DENSE_EMBEDDINGS_PATH")
    ensure_absolute(DENSE_DOC_IDS_PATH, "DENSE_DOC_IDS_PATH")
    ensure_absolute(SUMMARY_PATH, "SUMMARY_PATH")
    ensure_exists(DOCUMENTS_PARQUET, "DOCUMENTS_PARQUET")
    if INCLUDE_EMBEDDINGS:
        ensure_exists(DENSE_EMBEDDINGS_PATH, "DENSE_EMBEDDINGS_PATH")
        ensure_exists(DENSE_DOC_IDS_PATH, "DENSE_DOC_IDS_PATH")

    set_global_seed(SEED)

    dsn = pg_dsn_from_env(required=True)
    table_name = pg_table_from_env(default="ca_documents")

    try:
        from psycopg import connect
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "psycopg is required for Postgres integration. Install dependency and retry."
        ) from exc

    df = read_documents(DOCUMENTS_PARQUET)
    if MODE == "debug":
        df = df.head(DEBUG_MAX_DOCS).copy()
    if df.empty:
        raise RuntimeError("Input documents parquet is empty")

    embeddings = None
    embedding_index_by_doc_id: dict[str, int] = {}
    if INCLUDE_EMBEDDINGS:
        embeddings = np.load(DENSE_EMBEDDINGS_PATH, mmap_mode="r")
        dense_doc_ids = joblib.load(DENSE_DOC_IDS_PATH)
        embedding_index_by_doc_id = {
            str(doc_id): idx for idx, doc_id in enumerate(dense_doc_ids)
        }

    upsert_with_embedding_sql = f"""
    INSERT INTO {table_name} (doc_id, title, text, published_at, source, dense_embedding)
    VALUES (%s, %s, %s, %s, %s, %s::vector)
    ON CONFLICT (doc_id) DO UPDATE SET
      title = EXCLUDED.title,
      text = EXCLUDED.text,
      published_at = EXCLUDED.published_at,
      source = EXCLUDED.source,
      dense_embedding = EXCLUDED.dense_embedding;
    """
    upsert_without_embedding_sql = f"""
    INSERT INTO {table_name} (doc_id, title, text, published_at, source)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (doc_id) DO UPDATE SET
      title = EXCLUDED.title,
      text = EXCLUDED.text,
      published_at = EXCLUDED.published_at,
      source = EXCLUDED.source;
    """

    rows_written = 0
    rows_with_embedding = 0
    rows_without_embedding = 0
    pending_rows: list[tuple] = []
    pending_with_embedding = False

    with connect(dsn) as conn:
        with conn.cursor() as cursor:
            for row in df.itertuples(index=False):
                doc_id = str(row.doc_id)
                title = str(row.title or "")
                text = str(row.text or "")
                published_at = str(row.published_at or "")
                source = str(getattr(row, "source", "") or "")

                if INCLUDE_EMBEDDINGS:
                    emb_idx = embedding_index_by_doc_id.get(doc_id)
                    if emb_idx is not None:
                        emb_vector = embeddings[emb_idx]
                        emb_literal = vector_literal(emb_vector)
                        item = (doc_id, title, text, published_at, source, emb_literal)
                        with_embedding = True
                    else:
                        item = (doc_id, title, text, published_at, source)
                        with_embedding = False
                else:
                    item = (doc_id, title, text, published_at, source)
                    with_embedding = False

                if not pending_rows:
                    pending_with_embedding = with_embedding
                elif pending_with_embedding != with_embedding:
                    if pending_with_embedding:
                        cursor.executemany(upsert_with_embedding_sql, pending_rows)
                    else:
                        cursor.executemany(upsert_without_embedding_sql, pending_rows)
                    pending_rows = []
                    pending_with_embedding = with_embedding

                pending_rows.append(item)
                rows_written += 1
                if with_embedding:
                    rows_with_embedding += 1
                else:
                    rows_without_embedding += 1

                if len(pending_rows) >= BATCH_SIZE:
                    if pending_with_embedding:
                        cursor.executemany(upsert_with_embedding_sql, pending_rows)
                    else:
                        cursor.executemany(upsert_without_embedding_sql, pending_rows)
                    pending_rows = []

            if pending_rows:
                if pending_with_embedding:
                    cursor.executemany(upsert_with_embedding_sql, pending_rows)
                else:
                    cursor.executemany(upsert_without_embedding_sql, pending_rows)

        conn.commit()

    summary = {
        "mode": MODE,
        "seed": SEED,
        "table_name": table_name,
        "include_embeddings": INCLUDE_EMBEDDINGS,
        "rows_written": rows_written,
        "rows_with_embedding": rows_with_embedding,
        "rows_without_embedding": rows_without_embedding,
        "batch_size": BATCH_SIZE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Ingested rows into Postgres table {table_name}: {rows_written}")
    print(f"Rows with embedding: {rows_with_embedding}")
    print(f"Rows without embedding: {rows_without_embedding}")
    print(f"Summary: {SUMMARY_PATH}")
