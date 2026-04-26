from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.io_utils import ensure_absolute, ensure_exists, write_json
from corpusagent2.app_config import load_project_configuration
from corpusagent2.retrieval import dense_asset_health, pg_dsn_from_env, pg_table_from_env
from corpusagent2.seed import resolve_run_mode, set_global_seed


def parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "no", "off"}


def parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:  # pragma: no cover
        raise ValueError(f"Environment variable {name} must be int, got: {raw}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {name} must be > 0, got: {value}")
    return value


def vector_literal(vector: np.ndarray) -> str:
    return np.array2string(
        vector.astype(np.float32),
        separator=",",
        max_line_width=1_000_000,
        precision=8,
        suppress_small=False,
    ).replace("\n", "")


def iter_document_batches(
    path: Path,
    *,
    read_batch_size: int,
    debug_limit: int | None = None,
) -> Iterator[pd.DataFrame]:
    parquet = pq.ParquetFile(path)
    emitted = 0
    for batch in parquet.iter_batches(
        batch_size=max(read_batch_size, 1),
        columns=["doc_id", "title", "text", "published_at", "source"],
    ):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        if debug_limit is not None:
            remaining = debug_limit - emitted
            if remaining <= 0:
                break
            if len(frame) > remaining:
                frame = frame.head(remaining).copy()
        emitted += len(frame)
        yield frame
        if debug_limit is not None and emitted >= debug_limit:
            break


def flush_pending_rows(
    cursor,
    *,
    upsert_with_embedding_sql: str,
    upsert_without_embedding_sql: str,
    pending_rows: list[tuple],
    pending_with_embedding: bool,
) -> None:
    if not pending_rows:
        return
    if pending_with_embedding:
        cursor.executemany(upsert_with_embedding_sql, pending_rows)
    else:
        cursor.executemany(upsert_without_embedding_sql, pending_rows)


if __name__ == "__main__":
    MODE = resolve_run_mode("full")
    SEED = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DOCUMENTS_PARQUET = (PROJECT_ROOT / "data" / "processed" / "documents.parquet").resolve()
    DENSE_EMBEDDINGS_PATH = (PROJECT_ROOT / "data" / "indices" / "dense" / "dense_embeddings.npy").resolve()
    DENSE_DOC_IDS_PATH = (PROJECT_ROOT / "data" / "indices" / "dense" / "dense_doc_ids.joblib").resolve()
    SUMMARY_PATH = (PROJECT_ROOT / "outputs" / "postgres" / "ingest_summary.json").resolve()

    DEBUG_MAX_DOCS = 50_000
    DEFAULT_BATCH_SIZE = 250 if os.name == "nt" else 1000
    DEFAULT_READ_BATCH_SIZE = 2_000 if os.name == "nt" else 10_000
    DEFAULT_COMMIT_EVERY_BATCHES = 8 if os.name == "nt" else 32

    BATCH_SIZE = parse_int_env("CORPUSAGENT2_PG_INGEST_BATCH_SIZE", DEFAULT_BATCH_SIZE)
    READ_BATCH_SIZE = parse_int_env("CORPUSAGENT2_PG_INGEST_READ_BATCH_SIZE", DEFAULT_READ_BATCH_SIZE)
    COMMIT_EVERY_BATCHES = parse_int_env("CORPUSAGENT2_PG_INGEST_COMMIT_EVERY_BATCHES", DEFAULT_COMMIT_EVERY_BATCHES)
    PROGRESS_EVERY_ROWS = parse_int_env("CORPUSAGENT2_PG_INGEST_PROGRESS_EVERY_ROWS", 10_000)
    INCLUDE_EMBEDDINGS = parse_bool_env(
        "CORPUSAGENT2_PG_INCLUDE_EMBEDDINGS",
        parse_bool_env("CORPUSAGENT2_BUILD_DENSE_ASSETS", False),
    )

    load_project_configuration(PROJECT_ROOT)

    ensure_absolute(DOCUMENTS_PARQUET, "DOCUMENTS_PARQUET")
    ensure_absolute(DENSE_EMBEDDINGS_PATH, "DENSE_EMBEDDINGS_PATH")
    ensure_absolute(DENSE_DOC_IDS_PATH, "DENSE_DOC_IDS_PATH")
    ensure_absolute(SUMMARY_PATH, "SUMMARY_PATH")

    ensure_exists(DOCUMENTS_PARQUET, "DOCUMENTS_PARQUET")
    dense_health = dense_asset_health(DENSE_EMBEDDINGS_PATH.parent) if INCLUDE_EMBEDDINGS else {"ready": False, "error": ""}
    if INCLUDE_EMBEDDINGS and not dense_health.get("ready"):
        print(
            f"[warn] Dense assets are not usable ({dense_health.get('error', 'unknown issue')}); "
            "ingesting Postgres rows without embeddings."
        )
        INCLUDE_EMBEDDINGS = False

    if INCLUDE_EMBEDDINGS:
        ensure_exists(DENSE_EMBEDDINGS_PATH, "DENSE_EMBEDDINGS_PATH")
        ensure_exists(DENSE_DOC_IDS_PATH, "DENSE_DOC_IDS_PATH")
    set_global_seed(SEED)

    dsn = pg_dsn_from_env(required=True)
    table_name = pg_table_from_env(default="article_corpus")

    try:
        from psycopg import connect
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "psycopg is required for Postgres integration. Install dependency and retry."
        ) from exc

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
    flushed_batches = 0
    commit_count = 0

    debug_limit = DEBUG_MAX_DOCS if MODE == "debug" else None
    saw_any_rows = False

    with connect(dsn) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SET application_name = 'corpusagent2_ingest';")
            if os.name == "nt":
                # Keep each transaction smaller on Windows/Docker Desktop to avoid making the machine unresponsive.
                cursor.execute("SET synchronous_commit = OFF;")
            for frame in iter_document_batches(
                DOCUMENTS_PARQUET,
                read_batch_size=READ_BATCH_SIZE,
                debug_limit=debug_limit,
            ):
                saw_any_rows = True
                for row in frame.itertuples(index=False):
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
                        flush_pending_rows(
                            cursor,
                            upsert_with_embedding_sql=upsert_with_embedding_sql,
                            upsert_without_embedding_sql=upsert_without_embedding_sql,
                            pending_rows=pending_rows,
                            pending_with_embedding=pending_with_embedding,
                        )
                        pending_rows = []
                        flushed_batches += 1
                        if flushed_batches % COMMIT_EVERY_BATCHES == 0:
                            conn.commit()
                            commit_count += 1
                        pending_with_embedding = with_embedding

                    pending_rows.append(item)
                    rows_written += 1
                    if with_embedding:
                        rows_with_embedding += 1
                    else:
                        rows_without_embedding += 1

                    if len(pending_rows) >= BATCH_SIZE:
                        flush_pending_rows(
                            cursor,
                            upsert_with_embedding_sql=upsert_with_embedding_sql,
                            upsert_without_embedding_sql=upsert_without_embedding_sql,
                            pending_rows=pending_rows,
                            pending_with_embedding=pending_with_embedding,
                        )
                        pending_rows = []
                        flushed_batches += 1
                        if flushed_batches % COMMIT_EVERY_BATCHES == 0:
                            conn.commit()
                            commit_count += 1

                    if rows_written % PROGRESS_EVERY_ROWS == 0:
                        print(
                            f"[progress] written={rows_written} "
                            f"embedded={rows_with_embedding} "
                            f"plain={rows_without_embedding} "
                            f"commits={commit_count}"
                        )

            if pending_rows:
                flush_pending_rows(
                    cursor,
                    upsert_with_embedding_sql=upsert_with_embedding_sql,
                    upsert_without_embedding_sql=upsert_without_embedding_sql,
                    pending_rows=pending_rows,
                    pending_with_embedding=pending_with_embedding,
                )
                flushed_batches += 1
            conn.commit()
            commit_count += 1

    if not saw_any_rows:
        raise RuntimeError("Input documents parquet is empty")

    summary = {
        "mode": MODE,
        "seed": SEED,
        "table_name": table_name,
        "include_embeddings": INCLUDE_EMBEDDINGS,
        "rows_written": rows_written,
        "rows_with_embedding": rows_with_embedding,
        "rows_without_embedding": rows_without_embedding,
        "batch_size": BATCH_SIZE,
        "read_batch_size": READ_BATCH_SIZE,
        "commit_every_batches": COMMIT_EVERY_BATCHES,
        "commit_count": commit_count,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Ingested rows into Postgres table {table_name}: {rows_written}")
    print(f"Rows with embedding: {rows_with_embedding}")
    print(f"Rows without embedding: {rows_without_embedding}")
    print(f"Commits: {commit_count}")
    print(f"Summary: {SUMMARY_PATH}")
