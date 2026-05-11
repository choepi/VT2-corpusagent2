from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.app_config import load_project_configuration
from corpusagent2.io_utils import ensure_absolute, write_json
from corpusagent2.model_config import dense_model_id_from_env
from corpusagent2.retrieval import _load_sentence_transformer, dense_asset_health, pg_dsn_from_env, pg_table_from_env
from corpusagent2.seed import set_global_seed


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return max(int(raw), 1)


def vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector.astype(np.float32)) + "]"


def _safe_identifier(value: str) -> str:
    candidate = str(value).strip()
    if not candidate.replace("_", "").isalnum():
        raise ValueError(f"Invalid SQL identifier: {candidate}")
    return candidate


if __name__ == "__main__":
    SEED = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SUMMARY_PATH = (PROJECT_ROOT / "outputs" / "postgres" / "backfill_dense_embeddings_summary.json").resolve()
    DENSE_DIR = (PROJECT_ROOT / "data" / "indices" / "dense").resolve()
    DENSE_MODEL_ID = dense_model_id_from_env()
    FETCH_BATCH_SIZE = parse_int_env("CORPUSAGENT2_PG_BACKFILL_FETCH_BATCH_SIZE", 256)
    ENCODE_BATCH_SIZE = parse_int_env("CORPUSAGENT2_PG_BACKFILL_ENCODE_BATCH_SIZE", 64)
    LIMIT_ROWS = parse_int_env("CORPUSAGENT2_PG_BACKFILL_LIMIT_ROWS", 0) if os.getenv("CORPUSAGENT2_PG_BACKFILL_LIMIT_ROWS", "").strip() else 0
    PREFER_LOCAL_ASSETS = parse_bool_env("CORPUSAGENT2_PG_BACKFILL_PREFER_LOCAL_ASSETS", True)

    ensure_absolute(SUMMARY_PATH, "SUMMARY_PATH")
    load_project_configuration(PROJECT_ROOT)
    set_global_seed(SEED)

    dsn = pg_dsn_from_env(required=True)
    table_name = _safe_identifier(pg_table_from_env(default="article_corpus"))

    try:
        from psycopg import connect
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("psycopg is required for Postgres integration.") from exc

    dense_health = dense_asset_health(DENSE_DIR)
    use_local_assets = bool(PREFER_LOCAL_ASSETS and dense_health.get("ready"))
    dense_embeddings = None
    embedding_row_by_doc_id: dict[str, int] = {}
    model = None

    if use_local_assets:
        dense_embeddings = np.load(DENSE_DIR / "dense_embeddings.npy", mmap_mode="r")
        dense_doc_ids = joblib.load(DENSE_DIR / "dense_doc_ids.joblib")
        embedding_row_by_doc_id = {str(doc_id): idx for idx, doc_id in enumerate(dense_doc_ids)}
        source_mode = "local_dense_assets"
    else:
        model, _resolved_device = _load_sentence_transformer(DENSE_MODEL_ID, device=None)
        source_mode = "encode_on_the_fly"

    processed_rows = 0
    updated_rows = 0
    skipped_rows = 0
    remaining_rows = 0

    with connect(dsn) as conn:
        with conn.cursor() as cursor:
            while True:
                fetch_sql = (
                    f"SELECT doc_id, title, text FROM {table_name} "
                    "WHERE dense_embedding IS NULL "
                    "ORDER BY doc_id "
                    "LIMIT %s"
                )
                cursor.execute(fetch_sql, (FETCH_BATCH_SIZE,))
                rows = cursor.fetchall()
                if not rows:
                    break

                doc_ids = [str(row[0]) for row in rows]
                vectors_by_doc_id: dict[str, str] = {}

                if use_local_assets and dense_embeddings is not None:
                    for doc_id in doc_ids:
                        row_index = embedding_row_by_doc_id.get(doc_id)
                        if row_index is None:
                            continue
                        vectors_by_doc_id[doc_id] = vector_literal(np.asarray(dense_embeddings[row_index], dtype=np.float32))
                else:
                    texts = [f"{str(row[1] or '')} {str(row[2] or '')}".strip() for row in rows]
                    embeddings = model.encode(
                        texts,
                        batch_size=ENCODE_BATCH_SIZE,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    ).astype(np.float32)
                    for doc_id, vector in zip(doc_ids, embeddings, strict=False):
                        vectors_by_doc_id[doc_id] = vector_literal(vector)

                updates = [
                    (vector_literal_text, doc_id)
                    for doc_id, vector_literal_text in vectors_by_doc_id.items()
                    if vector_literal_text
                ]
                if updates:
                    cursor.executemany(
                        f"UPDATE {table_name} SET dense_embedding = %s::vector WHERE doc_id = %s",
                        updates,
                    )
                    conn.commit()

                processed_rows += len(rows)
                updated_rows += len(updates)
                skipped_rows += len(rows) - len(updates)

                if LIMIT_ROWS and processed_rows >= LIMIT_ROWS:
                    break

            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE dense_embedding IS NULL")
            remaining_rows = int(cursor.fetchone()[0])

    summary = {
        "seed": SEED,
        "table_name": table_name,
        "source_mode": source_mode,
        "dense_model_id": DENSE_MODEL_ID,
        "prefer_local_assets": PREFER_LOCAL_ASSETS,
        "local_dense_health": dense_health,
        "fetch_batch_size": FETCH_BATCH_SIZE,
        "encode_batch_size": ENCODE_BATCH_SIZE,
        "limit_rows": LIMIT_ROWS,
        "processed_rows": processed_rows,
        "updated_rows": updated_rows,
        "skipped_rows": skipped_rows,
        "remaining_rows_without_embedding": remaining_rows,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Backfilled dense embeddings in table {table_name}")
    print(f"Source mode: {source_mode}")
    print(f"Processed rows: {processed_rows}")
    print(f"Updated rows: {updated_rows}")
    print(f"Skipped rows: {skipped_rows}")
    print(f"Remaining rows without embedding: {remaining_rows}")
    print(f"Summary: {SUMMARY_PATH}")
