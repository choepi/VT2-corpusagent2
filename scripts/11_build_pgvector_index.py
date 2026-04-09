from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.io_utils import ensure_absolute, write_json
from corpusagent2.app_config import load_project_configuration
from corpusagent2.retrieval import pg_dsn_from_env, pg_table_from_env
from corpusagent2.seed import set_global_seed


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


if __name__ == "__main__":
    SEED = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SUMMARY_PATH = (PROJECT_ROOT / "outputs" / "postgres" / "build_index_summary.json").resolve()

    load_project_configuration(PROJECT_ROOT)

    IVF_LISTS = parse_int_env("CORPUSAGENT2_PG_IVF_LISTS", 2048)
    HNSW_M = parse_int_env("CORPUSAGENT2_PG_HNSW_M", 16)
    HNSW_EF_CONSTRUCTION = parse_int_env("CORPUSAGENT2_PG_HNSW_EF_CONSTRUCTION", 128)
    BUILD_IVFFLAT = parse_bool_env("CORPUSAGENT2_PG_BUILD_IVFFLAT", True)
    BUILD_HNSW = parse_bool_env("CORPUSAGENT2_PG_BUILD_HNSW", True)

    ensure_absolute(SUMMARY_PATH, "SUMMARY_PATH")
    set_global_seed(SEED)

    dsn = pg_dsn_from_env(required=True)
    table_name = pg_table_from_env(default="ca_documents")
    ivfflat_index_name = f"idx_{table_name}_embedding_ivfflat"
    hnsw_index_name = f"idx_{table_name}_embedding_hnsw"

    try:
        from psycopg import connect
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "psycopg is required for Postgres integration. Install dependency and retry."
        ) from exc

    dense_rows = 0
    total_rows = 0
    built_indices: list[str] = []

    with connect(dsn) as conn:
        with conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            total_rows = int(cursor.fetchone()[0])

            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE dense_embedding IS NOT NULL;")
            dense_rows = int(cursor.fetchone()[0])

            if dense_rows == 0:
                raise RuntimeError(
                    f"No rows with dense_embedding found in {table_name}. "
                    "Ingest embeddings first or disable embedding index build."
                )

            if BUILD_IVFFLAT:
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {ivfflat_index_name}
                    ON {table_name}
                    USING ivfflat (dense_embedding vector_cosine_ops)
                    WITH (lists = {IVF_LISTS});
                    """
                )
                built_indices.append(ivfflat_index_name)

            if BUILD_HNSW:
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {hnsw_index_name}
                    ON {table_name}
                    USING hnsw (dense_embedding vector_cosine_ops)
                    WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});
                    """
                )
                built_indices.append(hnsw_index_name)

            cursor.execute(f"ANALYZE {table_name};")
        conn.commit()

    summary = {
        "seed": SEED,
        "table_name": table_name,
        "total_rows": total_rows,
        "rows_with_dense_embedding": dense_rows,
        "build_ivfflat": BUILD_IVFFLAT,
        "build_hnsw": BUILD_HNSW,
        "ivf_lists": IVF_LISTS,
        "hnsw_m": HNSW_M,
        "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        "built_indices": built_indices,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Built pgvector indices on table: {table_name}")
    print(f"Indices: {built_indices}")
    print(f"Summary: {SUMMARY_PATH}")
