from __future__ import annotations

import math
import os
import re
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

DEFAULT_HNSW_EF_CONSTRUCTION = 128
DEFAULT_MAX_PARALLEL_MAINTENANCE_WORKERS = 6


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


def parse_memory_env(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    if not re.fullmatch(r"\d+\s*(?:[kKmMgGtT]?B)", raw):
        raise ValueError(
            f"Environment variable {name} must look like '512MB' or '2GB', got: {raw}"
        )
    return raw.replace(" ", "")


def parse_retrieval_mode() -> bool:
    return parse_bool_env(
        "CORPUSAGENT2_ENABLE_DENSE_RETRIEVAL",
        parse_bool_env(
            "CORPUSAGENT2_PG_INCLUDE_EMBEDDINGS",
            parse_bool_env("CORPUSAGENT2_BUILD_DENSE_ASSETS", False),
        ),
    )


def recommended_ivfflat_lists(total_rows: int) -> int:
    if total_rows <= 0:
        return 1
    if total_rows <= 1_000_000:
        return max(1, total_rows // 1_000)
    return max(1, int(math.sqrt(total_rows)))


def resolve_ivfflat_lists(total_rows: int) -> int:
    raw = os.getenv("CORPUSAGENT2_PG_IVF_LISTS", "").strip()
    if raw:
        return parse_int_env("CORPUSAGENT2_PG_IVF_LISTS", 2048)
    return recommended_ivfflat_lists(total_rows)


if __name__ == "__main__":
    SEED = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SUMMARY_PATH = (PROJECT_ROOT / "outputs" / "postgres" / "build_index_summary.json").resolve()

    load_project_configuration(PROJECT_ROOT)

    HNSW_M = parse_int_env("CORPUSAGENT2_PG_HNSW_M", 16)
    HNSW_EF_CONSTRUCTION = parse_int_env(
        "CORPUSAGENT2_PG_HNSW_EF_CONSTRUCTION",
        DEFAULT_HNSW_EF_CONSTRUCTION,
    )
    BUILD_IVFFLAT = parse_bool_env("CORPUSAGENT2_PG_BUILD_IVFFLAT", True)
    BUILD_HNSW = parse_bool_env("CORPUSAGENT2_PG_BUILD_HNSW", True)
    ENABLE_DENSE_RETRIEVAL = parse_retrieval_mode()
    MAINTENANCE_WORK_MEM = parse_memory_env("CORPUSAGENT2_PG_MAINTENANCE_WORK_MEM", "512MB")
    PARALLEL_MAINTENANCE_WORKERS = parse_int_env(
        "CORPUSAGENT2_PG_MAX_PARALLEL_MAINTENANCE_WORKERS",
        DEFAULT_MAX_PARALLEL_MAINTENANCE_WORKERS,
    )

    ensure_absolute(SUMMARY_PATH, "SUMMARY_PATH")
    set_global_seed(SEED)

    dsn = pg_dsn_from_env(required=True)
    table_name = pg_table_from_env(default="article_corpus")
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
    ivf_lists = 0

    if ENABLE_DENSE_RETRIEVAL:
        with connect(dsn) as conn:
            with conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                total_rows = int(cursor.fetchone()[0])

                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE dense_embedding IS NOT NULL;")
                dense_rows = int(cursor.fetchone()[0])

                if dense_rows == 0:
                    ENABLE_DENSE_RETRIEVAL = False
                else:
                    ivf_lists = resolve_ivfflat_lists(dense_rows)
                    if MAINTENANCE_WORK_MEM:
                        # Use a larger session-local budget for ANN index creation on full-corpus builds.
                        cursor.execute(f"SET maintenance_work_mem = '{MAINTENANCE_WORK_MEM}';")
                    # Request more workers for pgvector ANN index builds when the server allows it.
                    cursor.execute(
                        f"SET max_parallel_maintenance_workers = {PARALLEL_MAINTENANCE_WORKERS};"
                    )

                if ENABLE_DENSE_RETRIEVAL and BUILD_IVFFLAT:
                    cursor.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {ivfflat_index_name}
                        ON {table_name}
                        USING ivfflat (dense_embedding vector_cosine_ops)
                        WITH (lists = {ivf_lists});
                        """
                    )
                    built_indices.append(ivfflat_index_name)

                if ENABLE_DENSE_RETRIEVAL and BUILD_HNSW:
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
        "dense_retrieval_enabled": ENABLE_DENSE_RETRIEVAL,
        "build_ivfflat": BUILD_IVFFLAT,
        "build_hnsw": BUILD_HNSW,
        "ivf_lists": ivf_lists,
        "hnsw_m": HNSW_M,
        "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        "maintenance_work_mem": MAINTENANCE_WORK_MEM or "",
        "max_parallel_maintenance_workers": PARALLEL_MAINTENANCE_WORKERS,
        "built_indices": built_indices,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(SUMMARY_PATH, summary)

    if ENABLE_DENSE_RETRIEVAL:
        print(f"Built pgvector indices on table: {table_name}")
        print(f"Indices: {built_indices}")
    else:
        print(f"Skipped pgvector index build for table {table_name}; dense retrieval is disabled or embeddings are absent.")
    print(f"Summary: {SUMMARY_PATH}")
