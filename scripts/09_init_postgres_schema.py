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


if __name__ == "__main__":
    SEED = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SUMMARY_PATH = (PROJECT_ROOT / "outputs" / "postgres" / "init_schema_summary.json").resolve()

    load_project_configuration(PROJECT_ROOT)

    ensure_absolute(SUMMARY_PATH, "SUMMARY_PATH")
    set_global_seed(SEED)

    dsn = pg_dsn_from_env(required=True)
    table_name = pg_table_from_env(default="article_corpus")
    btree_source_index = f"idx_{table_name}_source"
    btree_published_index = f"idx_{table_name}_published_at"

    try:
        from psycopg import connect
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "psycopg is required for Postgres integration. Install dependency and retry."
        ) from exc

    create_extension_sql = "CREATE EXTENSION IF NOT EXISTS vector;"
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
      doc_id TEXT PRIMARY KEY,
      title TEXT NOT NULL DEFAULT '',
      text TEXT NOT NULL DEFAULT '',
      published_at TEXT NOT NULL DEFAULT '',
      source TEXT NOT NULL DEFAULT '',
      dense_embedding vector(768)
    );
    """
    create_source_index_sql = f"CREATE INDEX IF NOT EXISTS {btree_source_index} ON {table_name} (source);"
    create_published_index_sql = (
        f"CREATE INDEX IF NOT EXISTS {btree_published_index} ON {table_name} (published_at);"
    )

    with connect(dsn) as conn:
        with conn.cursor() as cursor:
            cursor.execute(create_extension_sql)
            cursor.execute(create_table_sql)
            cursor.execute(create_source_index_sql)
            cursor.execute(create_published_index_sql)
        conn.commit()

    summary = {
        "seed": SEED,
        "table_name": table_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_path": str(SUMMARY_PATH),
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Initialized Postgres schema for table: {table_name}")
    print(f"Summary: {SUMMARY_PATH}")
