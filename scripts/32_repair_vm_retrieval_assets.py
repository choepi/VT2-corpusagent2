from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.app_config import load_project_configuration
from corpusagent2.io_utils import ensure_absolute, write_json
from corpusagent2.retrieval import dense_asset_health, pg_dsn_from_env, pg_table_from_env
from corpusagent2.runtime_context import CorpusRuntime


DEFAULT_SUMMARY_PATH = (REPO_ROOT / "outputs" / "postgres" / "vm_repair_summary.json").resolve()
DEFAULT_IVF_LISTS_DIVISOR = 1_000
DEFAULT_IVF_LISTS_CAP = 4_096


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair a moved VM copy of the CorpusAgent2 retrieval assets without "
            "re-downloading the corpus or regenerating embeddings."
        )
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Where to write the repair summary JSON (default: {DEFAULT_SUMMARY_PATH})",
    )
    parser.add_argument(
        "--skip-doc-id-repair",
        action="store_true",
        help="Skip rebuilding dense_doc_ids.joblib from existing metadata.",
    )
    parser.add_argument(
        "--skip-collation-repair",
        action="store_true",
        help="Skip rebuilding Postgres indexes and refreshing the database collation version.",
    )
    parser.add_argument(
        "--skip-pgvector-index",
        action="store_true",
        help="Skip building a pgvector IVFFLAT index from the existing dense_embedding column.",
    )
    parser.add_argument(
        "--build-hnsw",
        action="store_true",
        help="Also build an HNSW pgvector index in addition to IVFFLAT.",
    )
    parser.add_argument(
        "--skip-local-dense-restore",
        action="store_true",
        help=(
            "Skip reconstructing dense_embeddings.npy from the existing Postgres "
            "dense_embedding column when the local dense asset is still invalid."
        ),
    )
    return parser.parse_args()


def _quote_ident(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _sample_positions(total_rows: int) -> list[int]:
    if total_rows <= 0:
        return []
    positions = {
        0,
        min(1, total_rows - 1),
        min(2, total_rows - 1),
        total_rows // 4,
        total_rows // 2,
        (total_rows * 3) // 4,
        total_rows - 1,
    }
    spread = np.linspace(0, total_rows - 1, num=min(total_rows, 8), dtype=np.int64)
    positions.update(int(value) for value in spread.tolist())
    return sorted(pos for pos in positions if 0 <= pos < total_rows)


def _recommended_ivfflat_lists(total_rows: int) -> int:
    if total_rows <= 0:
        return 1
    return max(1, min(DEFAULT_IVF_LISTS_CAP, total_rows // DEFAULT_IVF_LISTS_DIVISOR))


def _load_doc_id_source(
    metadata_path: Path,
    documents_path: Path,
    expected_rows: int,
) -> tuple[list[str], str]:
    for candidate in (metadata_path, documents_path):
        if not candidate.exists():
            continue
        frame = pd.read_parquet(candidate, columns=["doc_id"])
        doc_ids = frame["doc_id"].astype(str).tolist()
        if len(doc_ids) != expected_rows:
            continue
        if len(set(doc_ids)) != expected_rows:
            raise RuntimeError(f"Duplicate doc_id values detected in {candidate}")
        return doc_ids, str(candidate)
    raise RuntimeError(
        "Could not reconstruct dense_doc_ids.joblib because neither doc metadata nor processed documents "
        f"matched the existing embedding row count {expected_rows}."
    )


def _repair_dense_doc_ids(project_root: Path) -> dict[str, Any]:
    dense_dir = (project_root / "data" / "indices" / "dense").resolve()
    embeddings_path = dense_dir / "dense_embeddings.npy"
    doc_ids_path = dense_dir / "dense_doc_ids.joblib"
    metadata_path = (project_root / "data" / "indices" / "doc_metadata.parquet").resolve()
    documents_path = (project_root / "data" / "processed" / "documents.parquet").resolve()
    result: dict[str, Any] = {
        "embeddings_path": str(embeddings_path),
        "doc_ids_path": str(doc_ids_path),
        "metadata_path": str(metadata_path),
        "documents_path": str(documents_path),
        "embeddings_exists": embeddings_path.exists(),
        "doc_ids_existed_before": doc_ids_path.exists(),
        "doc_ids_written": False,
        "doc_id_source_path": "",
        "row_count": 0,
        "sample_positions": [],
        "ready_after": False,
        "error": "",
    }
    if not embeddings_path.exists():
        result["error"] = "dense_embeddings.npy is missing."
        return result

    embeddings = np.load(embeddings_path, mmap_mode="r")
    row_count = int(embeddings.shape[0])
    result["row_count"] = row_count

    doc_ids, source_path = _load_doc_id_source(
        metadata_path=metadata_path,
        documents_path=documents_path,
        expected_rows=row_count,
    )
    result["doc_id_source_path"] = source_path

    sample_positions = _sample_positions(row_count)
    result["sample_positions"] = sample_positions

    joblib.dump(doc_ids, doc_ids_path)
    result["doc_ids_written"] = True

    health = dense_asset_health(dense_dir, expected_rows=row_count)
    result["ready_after"] = bool(health.get("ready"))
    if not result["ready_after"]:
        result["error"] = str(health.get("error", "dense asset validation failed after doc_id repair"))
    return result


def _restore_local_dense_from_pgvector(
    project_root: Path,
    dsn: str,
    table_name: str,
) -> dict[str, Any]:
    from psycopg import connect

    dense_dir = (project_root / "data" / "indices" / "dense").resolve()
    embeddings_path = dense_dir / "dense_embeddings.npy"
    doc_ids_path = dense_dir / "dense_doc_ids.joblib"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    temp_embeddings_path = dense_dir / f"dense_embeddings.from_pgvector.{timestamp}.npy"
    temp_doc_ids_path = dense_dir / f"dense_doc_ids.from_pgvector.{timestamp}.joblib"
    backup_embeddings_path = dense_dir / f"dense_embeddings.vm_backup.{timestamp}.npy"
    backup_doc_ids_path = dense_dir / f"dense_doc_ids.vm_backup.{timestamp}.joblib"
    result: dict[str, Any] = {
        "restored": False,
        "table_name": table_name,
        "temp_embeddings_path": str(temp_embeddings_path),
        "temp_doc_ids_path": str(temp_doc_ids_path),
        "backup_embeddings_path": "",
        "backup_doc_ids_path": "",
        "rows_written": 0,
        "embedding_dim": 0,
        "ready_after": False,
        "error": "",
    }

    dense_rows = 0
    with connect(dsn) as count_conn:
        with count_conn.cursor() as count_cursor:
            count_cursor.execute(
                f"SELECT COUNT(*) FROM {_quote_ident(table_name)} WHERE dense_embedding IS NOT NULL"
            )
            dense_rows = int(count_cursor.fetchone()[0])
    if dense_rows <= 0:
        result["error"] = "No non-NULL dense_embedding rows were available in Postgres."
        return result

    doc_ids: list[str] = []
    dense_memmap = None
    rows_written = 0
    embedding_dim = 0

    with connect(dsn) as conn:
        with conn.cursor(name="dense_export") as cursor:
            cursor.execute(
                f"""
                SELECT doc_id::text, dense_embedding::text
                FROM {_quote_ident(table_name)}
                WHERE dense_embedding IS NOT NULL
                ORDER BY doc_id
                """
            )
            while True:
                rows = cursor.fetchmany(512)
                if not rows:
                    break
                for doc_id, vector_text in rows:
                    values = np.fromstring(str(vector_text).strip()[1:-1], sep=",", dtype=np.float32)
                    if values.size <= 0:
                        raise RuntimeError(f"Could not parse dense embedding for doc_id={doc_id}")
                    if dense_memmap is None:
                        embedding_dim = int(values.size)
                        dense_memmap = open_memmap(
                            temp_embeddings_path,
                            mode="w+",
                            dtype=np.float32,
                            shape=(dense_rows, embedding_dim),
                        )
                    elif int(values.size) != embedding_dim:
                        raise RuntimeError(
                            f"Inconsistent embedding dimension for doc_id={doc_id}: "
                            f"expected {embedding_dim}, got {int(values.size)}"
                        )
                    dense_memmap[rows_written] = values
                    doc_ids.append(str(doc_id))
                    rows_written += 1

    if dense_memmap is None or embedding_dim <= 0:
        raise RuntimeError("Postgres dense export did not yield any rows.")
    dense_memmap.flush()

    if rows_written != dense_rows:
        raise RuntimeError(f"Expected {dense_rows} dense rows but wrote {rows_written} rows.")

    joblib.dump(doc_ids, temp_doc_ids_path)

    if embeddings_path.exists():
        embeddings_path.replace(backup_embeddings_path)
        result["backup_embeddings_path"] = str(backup_embeddings_path)
    if doc_ids_path.exists():
        doc_ids_path.replace(backup_doc_ids_path)
        result["backup_doc_ids_path"] = str(backup_doc_ids_path)

    temp_embeddings_path.replace(embeddings_path)
    temp_doc_ids_path.replace(doc_ids_path)

    health = dense_asset_health(dense_dir, expected_rows=dense_rows)
    result["restored"] = True
    result["rows_written"] = rows_written
    result["embedding_dim"] = embedding_dim
    result["ready_after"] = bool(health.get("ready"))
    if not result["ready_after"]:
        result["error"] = str(health.get("error", "dense asset validation failed after pgvector restore"))
    return result


def _collation_versions(cursor) -> tuple[str, str]:
    cursor.execute(
        """
        SELECT
          COALESCE(datcollversion, '') AS stored_version,
          COALESCE(pg_database_collation_actual_version(oid), '') AS actual_version
        FROM pg_database
        WHERE datname = current_database()
        """
    )
    stored_version, actual_version = cursor.fetchone()
    return str(stored_version or ""), str(actual_version or "")


def _repair_postgres_collation(dsn: str) -> dict[str, Any]:
    from psycopg import connect

    result: dict[str, Any] = {
        "database": "",
        "stored_version_before": "",
        "actual_version_before": "",
        "mismatch_before": False,
        "reindexed_indices": [],
        "refreshed_database_collation_version": False,
        "stored_version_after": "",
        "actual_version_after": "",
        "mismatch_after": False,
        "error": "",
    }
    with connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT current_database()")
            result["database"] = str(cursor.fetchone()[0])
            stored_before, actual_before = _collation_versions(cursor)
            result["stored_version_before"] = stored_before
            result["actual_version_before"] = actual_before
            result["mismatch_before"] = bool(stored_before and actual_before and stored_before != actual_before)
            if not result["mismatch_before"]:
                result["stored_version_after"] = stored_before
                result["actual_version_after"] = actual_before
                result["mismatch_after"] = False
                return result

            cursor.execute(
                """
                SELECT schemaname, indexname
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY indexname
                """
            )
            public_indexes = [(str(row[0]), str(row[1])) for row in cursor.fetchall()]
            for schema_name, index_name in public_indexes:
                qualified_index = f"{_quote_ident(schema_name)}.{_quote_ident(index_name)}"
                cursor.execute(f"REINDEX INDEX {qualified_index}")
                result["reindexed_indices"].append(f"{schema_name}.{index_name}")

            cursor.execute(
                f"ALTER DATABASE {_quote_ident(result['database'])} REFRESH COLLATION VERSION"
            )
            result["refreshed_database_collation_version"] = True
            stored_after, actual_after = _collation_versions(cursor)
            result["stored_version_after"] = stored_after
            result["actual_version_after"] = actual_after
            result["mismatch_after"] = bool(stored_after and actual_after and stored_after != actual_after)
    return result


def _ensure_pgvector_index(
    dsn: str,
    table_name: str,
    *,
    build_hnsw: bool,
) -> dict[str, Any]:
    from psycopg import connect

    ivfflat_index_name = f"idx_{table_name}_embedding_ivfflat"
    hnsw_index_name = f"idx_{table_name}_embedding_hnsw"
    result: dict[str, Any] = {
        "table_name": table_name,
        "existing_indices_before": [],
        "existing_indices_after": [],
        "built_indices": [],
        "total_rows": 0,
        "dense_rows": 0,
        "ivfflat_lists": 0,
        "error": "",
    }
    with connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(f"SELECT COUNT(*), COUNT(*) FILTER (WHERE dense_embedding IS NOT NULL) FROM {_quote_ident(table_name)}")
            total_rows, dense_rows = cursor.fetchone()
            result["total_rows"] = int(total_rows)
            result["dense_rows"] = int(dense_rows)

            cursor.execute(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = 'public' AND tablename = %s
                ORDER BY indexname
                """,
                (table_name,),
            )
            existing_before = [str(row[0]) for row in cursor.fetchall()]
            result["existing_indices_before"] = existing_before

            if int(dense_rows) <= 0:
                result["existing_indices_after"] = existing_before
                return result

            cursor.execute("SET maintenance_work_mem = '512MB'")
            cursor.execute("SET max_parallel_maintenance_workers = 6")

            ivfflat_lists = _recommended_ivfflat_lists(int(dense_rows))
            result["ivfflat_lists"] = int(ivfflat_lists)

            if ivfflat_index_name not in existing_before:
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {_quote_ident(ivfflat_index_name)}
                    ON {_quote_ident(table_name)}
                    USING ivfflat (dense_embedding vector_cosine_ops)
                    WITH (lists = {ivfflat_lists})
                    """
                )
                result["built_indices"].append(ivfflat_index_name)

            if build_hnsw and hnsw_index_name not in existing_before:
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {_quote_ident(hnsw_index_name)}
                    ON {_quote_ident(table_name)}
                    USING hnsw (dense_embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 128)
                    """
                )
                result["built_indices"].append(hnsw_index_name)

            cursor.execute(f"ANALYZE {_quote_ident(table_name)}")
            cursor.execute(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = 'public' AND tablename = %s
                ORDER BY indexname
                """,
                (table_name,),
            )
            result["existing_indices_after"] = [str(row[0]) for row in cursor.fetchall()]
    return result


def main() -> None:
    args = _parse_args()
    project_root = REPO_ROOT.resolve()
    summary_path = args.summary_path.resolve()
    ensure_absolute(summary_path, "summary_path")

    load_project_configuration(project_root)

    dsn = pg_dsn_from_env(required=True)
    table_name = pg_table_from_env(default="article_corpus")

    summary: dict[str, Any] = {
        "project_root": str(project_root),
        "summary_path": str(summary_path),
        "table_name": table_name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "actions": {
            "doc_id_repair_requested": not args.skip_doc_id_repair,
            "collation_repair_requested": not args.skip_collation_repair,
            "pgvector_index_requested": not args.skip_pgvector_index,
            "build_hnsw": bool(args.build_hnsw),
            "local_dense_restore_requested": not args.skip_local_dense_restore,
        },
        "dense_doc_id_repair": {},
        "dense_from_pgvector_restore": {},
        "postgres_collation_repair": {},
        "pgvector_index_repair": {},
        "retrieval_health_after": {},
        "completed_at_utc": "",
    }

    if not args.skip_doc_id_repair:
        summary["dense_doc_id_repair"] = _repair_dense_doc_ids(project_root)

    if not args.skip_collation_repair:
        summary["postgres_collation_repair"] = _repair_postgres_collation(dsn)

    if not args.skip_pgvector_index:
        summary["pgvector_index_repair"] = _ensure_pgvector_index(
            dsn,
            table_name,
            build_hnsw=bool(args.build_hnsw),
        )

    dense_health_after_repairs = dense_asset_health((project_root / "data" / "indices" / "dense").resolve())
    if not args.skip_local_dense_restore and not dense_health_after_repairs.get("ready"):
        summary["dense_from_pgvector_restore"] = _restore_local_dense_from_pgvector(
            project_root,
            dsn,
            table_name,
        )

    runtime = CorpusRuntime.from_project_root(project_root)
    summary["retrieval_health_after"] = runtime.retrieval_health()
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()

    write_json(summary_path, summary)

    print("Repair summary written to:")
    print(summary_path)
    print("")
    print("Dense asset health:")
    print(summary["retrieval_health_after"]["local_dense"])
    print("")
    print("Postgres health:")
    print(summary["retrieval_health_after"]["pgvector"])


if __name__ == "__main__":
    main()
