"""Persistent run-history index in Postgres for the UI Run History tab.

Tracks one row per CorpusAgent2 query — metadata only; the full manifest
stays on disk at outputs/agent_runtime/<run_id>/run_manifest.json. The
index lets the frontend list and click recent runs without scanning the
filesystem.

The table is created lazily on first write. If Postgres is unreachable
the writes degrade silently and listing returns an empty array — run
history is a UI nicety, not a correctness requirement of the agent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import psycopg

from .retrieval import pg_connect_kwargs, pg_dsn_from_env


_TABLE_NAME = "agent_run_history"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
    run_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at_utc TIMESTAMPTZ,
    completed_at_utc TIMESTAMPTZ,
    duration_ms BIGINT,
    node_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    artifacts_dir TEXT,
    manifest_path TEXT
);
"""

_CREATE_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_{_TABLE_NAME}_created_at
    ON {_TABLE_NAME} (created_at_utc DESC);
"""

_UPSERT_SQL = f"""
INSERT INTO {_TABLE_NAME}
    (run_id, question, status, created_at_utc, completed_at_utc,
     duration_ms, node_count, failure_count, artifacts_dir, manifest_path)
VALUES
    (%(run_id)s, %(question)s, %(status)s, %(created_at_utc)s, %(completed_at_utc)s,
     %(duration_ms)s, %(node_count)s, %(failure_count)s, %(artifacts_dir)s, %(manifest_path)s)
ON CONFLICT (run_id) DO UPDATE SET
    question = EXCLUDED.question,
    status = EXCLUDED.status,
    completed_at_utc = EXCLUDED.completed_at_utc,
    duration_ms = EXCLUDED.duration_ms,
    node_count = EXCLUDED.node_count,
    failure_count = EXCLUDED.failure_count,
    artifacts_dir = EXCLUDED.artifacts_dir,
    manifest_path = EXCLUDED.manifest_path
"""

_LIST_SQL = f"""
SELECT run_id, question, status, created_at_utc, completed_at_utc,
       duration_ms, node_count, failure_count, artifacts_dir, manifest_path
FROM {_TABLE_NAME}
ORDER BY created_at_utc DESC NULLS LAST
LIMIT %(limit)s OFFSET %(offset)s
"""


def _enabled() -> bool:
    return bool(pg_dsn_from_env(required=False))


def _connect():
    dsn = pg_dsn_from_env(required=True)
    return psycopg.connect(dsn, **pg_connect_kwargs())


def ensure_table() -> bool:
    """Create the table and index if absent. Returns True on success."""
    if not _enabled():
        return False
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
                cur.execute(_CREATE_INDEX_SQL)
            conn.commit()
        return True
    except Exception:
        return False


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def record_run(manifest: dict[str, Any], *, manifest_path: str | None = None) -> bool:
    """Insert or update a run-history row from a manifest dict. Best-effort."""
    if not _enabled():
        return False
    run_id = str(manifest.get("run_id", "")).strip()
    if not run_id:
        return False
    if not ensure_table():
        return False

    created = _parse_iso(
        manifest.get("created_at_utc")
        or manifest.get("started_at_utc")
    )
    completed = _parse_iso(
        manifest.get("completed_at_utc")
        or manifest.get("finished_at_utc")
    )
    duration_ms: int | None = None
    if created and completed:
        try:
            duration_ms = int((completed - created).total_seconds() * 1000)
        except Exception:
            duration_ms = None

    node_records = manifest.get("node_records") or []
    failures = manifest.get("failures") or []

    params = {
        "run_id": run_id,
        "question": str(manifest.get("question", ""))[:8000],
        "status": str(manifest.get("status", ""))[:64],
        "created_at_utc": created,
        "completed_at_utc": completed,
        "duration_ms": duration_ms,
        "node_count": len(node_records),
        "failure_count": len(failures),
        "artifacts_dir": str(manifest.get("artifacts_dir") or "") or None,
        "manifest_path": manifest_path,
    }

    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_UPSERT_SQL, params)
            conn.commit()
        return True
    except Exception:
        return False


def list_runs(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    if not _enabled():
        return []
    if not ensure_table():
        return []
    safe_limit = max(1, min(int(limit), 500))
    safe_offset = max(0, int(offset))
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_LIST_SQL, {"limit": safe_limit, "offset": safe_offset})
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
    except Exception:
        return []

    result: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for column, value in zip(columns, row):
            if hasattr(value, "isoformat"):
                item[column] = value.isoformat()
            else:
                item[column] = value
        result.append(item)
    return result
