from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import closing
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from typing import Any, Callable, Protocol

from .agent_runtime import AgentRuntime, AgentRuntimeConfig, TERMINAL_RUN_STATUSES
from .retrieval import pg_connect_kwargs, pg_dsn_from_env


MCP_JOB_TERMINAL_STATUSES = TERMINAL_RUN_STATUSES | {"cancelled"}
MCP_JOB_ACTIVE_STATUSES = {"queued", "on_hold", "running", "aborting", "cancel_requested"}
DEFAULT_MCP_PORT = 8765


def _is_terminal_status(status: str) -> bool:
    return status in MCP_JOB_TERMINAL_STATUSES


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _json_loads(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return fallback


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


def _safe_identifier(value: str) -> str:
    candidate = str(value).strip()
    if not candidate or not candidate.replace("_", "").isalnum():
        raise ValueError(f"Invalid SQL identifier: {candidate}")
    return candidate


@dataclass(slots=True)
class MCPJobRecord:
    job_id: str
    run_id: str
    owner: str
    question: str
    force_answer: bool = False
    no_cache: bool = False
    clarification_history: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    hold_reason: str = ""
    worker_id: str = ""
    result_summary: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at_utc: str = ""
    updated_at_utc: str = ""
    started_at_utc: str = ""
    finished_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "owner": self.owner,
            "question": self.question,
            "force_answer": self.force_answer,
            "no_cache": self.no_cache,
            "clarification_history": list(self.clarification_history),
            "metadata": dict(self.metadata),
            "status": self.status,
            "hold_reason": self.hold_reason,
            "worker_id": self.worker_id,
            "result_summary": dict(self.result_summary),
            "error": self.error,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
        }


class MCPJobStore(Protocol):
    def ensure_schema(self) -> None: ...
    def create_job(self, job: MCPJobRecord) -> MCPJobRecord: ...
    def get_job(self, job_id: str) -> MCPJobRecord | None: ...
    def list_jobs(self, *, owner: str = "", status: str = "", limit: int = 50) -> list[MCPJobRecord]: ...
    def claim_next_job(self, *, worker_id: str, stale_running_after_s: int) -> MCPJobRecord | None: ...
    def set_job_status(
        self,
        job_id: str,
        status: str,
        *,
        hold_reason: str = "",
        worker_id: str | None = None,
        error: str = "",
        result_summary: dict[str, Any] | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> MCPJobRecord | None: ...
    def update_progress(self, job_id: str, progress: dict[str, Any]) -> MCPJobRecord | None: ...
    def cancel_job(self, job_id: str) -> MCPJobRecord | None: ...
    def queue_position(self, job_id: str) -> int | None: ...
    def active_count(self) -> int: ...


def _record_from_mapping(row: dict[str, Any]) -> MCPJobRecord:
    return MCPJobRecord(
        job_id=str(row.get("job_id", "")),
        run_id=str(row.get("run_id", "")),
        owner=str(row.get("owner", "")),
        question=str(row.get("question", "")),
        force_answer=bool(row.get("force_answer", False)),
        no_cache=bool(row.get("no_cache", False)),
        clarification_history=list(_json_loads(row.get("clarification_history"), [])),
        metadata=dict(_json_loads(row.get("metadata"), {})),
        status=str(row.get("status", "queued")),
        hold_reason=str(row.get("hold_reason", "")),
        worker_id=str(row.get("worker_id", "")),
        result_summary=dict(_json_loads(row.get("result_summary"), {})),
        error=str(row.get("error", "")),
        created_at_utc=str(row.get("created_at_utc") or row.get("created_at") or ""),
        updated_at_utc=str(row.get("updated_at_utc") or row.get("updated_at") or ""),
        started_at_utc=str(row.get("started_at_utc") or row.get("started_at") or ""),
        finished_at_utc=str(row.get("finished_at_utc") or row.get("finished_at") or ""),
    )


class SQLiteMCPJobStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._schema_ready = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ca_mcp_jobs (
                  job_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  owner TEXT NOT NULL DEFAULT '',
                  question TEXT NOT NULL,
                  force_answer INTEGER NOT NULL DEFAULT 0,
                  no_cache INTEGER NOT NULL DEFAULT 0,
                  clarification_history TEXT NOT NULL DEFAULT '[]',
                  metadata TEXT NOT NULL DEFAULT '{}',
                  status TEXT NOT NULL,
                  hold_reason TEXT NOT NULL DEFAULT '',
                  worker_id TEXT NOT NULL DEFAULT '',
                  result_summary TEXT NOT NULL DEFAULT '{}',
                  error TEXT NOT NULL DEFAULT '',
                  created_at_utc TEXT NOT NULL,
                  updated_at_utc TEXT NOT NULL,
                  started_at_utc TEXT NOT NULL DEFAULT '',
                  finished_at_utc TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ca_mcp_jobs_status_created ON ca_mcp_jobs (status, created_at_utc)")
            conn.commit()
        self._schema_ready = True

    def create_job(self, job: MCPJobRecord) -> MCPJobRecord:
        self.ensure_schema()
        now = _utc_now()
        job.created_at_utc = job.created_at_utc or now
        job.updated_at_utc = now
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO ca_mcp_jobs (
                  job_id, run_id, owner, question, force_answer, no_cache,
                  clarification_history, metadata, status, hold_reason, worker_id,
                  result_summary, error, created_at_utc, updated_at_utc,
                  started_at_utc, finished_at_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.run_id,
                    job.owner,
                    job.question,
                    int(job.force_answer),
                    int(job.no_cache),
                    _json_dumps(job.clarification_history),
                    _json_dumps(job.metadata),
                    job.status,
                    job.hold_reason,
                    job.worker_id,
                    _json_dumps(job.result_summary),
                    job.error,
                    job.created_at_utc,
                    job.updated_at_utc,
                    job.started_at_utc,
                    job.finished_at_utc,
                ),
            )
            conn.commit()
        return job

    def _fetch_one(self, query: str, params: tuple[Any, ...]) -> MCPJobRecord | None:
        self.ensure_schema()
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(query, params).fetchone()
        return _record_from_mapping(dict(row)) if row else None

    def get_job(self, job_id: str) -> MCPJobRecord | None:
        return self._fetch_one("SELECT * FROM ca_mcp_jobs WHERE job_id = ?", (job_id,))

    def list_jobs(self, *, owner: str = "", status: str = "", limit: int = 50) -> list[MCPJobRecord]:
        self.ensure_schema()
        clauses: list[str] = []
        params: list[Any] = []
        if owner:
            clauses.append("owner = ?")
            params.append(owner)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 500)))
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT * FROM ca_mcp_jobs {where} ORDER BY created_at_utc DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [_record_from_mapping(dict(row)) for row in rows]

    def claim_next_job(self, *, worker_id: str, stale_running_after_s: int) -> MCPJobRecord | None:
        self.ensure_schema()
        now = _utc_now()
        stale_cutoff = datetime.fromtimestamp(time.time() - stale_running_after_s, UTC).isoformat()
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE ca_mcp_jobs
                SET status = 'queued',
                    hold_reason = 'requeued_after_worker_heartbeat_timeout',
                    worker_id = '',
                    updated_at_utc = ?
                WHERE status = 'running' AND updated_at_utc < ?
                """,
                (now, stale_cutoff),
            )
            row = conn.execute(
                """
                SELECT * FROM ca_mcp_jobs
                WHERE status IN ('queued', 'on_hold')
                ORDER BY created_at_utc ASC
                LIMIT 1
                """
            ).fetchone()
            if not row:
                conn.commit()
                return None
            conn.execute(
                """
                UPDATE ca_mcp_jobs
                SET status = 'running',
                    hold_reason = '',
                    worker_id = ?,
                    started_at_utc = CASE WHEN started_at_utc = '' THEN ? ELSE started_at_utc END,
                    updated_at_utc = ?
                WHERE job_id = ?
                """,
                (worker_id, now, now, row["job_id"]),
            )
            conn.commit()
        return self.get_job(str(row["job_id"]))

    def set_job_status(
        self,
        job_id: str,
        status: str,
        *,
        hold_reason: str = "",
        worker_id: str | None = None,
        error: str = "",
        result_summary: dict[str, Any] | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> MCPJobRecord | None:
        self.ensure_schema()
        current = self.get_job(job_id)
        if current is None:
            return None
        if _is_terminal_status(current.status) and not _is_terminal_status(status):
            return current
        now = _utc_now()
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE ca_mcp_jobs
                SET status = ?,
                    hold_reason = ?,
                    worker_id = COALESCE(?, worker_id),
                    error = ?,
                    result_summary = ?,
                    started_at_utc = CASE WHEN ? THEN CASE WHEN started_at_utc = '' THEN ? ELSE started_at_utc END ELSE started_at_utc END,
                    finished_at_utc = CASE WHEN ? THEN ? ELSE finished_at_utc END,
                    updated_at_utc = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    hold_reason,
                    worker_id,
                    error,
                    _json_dumps(result_summary if result_summary is not None else current.result_summary),
                    int(started),
                    now,
                    int(finished),
                    now,
                    now,
                    job_id,
                ),
            )
            conn.commit()
        return self.get_job(job_id)

    def update_progress(self, job_id: str, progress: dict[str, Any]) -> MCPJobRecord | None:
        current = self.get_job(job_id)
        if current is None:
            return None
        summary = dict(current.result_summary)
        summary["live_status"] = progress
        return self.set_job_status(
            job_id,
            current.status,
            hold_reason=current.hold_reason,
            worker_id=current.worker_id,
            error=current.error,
            result_summary=summary,
        )

    def cancel_job(self, job_id: str) -> MCPJobRecord | None:
        current = self.get_job(job_id)
        if current is None:
            return None
        if current.status in MCP_JOB_TERMINAL_STATUSES:
            return current
        if current.status in {"queued", "on_hold"}:
            return self.set_job_status(job_id, "cancelled", hold_reason="", finished=True)
        return self.set_job_status(job_id, "cancel_requested", hold_reason="abort_requested")

    def queue_position(self, job_id: str) -> int | None:
        job = self.get_job(job_id)
        if job is None or job.status not in {"queued", "on_hold"}:
            return None
        self.ensure_schema()
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS position
                FROM ca_mcp_jobs
                WHERE status IN ('queued', 'on_hold') AND created_at_utc <= ?
                """,
                (job.created_at_utc,),
            ).fetchone()
        return int(row["position"]) if row else None

    def active_count(self) -> int:
        self.ensure_schema()
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM ca_mcp_jobs WHERE status = 'running'").fetchone()
        return int(row["count"]) if row else 0


class PostgresMCPJobStore:
    def __init__(self, *, dsn: str, table_name: str = "ca_mcp_jobs") -> None:
        self.dsn = dsn
        self.table_name = _safe_identifier(table_name)
        self._schema_ready = False

    def _connect(self):
        from psycopg import connect

        return connect(self.dsn, **pg_connect_kwargs())

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        table = self.table_name
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                      job_id TEXT PRIMARY KEY,
                      run_id TEXT NOT NULL,
                      owner TEXT NOT NULL DEFAULT '',
                      question TEXT NOT NULL,
                      force_answer BOOLEAN NOT NULL DEFAULT FALSE,
                      no_cache BOOLEAN NOT NULL DEFAULT FALSE,
                      clarification_history JSONB NOT NULL DEFAULT '[]'::jsonb,
                      metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                      status TEXT NOT NULL,
                      hold_reason TEXT NOT NULL DEFAULT '',
                      worker_id TEXT NOT NULL DEFAULT '',
                      result_summary JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                      error TEXT NOT NULL DEFAULT '',
                      created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                      updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                      started_at_utc TIMESTAMPTZ,
                      finished_at_utc TIMESTAMPTZ
                    )
                    """
                )
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_status_created ON {table} (status, created_at_utc)"
                )
            conn.commit()
        self._schema_ready = True

    def _row_to_record(self, row: Any, description: Any) -> MCPJobRecord:
        payload = {
            str(getattr(desc, "name", desc[0] if desc else "")): value
            for desc, value in zip(description, row)
        }
        for key in ("created_at_utc", "updated_at_utc", "started_at_utc", "finished_at_utc"):
            value = payload.get(key)
            if isinstance(value, datetime):
                payload[key] = value.isoformat()
            elif value is None:
                payload[key] = ""
        return _record_from_mapping(payload)

    def _fetch_one(self, query: str, params: tuple[Any, ...]) -> MCPJobRecord | None:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_record(row, cursor.description)

    def create_job(self, job: MCPJobRecord) -> MCPJobRecord:
        self.ensure_schema()
        from psycopg.types.json import Json

        table = self.table_name
        now = _utc_now()
        job.created_at_utc = job.created_at_utc or now
        job.updated_at_utc = now
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {table} (
                      job_id, run_id, owner, question, force_answer, no_cache,
                      clarification_history, metadata, status, hold_reason, worker_id,
                      result_summary, error, created_at_utc, updated_at_utc,
                      started_at_utc, finished_at_utc
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULLIF(%s, '')::timestamptz, NULLIF(%s, '')::timestamptz)
                    """,
                    (
                        job.job_id,
                        job.run_id,
                        job.owner,
                        job.question,
                        job.force_answer,
                        job.no_cache,
                        Json(job.clarification_history),
                        Json(job.metadata),
                        job.status,
                        job.hold_reason,
                        job.worker_id,
                        Json(job.result_summary),
                        job.error,
                        job.created_at_utc,
                        job.updated_at_utc,
                        job.started_at_utc,
                        job.finished_at_utc,
                    ),
                )
            conn.commit()
        return job

    def get_job(self, job_id: str) -> MCPJobRecord | None:
        return self._fetch_one(f"SELECT * FROM {self.table_name} WHERE job_id = %s", (job_id,))

    def list_jobs(self, *, owner: str = "", status: str = "", limit: int = 50) -> list[MCPJobRecord]:
        self.ensure_schema()
        clauses: list[str] = []
        params: list[Any] = []
        if owner:
            clauses.append("owner = %s")
            params.append(owner)
        if status:
            clauses.append("status = %s")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 500)))
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {self.table_name} {where} ORDER BY created_at_utc DESC LIMIT %s",
                    tuple(params),
                )
                rows = cursor.fetchall()
                return [self._row_to_record(row, cursor.description) for row in rows]

    def claim_next_job(self, *, worker_id: str, stale_running_after_s: int) -> MCPJobRecord | None:
        self.ensure_schema()
        table = self.table_name
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'queued',
                        hold_reason = 'requeued_after_worker_heartbeat_timeout',
                        worker_id = '',
                        updated_at_utc = NOW()
                    WHERE status = 'running'
                      AND updated_at_utc < NOW() - (%s * INTERVAL '1 second')
                    """,
                    (stale_running_after_s,),
                )
                cursor.execute(
                    f"""
                    WITH next_job AS (
                      SELECT job_id
                      FROM {table}
                      WHERE status IN ('queued', 'on_hold')
                      ORDER BY created_at_utc ASC
                      FOR UPDATE SKIP LOCKED
                      LIMIT 1
                    )
                    UPDATE {table}
                    SET status = 'running',
                        hold_reason = '',
                        worker_id = %s,
                        started_at_utc = COALESCE(started_at_utc, NOW()),
                        updated_at_utc = NOW()
                    WHERE job_id = (SELECT job_id FROM next_job)
                    RETURNING *
                    """,
                    (worker_id,),
                )
                row = cursor.fetchone()
                conn.commit()
                if row is None:
                    return None
                return self._row_to_record(row, cursor.description)

    def set_job_status(
        self,
        job_id: str,
        status: str,
        *,
        hold_reason: str = "",
        worker_id: str | None = None,
        error: str = "",
        result_summary: dict[str, Any] | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> MCPJobRecord | None:
        self.ensure_schema()
        from psycopg.types.json import Json

        current = self.get_job(job_id)
        if current is None:
            return None
        if _is_terminal_status(current.status) and not _is_terminal_status(status):
            return current
        table = self.table_name
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {table}
                    SET status = %s,
                        hold_reason = %s,
                        worker_id = COALESCE(%s, worker_id),
                        error = %s,
                        result_summary = %s,
                        started_at_utc = CASE WHEN %s THEN COALESCE(started_at_utc, NOW()) ELSE started_at_utc END,
                        finished_at_utc = CASE WHEN %s THEN NOW() ELSE finished_at_utc END,
                        updated_at_utc = NOW()
                    WHERE job_id = %s
                    RETURNING *
                    """,
                    (
                        status,
                        hold_reason,
                        worker_id,
                        error,
                        Json(result_summary if result_summary is not None else current.result_summary),
                        started,
                        finished,
                        job_id,
                    ),
                )
                row = cursor.fetchone()
                conn.commit()
                if row is None:
                    return None
                return self._row_to_record(row, cursor.description)

    def update_progress(self, job_id: str, progress: dict[str, Any]) -> MCPJobRecord | None:
        current = self.get_job(job_id)
        if current is None:
            return None
        summary = dict(current.result_summary)
        summary["live_status"] = progress
        return self.set_job_status(
            job_id,
            current.status,
            hold_reason=current.hold_reason,
            worker_id=current.worker_id,
            error=current.error,
            result_summary=summary,
        )

    def cancel_job(self, job_id: str) -> MCPJobRecord | None:
        current = self.get_job(job_id)
        if current is None:
            return None
        if current.status in MCP_JOB_TERMINAL_STATUSES:
            return current
        if current.status in {"queued", "on_hold"}:
            return self.set_job_status(job_id, "cancelled", finished=True)
        return self.set_job_status(job_id, "cancel_requested", hold_reason="abort_requested")

    def queue_position(self, job_id: str) -> int | None:
        job = self.get_job(job_id)
        if job is None or job.status not in {"queued", "on_hold"}:
            return None
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {self.table_name}
                    WHERE status IN ('queued', 'on_hold') AND created_at_utc <= %s::timestamptz
                    """,
                    (job.created_at_utc,),
                )
                row = cursor.fetchone()
        return int(row[0]) if row else None

    def active_count(self) -> int:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE status = 'running'")
                row = cursor.fetchone()
        return int(row[0]) if row else 0


def build_mcp_job_store(project_root: Path) -> MCPJobStore:
    store_kind = os.getenv("CORPUSAGENT2_MCP_JOB_STORE", "").strip().lower()
    dsn = os.getenv("CORPUSAGENT2_MCP_PG_DSN", "").strip() or pg_dsn_from_env(required=False)
    if store_kind in {"", "postgres", "pg"} and dsn:
        table_name = os.getenv("CORPUSAGENT2_MCP_JOB_TABLE", "ca_mcp_jobs").strip() or "ca_mcp_jobs"
        return PostgresMCPJobStore(dsn=dsn, table_name=table_name)
    sqlite_path = Path(
        os.getenv(
            "CORPUSAGENT2_MCP_SQLITE_PATH",
            str(project_root / "outputs" / "mcp_jobs" / "jobs.sqlite"),
        )
    )
    return SQLiteMCPJobStore(sqlite_path)


class MCPJobManager:
    def __init__(
        self,
        *,
        project_root: Path,
        store: MCPJobStore | None = None,
        runtime: AgentRuntime | None = None,
        runtime_factory: Callable[[], AgentRuntime] | None = None,
        max_concurrent_jobs: int | None = None,
        poll_seconds: float | None = None,
        heartbeat_seconds: float | None = None,
        stale_running_after_s: int | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.store = store or build_mcp_job_store(self.project_root)
        self.runtime = runtime
        self.runtime_factory = runtime_factory or (
            lambda: AgentRuntime(config=AgentRuntimeConfig.from_project_root(self.project_root))
        )
        self.max_concurrent_jobs = max_concurrent_jobs or _env_int(
            "CORPUSAGENT2_MCP_MAX_CONCURRENT_JOBS",
            1,
            minimum=1,
        )
        self.poll_seconds = poll_seconds if poll_seconds is not None else _env_float(
            "CORPUSAGENT2_MCP_WORKER_POLL_SECONDS",
            2.0,
            minimum=0.1,
        )
        self.heartbeat_seconds = heartbeat_seconds if heartbeat_seconds is not None else _env_float(
            "CORPUSAGENT2_MCP_HEARTBEAT_SECONDS",
            5.0,
            minimum=0.5,
        )
        self.stale_running_after_s = stale_running_after_s or _env_int(
            "CORPUSAGENT2_MCP_STALE_RUNNING_SECONDS",
            900,
            minimum=30,
        )
        self.worker_id = os.getenv("CORPUSAGENT2_MCP_WORKER_ID", "").strip() or f"mcp-{uuid.uuid4().hex[:12]}"
        self._started = False
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._active_runs: dict[str, str] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._started:
            return
        self.store.ensure_schema()
        self._stop_event.clear()
        self._started = True
        for index in range(self.max_concurrent_jobs):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(index,),
                daemon=True,
                name=f"corpusagent2-mcp-worker-{index}",
            )
            self._threads.append(thread)
            thread.start()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        for thread in list(self._threads):
            thread.join(timeout=timeout_s)
        self._threads.clear()
        self._started = False

    def submit_question(
        self,
        question: str,
        *,
        owner: str = "",
        force_answer: bool = False,
        no_cache: bool = False,
        clarification_history: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.start()
        active_count = self.store.active_count()
        status = "on_hold" if active_count >= self.max_concurrent_jobs else "queued"
        hold_reason = "waiting_for_worker_capacity" if status == "on_hold" else ""
        job = MCPJobRecord(
            job_id=f"mcpjob_{uuid.uuid4().hex[:16]}",
            run_id=f"agent_{uuid.uuid4().hex[:12]}",
            owner=str(owner or ""),
            question=str(question).strip(),
            force_answer=bool(force_answer),
            no_cache=bool(no_cache),
            clarification_history=list(clarification_history or []),
            metadata=dict(metadata or {}),
            status=status,
            hold_reason=hold_reason,
        )
        if not job.question:
            raise ValueError("question must not be empty")
        created = self.store.create_job(job)
        return self._status_payload(created)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if job is None:
            raise FileNotFoundError(f"MCP job not found: {job_id}")
        return self._status_payload(job)

    def list_jobs(self, *, owner: str = "", status: str = "", limit: int = 50) -> dict[str, Any]:
        jobs = [self._status_payload(job) for job in self.store.list_jobs(owner=owner, status=status, limit=limit)]
        return {
            "jobs": jobs,
            "count": len(jobs),
            "capacity": self._capacity_payload(),
            "worker_id": self.worker_id,
        }

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = self.store.cancel_job(job_id)
        if job is None:
            raise FileNotFoundError(f"MCP job not found: {job_id}")
        with self._lock:
            run_id = self._active_runs.get(job_id)
        if run_id and self.runtime is not None:
            try:
                self.runtime.abort_run(run_id)
            except Exception:
                pass
        return self._status_payload(self.store.get_job(job_id) or job)

    def get_job_result(self, job_id: str) -> dict[str, Any]:
        status = self.get_job_status(job_id)
        run_id = str(status.get("run_id", ""))
        if not run_id:
            return status
        runtime = self._runtime()
        try:
            manifest = runtime.get_run(run_id)
        except Exception:
            manifest = {}
        return {
            **status,
            "manifest": manifest,
        }

    def runtime_info(self) -> dict[str, Any]:
        return {
            "mcp": {
                "worker_id": self.worker_id,
                "capacity": self._capacity_payload(),
                "job_store": type(self.store).__name__,
            },
            "agent_runtime": self._runtime().runtime_info(),
        }

    def _runtime(self) -> AgentRuntime:
        if self.runtime is None:
            self.runtime = self.runtime_factory()
        return self.runtime

    def _worker_loop(self, worker_index: int) -> None:
        worker_id = f"{self.worker_id}-{worker_index}"
        while not self._stop_event.is_set():
            job = self.store.claim_next_job(
                worker_id=worker_id,
                stale_running_after_s=self.stale_running_after_s,
            )
            if job is None:
                self._stop_event.wait(self.poll_seconds)
                continue
            self._execute_job(job, worker_id=worker_id)

    def _execute_job(self, job: MCPJobRecord, *, worker_id: str) -> None:
        runtime = self._runtime()
        with self._lock:
            self._active_runs[job.job_id] = job.run_id
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(job.job_id, job.run_id, heartbeat_stop),
            daemon=True,
            name=f"corpusagent2-mcp-heartbeat-{job.job_id}",
        )
        heartbeat.start()
        try:
            latest = self.store.get_job(job.job_id)
            if latest is not None and latest.status == "cancel_requested":
                self.store.set_job_status(job.job_id, "cancelled", finished=True)
                return
            manifest = runtime.handle_query(
                job.question,
                force_answer=job.force_answer,
                no_cache=job.no_cache,
                clarification_history=job.clarification_history,
                run_id=job.run_id,
            )
            summary = self._manifest_summary(manifest)
            self.store.set_job_status(
                job.job_id,
                manifest.status,
                worker_id=worker_id,
                result_summary=summary,
                finished=manifest.status in MCP_JOB_TERMINAL_STATUSES,
            )
        except Exception as exc:
            self.store.set_job_status(
                job.job_id,
                "failed",
                worker_id=worker_id,
                error=str(exc),
                result_summary={"error_type": type(exc).__name__},
                finished=True,
            )
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=2.0)
            with self._lock:
                self._active_runs.pop(job.job_id, None)

    def _heartbeat_loop(self, job_id: str, run_id: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(self.heartbeat_seconds):
            try:
                progress = self._runtime().get_run_status(run_id)
                self.store.update_progress(job_id, progress)
            except Exception:
                current = self.store.get_job(job_id)
                if current is not None and current.status == "running":
                    self.store.set_job_status(
                        job_id,
                        "running",
                        hold_reason=current.hold_reason,
                        worker_id=current.worker_id,
                        error=current.error,
                        result_summary=current.result_summary,
                    )

    def _capacity_payload(self) -> dict[str, Any]:
        active_count = self.store.active_count()
        return {
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "running_jobs": active_count,
            "available_slots": max(0, self.max_concurrent_jobs - active_count),
        }

    def _status_payload(self, job: MCPJobRecord) -> dict[str, Any]:
        current = self.store.get_job(job.job_id) or job
        with self._lock:
            is_local_active = current.job_id in self._active_runs
        live_status: dict[str, Any] = {}
        if is_local_active:
            try:
                live_status = self._runtime().get_run_status(current.run_id)
            except Exception:
                live_status = dict(current.result_summary.get("live_status", {}))
        else:
            live_status = dict(current.result_summary.get("live_status", {}))
        payload = current.to_dict()
        queue_position = self.store.queue_position(current.job_id)
        if queue_position is not None:
            payload["queue_position"] = queue_position
        payload["capacity"] = self._capacity_payload()
        payload["live_status"] = live_status
        if current.status in {"queued", "on_hold"} and payload["capacity"]["available_slots"] <= 0:
            payload["status"] = "on_hold"
            payload["hold_reason"] = payload["hold_reason"] or "waiting_for_worker_capacity"
        return payload

    def _manifest_summary(self, manifest: Any) -> dict[str, Any]:
        manifest_path = Path(str(getattr(manifest, "artifacts_dir", ""))) / "run_manifest.json"
        final_answer = getattr(manifest, "final_answer", None)
        answer_text = str(getattr(final_answer, "answer_text", "") or "")
        return {
            "run_id": str(getattr(manifest, "run_id", "")),
            "status": str(getattr(manifest, "status", "")),
            "manifest_path": str(manifest_path),
            "artifacts_dir": str(getattr(manifest, "artifacts_dir", "")),
            "answer_preview": answer_text[:2000],
            "evidence_rows": len(getattr(manifest, "evidence_table", []) or []),
            "selected_docs": len(getattr(manifest, "selected_docs", []) or []),
            "node_records": len(getattr(manifest, "node_records", []) or []),
        }
