"""Backfill the agent_run_history Postgres index from authoritative run tables.

The Run History tab reads from `agent_run_history`, which is populated by
AgentRuntime as runs complete. Older runs that pre-date the table (or were
run on a different host) won't appear there. This script reconstructs the
index from the authoritative per-run tables that AgentRuntime always writes:

    ca_agent_runs                  one row per run (PK run_id) — question,
                                   rewritten_question, status, created_at
    ca_agent_run_tool_calls        many rows per run — node-level events
                                   used here to count completed/failed nodes
    outputs/agent_runtime/<run_id>/run_manifest.json   (optional)
                                   used when present for completion timestamp,
                                   duration, and node_records.

Strategy:
1. Universe of run_ids = SELECT run_id FROM ca_agent_runs.
2. For each run_id, count distinct nodes and failures from
   ca_agent_run_tool_calls.
3. If a run_manifest.json exists on disk, merge in completed_at_utc /
   duration_ms / node_count from there. Otherwise leave them NULL —
   the row still shows up in the UI, just without duration.
4. Upsert into agent_run_history.

Idempotent (UPSERT on run_id).

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\35_backfill_run_history.py
    .\\.venv\\Scripts\\python.exe scripts\\35_backfill_run_history.py --dry-run
    .\\.venv\\Scripts\\python.exe scripts\\35_backfill_run_history.py --root outputs/agent_runtime
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import psycopg

from corpusagent2 import run_history
from corpusagent2.retrieval import pg_connect_kwargs, pg_dsn_from_env


_CA_RUNS_QUERY = """
SELECT run_id, question, rewritten_question, status, created_at
FROM ca_agent_runs
ORDER BY created_at ASC
"""

# Counts distinct nodes per run. Each node typically has several tool-call
# rows (running, completed, failed). DISTINCT node_id gives the true node
# count; SUM CASE WHEN status='failed' gives a failure count.
_CA_TOOL_CALL_COUNTS_QUERY = """
SELECT
    run_id,
    COUNT(DISTINCT node_id) AS node_count,
    COUNT(DISTINCT CASE WHEN status = 'failed' THEN node_id END) AS failure_count
FROM ca_agent_run_tool_calls
GROUP BY run_id
"""


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _read_disk_manifest(root: Path, run_id: str) -> dict[str, Any] | None:
    manifest_path = root / run_id / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_postgres_runs(dsn: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Return (runs, counts_by_run_id) read from ca_agent_runs + ca_agent_run_tool_calls."""
    runs: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    with psycopg.connect(dsn, **pg_connect_kwargs()) as conn:
        with conn.cursor() as cur:
            cur.execute(_CA_RUNS_QUERY)
            for row in cur.fetchall():
                run_id = str(row[0])
                runs.append(
                    {
                        "run_id": run_id,
                        "question": str(row[1] or ""),
                        "rewritten_question": str(row[2] or ""),
                        "status": str(row[3] or "unknown"),
                        "created_at": row[4],
                    }
                )
            cur.execute(_CA_TOOL_CALL_COUNTS_QUERY)
            for row in cur.fetchall():
                counts[str(row[0])] = {
                    "node_count": int(row[1] or 0),
                    "failure_count": int(row[2] or 0),
                }
    return runs, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT / "outputs" / "agent_runtime"),
        help="Directory containing run_manifest.json files (default: outputs/agent_runtime).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read sources and report, but do not write to agent_run_history.",
    )
    args = parser.parse_args()

    dsn = pg_dsn_from_env(required=False)
    if not dsn:
        print(
            "CORPUSAGENT2_PG_DSN is not set; cannot read ca_agent_runs. Aborting.",
            file=sys.stderr,
        )
        return 1

    root = Path(args.root).resolve()
    print(f"Reading ca_agent_runs from Postgres")
    print(f"Optional on-disk manifests under {root}")

    try:
        runs, counts = _read_postgres_runs(dsn)
    except Exception as exc:
        print(f"Failed to read ca_agent_runs: {exc}", file=sys.stderr)
        return 1

    print(f"Found {len(runs)} runs in ca_agent_runs")
    print(f"Found {len(counts)} runs with tool-call rows")

    if not args.dry_run and not run_history.ensure_table():
        print("Failed to ensure agent_run_history table exists.", file=sys.stderr)
        return 1

    written = 0
    skipped: list[tuple[str, str]] = []
    on_disk_count = 0

    for run in runs:
        run_id = run["run_id"]
        question = run["question"] or run["rewritten_question"] or "(no question recorded)"
        status = run["status"] or "unknown"
        created_at = run["created_at"]
        node_count = counts.get(run_id, {}).get("node_count", 0)
        failure_count = counts.get(run_id, {}).get("failure_count", 0)

        completed_at = None
        duration_ms = None
        artifacts_dir = None
        manifest_path = None

        disk = _read_disk_manifest(root, run_id)
        if disk is not None:
            on_disk_count += 1
            completed_at = _parse_iso(
                disk.get("completed_at_utc") or disk.get("finished_at_utc")
            )
            # Disk often has a more complete created_at_utc than ca_agent_runs.
            disk_created = _parse_iso(
                disk.get("created_at_utc") or disk.get("started_at_utc")
            )
            if disk_created and not created_at:
                created_at = disk_created
            if completed_at and (created_at or disk_created):
                anchor = created_at or disk_created
                try:
                    duration_ms = int((completed_at - anchor).total_seconds() * 1000)
                except Exception:
                    duration_ms = None
            disk_nodes = disk.get("node_records") or []
            if isinstance(disk_nodes, list) and disk_nodes:
                node_count = max(node_count, len(disk_nodes))
            disk_failures = disk.get("failures") or []
            if isinstance(disk_failures, list) and disk_failures:
                failure_count = max(failure_count, len(disk_failures))
            artifacts_dir = str(disk.get("artifacts_dir") or (root / run_id))
            manifest_path = str(root / run_id / "run_manifest.json")
        else:
            # No on-disk manifest. Still synthesize an artifacts_dir so the
            # UI's review-mode link has somewhere to point.
            artifacts_dir = str(root / run_id)

        # Build a manifest-shaped dict that run_history.record_run can consume.
        synthetic = {
            "run_id": run_id,
            "question": question,
            "status": status,
            "created_at_utc": created_at.isoformat() if isinstance(created_at, datetime) else (created_at or ""),
            "completed_at_utc": completed_at.isoformat() if isinstance(completed_at, datetime) else "",
            "artifacts_dir": artifacts_dir,
            "node_records": [None] * node_count,
            "failures": [None] * failure_count,
        }

        if args.dry_run:
            written += 1
            continue

        ok = run_history.record_run(synthetic, manifest_path=manifest_path)
        if ok:
            written += 1
        else:
            skipped.append((run_id, "record_run returned False"))

    print(f"Runs with on-disk manifest:   {on_disk_count}")
    print(f"Runs Postgres-only (no disk): {len(runs) - on_disk_count}")
    print(f"{'Would write' if args.dry_run else 'Wrote'} rows: {written}")
    if skipped:
        print(f"Skipped: {len(skipped)}")
        for run_id, reason in skipped[:20]:
            print(f"  {run_id}: {reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
