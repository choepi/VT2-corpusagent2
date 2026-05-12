"""Backfill the agent_run_history Postgres index from on-disk run manifests.

The Run History tab in the frontend reads from the agent_run_history table,
which is populated by AgentRuntime as runs complete. Older runs that
finished before that table existed (or were created on a different host)
won't appear there. This script walks outputs/agent_runtime/<run_id>/
and upserts a row per existing run_manifest.json so the UI can list them.

Idempotent — re-running it is safe; the upsert overwrites existing rows
with the same run_id.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\35_backfill_run_history.py
    .\\.venv\\Scripts\\python.exe scripts\\35_backfill_run_history.py --root outputs/agent_runtime
    .\\.venv\\Scripts\\python.exe scripts\\35_backfill_run_history.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from corpusagent2 import run_history


def iter_run_manifests(root: Path):
    if not root.is_dir():
        return
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        yield run_dir, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT / "outputs" / "agent_runtime"),
        help="Directory containing one folder per run (default: outputs/agent_runtime).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report, but do not write to Postgres.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"Scanning {root}")

    if not args.dry_run:
        if not run_history._enabled():
            print(
                "CORPUSAGENT2_PG_DSN is not set or postgres is unreachable; "
                "agent_run_history is degraded-mode only. Aborting.",
                file=sys.stderr,
            )
            return 1
        if not run_history.ensure_table():
            print("Failed to ensure agent_run_history table exists.", file=sys.stderr)
            return 1

    seen = 0
    written = 0
    skipped: list[tuple[str, str]] = []

    for run_dir, manifest_path in iter_run_manifests(root):
        seen += 1
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            skipped.append((run_dir.name, f"json parse failed: {exc}"))
            continue
        if not manifest.get("run_id"):
            manifest["run_id"] = run_dir.name
        # Many older manifests omit artifacts_dir — synthesize one so the
        # UI's review-mode link works.
        if not manifest.get("artifacts_dir"):
            manifest["artifacts_dir"] = str(run_dir)

        if args.dry_run:
            written += 1
            continue

        ok = run_history.record_run(manifest, manifest_path=str(manifest_path))
        if ok:
            written += 1
        else:
            skipped.append((run_dir.name, "record_run returned False"))

    print(f"Seen run dirs:    {seen}")
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
