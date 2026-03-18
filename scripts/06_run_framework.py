from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.framework import run_workload_file
from corpusagent2.seed import resolve_run_mode


if __name__ == "__main__":
    mode = resolve_run_mode("full")
    workload_path = (REPO_ROOT / "config" / "framework_workload.jsonl").resolve()
    summary = run_workload_file(
        project_root=REPO_ROOT.resolve(),
        workload_path=workload_path,
        mode=mode,
    )
    print(f"Run completed: {summary['run_id']}")
    print(f"Output directory: {summary['output_dir']}")
