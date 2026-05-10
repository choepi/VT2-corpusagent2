from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_prebuilt_bundle_keeps_prebuilt_bundle_responsibility() -> None:
    script = (PROJECT_ROOT / "slurm" / "run_prebuilt_bundle.sbatch").read_text(encoding="utf-8")

    assert "scripts/27_build_prebuilt_bundle.py --clean-existing \"$@\"" in script
    assert "outputs/prebuilt/" in script
    assert "coding_agent_runner.py" not in script
    assert "vllm" not in script.lower()


def test_coding_agent_has_separate_slurm_entrypoint() -> None:
    script = (PROJECT_ROOT / "slurm" / "30_run_coding_agent.sbatch").read_text(encoding="utf-8")

    assert "coding_agent_runner.py" in script
    assert "vllm" in script.lower()


def test_generated_slurm_logs_are_ignored_and_not_tracked() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/slurm/*.out" in gitignore
    assert "/slurm/*.err" in gitignore

    if not (PROJECT_ROOT / ".git").exists():
        return

    result = subprocess.run(
        ["git", "ls-files", "slurm/*.out", "slurm/*.err"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == ""


def test_gitignore_has_no_utf8_bom() -> None:
    assert not (PROJECT_ROOT / ".gitignore").read_bytes().startswith(b"\xef\xbb\xbf")
