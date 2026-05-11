"""One-shot setup for CorpusAgent2.

Detects CUDA availability, writes a starter .env if missing, and runs `uv sync`.
After running this once, use scripts/run.py to start/stop/build the stack
without having to think about CPU vs GPU per command.

Usage:
    python scripts/setup.py                # auto-detect, write .env, uv sync
    python scripts/setup.py --no-sync      # skip the uv sync step
    python scripts/setup.py --force-cpu    # override detection
    python scripts/setup.py --force-gpu    # override detection
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"
ENV_EXAMPLE = REPO_ROOT / ".env.example"


def detect_cuda() -> bool:
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
        except (subprocess.TimeoutExpired, OSError):
            pass
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _existing_env_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        keys.add(stripped.split("=", 1)[0].strip())
    return keys


def write_env_if_missing(*, use_gpu: bool) -> str:
    profile = "cuda" if use_gpu else "cpu"
    device = "cuda" if use_gpu else "cpu"
    if not ENV_FILE.exists():
        if ENV_EXAMPLE.exists():
            ENV_FILE.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"Wrote {ENV_FILE} (copied from .env.example).")
        else:
            ENV_FILE.write_text("", encoding="utf-8")
            print(f"Wrote empty {ENV_FILE}.")

    existing = ENV_FILE.read_text(encoding="utf-8")
    existing_keys = _existing_env_keys(existing)
    appendix_lines: list[str] = []
    for key, value in {
        "CORPUSAGENT2_DEVICE": device,
        "CORPUSAGENT2_DOCKER_TORCH_PROFILE": profile,
    }.items():
        if key not in existing_keys:
            appendix_lines.append(f"{key}={value}")
    if appendix_lines:
        sep = "" if existing.endswith("\n") else "\n"
        ENV_FILE.write_text(existing + sep + "\n".join(appendix_lines) + "\n", encoding="utf-8")
        print("Appended to .env: " + ", ".join(appendix_lines))
    else:
        print(".env already pins CORPUSAGENT2_DEVICE / CORPUSAGENT2_DOCKER_TORCH_PROFILE — leaving them.")
    return profile


def run_uv_sync(*, use_gpu: bool) -> None:
    uv = shutil.which("uv")
    if not uv:
        print("uv not found on PATH. Install from https://github.com/astral-sh/uv and re-run.", file=sys.stderr)
        sys.exit(1)
    args = [uv, "sync"]
    print("Running: " + " ".join(args))
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-sync", action="store_true", help="Skip the uv sync step.")
    parser.add_argument("--force-cpu", action="store_true", help="Pretend no GPU is present.")
    parser.add_argument("--force-gpu", action="store_true", help="Pretend a GPU is present.")
    args = parser.parse_args()

    if args.force_cpu and args.force_gpu:
        parser.error("--force-cpu and --force-gpu are mutually exclusive.")

    if args.force_gpu:
        use_gpu = True
        print("Using GPU profile (forced).")
    elif args.force_cpu:
        use_gpu = False
        print("Using CPU profile (forced).")
    else:
        use_gpu = detect_cuda()
        print(f"Detected {'CUDA' if use_gpu else 'CPU-only'} environment.")

    profile = write_env_if_missing(use_gpu=use_gpu)

    if not args.no_sync:
        run_uv_sync(use_gpu=use_gpu)

    print()
    print("Setup complete.")
    print(f"Device profile: {profile}")
    print()
    print("Next steps:")
    print("  python scripts/run.py up         # start the full Docker stack")
    print("  python scripts/run.py local      # run API + frontend without Docker")
    print("  python scripts/run.py status     # show what's running")


if __name__ == "__main__":
    main()
