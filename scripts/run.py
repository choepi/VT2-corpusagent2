"""Unified dev runner for CorpusAgent2 — same commands for CPU and GPU hosts.

The active device profile is read from CORPUSAGENT2_DOCKER_TORCH_PROFILE in
.env (set by scripts/setup.py). When the profile is `cuda`, the GPU compose
override is automatically layered on top of the base compose files.

Subcommands:
    up        Bring up the full Docker stack (postgres + opensearch + api + mcp).
    up-nodb   Bring up the stack without the postgres profile.
    build     (Re)build the api/mcp images without recreating data services.
    down      Stop and remove containers (data volumes are preserved).
    stop      Stop containers without removing them.
    logs      Tail logs for api + mcp.
    status    Show docker compose ps and detected profile.
    local     Run scripts/15_start_local_stack.py (API + static frontend, no Docker).
    api       Run scripts/12_run_agent_api.py only (no Docker, no frontend server).
    mcp       Run scripts/31_run_mcp_server.py only.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO_ROOT / "deploy"
ENV_FILE = REPO_ROOT / ".env"


def _env_file_value(key: str) -> str | None:
    if not ENV_FILE.exists():
        return None
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return None


def active_profile() -> str:
    profile = os.getenv("CORPUSAGENT2_DOCKER_TORCH_PROFILE") or _env_file_value("CORPUSAGENT2_DOCKER_TORCH_PROFILE") or "cpu"
    profile = profile.strip().lower()
    if profile not in {"cpu", "cuda"}:
        print(f"Unknown CORPUSAGENT2_DOCKER_TORCH_PROFILE={profile!r}; defaulting to cpu.", file=sys.stderr)
        profile = "cpu"
    return profile


def compose_files(profile: str) -> list[Path]:
    files = [DEPLOY_DIR / "docker-compose.yml", DEPLOY_DIR / "docker-compose.mcp.yml"]
    if profile == "cuda":
        files.append(DEPLOY_DIR / "docker-compose.mcp.gpu.yml")
    return files


def compose_command(profile: str, *extra: str) -> list[str]:
    cmd = ["docker", "compose"]
    for f in compose_files(profile):
        cmd.extend(["-f", str(f)])
    cmd.extend(extra)
    return cmd


def _require_docker() -> None:
    if not shutil.which("docker"):
        print("docker not found on PATH.", file=sys.stderr)
        sys.exit(1)


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    print("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=REPO_ROOT, env=env)


def cmd_up(args: argparse.Namespace) -> int:
    _require_docker()
    profile = active_profile()
    print(f"Active profile: {profile}")
    rc = _run(compose_command(profile, "up", "-d", "--no-recreate", "postgres", "opensearch"))
    if rc != 0:
        return rc
    return _run(compose_command(profile, "up", "-d", "--build", "--no-deps", "corpusagent2-api", "corpusagent2-mcp"))


def cmd_up_nodb(args: argparse.Namespace) -> int:
    _require_docker()
    profile = active_profile()
    print(f"Active profile: {profile} (no DB profile)")
    return _run(compose_command(profile, "up", "-d", "--build", "--no-deps", "opensearch", "corpusagent2-api", "corpusagent2-mcp"))


def cmd_build(args: argparse.Namespace) -> int:
    _require_docker()
    profile = active_profile()
    return _run(compose_command(profile, "build", "corpusagent2-api", "corpusagent2-mcp"))


def cmd_down(args: argparse.Namespace) -> int:
    _require_docker()
    profile = active_profile()
    return _run(compose_command(profile, "down"))


def cmd_stop(args: argparse.Namespace) -> int:
    _require_docker()
    profile = active_profile()
    return _run(compose_command(profile, "stop"))


def cmd_logs(args: argparse.Namespace) -> int:
    _require_docker()
    profile = active_profile()
    return _run(compose_command(profile, "logs", "-f", "--tail=200", "corpusagent2-api", "corpusagent2-mcp"))


def cmd_status(args: argparse.Namespace) -> int:
    _require_docker()
    profile = active_profile()
    print(f"Active profile: {profile}")
    return _run(compose_command(profile, "ps"))


def cmd_local(args: argparse.Namespace) -> int:
    return _run([sys.executable, str(REPO_ROOT / "scripts" / "15_start_local_stack.py")])


def cmd_api(args: argparse.Namespace) -> int:
    return _run([sys.executable, str(REPO_ROOT / "scripts" / "12_run_agent_api.py")])


def cmd_mcp(args: argparse.Namespace) -> int:
    script = REPO_ROOT / "scripts" / "31_run_mcp_server.py"
    if not script.exists():
        script = REPO_ROOT / "scripts" / "07_mcp_server.py"
    return _run([sys.executable, str(script)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (
        ("up", cmd_up),
        ("up-nodb", cmd_up_nodb),
        ("build", cmd_build),
        ("down", cmd_down),
        ("stop", cmd_stop),
        ("logs", cmd_logs),
        ("status", cmd_status),
        ("local", cmd_local),
        ("api", cmd_api),
        ("mcp", cmd_mcp),
    ):
        p = sub.add_parser(name)
        p.set_defaults(handler=fn)
    args = parser.parse_args()
    sys.exit(args.handler(args))


if __name__ == "__main__":
    main()
