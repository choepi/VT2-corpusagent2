from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.deploy_resources import compute_docker_resource_plan, data_service_resource_updates, detect_host_hardware


def _upsert_dotenv(path: Path, updates: dict[str, str]) -> None:
    def render(key: str, value: str) -> str:
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        if any(char.isspace() for char in escaped):
            return f'{key}="{escaped}"'
        return f"{key}={escaped}"

    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    output: list[str] = []
    seen: set[str] = set()
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in updates:
            output.append(render(key, updates[key]))
            seen.add(key)
        else:
            output.append(line)
    for key, value in updates.items():
        if key not in seen:
            output.append(render(key, value))
    path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Docker CPU/RAM/GPU resource limits for CorpusAgent2 into .env.")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--gpu", choices=["auto", "on", "off"], default="auto")
    parser.add_argument(
        "--apply-data-services",
        action="store_true",
        help="Apply Postgres/OpenSearch CPU/RAM limits in-place with docker update. This does not recreate containers or volumes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the computed plan as JSON.")
    return parser.parse_args()


def _apply_data_service_resources(plan) -> None:
    for container_name, limits in data_service_resource_updates(plan).items():
        inspect = subprocess.run(
            ["docker", "inspect", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if inspect.returncode != 0:
            print(f"[skip] {container_name} is not present; no in-place resource update applied.")
            continue
        command = [
            "docker",
            "update",
            "--cpus",
            limits["cpus"],
            "--memory",
            limits["memory"],
            container_name,
        ]
        print(f"[run] {' '.join(command)}")
        subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    plan = compute_docker_resource_plan(detect_host_hardware(), gpu_mode=args.gpu)
    payload = {
        "host": {
            "logical_cpus": plan.host.logical_cpus,
            "total_memory_gib": round(plan.host.total_memory_bytes / (1024**3), 2),
            "gpu_available": plan.host.gpu_available,
        },
        "cpu_fraction": plan.cpu_fraction,
        "memory_fraction": plan.memory_fraction,
        "use_gpu": plan.use_gpu,
        "env": plan.env,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            "Detected "
            f"{payload['host']['logical_cpus']} logical CPUs, "
            f"{payload['host']['total_memory_gib']} GiB RAM, "
            f"GPU available={payload['host']['gpu_available']}."
        )
        print(f"Allocating {plan.cpu_fraction:.0%} CPU and {plan.memory_fraction:.0%} RAM to the Docker stack.")
        for key, value in plan.env.items():
            print(f"{key}={value}")
    if not args.dry_run:
        _upsert_dotenv(args.env_file, plan.env)
        print(f"Wrote Docker resource limits to {args.env_file}")
        if args.apply_data_services:
            _apply_data_service_resources(plan)


if __name__ == "__main__":
    main()
