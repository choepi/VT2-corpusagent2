from __future__ import annotations

import argparse
import os
from pathlib import Path
import pwd
import shlex
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.app_config import load_project_configuration
from corpusagent2.deploy_resources import compute_docker_resource_plan, detect_host_hardware


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _get_config_value(name: str, default: str = "") -> str:
    raw = os.getenv(name, "").strip()
    if raw:
        return raw
    return _load_dotenv(REPO_ROOT / ".env").get(name, default).strip()


def _run(command: list[str]) -> None:
    print(f"[run] {shlex.join(command)}")
    subprocess.run(command, check=True)


def _sudo_prefix() -> list[str]:
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None or geteuid() == 0:
        return []
    return ["sudo"] if shutil.which("sudo") else []


def _service_user_home(service_user: str) -> Path:
    return Path(pwd.getpwnam(service_user).pw_dir)


def _upsert_dotenv(path: Path, updates: dict[str, str]) -> None:
    def render(key: str, value: str) -> str:
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        if any(char.isspace() for char in escaped):
            return f'{key}="{escaped}"'
        return f"{key}={escaped}"

    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen: set[str] = set()
    output: list[str] = []
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


def _compose_exec_fragment(*, docker_bin: str, use_gpu: bool) -> str:
    files = [
        REPO_ROOT / "deploy" / "docker-compose.yml",
        REPO_ROOT / "deploy" / "docker-compose.mcp.yml",
    ]
    if use_gpu:
        files.append(REPO_ROOT / "deploy" / "docker-compose.mcp.gpu.yml")
    rendered_files = " ".join(f"-f {path}" for path in files)
    return f"{docker_bin} compose {rendered_files}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install reboot-safe systemd services for the Dockerized CorpusAgent2 stack and a named Cloudflare tunnel."
    )
    parser.add_argument("--service-user", default=_get_config_value("USER", os.getenv("USER", "")) or "root")
    parser.add_argument("--stack-service-name", default="corpusagent2-stack")
    parser.add_argument("--api-service-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--tunnel-service-name", default="corpusagent2-cloudflared")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8001")
    parser.add_argument("--gpu", choices=["auto", "on", "off"], default="auto", help="GPU compose selection for the Docker stack service.")
    parser.add_argument("--skip-tunnel", action="store_true", help="Install only the Docker stack service.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dotenv_path = REPO_ROOT / ".env"
    load_project_configuration(REPO_ROOT)

    service_user = args.service_user
    service_home = _service_user_home(service_user)
    cloudflared_dir = service_home / ".cloudflared"
    cloudflared_dir.mkdir(parents=True, exist_ok=True)

    tunnel_id = _get_config_value("CORPUSAGENT2_CLOUDFLARED_TUNNEL_ID")
    tunnel_name = _get_config_value("CORPUSAGENT2_CLOUDFLARED_TUNNEL_NAME")
    tunnel_hostname = _get_config_value("CORPUSAGENT2_CLOUDFLARED_HOSTNAME")
    credentials_file = _get_config_value(
        "CORPUSAGENT2_CLOUDFLARED_CREDENTIALS_FILE",
        str(cloudflared_dir / f"{tunnel_id}.json") if tunnel_id else "",
    )

    if not args.skip_tunnel and (not tunnel_id or not tunnel_name or not tunnel_hostname):
        raise RuntimeError(
            "Named tunnel setup requires CORPUSAGENT2_CLOUDFLARED_TUNNEL_ID, "
            "CORPUSAGENT2_CLOUDFLARED_TUNNEL_NAME, and CORPUSAGENT2_CLOUDFLARED_HOSTNAME in .env or the process environment."
        )

    frontend_api_base_url = _get_config_value(
        "CORPUSAGENT2_FRONTEND_API_BASE_URL",
        f"https://{tunnel_hostname}" if tunnel_hostname else "",
    )
    resource_plan = compute_docker_resource_plan(detect_host_hardware(), gpu_mode=args.gpu)
    _upsert_dotenv(dotenv_path, resource_plan.env)
    if frontend_api_base_url:
        _upsert_dotenv(
            dotenv_path,
            {"CORPUSAGENT2_FRONTEND_API_BASE_URL": frontend_api_base_url},
        )
        _run([sys.executable, str(REPO_ROOT / "scripts" / "13_write_frontend_config.py")])

    stack_service_name = args.api_service_name or args.stack_service_name
    docker_bin = shutil.which("docker") or "/usr/bin/docker"
    compose_exec = _compose_exec_fragment(docker_bin=docker_bin, use_gpu=resource_plan.use_gpu)
    stack_start = shlex.quote(
        f"{compose_exec} up -d --no-recreate postgres opensearch && "
        f"{compose_exec} up -d --build --no-deps corpusagent2-api corpusagent2-mcp"
    )

    stack_service = f"""[Unit]
Description=CorpusAgent2 Docker stack
After=network-online.target docker.service
Wants=network-online.target docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory={REPO_ROOT / 'deploy'}
EnvironmentFile={dotenv_path}
ExecStart=/bin/sh -lc {stack_start}
ExecStop={compose_exec} stop corpusagent2-api corpusagent2-mcp
TimeoutStartSec=0
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
"""
    service_dir = Path("/etc/systemd/system")
    prefix = _sudo_prefix()
    stack_service_path = service_dir / f"{stack_service_name}.service"
    temp_stack = REPO_ROOT / "deploy" / f"{stack_service_name}.service"
    temp_stack.write_text(stack_service, encoding="utf-8")
    _run(prefix + ["install", "-m", "0644", str(temp_stack), str(stack_service_path)])

    if not args.skip_tunnel:
        cloudflared_config = f"""tunnel: {tunnel_id}
credentials-file: {credentials_file}
ingress:
  - hostname: {tunnel_hostname}
    service: {args.backend_url}
  - service: http_status:404
"""
        config_path = cloudflared_dir / "config.yml"
        config_path.write_text(cloudflared_config, encoding="utf-8")

        tunnel_service = f"""[Unit]
Description=CorpusAgent2 Cloudflare Tunnel
After=network-online.target {stack_service_name}.service
Wants=network-online.target {stack_service_name}.service

[Service]
Type=simple
User={service_user}
ExecStart={shutil.which('cloudflared') or '/usr/bin/cloudflared'} tunnel --config {config_path} run {tunnel_name}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
        tunnel_service_path = service_dir / f"{args.tunnel_service_name}.service"
        temp_tunnel = REPO_ROOT / "deploy" / f"{args.tunnel_service_name}.service"
        temp_tunnel.write_text(tunnel_service, encoding="utf-8")
        _run(prefix + ["install", "-m", "0644", str(temp_tunnel), str(tunnel_service_path)])

    _run(prefix + ["systemctl", "daemon-reload"])
    _run(prefix + ["systemctl", "enable", "--now", f"{stack_service_name}.service"])
    if not args.skip_tunnel:
        _run(prefix + ["systemctl", "enable", "--now", f"{args.tunnel_service_name}.service"])

    print("")
    print(f"Docker stack service installed: {stack_service_name}.service")
    if not args.skip_tunnel:
        print(f"Tunnel service installed: {args.tunnel_service_name}.service")
        print(f"Stable public API URL: {frontend_api_base_url}")


if __name__ == "__main__":
    main()
