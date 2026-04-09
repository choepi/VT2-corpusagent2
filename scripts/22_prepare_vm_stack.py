from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import ssl
import subprocess
import sys
import time
from typing import Iterable
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = REPO_ROOT / "deploy"
COMPOSE_FILE = DEPLOY_ROOT / "docker-compose.yml"
DEFAULT_DOCKER_DATA_DIR = DEPLOY_ROOT / "data"
DEFAULT_PG_HOST = "127.0.0.1"
DEFAULT_PG_PORT = 5432
DEFAULT_OPENSEARCH_URL = "https://localhost:9200"
DEFAULT_OPENSEARCH_INDEX = "article-corpus-opensearch"
DEFAULT_OPENSEARCH_USERNAME = "admin"
DEFAULT_OPENSEARCH_PASSWORD = "VerySecurePassword123!"
DEFAULT_OPENSEARCH_VERIFY_SSL = False


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


DOTENV_VALUES = _load_dotenv(REPO_ROOT / ".env")


def _env_value(name: str, default: str) -> str:
    if name in os.environ:
        value = os.environ[name].strip()
        if value:
            return value
    value = DOTENV_VALUES.get(name, "").strip()
    if value:
        return value
    return default


def _truthy(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _run(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> None:
    print(f"[run] {shlex.join(command)}")
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def _capture(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _sudo_prefix() -> list[str]:
    if os.name == "nt":
        return []
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None or geteuid() == 0:
        return []
    if shutil.which("sudo"):
        return ["sudo"]
    return []


def _docker_compose_command() -> list[str]:
    docker = shutil.which("docker")
    if docker:
        probe = _capture([docker, "compose", "version"])
        if probe.returncode == 0:
            return _sudo_prefix() + [docker, "compose"]
    docker_compose = shutil.which("docker-compose")
    if docker_compose:
        return _sudo_prefix() + [docker_compose]
    raise RuntimeError(
        "Docker Compose is not available. Install docker + docker compose plugin first."
    )


def _docker_runtime_is_available() -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    docker_probe = _capture([docker, "--version"])
    if docker_probe.returncode != 0:
        return False
    compose_probe = _capture([docker, "compose", "version"])
    return compose_probe.returncode == 0


def _project_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in DOTENV_VALUES.items():
        env.setdefault(key, value)
    env.setdefault("CORPUSAGENT2_DOCKER_DATA_DIR", str(DEFAULT_DOCKER_DATA_DIR))
    return env


def _venv_python() -> Path:
    if os.name == "nt":
        return REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    return REPO_ROOT / ".venv" / "bin" / "python"


def _venv_uv() -> Path | None:
    if os.name == "nt":
        candidate = REPO_ROOT / ".venv" / "Scripts" / "uv.exe"
    else:
        candidate = REPO_ROOT / ".venv" / "bin" / "uv"
    return candidate if candidate.exists() else None


def _summary_is_present(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _ensure_git_checkout() -> None:
    if not (REPO_ROOT / ".git").exists():
        raise RuntimeError(f"Expected a git checkout at {REPO_ROOT}")


def _ensure_uv() -> str:
    uv_bin = shutil.which("uv")
    if uv_bin:
        return uv_bin
    if os.name == "nt":
        raise RuntimeError(
            "uv is required but not found on PATH. Install uv first on Windows or use the existing rebuild script."
        )
    install_command = "curl -LsSf https://astral.sh/uv/install.sh | sh"
    print("[info] uv not found, installing via astral installer.")
    _run(["sh", "-c", install_command], env=_project_env())
    candidate = Path.home() / ".local" / "bin" / "uv"
    if candidate.exists():
        return str(candidate)
    uv_bin = shutil.which("uv")
    if uv_bin:
        return uv_bin
    raise RuntimeError("uv installation did not produce a usable binary.")


def _install_system_packages() -> None:
    if os.name == "nt":
        print("[skip] system package install is intended for Ubuntu/Linux only.")
        return
    apt_get = shutil.which("apt-get")
    systemctl = shutil.which("systemctl")
    if not apt_get:
        raise RuntimeError("apt-get was not found. This bootstrap currently targets Ubuntu/Debian VMs.")
    prefix = _sudo_prefix()
    _run(prefix + [apt_get, "update"])
    base_packages = [
        "ca-certificates",
        "curl",
        "git",
        "python3",
        "python3-venv",
        "python3-pip",
    ]
    _run(prefix + [apt_get, "install", "-y", *base_packages])

    if _docker_runtime_is_available():
        print("[ready] Docker and Docker Compose are already available; skipping package install.")
        return

    docker_package_sets = [
        ["docker.io", "docker-compose-v2"],
        ["docker.io", "docker-compose-plugin"],
        ["docker.io", "docker-compose"],
    ]
    docker_installed = False
    for package_set in docker_package_sets:
        result = _capture(prefix + [apt_get, "install", "-y", *package_set])
        if result.returncode == 0:
            docker_installed = True
            print(f"[ready] installed Docker packages: {', '.join(package_set)}")
            break
        print(
            f"[warn] apt-get could not install Docker package set {package_set}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    if not docker_installed:
        raise RuntimeError(
            "Unable to install Docker packages with apt-get. Install docker.io plus a Docker Compose package manually."
        )

    if systemctl:
        docker_service_name = "docker"
        result = _capture(prefix + [systemctl, "enable", "--now", docker_service_name])
        if result.returncode != 0:
            print(
                f"[warn] could not enable/start {docker_service_name} service automatically: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )


def _ensure_venv_and_deps(*, skip_provider_assets: bool) -> Path:
    uv_bin = _ensure_uv()
    python_exe = _venv_python()
    if not python_exe.exists():
        _run([uv_bin, "venv", ".venv", "--python", sys.executable])
    running_inside_target_venv = False
    try:
        running_inside_target_venv = python_exe.resolve() == Path(sys.executable).resolve()
    except FileNotFoundError:
        running_inside_target_venv = False
    if os.name == "nt" and running_inside_target_venv:
        print(
            "[skip] uv sync skipped because the bootstrap is running from the managed Windows venv; "
            "rerun with system python if you need dependency refresh."
        )
    else:
        _run([uv_bin, "sync", "--extra", "nlp-providers"])
    if not python_exe.exists():
        raise RuntimeError(f"Expected virtualenv python at {python_exe}")
    if not skip_provider_assets:
        _run([str(python_exe), str(REPO_ROOT / "scripts" / "17_download_provider_assets.py")])
    return python_exe


def _ensure_data_pipeline(
    python_exe: Path,
    *,
    skip_data: bool,
    refresh_data: bool,
    refresh_assets: bool,
) -> None:
    if skip_data:
        print("[skip] data preparation disabled.")
        return

    incoming_file = REPO_ROOT / "data" / "raw" / "incoming" / "cc_news.jsonl.gz"
    staged_dir = REPO_ROOT / "data" / "raw" / "ccnews_staged"
    documents_parquet = REPO_ROOT / "data" / "processed" / "documents.parquet"
    dense_dir = REPO_ROOT / "data" / "indices" / "dense"
    lexical_dir = REPO_ROOT / "data" / "indices" / "lexical"
    metadata_path = REPO_ROOT / "data" / "indices" / "doc_metadata.parquet"
    build_lexical_assets = _truthy(_env_value("CORPUSAGENT2_BUILD_LEXICAL_ASSETS", "false"))
    build_dense_assets = _truthy(_env_value("CORPUSAGENT2_BUILD_DENSE_ASSETS", "true"))

    if refresh_data or not documents_parquet.exists():
        if refresh_data or not incoming_file.exists():
            _run([str(python_exe), str(REPO_ROOT / "scripts" / "00_1_downlaod.py")])
        if refresh_data or not any(staged_dir.rglob("*.jsonl")) and not any(staged_dir.rglob("*.jsonl.gz")):
            _run([str(python_exe), str(REPO_ROOT / "scripts" / "00_stage_ccnews_files.py")])
        _run([str(python_exe), str(REPO_ROOT / "scripts" / "01_prepare_dataset.py")])

    required_paths = [metadata_path]
    if build_lexical_assets:
        required_paths.extend(
            [
                lexical_dir / "tfidf_vectorizer.joblib",
                lexical_dir / "tfidf_matrix.joblib",
                lexical_dir / "tfidf_doc_ids.joblib",
            ]
        )
    if build_dense_assets:
        required_paths.extend(
            [
                dense_dir / "dense_embeddings.npy",
                dense_dir / "dense_doc_ids.joblib",
            ]
        )
    assets_ready = all(path.exists() for path in required_paths)
    if refresh_assets or refresh_data or not assets_ready:
        asset_env = os.environ.copy()
        asset_env["CORPUSAGENT2_BUILD_LEXICAL_ASSETS"] = "true" if build_lexical_assets else "false"
        asset_env["CORPUSAGENT2_BUILD_DENSE_ASSETS"] = "true" if build_dense_assets else "false"
        asset_env.setdefault("CORPUSAGENT2_STREAM_DENSE_ASSETS", "true")
        asset_env.setdefault("CORPUSAGENT2_DENSE_BATCH_SIZE", "64")
        asset_env.setdefault("CORPUSAGENT2_DENSE_CHUNK_SIZE", "2048")
        _run([str(python_exe), str(REPO_ROOT / "scripts" / "02_build_retrieval_assets.py")], env=asset_env)


def _wait_for_port(host: str, port: int, *, timeout_s: float, label: str) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2.0)
            if sock.connect_ex((host, port)) == 0:
                print(f"[ready] {label} on {host}:{port}")
                return
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for {label} on {host}:{port}")


def _pg_host_port() -> tuple[str, int]:
    dsn = _env_value(
        "CORPUSAGENT2_PG_DSN",
        f"postgresql://corpus:corpus@{DEFAULT_PG_HOST}:{DEFAULT_PG_PORT}/corpus_db",
    )
    parsed = urlparse(dsn)
    host = parsed.hostname or DEFAULT_PG_HOST
    port = int(parsed.port or DEFAULT_PG_PORT)
    return host, port


def _opensearch_host_port() -> tuple[str, int]:
    parsed = urlparse(_env_value("CORPUSAGENT2_OPENSEARCH_URL", DEFAULT_OPENSEARCH_URL))
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    return host, port


def _opensearch_request(path_suffix: str) -> dict[str, object]:
    base_url = _env_value("CORPUSAGENT2_OPENSEARCH_URL", DEFAULT_OPENSEARCH_URL).rstrip("/")
    username = _env_value("CORPUSAGENT2_OPENSEARCH_USERNAME", DEFAULT_OPENSEARCH_USERNAME)
    password = _env_value("CORPUSAGENT2_OPENSEARCH_PASSWORD", DEFAULT_OPENSEARCH_PASSWORD)
    verify_ssl = _truthy(_env_value("CORPUSAGENT2_OPENSEARCH_VERIFY_SSL", str(DEFAULT_OPENSEARCH_VERIFY_SSL).lower()))
    credentials = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    request = urlrequest.Request(f"{base_url}/{path_suffix.lstrip('/')}")
    request.add_header("Authorization", f"Basic {credentials}")
    context = ssl.create_default_context()
    if not verify_ssl:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    with urlrequest.urlopen(request, timeout=15.0, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_opensearch(timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            payload = _opensearch_request("")
        except (urlerror.URLError, TimeoutError, OSError, json.JSONDecodeError):
            time.sleep(2.0)
            continue
        if payload:
            print("[ready] OpenSearch HTTP endpoint is responding.")
            return
    raise TimeoutError("Timed out waiting for OpenSearch to respond.")


def _opensearch_count() -> int | None:
    index_name = _env_value("CORPUSAGENT2_OPENSEARCH_INDEX", DEFAULT_OPENSEARCH_INDEX)
    try:
        payload = _opensearch_request(f"{index_name}/_count")
    except Exception:
        return None
    count = payload.get("count")
    try:
        return int(count) if count is not None else None
    except (TypeError, ValueError):
        return None


def _ensure_docker_services(*, with_dashboards: bool) -> None:
    compose_command = _docker_compose_command()
    env = _project_env()
    services = ["postgres", "opensearch"]
    if with_dashboards:
        services.append("opensearch-dashboards")
    command = compose_command + ["-f", str(COMPOSE_FILE), "up", "-d", *services]
    _run(command, cwd=REPO_ROOT, env=env)
    pg_host, pg_port = _pg_host_port()
    os_host, os_port = _opensearch_host_port()
    _wait_for_port(pg_host, pg_port, timeout_s=180.0, label="Postgres")
    _wait_for_port(os_host, os_port, timeout_s=240.0, label="OpenSearch TCP")
    _wait_for_opensearch(timeout_s=240.0)


def _maybe_run_script(
    python_exe: Path,
    script_name: str,
    *,
    summary_path: Path | None = None,
    force: bool = False,
    should_skip: bool = False,
) -> None:
    if should_skip:
        print(f"[skip] {script_name} already prepared.")
        return
    if summary_path is not None and _summary_is_present(summary_path) and not force:
        print(f"[skip] {script_name} summary already present at {summary_path}")
        return
    _run([str(python_exe), str(REPO_ROOT / "scripts" / script_name)])


def _write_summary(payload: dict[str, object]) -> Path:
    summary_path = REPO_ROOT / "outputs" / "deployment" / "prepare_vm_stack_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the CorpusAgent2 VM stack: venv, data assets, Docker services, Postgres, and OpenSearch."
    )
    parser.add_argument("--install-system", action="store_true", help="Install Ubuntu system packages including Docker.")
    parser.add_argument("--skip-provider-assets", action="store_true", help="Skip spaCy/Stanza/NLTK/TextBlob asset downloads.")
    parser.add_argument("--skip-data", action="store_true", help="Skip dataset download/preprocessing/retrieval asset build.")
    parser.add_argument("--refresh-data", action="store_true", help="Re-download and rebuild the processed corpus.")
    parser.add_argument("--refresh-assets", action="store_true", help="Rebuild dense/lexical retrieval assets.")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker service startup and database/search preparation.")
    parser.add_argument("--refresh-postgres", action="store_true", help="Re-run Postgres schema, ingest, and pgvector indexing.")
    parser.add_argument("--refresh-opensearch", action="store_true", help="Re-run full OpenSearch bulk indexing.")
    parser.add_argument("--with-dashboards", action="store_true", help="Start OpenSearch Dashboards as well.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    _ensure_git_checkout()
    if args.install_system:
        _install_system_packages()

    python_exe = _ensure_venv_and_deps(skip_provider_assets=args.skip_provider_assets)
    _ensure_data_pipeline(
        python_exe,
        skip_data=args.skip_data,
        refresh_data=args.refresh_data,
        refresh_assets=args.refresh_assets,
    )
    _run([str(python_exe), str(REPO_ROOT / "scripts" / "13_write_frontend_config.py")])

    postgres_schema_summary = REPO_ROOT / "outputs" / "postgres" / "init_schema_summary.json"
    postgres_ingest_summary = REPO_ROOT / "outputs" / "postgres" / "ingest_summary.json"
    pgvector_index_summary = REPO_ROOT / "outputs" / "postgres" / "build_index_summary.json"
    opensearch_summary = REPO_ROOT / "outputs" / "opensearch" / "bulk_index_summary.json"

    if not args.skip_docker:
        _ensure_docker_services(with_dashboards=args.with_dashboards)
        _maybe_run_script(
            python_exe,
            "09_init_postgres_schema.py",
            summary_path=postgres_schema_summary,
            force=args.refresh_postgres,
        )
        _maybe_run_script(
            python_exe,
            "10_ingest_parquet_to_postgres.py",
            summary_path=postgres_ingest_summary,
            force=args.refresh_postgres,
        )
        _maybe_run_script(
            python_exe,
            "11_build_pgvector_index.py",
            summary_path=pgvector_index_summary,
            force=args.refresh_postgres,
        )

        documents_parquet = REPO_ROOT / "data" / "processed" / "documents.parquet"
        current_opensearch_count = _opensearch_count()
        expected_documents = None
        if documents_parquet.exists():
            try:
                import pyarrow.parquet as pq

                expected_documents = int(pq.ParquetFile(documents_parquet).metadata.num_rows)
            except Exception:
                expected_documents = None
        opensearch_needs_refresh = args.refresh_opensearch
        if not opensearch_needs_refresh and not _summary_is_present(opensearch_summary):
            opensearch_needs_refresh = True
        if not opensearch_needs_refresh and expected_documents is not None and current_opensearch_count != expected_documents:
            opensearch_needs_refresh = True
        _maybe_run_script(
            python_exe,
            "21_bulk_index_opensearch.py",
            summary_path=opensearch_summary,
            force=opensearch_needs_refresh,
            should_skip=not opensearch_needs_refresh,
        )

    summary_path = _write_summary(
        {
            "repo_root": str(REPO_ROOT),
            "venv_python": str(python_exe),
            "docker_compose": str(COMPOSE_FILE),
            "docker_data_dir": str(_project_env()["CORPUSAGENT2_DOCKER_DATA_DIR"]),
            "data_prepared": not args.skip_data,
            "docker_prepared": not args.skip_docker,
            "postgres_summary": str(postgres_ingest_summary) if postgres_ingest_summary.exists() else "",
            "opensearch_summary": str(opensearch_summary) if opensearch_summary.exists() else "",
        }
    )
    print("")
    print("VM stack preparation complete.")
    print(f"Summary: {summary_path}")
    print(f"Backend start: {python_exe} {REPO_ROOT / 'scripts' / '12_run_agent_api.py'}")
    print(f"Frontend config: {REPO_ROOT / 'web' / 'config.js'}")
    print(
        f"Cloudflared helper: {python_exe} {REPO_ROOT / 'scripts' / '23_start_cloudflared_tunnel.py'}"
    )


if __name__ == "__main__":
    main()
