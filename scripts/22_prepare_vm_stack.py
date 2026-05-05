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
MCP_COMPOSE_FILE = DEPLOY_ROOT / "docker-compose.mcp.yml"
GPU_COMPOSE_FILE = DEPLOY_ROOT / "docker-compose.mcp.gpu.yml"
DEFAULT_DOCKER_DATA_DIR = DEPLOY_ROOT / "data"
DEFAULT_PG_HOST = "127.0.0.1"
DEFAULT_PG_PORT = 5432
DEFAULT_OPENSEARCH_URL = "https://localhost:9200"
DEFAULT_OPENSEARCH_INDEX = "article-corpus-opensearch"
DEFAULT_OPENSEARCH_USERNAME = "admin"
DEFAULT_OPENSEARCH_PASSWORD = "VerySecurePassword123!"
DEFAULT_OPENSEARCH_VERIFY_SSL = False
DEFAULT_RETRIEVAL_PROFILE = "hybrid"

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.deploy_resources import compute_docker_resource_plan, detect_host_hardware


class BootstrapError(RuntimeError):
    pass


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


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes}m {remainder:.1f}s"
    hours = minutes // 60
    minute_remainder = minutes % 60
    return f"{hours}h {minute_remainder}m {remainder:.0f}s"


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
    raise BootstrapError(
        "Docker Compose is not available. Install docker + docker compose plugin first."
    )


def _compose_file_args(*, with_mcp: bool = False, use_gpu: bool = False) -> list[str]:
    args = ["-f", str(COMPOSE_FILE)]
    if with_mcp:
        args.extend(["-f", str(MCP_COMPOSE_FILE)])
    if use_gpu:
        args.extend(["-f", str(GPU_COMPOSE_FILE)])
    return args


def _docker_runtime_is_available() -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    docker_probe = _capture([docker, "--version"])
    if docker_probe.returncode != 0:
        return False
    compose_probe = _capture([docker, "compose", "version"])
    return compose_probe.returncode == 0


def _docker_active_context() -> str | None:
    docker = shutil.which("docker")
    if not docker:
        return None
    probe = _capture([docker, "context", "show"])
    if probe.returncode != 0:
        return None
    value = probe.stdout.strip()
    return value or None


def _ensure_docker_engine_ready() -> None:
    docker = shutil.which("docker")
    if not docker:
        return
    probe = _capture([docker, "info"])
    if probe.returncode == 0:
        return

    active_context = _docker_active_context()
    details = probe.stderr.strip() or probe.stdout.strip()
    message_lines = ["Docker is installed, but the Docker engine is not reachable."]
    if active_context:
        message_lines.append(f"Active Docker context: {active_context}.")
    if os.name == "nt":
        if active_context and active_context != "desktop-linux":
            message_lines.append(
                "This stack expects Docker Desktop to be running with the Linux container engine."
            )
        else:
            message_lines.append(
                "Start Docker Desktop and wait until the Linux engine reports healthy before rerunning this bootstrap."
            )
        if "dockerDesktopLinuxEngine" in details:
            message_lines.append(
                "The Docker Desktop Linux engine pipe (`//./pipe/dockerDesktopLinuxEngine`) is missing, which usually means Docker Desktop is not running yet."
            )
        message_lines.append(
            "If you only want to finish the Python/data setup right now, rerun with `--skip-docker`."
        )
    else:
        message_lines.append("Start the Docker daemon/service and rerun.")
    if details:
        message_lines.append(f"Docker reported: {details}")
    raise BootstrapError("\n".join(message_lines))


def _project_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in DOTENV_VALUES.items():
        env.setdefault(key, value)
    env.setdefault("CORPUSAGENT2_DOCKER_DATA_DIR", str(DEFAULT_DOCKER_DATA_DIR))
    return env


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
    DOTENV_VALUES.update(updates)


def _retrieval_profile_env(profile: str) -> dict[str, str]:
    normalized = profile.strip().lower()
    if normalized == "hybrid":
        return {
            "CORPUSAGENT2_VM_RETRIEVAL_PROFILE": "hybrid",
            "CORPUSAGENT2_BUILD_LEXICAL_ASSETS": "true",
            "CORPUSAGENT2_BUILD_DENSE_ASSETS": "false",
            "CORPUSAGENT2_PG_INCLUDE_EMBEDDINGS": "false",
            "CORPUSAGENT2_ENABLE_DENSE_RETRIEVAL": "true",
            "CORPUSAGENT2_RETRIEVAL_BACKEND": "pgvector",
            "CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE": "hybrid",
            "CORPUSAGENT2_PG_BACKFILL_FETCH_BATCH_SIZE": "128",
            "CORPUSAGENT2_PG_BACKFILL_ENCODE_BATCH_SIZE": "16",
            "CORPUSAGENT2_PG_BUILD_IVFFLAT": "true",
            "CORPUSAGENT2_PG_BUILD_HNSW": "false",
            "CORPUSAGENT2_PG_IVF_LISTS": "1024",
            "CORPUSAGENT2_PG_INGEST_BATCH_SIZE": "250",
            "CORPUSAGENT2_PG_INGEST_READ_BATCH_SIZE": "2000",
            "CORPUSAGENT2_PG_INGEST_COMMIT_EVERY_BATCHES": "8",
            "CORPUSAGENT2_PG_INGEST_PROGRESS_EVERY_ROWS": "10000",
        }
    return {
        "CORPUSAGENT2_VM_RETRIEVAL_PROFILE": "no-dense",
        "CORPUSAGENT2_BUILD_LEXICAL_ASSETS": "true",
        "CORPUSAGENT2_BUILD_DENSE_ASSETS": "false",
        "CORPUSAGENT2_PG_INCLUDE_EMBEDDINGS": "false",
        "CORPUSAGENT2_ENABLE_DENSE_RETRIEVAL": "false",
        "CORPUSAGENT2_RETRIEVAL_BACKEND": "local",
        "CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE": "lexical",
        "CORPUSAGENT2_PG_INGEST_BATCH_SIZE": "250",
        "CORPUSAGENT2_PG_INGEST_READ_BATCH_SIZE": "2000",
        "CORPUSAGENT2_PG_INGEST_COMMIT_EVERY_BATCHES": "8",
        "CORPUSAGENT2_PG_INGEST_PROGRESS_EVERY_ROWS": "10000",
    }


def _dense_retrieval_requested(env: dict[str, str]) -> bool:
    raw = env.get(
        "CORPUSAGENT2_ENABLE_DENSE_RETRIEVAL",
        _env_value("CORPUSAGENT2_ENABLE_DENSE_RETRIEVAL", "false"),
    )
    return _truthy(raw)


def _expected_pgvector_index_names(env: dict[str, str], *, table_name: str) -> set[str]:
    names: set[str] = set()
    if _truthy(env.get("CORPUSAGENT2_PG_BUILD_IVFFLAT", _env_value("CORPUSAGENT2_PG_BUILD_IVFFLAT", "true"))):
        names.add(f"idx_{table_name}_embedding_ivfflat")
    if _truthy(env.get("CORPUSAGENT2_PG_BUILD_HNSW", _env_value("CORPUSAGENT2_PG_BUILD_HNSW", "true"))):
        names.add(f"idx_{table_name}_embedding_hnsw")
    return names


def _pgvector_backfill_complete(
    *,
    expected_documents: int | None,
    current_postgres_count: int | None,
    current_embedding_count: int | None,
) -> bool:
    if current_postgres_count is None or current_embedding_count is None:
        return False
    if current_postgres_count <= 0 or current_embedding_count != current_postgres_count:
        return False
    if expected_documents is not None and current_postgres_count != expected_documents:
        return False
    return True


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


def _documents_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _postgres_scalar(query: str) -> int | None:
    try:
        from psycopg import connect
    except Exception:
        return None
    try:
        with connect(_pg_dsn()) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                row = cursor.fetchone()
                if not row:
                    return None
                return int(row[0])
    except Exception:
        return None


def _postgres_table_row_count() -> int | None:
    table_name = _env_value("CORPUSAGENT2_PG_TABLE", "article_corpus")
    if not table_name.replace("_", "").isalnum():
        return None
    return _postgres_scalar(f"SELECT COUNT(*) FROM {table_name};")


def _postgres_embedding_row_count() -> int | None:
    table_name = _env_value("CORPUSAGENT2_PG_TABLE", "article_corpus")
    if not table_name.replace("_", "").isalnum():
        return None
    return _postgres_scalar(f"SELECT COUNT(*) FROM {table_name} WHERE dense_embedding IS NOT NULL;")


def _postgres_index_names() -> set[str]:
    table_name = _env_value("CORPUSAGENT2_PG_TABLE", "article_corpus")
    if not table_name.replace("_", "").isalnum():
        return set()
    try:
        from psycopg import connect
    except Exception:
        return set()
    try:
        with connect(_pg_dsn()) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE schemaname = current_schema() AND tablename = %s;
                    """,
                    (table_name,),
                )
                return {str(row[0]) for row in cursor.fetchall()}
    except Exception:
        return set()


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
    env: dict[str, str],
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
    build_lexical_assets = _truthy(env.get("CORPUSAGENT2_BUILD_LEXICAL_ASSETS", _env_value("CORPUSAGENT2_BUILD_LEXICAL_ASSETS", "true")))
    build_dense_assets = _truthy(env.get("CORPUSAGENT2_BUILD_DENSE_ASSETS", _env_value("CORPUSAGENT2_BUILD_DENSE_ASSETS", "false")))

    if refresh_data or not documents_parquet.exists():
        if refresh_data or not incoming_file.exists():
            _run([str(python_exe), str(REPO_ROOT / "scripts" / "00_1_downlaod.py")], env=env)
        if refresh_data or not any(staged_dir.rglob("*.jsonl")) and not any(staged_dir.rglob("*.jsonl.gz")):
            _run([str(python_exe), str(REPO_ROOT / "scripts" / "00_stage_ccnews_files.py")], env=env)
        _run([str(python_exe), str(REPO_ROOT / "scripts" / "01_prepare_dataset.py")], env=env)

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
        asset_env = env.copy()
        asset_env["CORPUSAGENT2_BUILD_LEXICAL_ASSETS"] = "true" if build_lexical_assets else "false"
        asset_env["CORPUSAGENT2_BUILD_DENSE_ASSETS"] = "true" if build_dense_assets else "false"
        asset_env.setdefault("CORPUSAGENT2_STREAM_DENSE_ASSETS", "true")
        asset_env.setdefault("CORPUSAGENT2_DENSE_BATCH_SIZE", "128")
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


def _pg_dsn() -> str:
    return _env_value(
        "CORPUSAGENT2_PG_DSN",
        f"postgresql://corpus:corpus@{DEFAULT_PG_HOST}:{DEFAULT_PG_PORT}/corpus_db",
    )


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
    _ensure_docker_engine_ready()
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


def _ensure_mcp_service(*, build: bool, use_gpu: bool = False) -> None:
    compose_command = _docker_compose_command()
    _ensure_docker_engine_ready()
    env = _project_env()
    command = compose_command + _compose_file_args(with_mcp=True, use_gpu=use_gpu) + ["up", "-d"]
    if build:
        command.append("--build")
    command.append("corpusagent2-mcp")
    _run(command, cwd=REPO_ROOT, env=env)
    print(
        "[ready] MCP server requested at "
        f"http://127.0.0.1:{env.get('CORPUSAGENT2_MCP_PORT', '8765')}{env.get('CORPUSAGENT2_MCP_PATH', '/mcp')}"
    )


def _ensure_api_service(*, build: bool, use_gpu: bool = False) -> None:
    compose_command = _docker_compose_command()
    _ensure_docker_engine_ready()
    env = _project_env()
    command = compose_command + _compose_file_args(use_gpu=use_gpu) + ["up", "-d"]
    if build:
        command.append("--build")
    command.append("corpusagent2-api")
    _run(command, cwd=REPO_ROOT, env=env)
    print(f"[ready] API backend requested at http://127.0.0.1:{env.get('CORPUSAGENT2_SERVER_PORT', '8001')}")


def _maybe_run_script(
    python_exe: Path,
    script_name: str,
    *,
    env: dict[str, str],
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
    _run([str(python_exe), str(REPO_ROOT / "scripts" / script_name)], env=env)


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
    parser.add_argument("--start-api", action="store_true", help="Start the Dockerized FastAPI backend after preparation completes.")
    parser.add_argument(
        "--setup-mcp-only",
        action="store_true",
        help="Only start existing Docker services plus the MCP job server; do not run data, ingest, dense, or indexing steps.",
    )
    parser.add_argument("--start-mcp", action="store_true", help="Start the Docker-hosted MCP job server after stack preparation.")
    parser.add_argument("--no-mcp-build", action="store_true", help="Do not rebuild the MCP Docker image before starting it.")
    parser.add_argument("--gpu", choices=["auto", "on", "off"], default="auto", help="GPU compose selection for Dockerized API/MCP services.")
    parser.add_argument(
        "--retrieval-profile",
        choices=["no-dense", "hybrid"],
        default=_env_value("CORPUSAGENT2_VM_RETRIEVAL_PROFILE", DEFAULT_RETRIEVAL_PROFILE),
        help="Choose a stable VM retrieval profile. 'hybrid' is the default service-backed path.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    started = time.perf_counter()
    args = parse_args(argv)
    _ensure_git_checkout()
    if args.install_system:
        _install_system_packages()

    profile_env = _retrieval_profile_env(args.retrieval_profile)
    resource_plan = compute_docker_resource_plan(detect_host_hardware(), gpu_mode=args.gpu)
    _upsert_dotenv(REPO_ROOT / ".env", resource_plan.env)
    command_env = _project_env()
    command_env.update(profile_env)
    command_env.update(resource_plan.env)
    _upsert_dotenv(REPO_ROOT / ".env", profile_env)
    if args.setup_mcp_only:
        if args.skip_docker:
            raise BootstrapError("--setup-mcp-only requires Docker; remove --skip-docker.")
        _ensure_docker_services(with_dashboards=args.with_dashboards)
        _ensure_mcp_service(build=not args.no_mcp_build, use_gpu=resource_plan.use_gpu)
        elapsed_s = time.perf_counter() - started
        summary_path = _write_summary(
            {
                "repo_root": str(REPO_ROOT),
                "retrieval_profile": args.retrieval_profile,
                "docker_compose": str(COMPOSE_FILE),
                "mcp_docker_compose": str(MCP_COMPOSE_FILE),
                "docker_data_dir": str(_project_env()["CORPUSAGENT2_DOCKER_DATA_DIR"]),
                "data_prepared": False,
                "docker_prepared": True,
                "mcp_started": True,
                "mcp_url": f"http://127.0.0.1:{command_env.get('CORPUSAGENT2_MCP_PORT', '8765')}{command_env.get('CORPUSAGENT2_MCP_PATH', '/mcp')}",
                "resource_plan": resource_plan.env,
                "total_elapsed_seconds": round(elapsed_s, 3),
            }
        )
        print("")
        print("MCP-only preparation complete.")
        print(f"Total time: {_format_duration(elapsed_s)}")
        print(f"Summary: {summary_path}")
        return

    python_exe = _ensure_venv_and_deps(skip_provider_assets=args.skip_provider_assets)
    _ensure_data_pipeline(
        python_exe,
        skip_data=args.skip_data,
        refresh_data=args.refresh_data,
        refresh_assets=args.refresh_assets,
        env=command_env,
    )
    _run([str(python_exe), str(REPO_ROOT / "scripts" / "13_write_frontend_config.py")], env=command_env)

    postgres_schema_summary = REPO_ROOT / "outputs" / "postgres" / "init_schema_summary.json"
    postgres_ingest_summary = REPO_ROOT / "outputs" / "postgres" / "ingest_summary.json"
    pgvector_backfill_summary = REPO_ROOT / "outputs" / "postgres" / "backfill_dense_embeddings_summary.json"
    pgvector_index_summary = REPO_ROOT / "outputs" / "postgres" / "build_index_summary.json"
    opensearch_summary = REPO_ROOT / "outputs" / "opensearch" / "bulk_index_summary.json"

    dense_retrieval_requested = _dense_retrieval_requested(command_env)
    if not args.skip_docker:
        _ensure_docker_services(with_dashboards=args.with_dashboards)
        documents_parquet = REPO_ROOT / "data" / "processed" / "documents.parquet"
        expected_documents = _documents_row_count(documents_parquet)
        current_postgres_count = _postgres_table_row_count()
        current_embedding_count = _postgres_embedding_row_count()
        postgres_index_names = _postgres_index_names()
        table_name = command_env.get("CORPUSAGENT2_PG_TABLE", _env_value("CORPUSAGENT2_PG_TABLE", "article_corpus"))
        expected_pgvector_indices = _expected_pgvector_index_names(command_env, table_name=table_name)

        _maybe_run_script(
            python_exe,
            "09_init_postgres_schema.py",
            env=command_env,
            summary_path=postgres_schema_summary,
            force=args.refresh_postgres,
        )
        postgres_needs_refresh = args.refresh_postgres
        if not postgres_needs_refresh and not _summary_is_present(postgres_ingest_summary):
            postgres_needs_refresh = True
        if not postgres_needs_refresh and expected_documents is not None and current_postgres_count != expected_documents:
            postgres_needs_refresh = True
        _maybe_run_script(
            python_exe,
            "10_ingest_parquet_to_postgres.py",
            env=command_env,
            summary_path=postgres_ingest_summary,
            force=postgres_needs_refresh,
        )

        current_postgres_count = _postgres_table_row_count()
        current_embedding_count = _postgres_embedding_row_count()
        postgres_index_names = _postgres_index_names()

        pgvector_backfill_needs_refresh = args.refresh_postgres
        if not pgvector_backfill_needs_refresh and not _summary_is_present(pgvector_backfill_summary):
            pgvector_backfill_needs_refresh = True
        if (
            not pgvector_backfill_needs_refresh
            and dense_retrieval_requested
            and not _pgvector_backfill_complete(
                expected_documents=expected_documents,
                current_postgres_count=current_postgres_count,
                current_embedding_count=current_embedding_count,
            )
        ):
            pgvector_backfill_needs_refresh = True
        _maybe_run_script(
            python_exe,
            "26_backfill_pgvector_embeddings.py",
            env=command_env,
            summary_path=pgvector_backfill_summary,
            force=pgvector_backfill_needs_refresh,
            should_skip=not dense_retrieval_requested,
        )

        current_postgres_count = _postgres_table_row_count()
        current_embedding_count = _postgres_embedding_row_count()
        postgres_index_names = _postgres_index_names()
        pgvector_ready = _pgvector_backfill_complete(
            expected_documents=expected_documents,
            current_postgres_count=current_postgres_count,
            current_embedding_count=current_embedding_count,
        )

        pgvector_needs_refresh = args.refresh_postgres
        if not pgvector_needs_refresh and not _summary_is_present(pgvector_index_summary):
            pgvector_needs_refresh = True
        if (
            not pgvector_needs_refresh
            and dense_retrieval_requested
            and expected_pgvector_indices
            and not expected_pgvector_indices.issubset(postgres_index_names)
        ):
            pgvector_needs_refresh = True
        if dense_retrieval_requested and not pgvector_ready:
            expected_total = expected_documents if expected_documents is not None else current_postgres_count or 0
            ready_count = current_embedding_count or 0
            print(
                "[skip] 11_build_pgvector_index.py waiting for full dense backfill "
                f"({ready_count}/{expected_total} rows ready)."
            )
        _maybe_run_script(
            python_exe,
            "11_build_pgvector_index.py",
            env=command_env,
            summary_path=pgvector_index_summary,
            force=pgvector_needs_refresh,
            should_skip=not dense_retrieval_requested or not pgvector_ready or not expected_pgvector_indices,
        )

        current_opensearch_count = _opensearch_count()
        opensearch_needs_refresh = args.refresh_opensearch
        if not opensearch_needs_refresh and not _summary_is_present(opensearch_summary):
            opensearch_needs_refresh = True
        if not opensearch_needs_refresh and expected_documents is not None and current_opensearch_count != expected_documents:
            opensearch_needs_refresh = True
        _maybe_run_script(
            python_exe,
            "21_bulk_index_opensearch.py",
            env=command_env,
            summary_path=opensearch_summary,
            force=opensearch_needs_refresh,
            should_skip=not opensearch_needs_refresh,
        )
        if args.start_mcp:
            _ensure_mcp_service(build=not args.no_mcp_build, use_gpu=resource_plan.use_gpu)

    elapsed_s = time.perf_counter() - started
    summary_path = _write_summary(
        {
            "repo_root": str(REPO_ROOT),
            "retrieval_profile": args.retrieval_profile,
            "venv_python": str(python_exe),
            "docker_compose": str(COMPOSE_FILE),
            "mcp_docker_compose": str(MCP_COMPOSE_FILE),
            "docker_data_dir": str(_project_env()["CORPUSAGENT2_DOCKER_DATA_DIR"]),
            "data_prepared": not args.skip_data,
            "docker_prepared": not args.skip_docker,
            "api_started": bool(args.start_api and not args.skip_docker),
            "mcp_started": bool(args.start_mcp and not args.skip_docker),
            "resource_plan": resource_plan.env,
            "postgres_summary": str(postgres_ingest_summary) if postgres_ingest_summary.exists() else "",
            "pgvector_backfill_summary": str(pgvector_backfill_summary) if pgvector_backfill_summary.exists() else "",
            "pgvector_index_summary": str(pgvector_index_summary) if pgvector_index_summary.exists() else "",
            "opensearch_summary": str(opensearch_summary) if opensearch_summary.exists() else "",
            "total_elapsed_seconds": round(elapsed_s, 3),
        }
    )
    print("")
    print("VM stack preparation complete.")
    print(f"Total time: {_format_duration(elapsed_s)}")
    print(f"Summary: {summary_path}")
    print(f"Backend start: docker compose -f {COMPOSE_FILE} up -d corpusagent2-api")
    print(f"Frontend config: {REPO_ROOT / 'web' / 'config.js'}")
    print(
        f"Cloudflared helper: {python_exe} {REPO_ROOT / 'scripts' / '23_start_cloudflared_tunnel.py'}"
    )
    if args.start_api and not args.skip_docker:
        print("")
        print("[run] starting Dockerized API backend")
        _ensure_api_service(build=True, use_gpu=False)
    elif args.start_api:
        print("[skip] --start-api ignored because --skip-docker was set.")


if __name__ == "__main__":
    try:
        main()
    except BootstrapError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
