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
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.deploy_resources import (
    compose_files_for_stack,
    compute_docker_resource_plan,
    detect_host_hardware,
)


DEFAULT_PG_DSN = "postgresql://corpus:corpus@127.0.0.1:5432/corpus_db"
DEFAULT_DOCKER_PG_DSN = "postgresql://corpus:corpus@postgres:5432/corpus_db"
DEFAULT_PG_TABLE = "article_corpus"
DEFAULT_OPENSEARCH_URL = "https://127.0.0.1:9200"
DEFAULT_DOCKER_OPENSEARCH_URL = "https://opensearch:9200"
DEFAULT_OPENSEARCH_INDEX = "article-corpus-opensearch"
DEFAULT_OPENSEARCH_USERNAME = "admin"
DEFAULT_OPENSEARCH_PASSWORD = "VerySecurePassword123!"
DEFAULT_DENSE_CONTAINER_PATH = "/models/e5-base-v2"
DEFAULT_SMOKE_SOURCE = REPO_ROOT / "data" / "smoke" / "smoke_10.jsonl"
DOCUMENTS_PARQUET = REPO_ROOT / "data" / "processed" / "documents.parquet"


class StackPrepareError(RuntimeError):
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
        value = value.strip().strip('"').strip("'")
        if key.strip():
            values[key.strip()] = value
    return values


DOTENV_VALUES = _load_dotenv(REPO_ROOT / ".env")


def _env_value(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    value = DOTENV_VALUES.get(name, "").strip()
    if value:
        return value
    return default


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None or not str(value).strip():
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _run(command: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None) -> None:
    print(f"[run] {shlex.join(command)}")
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def _capture(command: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _venv_python() -> Path:
    return REPO_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _docker_compose_command() -> list[str]:
    docker = shutil.which("docker")
    if docker:
        probe = _capture([docker, "compose", "version"])
        if probe.returncode == 0:
            return [docker, "compose"]
    docker_compose = shutil.which("docker-compose")
    if docker_compose:
        return [docker_compose]
    raise StackPrepareError("Docker Compose is not available on PATH.")


def _validate_table_name(table_name: str) -> str:
    normalized = table_name.strip()
    if not normalized or not normalized.replace("_", "").isalnum():
        raise StackPrepareError(f"Unsafe Postgres table name: {table_name!r}")
    return normalized


def _pg_dsn() -> str:
    return _env_value("CORPUSAGENT2_PG_DSN", DEFAULT_PG_DSN)


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


def _postgres_count() -> int | None:
    table_name = _validate_table_name(_env_value("CORPUSAGENT2_PG_TABLE", DEFAULT_PG_TABLE))
    return _postgres_scalar(f"SELECT COUNT(*) FROM {table_name};")


def _postgres_dense_count() -> int | None:
    table_name = _validate_table_name(_env_value("CORPUSAGENT2_PG_TABLE", DEFAULT_PG_TABLE))
    return _postgres_scalar(f"SELECT COUNT(*) FROM {table_name} WHERE dense_embedding IS NOT NULL;")


def _opensearch_request(path_suffix: str) -> dict[str, object]:
    base_url = _env_value("CORPUSAGENT2_OPENSEARCH_URL", DEFAULT_OPENSEARCH_URL).rstrip("/")
    username = _env_value("CORPUSAGENT2_OPENSEARCH_USERNAME", DEFAULT_OPENSEARCH_USERNAME)
    password = _env_value("CORPUSAGENT2_OPENSEARCH_PASSWORD", DEFAULT_OPENSEARCH_PASSWORD)
    verify_ssl = _truthy(_env_value("CORPUSAGENT2_OPENSEARCH_VERIFY_SSL", "false"))
    credentials = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    request = urlrequest.Request(f"{base_url}/{path_suffix.lstrip('/')}")
    request.add_header("Authorization", f"Basic {credentials}")
    context = ssl.create_default_context()
    if not verify_ssl:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    with urlrequest.urlopen(request, timeout=20.0, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def _opensearch_count() -> int | None:
    index_name = _env_value("CORPUSAGENT2_OPENSEARCH_INDEX", DEFAULT_OPENSEARCH_INDEX)
    try:
        payload = _opensearch_request(f"{index_name}/_count")
    except Exception:
        return None
    try:
        return int(payload.get("count", 0))
    except (TypeError, ValueError):
        return None


def _documents_row_count(path: Path = DOCUMENTS_PARQUET) -> int | None:
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def should_restage_database(*, requested_rows: int | None, postgres_count: int | None, force_restage: bool) -> bool:
    if force_restage:
        return True
    if requested_rows is not None:
        return True
    return postgres_count is None or postgres_count <= 0


def _resolve_source_file(explicit: str, requested_rows: int | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if DEFAULT_SMOKE_SOURCE.exists():
        return DEFAULT_SMOKE_SOURCE
    if requested_rows is not None:
        return None
    return None


def _resolve_dense_model_host_path(explicit: str) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = _env_value("CORPUSAGENT2_DENSE_MODEL_HOST_PATH", "")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            REPO_ROOT.parent / "e5-base-v2",
            REPO_ROOT / "models" / "e5-base-v2",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and ((resolved / "modules.json").exists() or (resolved / "config.json").exists()):
            return resolved
    searched = ", ".join(str(path) for path in candidates)
    raise StackPrepareError(
        "Dense retrieval is enabled by default, but no local e5-base-v2 model directory was found. "
        f"Clone or copy it next to the repo, or pass --dense-model-host-path. Checked: {searched}"
    )


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


def _wait_for_opensearch(timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if _opensearch_request(""):
                print("[ready] OpenSearch HTTP endpoint is responding.")
                return
        except (urlerror.URLError, TimeoutError, OSError, json.JSONDecodeError):
            time.sleep(2.0)
    raise TimeoutError("Timed out waiting for OpenSearch to respond.")


def _service_env(*, gpu: str, dense_model_host_path: Path, dense_container_path: str) -> dict[str, str]:
    plan = compute_docker_resource_plan(detect_host_hardware(), gpu_mode=gpu)
    env = os.environ.copy()
    for key, value in DOTENV_VALUES.items():
        env.setdefault(key, value)
    env.update(plan.env)
    env.update(
        {
            "CORPUSAGENT2_PG_DSN": _env_value("CORPUSAGENT2_PG_DSN", DEFAULT_PG_DSN),
            "CORPUSAGENT2_DOCKER_PG_DSN": _env_value("CORPUSAGENT2_DOCKER_PG_DSN", DEFAULT_DOCKER_PG_DSN),
            "CORPUSAGENT2_PG_TABLE": _env_value("CORPUSAGENT2_PG_TABLE", DEFAULT_PG_TABLE),
            "CORPUSAGENT2_OPENSEARCH_URL": _env_value("CORPUSAGENT2_OPENSEARCH_URL", DEFAULT_OPENSEARCH_URL),
            "CORPUSAGENT2_DOCKER_OPENSEARCH_URL": _env_value("CORPUSAGENT2_DOCKER_OPENSEARCH_URL", DEFAULT_DOCKER_OPENSEARCH_URL),
            "CORPUSAGENT2_OPENSEARCH_INDEX": _env_value("CORPUSAGENT2_OPENSEARCH_INDEX", DEFAULT_OPENSEARCH_INDEX),
            "CORPUSAGENT2_OPENSEARCH_USERNAME": _env_value("CORPUSAGENT2_OPENSEARCH_USERNAME", DEFAULT_OPENSEARCH_USERNAME),
            "CORPUSAGENT2_OPENSEARCH_PASSWORD": _env_value("CORPUSAGENT2_OPENSEARCH_PASSWORD", DEFAULT_OPENSEARCH_PASSWORD),
            "CORPUSAGENT2_OPENSEARCH_VERIFY_SSL": "false",
            "CORPUSAGENT2_ENABLE_DENSE_RETRIEVAL": "true",
            "CORPUSAGENT2_RETRIEVAL_BACKEND": "pgvector",
            "CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE": "hybrid",
            "CORPUSAGENT2_PG_INCLUDE_EMBEDDINGS": "true",
            "CORPUSAGENT2_PG_BUILD_IVFFLAT": "true",
            "CORPUSAGENT2_PG_BUILD_HNSW": "false",
            "CORPUSAGENT2_DENSE_MODEL_ID": str(dense_model_host_path),
            "CORPUSAGENT2_DENSE_MODEL_HOST_PATH": str(dense_model_host_path),
            "CORPUSAGENT2_DOCKER_DENSE_MODEL_ID": dense_container_path,
            "CORPUSAGENT2_DOCKER_TORCH_PROFILE": "cuda" if plan.use_gpu else "cpu",
            "CORPUSAGENT2_DEVICE": "cuda" if plan.use_gpu else "cpu",
            "CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS": "false",
            "CORPUSAGENT2_DOCKER_DOWNLOAD_PROVIDER_ASSETS": "false",
            "CORPUSAGENT2_TIME_GRANULARITY": _env_value("CORPUSAGENT2_TIME_GRANULARITY", "year"),
        }
    )
    return env


def _compose_file_args(*, use_gpu: bool) -> list[str]:
    args: list[str] = []
    for file_name in compose_files_for_stack(use_gpu=use_gpu):
        args.extend(["-f", str(DEPLOY_ROOT / file_name)])
    return args


def _start_data_services(env: dict[str, str]) -> None:
    command = _docker_compose_command() + ["-f", str(DEPLOY_ROOT / "docker-compose.yml"), "up", "-d", "--no-recreate", "postgres", "opensearch"]
    _run(command, cwd=REPO_ROOT, env=env)
    parsed_pg = urlparse(env["CORPUSAGENT2_PG_DSN"])
    _wait_for_port(parsed_pg.hostname or "127.0.0.1", int(parsed_pg.port or 5432), timeout_s=180.0, label="Postgres")
    parsed_os = urlparse(env["CORPUSAGENT2_OPENSEARCH_URL"])
    _wait_for_port(parsed_os.hostname or "127.0.0.1", int(parsed_os.port or 9200), timeout_s=240.0, label="OpenSearch TCP")
    _wait_for_opensearch(timeout_s=240.0)


def _build_local_bundle(python_exe: Path, *, source_file: Path, rows: int | None, dense_model_host_path: Path, env: dict[str, str]) -> None:
    if not source_file.exists():
        raise StackPrepareError(f"Source file does not exist: {source_file}")
    bundle_name = f"local_stack_{rows or 'all'}_dense.zip"
    command = [
        str(python_exe),
        str(REPO_ROOT / "scripts" / "27_build_prebuilt_bundle.py"),
        "--clean-existing",
        "--source-file",
        str(source_file),
        "--granularity",
        env.get("CORPUSAGENT2_TIME_GRANULARITY", "year"),
        "--mode",
        "full",
        "--dense-model-id",
        str(dense_model_host_path),
        "--dense-batch-size",
        env.get("CORPUSAGENT2_DENSE_BATCH_SIZE", "4"),
        "--dense-chunk-size",
        env.get("CORPUSAGENT2_DENSE_CHUNK_SIZE", "10"),
        "--skip-nlp",
        "--bundle-path",
        str(REPO_ROOT / "outputs" / "prebuilt" / bundle_name),
    ]
    if rows is not None:
        command.extend(["--hf-max-rows", str(rows)])
    _run(command, env=env)


def _prepare_service_indexes(python_exe: Path, *, restage: bool, env: dict[str, str]) -> None:
    _run([str(python_exe), str(REPO_ROOT / "scripts" / "09_init_postgres_schema.py")], env=env)
    if restage:
        _run([str(python_exe), str(REPO_ROOT / "scripts" / "10_ingest_parquet_to_postgres.py")], env=env)
    else:
        print("[skip] Postgres restage skipped because the service DB already contains rows.")
    postgres_count = _postgres_count()
    dense_count = _postgres_dense_count()
    if postgres_count and dense_count != postgres_count:
        _run([str(python_exe), str(REPO_ROOT / "scripts" / "26_backfill_pgvector_embeddings.py")], env=env)
    _run([str(python_exe), str(REPO_ROOT / "scripts" / "11_build_pgvector_index.py")], env=env)
    expected_rows = postgres_count or _documents_row_count()
    current_os_count = _opensearch_count()
    if DOCUMENTS_PARQUET.exists() and (current_os_count is None or expected_rows is None or current_os_count != expected_rows):
        _run([str(python_exe), str(REPO_ROOT / "scripts" / "21_bulk_index_opensearch.py")], env=env)
    elif not DOCUMENTS_PARQUET.exists():
        print("[warn] OpenSearch indexing skipped because data/processed/documents.parquet is missing.")
    else:
        print("[skip] OpenSearch index already matches the expected row count.")


def _start_api_and_mcp(env: dict[str, str], *, use_gpu: bool) -> None:
    command = (
        _docker_compose_command()
        + _compose_file_args(use_gpu=use_gpu)
        + ["up", "-d", "--build", "--no-deps", "corpusagent2-api", "corpusagent2-mcp"]
    )
    _run(command, cwd=REPO_ROOT, env=env)


def _write_frontend_config(python_exe: Path, env: dict[str, str]) -> None:
    _run([str(python_exe), str(REPO_ROOT / "scripts" / "13_write_frontend_config.py")], env=env)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CorpusAgent2 from one entrypoint: start Docker data services, "
            "restage only when DB is empty or --rows/--force-restage is supplied, then start API/MCP."
        )
    )
    parser.add_argument("--rows", type=int, default=None, help="Restage exactly this many source rows. Omit to reuse a non-empty DB.")
    parser.add_argument("--source-file", default="", help="Local JSONL/JSON/CSV/Parquet source for restaging. Defaults to data/smoke/smoke_10.jsonl when present.")
    parser.add_argument("--force-restage", action="store_true", help="Restage even when Postgres already contains rows.")
    parser.add_argument("--gpu", choices=["auto", "on", "off"], default="auto", help="Default is auto: use CUDA Docker overlay only when NVIDIA is detected.")
    parser.add_argument("--dense-model-host-path", default="", help="Local dense model directory. Defaults to ../e5-base-v2 when present.")
    parser.add_argument("--dense-container-path", default=DEFAULT_DENSE_CONTAINER_PATH, help="Container mount path for the local dense model.")
    parser.add_argument("--skip-api", action="store_true", help="Prepare data services/indexes but do not start API/MCP.")
    parser.add_argument("--skip-frontend-config", action="store_true", help="Do not rewrite web/config.js.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.rows is not None and args.rows <= 0:
        raise StackPrepareError("--rows must be a positive integer when supplied.")
    python_exe = _venv_python()
    if not python_exe.exists():
        raise StackPrepareError(f"Virtualenv not found at {python_exe}. Create it before running this script.")
    dense_model_host_path = _resolve_dense_model_host_path(args.dense_model_host_path)
    resource_plan = compute_docker_resource_plan(detect_host_hardware(), gpu_mode=args.gpu)
    env = _service_env(
        gpu=args.gpu,
        dense_model_host_path=dense_model_host_path,
        dense_container_path=args.dense_container_path,
    )

    print(f"[info] repo={REPO_ROOT}")
    print(f"[info] dense_model_host_path={dense_model_host_path}")
    print(f"[info] docker_torch_profile={env['CORPUSAGENT2_DOCKER_TORCH_PROFILE']}")
    print("[info] provider asset downloads are not run by this script; Stanza paths are not hardcoded.")

    _start_data_services(env)
    postgres_count = _postgres_count()
    restage = should_restage_database(
        requested_rows=args.rows,
        postgres_count=postgres_count,
        force_restage=bool(args.force_restage),
    )
    print(f"[info] postgres_count_before={postgres_count}")
    print(f"[info] restage_required={restage}")
    if restage:
        source_file = _resolve_source_file(args.source_file, args.rows)
        if source_file is None:
            raise StackPrepareError(
                "Restage is required but no source file was found. Pass --source-file or create data/smoke/smoke_10.jsonl."
            )
        _build_local_bundle(
            python_exe,
            source_file=source_file,
            rows=args.rows,
            dense_model_host_path=dense_model_host_path,
            env=env,
        )
    _prepare_service_indexes(python_exe, restage=restage, env=env)
    if not args.skip_frontend_config:
        _write_frontend_config(python_exe, env)
    if not args.skip_api:
        _start_api_and_mcp(env, use_gpu=resource_plan.use_gpu)
    summary = {
        "repo_root": str(REPO_ROOT),
        "postgres_count": _postgres_count(),
        "postgres_dense_count": _postgres_dense_count(),
        "opensearch_count": _opensearch_count(),
        "documents_parquet_rows": _documents_row_count(),
        "restaged": restage,
        "rows_requested": args.rows,
        "gpu_mode": args.gpu,
        "docker_torch_profile": env["CORPUSAGENT2_DOCKER_TORCH_PROFILE"],
        "api_url": f"http://127.0.0.1:{env.get('CORPUSAGENT2_SERVER_PORT', '8001')}",
        "mcp_url": f"http://127.0.0.1:{env.get('CORPUSAGENT2_MCP_PORT', '8765')}{env.get('CORPUSAGENT2_MCP_PATH', '/mcp')}",
    }
    summary_path = REPO_ROOT / "outputs" / "deployment" / "prepare_full_stack_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ready] summary={summary_path}")
    print(f"[ready] API: {summary['api_url']}")
    print(f"[ready] Frontend: run {python_exe} {REPO_ROOT / 'scripts' / '14_run_static_frontend.py'}")


if __name__ == "__main__":
    try:
        main()
    except StackPrepareError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
