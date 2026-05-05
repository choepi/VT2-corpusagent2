from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_start_local_stack_module():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "15_start_local_stack.py"
    spec = importlib.util.spec_from_file_location("start_local_stack", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_stack_points_frontend_at_local_backend(monkeypatch) -> None:
    module = _load_start_local_stack_module()
    monkeypatch.delenv("CORPUSAGENT2_SERVER_HOST", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_SERVER_PORT", raising=False)

    assert module.local_api_base_url() == "http://127.0.0.1:8001"


def test_local_stack_uses_browser_reachable_host_for_wildcard_bind(monkeypatch) -> None:
    module = _load_start_local_stack_module()
    monkeypatch.setenv("CORPUSAGENT2_SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("CORPUSAGENT2_SERVER_PORT", "9000")

    assert module.local_api_base_url() == "http://127.0.0.1:9000"


def test_wait_for_backend_ready_polls_runtime_info(monkeypatch) -> None:
    module = _load_start_local_stack_module()
    requested_urls: list[str] = []

    class FakeProcess:
        def poll(self):
            return None

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_urlopen(url, timeout):
        requested_urls.append(str(url))
        assert timeout == 10
        return FakeResponse()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    module.wait_for_backend_ready("http://127.0.0.1:8001", FakeProcess(), timeout_s=1)

    assert requested_urls == ["http://127.0.0.1:8001/runtime-info"]


def test_mcp_compose_defaults_to_cpu_and_keeps_gpu_override_separate() -> None:
    project_root = Path(__file__).resolve().parents[1]
    base_compose = (project_root / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")
    mcp_compose = (project_root / "deploy" / "docker-compose.mcp.yml").read_text(encoding="utf-8")
    gpu_compose = (project_root / "deploy" / "docker-compose.mcp.gpu.yml").read_text(encoding="utf-8")

    assert "corpusagent2-api:" in base_compose
    assert "dockerfile: deploy/Dockerfile" in base_compose
    assert "dockerfile: deploy/Dockerfile" in mcp_compose
    assert "Dockerfile.mcp" not in mcp_compose
    assert "CORPUSAGENT2_DEVICE: ${CORPUSAGENT2_DEVICE:-cpu}" in mcp_compose
    assert "gpus: all" not in mcp_compose
    assert "corpusagent2-api:" in gpu_compose
    assert "CORPUSAGENT2_DEVICE: ${CORPUSAGENT2_DEVICE:-cuda}" in gpu_compose
    assert "gpus: all" in gpu_compose


def test_dockerized_api_and_mcp_share_runtime_image_and_sandbox_mounts() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runtime_dockerfile = project_root / "deploy" / "Dockerfile"
    base_compose = (project_root / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")
    mcp_compose = (project_root / "deploy" / "docker-compose.mcp.yml").read_text(encoding="utf-8")
    dockerfile_text = runtime_dockerfile.read_text(encoding="utf-8")

    assert runtime_dockerfile.exists()
    assert not (project_root / "deploy" / "Dockerfile.mcp").exists()
    assert "ARG CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS=false" in dockerfile_text
    assert "ARG CORPUSAGENT2_DOCKER_DOWNLOAD_PROVIDER_ASSETS=false" in dockerfile_text
    assert "uv sync --frozen --no-install-project" in dockerfile_text
    assert "uv sync --frozen --extra nlp-providers --no-install-project" in dockerfile_text
    assert "CMD [\"python\", \"/app/scripts/12_run_agent_api.py\"]" in dockerfile_text
    assert "image: corpusagent2-runtime:latest" in base_compose
    assert "image: corpusagent2-runtime:latest" in mcp_compose
    assert "CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS: ${CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS:-false}" in base_compose
    assert "CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS: ${CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS:-false}" in mcp_compose
    assert "CORPUSAGENT2_API_CPUS" in base_compose
    assert "CORPUSAGENT2_MCP_CPUS" in mcp_compose
    assert "CORPUSAGENT2_OPENSEARCH_MEM_LIMIT" not in base_compose
    assert "CORPUSAGENT2_POSTGRES_MEM_LIMIT" not in base_compose
    assert "/var/run/docker.sock:/var/run/docker.sock" in base_compose
    assert "/var/run/docker.sock:/var/run/docker.sock" in mcp_compose
    assert "CORPUSAGENT2_PYTHON_RUNNER_SHARED_TMP" in base_compose
    assert "CORPUSAGENT2_PYTHON_RUNNER_SHARED_TMP" in mcp_compose


def test_vm_services_are_docker_stack_first() -> None:
    project_root = Path(__file__).resolve().parents[1]
    configure_script = (project_root / "scripts" / "24_configure_vm_services.py").read_text(encoding="utf-8")
    stack_service = (project_root / "deploy" / "corpusagent2-stack.service").read_text(encoding="utf-8")
    tunnel_service = (project_root / "deploy" / "corpusagent2-cloudflared.service").read_text(encoding="utf-8")

    assert not (project_root / "deploy" / "corpusagent2-api.service").exists()
    assert "up -d --no-recreate postgres opensearch" in stack_service
    assert "up -d --build --no-deps corpusagent2-api corpusagent2-mcp" in stack_service
    assert "down --remove-orphans" not in stack_service
    assert "stop corpusagent2-api corpusagent2-mcp" in stack_service
    assert "EnvironmentFile=/home/dongtten/corpusagent2/.env" in stack_service
    assert "corpusagent2-stack.service" in tunnel_service
    assert "scripts/12_run_agent_api.py" not in stack_service
    assert "compute_docker_resource_plan(detect_host_hardware(), gpu_mode=args.gpu)" in configure_script
    assert "--gpu" in configure_script
    assert "up -d --no-recreate postgres opensearch" in configure_script
    assert "up -d --build --no-deps corpusagent2-api corpusagent2-mcp" in configure_script
    assert "down --remove-orphans" not in configure_script
    assert "scripts' / '12_run_agent_api.py" not in configure_script
