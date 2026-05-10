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


def _load_static_frontend_module():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "14_run_static_frontend.py"
    spec = importlib.util.spec_from_file_location("run_static_frontend", script_path)
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


def test_static_frontend_defaults_to_local_backend_for_manual_launch(monkeypatch) -> None:
    module = _load_static_frontend_module()
    monkeypatch.delenv("CORPUSAGENT2_FRONTEND_API_BASE_URL", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_SERVER_HOST", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_SERVER_PORT", raising=False)

    payload = module.static_frontend_runtime_payload()

    assert payload["apiBaseUrl"] == "http://127.0.0.1:8001"
    assert payload["preferRuntimeApiBase"] is True


def test_static_frontend_honors_explicit_api_base_override(monkeypatch) -> None:
    module = _load_static_frontend_module()
    monkeypatch.setenv("CORPUSAGENT2_FRONTEND_API_BASE_URL", "https://demo.example.com/api")

    payload = module.static_frontend_runtime_payload()

    assert payload["apiBaseUrl"] == "https://demo.example.com/api"
    assert payload["preferRuntimeApiBase"] is True


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
    assert "ARG CORPUSAGENT2_DOCKER_TORCH_PROFILE=cpu" in dockerfile_text
    assert "ARG CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS=false" in dockerfile_text
    assert "ARG CORPUSAGENT2_DOCKER_DOWNLOAD_PROVIDER_ASSETS=false" in dockerfile_text
    assert "deploy/requirements.docker-cpu.txt" in dockerfile_text
    assert "deploy/requirements.docker-nlp-providers.txt" in dockerfile_text
    assert "RUN set -eu;" in dockerfile_text
    assert "--index-url https://pypi.org/simple" in dockerfile_text
    assert "--torch-backend cpu" not in dockerfile_text
    assert "https://download.pytorch.org/whl/cpu" not in dockerfile_text
    assert "Unsupported CORPUSAGENT2_DOCKER_TORCH_PROFILE" in dockerfile_text
    assert "PYTHONPATH=/app/src" in dockerfile_text
    assert "CMD [\"python\", \"/app/scripts/12_run_agent_api.py\"]" in dockerfile_text
    assert "image: corpusagent2-runtime:latest" in base_compose
    assert "image: corpusagent2-runtime:latest" in mcp_compose
    assert "CORPUSAGENT2_DOCKER_TORCH_PROFILE: ${CORPUSAGENT2_DOCKER_TORCH_PROFILE:-cpu}" in base_compose
    assert "CORPUSAGENT2_DOCKER_TORCH_PROFILE: ${CORPUSAGENT2_DOCKER_TORCH_PROFILE:-cpu}" in mcp_compose
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


def test_docker_cpu_requirements_use_cpu_torch_and_real_provider_stack() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cpu_requirements = (project_root / "deploy" / "requirements.docker-cpu.txt").read_text(encoding="utf-8")
    provider_requirements = (project_root / "deploy" / "requirements.docker-nlp-providers.txt").read_text(encoding="utf-8")
    gpu_compose = (project_root / "deploy" / "docker-compose.mcp.gpu.yml").read_text(encoding="utf-8")

    assert "torch==2.3.1" in cpu_requirements
    assert "torchvision==0.18.1" in cpu_requirements
    assert "torchaudio==2.3.1" in cpu_requirements
    assert "torch==2.3.1+cpu" not in cpu_requirements
    assert "cu118" not in cpu_requirements
    assert "cu118" not in provider_requirements
    assert "download.pytorch.org" not in cpu_requirements
    assert "sentence-transformers" in cpu_requirements
    assert "bertopic" in cpu_requirements
    assert "flair" in provider_requirements
    assert "stanza" in provider_requirements
    assert "textacy" in provider_requirements
    assert "CORPUSAGENT2_DOCKER_TORCH_PROFILE: ${CORPUSAGENT2_DOCKER_TORCH_PROFILE:-cuda}" in gpu_compose
    assert "CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS: ${CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS:-true}" in gpu_compose


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


def test_vm_bootstrap_guidance_does_not_recreate_data_services() -> None:
    project_root = Path(__file__).resolve().parents[1]
    bootstrap_script = (project_root / "scripts" / "bootstrap_ubuntu_vm.sh").read_text(encoding="utf-8")
    prepare_script = (project_root / "scripts" / "22_prepare_vm_stack.py").read_text(encoding="utf-8")

    assert "up -d --no-recreate postgres opensearch" in bootstrap_script
    assert "up -d --build --no-deps corpusagent2-api corpusagent2-mcp" in bootstrap_script
    assert "up -d corpusagent2-api" not in bootstrap_script
    assert "up -d --no-recreate postgres opensearch" in prepare_script
    assert "up -d --build --no-deps corpusagent2-api corpusagent2-mcp" in prepare_script
    assert "_ensure_api_service(build=True, use_gpu=resource_plan.use_gpu)" in prepare_script
    assert "docker compose -f {COMPOSE_FILE} up -d corpusagent2-api" not in prepare_script
