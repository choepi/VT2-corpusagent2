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
