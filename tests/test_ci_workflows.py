from __future__ import annotations

from pathlib import Path


def test_ci_installs_cpu_runtime_and_provider_dependencies() -> None:
    project_root = Path(__file__).resolve().parents[1]
    ci_workflow = (project_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "deploy/requirements.docker-cpu.txt" in ci_workflow
    assert "deploy/requirements.docker-nlp-providers.txt" in ci_workflow
    assert "pip install -e . --no-deps" in ci_workflow
    assert "CORPUSAGENT2_RUN_DOCKER_TESTS: \"0\"" in ci_workflow
    assert "cache-dependency-path" in ci_workflow
