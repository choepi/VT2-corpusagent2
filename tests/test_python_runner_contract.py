from __future__ import annotations

import os
from pathlib import Path

import pytest

from corpusagent2.python_runner_service import DockerPythonRunnerService


def test_python_runner_reads_container_deployment_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CORPUSAGENT2_PYTHON_RUNNER_IMAGE", "python:3.11-alpine")
    monkeypatch.setenv("CORPUSAGENT2_PYTHON_RUNNER_TIMEOUT_S", "17")
    monkeypatch.setenv("CORPUSAGENT2_PYTHON_RUNNER_CPUS", "0.5")
    monkeypatch.setenv("CORPUSAGENT2_PYTHON_RUNNER_MEMORY", "256m")
    monkeypatch.setenv("CORPUSAGENT2_PYTHON_RUNNER_SHARED_TMP", str(tmp_path))

    runner = DockerPythonRunnerService()

    assert runner.image == "python:3.11-alpine"
    assert runner.timeout_s == 17
    assert runner.cpus == "0.5"
    assert runner.memory == "256m"
    assert runner.shared_tmp_dir == tmp_path.resolve()


@pytest.mark.skipif(
    os.getenv("CORPUSAGENT2_RUN_DOCKER_TESTS", "").strip() != "1",
    reason="Docker sandbox integration test disabled unless CORPUSAGENT2_RUN_DOCKER_TESTS=1",
)
def test_python_runner_returns_artifacts() -> None:
    runner = DockerPythonRunnerService(timeout_s=30)
    code = (
        "from pathlib import Path\n"
        "Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)\n"
        "Path(OUTPUT_DIR, 'artifact.txt').write_text('ok', encoding='utf-8')\n"
        "print('done')\n"
    )

    result = runner.run(code=code, inputs_json={"hello": "world"})

    assert result.exit_code == 0
    assert any(item.name == "artifact.txt" for item in result.artifacts)
