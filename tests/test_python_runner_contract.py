from __future__ import annotations

import os

import pytest

from corpusagent2.python_runner_service import DockerPythonRunnerService


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
