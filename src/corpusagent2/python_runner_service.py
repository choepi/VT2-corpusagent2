from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


@dataclass(slots=True)
class SandboxArtifact:
    name: str
    mime: str
    bytes_b64: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class PythonRunnerResult:
    stdout: str
    stderr: str
    artifacts: list[SandboxArtifact]
    exit_code: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "exit_code": self.exit_code,
        }


class DockerPythonRunnerService:
    def __init__(
        self,
        image: str | None = None,
        *,
        timeout_s: int | None = None,
        cpus: str | None = None,
        memory: str | None = None,
        shared_tmp_dir: str | Path | None = None,
    ) -> None:
        self.image = image or os.getenv("CORPUSAGENT2_PYTHON_RUNNER_IMAGE", "python:3.11-slim")
        self.timeout_s = int(timeout_s or os.getenv("CORPUSAGENT2_PYTHON_RUNNER_TIMEOUT_S", "60"))
        self.cpus = cpus or os.getenv("CORPUSAGENT2_PYTHON_RUNNER_CPUS", "1")
        self.memory = memory or os.getenv("CORPUSAGENT2_PYTHON_RUNNER_MEMORY", "512m")
        raw_shared_tmp = shared_tmp_dir or os.getenv("CORPUSAGENT2_PYTHON_RUNNER_SHARED_TMP", "")
        self.shared_tmp_dir = Path(raw_shared_tmp).resolve() if raw_shared_tmp else None

    def run(self, code: str, inputs_json: dict[str, Any]) -> PythonRunnerResult:
        if self.shared_tmp_dir is not None:
            self.shared_tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ca2_py_runner_", dir=str(self.shared_tmp_dir) if self.shared_tmp_dir else None) as temp_dir:
            temp_root = Path(temp_dir)
            workspace = temp_root / "workspace"
            outputs = temp_root / "outputs"
            workspace.mkdir(parents=True, exist_ok=True)
            outputs.mkdir(parents=True, exist_ok=True)
            workspace.chmod(0o755)
            outputs.chmod(0o777)

            code_path = workspace / "code.py"
            inputs_path = workspace / "inputs.json"
            driver_path = workspace / "driver.py"

            code_path.write_text(code, encoding="utf-8")
            inputs_path.write_text(json.dumps(inputs_json, ensure_ascii=True), encoding="utf-8")
            driver_path.write_text(
                """
from pathlib import Path
import json
import runpy
import sys

workspace = Path('/workspace')
outputs = Path('/outputs')
inputs_path = workspace / 'inputs.json'
code_path = workspace / 'code.py'

payload = json.loads(inputs_path.read_text(encoding='utf-8'))
globals_dict = {
    'INPUTS_JSON': payload,
    'OUTPUT_DIR': str(outputs),
}
runpy.run_path(str(code_path), init_globals=globals_dict, run_name='__main__')
""".strip(),
                encoding="utf-8",
            )

            command = [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--read-only",
                "--cpus",
                self.cpus,
                "--memory",
                self.memory,
                "--user",
                "65534:65534",
                "--mount",
                f"type=bind,src={workspace},dst=/workspace,readonly",
                "--mount",
                f"type=bind,src={outputs},dst=/outputs",
                self.image,
                "python",
                "/workspace/driver.py",
            ]

            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )

            artifacts: list[SandboxArtifact] = []
            for path in sorted(outputs.iterdir()):
                if not path.is_file():
                    continue
                mime = "application/octet-stream"
                suffix = path.suffix.lower()
                if suffix == ".png":
                    mime = "image/png"
                elif suffix in {".jpg", ".jpeg"}:
                    mime = "image/jpeg"
                elif suffix == ".json":
                    mime = "application/json"
                elif suffix == ".txt":
                    mime = "text/plain"
                artifacts.append(
                    SandboxArtifact(
                        name=path.name,
                        mime=mime,
                        bytes_b64=base64.b64encode(path.read_bytes()).decode("ascii"),
                    )
                )

            return PythonRunnerResult(
                stdout=completed.stdout,
                stderr=completed.stderr,
                artifacts=artifacts,
                exit_code=int(completed.returncode),
            )
