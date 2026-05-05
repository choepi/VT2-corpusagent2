from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.app_config import AppConfig


def local_api_base_url() -> str:
    config = AppConfig.from_project_root(REPO_ROOT)
    host = os.getenv("CORPUSAGENT2_SERVER_HOST", config.server.host).strip() or config.server.host
    port = int(os.getenv("CORPUSAGENT2_SERVER_PORT", str(config.server.port)).strip() or str(config.server.port))
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host.strip("[]")
    return f"http://{browser_host}:{port}"


def wait_for_backend_ready(api_base_url: str, process: subprocess.Popen, *, timeout_s: float = 240.0) -> None:
    deadline = time.monotonic() + timeout_s
    runtime_info_url = api_base_url.rstrip("/") + "/runtime-info"
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Backend process exited before becoming ready.")
        try:
            with urlopen(runtime_info_url, timeout=10) as response:
                if 200 <= int(response.status) < 300:
                    return
        except (OSError, URLError) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"Backend did not become ready within {timeout_s:.0f}s. Last error: {last_error}")


if __name__ == "__main__":
    api_base_url = local_api_base_url()
    os.environ["CORPUSAGENT2_FRONTEND_API_BASE_URL"] = api_base_url
    frontend_config = subprocess.run(
        [PYTHON, str(REPO_ROOT / "scripts" / "13_write_frontend_config.py")],
        cwd=REPO_ROOT,
        check=True,
    )
    backend = subprocess.Popen(
        [PYTHON, str(REPO_ROOT / "scripts" / "12_run_agent_api.py")],
        cwd=REPO_ROOT,
    )
    print(f"Waiting for backend runtime info on {api_base_url} ...", flush=True)
    wait_for_backend_ready(api_base_url, backend)
    frontend = subprocess.Popen(
        [PYTHON, str(REPO_ROOT / "scripts" / "14_run_static_frontend.py")],
        cwd=REPO_ROOT,
    )
    print(f"Backend:  {api_base_url}")
    print("Frontend: http://127.0.0.1:5500")
    print("Press Ctrl+C to stop both services.")
    try:
        while True:
            if backend.poll() is not None:
                raise RuntimeError("Backend process exited unexpectedly.")
            if frontend.poll() is not None:
                raise RuntimeError("Frontend process exited unexpectedly.")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        for process in (frontend, backend):
            if process.poll() is None:
                process.terminate()
        for process in (frontend, backend):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
