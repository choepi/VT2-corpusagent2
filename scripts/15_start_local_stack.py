from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


if __name__ == "__main__":
    frontend_config = subprocess.run(
        [PYTHON, str(REPO_ROOT / "scripts" / "13_write_frontend_config.py")],
        cwd=REPO_ROOT,
        check=True,
    )
    backend = subprocess.Popen(
        [PYTHON, str(REPO_ROOT / "scripts" / "12_run_agent_api.py")],
        cwd=REPO_ROOT,
    )
    frontend = subprocess.Popen(
        [PYTHON, str(REPO_ROOT / "scripts" / "14_run_static_frontend.py")],
        cwd=REPO_ROOT,
    )
    print("Backend:  http://127.0.0.1:8001")
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
