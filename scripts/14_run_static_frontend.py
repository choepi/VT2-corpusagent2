from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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


class NoCacheStaticHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


if __name__ == "__main__":
    os.environ.setdefault("CORPUSAGENT2_FRONTEND_API_BASE_URL", local_api_base_url())
    os.environ.setdefault("CORPUSAGENT2_PREFER_RUNTIME_API_BASE", "1")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "13_write_frontend_config.py")],
        cwd=REPO_ROOT,
        check=True,
    )

    web_root = REPO_ROOT / "web"
    host = "127.0.0.1"
    port = 5500
    handler = partial(NoCacheStaticHandler, directory=str(web_root))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving static frontend on http://{host}:{port}")
    print(f"Using web root {web_root}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
