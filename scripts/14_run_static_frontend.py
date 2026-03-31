from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.app_config import AppConfig, frontend_runtime_payload


if __name__ == "__main__":
    config = AppConfig.from_project_root(REPO_ROOT)
    web_root = REPO_ROOT / "web"
    (web_root / "config.js").write_text(
        "window.CORPUSAGENT2_CONFIG = " + __import__("json").dumps(frontend_runtime_payload(REPO_ROOT), indent=2) + ";\n",
        encoding="utf-8",
    )
    host = "127.0.0.1"
    port = 5500
    handler = partial(SimpleHTTPRequestHandler, directory=str(web_root))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving static frontend on http://{host}:{port}")
    print(f"Using web root {web_root}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
