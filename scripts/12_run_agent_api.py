from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.api import build_app
from corpusagent2.app_config import load_project_configuration
from corpusagent2.agent_runtime import AgentRuntime, AgentRuntimeConfig


if __name__ == "__main__":
    app_config = load_project_configuration(REPO_ROOT)
    host = app_config.server.host
    port = app_config.server.port
    runtime = AgentRuntime(config=AgentRuntimeConfig.from_project_root(REPO_ROOT))
    app = build_app(runtime=runtime, project_root=REPO_ROOT)
    uvicorn.run(app, host=host, port=port)
