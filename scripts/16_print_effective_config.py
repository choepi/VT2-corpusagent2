from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.app_config import AppConfig, load_project_configuration


if __name__ == "__main__":
    config = load_project_configuration(REPO_ROOT)
    payload = {
        "source_path": config.source_path,
        "server": {
            "host": config.server.host,
            "port": config.server.port,
            "cors_origins": list(config.server.cors_origins),
        },
        "frontend": {
            "api_base_url": config.frontend.api_base_url,
            "title": config.frontend.title,
        },
        "llm": {
            "use_openai": config.llm.use_openai,
            "openai_base_url": config.llm.openai_base_url,
            "openai_planner_model": config.llm.openai_planner_model,
            "openai_synthesis_model": config.llm.openai_synthesis_model,
            "unclose_base_url": config.llm.unclose_base_url,
            "unclose_planner_model": config.llm.unclose_planner_model,
            "unclose_synthesis_model": config.llm.unclose_synthesis_model,
            "timeout_s": config.llm.timeout_s,
            "verify_ssl": config.llm.verify_ssl,
        },
        "effective_env": {
            key: os.environ.get(key, "")
            for key in sorted(config.env_map.keys())
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
