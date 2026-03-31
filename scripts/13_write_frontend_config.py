from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.app_config import frontend_runtime_payload


if __name__ == "__main__":
    payload = frontend_runtime_payload(REPO_ROOT)
    target = REPO_ROOT / "web" / "config.js"
    target.write_text(
        "window.CORPUSAGENT2_CONFIG = "
        + json.dumps(payload, ensure_ascii=True, indent=2)
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote frontend config to {target}")
