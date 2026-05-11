from __future__ import annotations

import os


DEFAULT_DENSE_MODEL_ID = "intfloat/e5-base-v2"


def dense_model_id_from_env(default: str = DEFAULT_DENSE_MODEL_ID) -> str:
    return os.getenv("CORPUSAGENT2_DENSE_MODEL_ID", default).strip() or default
