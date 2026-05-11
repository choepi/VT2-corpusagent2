from __future__ import annotations

from pathlib import Path

from corpusagent2.model_config import DEFAULT_DENSE_MODEL_ID, dense_model_id_from_env
from corpusagent2.runtime_context import CorpusRuntime


def test_dense_model_id_defaults_to_huggingface_model(monkeypatch) -> None:
    monkeypatch.delenv("CORPUSAGENT2_DENSE_MODEL_ID", raising=False)

    assert dense_model_id_from_env() == DEFAULT_DENSE_MODEL_ID


def test_dense_model_id_can_point_to_local_model_path(monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_DENSE_MODEL_ID", "C:/models/e5-base-v2")

    assert dense_model_id_from_env() == "C:/models/e5-base-v2"


def test_runtime_uses_configured_dense_model_path(monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_DENSE_MODEL_ID", "C:/models/e5-base-v2")

    runtime = CorpusRuntime.from_project_root(Path(__file__).resolve().parents[1])

    assert runtime.dense_model_id == "C:/models/e5-base-v2"
