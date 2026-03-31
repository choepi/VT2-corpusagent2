from __future__ import annotations

import os
from pathlib import Path

from corpusagent2.app_config import AppConfig, frontend_runtime_payload, load_project_configuration


def test_app_config_loads_defaults_from_project_root() -> None:
    project_root = Path(__file__).resolve().parents[1]

    config = AppConfig.from_project_root(project_root)

    assert config.server.port == 8001
    assert config.frontend.api_base_url == "http://127.0.0.1:8001"
    assert config.env_map["CORPUSAGENT2_LLM_PROVIDER"] == "uncloseai"
    assert config.env_map["CORPUSAGENT2_PG_TABLE"] == "article_corpus"


def test_frontend_runtime_payload_uses_app_config() -> None:
    project_root = Path(__file__).resolve().parents[1]

    payload = frontend_runtime_payload(project_root)

    assert payload["apiBaseUrl"] == "http://127.0.0.1:8001"
    assert "CorpusAgent2" in payload["title"]


def test_load_project_configuration_prefers_dotenv_over_repo_defaults(tmp_path, monkeypatch) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "app_config.toml").write_text(
        """
[llm]
provider = "uncloseai"
base_url = "https://hermes.ai.unturf.com/v1"
        """.strip(),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "CORPUSAGENT2_LLM_PROVIDER=openai\nCORPUSAGENT2_LLM_BASE_URL=https://api.openai.com/v1\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CORPUSAGENT2_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_LLM_BASE_URL", raising=False)

    load_project_configuration(tmp_path)

    assert os.environ["CORPUSAGENT2_LLM_PROVIDER"] == "openai"
    assert os.environ["CORPUSAGENT2_LLM_BASE_URL"] == "https://api.openai.com/v1"


def test_load_project_configuration_prefers_process_env_over_dotenv(tmp_path, monkeypatch) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "app_config.toml").write_text(
        """
[llm]
provider = "uncloseai"
        """.strip(),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("CORPUSAGENT2_LLM_PROVIDER=openai\n", encoding="utf-8")
    monkeypatch.setenv("CORPUSAGENT2_LLM_PROVIDER", "manual-override")

    load_project_configuration(tmp_path)

    assert os.environ["CORPUSAGENT2_LLM_PROVIDER"] == "manual-override"


def test_frontend_runtime_payload_respects_env_override(tmp_path, monkeypatch) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "app_config.toml").write_text(
        """
[frontend]
api_base_url = "http://127.0.0.1:8001"
title = "CorpusAgent2 Prototype"
        """.strip(),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "CORPUSAGENT2_FRONTEND_API_BASE_URL=https://demo.example.com/api\nCORPUSAGENT2_FRONTEND_TITLE=CorpusAgent2 Demo\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CORPUSAGENT2_FRONTEND_API_BASE_URL", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_FRONTEND_TITLE", raising=False)

    payload = frontend_runtime_payload(tmp_path)

    assert payload["apiBaseUrl"] == "https://demo.example.com/api"
    assert payload["title"] == "CorpusAgent2 Demo"
