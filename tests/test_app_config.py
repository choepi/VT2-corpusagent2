from __future__ import annotations

import os
from pathlib import Path

from corpusagent2.app_config import AppConfig, frontend_runtime_payload, load_project_configuration


def test_app_config_loads_defaults_from_project_root() -> None:
    project_root = Path(__file__).resolve().parents[1]

    config = AppConfig.from_project_root(project_root)

    assert config.server.port == 8001
    assert config.frontend.api_base_url == "https://api.dongtse.com"
    assert config.llm.use_openai is True
    assert config.env_map["CORPUSAGENT2_USE_OPENAI"] == "True"
    assert config.env_map["CORPUSAGENT2_PG_TABLE"] == "article_corpus"


def test_frontend_runtime_payload_uses_app_config() -> None:
    project_root = Path(__file__).resolve().parents[1]

    payload = frontend_runtime_payload(project_root)

    assert payload["apiBaseUrl"] == "https://api.dongtse.com"
    assert "Corpusagent v2" in payload["title"]
    assert payload["useOpenAI"] is True


def test_load_project_configuration_prefers_dotenv_over_repo_defaults(tmp_path, monkeypatch) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "app_config.toml").write_text(
        """
[llm]
use_openai = false
unclose_base_url = "https://hermes.ai.unturf.com/v1"
        """.strip(),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "CORPUSAGENT2_USE_OPENAI=true\nCORPUSAGENT2_OPENAI_BASE_URL=https://api.openai.com/v1\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CORPUSAGENT2_USE_OPENAI", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_OPENAI_BASE_URL", raising=False)

    load_project_configuration(tmp_path)

    assert os.environ["CORPUSAGENT2_USE_OPENAI"] == "true"
    assert os.environ["CORPUSAGENT2_OPENAI_BASE_URL"] == "https://api.openai.com/v1"


def test_load_project_configuration_prefers_process_env_over_dotenv(tmp_path, monkeypatch) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "app_config.toml").write_text(
        """
[llm]
use_openai = false
        """.strip(),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("CORPUSAGENT2_USE_OPENAI=true\n", encoding="utf-8")
    monkeypatch.setenv("CORPUSAGENT2_USE_OPENAI", "manual-override")

    load_project_configuration(tmp_path)

    assert os.environ["CORPUSAGENT2_USE_OPENAI"] == "manual-override"


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


def test_frontend_runtime_payload_includes_access_gate(tmp_path, monkeypatch) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "app_config.toml").write_text(
        """
[frontend]
api_base_url = "https://api.dongtse.com"
title = "CorpusAgent2 Prototype"
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("CORPUSAGENT2_FRONTEND_ACCESS_GATE", "true")
    monkeypatch.setenv("CORPUSAGENT2_FRONTEND_ACCESS_PASSWORD_SHA256", "abc123")
    monkeypatch.setenv("CORPUSAGENT2_FRONTEND_ACCESS_TITLE", "Private Demo Access")
    monkeypatch.setenv("CORPUSAGENT2_FRONTEND_ACCESS_SUBTITLE", "Enter the shared passphrase to continue.")

    payload = frontend_runtime_payload(tmp_path)

    assert payload["accessGate"]["enabled"] is True
    assert payload["accessGate"]["passwordSha256"] == "abc123"
    assert payload["accessGate"]["title"] == "Private Demo Access"
    assert payload["accessGate"]["subtitle"] == "Enter the shared passphrase to continue."
