from __future__ import annotations

from corpusagent2.llm_provider import LLMProviderConfig


def test_provider_config_reads_openai_toggle_and_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_USE_OPENAI", "true")
    monkeypatch.setenv("CORPUSAGENT2_OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CORPUSAGENT2_OPENAI_PLANNER_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("CORPUSAGENT2_OPENAI_SYNTHESIS_MODEL", "gpt-4.1-mini")
    monkeypatch.delenv("CORPUSAGENT2_LLM_API_KEY", raising=False)

    config = LLMProviderConfig.from_env()

    assert config.use_openai is True
    assert config.provider_name == "openai"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.api_key == "test-key"
    assert config.planner_model == "gpt-4.1-mini"
    assert config.synthesis_model == "gpt-4.1-mini"


def test_provider_config_defaults_to_uncloseai(monkeypatch) -> None:
    monkeypatch.delenv("CORPUSAGENT2_USE_OPENAI", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_UNCLOSE_BASE_URL", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_UNCLOSE_PLANNER_MODEL", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_UNCLOSE_SYNTHESIS_MODEL", raising=False)

    config = LLMProviderConfig.from_env()

    assert config.use_openai is False
    assert config.provider_name == "uncloseai"
    assert config.base_url == "https://hermes.ai.unturf.com/v1"
    assert config.api_key == "choose-any-value"
    assert "Hermes" in config.planner_model
