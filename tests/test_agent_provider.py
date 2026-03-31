from __future__ import annotations

from corpusagent2.llm_provider import LLMProviderConfig


def test_provider_config_reads_openai_compatible_env(monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_LLM_PROVIDER", "openai")
    monkeypatch.setenv("CORPUSAGENT2_LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("CORPUSAGENT2_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CORPUSAGENT2_LLM_PLANNER_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("CORPUSAGENT2_LLM_SYNTHESIS_MODEL", "gpt-4.1-mini")

    config = LLMProviderConfig.from_env()

    assert config.provider_name == "openai"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.api_key == "test-key"
    assert config.planner_model == "gpt-4.1-mini"
    assert config.synthesis_model == "gpt-4.1-mini"


def test_provider_config_defaults_to_uncloseai(monkeypatch) -> None:
    monkeypatch.delenv("CORPUSAGENT2_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_LLM_PLANNER_MODEL", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_LLM_SYNTHESIS_MODEL", raising=False)

    config = LLMProviderConfig.from_env()

    assert config.provider_name == "uncloseai"
    assert config.base_url == "https://hermes.ai.unturf.com/v1"
    assert config.api_key == "choose-any-value"
    assert "Hermes" in config.planner_model
