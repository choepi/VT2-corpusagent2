from __future__ import annotations

from corpusagent2.llm_provider import LLMProviderConfig, _extract_json_object


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


def test_provider_config_ignores_legacy_openai_base_url_when_unclose_toggle_is_false(monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_USE_OPENAI", "false")
    monkeypatch.setenv("CORPUSAGENT2_LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("CORPUSAGENT2_UNCLOSE_BASE_URL", "https://hermes.ai.unturf.com/v1")
    monkeypatch.delenv("CORPUSAGENT2_LLM_PROVIDER", raising=False)

    config = LLMProviderConfig.from_env()

    assert config.provider_name == "uncloseai"
    assert config.base_url == "https://hermes.ai.unturf.com/v1"


def test_extract_json_object_recovers_from_unescaped_control_chars() -> None:
    raw = '{\n  "action": "emit_plan_dag",\n  "message": "line one\nline two",\n  "plan_dag": {"nodes": [{"node_id": "search", "capability": "db_search"}]}\n}'

    payload = _extract_json_object(raw)

    assert payload["action"] == "emit_plan_dag"
    assert payload["message"] == "line one\nline two"


def test_extract_json_object_recovers_from_python_literal_dict() -> None:
    raw = """{'action': 'ask_clarification', 'rewritten_question': 'How did the media tone change?', 'clarification_question': 'Which outlets should I compare?', 'assumptions': ['Assume major English-language outlets if none are supplied.']}"""

    payload = _extract_json_object(raw)

    assert payload["action"] == "ask_clarification"
    assert payload["clarification_question"] == "Which outlets should I compare?"
