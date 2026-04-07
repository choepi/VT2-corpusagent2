from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Protocol

import httpx


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return text not in {"0", "false", "no", "off"}


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        raise ValueError("LLM response did not contain choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return str(content).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(stripped[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Could not parse JSON object from LLM response: {text[:240]}")


@dataclass(slots=True)
class LLMProviderConfig:
    use_openai: bool = False
    provider_name: str = "uncloseai"
    base_url: str = "https://hermes.ai.unturf.com/v1"
    api_key: str = ""
    planner_model: str = "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic"
    synthesis_model: str = "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic"
    timeout_s: float = 60.0
    verify_ssl: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LLMProviderConfig":
        raw_toggle = os.getenv("CORPUSAGENT2_USE_OPENAI")
        legacy_provider = os.getenv("CORPUSAGENT2_LLM_PROVIDER", "").strip().lower()
        use_openai = _truthy(raw_toggle) if raw_toggle is not None else legacy_provider == "openai"
        provider_name = "openai" if use_openai else "uncloseai"

        openai_base_url = os.getenv("CORPUSAGENT2_OPENAI_BASE_URL", "https://api.openai.com/v1").strip() or "https://api.openai.com/v1"
        openai_planner_model = os.getenv("CORPUSAGENT2_OPENAI_PLANNER_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
        openai_synthesis_model = os.getenv("CORPUSAGENT2_OPENAI_SYNTHESIS_MODEL", openai_planner_model).strip() or openai_planner_model

        unclose_base_url = os.getenv("CORPUSAGENT2_UNCLOSE_BASE_URL", "https://hermes.ai.unturf.com/v1").strip() or "https://hermes.ai.unturf.com/v1"
        unclose_planner_model = os.getenv(
            "CORPUSAGENT2_UNCLOSE_PLANNER_MODEL",
            "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic",
        ).strip() or "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic"
        unclose_synthesis_model = os.getenv(
            "CORPUSAGENT2_UNCLOSE_SYNTHESIS_MODEL",
            unclose_planner_model,
        ).strip() or unclose_planner_model

        api_key = os.getenv("CORPUSAGENT2_LLM_API_KEY", "").strip()
        if use_openai:
            api_key = os.getenv("OPENAI_API_KEY", "").strip() or api_key
        elif not api_key:
            api_key = "choose-any-value"

        default_base_url = openai_base_url if use_openai else unclose_base_url
        default_planner_model = openai_planner_model if use_openai else unclose_planner_model
        default_synthesis_model = openai_synthesis_model if use_openai else unclose_synthesis_model

        return cls(
            use_openai=use_openai,
            provider_name=provider_name,
            base_url=os.getenv("CORPUSAGENT2_LLM_BASE_URL", default_base_url).strip() or default_base_url,
            api_key=api_key,
            planner_model=os.getenv(
                "CORPUSAGENT2_LLM_PLANNER_MODEL",
                default_planner_model,
            ).strip()
            or default_planner_model,
            synthesis_model=os.getenv(
                "CORPUSAGENT2_LLM_SYNTHESIS_MODEL",
                default_synthesis_model,
            ).strip()
            or default_synthesis_model,
            timeout_s=float(os.getenv("CORPUSAGENT2_LLM_TIMEOUT_S", "60").strip() or "60"),
            verify_ssl=os.getenv("CORPUSAGENT2_LLM_VERIFY_SSL", "true").strip().lower()
            not in {"0", "false", "no", "off"},
        )


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
    ) -> str:
        ...

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        ...

    def complete_json_trace(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        ...


class OpenAICompatibleLLMClient:
    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers.update(self.config.extra_headers)
        return headers

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        with httpx.Client(timeout=self.config.timeout_s, verify=self.config.verify_ssl) as client:
            response = client.post(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            return _extract_content(response.json())

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        trace = self.complete_json_trace(messages, model=model, temperature=temperature)
        return dict(trace["parsed_json"])

    def complete_json_trace(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        augmented = list(messages)
        augmented.append(
            {
                "role": "system",
                "content": "Return valid JSON only. Do not wrap the JSON in markdown fences.",
            }
        )
        content = self.complete(augmented, model=model, temperature=temperature)
        return {
            "provider_name": self.config.provider_name,
            "base_url": self.config.base_url,
            "model": model,
            "temperature": temperature,
            "messages": augmented,
            "raw_text": content,
            "parsed_json": _extract_json_object(content),
        }
