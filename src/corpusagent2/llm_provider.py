from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Protocol

import httpx


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
        provider_name = os.getenv("CORPUSAGENT2_LLM_PROVIDER", "uncloseai").strip() or "uncloseai"
        api_key = os.getenv("CORPUSAGENT2_LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
        if not api_key and provider_name == "uncloseai":
            api_key = "choose-any-value"
        return cls(
            provider_name=provider_name,
            base_url=os.getenv("CORPUSAGENT2_LLM_BASE_URL", "https://hermes.ai.unturf.com/v1").strip() or "https://hermes.ai.unturf.com/v1",
            api_key=api_key,
            planner_model=os.getenv(
                "CORPUSAGENT2_LLM_PLANNER_MODEL",
                "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic",
            ).strip()
            or "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic",
            synthesis_model=os.getenv(
                "CORPUSAGENT2_LLM_SYNTHESIS_MODEL",
                "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic",
            ).strip()
            or "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic",
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
        augmented = list(messages)
        augmented.append(
            {
                "role": "system",
                "content": "Return valid JSON only. Do not wrap the JSON in markdown fences.",
            }
        )
        content = self.complete(augmented, model=model, temperature=temperature)
        return _extract_json_object(content)
