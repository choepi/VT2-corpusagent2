from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
import os
import re
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


def _escape_control_chars_in_json_strings(text: str) -> str:
    result: list[str] = []
    in_string = False
    escape_next = False
    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == "\\":
            result.append(char)
            escape_next = True
            continue
        if char == '"':
            result.append(char)
            in_string = not in_string
            continue
        if in_string and char in {"\n", "\r", "\t"}:
            result.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[char])
            continue
        result.append(char)
    return "".join(result)


def _extract_json_object(text: str) -> dict[str, Any]:
    def _parse_python_literal(candidate: str) -> dict[str, Any] | None:
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _partial_object_recovery(candidate: str) -> dict[str, Any] | None:
        keys = [
            "action",
            "rewritten_question",
            "clarification_question",
            "rejection_reason",
            "message",
        ]
        recovered: dict[str, Any] = {}
        action_match = re.search(r'["\']action["\']\s*:\s*["\']([^"\']+)["\']', candidate, flags=re.DOTALL)
        if action_match:
            recovered["action"] = action_match.group(1).strip()

        for key in keys[1:]:
            anchored = re.search(
                rf'["\']{re.escape(key)}["\']\s*:\s*["\'](?P<value>.*?)(?=["\']\s*,\s*["\'](?:action|rewritten_question|clarification_question|assumptions|plan_dag|rejection_reason|message)["\']\s*:|["\']\s*\}})',
                candidate,
                flags=re.DOTALL,
            )
            if anchored:
                recovered[key] = anchored.group("value").strip()
                continue
            loose = re.search(
                rf'["\']{re.escape(key)}["\']\s*:\s*["\'](?P<value>.+)$',
                candidate,
                flags=re.DOTALL,
            )
            if loose:
                recovered[key] = loose.group("value").strip().rstrip(",").strip()
        if recovered:
            recovered.setdefault("assumptions", [])
            return recovered
        return None

    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    sanitized = _escape_control_chars_in_json_strings(stripped)
    if sanitized != stripped:
        try:
            parsed = json.loads(sanitized)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidate = stripped[start : end + 1]
        for attempt in (candidate, _escape_control_chars_in_json_strings(candidate)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                literal = _parse_python_literal(attempt)
                if literal is not None:
                    return literal
                continue
    literal = _parse_python_literal(stripped)
    if literal is not None:
        return literal
    partial = _partial_object_recovery(stripped)
    if partial is not None:
        return partial
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

    def with_runtime_overrides(
        self,
        *,
        use_openai: bool | None = None,
        planner_model: str | None = None,
        synthesis_model: str | None = None,
    ) -> "LLMProviderConfig":
        resolved_use_openai = self.use_openai if use_openai is None else bool(use_openai)
        provider_name = "openai" if resolved_use_openai else "uncloseai"
        if resolved_use_openai:
            base_url = os.getenv("CORPUSAGENT2_OPENAI_BASE_URL", self.openai_fallback_base_url()).strip() or self.openai_fallback_base_url()
            api_key = os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("CORPUSAGENT2_LLM_API_KEY", "").strip()
            default_planner = os.getenv("CORPUSAGENT2_OPENAI_PLANNER_MODEL", self.openai_fallback_planner_model()).strip() or self.openai_fallback_planner_model()
            default_synthesis = os.getenv("CORPUSAGENT2_OPENAI_SYNTHESIS_MODEL", default_planner).strip() or default_planner
        else:
            base_url = os.getenv("CORPUSAGENT2_UNCLOSE_BASE_URL", self.unclose_fallback_base_url()).strip() or self.unclose_fallback_base_url()
            api_key = os.getenv("CORPUSAGENT2_LLM_API_KEY", "").strip() or "choose-any-value"
            default_planner = os.getenv("CORPUSAGENT2_UNCLOSE_PLANNER_MODEL", self.unclose_fallback_planner_model()).strip() or self.unclose_fallback_planner_model()
            default_synthesis = os.getenv("CORPUSAGENT2_UNCLOSE_SYNTHESIS_MODEL", default_planner).strip() or default_planner

        return LLMProviderConfig(
            use_openai=resolved_use_openai,
            provider_name=provider_name,
            base_url=base_url,
            api_key=api_key,
            planner_model=(planner_model or default_planner).strip(),
            synthesis_model=(synthesis_model or default_synthesis).strip(),
            timeout_s=self.timeout_s,
            verify_ssl=self.verify_ssl,
            extra_headers=dict(self.extra_headers),
        )

    @staticmethod
    def openai_fallback_base_url() -> str:
        return "https://api.openai.com/v1"

    @staticmethod
    def openai_fallback_planner_model() -> str:
        return "gpt-5.4-2026-03-05"

    @staticmethod
    def unclose_fallback_base_url() -> str:
        return "https://hermes.ai.unturf.com/v1"

    @staticmethod
    def unclose_fallback_planner_model() -> str:
        return "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic"

    @classmethod
    def from_env(cls) -> "LLMProviderConfig":
        raw_toggle = os.getenv("CORPUSAGENT2_USE_OPENAI")
        legacy_provider = os.getenv("CORPUSAGENT2_LLM_PROVIDER", "").strip().lower()
        explicit_provider_specific = any(
            os.getenv(name) is not None
            for name in (
                "CORPUSAGENT2_OPENAI_BASE_URL",
                "CORPUSAGENT2_OPENAI_PLANNER_MODEL",
                "CORPUSAGENT2_OPENAI_SYNTHESIS_MODEL",
                "CORPUSAGENT2_UNCLOSE_BASE_URL",
                "CORPUSAGENT2_UNCLOSE_PLANNER_MODEL",
                "CORPUSAGENT2_UNCLOSE_SYNTHESIS_MODEL",
            )
        )
        legacy_mode = raw_toggle is None and not explicit_provider_specific and bool(legacy_provider)
        use_openai = _truthy(raw_toggle) if raw_toggle is not None else legacy_provider == "openai"
        provider_name = "openai" if use_openai else "uncloseai"

        openai_base_url = os.getenv("CORPUSAGENT2_OPENAI_BASE_URL", cls.openai_fallback_base_url()).strip() or cls.openai_fallback_base_url()
        openai_planner_model = os.getenv("CORPUSAGENT2_OPENAI_PLANNER_MODEL", cls.openai_fallback_planner_model()).strip() or cls.openai_fallback_planner_model()
        openai_synthesis_model = os.getenv("CORPUSAGENT2_OPENAI_SYNTHESIS_MODEL", openai_planner_model).strip() or openai_planner_model

        unclose_base_url = os.getenv("CORPUSAGENT2_UNCLOSE_BASE_URL", cls.unclose_fallback_base_url()).strip() or cls.unclose_fallback_base_url()
        unclose_planner_model = os.getenv(
            "CORPUSAGENT2_UNCLOSE_PLANNER_MODEL",
            cls.unclose_fallback_planner_model(),
        ).strip() or cls.unclose_fallback_planner_model()
        unclose_synthesis_model = os.getenv(
            "CORPUSAGENT2_UNCLOSE_SYNTHESIS_MODEL",
            unclose_planner_model,
        ).strip() or unclose_planner_model

        legacy_base_url = os.getenv("CORPUSAGENT2_LLM_BASE_URL", "").strip()
        legacy_planner_model = os.getenv("CORPUSAGENT2_LLM_PLANNER_MODEL", "").strip()
        legacy_synthesis_model = os.getenv("CORPUSAGENT2_LLM_SYNTHESIS_MODEL", "").strip()
        api_key = os.getenv("CORPUSAGENT2_LLM_API_KEY", "").strip()
        if use_openai:
            api_key = os.getenv("OPENAI_API_KEY", "").strip() or api_key
        elif not api_key:
            api_key = "choose-any-value"

        default_base_url = openai_base_url if use_openai else unclose_base_url
        default_planner_model = openai_planner_model if use_openai else unclose_planner_model
        default_synthesis_model = openai_synthesis_model if use_openai else unclose_synthesis_model
        resolved_base_url = default_base_url
        resolved_planner_model = default_planner_model
        resolved_synthesis_model = default_synthesis_model
        if legacy_mode:
            if legacy_base_url:
                resolved_base_url = legacy_base_url
            if legacy_planner_model:
                resolved_planner_model = legacy_planner_model
            if legacy_synthesis_model:
                resolved_synthesis_model = legacy_synthesis_model

        return cls(
            use_openai=use_openai,
            provider_name=provider_name,
            base_url=resolved_base_url,
            api_key=api_key,
            planner_model=resolved_planner_model,
            synthesis_model=resolved_synthesis_model,
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
