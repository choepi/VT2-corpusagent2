from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib
from typing import Any


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)

@dataclass(slots=True)
class FrontendConfig:
    api_base_url: str = "http://127.0.0.1:8001"
    title: str = "CorpusAgent2 Prototype"


@dataclass(slots=True)
class LLMAppConfig:
    use_openai: bool = False
    openai_base_url: str = "https://api.openai.com/v1"
    openai_planner_model: str = "gpt-4.1-mini"
    openai_synthesis_model: str = "gpt-4.1-mini"
    unclose_base_url: str = "https://hermes.ai.unturf.com/v1"
    unclose_planner_model: str = "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic"
    unclose_synthesis_model: str = "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic"
    timeout_s: float = 60.0
    verify_ssl: bool = True


@dataclass(slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8001
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass(slots=True)
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    llm: LLMAppConfig = field(default_factory=LLMAppConfig)
    env_map: dict[str, str] = field(default_factory=dict)
    source_path: str = ""

    @classmethod
    def from_project_root(cls, project_root: Path) -> "AppConfig":
        config_path = project_root / "config" / "app_config.toml"
        raw = _read_toml(config_path)
        server_payload = raw.get("server", {})
        frontend_payload = raw.get("frontend", {})
        llm_payload = raw.get("llm", {})
        env_map = cls._env_map_from_payload(raw)
        return cls(
            server=ServerConfig(
                host=str(server_payload.get("host", "127.0.0.1")),
                port=int(server_payload.get("port", 8001)),
                cors_origins=[
                    str(item).strip()
                    for item in server_payload.get("cors_origins", ["*"])
                    if str(item).strip()
                ]
                or ["*"],
            ),
            frontend=FrontendConfig(
                api_base_url=str(frontend_payload.get("api_base_url", "http://127.0.0.1:8001")),
                title=str(frontend_payload.get("title", "CorpusAgent2 Prototype")),
            ),
            llm=LLMAppConfig(
                use_openai=str(llm_payload.get("use_openai", "false")).strip().lower() in {"1", "true", "yes", "on"},
                openai_base_url=str(llm_payload.get("openai_base_url", "https://api.openai.com/v1")),
                openai_planner_model=str(llm_payload.get("openai_planner_model", "gpt-4.1-mini")),
                openai_synthesis_model=str(llm_payload.get("openai_synthesis_model", "gpt-4.1-mini")),
                unclose_base_url=str(llm_payload.get("unclose_base_url", "https://hermes.ai.unturf.com/v1")),
                unclose_planner_model=str(
                    llm_payload.get(
                        "unclose_planner_model",
                        "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic",
                    )
                ),
                unclose_synthesis_model=str(
                    llm_payload.get(
                        "unclose_synthesis_model",
                        "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic",
                    )
                ),
                timeout_s=float(llm_payload.get("timeout_s", 60)),
                verify_ssl=str(llm_payload.get("verify_ssl", "true")).strip().lower()
                not in {"0", "false", "no", "off"},
            ),
            env_map=env_map,
            source_path=str(config_path),
        )

    @staticmethod
    def _env_map_from_payload(raw: dict[str, Any]) -> dict[str, str]:
        env_map: dict[str, str] = {}
        mappings = [
            ("server", "host", "CORPUSAGENT2_SERVER_HOST"),
            ("server", "port", "CORPUSAGENT2_SERVER_PORT"),
            ("frontend", "api_base_url", "CORPUSAGENT2_FRONTEND_API_BASE_URL"),
            ("frontend", "title", "CORPUSAGENT2_FRONTEND_TITLE"),
            ("llm", "use_openai", "CORPUSAGENT2_USE_OPENAI"),
            ("llm", "openai_base_url", "CORPUSAGENT2_OPENAI_BASE_URL"),
            ("llm", "openai_planner_model", "CORPUSAGENT2_OPENAI_PLANNER_MODEL"),
            ("llm", "openai_synthesis_model", "CORPUSAGENT2_OPENAI_SYNTHESIS_MODEL"),
            ("llm", "unclose_base_url", "CORPUSAGENT2_UNCLOSE_BASE_URL"),
            ("llm", "unclose_planner_model", "CORPUSAGENT2_UNCLOSE_PLANNER_MODEL"),
            ("llm", "unclose_synthesis_model", "CORPUSAGENT2_UNCLOSE_SYNTHESIS_MODEL"),
            ("llm", "timeout_s", "CORPUSAGENT2_LLM_TIMEOUT_S"),
            ("llm", "verify_ssl", "CORPUSAGENT2_LLM_VERIFY_SSL"),
            ("retrieval", "backend", "CORPUSAGENT2_RETRIEVAL_BACKEND"),
            ("retrieval", "default_mode", "CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE"),
            ("retrieval", "use_rerank", "CORPUSAGENT2_RETRIEVAL_USE_RERANK"),
            ("retrieval", "rerank_top_k", "CORPUSAGENT2_RETRIEVAL_RERANK_TOP_K"),
            ("retrieval", "fusion_k", "CORPUSAGENT2_RETRIEVAL_FUSION_K"),
            ("retrieval", "require_backend_services", "CORPUSAGENT2_REQUIRE_BACKEND_SERVICES"),
            ("retrieval", "allow_local_fallback", "CORPUSAGENT2_ALLOW_LOCAL_FALLBACK"),
            ("retrieval", "default_time_granularity", "CORPUSAGENT2_TIME_GRANULARITY"),
            ("runtime", "device", "CORPUSAGENT2_DEVICE"),
            ("postgres", "dsn", "CORPUSAGENT2_PG_DSN"),
            ("postgres", "table", "CORPUSAGENT2_PG_TABLE"),
            ("opensearch", "url", "CORPUSAGENT2_OPENSEARCH_URL"),
            ("opensearch", "index", "CORPUSAGENT2_OPENSEARCH_INDEX"),
            ("opensearch", "username", "CORPUSAGENT2_OPENSEARCH_USERNAME"),
            ("opensearch", "password", "CORPUSAGENT2_OPENSEARCH_PASSWORD"),
            ("opensearch", "verify_ssl", "CORPUSAGENT2_OPENSEARCH_VERIFY_SSL"),
            ("opensearch", "timeout_s", "CORPUSAGENT2_OPENSEARCH_TIMEOUT_S"),
        ]
        for section, key, env_name in mappings:
            section_payload = raw.get(section, {})
            if key in section_payload:
                env_map[env_name] = str(section_payload[key])

        cors_origins = raw.get("server", {}).get("cors_origins", [])
        if cors_origins:
            env_map["CORPUSAGENT2_CORS_ORIGINS"] = ",".join(str(item).strip() for item in cors_origins if str(item).strip())

        provider_order = raw.get("provider_order", {})
        for capability, providers in provider_order.items():
            env_name = f"CORPUSAGENT2_PROVIDER_ORDER_{str(capability).upper()}"
            env_map[env_name] = ",".join(str(item).strip() for item in providers if str(item).strip())
        return env_map

    def apply_to_environ(self) -> None:
        for key, value in self.env_map.items():
            if key not in os.environ:
                os.environ[key] = value


def load_project_configuration(project_root: Path) -> AppConfig:
    preexisting = set(os.environ.keys())
    config = AppConfig.from_project_root(project_root)
    for key, value in config.env_map.items():
        if key not in preexisting:
            os.environ[key] = value
    env_path = project_root / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in preexisting:
                os.environ[key] = value
    config.server.host = os.getenv("CORPUSAGENT2_SERVER_HOST", config.server.host).strip() or config.server.host
    config.server.port = int(os.getenv("CORPUSAGENT2_SERVER_PORT", str(config.server.port)).strip() or str(config.server.port))
    cors_raw = os.getenv("CORPUSAGENT2_CORS_ORIGINS", "").strip()
    if cors_raw:
        config.server.cors_origins = [item.strip() for item in cors_raw.split(",") if item.strip()] or ["*"]
    config.frontend.api_base_url = os.getenv("CORPUSAGENT2_FRONTEND_API_BASE_URL", config.frontend.api_base_url).strip() or config.frontend.api_base_url
    config.frontend.title = os.getenv("CORPUSAGENT2_FRONTEND_TITLE", config.frontend.title).strip() or config.frontend.title
    raw_toggle = os.getenv("CORPUSAGENT2_USE_OPENAI")
    if raw_toggle is not None:
        config.llm.use_openai = raw_toggle.strip().lower() in {"1", "true", "yes", "on"}
    config.llm.openai_base_url = os.getenv("CORPUSAGENT2_OPENAI_BASE_URL", config.llm.openai_base_url).strip() or config.llm.openai_base_url
    config.llm.openai_planner_model = os.getenv("CORPUSAGENT2_OPENAI_PLANNER_MODEL", config.llm.openai_planner_model).strip() or config.llm.openai_planner_model
    config.llm.openai_synthesis_model = os.getenv("CORPUSAGENT2_OPENAI_SYNTHESIS_MODEL", config.llm.openai_synthesis_model).strip() or config.llm.openai_synthesis_model
    config.llm.unclose_base_url = os.getenv("CORPUSAGENT2_UNCLOSE_BASE_URL", config.llm.unclose_base_url).strip() or config.llm.unclose_base_url
    config.llm.unclose_planner_model = os.getenv("CORPUSAGENT2_UNCLOSE_PLANNER_MODEL", config.llm.unclose_planner_model).strip() or config.llm.unclose_planner_model
    config.llm.unclose_synthesis_model = os.getenv("CORPUSAGENT2_UNCLOSE_SYNTHESIS_MODEL", config.llm.unclose_synthesis_model).strip() or config.llm.unclose_synthesis_model
    config.llm.timeout_s = float(os.getenv("CORPUSAGENT2_LLM_TIMEOUT_S", str(config.llm.timeout_s)).strip() or str(config.llm.timeout_s))
    config.llm.verify_ssl = os.getenv("CORPUSAGENT2_LLM_VERIFY_SSL", str(config.llm.verify_ssl)).strip().lower() not in {"0", "false", "no", "off"}
    return config


def frontend_runtime_payload(project_root: Path) -> dict[str, Any]:
    config = load_project_configuration(project_root)
    prefer_runtime = os.getenv("CORPUSAGENT2_PREFER_RUNTIME_API_BASE", "").strip().lower() in {"1", "true", "yes", "on"}
    payload: dict[str, Any] = {
        "apiBaseUrl": config.frontend.api_base_url,
        "title": config.frontend.title,
        "useOpenAI": config.llm.use_openai,
        "accessGate": {
            "enabled": os.getenv("CORPUSAGENT2_FRONTEND_ACCESS_GATE", "").strip().lower() in {"1", "true", "yes", "on"},
            "title": os.getenv("CORPUSAGENT2_FRONTEND_ACCESS_TITLE", "Private Demo Access").strip() or "Private Demo Access",
            "subtitle": os.getenv(
                "CORPUSAGENT2_FRONTEND_ACCESS_SUBTITLE",
                "Use Cloudflare Access for real protection. This frontend gate only blocks casual access.",
            ).strip(),
            "hint": os.getenv("CORPUSAGENT2_FRONTEND_ACCESS_HINT", "").strip(),
            "passwordSha256": os.getenv("CORPUSAGENT2_FRONTEND_ACCESS_PASSWORD_SHA256", "").strip().lower(),
        },
    }
    if prefer_runtime:
        payload["preferRuntimeApiBase"] = True
    return payload
