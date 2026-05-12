from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .agent_runtime import AgentRuntime, AgentRuntimeConfig
from .app_config import frontend_runtime_payload


class QueryRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(min_length=1)
    force_answer: bool = Field(default=False, validation_alias=AliasChoices("force_answer", "forceAnswer"))
    no_cache: bool = Field(default=False, validation_alias=AliasChoices("no_cache", "noCache"))
    async_mode: bool = Field(default=False, validation_alias=AliasChoices("async_mode", "asyncMode"))
    clarification_history: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("clarification_history", "clarificationHistory"),
    )


class LLMSettingsRequest(BaseModel):
    use_openai: bool
    planner_model: str = Field(default="")
    synthesis_model: str = Field(default="")


class LLMSettingsResetRequest(BaseModel):
    reset_to_startup: bool = True


def build_app(runtime: AgentRuntime | None = None, project_root: Path | None = None) -> FastAPI:
    resolved_root = project_root or Path(__file__).resolve().parents[2]
    resolved_runtime = runtime or AgentRuntime(
        config=AgentRuntimeConfig.from_project_root(resolved_root)
    )
    app = FastAPI(title="CorpusAgent2 Agent Runtime", version="0.1.0")
    cors_origins = [
        item.strip()
        for item in os.getenv("CORPUSAGENT2_CORS_ORIGINS", "*").split(",")
        if item.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        warmup = resolved_runtime.warmup_info()
        return {
            "status": "ok",
            "service": "corpusagent2-agent-runtime",
            "ready": bool(warmup.get("complete")),
            "warming": list(warmup.get("pending_stages", [])),
            "warmup_errors": list(warmup.get("errors", [])),
        }

    @app.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        return {"capabilities": resolved_runtime.capability_catalog()}

    @app.get("/runtime-info")
    def runtime_info() -> dict[str, Any]:
        info = resolved_runtime.runtime_info()
        info["warmup"] = resolved_runtime.warmup_info()
        return info

    @app.get("/tool-usage")
    def tool_usage() -> dict[str, Any]:
        return resolved_runtime.tool_usage_summary()

    @app.get("/settings/llm")
    def get_llm_settings() -> dict[str, Any]:
        return resolved_runtime.runtime_info()["llm"]

    @app.post("/settings/llm")
    def update_llm_settings(request: LLMSettingsRequest) -> dict[str, Any]:
        try:
            return resolved_runtime.update_llm_runtime_settings(
                use_openai=request.use_openai,
                planner_model=request.planner_model,
                synthesis_model=request.synthesis_model,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/settings/llm/reset")
    def reset_llm_settings(request: LLMSettingsResetRequest) -> dict[str, Any]:
        if not request.reset_to_startup:
            raise HTTPException(status_code=400, detail="reset_to_startup must be true.")
        try:
            return resolved_runtime.reset_llm_runtime_settings()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/diagnostics/dry-run")
    def diagnostics_dry_run() -> dict[str, Any]:
        """Smoke-test the API surface without prompting an LLM.

        Verifies warmup state, capability registry, LLM endpoint reachability,
        retrieval health, and python_runner availability. Every check is
        wrapped so one failure does not mask the others.
        """
        import socket
        import time
        from urllib.parse import urlparse

        started_total = time.monotonic()
        checks: dict[str, Any] = {}
        overall_ok = True

        warmup = resolved_runtime.warmup_info()
        checks["warmup"] = {
            "ok": bool(warmup.get("complete")),
            "ready": bool(warmup.get("complete")),
            "pending_stages": list(warmup.get("pending_stages", [])),
            "errors": list(warmup.get("errors", [])),
            "duration_ms": warmup.get("duration_ms"),
        }
        if not checks["warmup"]["ok"]:
            overall_ok = False

        try:
            tools = list(resolved_runtime.registry.list_tools())
            checks["capability_registry"] = {"ok": True, "tool_count": len(tools)}
        except Exception as exc:
            checks["capability_registry"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            overall_ok = False

        started = time.monotonic()
        try:
            parsed = urlparse(resolved_runtime.llm_config.base_url or "")
            host = parsed.hostname
            if not host:
                raise RuntimeError("LLM base_url has no host")
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            with socket.create_connection((host, port), timeout=5):
                pass
            checks["llm_endpoint"] = {
                "ok": True,
                "host": host,
                "port": port,
                "duration_ms": round((time.monotonic() - started) * 1000, 1),
            }
        except Exception as exc:
            checks["llm_endpoint"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "duration_ms": round((time.monotonic() - started) * 1000, 1),
            }
            overall_ok = False

        started = time.monotonic()
        try:
            retrieval_health = resolved_runtime._cached_retrieval_health()
            checks["retrieval"] = {
                "ok": True,
                "health": retrieval_health,
                "duration_ms": round((time.monotonic() - started) * 1000, 1),
            }
        except Exception as exc:
            checks["retrieval"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            overall_ok = False

        started = time.monotonic()
        try:
            result = resolved_runtime.python_runner.run(
                code="import json; open(OUTPUT_DIR + '/result.json', 'w').write(json.dumps({'value': 42}))",
                inputs_json={},
            )
            artifact_count = len(getattr(result, "artifacts", []) or [])
            exit_code = getattr(result, "exit_code", None)
            checks["python_runner"] = {
                "ok": exit_code == 0,
                "exit_code": exit_code,
                "artifact_count": artifact_count,
                "duration_ms": round((time.monotonic() - started) * 1000, 1),
                "stdout_tail": (getattr(result, "stdout", "") or "")[-400:],
                "stderr_tail": (getattr(result, "stderr", "") or "")[-400:],
            }
            if not checks["python_runner"]["ok"]:
                overall_ok = False
        except Exception as exc:
            checks["python_runner"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "duration_ms": round((time.monotonic() - started) * 1000, 1),
            }
            overall_ok = False

        return {
            "ok": overall_ok,
            "duration_ms": round((time.monotonic() - started_total) * 1000, 1),
            "checks": checks,
        }

    @app.post("/query")
    def query(request: QueryRequest) -> dict[str, Any]:
        if request.async_mode:
            status = resolved_runtime.submit_query(
                request.question,
                force_answer=request.force_answer,
                no_cache=request.no_cache,
                clarification_history=request.clarification_history,
            )
            return status.to_dict()
        manifest = resolved_runtime.handle_query(
            request.question,
            force_answer=request.force_answer,
            no_cache=request.no_cache,
            clarification_history=request.clarification_history,
        )
        return manifest.to_dict()

    @app.post("/query/submit")
    def submit_query(request: QueryRequest) -> dict[str, Any]:
        status = resolved_runtime.submit_query(
            request.question,
            force_answer=request.force_answer,
            no_cache=request.no_cache,
            clarification_history=request.clarification_history,
        )
        return status.to_dict()

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return resolved_runtime.get_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/artifact")
    def get_run_artifact(run_id: str, artifact_path: str) -> FileResponse:
        try:
            resolved = resolved_runtime.resolve_artifact_path(run_id, artifact_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return FileResponse(resolved)

    @app.get("/runs/{run_id}/status")
    def get_run_status(run_id: str) -> dict[str, Any]:
        try:
            return resolved_runtime.get_run_status(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/abort")
    def abort_run(run_id: str) -> dict[str, Any]:
        try:
            return resolved_runtime.abort_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/runs/abort-all")
    def abort_all_runs() -> dict[str, Any]:
        return resolved_runtime.abort_all_runs()

    serve_frontend = os.getenv("CORPUSAGENT2_SERVE_FRONTEND", "1").strip().lower() not in {"0", "false", "no", "off"}
    web_root = resolved_root / "web"
    if web_root.is_dir() and serve_frontend:

        @app.get("/config.js")
        def runtime_config_js(request: Request) -> Response:
            payload = frontend_runtime_payload(resolved_root)
            forwarded_proto = request.headers.get("x-forwarded-proto", "").strip().lower()
            forwarded_host = request.headers.get("x-forwarded-host", "").strip()
            scheme = forwarded_proto or request.url.scheme
            netloc = forwarded_host or request.url.netloc
            if netloc:
                payload["apiBaseUrl"] = f"{scheme}://{netloc}"
                payload["preferRuntimeApiBase"] = True
            body = (
                "window.CORPUSAGENT2_CONFIG = "
                + json.dumps(payload, ensure_ascii=False, indent=2)
                + ";\n"
            )
            return Response(
                content=body,
                media_type="application/javascript",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        app.mount("/", StaticFiles(directory=str(web_root), html=True), name="frontend")

    return app
