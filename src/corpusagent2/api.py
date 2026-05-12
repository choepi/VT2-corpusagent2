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
