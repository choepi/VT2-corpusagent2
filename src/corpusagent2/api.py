from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent_runtime import AgentRuntime, AgentRuntimeConfig


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    force_answer: bool = False
    no_cache: bool = False
    async_mode: bool = False
    clarification_history: list[str] = Field(default_factory=list)


class LLMSettingsRequest(BaseModel):
    use_openai: bool
    planner_model: str = Field(default="")
    synthesis_model: str = Field(default="")


class LLMSettingsResetRequest(BaseModel):
    reset_to_startup: bool = True


def build_app(runtime: AgentRuntime | None = None, project_root: Path | None = None) -> FastAPI:
    resolved_runtime = runtime or AgentRuntime(
        config=AgentRuntimeConfig.from_project_root(project_root or Path(__file__).resolve().parents[2])
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
        return {"status": "ok", "service": "corpusagent2-agent-runtime"}

    @app.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        return {"capabilities": resolved_runtime.capability_catalog()}

    @app.get("/runtime-info")
    def runtime_info() -> dict[str, Any]:
        return resolved_runtime.runtime_info()

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

    return app
