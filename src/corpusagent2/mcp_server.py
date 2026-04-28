from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .mcp_jobs import DEFAULT_MCP_PORT, MCPJobManager


_MANAGER: MCPJobManager | None = None


def _project_root() -> Path:
    return Path(os.getenv("CORPUSAGENT2_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def _manager() -> MCPJobManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = MCPJobManager(project_root=_project_root())
        _MANAGER.start()
    return _MANAGER


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def create_mcp_server(manager: MCPJobManager | None = None) -> FastMCP:
    if manager is not None:
        global _MANAGER
        _MANAGER = manager
        _MANAGER.start()

    host = os.getenv("CORPUSAGENT2_MCP_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("CORPUSAGENT2_MCP_PORT", str(DEFAULT_MCP_PORT)).strip() or str(DEFAULT_MCP_PORT))
    mcp = FastMCP(
        "CorpusAgent2 Jobs",
        instructions=(
            "Submit long-running CorpusAgent2 analysis jobs, poll durable job status, "
            "cancel running work, and read completed run manifests/artifact indexes. "
            "The server executes the real CorpusAgent2 AgentRuntime; it does not mock "
            "retrieval, NLP, plotting, or synthesis."
        ),
        host=host,
        port=port,
        streamable_http_path=os.getenv("CORPUSAGENT2_MCP_PATH", "/mcp").strip() or "/mcp",
        stateless_http=False,
    )

    @mcp.tool()
    def submit_question(
        question: str,
        owner: str = "",
        force_answer: bool = False,
        no_cache: bool = False,
        clarification_history: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a CorpusAgent2 question as a durable multi-user job."""

        return _manager().submit_question(
            question,
            owner=owner,
            force_answer=force_answer,
            no_cache=no_cache,
            clarification_history=clarification_history,
            metadata=metadata,
        )

    @mcp.tool()
    def get_job_status(job_id: str) -> dict[str, Any]:
        """Return queue/running/completed status for one MCP job."""

        return _manager().get_job_status(job_id)

    @mcp.tool()
    def list_jobs(owner: str = "", status: str = "", limit: int = 50) -> dict[str, Any]:
        """List recent jobs, optionally scoped by owner or status."""

        return _manager().list_jobs(owner=owner, status=status, limit=limit)

    @mcp.tool()
    def cancel_job(job_id: str) -> dict[str, Any]:
        """Cancel a queued job or request abort for a running AgentRuntime run."""

        return _manager().cancel_job(job_id)

    @mcp.tool()
    def get_job_result(job_id: str) -> dict[str, Any]:
        """Return job status plus the completed AgentRuntime manifest when available."""

        return _manager().get_job_result(job_id)

    @mcp.tool()
    def get_runtime_info() -> dict[str, Any]:
        """Return MCP worker capacity plus the underlying CorpusAgent2 runtime profile."""

        return _manager().runtime_info()

    @mcp.resource("corpusagent2://jobs/{job_id}")
    def job_resource(job_id: str) -> str:
        return _json(_manager().get_job_status(job_id))

    @mcp.resource("corpusagent2://jobs/{job_id}/manifest")
    def job_manifest_resource(job_id: str) -> str:
        return _json(_manager().get_job_result(job_id).get("manifest", {}))

    @mcp.resource("corpusagent2://jobs/{job_id}/artifacts")
    def job_artifacts_resource(job_id: str) -> str:
        result = _manager().get_job_result(job_id)
        manifest = result.get("manifest", {})
        artifacts_dir = str(manifest.get("artifacts_dir", ""))
        artifact_paths: list[dict[str, Any]] = []
        if artifacts_dir:
            base = Path(artifacts_dir)
            if base.exists():
                for path in sorted(base.rglob("*")):
                    if path.is_file():
                        artifact_paths.append(
                            {
                                "name": path.name,
                                "relative_path": str(path.relative_to(base)),
                                "absolute_path": str(path),
                                "size_bytes": path.stat().st_size,
                            }
                        )
        return _json(
            {
                "job_id": job_id,
                "run_id": result.get("run_id", ""),
                "artifacts_dir": artifacts_dir,
                "artifacts": artifact_paths,
            }
        )

    return mcp


def main() -> None:
    transport = os.getenv("CORPUSAGENT2_MCP_TRANSPORT", "streamable-http").strip() or "streamable-http"
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise SystemExit(f"Unsupported CORPUSAGENT2_MCP_TRANSPORT={transport!r}")
    _manager().start()
    server = create_mcp_server()
    server.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
