from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

from corpusagent2.mcp_jobs import MCPJobManager, MCPJobRecord, SQLiteMCPJobStore
from corpusagent2.mcp_server import create_mcp_server


@dataclass(slots=True)
class _FakeFinalAnswer:
    answer_text: str

    def to_dict(self) -> dict[str, Any]:
        return {"answer_text": self.answer_text}


@dataclass(slots=True)
class _FakeManifest:
    run_id: str
    status: str
    artifacts_dir: str
    final_answer: _FakeFinalAnswer
    evidence_table: list[dict[str, Any]]
    selected_docs: list[dict[str, Any]]
    node_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "artifacts_dir": self.artifacts_dir,
            "final_answer": self.final_answer.to_dict(),
            "evidence_table": list(self.evidence_table),
            "selected_docs": list(self.selected_docs),
            "node_records": list(self.node_records),
        }


class _FakeRuntime:
    def __init__(self, artifacts_root: Path, sleep_s: float = 0.05) -> None:
        self.artifacts_root = artifacts_root
        self.sleep_s = sleep_s
        self.manifests: dict[str, dict[str, Any]] = {}
        self.aborted: set[str] = set()

    def handle_query(
        self,
        question: str,
        *,
        force_answer: bool = False,
        no_cache: bool = False,
        clarification_history: list[str] | None = None,
        run_id: str | None = None,
    ) -> _FakeManifest:
        assert run_id
        time.sleep(self.sleep_s)
        artifacts_dir = self.artifacts_root / run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        status = "aborted" if run_id in self.aborted else "completed"
        manifest = _FakeManifest(
            run_id=run_id,
            status=status,
            artifacts_dir=str(artifacts_dir),
            final_answer=_FakeFinalAnswer(f"answered: {question}"),
            evidence_table=[{"doc_id": "d1"}],
            selected_docs=[{"doc_id": "d1"}],
            node_records=[{"node_id": "n1"}],
        )
        (artifacts_dir / "run_manifest.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
        self.manifests[run_id] = manifest.to_dict()
        return manifest

    def get_run_status(self, run_id: str) -> dict[str, Any]:
        return {"run_id": run_id, "status": "running", "detail": "fake active status"}

    def abort_run(self, run_id: str) -> dict[str, Any]:
        self.aborted.add(run_id)
        return {"run_id": run_id, "status": "aborting"}

    def get_run(self, run_id: str) -> dict[str, Any]:
        if run_id not in self.manifests:
            raise FileNotFoundError(run_id)
        return self.manifests[run_id]

    def runtime_info(self) -> dict[str, Any]:
        return {"fake": True}


def _wait_until(predicate, *, timeout_s: float = 10.0, describe=None) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    assert predicate(), describe() if describe is not None else "condition was not met"


def test_mcp_job_manager_runs_real_runtime_contract_with_preallocated_run_id(tmp_path: Path) -> None:
    store = SQLiteMCPJobStore(tmp_path / "jobs.sqlite")
    runtime = _FakeRuntime(tmp_path / "artifacts")
    manager = MCPJobManager(
        project_root=tmp_path,
        store=store,
        runtime=runtime,  # type: ignore[arg-type]
        max_concurrent_jobs=1,
        poll_seconds=0.01,
        heartbeat_seconds=0.02,
        stale_running_after_s=30,
    )
    try:
        first = manager.submit_question("first question", owner="u1")
        second = manager.submit_question("second question", owner="u2")
        assert first["job_id"] != second["job_id"]
        assert first["run_id"].startswith("agent_")
        assert second["run_id"].startswith("agent_")

        _wait_until(
            lambda: manager.get_job_status(first["job_id"])["status"] in {"completed", "failed"},
            describe=lambda: manager.get_job_status(first["job_id"]),
        )
        _wait_until(
            lambda: manager.get_job_status(second["job_id"])["status"] in {"completed", "failed"},
            describe=lambda: manager.get_job_status(second["job_id"]),
        )
        first_status = manager.get_job_status(first["job_id"])
        second_status = manager.get_job_status(second["job_id"])
        assert first_status["status"] == "completed", first_status
        assert second_status["status"] == "completed", second_status

        first_result = manager.get_job_result(first["job_id"])
        assert first_result["result_summary"]["run_id"] == first["run_id"]
        assert first_result["result_summary"]["evidence_rows"] == 1
        assert first_result["manifest"]["final_answer"]["answer_text"] == "answered: first question"
    finally:
        manager.stop()


def test_mcp_job_status_reports_on_hold_when_capacity_is_full(tmp_path: Path) -> None:
    store = SQLiteMCPJobStore(tmp_path / "jobs.sqlite")
    runtime = _FakeRuntime(tmp_path / "artifacts", sleep_s=0.2)
    manager = MCPJobManager(
        project_root=tmp_path,
        store=store,
        runtime=runtime,  # type: ignore[arg-type]
        max_concurrent_jobs=1,
        poll_seconds=0.01,
        heartbeat_seconds=0.02,
        stale_running_after_s=30,
    )
    try:
        first = manager.submit_question("slow question")
        _wait_until(lambda: manager.get_job_status(first["job_id"])["capacity"]["running_jobs"] == 1)
        second = manager.submit_question("waiting question")
        second_status = manager.get_job_status(second["job_id"])
        assert second_status["status"] == "on_hold"
        assert second_status["hold_reason"] == "waiting_for_worker_capacity"
        assert second_status["queue_position"] == 1
    finally:
        manager.stop()


def test_mcp_job_store_does_not_regress_terminal_status_from_heartbeat(tmp_path: Path) -> None:
    store = SQLiteMCPJobStore(tmp_path / "jobs.sqlite")
    job = store.create_job(
        MCPJobRecord(
            job_id="mcpjob_terminal_guard",
            run_id="agent_terminal_guard",
            owner="",
            question="question",
            status="running",
        )
    )
    completed = store.set_job_status(
        job.job_id,
        "completed",
        result_summary={"status": "completed"},
        finished=True,
    )
    assert completed is not None
    assert completed.status == "completed"

    stale_heartbeat = store.set_job_status(
        job.job_id,
        "running",
        result_summary={"live_status": {"status": "running"}},
    )
    assert stale_heartbeat is not None
    assert stale_heartbeat.status == "completed"
    assert stale_heartbeat.result_summary == {"status": "completed"}


def test_create_mcp_server_registers_job_tools(tmp_path: Path) -> None:
    manager = MCPJobManager(
        project_root=tmp_path,
        store=SQLiteMCPJobStore(tmp_path / "jobs.sqlite"),
        runtime=_FakeRuntime(tmp_path / "artifacts"),  # type: ignore[arg-type]
        max_concurrent_jobs=1,
        poll_seconds=0.01,
        heartbeat_seconds=0.02,
        stale_running_after_s=30,
    )
    try:
        server = create_mcp_server(manager=manager)
        assert server is not None
    finally:
        manager.stop()
