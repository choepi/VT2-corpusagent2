from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import uuid
from typing import Any

from .execution_engine import PlanExecutor
from .io_utils import read_json, read_jsonl, write_json, write_jsonl
from .planner import PlanningContext, QuestionPlanner
from .provider_adapters import build_default_registry
from .run_manifest import RunManifest
from .runtime_context import CorpusRuntime
from .tool_registry import ToolRegistry


def build_planning_context(runtime: CorpusRuntime) -> PlanningContext:
    try:
        metadata = runtime.load_metadata()
        metadata_columns = set(str(column) for column in metadata.columns)
        total_documents = int(metadata.shape[0])
    except Exception:
        metadata_columns = set()
        total_documents = 0

    retrieval_ready = False
    dense_ready = False
    notes: list[str] = []
    try:
        runtime.load_lexical_assets()
        retrieval_ready = True
    except Exception as exc:
        notes.append(f"lexical assets unavailable: {exc}")

    if runtime.retrieval_backend == "local":
        try:
            runtime.load_dense_assets()
            dense_ready = True
        except Exception as exc:
            notes.append(f"dense assets unavailable: {exc}")
    else:
        dense_ready = True

    artifact_names = {"entity_trend", "sentiment_series", "topics_over_time", "burst_events", "keyphrases"}
    available_artifacts = {name for name in artifact_names if runtime.artifact_available(name)}

    return PlanningContext(
        total_documents=total_documents,
        metadata_columns=metadata_columns,
        available_artifacts=available_artifacts,
        retrieval_ready=retrieval_ready,
        dense_ready=dense_ready,
        notes=notes,
    )


@dataclass(slots=True)
class WorkloadQuestion:
    question_id: str
    raw_question: str


def load_workload_questions(path: Path) -> list[WorkloadQuestion]:
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        rows = list(payload if isinstance(payload, list) else [])
    else:
        rows = read_jsonl(path)
    results: list[WorkloadQuestion] = []
    for idx, row in enumerate(rows, start=1):
        question = str(row.get("raw_question") or row.get("query") or "").strip()
        if not question:
            continue
        question_id = str(row.get("question_id") or row.get("query_id") or f"q{idx:03d}")
        results.append(WorkloadQuestion(question_id=question_id, raw_question=question))
    return results


class FrameworkRunner:
    def __init__(
        self,
        runtime: CorpusRuntime,
        registry: ToolRegistry | None = None,
        planner: QuestionPlanner | None = None,
        executor: PlanExecutor | None = None,
    ) -> None:
        self.runtime = runtime
        self.registry = registry or build_default_registry()
        self.planner = planner or QuestionPlanner(registry=self.registry)
        self.executor = executor or PlanExecutor(registry=self.registry)

    def run_question(self, raw_question: str, question_id: str | None = None, artifacts_root: Path | None = None) -> RunManifest:
        planning_context = build_planning_context(self.runtime)
        question_spec = self.planner.build_question_spec(
            raw_question=raw_question,
            planning_context=planning_context,
            question_id=question_id,
        )
        plan_graph = self.planner.build_plan(question_spec)
        manifest = self.executor.execute(
            plan_graph=plan_graph,
            question_spec=question_spec,
            runtime=self.runtime,
            artifacts_root=artifacts_root,
        )
        write_json(Path(manifest.artifacts_dir) / "run_manifest.json", manifest.to_dict())
        return manifest

    def run_workload(self, questions: list[WorkloadQuestion], artifacts_root: Path | None = None) -> list[RunManifest]:
        manifests: list[RunManifest] = []
        for question in questions:
            manifests.append(
                self.run_question(
                    raw_question=question.raw_question,
                    question_id=question.question_id,
                    artifacts_root=artifacts_root,
                )
            )
        return manifests


def run_workload_file(
    *,
    project_root: Path,
    workload_path: Path,
    mode: str = "full",
    output_root: Path | None = None,
) -> dict[str, Any]:
    runtime = CorpusRuntime.from_project_root(project_root)
    runner = FrameworkRunner(runtime=runtime)
    questions = load_workload_questions(workload_path)
    if mode == "debug":
        questions = questions[: min(len(questions), 5)]
    if not questions:
        raise RuntimeError("No workload entries found.")

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    run_root = (output_root or (project_root / "outputs" / "framework")) / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    manifests = runner.run_workload(questions=questions, artifacts_root=run_root)

    reports: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for manifest in manifests:
        reports.append(
            {
                "run_id": manifest.run_id,
                "question_id": manifest.question_spec.question_id,
                "question": manifest.question_spec.raw_question,
                "status": manifest.status,
                "final_answer": manifest.final_answer.to_dict(),
                "failures": [item.to_dict() for item in manifest.failures],
                "manifest_path": str(Path(manifest.artifacts_dir) / "run_manifest.json"),
            }
        )
        provenance_rows.extend(manifest.provenance_records)

    write_jsonl(run_root / "reports.jsonl", reports)
    write_jsonl(run_root / "provenance.jsonl", provenance_rows)

    summary = {
        "run_id": run_id,
        "mode": mode,
        "workload_size": len(questions),
        "completed_runs": sum(1 for item in manifests if item.status == "completed"),
        "partial_runs": sum(1 for item in manifests if item.status == "partial"),
        "failed_runs": sum(1 for item in manifests if item.status == "failed"),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "workload_path": str(workload_path),
        "output_dir": str(run_root),
    }
    write_json(run_root / "run_summary.json", summary)
    return summary
