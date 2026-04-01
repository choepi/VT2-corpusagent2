from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import threading
import uuid
from typing import Any

from .agent_backends import (
    InMemoryWorkingSetStore,
    LocalSearchBackend,
    OpenSearchBackend,
    OpenSearchConfig,
    PostgresWorkingSetStore,
    WorkingSetStore,
    save_agent_manifest,
)
from .agent_capabilities import AgentExecutionContext, build_agent_registry
from .agent_executor import AgentExecutionSnapshot, AsyncPlanExecutor
from .agent_models import (
    AgentFailure,
    AgentPlanDAG,
    AgentPlanNode,
    AgentRunManifest,
    AgentRunState,
    LiveRunStatus,
    PlannerAction,
)
from .agent_policy import evidence_required, rejection_reason_for_question, rewrite_special_cases
from .app_config import load_project_configuration
from .llm_provider import LLMClient, LLMProviderConfig, OpenAICompatibleLLMClient
from .retrieval import pg_dsn_from_env, pg_table_from_env
from .run_manifest import FinalAnswerPayload
from .runtime_context import CorpusRuntime
from .tool_registry import ToolRegistry
from .python_runner_service import DockerPythonRunnerService


@dataclass(slots=True)
class AgentRuntimeConfig:
    project_root: Path
    outputs_root: Path
    planner_calls_max: int = 6

    @classmethod
    def from_project_root(cls, project_root: Path) -> "AgentRuntimeConfig":
        project_root = project_root.resolve()
        return cls(
            project_root=project_root,
            outputs_root=(project_root / "outputs" / "agent_runtime").resolve(),
        )

class MagicBoxOrchestrator:
    def __init__(self, llm_client: LLMClient | None = None, llm_config: LLMProviderConfig | None = None) -> None:
        self.llm_client = llm_client
        self.llm_config = llm_config or LLMProviderConfig.from_env()

    def _question_with_clarifications(self, state: AgentRunState) -> str:
        history = [str(item).strip() for item in state.clarification_history if str(item).strip()]
        if not history:
            return state.question
        suffix = "\n".join(f"- {item}" for item in history)
        return f"{state.question}\n\nUser clarification history:\n{suffix}"

    def rephrase_or_clarify(self, state: AgentRunState) -> PlannerAction:
        rejection_reason = rejection_reason_for_question(state.question)
        if rejection_reason:
            return PlannerAction(action="grounded_rejection", rejection_reason=rejection_reason)

        enriched_question = self._question_with_clarifications(state)
        rewritten, assumptions = rewrite_special_cases(enriched_question)
        if self.llm_client is None:
            if state.clarification_history:
                return PlannerAction(
                    action="accept_with_assumptions",
                    rewritten_question=rewritten,
                    assumptions=assumptions,
                )
            if " between groups " in f" {state.question.lower()} " and not state.force_answer:
                return PlannerAction(
                    action="ask_clarification",
                    rewritten_question=rewritten,
                    clarification_question="Which exact groups, outlets, or entities should be compared?",
                    assumptions=assumptions,
                )
            return PlannerAction(
                action="accept_with_assumptions",
                rewritten_question=rewritten,
                assumptions=assumptions,
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the rephrasing and clarification module for a corpus-analysis agent. "
                    "Return JSON with keys action, rewritten_question, clarification_question, assumptions, rejection_reason, message. "
                    "Allowed actions: ask_clarification, accept_with_assumptions, grounded_rejection. "
                    "Reject hidden-motive questions. Ask clarification only if workflow changes materially."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": state.question,
                        "clarification_history": list(state.clarification_history),
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        try:
            payload = self.llm_client.complete_json(
                messages,
                model=self.llm_config.planner_model,
                temperature=0.0,
            )
            action = PlannerAction.from_dict(payload)
            if action.action == "ask_clarification" and state.force_answer:
                forced_rewrite = rewritten or state.question
                forced_assumptions = list(dict.fromkeys(list(action.assumptions) + assumptions + [
                    "force_answer=true: proceeded with best-effort assumptions instead of waiting for clarification."
                ]))
                return PlannerAction(
                    action="accept_with_assumptions",
                    rewritten_question=forced_rewrite,
                    assumptions=forced_assumptions,
                    message=action.message,
                )
            if not action.rewritten_question:
                action.rewritten_question = rewritten
            if assumptions:
                action.assumptions = list(dict.fromkeys(list(action.assumptions) + assumptions))
            return action
        except Exception:
            return PlannerAction(
                action="accept_with_assumptions",
                rewritten_question=rewritten,
                assumptions=assumptions,
            )

    def _heuristic_plan(self, state: AgentRunState) -> PlannerAction:
        text = (state.rewritten_question or self._question_with_clarifications(state)).lower()
        if "distribution" in text and "noun" in text:
            dag = AgentPlanDAG(
                nodes=[
                    AgentPlanNode("search", "db_search", {"top_k": 40}),
                    AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                    AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                    AgentPlanNode("pos", "pos_morph", depends_on=["fetch"]),
                    AgentPlanNode("lemmas", "lemmatize", depends_on=["fetch"]),
                    AgentPlanNode("plot", "plot_artifact", {"plot_name": "noun_distribution"}, depends_on=["pos"], optional=True),
                ],
                metadata={"question_family": "noun_distribution"},
            )
            return PlannerAction(action="emit_plan_dag", rewritten_question=state.rewritten_question, plan_dag=dag)
        if "named entit" in text or ("climate" in text and "entity" in text):
            dag = AgentPlanDAG(
                nodes=[
                    AgentPlanNode("search", "db_search", {"top_k": 60}),
                    AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                    AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                    AgentPlanNode("ner", "ner", depends_on=["fetch"]),
                    AgentPlanNode("series", "time_series_aggregate", depends_on=["ner"]),
                    AgentPlanNode("changes", "change_point_detect", depends_on=["series"], optional=True),
                    AgentPlanNode("plot", "plot_artifact", {"plot_name": "entity_trend"}, depends_on=["series"], optional=True),
                ],
                metadata={"question_family": "entity_trend"},
            )
            return PlannerAction(action="emit_plan_dag", rewritten_question=state.rewritten_question, plan_dag=dag)
        if "ukraine" in text and "predict" in text:
            dag = AgentPlanDAG(
                nodes=[
                    AgentPlanNode("search", "db_search", {"top_k": 80, "date_to": "2022-02-23"}),
                    AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                    AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                    AgentPlanNode("claim_spans", "claim_span_extract", depends_on=["fetch"]),
                    AgentPlanNode("claim_scores", "claim_strength_score", depends_on=["claim_spans"]),
                    AgentPlanNode("evidence", "build_evidence_table", depends_on=["claim_scores"]),
                ],
                metadata={"question_family": "prediction_evidence"},
            )
            return PlannerAction(action="emit_plan_dag", rewritten_question=state.rewritten_question, plan_dag=dag)
        dag = AgentPlanDAG(
            nodes=[
                AgentPlanNode("search", "db_search", {"top_k": 40}),
                AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                AgentPlanNode("keyterms", "extract_keyterms", depends_on=["fetch"], optional=True),
            ],
            metadata={"question_family": "generic"},
        )
        return PlannerAction(action="emit_plan_dag", rewritten_question=state.rewritten_question, plan_dag=dag)

    def plan(self, state: AgentRunState) -> PlannerAction:
        if self.llm_client is None:
            return self._heuristic_plan(state)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the planning module for a corpus-analysis agent. "
                    "Return JSON with keys action, rewritten_question, assumptions, clarification_question, rejection_reason, message, plan_dag. "
                    "Allowed actions: ask_clarification, emit_plan_dag, grounded_rejection. "
                    "Choose capabilities, not libraries. Keep plans compact and parallel where possible."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": state.question,
                        "rewritten_question": state.rewritten_question,
                        "clarification_history": list(state.clarification_history),
                        "available_capabilities": state.available_capabilities,
                        "corpus_schema": state.corpus_schema,
                        "failures": state.failures,
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        try:
            payload = self.llm_client.complete_json(
                messages,
                model=self.llm_config.planner_model,
                temperature=0.0,
            )
            action = PlannerAction.from_dict(payload)
            if action.action == "ask_clarification" and state.force_answer:
                heuristic = self._heuristic_plan(state)
                heuristic.assumptions = list(
                    dict.fromkeys(
                        list(action.assumptions)
                        + ["force_answer=true: planner clarification skipped and best-effort heuristic plan was used."]
                    )
                )
                return heuristic
            return action
        except Exception:
            return self._heuristic_plan(state)

    def revise_after_failure(self, state: AgentRunState, failure: AgentFailure) -> PlannerAction | None:
        if failure.capability == "python_runner":
            return None
        code = (
            "from pathlib import Path\n"
            "import json\n"
            "summary = {'received_keys': sorted(INPUTS_JSON.keys())}\n"
            "Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)\n"
            "Path(OUTPUT_DIR, 'fallback_summary.json').write_text(json.dumps(summary), encoding='utf-8')\n"
            "print(json.dumps(summary))\n"
        )
        dag = AgentPlanDAG(
            nodes=[
                AgentPlanNode(
                    "python_fallback",
                    "python_runner",
                    {"code": code},
                )
            ],
            metadata={"revision_for": failure.capability},
        )
        return PlannerAction(
            action="revise_plan_after_failure",
            rewritten_question=state.rewritten_question,
            assumptions=[f"Used python_runner fallback after failure in capability '{failure.capability}'."],
            plan_dag=dag,
        )

    def synthesize(self, state: AgentRunState, snapshot: AgentExecutionSnapshot) -> FinalAnswerPayload:
        evidence_rows = self._extract_evidence(snapshot)
        summary = self._derive_summary(snapshot)
        if self.llm_client is None:
            return self._fallback_synthesis(state, evidence_rows, summary, snapshot)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the grounded synthesis module for a corpus-analysis agent. "
                    "Return JSON with keys answer_text, evidence_items, artifacts_used, unsupported_parts, caveats, claim_verdicts. "
                    "Use only the provided summaries, tool outputs, and evidence."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": state.question,
                        "rewritten_question": state.rewritten_question,
                        "summary": summary,
                        "evidence_rows": evidence_rows,
                        "failures": [item.to_dict() for item in snapshot.failures],
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        try:
            payload = self.llm_client.complete_json(
                messages,
                model=self.llm_config.synthesis_model,
                temperature=0.1,
            )
            answer = FinalAnswerPayload.from_payload(payload)
            if evidence_required(state.question):
                answer.evidence_items = evidence_rows
            return answer
        except Exception:
            return self._fallback_synthesis(state, evidence_rows, summary, snapshot)

    def _extract_evidence(self, snapshot: AgentExecutionSnapshot) -> list[dict[str, Any]]:
        for node_id, result in snapshot.node_results.items():
            payload = result.payload
            if isinstance(payload, dict) and "rows" in payload and payload["rows"]:
                first = payload["rows"][0]
                if {"doc_id", "outlet", "date", "excerpt", "score"}.issubset(first.keys()):
                    return list(payload["rows"])
        return []

    def _derive_summary(self, snapshot: AgentExecutionSnapshot) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for node_id, result in snapshot.node_results.items():
            payload = result.payload if isinstance(result.payload, dict) else {}
            rows = list(payload.get("rows", []))
            if not rows:
                continue
            first = rows[0]
            if "pos" in first and "lemma" in first:
                counts: dict[str, int] = {}
                for row in rows:
                    if str(row.get("pos", "")) not in {"NOUN", "PROPN"}:
                        continue
                    lemma = str(row.get("lemma", "")).lower()
                    if not lemma or len(lemma) < 3:
                        continue
                    counts[lemma] = counts.get(lemma, 0) + 1
                summary["noun_distribution"] = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:15]
            elif "entity" in first and "time_bin" in first:
                grouped: dict[str, int] = {}
                for row in rows:
                    entity = str(row.get("entity", ""))
                    grouped[entity] = grouped.get(entity, 0) + int(row.get("count", 1))
                summary["entity_trend"] = sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:15]
            elif {"doc_id", "outlet", "date", "excerpt", "score"}.issubset(first.keys()):
                summary["evidence_rows"] = rows[:10]
        return summary

    def _fallback_synthesis(
        self,
        state: AgentRunState,
        evidence_rows: list[dict[str, Any]],
        summary: dict[str, Any],
        snapshot: AgentExecutionSnapshot,
    ) -> FinalAnswerPayload:
        caveats = [failure.message for failure in snapshot.failures]
        if "noun_distribution" in summary:
            top = ", ".join(f"{term} ({count})" for term, count in summary["noun_distribution"][:8])
            answer_text = f"Top noun lemmas in the retrieved slice are: {top}."
        elif "entity_trend" in summary:
            top = ", ".join(f"{entity} ({count})" for entity, count in summary["entity_trend"][:8])
            answer_text = f"Most prominent entities in the retrieved slice are: {top}."
        elif evidence_rows:
            answer_text = (
                "The strongest immediate pre-invasion prediction or warning evidence comes from the returned articles. "
                "Higher-scoring rows are more explicit about near-term invasion."
            )
            caveats.append(
                "Warnings, scenarios, and explicit predictions are not identical; the evidence table is ranked by heuristic claim strength."
            )
        else:
            answer_text = "The agent completed the available analysis steps and returned the grounded artifacts for inspection."
        return FinalAnswerPayload(
            answer_text=answer_text,
            evidence_items=evidence_rows if evidence_required(state.question) else [],
            artifacts_used=[
                artifact
                for record in snapshot.node_records
                for artifact in record.artifacts_used
            ],
            unsupported_parts=[failure.message for failure in snapshot.failures if not failure.retriable],
            caveats=list(dict.fromkeys(item for item in caveats if item)),
            claim_verdicts=[],
        )


class AgentRuntime:
    def __init__(
        self,
        *,
        config: AgentRuntimeConfig,
        runtime: CorpusRuntime | None = None,
        llm_client: LLMClient | None = None,
        llm_config: LLMProviderConfig | None = None,
        registry: ToolRegistry | None = None,
        search_backend: Any | None = None,
        working_store: WorkingSetStore | None = None,
        python_runner: DockerPythonRunnerService | None = None,
    ) -> None:
        self.config = config
        self.app_config = load_project_configuration(config.project_root)
        self.runtime = runtime or CorpusRuntime.from_project_root(config.project_root)
        self.llm_config = llm_config or LLMProviderConfig.from_env()
        self.llm_client = llm_client or OpenAICompatibleLLMClient(self.llm_config)
        self.registry = registry or build_agent_registry()
        self.search_backend = search_backend or self._build_search_backend()
        self.working_store = working_store or self._build_working_store()
        self.python_runner = python_runner or DockerPythonRunnerService()
        self.orchestrator = MagicBoxOrchestrator(self.llm_client, self.llm_config)
        self.executor = AsyncPlanExecutor(self.registry)
        self._live_runs: dict[str, LiveRunStatus] = {}
        self._run_lock = threading.Lock()

    def _build_search_backend(self):
        try:
            return OpenSearchBackend(OpenSearchConfig.from_env())
        except Exception:
            return LocalSearchBackend(self.runtime)

    def _build_working_store(self) -> WorkingSetStore:
        try:
            dsn = pg_dsn_from_env(required=False)
        except Exception:
            dsn = ""
        if dsn:
            return PostgresWorkingSetStore(dsn=dsn, documents_table=pg_table_from_env(default="article_corpus"))
        store = InMemoryWorkingSetStore()
        store.document_lookup.update(self.runtime.doc_lookup())
        return store

    def capability_catalog(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self.registry.list_tools()]

    def _set_live_status(self, run_id: str, **updates: Any) -> LiveRunStatus:
        with self._run_lock:
            status = self._live_runs.get(run_id)
            if status is None:
                status = LiveRunStatus(
                    run_id=run_id,
                    question=str(updates.get("question", "")),
                    status=str(updates.get("status", "queued")),
                )
                self._live_runs[run_id] = status
            for key, value in updates.items():
                if hasattr(status, key):
                    setattr(status, key, value)
            status.updated_at_utc = datetime.now(UTC).isoformat()
            return status

    def _record_step_event(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._run_lock:
            status = self._live_runs.get(run_id)
            if status is None:
                return
            event = str(payload.get("event", ""))
            entry = {
                "node_id": str(payload.get("node_id", "")),
                "capability": str(payload.get("capability", "")),
                "status": str(payload.get("status", "")),
            }
            if payload.get("error"):
                entry["error"] = str(payload["error"])
            if event == "node_started":
                status.active_steps = [
                    item for item in status.active_steps if item.get("node_id") != entry["node_id"]
                ] + [entry]
                status.current_phase = "executing"
                status.detail = f"Running {entry['capability']}"
            elif event == "node_completed":
                status.active_steps = [
                    item for item in status.active_steps if item.get("node_id") != entry["node_id"]
                ]
                status.completed_steps.append(entry)
                status.detail = f"Completed {entry['capability']}"
            elif event == "node_failed":
                status.active_steps = [
                    item for item in status.active_steps if item.get("node_id") != entry["node_id"]
                ]
                status.failed_steps.append(entry)
                status.detail = f"Failed {entry['capability']}"
            status.updated_at_utc = datetime.now(UTC).isoformat()

    def corpus_schema(self) -> dict[str, Any]:
        metadata = self.runtime.load_metadata().head(1)
        return {
            "metadata_fields": sorted(str(column) for column in metadata.columns),
            "retrieval_backend": "opensearch+postgres",
            "document_count": int(self.runtime.load_metadata().shape[0]),
        }

    def _build_state(self, question: str, force_answer: bool, no_cache: bool) -> AgentRunState:
        return AgentRunState(
            question=question,
            force_answer=force_answer,
            available_capabilities=sorted(
                {capability for spec in self.registry.list_tools() for capability in spec.capabilities}
            ),
            corpus_schema=self.corpus_schema(),
            planner_calls_max=self.config.planner_calls_max,
            no_cache=no_cache,
        )

    def _needs_clarification_manifest(
        self,
        *,
        run_id: str,
        question: str,
        rewritten_question: str,
        clarification_question: str,
        assumptions: list[str],
        artifacts_dir: Path,
        state: AgentRunState,
    ) -> AgentRunManifest:
        return AgentRunManifest(
            run_id=run_id,
            question=question,
            rewritten_question=rewritten_question,
            status="needs_clarification",
            clarification_questions=[clarification_question],
            assumptions=assumptions,
            planner_actions=list(state.planner_actions),
            plan_dags=[],
            selected_docs=[],
            node_records=[],
            provenance_records=[],
            evidence_table=[],
            final_answer=FinalAnswerPayload(
                answer_text="",
                caveats=["Clarification is required before planning can continue."],
            ),
            artifacts_dir=str(artifacts_dir),
            metadata={
                "clarification_question": clarification_question,
                "clarification_history": list(state.clarification_history),
            },
        )

    def _run_query(
        self,
        run_id: str,
        question: str,
        *,
        force_answer: bool = False,
        no_cache: bool = False,
        clarification_history: list[str] | None = None,
    ) -> AgentRunManifest:
        artifacts_dir = (self.config.outputs_root / run_id).resolve()
        (artifacts_dir / "nodes").mkdir(parents=True, exist_ok=True)
        self._set_live_status(
            run_id,
            question=question,
            status="running",
            current_phase="initializing",
            detail="Preparing run state",
        )

        state = self._build_state(question=question, force_answer=force_answer, no_cache=no_cache)
        if clarification_history:
            state.clarification_history = list(clarification_history)

        try:
            self.working_store.create_run(
                run_id=run_id,
                question=question,
                rewritten_question="",
                force_answer=force_answer,
                no_cache=no_cache,
            )
        except Exception:
            fallback_store = InMemoryWorkingSetStore()
            fallback_store.document_lookup.update(self.runtime.doc_lookup())
            self.working_store = fallback_store
            self.working_store.create_run(
                run_id=run_id,
                question=question,
                rewritten_question="",
                force_answer=force_answer,
                no_cache=no_cache,
            )

        self._set_live_status(run_id, current_phase="rephrase_or_clarify", detail="Rephrasing or clarifying question")
        rephrase_action = self.orchestrator.rephrase_or_clarify(state)
        state.planner_calls_used += 1
        state.planner_actions.append(rephrase_action.to_dict())
        self._set_live_status(run_id, planner_actions=list(state.planner_actions))
        if rephrase_action.assumptions:
            state.assumptions = list(dict.fromkeys(state.assumptions + rephrase_action.assumptions))
            self._set_live_status(run_id, assumptions=list(state.assumptions))
        if rephrase_action.action == "grounded_rejection":
            manifest = AgentRunManifest(
                run_id=run_id,
                question=question,
                rewritten_question=rephrase_action.rewritten_question or question,
                status="rejected",
                clarification_questions=[],
                assumptions=list(state.assumptions),
                planner_actions=list(state.planner_actions),
                plan_dags=[],
                selected_docs=[],
                node_records=[],
                provenance_records=[],
                evidence_table=[],
                final_answer=FinalAnswerPayload(
                    answer_text=rephrase_action.rejection_reason,
                    unsupported_parts=[rephrase_action.rejection_reason],
                    caveats=[],
                ),
                artifacts_dir=str(artifacts_dir),
            )
            self._persist_manifest(manifest)
            return manifest

        state.rewritten_question = rephrase_action.rewritten_question or question

        if (
            rephrase_action.action == "ask_clarification"
            and not force_answer
            and len(state.clarification_history) < 2
        ):
            manifest = self._needs_clarification_manifest(
                run_id=run_id,
                question=question,
                rewritten_question=state.rewritten_question,
                clarification_question=rephrase_action.clarification_question,
                assumptions=list(state.assumptions),
                artifacts_dir=artifacts_dir,
                state=state,
            )
            self._set_live_status(
                run_id,
                status="needs_clarification",
                current_phase="waiting_for_user",
                clarification_questions=[rephrase_action.clarification_question],
                assumptions=list(state.assumptions),
                planner_actions=list(state.planner_actions),
                detail="Waiting for clarification before planning",
            )
            self._persist_manifest(manifest)
            return manifest

        self._set_live_status(run_id, current_phase="planning", detail="Building plan DAG")
        plan_action = self.orchestrator.plan(state)
        state.planner_calls_used += 1
        state.planner_actions.append(plan_action.to_dict())
        self._set_live_status(run_id, planner_actions=list(state.planner_actions))
        if plan_action.action == "ask_clarification" and not force_answer and len(state.clarification_history) < 2:
            manifest = self._needs_clarification_manifest(
                run_id=run_id,
                question=question,
                rewritten_question=state.rewritten_question,
                clarification_question=plan_action.clarification_question,
                assumptions=list(state.assumptions),
                artifacts_dir=artifacts_dir,
                state=state,
            )
            self._set_live_status(
                run_id,
                status="needs_clarification",
                current_phase="waiting_for_user",
                clarification_questions=[plan_action.clarification_question],
                assumptions=list(state.assumptions),
                planner_actions=list(state.planner_actions),
                detail="Planner requested clarification",
            )
            self._persist_manifest(manifest)
            return manifest

        if plan_action.plan_dag is None:
            raise RuntimeError("Planner did not produce a PlanDAG.")

        state.last_plan = plan_action.plan_dag.to_dict()
        context = AgentExecutionContext(
            run_id=run_id,
            artifacts_dir=artifacts_dir,
            search_backend=self.search_backend,
            working_store=self.working_store,
            llm_client=self.llm_client,
            python_runner=self.python_runner,
            runtime=self.runtime,
            state=state,
            event_callback=lambda payload: self._record_step_event(run_id, payload),
        )

        self._set_live_status(run_id, current_phase="executing", detail="Executing plan DAG")
        snapshot = asyncio.run(self.executor.execute(plan_action.plan_dag, context))
        plan_dags = [plan_action.plan_dag.to_dict()]

        if snapshot.failures:
            state.failures = [item.to_dict() for item in snapshot.failures]
            self._set_live_status(run_id, current_phase="revising_after_failure", detail="Revising plan after failure")
            revised = self.orchestrator.revise_after_failure(state, snapshot.failures[0])
            if revised is not None and revised.plan_dag is not None:
                state.planner_calls_used += 1
                state.planner_actions.append(revised.to_dict())
                self._set_live_status(run_id, planner_actions=list(state.planner_actions))
                revised_snapshot = asyncio.run(self.executor.execute(revised.plan_dag, context))
                plan_dags.append(revised.plan_dag.to_dict())
                snapshot = revised_snapshot

        self._set_live_status(run_id, current_phase="final_synthesis", detail="Synthesizing grounded answer")
        final_answer = self.orchestrator.synthesize(state, snapshot)
        manifest = AgentRunManifest(
            run_id=run_id,
            question=question,
            rewritten_question=state.rewritten_question,
            status=snapshot.status,
            clarification_questions=[],
            assumptions=list(state.assumptions),
            planner_actions=list(state.planner_actions),
            plan_dags=plan_dags,
            selected_docs=list(snapshot.selected_docs),
            node_records=list(snapshot.node_records),
            provenance_records=list(snapshot.provenance_records),
            evidence_table=list(self.orchestrator._extract_evidence(snapshot)),
            final_answer=final_answer,
            artifacts_dir=str(artifacts_dir),
            failures=list(snapshot.failures),
            metadata={
                "planner_calls_used": state.planner_calls_used,
                "clarification_history": list(state.clarification_history),
            },
        )
        self._persist_manifest(manifest)
        return manifest

    def handle_query(
        self,
        question: str,
        *,
        force_answer: bool = False,
        no_cache: bool = False,
        clarification_history: list[str] | None = None,
    ) -> AgentRunManifest:
        run_id = f"agent_{uuid.uuid4().hex[:12]}"
        return self._run_query(
            run_id,
            question,
            force_answer=force_answer,
            no_cache=no_cache,
            clarification_history=clarification_history,
        )

    def submit_query(
        self,
        question: str,
        *,
        force_answer: bool = False,
        no_cache: bool = False,
        clarification_history: list[str] | None = None,
    ) -> LiveRunStatus:
        run_id = f"agent_{uuid.uuid4().hex[:12]}"
        status = self._set_live_status(
            run_id,
            question=question,
            status="queued",
            current_phase="queued",
            detail="Queued for execution",
        )

        def _runner() -> None:
            try:
                self._run_query(
                    run_id,
                    question,
                    force_answer=force_answer,
                    no_cache=no_cache,
                    clarification_history=clarification_history,
                )
            except Exception as exc:
                self._set_live_status(
                    run_id,
                    status="failed",
                    current_phase="failed",
                    detail=str(exc),
                )

        thread = threading.Thread(target=_runner, daemon=True, name=f"corpusagent2-{run_id}")
        thread.start()
        return status

    def _persist_manifest(self, manifest: AgentRunManifest) -> None:
        manifest_path = Path(manifest.artifacts_dir) / "run_manifest.json"
        save_agent_manifest(manifest_path, manifest.to_dict())
        try:
            self.working_store.record_output(manifest.run_id, "final_answer", manifest.final_answer.to_dict())
            self.working_store.finalize_run(manifest.run_id, manifest.status)
        except Exception:
            pass
        self._set_live_status(
            manifest.run_id,
            question=manifest.question,
            status=manifest.status,
            current_phase="completed" if manifest.status in {"completed", "partial"} else manifest.status,
            detail="Run finished",
            assumptions=list(manifest.assumptions),
            planner_actions=list(manifest.planner_actions),
            clarification_questions=list(manifest.clarification_questions),
            final_manifest_path=str(manifest_path),
        )

    def get_run_status(self, run_id: str) -> dict[str, Any]:
        with self._run_lock:
            live = self._live_runs.get(run_id)
            if live is not None:
                return live.to_dict()
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        manifest_path = self.config.outputs_root / run_id / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run manifest not found for run_id={run_id}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
