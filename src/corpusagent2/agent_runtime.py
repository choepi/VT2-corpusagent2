from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import threading
import uuid
from typing import Any

from .agent_backends import (
    HybridSearchBackend,
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
from .agent_policy import rejection_reason_for_question, rewrite_special_cases
from .app_config import load_project_configuration
from .llm_provider import LLMClient, LLMProviderConfig, OpenAICompatibleLLMClient
from .retrieval import pg_dsn_from_env, pg_table_from_env
from .run_manifest import FinalAnswerPayload
from .runtime_context import CorpusRuntime
from .seed import runtime_device_report
from .tool_registry import ToolRegistry
from .python_runner_service import DockerPythonRunnerService

TERMINAL_RUN_STATUSES = {"completed", "partial", "failed", "rejected", "needs_clarification", "aborted"}


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

    def _record_llm_trace(
        self,
        state: AgentRunState,
        *,
        stage: str,
        trace: dict[str, Any] | None = None,
        used_fallback: bool = False,
        error: str = "",
        note: str = "",
    ) -> None:
        entry = {
            "stage": stage,
            "provider_name": self.llm_config.provider_name if self.llm_client is not None else "heuristic",
            "base_url": self.llm_config.base_url if self.llm_client is not None else "",
            "model": trace.get("model", "") if trace else "",
            "temperature": trace.get("temperature", 0.0) if trace else 0.0,
            "messages": trace.get("messages", []) if trace else [],
            "raw_text": trace.get("raw_text", "") if trace else "",
            "parsed_json": trace.get("parsed_json", {}) if trace else {},
            "used_fallback": used_fallback,
            "error": error,
            "note": note,
        }
        state.llm_traces.append(entry)

    def _question_with_clarifications(self, state: AgentRunState) -> str:
        history = [str(item).strip() for item in state.clarification_history if str(item).strip()]
        if not history:
            return state.question
        suffix = "\n".join(f"- {item}" for item in history)
        return f"{state.question}\n\nUser clarification history:\n{suffix}"

    def _extract_date_window(self, text: str) -> dict[str, str]:
        year_values = sorted({int(item) for item in re.findall(r"\b(?:19|20)\d{2}\b", text)})
        if not year_values:
            return {}
        if len(year_values) == 1:
            year = year_values[0]
            return {"date_from": f"{year}-01-01", "date_to": f"{year}-12-31"}
        return {
            "date_from": f"{year_values[0]}-01-01",
            "date_to": f"{year_values[-1]}-12-31",
        }

    def _compact_query_terms(self, text: str, preferred_terms: list[str]) -> str:
        found_terms: list[str] = []
        lowered = text.lower()
        for term in preferred_terms:
            if term.lower() in lowered and term not in found_terms:
                found_terms.append(term)
        if found_terms:
            return " ".join(found_terms)
        tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text) if len(token) > 2]
        stopwords = {
            "how", "did", "from", "into", "with", "what", "when", "which", "that", "this", "those",
            "coverage", "framing", "frame", "around", "during", "between", "across", "their", "there",
            "correspond", "corresponded", "drawdowns", "stock", "media", "shift", "shifted",
        }
        filtered = [token for token in tokens if token.lower() not in stopwords]
        return " ".join(filtered[:8]).strip()

    def _broad_scope_clarification_is_sufficient(self, state: AgentRunState) -> tuple[bool, list[str]]:
        if not state.clarification_history:
            return False, []
        question_text = f"{state.question}\n" + "\n".join(state.clarification_history)
        lowered = question_text.lower()
        assumptions: list[str] = []
        sufficient = False

        has_broad_region = any(
            token in lowered
            for token in ("europe", "european", "overall europe", "across europe", "europe-wide")
        )
        has_broad_aggregation = any(
            token in lowered
            for token in ("overall", "aggregate", "broadly", "quantitative", "quantitativ")
        )
        asks_regional = any(token in state.question.lower() for token in ("region", "regions", "across regions"))
        asks_outlet_types = "outlet type" in state.question.lower() or "outlet types" in state.question.lower()

        if has_broad_region and asks_regional:
            sufficient = True
            assumptions.append("Interpret the requested region scope as Europe-wide overall coverage rather than country-by-country subregions.")
        if has_broad_aggregation:
            sufficient = True
            assumptions.append("Prefer a quantitative aggregate summary where possible and note any metadata limits explicitly.")
        if asks_outlet_types and sufficient:
            assumptions.append("Use available outlet/source metadata as the outlet-type proxy; if outlet-type labels are sparse, report the overall pattern and the limitation.")
        return sufficient, assumptions

    def rephrase_or_clarify(self, state: AgentRunState) -> PlannerAction:
        rejection_reason = rejection_reason_for_question(state.question)
        if rejection_reason:
            return PlannerAction(action="grounded_rejection", rejection_reason=rejection_reason)

        enriched_question = self._question_with_clarifications(state)
        rewritten, assumptions = rewrite_special_cases(enriched_question)
        if self.llm_client is None:
            self._record_llm_trace(
                state,
                stage="rephrase_or_clarify",
                used_fallback=True,
                note="No LLM client configured; heuristic clarification policy used.",
            )
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
            trace = self.llm_client.complete_json_trace(
                messages,
                model=self.llm_config.planner_model,
                temperature=0.0,
            )
            payload = dict(trace["parsed_json"])
            self._record_llm_trace(state, stage="rephrase_or_clarify", trace=trace)
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
            sufficient, clarification_assumptions = self._broad_scope_clarification_is_sufficient(state)
            if action.action == "ask_clarification" and sufficient:
                return PlannerAction(
                    action="accept_with_assumptions",
                    rewritten_question=action.rewritten_question or rewritten or state.question,
                    assumptions=list(dict.fromkeys(list(action.assumptions) + assumptions + clarification_assumptions)),
                    message="Broad-scope clarification accepted; proceeding with explicit assumptions.",
                )
            if not action.rewritten_question:
                action.rewritten_question = rewritten
            if assumptions:
                action.assumptions = list(dict.fromkeys(list(action.assumptions) + assumptions))
            return action
        except Exception as exc:
            self._record_llm_trace(
                state,
                stage="rephrase_or_clarify",
                used_fallback=True,
                error=str(exc),
                note="LLM rephrase step failed; heuristic fallback used.",
            )
            return PlannerAction(
                action="accept_with_assumptions",
                rewritten_question=rewritten,
                assumptions=assumptions,
            )

    def _heuristic_plan(self, state: AgentRunState) -> PlannerAction:
        rewritten = state.rewritten_question or self._question_with_clarifications(state)
        text = rewritten.lower()
        if "distribution" in text and "noun" in text:
            dag = AgentPlanDAG(
                nodes=[
                    AgentPlanNode("search", "db_search", {"top_k": 40, "retrieval_mode": "hybrid", "use_rerank": False}),
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
                    AgentPlanNode("search", "db_search", {"top_k": 60, "retrieval_mode": "hybrid", "use_rerank": True, "rerank_top_k": 30}),
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
                    AgentPlanNode("search", "db_search", {"top_k": 80, "date_to": "2022-02-23", "retrieval_mode": "hybrid", "use_rerank": True, "rerank_top_k": 40}),
                    AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                    AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                    AgentPlanNode("claim_spans", "claim_span_extract", depends_on=["fetch"]),
                    AgentPlanNode("claim_scores", "claim_strength_score", depends_on=["claim_spans"]),
                    AgentPlanNode("evidence", "build_evidence_table", depends_on=["claim_scores"]),
                ],
                metadata={"question_family": "prediction_evidence"},
            )
            return PlannerAction(action="emit_plan_dag", rewritten_question=state.rewritten_question, plan_dag=dag)
        framing_terms = ["framing", "frame", "privacy", "regulation", "innovation", "growth", "scandal", "cambridge analytica", "facebook", "meta"]
        if any(term in text for term in framing_terms) and any(term in text for term in ["shift", "changed", "change", "correspond", "over time"]):
            search_inputs: dict[str, Any] = {
                "top_k": 100,
                "retrieval_mode": "hybrid",
                "use_rerank": True,
                "rerank_top_k": 50,
                "query": self._compact_query_terms(
                    rewritten,
                    [
                        "Facebook",
                        "Meta",
                        "Cambridge Analytica",
                        "privacy",
                        "regulation",
                        "innovation",
                        "growth",
                        "stock",
                        "drawdown",
                    ],
                ),
            }
            search_inputs.update(self._extract_date_window(rewritten))
            nodes = [
                AgentPlanNode("search", "db_search", search_inputs),
                AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                AgentPlanNode("keyterms", "extract_keyterms", depends_on=["fetch"]),
                AgentPlanNode("topics", "topic_model", {"num_topics": 6}, depends_on=["fetch"]),
                AgentPlanNode("sentiment", "sentiment", depends_on=["fetch"]),
                AgentPlanNode("series", "time_series_aggregate", depends_on=["sentiment"]),
                AgentPlanNode("changes", "change_point_detect", depends_on=["series"], optional=True),
                AgentPlanNode("plot_sentiment", "plot_artifact", {"plot_name": "framing_sentiment_series"}, depends_on=["series"], optional=True),
                AgentPlanNode("plot_topics", "plot_artifact", {"plot_name": "framing_topics"}, depends_on=["topics"], optional=True),
            ]
            assumptions: list[str] = []
            if any(term in text for term in ["stock", "drawdown", "share price", "market", "valuation"]):
                assumptions.append(
                    "External market data is not automatically attached in the first-trial runtime, so the stock-drawdown correspondence can only be assessed if matching series are supplied later."
                )
            dag = AgentPlanDAG(
                nodes=nodes,
                metadata={
                    "question_family": "framing_shift",
                    "requires_external_series": bool(assumptions),
                },
            )
            return PlannerAction(
                action="emit_plan_dag",
                rewritten_question=state.rewritten_question or rewritten,
                assumptions=assumptions,
                plan_dag=dag,
            )
        dag = AgentPlanDAG(
            nodes=[
                AgentPlanNode("search", "db_search", {"top_k": 40, "retrieval_mode": "hybrid", "use_rerank": True, "rerank_top_k": 25}),
                AgentPlanNode("fetch", "fetch_documents", depends_on=["search"]),
                AgentPlanNode("working_set", "create_working_set", depends_on=["fetch"]),
                AgentPlanNode("keyterms", "extract_keyterms", depends_on=["fetch"], optional=True),
            ],
            metadata={"question_family": "generic"},
        )
        return PlannerAction(action="emit_plan_dag", rewritten_question=state.rewritten_question, plan_dag=dag)

    def plan(self, state: AgentRunState) -> PlannerAction:
        if self.llm_client is None:
            self._record_llm_trace(
                state,
                stage="plan",
                used_fallback=True,
                note="No LLM client configured; heuristic planning policy used.",
            )
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
        repair_system_message = (
            "You previously returned an invalid or empty plan for a corpus-analysis agent. "
            "Return valid JSON with keys action, rewritten_question, assumptions, clarification_question, rejection_reason, message, plan_dag. "
            "Allowed actions: ask_clarification, emit_plan_dag, grounded_rejection. "
            "If action is emit_plan_dag, plan_dag.nodes must contain at least one executable node with node_id, capability, inputs, and depends_on."
        )
        try:
            trace = self.llm_client.complete_json_trace(
                messages,
                model=self.llm_config.planner_model,
                temperature=0.0,
            )
            payload = dict(trace["parsed_json"])
            self._record_llm_trace(state, stage="plan", trace=trace)
            action = PlannerAction.from_dict(payload)
            if action.action == "emit_plan_dag" and action.plan_dag is None:
                repair_messages = [
                    {"role": "system", "content": repair_system_message},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": state.question,
                                "rewritten_question": state.rewritten_question,
                                "available_capabilities": state.available_capabilities,
                                "corpus_schema": state.corpus_schema,
                                "previous_invalid_output": trace.get("raw_text", ""),
                            },
                            ensure_ascii=True,
                        ),
                    },
                ]
                repair_trace = self.llm_client.complete_json_trace(
                    repair_messages,
                    model=self.llm_config.planner_model,
                    temperature=0.0,
                )
                self._record_llm_trace(state, stage="plan_repair", trace=repair_trace)
                action = PlannerAction.from_dict(dict(repair_trace["parsed_json"]))
                if action.action == "emit_plan_dag" and action.plan_dag is None:
                    raise ValueError("Planner returned no executable nodes.")
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
        except Exception as exc:
            self._record_llm_trace(
                state,
                stage="plan",
                used_fallback=True,
                error=str(exc),
                note="LLM planning step failed; heuristic planning fallback used.",
            )
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
            self._record_llm_trace(
                state,
                stage="final_synthesis",
                used_fallback=True,
                note="No LLM client configured; fallback synthesis used.",
            )
            return self._apply_answer_guardrails(
                state,
                snapshot,
                self._fallback_synthesis(state, evidence_rows, summary, snapshot),
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the grounded synthesis module for a corpus-analysis agent. "
                    "Return JSON with keys answer_text, evidence_items, artifacts_used, unsupported_parts, caveats, claim_verdicts. "
                    "Use only the provided summaries, tool outputs, and evidence. "
                    "Do not claim direct correspondence to stock-price moves or external market behavior unless an external series was explicitly attached."
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
                        "has_external_series": self._has_external_series(snapshot),
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        try:
            trace = self.llm_client.complete_json_trace(
                messages,
                model=self.llm_config.synthesis_model,
                temperature=0.1,
            )
            payload = dict(trace["parsed_json"])
            self._record_llm_trace(state, stage="final_synthesis", trace=trace)
            answer = FinalAnswerPayload.from_payload(payload)
            if evidence_rows:
                answer.evidence_items = evidence_rows
            return self._apply_answer_guardrails(state, snapshot, answer)
        except Exception as exc:
            self._record_llm_trace(
                state,
                stage="final_synthesis",
                used_fallback=True,
                error=str(exc),
                note="LLM synthesis failed; fallback synthesis used.",
            )
            return self._apply_answer_guardrails(
                state,
                snapshot,
                self._fallback_synthesis(state, evidence_rows, summary, snapshot),
            )

    def _extract_evidence(self, snapshot: AgentExecutionSnapshot) -> list[dict[str, Any]]:
        for node_id, result in snapshot.node_results.items():
            payload = result.payload
            if isinstance(payload, dict) and "rows" in payload and payload["rows"]:
                first = payload["rows"][0]
                if {"doc_id", "outlet", "date", "excerpt", "score"}.issubset(first.keys()):
                    rows = []
                    for row in payload["rows"]:
                        copied = dict(row)
                        score = copied.get("score", 0.0)
                        try:
                            numeric = float(score)
                        except (TypeError, ValueError):
                            numeric = 0.0
                        copied.setdefault("score_display", f"{numeric:.4f}".rstrip("0").rstrip(".") if abs(numeric) >= 0.01 else (f"{numeric:.2e}" if abs(numeric) > 0 else "0"))
                        rows.append(copied)
                    return rows
        fallback_rows: list[dict[str, Any]] = []
        for row in snapshot.selected_docs[:10]:
            doc_id = str(row.get("doc_id", "")).strip()
            if not doc_id:
                continue
            excerpt_source = (
                row.get("snippet")
                or row.get("excerpt")
                or row.get("text")
                or row.get("body")
                or row.get("title")
                or ""
            )
            excerpt = str(excerpt_source).strip().replace("\n", " ")
            if len(excerpt) > 280:
                excerpt = excerpt[:277].rstrip() + "..."
            fallback_rows.append(
                {
                    "doc_id": doc_id,
                    "outlet": str(
                        row.get("outlet")
                        or row.get("source")
                        or row.get("source_domain")
                        or ""
                    ),
                    "date": str(row.get("published_at") or row.get("date") or row.get("year") or ""),
                    "excerpt": excerpt,
                    "score": float(row.get("score", 0.0) or 0.0),
                    "score_display": str(row.get("score_display") or ""),
                }
            )
        return fallback_rows

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
            elif "term" in first and "score" in first:
                summary["keyterms"] = rows[:15]
            elif "topic_id" in first and "top_terms" in first:
                summary["topics"] = rows[:10]
            elif "label" in first and "time_bin" in first and "score" in first:
                by_time: dict[str, dict[str, float]] = {}
                for row in rows:
                    time_bin = str(row.get("time_bin", "unknown"))
                    bucket = by_time.setdefault(time_bin, {"count": 0.0, "score_total": 0.0})
                    bucket["count"] += 1.0
                    bucket["score_total"] += float(row.get("score", 0.0))
                summary["sentiment_trend"] = [
                    {
                        "time_bin": time_bin,
                        "average_score": round(values["score_total"] / max(values["count"], 1.0), 4),
                        "documents": int(values["count"]),
                    }
                    for time_bin, values in sorted(by_time.items())
                ]
            elif "delta" in first and "time_bin" in first:
                summary["change_points"] = rows[:10]
            elif {"doc_id", "outlet", "date", "excerpt", "score"}.issubset(first.keys()):
                summary["evidence_rows"] = rows[:10]
        return summary

    def _has_external_series(self, snapshot: AgentExecutionSnapshot) -> bool:
        for node_id, result in snapshot.node_results.items():
            payload = result.payload if isinstance(result.payload, dict) else {}
            if node_id == "join_external_series":
                rows = list(payload.get("rows", []))
                if rows:
                    return True
        return False

    def _apply_answer_guardrails(
        self,
        state: AgentRunState,
        snapshot: AgentExecutionSnapshot,
        answer: FinalAnswerPayload,
    ) -> FinalAnswerPayload:
        question_text = f"{state.question} {state.rewritten_question}".lower()
        needs_market_guardrail = any(
            term in question_text
            for term in ["stock", "drawdown", "share price", "valuation", "market"]
        )
        if needs_market_guardrail and not self._has_external_series(snapshot):
            note = (
                "This run did not attach an external market time series, so any direct correspondence to stock drawdowns remains unverified."
            )
            unsupported = "Direct stock-price or drawdown correspondence was not verified because no external market series was attached."
            if note not in answer.caveats:
                answer.caveats.append(note)
            if unsupported not in answer.unsupported_parts:
                answer.unsupported_parts.append(unsupported)
            lowered_answer = answer.answer_text.lower()
            if "unverified" not in lowered_answer and "not verified" not in lowered_answer and "not attach" not in lowered_answer:
                answer.answer_text = f"{answer.answer_text.rstrip()} {note}".strip()
        if answer.evidence_items:
            answer.evidence_items = list(answer.evidence_items)[:10]
        return answer

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
        elif "topics" in summary or "keyterms" in summary or "sentiment_trend" in summary:
            topic_text = ""
            keyterm_text = ""
            sentiment_text = ""
            if "topics" in summary:
                topic_bits = []
                for topic in summary["topics"][:4]:
                    top_terms = ", ".join(str(term) for term in topic.get("top_terms", [])[:5])
                    topic_bits.append(f"{topic.get('time_bin', 'all')}: {top_terms}")
                topic_text = "Topic slices: " + " | ".join(topic_bits) + "."
            if "keyterms" in summary:
                keyterm_text = "Key terms: " + ", ".join(
                    f"{row.get('term', '')} ({row.get('score', 0):.2f})"
                    for row in summary["keyterms"][:8]
                ) + "."
            if "sentiment_trend" in summary:
                sentiment_text = "Sentiment over time: " + " | ".join(
                    f"{row.get('time_bin', 'unknown')} avg={row.get('average_score', 0):.2f}"
                    for row in summary["sentiment_trend"][:6]
                ) + "."
            answer_text = " ".join(part for part in [topic_text, keyterm_text, sentiment_text] if part).strip()
            if any(term in state.question.lower() for term in ["stock", "drawdown", "share price", "valuation", "market"]):
                caveats.append(
                    "The current run did not attach an external market series, so stock-price correspondence cannot be measured directly yet."
                )
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
            evidence_items=evidence_rows,
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
        self._startup_llm_config = llm_config or LLMProviderConfig.from_env()
        self.llm_config = self._startup_llm_config
        self.llm_client = llm_client or OpenAICompatibleLLMClient(self.llm_config)
        self.registry = registry or build_agent_registry()
        self.search_backend = search_backend or self._build_search_backend()
        self.working_store = working_store or self._build_working_store()
        self.python_runner = python_runner or DockerPythonRunnerService()
        self.orchestrator = MagicBoxOrchestrator(self.llm_client, self.llm_config)
        self.executor = AsyncPlanExecutor(self.registry)
        self._live_runs: dict[str, LiveRunStatus] = {}
        self._run_cancel_events: dict[str, threading.Event] = {}
        self._run_threads: dict[str, threading.Thread] = {}
        self._run_lock = threading.Lock()
        self._llm_override_active = False

    def _build_search_backend(self):
        try:
            lexical_backend = OpenSearchBackend(OpenSearchConfig.from_env())
        except Exception:
            lexical_backend = None
        return HybridSearchBackend(self.runtime, lexical_backend=lexical_backend)

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

    def _active_run_ids(self) -> list[str]:
        with self._run_lock:
            return [
                run_id
                for run_id, status in self._live_runs.items()
                if status.status not in TERMINAL_RUN_STATUSES
            ]

    def update_llm_runtime_settings(
        self,
        *,
        use_openai: bool,
        planner_model: str = "",
        synthesis_model: str = "",
    ) -> dict[str, Any]:
        active_run_ids = self._active_run_ids()
        if active_run_ids:
            raise RuntimeError(
                "LLM settings can only be changed when no query is running. Abort active runs first."
            )

        resolved = self._startup_llm_config.with_runtime_overrides(
            use_openai=use_openai,
            planner_model=planner_model or None,
            synthesis_model=synthesis_model or None,
        )
        self.llm_config = resolved
        self.llm_client = OpenAICompatibleLLMClient(resolved)
        self.orchestrator = MagicBoxOrchestrator(self.llm_client, self.llm_config)
        self._llm_override_active = (
            resolved.use_openai != self._startup_llm_config.use_openai
            or resolved.planner_model != self._startup_llm_config.planner_model
            or resolved.synthesis_model != self._startup_llm_config.synthesis_model
            or resolved.base_url != self._startup_llm_config.base_url
        )
        return self.runtime_info()

    def reset_llm_runtime_settings(self) -> dict[str, Any]:
        active_run_ids = self._active_run_ids()
        if active_run_ids:
            raise RuntimeError(
                "LLM settings can only be reset when no query is running. Abort active runs first."
            )
        self.llm_config = self._startup_llm_config
        self.llm_client = OpenAICompatibleLLMClient(self.llm_config)
        self.orchestrator = MagicBoxOrchestrator(self.llm_client, self.llm_config)
        self._llm_override_active = False
        return self.runtime_info()

    def runtime_info(self) -> dict[str, Any]:
        provider_modules = {}
        for module_name in ["spacy", "textacy", "stanza", "nltk", "gensim", "flair", "textblob", "torch"]:
            provider_modules[module_name] = importlib.util.find_spec(module_name) is not None
        device_report = runtime_device_report()
        llm_warnings: list[str] = []
        if self.llm_config.use_openai and "hermes.ai.unturf.com" in self.llm_config.base_url:
            llm_warnings.append("OpenAI mode is enabled but the resolved base URL still points to UncloseAI/Hermes.")
        if not self.llm_config.use_openai and "api.openai.com" in self.llm_config.base_url:
            llm_warnings.append("UncloseAI mode is enabled but the resolved base URL still points to api.openai.com.")
        gpu_warnings: list[str] = []
        if device_report.get("nvidia_smi_ok") and not device_report.get("cuda_available"):
            gpu_warnings.append(
                "An NVIDIA GPU is visible through nvidia-smi, but PyTorch is a CPU-only build. Reinstall a CUDA-enabled torch wheel to use the GPU."
            )

        provider_orders = {
            capability.lower().replace("CORPUSAGENT2_PROVIDER_ORDER_", ""): value
            for capability, value in os.environ.items()
            if capability.startswith("CORPUSAGENT2_PROVIDER_ORDER_")
        }
        return {
            "llm": {
                "use_openai": self.llm_config.use_openai,
                "provider_name": self.llm_config.provider_name,
                "base_url": self.llm_config.base_url,
                "planner_model": self.llm_config.planner_model,
                "synthesis_model": self.llm_config.synthesis_model,
                "api_key_present": bool(self.llm_config.api_key),
                "override_active": self._llm_override_active,
                "startup_defaults": {
                    "use_openai": self._startup_llm_config.use_openai,
                    "provider_name": self._startup_llm_config.provider_name,
                    "base_url": self._startup_llm_config.base_url,
                    "planner_model": self._startup_llm_config.planner_model,
                    "synthesis_model": self._startup_llm_config.synthesis_model,
                },
                "warnings": llm_warnings,
            },
            "device": {
                **device_report,
                "warnings": gpu_warnings,
            },
            "providers_installed": provider_modules,
            "provider_order": provider_orders,
            "capability_count": len(self.registry.list_tools()),
            "retrieval": {
                "backend": self.runtime.retrieval_backend,
                "default_mode": os.getenv("CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE", "hybrid"),
                "rerank_enabled": os.getenv("CORPUSAGENT2_RETRIEVAL_USE_RERANK", "true").strip().lower()
                not in {"0", "false", "no", "off"},
                "rerank_top_k": int(os.getenv("CORPUSAGENT2_RETRIEVAL_RERANK_TOP_K", "25").strip() or "25"),
                "fusion_k": int(os.getenv("CORPUSAGENT2_RETRIEVAL_FUSION_K", "60").strip() or "60"),
                "dense_model_id": self.runtime.dense_model_id,
                "rerank_model_id": self.runtime.rerank_model_id,
            },
            "analysis_notes": [
                "UI model/provider changes are process-local runtime overrides. They affect future runs only and reset when the backend restarts unless you also update .env.",
                "Classical spaCy/textacy/gensim analytics are usually CPU-bound even when CUDA is available.",
                "GPU is mainly relevant for torch- or Flair-backed models and only when those providers are selected.",
                "Per-node provider choice and artifacts are captured in the run manifest so you can verify what really ran.",
            ],
            "active_run_ids": self._active_run_ids(),
        }

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

    def _register_cancel_event(self, run_id: str) -> threading.Event:
        with self._run_lock:
            event = self._run_cancel_events.get(run_id)
            if event is None:
                event = threading.Event()
                self._run_cancel_events[run_id] = event
            return event

    def _is_cancelled(self, run_id: str) -> bool:
        with self._run_lock:
            event = self._run_cancel_events.get(run_id)
            return bool(event is not None and event.is_set())

    def _build_aborted_manifest(
        self,
        *,
        run_id: str,
        question: str,
        rewritten_question: str,
        state: AgentRunState,
        artifacts_dir: Path,
        snapshot: AgentExecutionSnapshot | None = None,
        plan_dags: list[dict[str, Any]] | None = None,
        clarification_questions: list[str] | None = None,
    ) -> AgentRunManifest:
        empty_snapshot = AgentExecutionSnapshot(
            node_records=[],
            node_results={},
            failures=[],
            provenance_records=[],
            selected_docs=[],
            status="aborted",
        )
        resolved_snapshot = snapshot or empty_snapshot
        return AgentRunManifest(
            run_id=run_id,
            question=question,
            rewritten_question=rewritten_question or question,
            status="aborted",
            clarification_questions=list(clarification_questions or []),
            assumptions=list(state.assumptions),
            planner_actions=list(state.planner_actions),
            plan_dags=list(plan_dags or []),
            selected_docs=list(resolved_snapshot.selected_docs),
            node_records=list(resolved_snapshot.node_records),
            provenance_records=list(resolved_snapshot.provenance_records),
            evidence_table=list(self.orchestrator._extract_evidence(resolved_snapshot)),
            final_answer=FinalAnswerPayload(
                answer_text="Run aborted by user.",
                caveats=["Run was aborted before completion; any partial outputs may be incomplete."],
            ),
            artifacts_dir=str(artifacts_dir),
            failures=list(resolved_snapshot.failures),
            metadata={
                "clarification_history": list(state.clarification_history),
                "llm_traces": list(state.llm_traces),
                "runtime_info": self.runtime_info(),
                "aborted": True,
            },
        )

    def _maybe_abort(
        self,
        *,
        run_id: str,
        question: str,
        rewritten_question: str,
        state: AgentRunState,
        artifacts_dir: Path,
        snapshot: AgentExecutionSnapshot | None = None,
        plan_dags: list[dict[str, Any]] | None = None,
        clarification_questions: list[str] | None = None,
    ) -> AgentRunManifest | None:
        if not self._is_cancelled(run_id):
            return None
        self._set_live_status(
            run_id,
            status="aborted",
            current_phase="aborted",
            detail="Run aborted by user",
            assumptions=list(state.assumptions),
            planner_actions=list(state.planner_actions),
            llm_traces=list(state.llm_traces),
            clarification_questions=list(clarification_questions or []),
        )
        manifest = self._build_aborted_manifest(
            run_id=run_id,
            question=question,
            rewritten_question=rewritten_question,
            state=state,
            artifacts_dir=artifacts_dir,
            snapshot=snapshot,
            plan_dags=plan_dags,
            clarification_questions=clarification_questions,
        )
        self._persist_manifest(manifest)
        return manifest

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
                "llm_traces": list(state.llm_traces),
                "runtime_info": self.runtime_info(),
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
        maybe_aborted = self._maybe_abort(
            run_id=run_id,
            question=question,
            rewritten_question=question,
            state=state,
            artifacts_dir=artifacts_dir,
        )
        if maybe_aborted is not None:
            return maybe_aborted

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
        self._set_live_status(run_id, planner_actions=list(state.planner_actions), llm_traces=list(state.llm_traces))
        if rephrase_action.assumptions:
            state.assumptions = list(dict.fromkeys(state.assumptions + rephrase_action.assumptions))
            self._set_live_status(run_id, assumptions=list(state.assumptions))
        maybe_aborted = self._maybe_abort(
            run_id=run_id,
            question=question,
            rewritten_question=rephrase_action.rewritten_question or question,
            state=state,
            artifacts_dir=artifacts_dir,
        )
        if maybe_aborted is not None:
            return maybe_aborted
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
                metadata={
                    "llm_traces": list(state.llm_traces),
                    "runtime_info": self.runtime_info(),
                },
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
        if plan_action.assumptions:
            state.assumptions = list(dict.fromkeys(state.assumptions + plan_action.assumptions))
        self._set_live_status(run_id, planner_actions=list(state.planner_actions), llm_traces=list(state.llm_traces))
        maybe_aborted = self._maybe_abort(
            run_id=run_id,
            question=question,
            rewritten_question=state.rewritten_question,
            state=state,
            artifacts_dir=artifacts_dir,
            clarification_questions=[plan_action.clarification_question] if plan_action.clarification_question else [],
        )
        if maybe_aborted is not None:
            return maybe_aborted
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
            cancel_requested=lambda: self._is_cancelled(run_id),
        )

        self._set_live_status(run_id, current_phase="executing", detail="Executing plan DAG")
        snapshot = asyncio.run(self.executor.execute(plan_action.plan_dag, context))
        plan_dags = [plan_action.plan_dag.to_dict()]
        maybe_aborted = self._maybe_abort(
            run_id=run_id,
            question=question,
            rewritten_question=state.rewritten_question,
            state=state,
            artifacts_dir=artifacts_dir,
            snapshot=snapshot,
            plan_dags=plan_dags,
        )
        if maybe_aborted is not None:
            return maybe_aborted

        if snapshot.failures:
            state.failures = [item.to_dict() for item in snapshot.failures]
            self._set_live_status(run_id, current_phase="revising_after_failure", detail="Revising plan after failure")
            revised = self.orchestrator.revise_after_failure(state, snapshot.failures[0])
            if revised is not None and revised.plan_dag is not None:
                state.planner_calls_used += 1
                state.planner_actions.append(revised.to_dict())
                if revised.assumptions:
                    state.assumptions = list(dict.fromkeys(state.assumptions + revised.assumptions))
                self._set_live_status(run_id, planner_actions=list(state.planner_actions), llm_traces=list(state.llm_traces))
                revised_snapshot = asyncio.run(self.executor.execute(revised.plan_dag, context))
                plan_dags.append(revised.plan_dag.to_dict())
                snapshot = revised_snapshot
                maybe_aborted = self._maybe_abort(
                    run_id=run_id,
                    question=question,
                    rewritten_question=state.rewritten_question,
                    state=state,
                    artifacts_dir=artifacts_dir,
                    snapshot=snapshot,
                    plan_dags=plan_dags,
                )
                if maybe_aborted is not None:
                    return maybe_aborted

        self._set_live_status(run_id, current_phase="final_synthesis", detail="Synthesizing grounded answer")
        maybe_aborted = self._maybe_abort(
            run_id=run_id,
            question=question,
            rewritten_question=state.rewritten_question,
            state=state,
            artifacts_dir=artifacts_dir,
            snapshot=snapshot,
            plan_dags=plan_dags,
        )
        if maybe_aborted is not None:
            return maybe_aborted
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
                "llm_traces": list(state.llm_traces),
                "runtime_info": self.runtime_info(),
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
        self._register_cancel_event(run_id)

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
            finally:
                with self._run_lock:
                    self._run_threads.pop(run_id, None)

        thread = threading.Thread(target=_runner, daemon=True, name=f"corpusagent2-{run_id}")
        with self._run_lock:
            self._run_threads[run_id] = thread
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
            llm_traces=list(manifest.metadata.get("llm_traces", [])),
            clarification_questions=list(manifest.clarification_questions),
            final_manifest_path=str(manifest_path),
        )

    def get_run_status(self, run_id: str) -> dict[str, Any]:
        with self._run_lock:
            live = self._live_runs.get(run_id)
            if live is not None:
                return live.to_dict()
        return self.get_run(run_id)

    def abort_run(self, run_id: str) -> dict[str, Any]:
        with self._run_lock:
            status = self._live_runs.get(run_id)
            if status is None:
                raise FileNotFoundError(f"Run not found for run_id={run_id}")
            current_status = str(status.status)
            if current_status in TERMINAL_RUN_STATUSES:
                return status.to_dict()
            event = self._run_cancel_events.get(run_id)
            if event is None:
                event = threading.Event()
                self._run_cancel_events[run_id] = event
            event.set()
        return self._set_live_status(
            run_id,
            status="aborting",
            current_phase="aborting",
            detail="Abort requested; waiting for current step to stop",
        ).to_dict()

    def abort_all_runs(self) -> dict[str, Any]:
        aborted_run_ids: list[str] = []
        with self._run_lock:
            run_ids = list(self._live_runs.keys())
        for run_id in run_ids:
            with self._run_lock:
                status = self._live_runs.get(run_id)
                current_status = str(status.status) if status is not None else ""
            if current_status and current_status not in TERMINAL_RUN_STATUSES and current_status != "aborting":
                self.abort_run(run_id)
                aborted_run_ids.append(run_id)
        return {"aborted_run_ids": aborted_run_ids, "count": len(aborted_run_ids)}

    def get_run(self, run_id: str) -> dict[str, Any]:
        manifest_path = self.config.outputs_root / run_id / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run manifest not found for run_id={run_id}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def resolve_artifact_path(self, run_id: str, artifact_path: str) -> Path:
        manifest = self.get_run(run_id)
        base_dir = Path(str(manifest.get("artifacts_dir", ""))).resolve()
        target = Path(artifact_path)
        resolved = target.resolve() if target.is_absolute() else (base_dir / target).resolve()
        try:
            resolved.relative_to(base_dir)
        except ValueError as exc:
            raise PermissionError("Artifact path escapes the run artifact directory.") from exc
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        return resolved
