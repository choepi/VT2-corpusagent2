from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .plan_graph import NodeType, PlanGraph, PlanNode
from .question_spec import (
    FeasibilityCheck,
    FeasibilityReport,
    FeasibilityStage,
    FeasibilityStatus,
    QuestionClass,
    QuestionSpec,
    extract_entities_from_question,
    infer_group_by,
    make_question_id,
    normalize_question_text,
    parse_time_range,
)
from .tool_registry import ToolRegistry


_BASE_METADATA_COLUMNS = {"doc_id", "title", "text", "published_at"}
_ANALYSIS_SIGNAL_CAPABILITIES = {
    "sentiment": "analyze.sentiment_series",
    "entities": "analyze.entity_trend",
    "topics": "analyze.topics_over_time",
    "bursts": "analyze.burst_events",
    "keyphrases": "analyze.keyphrases",
}
_ANALYSIS_ARTIFACTS = {
    "analyze.entity_trend": "entity_trend",
    "analyze.sentiment_series": "sentiment_series",
    "analyze.topics_over_time": "topics_over_time",
    "analyze.burst_events": "burst_events",
    "analyze.keyphrases": "keyphrases",
}
_QUESTION_CLASS_KEYWORDS: dict[str, set[str]] = {
    "sentiment": {"sentiment", "tone", "positive", "negative"},
    "entities": {"entity", "entities", "organization", "organisations", "person", "people", "actor", "actors"},
    "topics": {"topic", "topics", "theme", "themes", "narrative", "narratives"},
    "bursts": {"burst", "spike", "surge", "peak"},
    "keyphrases": {"keyphrase", "keyphrases", "keyword", "keywords", "terms", "phrases"},
    "verification": {"verify", "verified", "support", "supported", "contradict", "contradiction", "true", "false", "claim"},
}


@dataclass(slots=True)
class PlanningContext:
    total_documents: int = 0
    metadata_columns: set[str] = field(default_factory=set)
    available_artifacts: set[str] = field(default_factory=set)
    retrieval_ready: bool = False
    dense_ready: bool = False
    notes: list[str] = field(default_factory=list)

    def has_artifact(self, artifact_name: str) -> bool:
        return artifact_name in self.available_artifacts


class QuestionPlanner:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry

    def build_question_spec(
        self,
        raw_question: str,
        planning_context: PlanningContext,
        question_id: str | None = None,
    ) -> QuestionSpec:
        normalized = normalize_question_text(raw_question)
        question_class = self.classify_question(normalized)
        entities = extract_entities_from_question(raw_question)
        required_capabilities = self.required_capabilities_for_question(question_class, normalized)
        metadata_requirements = self.metadata_requirements_for_capabilities(required_capabilities)
        ambiguity_flags = self.detect_ambiguity(normalized, question_class, required_capabilities, entities)
        metadata_check = self.metadata_schema_feasibility(
            question_class=question_class,
            required_capabilities=required_capabilities,
            metadata_requirements=metadata_requirements,
            ambiguity_flags=ambiguity_flags,
            planning_context=planning_context,
        )
        retrieval_check = self.lightweight_retrieval_feasibility(
            raw_question=raw_question,
            question_class=question_class,
            entities=entities,
            previous_check=metadata_check,
        )
        return QuestionSpec(
            question_id=question_id or make_question_id(),
            raw_question=raw_question,
            normalized_question=normalized,
            question_class=question_class,
            entities=entities,
            time_range=parse_time_range(raw_question),
            group_by=infer_group_by(raw_question),
            required_capabilities=required_capabilities,
            metadata_requirements=metadata_requirements,
            expected_output_types=self.expected_outputs_for_question(question_class),
            ambiguity_flags=ambiguity_flags,
            clarification_question=self.build_clarification_question(ambiguity_flags, raw_question),
            feasibility=FeasibilityReport(checks=[metadata_check, retrieval_check]),
        )

    def classify_question(self, normalized_question: str) -> str:
        flags = {name: any(token in normalized_question for token in keywords) for name, keywords in _QUESTION_CLASS_KEYWORDS.items()}
        analysis_signals = [signal for signal in ("sentiment", "entities", "topics", "bursts", "keyphrases") if flags.get(signal, False)]
        if flags.get("verification", False) and not analysis_signals:
            return QuestionClass.CLAIM_VERIFICATION.value
        if len(analysis_signals) >= 2:
            return QuestionClass.COMPARATIVE_ANALYSIS.value if "compare" in normalized_question else QuestionClass.MULTI_ANALYSIS.value
        if flags.get("sentiment", False):
            return QuestionClass.SENTIMENT_TREND.value
        if flags.get("entities", False):
            return QuestionClass.ENTITY_TREND.value
        if flags.get("topics", False):
            return QuestionClass.TOPIC_TREND.value
        if flags.get("bursts", False):
            return QuestionClass.BURST_ANALYSIS.value
        if flags.get("keyphrases", False):
            return QuestionClass.KEYPHRASE_ANALYSIS.value
        return QuestionClass.RETRIEVAL_QA.value

    def required_capabilities_for_question(self, question_class: str, normalized_question: str) -> list[str]:
        rows = ["retrieve.documents", "filter.documents"]
        if question_class == QuestionClass.ENTITY_TREND.value:
            rows.append("analyze.entity_trend")
        elif question_class == QuestionClass.SENTIMENT_TREND.value:
            rows.append("analyze.sentiment_series")
        elif question_class == QuestionClass.TOPIC_TREND.value:
            rows.append("analyze.topics_over_time")
        elif question_class == QuestionClass.BURST_ANALYSIS.value:
            rows.extend(["analyze.entity_trend", "analyze.burst_events"])
        elif question_class == QuestionClass.KEYPHRASE_ANALYSIS.value:
            rows.append("analyze.keyphrases")
        elif question_class in {QuestionClass.MULTI_ANALYSIS.value, QuestionClass.COMPARATIVE_ANALYSIS.value}:
            for signal, capability in _ANALYSIS_SIGNAL_CAPABILITIES.items():
                if signal in normalized_question:
                    rows.append(capability)
        rows.extend(["aggregate.findings", "verify.claims", "synthesize.answer"])
        return list(dict.fromkeys(rows))

    def metadata_requirements_for_capabilities(self, capabilities: list[str]) -> list[str]:
        rows = {"doc_metadata.doc_id", "doc_metadata.title", "doc_metadata.text", "doc_metadata.published_at"}
        by_capability = {
            "retrieve.documents": {"retrieval.lexical_assets", "retrieval.dense_assets_or_pgvector"},
            "analyze.entity_trend": {"artifact.entity_trend_or_retrieved_docs"},
            "analyze.sentiment_series": {"artifact.sentiment_series_or_retrieved_docs"},
            "analyze.topics_over_time": {"artifact.topics_over_time_or_retrieved_docs"},
            "analyze.burst_events": {"artifact.burst_events_or_entity_trend"},
            "analyze.keyphrases": {"artifact.keyphrases_or_retrieved_docs"},
            "verify.claims": {"doc_metadata.text"},
        }
        for capability in capabilities:
            rows.update(by_capability.get(capability, set()))
        return sorted(rows)

    def expected_outputs_for_question(self, question_class: str) -> list[str]:
        rows = ["answer_text", "evidence_items", "artifacts_used", "unsupported_parts", "caveats", "claim_verdicts"]
        if question_class != QuestionClass.RETRIEVAL_QA.value:
            rows.append("analysis_table")
        return rows

    def detect_ambiguity(self, normalized_question: str, question_class: str, required_capabilities: list[str], entities: list[str]) -> list[str]:
        rows: list[str] = []
        if "compare" in normalized_question and question_class not in {QuestionClass.MULTI_ANALYSIS.value, QuestionClass.COMPARATIVE_ANALYSIS.value} and len(entities) < 2:
            rows.append("missing_comparison_target")
        if "between" in normalized_question and len(entities) < 2:
            rows.append("missing_between_targets")
        if question_class == QuestionClass.CLAIM_VERIFICATION.value and "claim" not in normalized_question:
            rows.append("verification_target_unclear")
        if "analyze.sentiment_series" in required_capabilities and " by " in normalized_question and len(entities) == 0:
            rows.append("sentiment_group_missing")
        return rows

    def build_clarification_question(self, ambiguity_flags: list[str], raw_question: str) -> str | None:
        if not ambiguity_flags:
            return None
        if "missing_comparison_target" in ambiguity_flags or "missing_between_targets" in ambiguity_flags:
            return f"Which entities, groups, or time periods should be compared in '{raw_question}'?"
        if "verification_target_unclear" in ambiguity_flags:
            return "Which exact claim should be verified against the corpus?"
        if "sentiment_group_missing" in ambiguity_flags:
            return "Should sentiment be grouped by time, source, or entity?"
        return None

    def metadata_schema_feasibility(
        self,
        question_class: str,
        required_capabilities: list[str],
        metadata_requirements: list[str],
        ambiguity_flags: list[str],
        planning_context: PlanningContext,
    ) -> FeasibilityCheck:
        reasons: list[str] = []
        if ambiguity_flags:
            return FeasibilityCheck(
                stage=FeasibilityStage.METADATA_SCHEMA.value,
                status=FeasibilityStatus.NEEDS_CLARIFICATION.value,
                reasons=["Clarification is required because workflow selection would change materially."],
                details={"ambiguity_flags": list(ambiguity_flags)},
            )
        if planning_context.total_documents <= 0:
            reasons.append("The corpus contains no documents.")
        missing_columns = sorted(_BASE_METADATA_COLUMNS.difference(planning_context.metadata_columns))
        if missing_columns:
            reasons.append(f"Missing required metadata columns: {missing_columns}.")
        if "retrieve.documents" in required_capabilities and not planning_context.retrieval_ready:
            reasons.append("Retrieval assets are not ready.")
        for capability in required_capabilities:
            if capability.startswith("analyze.") and not self._analysis_capability_feasible(capability, planning_context):
                reasons.append(f"Capability '{capability}' lacks both artifact and retrieval-backed fallback.")
        if self.registry is not None:
            missing_impls = [capability for capability in required_capabilities if not self.registry.list_tools(capability=capability)]
            if missing_impls:
                reasons.append(f"No tool implementations registered for capabilities: {missing_impls}.")
        status = FeasibilityStatus.NOT_FEASIBLE.value if reasons else FeasibilityStatus.FEASIBLE.value
        return FeasibilityCheck(
            stage=FeasibilityStage.METADATA_SCHEMA.value,
            status=status,
            reasons=reasons,
            details={"question_class": question_class, "metadata_requirements": list(metadata_requirements)},
        )

    def lightweight_retrieval_feasibility(self, raw_question: str, question_class: str, entities: list[str], previous_check: FeasibilityCheck) -> FeasibilityCheck:
        if previous_check.status in {FeasibilityStatus.NOT_FEASIBLE.value, FeasibilityStatus.NEEDS_CLARIFICATION.value}:
            skipped_status = previous_check.status
            return FeasibilityCheck(
                stage=FeasibilityStage.RETRIEVAL_LIGHTWEIGHT.value,
                status=skipped_status,
                reasons=["Skipped retrieval feasibility probe because metadata/schema stage failed."],
                details={"skipped": True, "propagated_status": skipped_status},
            )
        if self.registry is None:
            return FeasibilityCheck(
                stage=FeasibilityStage.RETRIEVAL_LIGHTWEIGHT.value,
                status=FeasibilityStatus.FEASIBLE.value,
                reasons=["Registry unavailable, retrieval probe skipped."],
                details={"skipped": True},
            )
        try:
            params = {"query": raw_question, "top_k": 5, "lightweight": True}
            resolution = self.registry.resolve("retrieve.documents", context={"question_class": question_class, "planning_probe": True}, params=params)
            result = resolution.adapter.run(params=params, dependency_results={}, context={"question_class": question_class, "planning_probe": True})
        except Exception as exc:
            return FeasibilityCheck(
                stage=FeasibilityStage.RETRIEVAL_LIGHTWEIGHT.value,
                status=FeasibilityStatus.NOT_FEASIBLE.value,
                reasons=[f"Retrieval feasibility probe failed: {exc}"],
                details={"error": str(exc)},
            )
        rows = result.payload.get("results", []) if isinstance(result.payload, dict) else []
        if not rows:
            return FeasibilityCheck(
                stage=FeasibilityStage.RETRIEVAL_LIGHTWEIGHT.value,
                status=FeasibilityStatus.NOT_FEASIBLE.value,
                reasons=["Lightweight retrieval found no supporting documents."],
                details={"result_count": 0},
            )
        if entities:
            haystack = " ".join(f"{row.get('title', '')} {row.get('snippet', '')}".lower() for row in rows)
            missing = [entity for entity in entities[:2] if entity.lower() not in haystack]
            if missing:
                return FeasibilityCheck(
                    stage=FeasibilityStage.RETRIEVAL_LIGHTWEIGHT.value,
                    status=FeasibilityStatus.PARTIALLY_FEASIBLE.value,
                    reasons=["Probe retrieved documents, but not all requested entities were visible in top results."],
                    details={"missing_entities_in_probe": missing, "result_count": len(rows)},
                )
        return FeasibilityCheck(
            stage=FeasibilityStage.RETRIEVAL_LIGHTWEIGHT.value,
            status=FeasibilityStatus.FEASIBLE.value,
            reasons=[f"Lightweight retrieval returned {len(rows)} support documents."],
            details={"result_count": len(rows)},
        )

    def _analysis_capability_feasible(self, capability: str, planning_context: PlanningContext) -> bool:
        artifact_name = _ANALYSIS_ARTIFACTS.get(capability)
        if artifact_name and planning_context.has_artifact(artifact_name):
            return True
        return planning_context.retrieval_ready

    def build_plan(self, question_spec: QuestionSpec) -> PlanGraph:
        if question_spec.feasibility_status == FeasibilityStatus.NOT_FEASIBLE.value:
            return PlanGraph(
                question_id=question_spec.question_id,
                template_name="unsupported",
                nodes=[PlanNode("synthesize_unavailable", NodeType.SYNTHESIZE.value, "synthesize.answer", "final_answer", params={"mode": "unsupported"}, cacheable=False)],
                metadata={"question_class": question_spec.question_class},
            )

        nodes = [
            PlanNode("retrieve_support", NodeType.RETRIEVE.value, "retrieve.documents", "retrieval_results", params={"query": question_spec.raw_question, "top_k": 12}),
            PlanNode("filter_support", NodeType.FILTER.value, "filter.documents", "filtered_results", dependencies=["retrieve_support"], params={"entities": list(question_spec.entities), "time_range": question_spec.time_range.to_dict(), "max_results": 8}),
        ]
        analysis_nodes: list[str] = []
        for capability, output_key in (("analyze.entity_trend", "entity_trend"), ("analyze.sentiment_series", "sentiment_series"), ("analyze.topics_over_time", "topics_over_time"), ("analyze.keyphrases", "keyphrases")):
            if capability in question_spec.required_capabilities:
                node_id = f"{output_key}_analysis"
                nodes.append(PlanNode(node_id, NodeType.ANALYZE.value, capability, output_key, dependencies=["filter_support"], params={"entities": list(question_spec.entities), "time_range": question_spec.time_range.to_dict()}))
                analysis_nodes.append(node_id)
        if "analyze.burst_events" in question_spec.required_capabilities:
            deps = ["entity_trend_analysis"] if "entity_trend_analysis" in analysis_nodes else ["filter_support"]
            nodes.append(PlanNode("burst_events_analysis", NodeType.ANALYZE.value, "analyze.burst_events", "burst_events", dependencies=deps, params={"time_range": question_spec.time_range.to_dict()}))
            analysis_nodes.append("burst_events_analysis")
        nodes.append(PlanNode("aggregate_findings", NodeType.AGGREGATE.value, "aggregate.findings", "aggregate_summary", dependencies=["filter_support", *analysis_nodes], params={"question_class": question_spec.question_class, "raw_question": question_spec.raw_question, "expected_output_types": list(question_spec.expected_output_types)}))
        nodes.append(PlanNode("verify_claims", NodeType.VERIFY.value, "verify.claims", "claim_verification", dependencies=["aggregate_findings", "filter_support"], params={"question_class": question_spec.question_class, "max_claims": 3}, optional=True))
        nodes.append(PlanNode("synthesize_answer", NodeType.SYNTHESIZE.value, "synthesize.answer", "final_answer", dependencies=["aggregate_findings", "verify_claims", "filter_support"], params={"question_text": question_spec.raw_question, "question_class": question_spec.question_class}, cacheable=False))
        return PlanGraph(question_id=question_spec.question_id, template_name=f"{question_spec.question_class}_template", nodes=nodes, metadata={"question_class": question_spec.question_class})
