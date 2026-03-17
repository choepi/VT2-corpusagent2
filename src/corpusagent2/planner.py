from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .plan_graph import NodeType, PlanGraph, PlanNode
from .question_spec import (
    FeasibilityStatus,
    QuestionClass,
    QuestionSpec,
    extract_entities_from_question,
    infer_group_by,
    make_question_id,
    normalize_question_text,
    parse_time_range,
)


_BASE_METADATA_COLUMNS = {"doc_id", "title", "text", "published_at"}

_QUESTION_CLASS_KEYWORDS: dict[str, set[str]] = {
    "sentiment": {"sentiment", "tone", "positive", "negative"},
    "entities": {"entity", "entities", "organization", "organisations", "person", "people", "actor", "actors"},
    "topics": {"topic", "topics", "theme", "themes", "narrative", "narratives"},
    "bursts": {"burst", "spike", "surge", "peak"},
    "keyphrases": {"keyphrase", "keyphrases", "keyword", "keywords", "terms", "phrases"},
    "verification": {"verify", "verified", "support", "supported", "contradict", "contradiction", "true", "false", "claim"},
}


_ANALYZE_CAPABILITY_BY_SIGNAL = {
    "sentiment": "analyze.sentiment_series",
    "entities": "analyze.entity_trend",
    "topics": "analyze.topics_over_time",
    "bursts": "analyze.burst_events",
    "keyphrases": "analyze.keyphrases",
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
    def __init__(self, registry: Any | None = None) -> None:
        self.registry = registry

    def build_question_spec(
        self,
        raw_question: str,
        planning_context: PlanningContext,
        question_id: str | None = None,
    ) -> QuestionSpec:
        normalized = normalize_question_text(raw_question)
        entities = extract_entities_from_question(raw_question)
        time_range = parse_time_range(raw_question)
        group_by = infer_group_by(raw_question)
        question_class = self.classify_question(normalized)
        required_capabilities = self.required_capabilities_for_question(question_class, normalized)
        metadata_requirements = self.metadata_requirements_for_capabilities(required_capabilities)
        expected_output_types = self.expected_outputs_for_question(question_class)
        ambiguity_flags = self.detect_ambiguity(normalized, question_class, required_capabilities, entities)
        clarification_question = self.build_clarification_question(ambiguity_flags, raw_question)
        feasibility_status, unsupported_reasons = self.assess_feasibility(
            raw_question=raw_question,
            normalized_question=normalized,
            question_class=question_class,
            entities=entities,
            required_capabilities=required_capabilities,
            metadata_requirements=metadata_requirements,
            ambiguity_flags=ambiguity_flags,
            planning_context=planning_context,
        )

        return QuestionSpec(
            question_id=question_id or make_question_id(),
            raw_question=raw_question,
            normalized_question=normalized,
            question_class=question_class,
            ambiguity_flags=ambiguity_flags,
            clarification_question=clarification_question,
            entities=entities,
            time_range=time_range,
            group_by=group_by,
            required_capabilities=required_capabilities,
            metadata_requirements=metadata_requirements,
            expected_output_types=expected_output_types,
            feasibility_status=feasibility_status,
            unsupported_reasons=unsupported_reasons,
        )

    def classify_question(self, normalized_question: str) -> str:
        flags = {name: any(token in normalized_question for token in keywords) for name, keywords in _QUESTION_CLASS_KEYWORDS.items()}

        active_analysis_signals = [
            signal
            for signal in ("sentiment", "entities", "topics", "bursts", "keyphrases")
            if flags.get(signal, False)
        ]

        if flags.get("verification", False) and not active_analysis_signals:
            return QuestionClass.CLAIM_VERIFICATION.value
        if len(active_analysis_signals) >= 2:
            if "compare" in normalized_question:
                return QuestionClass.COMPARATIVE_ANALYSIS.value
            return QuestionClass.MULTI_ANALYSIS.value
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
        required = ["retrieve.documents", "filter.documents"]

        if question_class == QuestionClass.ENTITY_TREND.value:
            required.append("analyze.entity_trend")
        elif question_class == QuestionClass.SENTIMENT_TREND.value:
            required.append("analyze.sentiment_series")
        elif question_class == QuestionClass.TOPIC_TREND.value:
            required.append("analyze.topics_over_time")
        elif question_class == QuestionClass.BURST_ANALYSIS.value:
            required.extend(["analyze.entity_trend", "analyze.burst_events"])
        elif question_class == QuestionClass.KEYPHRASE_ANALYSIS.value:
            required.append("analyze.keyphrases")
        elif question_class in {
            QuestionClass.MULTI_ANALYSIS.value,
            QuestionClass.COMPARATIVE_ANALYSIS.value,
        }:
            for signal, capability in _ANALYZE_CAPABILITY_BY_SIGNAL.items():
                if signal in normalized_question:
                    required.append(capability)

        required.extend(["aggregate.findings", "verify.claims", "synthesize.answer"])
        deduped: list[str] = []
        seen: set[str] = set()
        for capability in required:
            if capability not in seen:
                seen.add(capability)
                deduped.append(capability)
        return deduped

    def metadata_requirements_for_capabilities(self, capabilities: list[str]) -> list[str]:
        requirements = {
            "doc_metadata.doc_id",
            "doc_metadata.title",
            "doc_metadata.text",
            "doc_metadata.published_at",
        }

        capability_to_requirements = {
            "retrieve.documents": {"retrieval.lexical_assets", "retrieval.dense_assets_or_pgvector"},
            "analyze.entity_trend": {"artifact.entity_trend_or_retrieved_docs"},
            "analyze.sentiment_series": {"artifact.sentiment_series_or_retrieved_docs"},
            "analyze.topics_over_time": {"artifact.topics_over_time_or_retrieved_docs"},
            "analyze.burst_events": {"artifact.burst_events_or_entity_trend"},
            "analyze.keyphrases": {"artifact.keyphrases_or_retrieved_docs"},
            "verify.claims": {"doc_metadata.text"},
        }
        for capability in capabilities:
            requirements.update(capability_to_requirements.get(capability, set()))
        return sorted(requirements)

    def expected_outputs_for_question(self, question_class: str) -> list[str]:
        base = ["answer_text", "evidence_items", "claim_verdicts"]
        if question_class in {
            QuestionClass.ENTITY_TREND.value,
            QuestionClass.SENTIMENT_TREND.value,
            QuestionClass.TOPIC_TREND.value,
            QuestionClass.BURST_ANALYSIS.value,
            QuestionClass.KEYPHRASE_ANALYSIS.value,
            QuestionClass.MULTI_ANALYSIS.value,
            QuestionClass.COMPARATIVE_ANALYSIS.value,
        }:
            base.append("analysis_table")
        return base

    def detect_ambiguity(
        self,
        normalized_question: str,
        question_class: str,
        required_capabilities: list[str],
        entities: list[str],
    ) -> list[str]:
        ambiguity_flags: list[str] = []
        if "compare" in normalized_question and question_class not in {
            QuestionClass.MULTI_ANALYSIS.value,
            QuestionClass.COMPARATIVE_ANALYSIS.value,
        } and len(entities) < 2:
            ambiguity_flags.append("missing_comparison_target")
        if "between" in normalized_question and len(entities) < 2:
            ambiguity_flags.append("missing_between_targets")
        if question_class == QuestionClass.CLAIM_VERIFICATION.value and "?" in normalized_question and "claim" not in normalized_question:
            ambiguity_flags.append("verification_target_unclear")
        if (
            "analyze.sentiment_series" in required_capabilities
            and "analyze.entity_trend" not in required_capabilities
            and " by " in normalized_question
            and len(entities) == 0
        ):
            ambiguity_flags.append("sentiment_group_missing")
        return ambiguity_flags

    def build_clarification_question(self, ambiguity_flags: list[str], raw_question: str) -> str | None:
        if not ambiguity_flags:
            return None
        if "missing_comparison_target" in ambiguity_flags or "missing_between_targets" in ambiguity_flags:
            return f"Which entities, groups, or time periods should be compared in '{raw_question}'?"
        if "verification_target_unclear" in ambiguity_flags:
            return "Which concrete claim should be verified against the corpus?"
        if "sentiment_group_missing" in ambiguity_flags:
            return "Should the sentiment analysis be grouped by time, source, or entity?"
        return None

    def assess_feasibility(
        self,
        raw_question: str,
        normalized_question: str,
        question_class: str,
        entities: list[str],
        required_capabilities: list[str],
        metadata_requirements: list[str],
        ambiguity_flags: list[str],
        planning_context: PlanningContext,
    ) -> tuple[str, list[str]]:
        unsupported_reasons: list[str] = []

        if ambiguity_flags:
            return FeasibilityStatus.NEEDS_CLARIFICATION.value, [
                "Clarification is required because the requested workflow would change materially.",
            ]

        missing_columns = sorted(_BASE_METADATA_COLUMNS.difference(planning_context.metadata_columns))
        if planning_context.total_documents <= 0:
            unsupported_reasons.append("The corpus contains no documents.")
        if missing_columns:
            unsupported_reasons.append(f"Missing required metadata columns: {missing_columns}.")
        if "retrieve.documents" in required_capabilities and not planning_context.retrieval_ready:
            unsupported_reasons.append("Retrieval assets are not ready.")

        if "analyze.burst_events" in required_capabilities:
            if not (planning_context.has_artifact("burst_events") or planning_context.has_artifact("entity_trend")):
                unsupported_reasons.append(
                    "Burst analysis requires either a precomputed burst artifact or entity trend rows."
                )

        for capability in required_capabilities:
            if capability.startswith("analyze.") and not self._analysis_capability_feasible(capability, planning_context):
                unsupported_reasons.append(
                    f"Capability '{capability}' is unsupported because neither a matching artifact nor retrieval-backed fallback is available."
                )

        if unsupported_reasons:
            return FeasibilityStatus.NOT_FEASIBLE.value, unsupported_reasons

        retrieval_status, retrieval_reasons = self.assess_retrieval_feasibility(
            raw_question=raw_question,
            normalized_question=normalized_question,
            question_class=question_class,
            entities=entities,
        )
        if retrieval_status != FeasibilityStatus.FEASIBLE.value:
            return retrieval_status, retrieval_reasons
        return FeasibilityStatus.FEASIBLE.value, []

    def assess_retrieval_feasibility(
        self,
        raw_question: str,
        normalized_question: str,
        question_class: str,
        entities: list[str],
    ) -> tuple[str, list[str]]:
        if self.registry is None:
            return FeasibilityStatus.FEASIBLE.value, []

        try:
            context = {"question_class": question_class}
            resolution = self.registry.resolve(
                capability="retrieve.documents",
                context=context,
                params={"query": raw_question, "top_k": 5, "lightweight": True},
            )
            result = resolution.adapter.run(
                params={"query": raw_question, "top_k": 5, "lightweight": True},
                dependency_results={},
                context=context,
            )
        except Exception as exc:
            return FeasibilityStatus.NOT_FEASIBLE.value, [f"Retrieval feasibility check failed: {exc}"]

        rows = result.payload.get("results", []) if isinstance(result.payload, dict) else []
        if not rows:
            return FeasibilityStatus.NOT_FEASIBLE.value, [
                "Lightweight retrieval did not find supporting documents for the question.",
            ]

        if entities:
            haystack = " ".join(
                " ".join(
                    [
                        str(row.get("title", "")),
                        str(row.get("snippet", "")),
                    ]
                )
                for row in rows
            ).lower()
            if not all(entity.lower() in haystack for entity in entities[:2]):
                return FeasibilityStatus.PARTIALLY_FEASIBLE.value, [
                    "Retrieval found support documents, but not all requested entities were visible in the top lightweight evidence set.",
                ]
        return FeasibilityStatus.FEASIBLE.value, []

    def _analysis_capability_feasible(self, capability: str, planning_context: PlanningContext) -> bool:
        capability_to_artifact = {
            "analyze.entity_trend": "entity_trend",
            "analyze.sentiment_series": "sentiment_series",
            "analyze.topics_over_time": "topics_over_time",
            "analyze.burst_events": "burst_events",
            "analyze.keyphrases": "keyphrases",
        }
        artifact_name = capability_to_artifact.get(capability)
        if artifact_name and planning_context.has_artifact(artifact_name):
            return True
        return planning_context.retrieval_ready

    def build_plan(self, question_spec: QuestionSpec) -> PlanGraph:
        if question_spec.feasibility_status == FeasibilityStatus.NOT_FEASIBLE.value:
            return PlanGraph(
                question_id=question_spec.question_id,
                nodes=[
                    PlanNode(
                        node_id="synthesize_unavailable",
                        node_type=NodeType.SYNTHESIZE.value,
                        capability="synthesize.answer",
                        params={"mode": "unsupported"},
                        dependencies=[],
                        output_key="final_answer",
                    )
                ],
            )

        nodes: list[PlanNode] = [
            PlanNode(
                node_id="retrieve_support",
                node_type=NodeType.RETRIEVE.value,
                capability="retrieve.documents",
                params={"query": question_spec.raw_question, "top_k": 12},
                dependencies=[],
                output_key="retrieval_results",
            ),
            PlanNode(
                node_id="filter_support",
                node_type=NodeType.FILTER.value,
                capability="filter.documents",
                params={
                    "entities": list(question_spec.entities),
                    "time_range": question_spec.time_range.to_dict(),
                    "max_results": 8,
                },
                dependencies=["retrieve_support"],
                output_key="filtered_results",
            ),
        ]

        analysis_node_ids: list[str] = []
        if "analyze.entity_trend" in question_spec.required_capabilities:
            nodes.append(
                PlanNode(
                    node_id="analyze_entity_trend",
                    node_type=NodeType.ANALYZE.value,
                    capability="analyze.entity_trend",
                    params={
                        "entities": list(question_spec.entities),
                        "time_range": question_spec.time_range.to_dict(),
                    },
                    dependencies=["filter_support"],
                    output_key="entity_trend",
                )
            )
            analysis_node_ids.append("analyze_entity_trend")

        if "analyze.sentiment_series" in question_spec.required_capabilities:
            nodes.append(
                PlanNode(
                    node_id="analyze_sentiment_series",
                    node_type=NodeType.ANALYZE.value,
                    capability="analyze.sentiment_series",
                    params={
                        "entities": list(question_spec.entities),
                        "time_range": question_spec.time_range.to_dict(),
                    },
                    dependencies=["filter_support"],
                    output_key="sentiment_series",
                )
            )
            analysis_node_ids.append("analyze_sentiment_series")

        if "analyze.topics_over_time" in question_spec.required_capabilities:
            nodes.append(
                PlanNode(
                    node_id="analyze_topics_over_time",
                    node_type=NodeType.ANALYZE.value,
                    capability="analyze.topics_over_time",
                    params={"time_range": question_spec.time_range.to_dict()},
                    dependencies=["filter_support"],
                    output_key="topics_over_time",
                )
            )
            analysis_node_ids.append("analyze_topics_over_time")

        if "analyze.keyphrases" in question_spec.required_capabilities:
            nodes.append(
                PlanNode(
                    node_id="analyze_keyphrases",
                    node_type=NodeType.ANALYZE.value,
                    capability="analyze.keyphrases",
                    params={"time_range": question_spec.time_range.to_dict()},
                    dependencies=["filter_support"],
                    output_key="keyphrases",
                )
            )
            analysis_node_ids.append("analyze_keyphrases")

        if "analyze.burst_events" in question_spec.required_capabilities:
            dependencies = ["analyze_entity_trend"] if "analyze_entity_trend" in analysis_node_ids else ["filter_support"]
            nodes.append(
                PlanNode(
                    node_id="analyze_burst_events",
                    node_type=NodeType.ANALYZE.value,
                    capability="analyze.burst_events",
                    params={"time_range": question_spec.time_range.to_dict()},
                    dependencies=dependencies,
                    output_key="burst_events",
                )
            )
            analysis_node_ids.append("analyze_burst_events")

        aggregate_dependencies = ["filter_support", *analysis_node_ids]
        nodes.append(
            PlanNode(
                node_id="aggregate_findings",
                node_type=NodeType.AGGREGATE.value,
                capability="aggregate.findings",
                params={
                    "question_class": question_spec.question_class,
                    "raw_question": question_spec.raw_question,
                    "expected_output_types": list(question_spec.expected_output_types),
                },
                dependencies=aggregate_dependencies,
                output_key="aggregate_summary",
            )
        )
        nodes.append(
            PlanNode(
                node_id="verify_claims",
                node_type=NodeType.VERIFY.value,
                capability="verify.claims",
                params={"question_class": question_spec.question_class, "max_claims": 3},
                dependencies=["aggregate_findings", "filter_support"],
                output_key="claim_verification",
                optional=True,
            )
        )
        nodes.append(
            PlanNode(
                node_id="synthesize_answer",
                node_type=NodeType.SYNTHESIZE.value,
                capability="synthesize.answer",
                params={
                    "question_text": question_spec.raw_question,
                    "question_class": question_spec.question_class,
                },
                dependencies=["aggregate_findings", "verify_claims", "filter_support"],
                output_key="final_answer",
            )
        )
        return PlanGraph(question_id=question_spec.question_id, nodes=nodes)
