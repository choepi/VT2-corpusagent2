from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from typing import Any

from .plan_graph import PlanGraph
from .question_spec import QuestionSpec


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


@dataclass(slots=True)
class EvidenceItem:
    doc_id: str
    chunk_id: str
    title: str
    snippet: str
    score: float
    score_components: dict[str, float] = field(default_factory=dict)
    published_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClaimVerdictPayload:
    claim_id: str
    claim: str
    label: str
    entailment: float
    contradiction: float
    neutral: float
    best_evidence_sentence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FinalAnswerPayload:
    answer_text: str
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    artifacts_used: list[str] = field(default_factory=list)
    unsupported_parts: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    claim_verdicts: list[ClaimVerdictPayload] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "artifacts_used": list(self.artifacts_used),
            "unsupported_parts": list(self.unsupported_parts),
            "caveats": list(self.caveats),
            "claim_verdicts": [item.to_dict() for item in self.claim_verdicts],
        }


@dataclass(slots=True)
class StructuredFailure:
    node_id: str
    capability: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class NodeExecutionRecord:
    node_id: str
    node_type: str
    capability: str
    output_key: str
    status: str
    tool_name: str = ""
    provider: str = ""
    tool_reason: str = ""
    cache_hit: bool = False
    artifacts: list[str] = field(default_factory=list)
    evidence_count: int = 0
    caveats: list[str] = field(default_factory=list)
    unsupported_parts: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunManifest:
    run_id: str
    question_spec: QuestionSpec
    plan_graph: PlanGraph
    node_results: list[NodeExecutionRecord]
    provenance_records: list[dict[str, Any]]
    final_answer: FinalAnswerPayload
    artifacts_dir: str
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    failures: list[StructuredFailure] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "artifacts_dir": self.artifacts_dir,
            "question_spec": self.question_spec.to_dict(),
            "plan_graph": self.plan_graph.to_dict(),
            "node_results": [item.to_dict() for item in self.node_results],
            "provenance_records": _serialize(self.provenance_records),
            "final_answer": self.final_answer.to_dict(),
            "failures": [item.to_dict() for item in self.failures],
        }
