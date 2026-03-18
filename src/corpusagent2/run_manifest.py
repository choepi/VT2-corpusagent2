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
class FinalAnswerPayload:
    answer_text: str
    evidence_items: list[dict[str, Any]] = field(default_factory=list)
    artifacts_used: list[str] = field(default_factory=list)
    unsupported_parts: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    claim_verdicts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "FinalAnswerPayload":
        payload = payload or {}
        return cls(
            answer_text=str(payload.get("answer_text", "")).strip(),
            evidence_items=list(payload.get("evidence_items", [])),
            artifacts_used=list(payload.get("artifacts_used", [])),
            unsupported_parts=list(payload.get("unsupported_parts", [])),
            caveats=list(payload.get("caveats", [])),
            claim_verdicts=list(payload.get("claim_verdicts", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "evidence_items": _serialize(self.evidence_items),
            "artifacts_used": list(self.artifacts_used),
            "unsupported_parts": list(self.unsupported_parts),
            "caveats": list(self.caveats),
            "claim_verdicts": _serialize(self.claim_verdicts),
        }


@dataclass(slots=True)
class StructuredFailure:
    node_id: str
    capability: str
    error_type: str
    message: str
    retriable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NodeExecutionRecord:
    node_id: str
    node_type: str
    capability: str
    output_key: str
    status: str
    started_at_utc: str = ""
    finished_at_utc: str = ""
    duration_ms: float = 0.0
    cache_key: str = ""
    cache_hit: bool = False
    tool_name: str = ""
    provider: str = ""
    tool_version: str = ""
    model_id: str = ""
    tool_reason: str = ""
    artifacts_used: list[str] = field(default_factory=list)
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
    node_records: list[NodeExecutionRecord]
    provenance_records: list[dict[str, Any]]
    final_answer: FinalAnswerPayload
    artifacts_dir: str
    status: str
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    failures: list[StructuredFailure] = field(default_factory=list)
    cache_entries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "status": self.status,
            "artifacts_dir": self.artifacts_dir,
            "cache_entries": self.cache_entries,
            "question_spec": self.question_spec.to_dict(),
            "plan_graph": self.plan_graph.to_dict(),
            "node_records": [item.to_dict() for item in self.node_records],
            "provenance_records": _serialize(self.provenance_records),
            "final_answer": self.final_answer.to_dict(),
            "failures": [item.to_dict() for item in self.failures],
        }
