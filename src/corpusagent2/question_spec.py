from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
import uuid
from typing import Any


_MONTH_PATTERN = re.compile(r"(?<!\d)(\d{4})-(\d{2})(?!\d)")
_YEAR_PATTERN = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_QUOTED_ENTITY_PATTERN = re.compile(r"['\"]([^'\"]{2,80})['\"]")
_TITLE_CASE_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")


class QuestionClass(str, Enum):
    RETRIEVAL_QA = "retrieval_qa"
    ENTITY_TREND = "entity_trend"
    SENTIMENT_TREND = "sentiment_trend"
    TOPIC_TREND = "topic_trend"
    BURST_ANALYSIS = "burst_analysis"
    KEYPHRASE_ANALYSIS = "keyphrase_analysis"
    CLAIM_VERIFICATION = "claim_verification"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    MULTI_ANALYSIS = "multi_analysis"


class FeasibilityStatus(str, Enum):
    FEASIBLE = "feasible"
    PARTIALLY_FEASIBLE = "partially_feasible"
    NEEDS_CLARIFICATION = "needs_clarification"
    NOT_FEASIBLE = "not_feasible"


class FeasibilityStage(str, Enum):
    METADATA_SCHEMA = "metadata_schema"
    RETRIEVAL_LIGHTWEIGHT = "retrieval_lightweight"


@dataclass(slots=True)
class TimeRange:
    start: str = ""
    end: str = ""
    granularity: str = "year"

    def is_defined(self) -> bool:
        return bool(self.start or self.end)

    def to_dict(self) -> dict[str, str]:
        return {
            "start": self.start,
            "end": self.end,
            "granularity": self.granularity,
        }


@dataclass(slots=True)
class FeasibilityCheck:
    stage: str
    status: str
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "reasons": list(self.reasons),
            "details": dict(self.details),
        }


@dataclass(slots=True)
class FeasibilityReport:
    checks: list[FeasibilityCheck] = field(default_factory=list)

    @property
    def overall_status(self) -> str:
        statuses = {item.status for item in self.checks}
        if FeasibilityStatus.NOT_FEASIBLE.value in statuses:
            return FeasibilityStatus.NOT_FEASIBLE.value
        if FeasibilityStatus.NEEDS_CLARIFICATION.value in statuses:
            return FeasibilityStatus.NEEDS_CLARIFICATION.value
        if FeasibilityStatus.PARTIALLY_FEASIBLE.value in statuses:
            return FeasibilityStatus.PARTIALLY_FEASIBLE.value
        return FeasibilityStatus.FEASIBLE.value

    @property
    def unsupported_reasons(self) -> list[str]:
        rows: list[str] = []
        for check in self.checks:
            if check.status in {
                FeasibilityStatus.NOT_FEASIBLE.value,
                FeasibilityStatus.NEEDS_CLARIFICATION.value,
                FeasibilityStatus.PARTIALLY_FEASIBLE.value,
            }:
                rows.extend(check.reasons)
        return list(dict.fromkeys(item for item in rows if item))

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "checks": [item.to_dict() for item in self.checks],
            "unsupported_reasons": self.unsupported_reasons,
        }


@dataclass(slots=True)
class QuestionSpec:
    question_id: str
    raw_question: str
    normalized_question: str
    question_class: str
    entities: list[str] = field(default_factory=list)
    time_range: TimeRange = field(default_factory=TimeRange)
    group_by: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    metadata_requirements: list[str] = field(default_factory=list)
    expected_output_types: list[str] = field(default_factory=list)
    ambiguity_flags: list[str] = field(default_factory=list)
    clarification_question: str | None = None
    feasibility: FeasibilityReport = field(default_factory=FeasibilityReport)

    @property
    def feasibility_status(self) -> str:
        return self.feasibility.overall_status

    @property
    def unsupported_reasons(self) -> list[str]:
        return self.feasibility.unsupported_reasons

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["time_range"] = self.time_range.to_dict()
        payload["feasibility"] = self.feasibility.to_dict()
        payload["feasibility_status"] = self.feasibility_status
        payload["unsupported_reasons"] = self.unsupported_reasons
        return payload


def make_question_id(prefix: str = "q") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def normalize_question_text(question: str) -> str:
    return " ".join(str(question).strip().split()).lower()


def infer_group_by(question: str) -> list[str]:
    normalized = normalize_question_text(question)
    group_by: list[str] = []
    if "by month" in normalized or "monthly" in normalized:
        group_by.append("month")
    elif "by year" in normalized or "yearly" in normalized or "over time" in normalized or "trend" in normalized:
        group_by.append("year")

    if "by source" in normalized or "per source" in normalized:
        group_by.append("source")
    if "by organization" in normalized or "by entity" in normalized:
        group_by.append("entity")
    return group_by


def extract_entities_from_question(question: str) -> list[str]:
    seen: set[str] = set()
    entities: list[str] = []

    for pattern in (_QUOTED_ENTITY_PATTERN, _TITLE_CASE_ENTITY_PATTERN):
        for match in pattern.finditer(question):
            value = " ".join(match.group(1 if pattern is _QUOTED_ENTITY_PATTERN else 0).split()).strip()
            if not value:
                continue
            if len(value) < 2:
                continue
            lowered = value.lower()
            if lowered in {"what", "when", "where", "which", "who", "how", "compare"}:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            entities.append(value)
    return entities


def parse_time_range(question: str) -> TimeRange:
    months = [match.group(0) for match in _MONTH_PATTERN.finditer(question)]
    if months:
        ordered = sorted(months)
        return TimeRange(start=ordered[0], end=ordered[-1], granularity="month")

    years = [match.group(1) for match in _YEAR_PATTERN.finditer(question)]
    if years:
        ordered = sorted(years)
        return TimeRange(start=ordered[0], end=ordered[-1], granularity="year")

    normalized = normalize_question_text(question)
    if "monthly" in normalized or "by month" in normalized:
        return TimeRange(granularity="month")
    return TimeRange(granularity="year")
