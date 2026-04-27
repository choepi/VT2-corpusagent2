from __future__ import annotations

import re


_MOTIVE_PATTERNS = [
    re.compile(r"\bwhy did journalists\b", re.IGNORECASE),
    re.compile(r"\bwhy did the media\b", re.IGNORECASE),
    re.compile(r"\bmotivation of journalists\b", re.IGNORECASE),
    re.compile(r"\bwhy .* want(?:ed)? to\b", re.IGNORECASE),
]


def is_motive_question(question: str) -> bool:
    text = str(question).strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _MOTIVE_PATTERNS)


def rejection_reason_for_question(question: str) -> str | None:
    if is_motive_question(question):
        return (
            "The corpus can support observable content, framing, timing, quotations, and evidence, "
            "but not hidden motives or mental states of journalists."
        )
    return None


def evidence_required(question: str) -> bool:
    lowered = str(question).lower()
    keywords = (
        "predict",
        "prediction",
        "verify",
        "verification",
        "claim",
        "evidence",
        "contradict",
        "contradiction",
        "which media",
        "which article",
    )
    return any(keyword in lowered for keyword in keywords)


def normalize_question_text(question: str) -> tuple[str, list[str]]:
    rewritten = str(question).strip()
    rewritten = re.sub(
        r"\b([A-Za-z][A-Za-z-]{2,})/([A-Za-z][A-Za-z-]{2,})\b",
        r"\1 and \2",
        rewritten,
    )
    return rewritten, []
