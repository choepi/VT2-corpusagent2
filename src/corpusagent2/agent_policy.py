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


def rewrite_special_cases(question: str) -> tuple[str, list[str]]:
    lowered = str(question).lower()
    assumptions: list[str] = []
    rewritten = str(question).strip()
    if "ukraine" in lowered and "predict" in lowered:
        rewritten = (
            "Which media published explicit immediate pre-invasion warnings or predictions "
            "about a Russian invasion of Ukraine before 2022-02-24?"
        )
        assumptions.append(
            "Interpret 'predicted the outbreak of the Ukraine war' as explicit immediate pre-invasion warning or prediction before 2022-02-24."
        )
    if "risk/regulation" in lowered:
        rewritten = rewritten.replace("risk/regulation", "risk and regulation")
    return rewritten, assumptions
