from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Mapping


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


@dataclass(frozen=True, slots=True)
class RetrievalBudget:
    scope: str
    retrieval_strategy: str
    top_k: int
    lexical_top_k: int
    dense_top_k: int
    use_rerank: bool
    rerank_top_k: int
    retrieval_mode: str
    fusion_k: int
    retrieve_all_requested: bool = False

    def to_inputs(self) -> dict[str, Any]:
        payload = {
            "retrieval_strategy": self.retrieval_strategy,
            "top_k": 0 if self.retrieve_all_requested else self.top_k,
            "retrieval_mode": self.retrieval_mode,
            "lexical_top_k": self.lexical_top_k,
            "dense_top_k": self.dense_top_k,
            "use_rerank": self.use_rerank,
            "fusion_k": self.fusion_k,
        }
        if self.retrieve_all_requested:
            payload["fallback_top_k"] = self.top_k
        if self.use_rerank:
            payload["rerank_top_k"] = self.rerank_top_k
        else:
            payload["rerank_top_k"] = 0
        if self.retrieve_all_requested:
            payload["retrieve_all"] = True
        return payload


@dataclass(frozen=True, slots=True)
class _BudgetProfile:
    top_k: int
    candidate_multiplier: int
    min_candidates: int
    rerank_multiplier: int
    use_rerank: bool


_EXHAUSTIVE_PATTERNS = (
    re.compile(
        r"\b(?:all|every|entire|whole|full|complete)\b"
        r"(?:\s+[a-z0-9-]+){0,6}\s+"
        r"(?:corpus|dataset|collection|archive|records?|documents?|articles?|reports?|stories?|items?|pieces?|docs?)\b"
    ),
    re.compile(r"\b(?:all|every|entire|whole|full|complete)\s+(?:matching|relevant|related|available)\s+(?:results?|hits?)\b"),
    re.compile(r"\bacross all\b"),
    re.compile(r"\bfor each\b"),
    re.compile(r"\bevery single\b"),
)
_BROAD_ANALYSIS_PATTERNS = (
    re.compile(r"\bdistribution\b"),
    re.compile(r"\bfrequency\b"),
    re.compile(r"\bshare\b"),
    re.compile(r"\bproportion\b"),
    re.compile(r"\bbreakdown\b"),
    re.compile(r"\boverall\b"),
    re.compile(r"\baggregate\b"),
    re.compile(r"\btrend\b"),
    re.compile(r"\bevolution\b"),
    re.compile(r"\bover time\b"),
    re.compile(r"\btime series\b"),
    re.compile(r"\bmonthly\b"),
    re.compile(r"\bweekly\b"),
    re.compile(r"\bdaily\b"),
    re.compile(r"\bquarterly\b"),
    re.compile(r"\bacross\b"),
    re.compile(r"\bbetween\b"),
    re.compile(r"\bcompare\b"),
    re.compile(r"\bcomparison\b"),
    re.compile(r"\bgrouped by\b"),
    re.compile(r"\bby (?:year|month|week|outlet|source|publisher|region|country|language)\b"),
)
_ANALYTIC_INTENT_PATTERNS = (
    re.compile(r"\bdistribution\b"),
    re.compile(r"\bfrequency\b"),
    re.compile(r"\bshare\b"),
    re.compile(r"\bproportion\b"),
    re.compile(r"\bbreakdown\b"),
    re.compile(r"\boverall\b"),
    re.compile(r"\baggregate\b"),
    re.compile(r"\btrend\b"),
    re.compile(r"\bevolution\b"),
    re.compile(r"\bover time\b"),
    re.compile(r"\btime series\b"),
    re.compile(r"\bcompare\b"),
    re.compile(r"\bcomparison\b"),
    re.compile(r"\bgrouped by\b"),
    re.compile(r"\bby (?:year|month|week|outlet|source|publisher|region|country|language)\b"),
)
_SEMANTIC_EXPLORATION_PATTERNS = (
    re.compile(r"\bsimilar\b"),
    re.compile(r"\bsemantic(?:ally)?\b"),
    re.compile(r"\bclosest\b"),
    re.compile(r"\bnearest\b"),
    re.compile(r"\brelated\b"),
    re.compile(r"\bparaphrase\b"),
    re.compile(r"\banalog(?:y|ous)\b"),
    re.compile(r"\btheme(?:s)?\b"),
    re.compile(r"\bconcept(?:s)?\b"),
    re.compile(r"\bembedding(?:s)?\b"),
)
_TARGETED_PATTERNS = (
    re.compile(r"\bwhich\b"),
    re.compile(r"\bwho\b"),
    re.compile(r"\bwhen\b"),
    re.compile(r"\bwhere\b"),
    re.compile(r"\bpredict(?:ed|ion)?\b"),
    re.compile(r"\bwarn(?:ed|ing)?\b"),
    re.compile(r"\bquote\b"),
    re.compile(r"\bclaim\b"),
    re.compile(r"\bexample\b"),
    re.compile(r"\bfind\b"),
    re.compile(r"\bevidence\b"),
)
_REQUESTED_LIMIT_PATTERNS = (
    re.compile(r"\btop\s+(\d{1,4})\b"),
    re.compile(r"\bfirst\s+(\d{1,4})\b"),
    re.compile(r"\b(\d{1,4})\s+(?:most common|most frequent|results|items|rows|documents|articles|reports|nouns|lemmas)\b"),
)


def default_retrieval_mode(configured_mode: str | None = None) -> str:
    configured = str(
        configured_mode
        or os.getenv("CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE", "hybrid")
    ).strip().lower()
    return configured or "hybrid"


def infer_retrieval_scope(text: str) -> str:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return "focused"
    if any(pattern.search(lowered) for pattern in _EXHAUSTIVE_PATTERNS):
        return "exhaustive"
    broad_hits = sum(1 for pattern in _BROAD_ANALYSIS_PATTERNS if pattern.search(lowered))
    targeted_hits = sum(1 for pattern in _TARGETED_PATTERNS if pattern.search(lowered))
    if broad_hits >= 3:
        return "broad"
    if broad_hits >= 1 and any(token in lowered for token in ("all ", "overall", "across", "between", "grouped", "timeline")):
        return "broad"
    if broad_hits >= 1:
        return "comparative"
    if targeted_hits >= 1:
        return "focused"
    return "focused"


def infer_requested_output_limit(
    text: str,
    *,
    default: int,
    minimum: int = 1,
    maximum: int = 1000,
) -> int:
    for pattern in _REQUESTED_LIMIT_PATTERNS:
        match = pattern.search(str(text or ""))
        if not match:
            continue
        try:
            value = int(match.group(1))
        except ValueError:
            continue
        return max(minimum, min(maximum, value))
    return max(minimum, min(maximum, default))


def infer_retrieval_strategy(
    text: str,
    *,
    inputs: Mapping[str, Any] | None = None,
    scope: str | None = None,
    lightweight: bool = False,
) -> str:
    params = dict(inputs or {})
    explicit = str(params.get("retrieval_strategy", "")).strip().lower()
    if explicit in {"exhaustive_analytic", "precision_ranked", "semantic_exploratory"}:
        return explicit
    if lightweight:
        return "precision_ranked"
    requested_all = _coerce_bool(params.get("retrieve_all")) is True
    explicit_top_k = _coerce_int(params.get("top_k")) if "top_k" in params else None
    if explicit_top_k is not None and explicit_top_k <= 0:
        requested_all = True
    resolved_scope = scope or ("exhaustive" if requested_all else infer_retrieval_scope(text))
    lowered = str(text or "").strip().lower()
    analytic_hits = sum(1 for pattern in _ANALYTIC_INTENT_PATTERNS if pattern.search(lowered))
    semantic_hits = sum(1 for pattern in _SEMANTIC_EXPLORATION_PATTERNS if pattern.search(lowered))
    targeted_hits = sum(1 for pattern in _TARGETED_PATTERNS if pattern.search(lowered))
    if requested_all or resolved_scope == "exhaustive":
        return "exhaustive_analytic"
    if semantic_hits >= 1 and analytic_hits == 0:
        return "semantic_exploratory"
    if analytic_hits >= 1 and resolved_scope in {"comparative", "broad"}:
        return "exhaustive_analytic"
    if analytic_hits >= 2:
        return "exhaustive_analytic"
    if semantic_hits >= 1 and targeted_hits == 0:
        return "semantic_exploratory"
    return "precision_ranked"


def _profile_for_scope(scope: str, *, lightweight: bool) -> _BudgetProfile:
    if lightweight:
        lightweight_top_k = _env_int("CORPUSAGENT2_RETRIEVAL_LIGHTWEIGHT_TOP_K", 12)
        return _BudgetProfile(
            top_k=lightweight_top_k,
            candidate_multiplier=3,
            min_candidates=max(15, lightweight_top_k),
            rerank_multiplier=2,
            use_rerank=False,
        )
    if scope == "exhaustive":
        return _BudgetProfile(
            top_k=_env_int("CORPUSAGENT2_RETRIEVAL_EXHAUSTIVE_TOP_K", 320),
            candidate_multiplier=8,
            min_candidates=320,
            rerank_multiplier=2,
            use_rerank=True,
        )
    if scope == "broad":
        return _BudgetProfile(
            top_k=_env_int("CORPUSAGENT2_RETRIEVAL_BROAD_TOP_K", 160),
            candidate_multiplier=7,
            min_candidates=160,
            rerank_multiplier=2,
            use_rerank=True,
        )
    if scope == "comparative":
        return _BudgetProfile(
            top_k=_env_int("CORPUSAGENT2_RETRIEVAL_COMPARATIVE_TOP_K", 80),
            candidate_multiplier=6,
            min_candidates=80,
            rerank_multiplier=2,
            use_rerank=True,
        )
    return _BudgetProfile(
        top_k=_env_int("CORPUSAGENT2_RETRIEVAL_FOCUSED_TOP_K", 40),
        candidate_multiplier=5,
        min_candidates=40,
        rerank_multiplier=2,
        use_rerank=True,
    )


def infer_retrieval_budget(
    text: str,
    *,
    inputs: Mapping[str, Any] | None = None,
    configured_mode: str | None = None,
    lightweight: bool = False,
) -> RetrievalBudget:
    params = dict(inputs or {})
    requested_all = _coerce_bool(params.get("retrieve_all")) is True
    explicit_top_k = _coerce_int(params.get("top_k")) if "top_k" in params else None
    if explicit_top_k is not None and explicit_top_k <= 0:
        requested_all = True
        explicit_top_k = None
    scope = "exhaustive" if requested_all else infer_retrieval_scope(text)
    retrieval_strategy = infer_retrieval_strategy(
        text,
        inputs=params,
        scope=scope,
        lightweight=lightweight,
    )
    effective_retrieve_all = requested_all or (
        retrieval_strategy == "exhaustive_analytic" and not lightweight
    )
    profile = _profile_for_scope(scope, lightweight=lightweight)
    top_k = explicit_top_k if explicit_top_k is not None else profile.top_k
    # Guardrail broad analytical questions away from tiny planner defaults.
    if explicit_top_k is not None and scope in {"comparative", "broad", "exhaustive"}:
        top_k = max(top_k, profile.top_k)
    top_k = max(1, top_k)
    default_mode = default_retrieval_mode(configured_mode)
    if "retrieval_mode" in params and str(params.get("retrieval_mode", "")).strip():
        retrieval_mode = str(params.get("retrieval_mode", default_mode)).strip().lower() or default_mode
    elif retrieval_strategy == "semantic_exploratory":
        retrieval_mode = "dense"
    else:
        retrieval_mode = default_mode
    explicit_use_rerank = _coerce_bool(params.get("use_rerank")) if "use_rerank" in params else None
    default_use_rerank = _env_flag("CORPUSAGENT2_RETRIEVAL_USE_RERANK", True)
    use_rerank = explicit_use_rerank if explicit_use_rerank is not None else (default_use_rerank and profile.use_rerank)
    candidate_floor = max(profile.min_candidates, top_k * profile.candidate_multiplier)
    lexical_top_k = _coerce_int(params.get("lexical_top_k")) or candidate_floor
    dense_top_k = _coerce_int(params.get("dense_top_k")) or candidate_floor
    lexical_top_k = max(lexical_top_k, top_k)
    dense_top_k = max(dense_top_k, top_k)
    max_candidates = max(lexical_top_k, dense_top_k)
    if use_rerank:
        default_rerank = min(max(top_k * profile.rerank_multiplier, top_k), max_candidates)
        rerank_top_k = _coerce_int(params.get("rerank_top_k")) or default_rerank
        rerank_top_k = max(top_k, min(rerank_top_k, max_candidates))
    else:
        rerank_top_k = 0
    fusion_k = _coerce_int(params.get("fusion_k")) or _env_int("CORPUSAGENT2_RETRIEVAL_FUSION_K", 60)
    return RetrievalBudget(
        scope=scope,
        retrieval_strategy=retrieval_strategy,
        top_k=top_k,
        lexical_top_k=lexical_top_k,
        dense_top_k=dense_top_k,
        use_rerank=use_rerank,
        rerank_top_k=rerank_top_k,
        retrieval_mode=retrieval_mode,
        fusion_k=fusion_k,
        retrieve_all_requested=effective_retrieve_all,
    )
