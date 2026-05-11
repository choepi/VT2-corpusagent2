from __future__ import annotations

from collections import Counter, defaultdict
import base64
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import threading
import textwrap
from typing import Any, Callable, Iterable
from urllib.parse import quote

import pandas as pd

from .agent_backends import PostgresWorkingSetStore, SearchBackend, WorkingSetStore
from .agent_models import EvidenceRow
from .analysis_tools import textrank_keywords
from .io_utils import sentence_split as simple_sentence_split
from .python_runner_service import DockerPythonRunnerService
from .retrieval_budgeting import infer_retrieval_budget
from .retrieval import _load_sentence_transformer, pg_dsn_from_env, retrieve_tfidf
from .model_config import dense_model_id_from_env
from .seed import resolve_device
from .tool_registry import CapabilityToolAdapter, SchemaDescriptor, ToolExecutionResult, ToolRegistry, ToolSpec


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']+")
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
QUOTE_PATTERN = re.compile(r'["“](.*?)["”]')
SPEAKER_PATTERN = re.compile(
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(said|says|warned|argued|claimed|according to)",
    re.IGNORECASE,
)
MATPLOTLIB_PLOT_LOCK = threading.RLock()

STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "have", "were", "their", "about", "into", "there",
    "would", "could", "should", "while", "after", "before", "where", "which", "what", "when", "been", "being",
    "over", "under", "between", "across", "relationship", "systematic", "movements", "movement",
    "coverage", "media", "sentiment", "price", "prices", "analysis",
}
POSITIVE_WORDS = {"good", "strong", "gain", "improve", "success", "positive", "optimistic"}
NEGATIVE_WORDS = {"bad", "weak", "loss", "drop", "risk", "fear", "negative", "warn", "crisis"}
CLAIM_KEYWORDS = {
    "anticipate",
    "anticipated",
    "anticipates",
    "claim",
    "claimed",
    "claims",
    "expect",
    "expected",
    "expects",
    "forecast",
    "forecasted",
    "forecasts",
    "foresaw",
    "foresee",
    "imminent",
    "likely",
    "predict",
    "predicted",
    "prediction",
    "risk",
    "warn",
    "warned",
    "warning",
}
LANGUAGE_HINTS = {
    "de": {"und", "der", "die", "das", "nicht", "mit", "ist", "ein"},
    "fr": {"le", "la", "les", "des", "une", "est", "avec", "dans"},
    "it": {"il", "lo", "la", "gli", "che", "con", "per", "non"},
    "en": {"the", "and", "with", "from", "that", "this", "have", "will"},
}
STRUCTURAL_TERM_STOPWORDS = {
    "title",
    "url",
    "description",
    "author",
    "authors",
    "content",
    "article",
    "articles",
    "report",
    "reports",
    "document",
    "documents",
    "source",
    "sources",
    "image",
    "images",
    "caption",
    "tags",
    "news",
    "reuters",
    "said",
    "say",
    "id",
    "data",
    "json",
}
FUNCTION_WORD_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet", "if", "then", "than", "because", "as",
    "of", "to", "in", "on", "at", "by", "for", "from", "with", "without", "into", "onto", "out", "up",
    "down", "off", "over", "under", "through", "across", "between", "among", "against", "during",
    "since", "until", "via", "per", "above", "below", "near",
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours", "he", "him", "his",
    "she", "her", "hers", "it", "its", "they", "them", "their", "theirs", "who", "whom", "whose",
    "whoever", "what", "whatever", "which", "whichever", "that", "this", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done", "doing",
    "have", "has", "had", "having", "will", "would", "shall", "should", "can", "could", "may",
    "might", "must",
    "not", "no", "yes", "just", "very", "also", "only", "even", "ever", "never", "again", "still",
    "now", "here", "there", "why", "where", "when", "how", "back",
    "more", "most", "less", "least", "many", "much", "some", "any", "all", "both", "each", "either",
    "neither", "one", "two", "three", "first", "second", "last", "new", "latest", "former", "other",
    "another", "same", "own", "few", "several", "such",
    "said", "say", "says", "told", "according", "reported", "announced", "asked", "added", "called",
    "made", "make", "makes", "got", "get", "gets", "take", "takes", "took", "taken", "won",
    "go", "goes", "going", "went", "gone", "want", "wants", "wanted", "need", "needs", "needed",
    "see", "sees", "saw", "seen", "know", "knows", "knew", "known", "think", "thinks", "thought",
    "come", "comes", "came", "coming", "look", "looks", "looked", "looking", "use", "uses", "used",
    "work", "works", "worked", "working",
    "like", "well", "good", "better", "best", "right",
}
ENTITY_SURFACE_STOPWORDS = {
    "unknown", "none", "null", "n/a",
    "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "theirs", "we", "our", "ours", "you", "your", "yours",
    "this", "that", "these", "those", "who", "whom", "whose", "which", "what",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done", "doing", "have", "has", "had", "having",
    "said", "say", "says", "told", "reported", "announced", "asked", "added",
    "report", "reports", "article", "articles", "story", "stories", "news",
}
NULL_AXIS_LABELS = {"", "unknown", "unkn", "__unknown__", "none", "null", "nan", "nat", "n/a", "na", "-", "--"}
TEMPLATE_TERM_STOPWORDS = {
    "taboola", "window", "push", "container", "placement", "target_type", "target", "type", "mode",
    "mix", "thumbnail", "thumbnails", "interstitial", "gallery", "caption", "close", "photo", "image",
    "javascript", "subscribe", "subscription", "email", "advertisement", "advertisements", "ad", "ads",
    "script", "function", "return", "var", "let", "const", "try", "catch", "error", "undefined",
    "tmr", "banner", "click", "scroll", "start", "continue", "read", "watch", "video", "photo",
}
SERIES_SURFACE_STOPWORDS = (
    STOPWORDS | STRUCTURAL_TERM_STOPWORDS | FUNCTION_WORD_STOPWORDS | TEMPLATE_TERM_STOPWORDS | ENTITY_SURFACE_STOPWORDS
)

# Stopword set used by the *display layer* to scrub topic / frame labels
# before they reach plots or the answer. The vectorizers in _topic_model
# already pass stop_words="english", but the heuristic fallback skips that,
# and gensim/textacy occasionally still surface function words. This is
# the final safety net so charts never show labels like "in of league is".
TOPIC_LABEL_STOPWORDS = (
    STOPWORDS | FUNCTION_WORD_STOPWORDS | STRUCTURAL_TERM_STOPWORDS | ENTITY_SURFACE_STOPWORDS
)


def clean_topic_terms(top_terms: Any, *, max_count: int = 4, min_length: int = 3) -> list[str]:
    """Return display-safe topic terms.

    Drops English function words, structural article-template tokens, and
    very short fragments (<3 chars). Preserves order so the strongest
    remaining terms surface first. Lower-cases for stopword matching but
    returns the original casing.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for term in top_terms or []:
        text = str(term or "").strip()
        if not text:
            continue
        normalized = text.lower()
        if len(normalized) < min_length:
            continue
        if normalized in TOPIC_LABEL_STOPWORDS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(text)
        if len(cleaned) >= max_count:
            break
    return cleaned


def _is_placeholder_axis_label(value: Any) -> bool:
    return str(value or "").strip().lower() in NULL_AXIS_LABELS
MACHINE_PAYLOAD_MARKERS = {
    "analytics",
    "article_type",
    "author_name",
    "content_type",
    "duration",
    "embed_url",
    "image_url",
    "playlist",
    "thumbnail_url",
    "video_id",
    "video_playlist",
}
HUMAN_TEXT_JSON_KEYS = {
    "article",
    "body",
    "caption",
    "dek",
    "description",
    "headline",
    "lede",
    "summary",
    "text",
    "title",
}
NOUN_LEMMA_STOPWORDS = (
    STOPWORDS | STRUCTURAL_TERM_STOPWORDS | FUNCTION_WORD_STOPWORDS | TEMPLATE_TERM_STOPWORDS | MACHINE_PAYLOAD_MARKERS
)


def _valid_noun_lemma(lemma: str, *, min_length: int = 2) -> bool:
    normalized = str(lemma or "").strip().lower()
    if not normalized or len(normalized) < min_length:
        return False
    if normalized in NOUN_LEMMA_STOPWORDS:
        return False
    return bool(re.search(r"[a-z]", normalized))


def _collapse_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split()).strip()


def _looks_machine_payload(text: str) -> bool:
    sample = str(text or "")[:12000]
    if not sample.strip():
        return False
    lower = sample.lower()
    marker_hits = sum(1 for marker in MACHINE_PAYLOAD_MARKERS if marker in lower)
    json_key_hits = len(re.findall(r'"[A-Za-z_][A-Za-z0-9_\-]{1,48}"\s*:', sample))
    structural_chars = sum(sample.count(char) for char in "{}[]:,")
    token_count = max(1, len(re.findall(r"\w+", sample)))
    structural_density = structural_chars / max(1, len(sample))
    key_density = json_key_hits / token_count
    starts_like_payload = sample.lstrip().startswith(("{", "[", '","', '":'))
    return bool(
        (starts_like_payload and json_key_hits >= 3)
        or (json_key_hits >= 8 and marker_hits >= 2)
        or (json_key_hits >= 18 and key_density >= 0.08)
        or (marker_hits >= 4 and structural_density >= 0.08)
    )


def _decode_jsonish_string(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        decoded = json.loads(f'"{raw}"')
    except Exception:
        decoded = raw.replace('\\"', '"').replace("\\n", " ").replace("\\/", "/")
    return _collapse_text(decoded)


def _extract_human_text_from_payload(text: str) -> list[str]:
    sample = str(text or "")[:60000]
    chunks: list[str] = []
    seen: set[str] = set()
    key_pattern = "|".join(re.escape(key) for key in sorted(HUMAN_TEXT_JSON_KEYS))
    for match in re.finditer(rf'"(?:{key_pattern})"\s*:\s*"((?:\\.|[^"\\])*)"', sample, flags=re.IGNORECASE):
        candidate = _decode_jsonish_string(match.group(1))
        normalized = candidate.lower()
        if len(candidate) < 20 or normalized in {"null", "none", "undefined"}:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        chunks.append(candidate)
    return chunks[:8]


def _analysis_text_char_limit() -> int:
    raw = os.getenv("CORPUSAGENT2_ANALYSIS_TEXT_CHAR_LIMIT", "20000").strip()
    try:
        return max(1000, int(raw))
    except ValueError:
        return 20000


def _noun_spacy_max_documents() -> int | None:
    raw = os.getenv("CORPUSAGENT2_NOUN_SPACY_MAX_DOCS", "-1").strip()
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 0:
        return None
    return max(0, value)


def _sql_aggregate_text_char_limit() -> int:
    raw = os.getenv("CORPUSAGENT2_SQL_AGGREGATE_TEXT_CHAR_LIMIT", "4000").strip()
    try:
        return max(500, int(raw))
    except ValueError:
        return 4000


def _working_set_analysis_max_documents() -> int | None:
    configured = os.getenv("CORPUSAGENT2_WORKING_SET_ANALYSIS_MAX_DOCS")
    if configured is None:
        return None
    raw = configured.strip()
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 0:
        return None
    return max(1, value)


def _working_set_analysis_limit_explicit() -> bool:
    return os.getenv("CORPUSAGENT2_WORKING_SET_ANALYSIS_MAX_DOCS") is not None


def _clean_analysis_text(title: Any = "", text: Any = "") -> tuple[str, dict[str, bool]]:
    title_text = _collapse_text(title)
    raw_body_text = str(text or "").replace("\x00", " ")
    flags = {"machine_payload": False, "truncated": False}
    if not raw_body_text.strip():
        return title_text, flags
    limit = _sql_aggregate_text_char_limit()
    if _looks_machine_payload(raw_body_text):
        flags["machine_payload"] = True
        chunks = [title_text, *_extract_human_text_from_payload(raw_body_text)]
        cleaned = _collapse_text(" ".join(chunk for chunk in chunks if chunk))
    else:
        body_slice = raw_body_text
        if len(body_slice) > limit * 2:
            flags["truncated"] = True
            body_slice = body_slice[: limit * 2]
        cleaned = _collapse_text(f"{title_text} {body_slice}".strip())
        cleaned = re.sub(r"_taboola\s*=\s*window\._taboola\s*\|\|\s*\[\]\s*;?", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"_taboola\.push\([^)]*\)\s*;?", " ", cleaned, flags=re.IGNORECASE)
        cleaned = _collapse_text(cleaned)
    if len(cleaned) > limit:
        flags["truncated"] = True
        cleaned = cleaned[:limit]
    return cleaned, flags


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_time_granularity() -> str:
    return (os.getenv("CORPUSAGENT2_TIME_GRANULARITY", "month").strip().lower() or "month")


QUERY_ANCHOR_STOPWORDS = {
    "about",
    "across",
    "after",
    "aggregate",
    "aggregated",
    "all",
    "america",
    "american",
    "analysed",
    "analysis",
    "analyze",
    "analyzed",
    "and",
    "association",
    "associations",
    "associated",
    "attitude",
    "attitudes",
    "around",
    "article",
    "articles",
    "actor",
    "actors",
    "available",
    "before",
    "been",
    "being",
    "between",
    "breakdown",
    "administration",
    "change",
    "changed",
    "changes",
    "changing",
    "collection",
    "common",
    "complete",
    "compare",
    "compared",
    "comparison",
    "corpus",
    "coverage",
    "could",
    "daily",
    "dataset",
    "did",
    "different",
    "differently",
    "difference",
    "differences",
    "dominate",
    "dominates",
    "dominated",
    "dominant",
    "distribution",
    "document",
    "documents",
    "does",
    "during",
    "each",
    "entire",
    "every",
    "evolve",
    "evolved",
    "evolution",
    "explain",
    "explained",
    "explains",
    "for",
    "frame",
    "framing",
    "frequencies",
    "frequency",
    "from",
    "full",
    "heightened",
    "her",
    "have",
    "his",
    "how",
    "identified",
    "identifying",
    "identify",
    "include",
    "included",
    "including",
    "individual",
    "into",
    "its",
    "lemma",
    "lemmas",
    "media",
    "monthly",
    "most",
    "news",
    "newspaper",
    "newspapers",
    "noun",
    "nouns",
    "overall",
    "outlet",
    "outlets",
    "over",
    "pattern",
    "patterns",
    "period",
    "periods",
    "perception",
    "perceptions",
    "perceived",
    "portrayal",
    "portrayed",
    "presidency",
    "public",
    "record",
    "records",
    "relative",
    "related",
    "relation",
    "relations",
    "relationship",
    "relationships",
    "relevant",
    "discourse",
    "report",
    "reports",
    "result",
    "results",
    "row",
    "rows",
    "sentiment",
    "shift",
    "shifted",
    "should",
    "specific",
    "state",
    "states",
    "stories",
    "story",
    "subperiod",
    "subperiods",
    "such",
    "swiss",
    "switzerland",
    "that",
    "the",
    "their",
    "there",
    "this",
    "through",
    "time",
    "tone",
    "tones",
    "toward",
    "towards",
    "trend",
    "trends",
    "under",
    "until",
    "united",
    "usa",
    "used",
    "using",
    "weekly",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whole",
    "within",
    "why",
    "with",
    "would",
    "yearly",
    "versus",
}


def _add_query_anchor(anchor_terms: list[str], seen: set[str], token: str, *, blocked: set[str] | None = None) -> None:
    blocked = blocked or set()
    for part in re.findall(r"[A-Za-z][A-Za-z0-9]+", str(token or "")):
        lowered = part.lower()
        if len(lowered) < 3 or lowered in QUERY_ANCHOR_STOPWORDS or lowered in seen or lowered in blocked:
            continue
        seen.add(lowered)
        anchor_terms.append(lowered)


GROUPED_FIELD_FILTER_PATTERN = re.compile(
    r"\b(?P<field>source|outlet|site|domain)\s*:\s*\((?P<group>[^)]*)\)",
    flags=re.IGNORECASE,
)
FIELD_FILTER_PATTERN = re.compile(
    r"\b(?P<field>source|outlet|site|domain)\s*:\s*(?:\"(?P<quoted>[^\"]+)\"|'(?P<single>[^']+)'|(?P<bare>[^\s)]+))",
    flags=re.IGNORECASE,
)


def _normalize_source_filter(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _query_source_filters(query: str) -> list[str]:
    filters: list[str] = []
    seen: set[str] = set()
    for group_match in GROUPED_FIELD_FILTER_PATTERN.finditer(str(query or "")):
        group = group_match.group("group") or ""
        values = re.findall(r'"([^"]+)"|\'([^\']+)\'|([A-Za-z0-9_.-]+)', group)
        for parts in values:
            value = next((part for part in parts if part), "")
            if value.upper() in {"OR", "AND", "NOT"}:
                continue
            normalized = _normalize_source_filter(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                filters.append(normalized)
    for match in FIELD_FILTER_PATTERN.finditer(str(query or "")):
        value = match.group("quoted") or match.group("single") or match.group("bare") or ""
        if str(value).startswith("("):
            continue
        normalized = _normalize_source_filter(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            filters.append(normalized)
    return filters


def _query_without_field_filters(query: str) -> str:
    text = GROUPED_FIELD_FILTER_PATTERN.sub(" ", str(query or ""))
    text = FIELD_FILTER_PATTERN.sub(" ", text)
    text = re.sub(r"\b(?:AND|OR|NOT)\b\s*(?=[)\s]*$)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\)", " ", text)
    return " ".join(text.split())


def _row_matches_source_filters(source: Any, filters: list[str]) -> bool:
    if not filters:
        return True
    normalized = _normalize_source_filter(str(source or ""))
    return any(item in normalized for item in filters)


def _source_filter_match_counts(context: "AgentExecutionContext", filters: list[str]) -> dict[str, int]:
    if not filters or context.runtime is None:
        return {}
    try:
        metadata = context.runtime.load_metadata()
    except Exception:
        return {}
    if metadata is None or metadata.empty or "source" not in metadata.columns:
        return {}
    normalized_sources = metadata["source"].fillna("").astype(str).map(_normalize_source_filter)
    counts: dict[str, int] = {}
    for item in filters:
        counts[item] = int(normalized_sources.str.contains(re.escape(item), regex=True, na=False).sum())
    return counts


def _source_filtered_no_data_caveat(context: "AgentExecutionContext", filters: list[str]) -> str:
    counts = _source_filter_match_counts(context, filters)
    if counts and not any(counts.values()):
        rendered = ", ".join(filters)
        return f"No corpus documents matched the requested source filters ({rendered}); the requested outlets may be absent from this corpus."
    if counts:
        rendered_counts = ", ".join(f"{source}={count}" for source, count in counts.items())
        return (
            "No documents matched both the requested source filters and the main query terms. "
            f"Corpus source-filter coverage: {rendered_counts}."
        )
    return f"No documents matched source filters: {', '.join(filters)}."


def _query_anchor_terms(query: str) -> list[str]:
    text = _query_without_field_filters(str(query or "")).strip()
    if not text:
        return []
    resolved_terms: list[str] = []
    resolved_seen: set[str] = set()
    blocked_terms: set[str] = set()
    if re.search(r"\b(?:19|20)\d{2}\b", text) and re.search(
        r"\b(?:from|since|starting|started|beginning|began|after|through|until|to)\b",
        text,
        flags=re.IGNORECASE,
    ):
        blocked_terms.update({"campaign", "presidency", "administration"})
    for match in re.finditer(
        r"['\"]?([A-Za-z][A-Za-z0-9-]+)['\"]?\s+(?:means|refers\s+to|interpreted\s+as)\s+['\"]?([A-Za-z][A-Za-z0-9-]+)['\"]?",
        text,
        flags=re.IGNORECASE,
    ):
        blocked_terms.update(part.lower() for part in re.findall(r"[A-Za-z][A-Za-z0-9]+", match.group(1)))
        _add_query_anchor(resolved_terms, resolved_seen, match.group(2), blocked=blocked_terms)
    if resolved_terms:
        return resolved_terms[:16]
    preferred: list[str] = []
    preferred_seen: set[str] = set()
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Z]?[A-Za-z0-9]+)?\b", text):
        token = match.group(0).strip()
        _add_query_anchor(preferred, preferred_seen, token, blocked=blocked_terms)
    fallback: list[str] = []
    fallback_seen: set[str] = set(preferred_seen)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9-]+", text):
        _add_query_anchor(fallback, fallback_seen, token, blocked=blocked_terms)
    return [*preferred, *fallback][:16]


def _query_or_groups(query: str) -> list[str]:
    text = _strip_wrapping_parentheses(_query_without_field_filters(str(query or "")).strip())
    if not text or not re.search(r"\bOR\b|\|", text, flags=re.IGNORECASE):
        return []
    groups: list[str] = []
    seen: set[str] = set()
    for raw_group in _split_top_level_boolean(text, "OR"):
        anchors = _query_anchor_terms(raw_group)
        if not anchors:
            continue
        group = " ".join(anchors[:16]).strip()
        if group and group not in seen:
            seen.add(group)
            groups.append(group)
    return groups


def _strip_wrapping_parentheses(text: str) -> str:
    value = str(text or "").strip()
    while value.startswith("(") and value.endswith(")"):
        depth = 0
        balanced = True
        for index, char in enumerate(value):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(value) - 1:
                    balanced = False
                    break
        if not balanced:
            break
        value = value[1:-1].strip()
    return value


def _split_top_level_boolean(text: str, operator: str) -> list[str]:
    value = str(text or "").strip()
    operator = str(operator or "").strip().upper()
    if not value or operator not in {"AND", "OR"}:
        return [value] if value else []
    separator_pattern = re.compile(
        rf"(?:\s+\b{re.escape(operator)}\b\s+|\s*\|\s*)" if operator == "OR" else rf"\s+\b{re.escape(operator)}\b\s+",
        flags=re.IGNORECASE,
    )
    groups: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(value):
        char = value[index]
        if quote:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            index += 1
            continue
        if depth == 0:
            match = separator_pattern.match(value, index)
            if match:
                groups.append(_strip_wrapping_parentheses(value[start:index]))
                index = match.end()
                start = index
                continue
        index += 1
    groups.append(_strip_wrapping_parentheses(value[start:]))
    return [group for group in groups if group]


def _split_top_level_and_groups(query: str) -> list[str]:
    text = _strip_wrapping_parentheses(_query_without_field_filters(str(query or "")).strip())
    if not text:
        return []
    return _split_top_level_boolean(text, "AND")


def _sql_websearch_query_text(query: str, tokens: list[str]) -> str:
    # Preserve an explicit planner/user disjunction. Joining these tokens with
    # spaces would turn a broad OR query into an accidental AND query.
    or_groups = _query_or_groups(query)
    if len(or_groups) > 1:
        return " OR ".join(or_groups[:32])
    return " ".join(tokens)


def _sql_websearch_query_clauses(query: str, tokens: list[str]) -> list[str]:
    and_groups = _split_top_level_and_groups(query)
    if len(and_groups) > 1:
        clauses = []
        for group in and_groups:
            group_tokens = _query_anchor_terms(group)
            if not group_tokens:
                continue
            clauses.append(_sql_websearch_query_text(group, group_tokens))
        if len(clauses) > 1:
            return clauses[:8]
    query_text = _sql_websearch_query_text(query, tokens)
    return [query_text] if query_text else []


def _row_matches_query_expression(row: dict[str, Any], query: str) -> bool:
    text = " ".join(
        str(row.get(field, "") or "")
        for field in ("title", "text", "cleaned_text", "snippet", "source", "outlet")
    ).lower()
    if not text.strip():
        return False

    def matches_expression(expression: str) -> bool:
        current = _strip_wrapping_parentheses(_query_without_field_filters(str(expression or "")).strip())
        if not current:
            return True
        and_groups = _split_top_level_boolean(current, "AND")
        if len(and_groups) > 1:
            return all(matches_expression(group) for group in and_groups)
        or_groups = _split_top_level_boolean(current, "OR")
        if len(or_groups) > 1:
            return any(matches_expression(group) for group in or_groups)
        anchors = _query_anchor_terms(current)
        if not anchors:
            return True
        return _anchor_hit_count(text, anchors) >= _min_exhaustive_anchor_hits(anchors)

    return matches_expression(query)


def _payload_or_params(params: dict[str, Any]) -> dict[str, Any]:
    payload = params.get("payload")
    if not isinstance(payload, dict):
        return dict(params)
    merged = dict(payload)
    for key, value in params.items():
        if key == "payload" or key in merged:
            continue
        merged[key] = value
    return merged


def _dependency_rows(dependency_results: dict[str, ToolExecutionResult]) -> list[dict[str, Any]]:
    for result in dependency_results.values():
        payload = result.payload
        if not isinstance(payload, dict):
            continue
        for key in ("documents", "results", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
    return []


TEXT_ROW_FIELDS = ("text", "cleaned_text", "claim_span", "excerpt", "quote")


def _coerce_text_document_row(row: dict[str, Any]) -> dict[str, Any]:
    copy = dict(row)
    if not str(copy.get("text", copy.get("cleaned_text", ""))).strip():
        sentences = copy.get("sentences")
        if isinstance(sentences, list):
            joined = " ".join(str(item).strip() for item in sentences if str(item).strip())
            if joined:
                copy["text"] = joined
        if not str(copy.get("text", copy.get("cleaned_text", ""))).strip():
            for field in ("claim_span", "excerpt", "quote"):
                value = str(copy.get(field, "") or "").strip()
                if value:
                    copy["text"] = value
                    break
    return _with_cleaned_document_text(copy)


def _row_has_text_payload(row: dict[str, Any]) -> bool:
    if isinstance(row.get("sentences"), list) and row.get("sentences"):
        return True
    return any(str(row.get(field, "") or "").strip() for field in TEXT_ROW_FIELDS)


def _with_cleaned_document_text(row: dict[str, Any]) -> dict[str, Any]:
    copy = dict(row)
    if str(copy.get("cleaned_text", "")).strip():
        return copy
    cleaned, flags = _clean_analysis_text(copy.get("title", ""), copy.get("text", ""))
    if cleaned:
        copy["cleaned_text"] = cleaned
        if flags.get("machine_payload"):
            copy["text"] = cleaned
    if flags.get("machine_payload"):
        copy["text_is_machine_payload"] = True
    if flags.get("truncated"):
        copy["analysis_text_truncated"] = True
    return copy


def _text_rows(dependency_results: dict[str, ToolExecutionResult]) -> list[dict[str, Any]]:
    direct = _doc_rows(dependency_results)
    if direct:
        return direct
    rows: list[dict[str, Any]] = []
    for row in _dependency_rows(dependency_results):
        text = str(row.get("text", row.get("cleaned_text", ""))).strip()
        if not text and not str(row.get("title", "")).strip():
            continue
        copy = _coerce_text_document_row(dict(row))
        if "text" not in copy and text:
            copy["text"] = text
        rows.append(copy)
    return rows


def _entity_analysis_max_documents() -> int | None:
    return _optional_positive_int_env("CORPUSAGENT2_ENTITY_ANALYSIS_MAX_DOCS")


def _topic_model_analysis_max_documents() -> int | None:
    return _optional_positive_int_env("CORPUSAGENT2_TOPIC_MODEL_ANALYSIS_MAX_DOCS")


def _sentiment_analysis_max_documents() -> int | None:
    return _optional_positive_int_env("CORPUSAGENT2_SENTIMENT_ANALYSIS_MAX_DOCS")


def _optional_positive_int_env(name: str) -> int | None:
    raw_env = os.getenv(name)
    if raw_env is None:
        return None
    raw = raw_env.strip()
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 0:
        return None
    return max(1, value)


def _entity_provider_max_documents() -> int | None:
    raw = os.getenv("CORPUSAGENT2_ENTITY_PROVIDER_MAX_DOCS", "-1").strip()
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 0:
        return None
    return max(1, value)


def _analysis_document_rows_from_deps(
    deps: dict[str, ToolExecutionResult],
    context: "AgentExecutionContext",
    *,
    max_documents: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    documents = _doc_rows(deps)
    working_set_ref = _working_set_ref(deps)
    documents_truncated = _dependency_payload_flag(deps, "documents_truncated") or _dependency_payload_flag(deps, "working_set_truncated")
    if documents and not (documents_truncated and working_set_ref):
        return documents, {"analyzed_document_count": len(documents), "documents_from": "dependency_rows"}, []
    if not working_set_ref:
        if documents:
            caveats = [
                "Upstream documents were marked truncated, but no working_set_ref was available; analysis used preview rows only."
            ] if documents_truncated else []
            return documents, {
                "analyzed_document_count": len(documents),
                "documents_from": "dependency_rows",
                "preview_only": bool(documents_truncated),
            }, caveats
        return [], {"analyzed_document_count": 0, "documents_from": "none"}, []
    rows = [
        _with_cleaned_document_text(row)
        for row in _iter_working_set_documents(context, working_set_ref, max_documents=max_documents)
    ]
    cancelled = _cancel_requested(context)
    working_set_count = _count_working_set(context, working_set_ref, len(rows))
    caveats = [
        "Upstream fetched documents were only a preview, so documents were streamed from working_set_ref for analysis."
        if documents_truncated
        else "No fetched document rows were available, so documents were streamed from working_set_ref for analysis."
    ]
    if cancelled:
        caveats.append("Run abort was requested; analysis document streaming stopped early.")
    if max_documents is not None and working_set_count > max_documents:
        caveats.append(
            f"Working-set analysis was capped at {max_documents} documents from a working set of {working_set_count}; "
            "set the relevant *_ANALYSIS_MAX_DOCS environment variable to -1 for an uncapped offline run."
        )
    return rows, {
        "analyzed_document_count": len(rows),
        "working_set_document_count": working_set_count,
        "working_set_ref": working_set_ref,
        "documents_from": "working_set_ref",
        "analysis_document_limit": max_documents,
        "cancelled": cancelled,
    }, caveats


def _working_set_doc_ids(dependency_results: dict[str, ToolExecutionResult]) -> list[str]:
    for result in dependency_results.values():
        payload = result.payload
        if not isinstance(payload, dict):
            continue
        ids = payload.get("working_set_doc_ids")
        if isinstance(ids, list):
            return [str(item).strip() for item in ids if str(item).strip()]
    return []


def _working_set_ref(dependency_results: dict[str, ToolExecutionResult]) -> str:
    for result in dependency_results.values():
        payload = result.payload
        if not isinstance(payload, dict):
            continue
        ref = str(payload.get("working_set_ref", "")).strip()
        if ref:
            return ref
    return ""


def _looks_like_plan_node_ref(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    return bool(re.fullmatch(r"n\d+", normalized)) or normalized in {
        "search",
        "fetch",
        "documents",
        "working_set",
        "working-set",
    }


def _resolve_working_set_ref(params: dict[str, Any], dependency_results: dict[str, ToolExecutionResult]) -> str:
    for key in (
        "working_set_ref",
        "working_set",
        "working_set_from",
        "working_set_source",
        "working_set_source_node",
        "working_set_source_node_id",
        "source_node",
        "source_node_id",
    ):
        raw = str(params.get(key, "") or "").strip()
        if not raw:
            continue
        if raw in dependency_results:
            payload = dependency_results[raw].payload
            if isinstance(payload, dict) and str(payload.get("working_set_ref", "")).strip():
                return str(payload.get("working_set_ref", "")).strip()
        if _looks_like_plan_node_ref(raw):
            continue
        return raw
    return _working_set_ref(dependency_results)


def _merged_payload_params(params: dict[str, Any]) -> dict[str, Any]:
    payload = params.get("payload") if isinstance(params.get("payload"), dict) else {}
    merged = dict(payload)
    for key, value in params.items():
        if key != "payload":
            merged[key] = value
    return merged


def _dependency_payload_flag(dependency_results: dict[str, ToolExecutionResult], key: str) -> bool:
    for result in dependency_results.values():
        payload = result.payload
        if isinstance(payload, dict) and bool(payload.get(key)):
            return True
    return False


def _dependency_payload_int(dependency_results: dict[str, ToolExecutionResult], key: str, default: int = 0) -> int:
    for result in dependency_results.values():
        payload = result.payload
        if not isinstance(payload, dict) or payload.get(key) in (None, ""):
            continue
        try:
            return int(payload.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return default


def _dedupe_doc_ids(doc_ids: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for doc_id in doc_ids:
        normalized = str(doc_id).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _result_preview_limit() -> int:
    raw = os.getenv("CORPUSAGENT2_RESULT_PREVIEW_ROWS", "50").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 50


def _count_working_set(context: "AgentExecutionContext", label: str, fallback: int = 0) -> int:
    counter = getattr(context.working_store, "count_working_set", None)
    if not callable(counter):
        return fallback
    try:
        return int(counter(context.run_id, label))
    except Exception:
        return fallback


def _fetch_working_set_ids(
    context: "AgentExecutionContext",
    label: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[str]:
    fetcher = getattr(context.working_store, "fetch_working_set_doc_ids", None)
    if not callable(fetcher):
        return []
    try:
        return [str(item).strip() for item in fetcher(context.run_id, label, limit=limit, offset=offset) if str(item).strip()]
    except Exception:
        return []


def _iter_working_set_documents(
    context: "AgentExecutionContext",
    label: str,
    *,
    batch_size: int | None = None,
    max_documents: int | None = None,
):
    if not label:
        return
    fetcher = getattr(context.working_store, "fetch_working_set_documents", None)
    if not callable(fetcher):
        return
    resolved_batch_size = int(batch_size or os.getenv("CORPUSAGENT2_WORKING_SET_ANALYSIS_BATCH_SIZE", "1000") or 1000)
    resolved_batch_size = max(1, resolved_batch_size)
    offset = 0
    yielded = 0
    while True:
        if _cancel_requested(context):
            break
        if max_documents is not None and yielded >= max_documents:
            break
        limit = resolved_batch_size
        if max_documents is not None:
            limit = min(limit, max(0, max_documents - yielded))
        if limit <= 0:
            break
        rows = fetcher(context.run_id, label, limit=limit, offset=offset)
        if not rows:
            break
        for row in rows:
            if _cancel_requested(context):
                break
            if isinstance(row, dict):
                yield dict(row)
                yielded += 1
                if max_documents is not None and yielded >= max_documents:
                    break
        if _cancel_requested(context):
            break
        if len(rows) < resolved_batch_size:
            break
        offset += len(rows)


def _cancel_requested(context: "AgentExecutionContext") -> bool:
    return bool(context.cancel_requested is not None and context.cancel_requested())


def _sql_noun_frequency_rows_from_working_set(
    context: "AgentExecutionContext",
    working_set_ref: str,
    *,
    top_k: int,
    max_documents: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    if not isinstance(context.working_store, PostgresWorkingSetStore):
        return None
    store = context.working_store
    try:
        columns = store._document_columns()
    except Exception:
        return None
    table_name = store._safe_identifier(store.documents_table)
    title_expr = f"COALESCE(doc.{columns['title']}::text, '')" if columns["title"] else "''"
    text_expr = f"COALESCE(doc.{columns['text']}::text, '')"
    machine_checks = " OR ".join(
        f"LOWER({text_expr}) LIKE %s"
        for _ in ("duration", "thumbnail", "analytics", "video_playlist")
    )
    stopwords = sorted(NOUN_LEMMA_STOPWORDS)
    limit = _sql_aggregate_text_char_limit()
    token_regex = r"[A-Za-z][A-Za-z'-]+"
    sql = f"""
        WITH selected_ws AS (
            SELECT doc_id, rank
            FROM ca_agent_working_set_docs
            WHERE run_id = %s AND label = %s
            ORDER BY rank, doc_id
            {"LIMIT %s" if max_documents is not None else ""}
        ),
        docs AS (
            SELECT
                doc.{columns['doc_id']}::text AS doc_id,
                (
                    {title_expr}
                    || ' '
                    || CASE WHEN ({machine_checks}) THEN '' ELSE LEFT({text_expr}, %s) END
                ) AS analysis_text
            FROM selected_ws ws
            JOIN {table_name} doc ON doc.{columns['doc_id']}::text = ws.doc_id
        ),
        tokens AS (
            SELECT
                docs.doc_id,
                LOWER(match.token[1]) AS token
            FROM docs
            CROSS JOIN LATERAL regexp_matches(docs.analysis_text, %s, 'g') AS match(token)
        ),
        lemmas AS (
            SELECT
                doc_id,
                CASE
                    WHEN token LIKE '%%ies' AND char_length(token) > 4
                        THEN substring(token from 1 for char_length(token) - 3) || 'y'
                    WHEN token LIKE '%%s' AND char_length(token) > 3
                        THEN substring(token from 1 for char_length(token) - 1)
                    ELSE token
                END AS lemma
            FROM tokens
            WHERE char_length(token) >= 3 AND NOT (token = ANY(%s))
        ),
        grouped AS (
            SELECT lemma, COUNT(*)::bigint AS count, COUNT(DISTINCT doc_id)::bigint AS document_frequency
            FROM lemmas
            WHERE char_length(lemma) >= 3
              AND lemma ~ '[a-z]'
              AND NOT (lemma = ANY(%s))
            GROUP BY lemma
        ),
        totals AS (
            SELECT COALESCE(SUM(count), 0)::bigint AS total_count FROM grouped
        )
        SELECT grouped.lemma, grouped.count, grouped.document_frequency, totals.total_count
        FROM grouped
        CROSS JOIN totals
        ORDER BY grouped.count DESC, grouped.lemma
        LIMIT %s
    """
    params: list[Any] = [
        context.run_id,
        working_set_ref,
    ]
    if max_documents is not None:
        params.append(max(1, int(max_documents)))
    params.extend(
        [
        "%\"duration\"%",
        "%thumbnail%",
        "%analytics%",
        "%video_playlist%",
        limit,
        token_regex,
        stopwords,
        stopwords,
        max(1, int(top_k)),
        ]
    )
    try:
        with store._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()
    except Exception:
        return None
    total_tokens = int(rows[0][3] or 0) if rows else 0
    payload_rows = [
        {
            "lemma": str(row[0]),
            "count": int(row[1] or 0),
            "relative_frequency": round(int(row[1] or 0) / max(total_tokens, 1), 6),
            "document_frequency": int(row[2] or 0),
            "rank": rank,
        }
        for rank, row in enumerate(rows, start=1)
        if _valid_noun_lemma(str(row[0]), min_length=3)
    ]
    return payload_rows, {
        "provider": "postgres_token_aggregate",
        "analyzed_document_count": min(_count_working_set(context, working_set_ref, 0), max_documents)
        if max_documents is not None
        else _count_working_set(context, working_set_ref, 0),
        "total_noun_tokens": total_tokens,
        "full_working_set": True,
        "working_set_ref": working_set_ref,
        "analysis_text_char_limit": limit,
        "analysis_text_limit_scope": "postgres_token_aggregate_per_document",
        "analysis_document_limit": max_documents,
    }


def _materialize_result_working_set(
    context: "AgentExecutionContext",
    *,
    query: str,
    retrieval_mode: str,
    rows: list[dict[str, Any]],
) -> tuple[str, int]:
    recorder = getattr(context.working_store, "record_working_set", None)
    if not callable(recorder) or not rows:
        return "", 0
    digest = sha256(f"{retrieval_mode}\n{query}".encode("utf-8")).hexdigest()[:12]
    label = f"{retrieval_mode}_search_{digest}"
    recorder(context.run_id, label, rows)
    return label, _count_working_set(context, label, len(rows))


def _search_result(
    *,
    context: "AgentExecutionContext",
    query: str,
    retrieval_mode: str,
    retrieval_strategy: str,
    rows: list[dict[str, Any]],
    caveats: list[str],
) -> ToolExecutionResult:
    preview_limit = _result_preview_limit()
    preview_rows = list(rows[:preview_limit])
    label, materialized_count = _materialize_result_working_set(
        context,
        query=query,
        retrieval_mode=retrieval_mode,
        rows=rows,
    )
    result_count = materialized_count or len(rows)
    payload = {
        "results": preview_rows if label else rows,
        "query": query,
        "retrieval_mode": retrieval_mode,
        "retrieval_strategy": retrieval_strategy,
        "result_count": result_count,
        "document_count": result_count,
    }
    if label:
        payload.update(
            {
                "working_set_ref": label,
                "preview_count": len(preview_rows),
                "results_truncated": result_count > len(preview_rows),
            }
        )
        if result_count > len(preview_rows):
            caveats = list(caveats) + [
                f"Full retrieval population was materialized as working_set_ref='{label}'; only {len(preview_rows)} preview rows are shown in JSON/UI."
            ]
    return ToolExecutionResult(
        payload=payload,
        evidence=preview_rows if label else list(rows),
        caveats=caveats,
        metadata={
            "working_set_ref": label,
            "full_result_count": result_count,
            "payload_truncated": bool(label and result_count > len(preview_rows)),
        },
    )


def _min_exhaustive_anchor_hits(anchors: list[str]) -> int:
    count = len([anchor for anchor in anchors if str(anchor).strip()])
    if count <= 1:
        return count
    if count == 2:
        return 2
    return max(2, (count * 2 + 2) // 3)


def _min_required_anchor_hits(query: str, anchors: list[str], *, top_k: int) -> int:
    if top_k > 0:
        return 1
    or_groups = _query_or_groups(query)
    raw_or_groups = _split_top_level_boolean(
        _strip_wrapping_parentheses(_query_without_field_filters(str(query or "")).strip()),
        "OR",
    )
    if len(or_groups) > 1 and not any(re.search(r"\bAND\b", group, flags=re.IGNORECASE) for group in raw_or_groups):
        return 1
    return _min_exhaustive_anchor_hits(anchors)


def _anchor_hit_count(text: str, anchors: list[str]) -> int:
    haystack = str(text or "").lower()
    return sum(1 for anchor in anchors if re.search(rf"\b{re.escape(str(anchor).lower())}\b", haystack))


def _normalized_pos_label(value: Any) -> str:
    label = str(value or "").strip().upper()
    if not label:
        return ""
    if label in {"NN", "NNS"}:
        return "NOUN"
    if label in {"NNP", "NNPS"}:
        return "PROPN"
    return label


def _noun_frequency_rows(
    documents: list[dict[str, Any]],
    pos_rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if not pos_rows:
        return []
    doc_counts: defaultdict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for row in pos_rows:
        label = _normalized_pos_label(row.get("pos"))
        if label not in {"NOUN", "PROPN"}:
            continue
        lemma = str(row.get("lemma", row.get("token", ""))).strip().lower()
        if not _valid_noun_lemma(lemma, min_length=2):
            continue
        doc_id = str(row.get("doc_id", "")).strip()
        counts[lemma] += 1
        if doc_id:
            doc_counts[lemma].add(doc_id)
    total = sum(counts.values())
    rows: list[dict[str, Any]] = []
    for rank, (lemma, count) in enumerate(counts.most_common(max(1, top_k)), start=1):
        rows.append(
            {
                "lemma": lemma,
                "count": int(count),
                "relative_frequency": round(count / max(total, 1), 6),
                "document_frequency": len(doc_counts.get(lemma, set())),
                "rank": rank,
            }
        )
    return rows


def _noun_frequency_rows_from_working_set(
    context: "AgentExecutionContext",
    working_set_ref: str,
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    working_set_count = _count_working_set(context, working_set_ref, 0)
    spacy_limit = _noun_spacy_max_documents()
    prefer_spacy = spacy_limit is None or working_set_count <= spacy_limit
    analysis_max_documents = _working_set_analysis_max_documents()
    if analysis_max_documents is not None and working_set_count > analysis_max_documents:
        prefer_spacy = False
    if _env_flag("CORPUSAGENT2_USE_SQL_TOKEN_AGGREGATE", False):
        sql_max_documents = analysis_max_documents if _working_set_analysis_limit_explicit() else None
        sql_result = _sql_noun_frequency_rows_from_working_set(
            context,
            working_set_ref,
            top_k=top_k,
            max_documents=sql_max_documents,
        )
        if sql_result is not None:
            rows, metadata = sql_result
            metadata["working_set_document_count"] = working_set_count
            metadata["provider_fallback_reason"] = "CORPUSAGENT2_USE_SQL_TOKEN_AGGREGATE=true"
            return rows, metadata
    rows, metadata = _noun_frequency_rows_from_documents(
        _iter_working_set_documents(context, working_set_ref, max_documents=analysis_max_documents),
        top_k=top_k,
        full_working_set=True,
        working_set_ref=working_set_ref,
        prefer_spacy=prefer_spacy,
    )
    metadata["working_set_document_count"] = working_set_count
    metadata["analysis_document_limit"] = analysis_max_documents
    if not prefer_spacy:
        metadata["provider_fallback_reason"] = (
            f"working_set_document_count {working_set_count} exceeds CORPUSAGENT2_NOUN_SPACY_MAX_DOCS={spacy_limit}"
        )
    return rows, metadata


def _noun_frequency_rows_from_documents(
    documents: Iterable[dict[str, Any]],
    *,
    top_k: int,
    full_working_set: bool = False,
    working_set_ref: str = "",
    prefer_spacy: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lemma_counts: Counter[str] = Counter()
    doc_counts: defaultdict[str, set[str]] = defaultdict(set)
    analyzed_documents = 0
    total_tokens = 0
    machine_payload_documents = 0
    truncated_documents = 0
    provider = "heuristic_batch"
    nlp = _load_spacy_model() if prefer_spacy else None
    use_spacy = bool(prefer_spacy and nlp is not None and _spacy_supports_pos(nlp))
    batch_size_raw = os.getenv("CORPUSAGENT2_NOUN_BATCH_SIZE", "64").strip()
    try:
        batch_size = max(1, int(batch_size_raw))
    except ValueError:
        batch_size = 64

    def prepared_documents() -> Iterable[tuple[str, str]]:
        nonlocal analyzed_documents, machine_payload_documents, truncated_documents
        for document in documents:
            doc_id = str(document.get("doc_id", "")).strip()
            if not doc_id:
                continue
            text, flags = _clean_analysis_text(document.get("title", ""), document.get("text", document.get("cleaned_text", "")))
            if document.get("text_is_machine_payload"):
                flags["machine_payload"] = True
            if not text:
                continue
            analyzed_documents += 1
            if flags.get("machine_payload"):
                machine_payload_documents += 1
            if flags.get("truncated"):
                truncated_documents += 1
            yield doc_id, text

    if use_spacy:
        provider = "spacy_batch"
        disabled = _spacy_noun_pipe_disabled(nlp)
        context_manager = nlp.select_pipes(disable=disabled) if disabled else None
        try:
            if context_manager is not None:
                context_manager.__enter__()
            for batch in _iter_batches(prepared_documents(), batch_size):
                doc_ids = [item[0] for item in batch]
                texts = [item[1] for item in batch]
                for doc_id, doc in zip(doc_ids, nlp.pipe(texts, batch_size=batch_size), strict=False):
                    for token in doc:
                        label = _normalized_pos_label(getattr(token, "pos_", ""))
                        if label not in {"NOUN", "PROPN"}:
                            continue
                        lemma = str(getattr(token, "lemma_", "") or getattr(token, "text", "") or "").strip().lower()
                        if lemma == "-pron-":
                            lemma = str(getattr(token, "text", "") or "").strip().lower()
                        if not _valid_noun_lemma(lemma, min_length=3):
                            continue
                        lemma_counts[lemma] += 1
                        doc_counts[lemma].add(doc_id)
                        total_tokens += 1
        finally:
            if context_manager is not None:
                context_manager.__exit__(None, None, None)
    else:
        for doc_id, text in prepared_documents():
            for token in _tokenize(text):
                pos = _heuristic_pos(token)
                if pos not in {"NOUN", "PROPN"}:
                    continue
                lemma = _lemma(token)
                if not _valid_noun_lemma(lemma, min_length=3):
                    continue
                lemma_counts[lemma] += 1
                doc_counts[lemma].add(doc_id)
                total_tokens += 1
    rows: list[dict[str, Any]] = []
    for rank, (lemma, count) in enumerate(lemma_counts.most_common(top_k), start=1):
        rows.append(
            {
                "lemma": lemma,
                "count": int(count),
                "relative_frequency": round(count / max(total_tokens, 1), 6),
                "document_frequency": len(doc_counts[lemma]),
                "rank": rank,
            }
        )
    return rows, {
        "analyzed_document_count": analyzed_documents,
        "total_noun_tokens": total_tokens,
        "full_working_set": bool(full_working_set),
        "working_set_ref": working_set_ref,
        "provider": provider,
        "machine_payload_document_count": machine_payload_documents,
        "analysis_text_truncated_document_count": truncated_documents,
    }


def _summary_stat_rows(
    documents: list[dict[str, Any]],
    upstream_rows: list[dict[str, Any]],
    *,
    matched_document_count: int | None = None,
) -> list[dict[str, Any]]:
    total_noun_tokens = sum(int(row.get("count", 0) or 0) for row in upstream_rows)
    top_nouns = ", ".join(
        f"{row.get('lemma', '')} ({int(row.get('count', 0) or 0)})"
        for row in upstream_rows[:10]
        if str(row.get("lemma", "")).strip()
    )
    return [
        {"metric": "matched_document_count", "value": matched_document_count if matched_document_count is not None else len(documents)},
        {"metric": "total_noun_tokens", "value": total_noun_tokens},
        {"metric": "unique_noun_lemmas", "value": len(upstream_rows)},
        {"metric": "top_nouns", "value": top_nouns},
    ]


def _rows_match_query_anchor_terms(rows: list[dict[str, Any]], query: str) -> bool:
    anchors = _query_anchor_terms(query)
    if not anchors:
        return True
    haystack = " ".join(
        f"{str(row.get('title', ''))} {str(row.get('snippet', ''))} {str(row.get('outlet', ''))}"
        for row in rows
    ).lower()
    if not haystack:
        return False
    return any(anchor.lower() in haystack for anchor in anchors)


def _parse_year(value: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", str(value or ""))
    if not match:
        return None
    return int(match.group(0))


def _year_range(date_from: str, date_to: str) -> list[int]:
    start_year = _parse_year(date_from)
    end_year = _parse_year(date_to)
    if start_year is None or end_year is None or end_year < start_year:
        return []
    if end_year - start_year > 8:
        return []
    return list(range(start_year, end_year + 1))


def _normalize_duplicate_text(text: str) -> str:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    ]
    return " ".join(tokens[:20])


def _token_set(text: str) -> set[str]:
    return set(_normalize_duplicate_text(text).split())


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _row_year(row: dict[str, Any]) -> int | None:
    return _parse_year(str(row.get("date", row.get("published_at", ""))))


def _rows_are_near_duplicates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if str(left.get("doc_id", "")) == str(right.get("doc_id", "")):
        return True
    left_date = str(left.get("date", ""))[:10]
    right_date = str(right.get("date", ""))[:10]
    left_title = _token_set(str(left.get("title", "")))
    right_title = _token_set(str(right.get("title", "")))
    left_snippet = _token_set(str(left.get("snippet", "")))
    right_snippet = _token_set(str(right.get("snippet", "")))
    title_overlap = _jaccard(left_title, right_title)
    snippet_overlap = _jaccard(left_snippet, right_snippet)
    if title_overlap >= 0.9:
        return True
    if left_date and right_date and left_date == right_date and title_overlap >= 0.72 and snippet_overlap >= 0.45:
        return True
    return False


def _duplicate_candidate_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    doc_id = str(row.get("doc_id", "")).strip()
    if doc_id:
        keys.append(f"doc:{doc_id}")
    date_key = str(row.get("date", row.get("published_at", "")))[:10]
    title_norm = _normalize_duplicate_text(str(row.get("title", "")))
    snippet_norm = _normalize_duplicate_text(str(row.get("snippet", "")))
    if title_norm:
        keys.append(f"title:{title_norm}")
        if date_key:
            keys.append(f"date-title:{date_key}:{title_norm}")
    if snippet_norm:
        snippet_key = " ".join(snippet_norm.split()[:12])
        keys.append(f"snippet:{snippet_key}")
        if date_key:
            keys.append(f"date-snippet:{date_key}:{snippet_key}")
    if title_norm and snippet_norm and date_key:
        keys.append(f"date-combo:{date_key}:{title_norm}:{' '.join(snippet_norm.split()[:8])}")
    return keys


def _display_snippet(title: Any, snippet: Any) -> tuple[str, dict[str, bool]]:
    cleaned, flags = _clean_analysis_text(title, snippet)
    if not cleaned:
        cleaned = _collapse_text(title)
    return textwrap.shorten(cleaned, width=360, placeholder="..."), flags


def _retrieval_quality_multiplier(title: Any, snippet: Any) -> tuple[float, dict[str, bool]]:
    _, flags = _display_snippet(title, snippet)
    if flags.get("machine_payload"):
        return 0.35, flags
    return 1.0, flags


def _normalize_result_rows(rows: list[dict[str, Any]], retrieval_mode: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    prepared: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        raw_snippet = copy.get("snippet", copy.get("text", ""))
        multiplier = 1.0
        multiplier_flags: dict[str, bool] = {}
        if not bool(copy.get("score_quality_adjusted")):
            multiplier, multiplier_flags = _retrieval_quality_multiplier(copy.get("title", ""), raw_snippet)
        if "snippet" in copy or "text" in copy:
            snippet, flags = _display_snippet(copy.get("title", ""), raw_snippet)
            copy["snippet"] = snippet
            if flags.get("machine_payload") or multiplier_flags.get("machine_payload"):
                copy["snippet_is_machine_payload"] = True
        score = _coerce_score(copy.get("score", 0.0))
        if not bool(copy.get("score_quality_adjusted")):
            if multiplier != 1.0:
                copy["raw_score"] = score
                copy["score"] = score * multiplier
                copy["score_quality_adjusted"] = True
                copy["score_quality_multiplier"] = multiplier
                if multiplier_flags.get("machine_payload"):
                    copy["snippet_is_machine_payload"] = True
            else:
                copy["score"] = score
        else:
            copy["score"] = score
        prepared.append(copy)
    ordered = sorted(prepared, key=lambda item: _coerce_score(item.get("score", 0.0)), reverse=True)
    max_score = max((_coerce_score(item.get("score", 0.0)) for item in ordered), default=0.0)
    normalized: list[dict[str, Any]] = []
    for rank, row in enumerate(ordered, start=1):
        copy = dict(row)
        score = _coerce_score(copy.get("score", 0.0))
        copy["score"] = score
        copy["rank"] = rank
        copy["retrieval_mode"] = str(copy.get("retrieval_mode", retrieval_mode) or retrieval_mode)
        copy["score_display"] = _score_display(score / max_score if max_score > 0 else score)
        score_components = copy.get("score_components", {})
        if not isinstance(score_components, dict) or not score_components:
            score_components = {copy["retrieval_mode"]: round(score, 6)}
        copy["score_components"] = {str(key): round(_coerce_score(value), 6) for key, value in score_components.items()}
        normalized.append(copy)
    return normalized


def _dedupe_wire_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if not rows:
        return [], 0
    ordered = sorted(rows, key=lambda item: _coerce_score(item.get("score", 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    buckets: defaultdict[str, list[int]] = defaultdict(list)
    duplicates_removed = 0
    for row in ordered:
        duplicate_of: dict[str, Any] | None = None
        candidate_indexes: list[int] = []
        seen_indexes: set[int] = set()
        row_keys = _duplicate_candidate_keys(row)
        for key in row_keys:
            for idx in buckets.get(key, []):
                if idx in seen_indexes:
                    continue
                seen_indexes.add(idx)
                candidate_indexes.append(idx)
        for idx in candidate_indexes:
            candidate = kept[idx]
            if _rows_are_near_duplicates(row, candidate):
                duplicate_of = candidate
                break
        if duplicate_of is None:
            copy = dict(row)
            copy["duplicate_cluster_size"] = 1
            kept.append(copy)
            kept_index = len(kept) - 1
            for key in row_keys:
                buckets[key].append(kept_index)
            continue
        duplicate_of["duplicate_cluster_size"] = int(duplicate_of.get("duplicate_cluster_size", 1)) + 1
        duplicates_removed += 1
    return kept, duplicates_removed


def _round_robin_year_balance(rows: list[dict[str, Any]], years: list[int], top_k: int) -> list[dict[str, Any]]:
    if not rows or not years:
        return rows[:top_k] if top_k > 0 else list(rows)
    if top_k <= 0:
        return list(rows)
    buckets: dict[int, list[dict[str, Any]]] = {year: [] for year in years}
    leftovers: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: _coerce_score(item.get("score", 0.0)), reverse=True):
        year = _row_year(row)
        if year in buckets:
            buckets[year].append(row)
        else:
            leftovers.append(row)
    balanced: list[dict[str, Any]] = []
    while len(balanced) < top_k:
        progressed = False
        for year in years:
            bucket = buckets[year]
            if bucket:
                balanced.append(bucket.pop(0))
                progressed = True
                if len(balanced) >= top_k:
                    break
        if not progressed:
            break
    if len(balanced) < top_k:
        remaining = []
        for year in years:
            remaining.extend(buckets[year])
        remaining.extend(leftovers)
        remaining.sort(key=lambda item: _coerce_score(item.get("score", 0.0)), reverse=True)
        for row in remaining:
            if len(balanced) >= top_k:
                break
            balanced.append(row)
    return balanced[:top_k]


def _prepare_result_rows(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    retrieval_mode: str,
    years: list[int] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    deduped, duplicates_removed = _dedupe_wire_rows(rows)
    if top_k <= 0:
        return _normalize_result_rows(deduped, retrieval_mode), duplicates_removed
    selected = _round_robin_year_balance(deduped, list(years or []), top_k) if years else deduped[:top_k]
    return _normalize_result_rows(selected[:top_k], retrieval_mode), duplicates_removed

_SPACY_NLP = None
_STANZA_PIPELINES: dict[tuple[str, str], Any] = {}
_FLAIR_OBJECTS: dict[str, Any] = {}
_YFINANCE_SERIES_CACHE: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}


def _sql_fallback_store(context: "AgentExecutionContext") -> PostgresWorkingSetStore | None:
    if isinstance(context.working_store, PostgresWorkingSetStore):
        return context.working_store
    dsn = pg_dsn_from_env(required=False)
    if not dsn:
        return None
    table = os.getenv("CORPUSAGENT2_PG_TABLE", "article_corpus").strip() or "article_corpus"
    return PostgresWorkingSetStore(dsn=dsn, documents_table=table)


def _queryable_sql_store(context: "AgentExecutionContext") -> tuple[PostgresWorkingSetStore | None, str]:
    store = _sql_fallback_store(context)
    if store is None:
        return None, "Postgres corpus store is not configured."
    try:
        store._document_columns()
    except Exception as exc:
        return None, str(exc)
    return store, ""


def _sql_date_filters(date_column: str, date_from: str, date_to: str) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if not date_column:
        return clauses, params
    if date_column == "year":
        if date_from:
            clauses.append(f"CAST({date_column} AS INT) >= %s")
            params.append(int(str(date_from)[:4]))
        if date_to:
            clauses.append(f"CAST({date_column} AS INT) <= %s")
            params.append(int(str(date_to)[:4]))
        return clauses, params
    if date_from:
        clauses.append(f"LEFT(COALESCE({date_column}::text, ''), 10) >= %s")
        params.append(str(date_from)[:10])
    if date_to:
        clauses.append(f"LEFT(COALESCE({date_column}::text, ''), 10) <= %s")
        params.append(str(date_to)[:10])
    return clauses, params


def _sql_search_rows(
    *,
    query: str,
    top_k: int,
    date_from: str,
    date_to: str,
    context: "AgentExecutionContext",
) -> list[dict[str, Any]]:
    store = _sql_fallback_store(context)
    if store is None:
        return []
    tokens = _query_anchor_terms(query)
    if not tokens:
        return []
    columns = store._document_columns()
    table_name = store._safe_identifier(store.documents_table)
    title_expr = f"COALESCE({columns['title']}::text, '')" if columns["title"] else "''"
    text_expr = f"COALESCE({columns['text']}::text, '')"
    date_expr = f"COALESCE({columns['date']}::text, '')" if columns["date"] else "''"
    source_expr = f"COALESCE({columns['source']}::text, '')" if columns["source"] else "''"
    source_filters = _query_source_filters(query)
    source_clauses = [
        f"regexp_replace(LOWER({source_expr}), '[^a-z0-9]+', '', 'g') LIKE %s"
        for _ in source_filters
    ]
    source_params = [f"%{item}%" for item in source_filters]
    vector_expr = (
        f"setweight(to_tsvector('simple', {title_expr}), 'A') || "
        f"setweight(to_tsvector('simple', {text_expr}), 'B')"
    )
    date_clauses, date_params = _sql_date_filters(columns["date"], date_from, date_to)
    query_clauses = _sql_websearch_query_clauses(query, tokens)
    if not query_clauses:
        return []
    rank_expr = " + ".join(
        f"ts_rank_cd({vector_expr}, websearch_to_tsquery('simple', %s))"
        for _ in query_clauses
    )
    match_expr = " AND ".join(
        f"{vector_expr} @@ websearch_to_tsquery('simple', %s)"
        for _ in query_clauses
    )
    sql = (
        f"SELECT "
        f"{columns['doc_id']}::text AS doc_id, "
        f"{title_expr} AS title, "
        f"SUBSTRING({text_expr} FROM 1 FOR 360) AS snippet, "
        f"{source_expr} AS outlet, "
        f"{date_expr} AS date, "
        f"({rank_expr}) AS score "
        f"FROM {table_name} "
        f"WHERE {match_expr}"
    )
    if date_clauses:
        sql += f" AND {' AND '.join(date_clauses)}"
    if source_clauses:
        sql += f" AND ({' OR '.join(source_clauses)})"
    sql += " ORDER BY score DESC"
    params: list[Any] = [*query_clauses, *query_clauses, *date_params, *source_params]
    if top_k > 0:
        sql += " LIMIT %s"
        params.append(int(top_k))

    def _run_relaxed_anchor_search() -> list[Any]:
        hit_parts: list[str] = []
        like_score_parts: list[str] = []
        score_params: list[Any] = []
        hit_params: list[Any] = []
        for token in tokens:
            needle = f"%{token.lower()}%"
            title_match = f"LOWER({title_expr}) LIKE %s"
            text_match = f"LOWER({text_expr}) LIKE %s"
            hit_parts.append(f"(CASE WHEN ({title_match} OR {text_match}) THEN 1 ELSE 0 END)")
            like_score_parts.append(
                f"(CASE WHEN {title_match} THEN 2 ELSE 0 END + CASE WHEN {text_match} THEN 1 ELSE 0 END)"
            )
            score_params.extend([needle, needle])
            hit_params.extend([needle, needle])
        min_hits = _min_required_anchor_hits(query, tokens, top_k=top_k)
        hit_expr = " + ".join(hit_parts)
        fallback_sql = (
            f"SELECT "
            f"{columns['doc_id']}::text AS doc_id, "
            f"{title_expr} AS title, "
            f"SUBSTRING({text_expr} FROM 1 FOR 360) AS snippet, "
            f"{source_expr} AS outlet, "
            f"{date_expr} AS date, "
            f"({' + '.join(like_score_parts)})::float AS score "
            f"FROM {table_name} "
            f"WHERE ({hit_expr}) >= %s"
        )
        if date_clauses:
            fallback_sql += f" AND {' AND '.join(date_clauses)}"
        if source_clauses:
            fallback_sql += f" AND ({' OR '.join(source_clauses)})"
        fallback_sql += " ORDER BY score DESC"
        fallback_params: list[Any] = score_params + hit_params + [min_hits] + date_params + source_params
        if top_k > 0:
            fallback_sql += " LIMIT %s"
            fallback_params.append(int(top_k))
        with store._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(fallback_sql, tuple(fallback_params))
                return cursor.fetchall()

    try:
        with store._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()
    except Exception:
        rows = _run_relaxed_anchor_search()
    if not rows:
        rows = _run_relaxed_anchor_search()
    if not rows:
        return []
    max_score = max((_coerce_score(row[5]) for row in rows), default=0.0)
    normalized: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        score = _coerce_score(row[5])
        normalized.append(
            {
                "doc_id": str(row[0] or ""),
                "title": str(row[1] or ""),
                "snippet": str(row[2] or ""),
                "outlet": str(row[3] or ""),
                "date": str(row[4] or ""),
                "score": score,
                "score_display": _score_display(score / max_score if max_score > 0 else score),
                "rank": rank,
                "retrieval_mode": "sql",
                "score_components": {"sql": round(score, 6)},
            }
        )
    return normalized


def _local_exhaustive_rows(
    *,
    query: str,
    top_k: int,
    date_from: str,
    date_to: str,
    context: "AgentExecutionContext",
) -> list[dict[str, Any]]:
    if context.runtime is None:
        return []
    anchors = [anchor.lower() for anchor in _query_anchor_terms(query)]
    if not anchors:
        return []
    min_hits = _min_required_anchor_hits(query, anchors, top_k=top_k)
    source_filters = _query_source_filters(query)
    def _metadata_scan_rows() -> list[dict[str, Any]]:
        try:
            metadata = context.runtime.load_metadata()
        except Exception:
            return []
        if metadata.empty:
            return []
        date_from_key = str(date_from or "")[:10]
        date_to_key = str(date_to or "")[:10]
        rows: list[dict[str, Any]] = []
        for row in metadata.itertuples(index=False):
            published_at = str(getattr(row, "published_at", "") or "")
            published_key = published_at[:10]
            if date_from_key and (not published_key or published_key < date_from_key):
                continue
            if date_to_key and (not published_key or published_key > date_to_key):
                continue
            title = str(getattr(row, "title", "") or "")
            text = str(getattr(row, "text", "") or "")
            source = str(getattr(row, "source", "") or "")
            if not _row_matches_source_filters(source, source_filters):
                continue
            title_lower = title.lower()
            text_lower = text.lower()
            source_lower = source.lower()
            title_hits = _anchor_hit_count(title_lower, anchors)
            text_hits = _anchor_hit_count(text_lower, anchors)
            source_hits = _anchor_hit_count(source_lower, anchors)
            if _anchor_hit_count(f"{title} {text} {source}", anchors) < min_hits:
                continue
            score = float((title_hits * 4.0) + (text_hits * 1.5) + (source_hits * 1.0))
            rows.append(
                {
                    "doc_id": str(getattr(row, "doc_id", "") or ""),
                    "title": title,
                    "snippet": text[:360],
                    "outlet": source,
                    "date": published_at,
                    "score": score,
                    "retrieval_mode": "local_exhaustive",
                    "score_components": {"local_exhaustive": round(score, 6)},
                }
            )
        rows.sort(key=lambda item: (item["score"], str(item.get("doc_id", ""))), reverse=True)
        return rows
    try:
        lexical_vectorizer, lexical_matrix, lexical_doc_ids = context.runtime.load_lexical_assets()
        query_text = " ".join(anchors) if anchors else str(query or "")
        retrieval_results = retrieve_tfidf(
            query=query_text,
            vectorizer=lexical_vectorizer,
            matrix=lexical_matrix,
            doc_ids=lexical_doc_ids,
            top_k=len(lexical_doc_ids),
        )
        if not retrieval_results:
            return _metadata_scan_rows()
        score_by_id = {result.doc_id: float(result.score) for result in retrieval_results}
        metadata = context.runtime.load_docs(list(score_by_id.keys()))
    except Exception:
        return _metadata_scan_rows()
    if metadata.empty:
        return []
    date_from_key = str(date_from or "")[:10]
    date_to_key = str(date_to or "")[:10]
    rows: list[dict[str, Any]] = []
    for row in metadata.itertuples(index=False):
        doc_id = str(getattr(row, "doc_id", "") or "")
        published_at = str(getattr(row, "published_at", "") or "")
        published_key = published_at[:10]
        if date_from_key and (not published_key or published_key < date_from_key):
            continue
        if date_to_key and (not published_key or published_key > date_to_key):
            continue
        score = float(score_by_id.get(doc_id, 0.0))
        if score <= 0.0:
            continue
        combined = f"{getattr(row, 'title', '')} {getattr(row, 'text', '')} {getattr(row, 'source', '')}"
        if not _row_matches_source_filters(str(getattr(row, "source", "") or ""), source_filters):
            continue
        if _anchor_hit_count(combined, anchors) < min_hits:
            continue
        rows.append(
            {
                "doc_id": doc_id,
                "title": str(getattr(row, "title", "") or ""),
                "snippet": str(getattr(row, "text", "") or "")[:360],
                "outlet": str(getattr(row, "source", "") or ""),
                "date": published_at,
                "score": score,
                "retrieval_mode": "local_exhaustive",
                "score_components": {"local_exhaustive": round(score, 6)},
            }
        )
    rows.sort(key=lambda item: (item["score"], str(item.get("doc_id", ""))), reverse=True)
    return rows


def _sandbox_candidate_rows(
    *,
    query: str,
    top_k: int,
    date_from: str,
    date_to: str,
    context: "AgentExecutionContext",
) -> list[dict[str, Any]]:
    anchors = _query_anchor_terms(query)
    if not anchors or context.runtime is None:
        return []
    metadata = context.runtime.load_metadata()
    if metadata.empty:
        return []
    frame = metadata.copy()
    if "published_at" in frame.columns:
        published = frame["published_at"].astype(str)
        if date_from:
            frame = frame[published.str.slice(0, 10) >= str(date_from)[:10]]
        if date_to:
            frame = frame[published.str.slice(0, 10) <= str(date_to)[:10]]
    if frame.empty:
        return []
    source_filters = _query_source_filters(query)
    if source_filters and "source" in frame.columns:
        normalized_sources = frame["source"].fillna("").astype(str).map(_normalize_source_filter)
        mask = pd.Series(False, index=frame.index)
        for source_filter in source_filters:
            mask = mask | normalized_sources.str.contains(re.escape(source_filter), regex=True, na=False)
        frame = frame[mask]
    if frame.empty:
        return []
    escaped = "|".join(re.escape(term) for term in anchors)
    if not escaped:
        return []
    haystack = (
        frame.get("title", pd.Series("", index=frame.index)).fillna("").astype(str)
        + " "
        + frame.get("text", pd.Series("", index=frame.index)).fillna("").astype(str)
        + " "
        + frame.get("source", pd.Series("", index=frame.index)).fillna("").astype(str)
    )
    matches = frame[haystack.str.contains(escaped, case=False, regex=True, na=False)].head(max(200, top_k * 15))
    candidates: list[dict[str, Any]] = []
    for row in matches.itertuples(index=False):
        candidates.append(
            {
                "doc_id": str(getattr(row, "doc_id", "")),
                "title": str(getattr(row, "title", "")),
                "text": str(getattr(row, "text", ""))[:2000],
                "outlet": str(getattr(row, "source", "")),
                "date": str(getattr(row, "published_at", "")),
            }
        )
    return candidates


def _sandbox_retrieval_rows(
    *,
    query: str,
    top_k: int,
    date_from: str,
    date_to: str,
    context: "AgentExecutionContext",
) -> list[dict[str, Any]]:
    if context.python_runner is None:
        return []
    candidates = _sandbox_candidate_rows(
        query=query,
        top_k=top_k,
        date_from=date_from,
        date_to=date_to,
        context=context,
    )
    if not candidates:
        return []
    code = """
from pathlib import Path
import json
import re

def tokens(text):
    return [token for token in re.findall(r"[A-Za-z0-9]+", str(text).lower()) if len(token) > 2]

payload = INPUTS_JSON
query = str(payload.get("query", ""))
top_k = int(payload.get("top_k", 20))
anchors = [token for token in tokens(query) if token not in {"how", "did", "what", "which", "from", "into", "with", "between", "across"}]
bigram_terms = {" ".join(pair) for pair in zip(anchors, anchors[1:])}
rows = []
for candidate in payload.get("candidates", []):
    title = str(candidate.get("title", ""))
    text = str(candidate.get("text", ""))
    outlet = str(candidate.get("outlet", ""))
    combined = f"{title} {text} {outlet}".lower()
    title_tokens = set(tokens(title))
    body_tokens = set(tokens(text))
    overlap = sum(2 for token in anchors if token in title_tokens) + sum(1 for token in anchors if token in body_tokens)
    phrase_bonus = sum(3 for phrase in bigram_terms if phrase and phrase in combined)
    exact_bonus = sum(2 for token in anchors if token and token in combined)
    score = float(overlap + phrase_bonus + exact_bonus)
    if score <= 0:
        continue
    rows.append(
        {
            "doc_id": str(candidate.get("doc_id", "")),
            "title": title,
            "snippet": text[:360],
            "outlet": outlet,
            "date": str(candidate.get("date", "")),
            "score": score,
            "retrieval_mode": "sandbox",
            "score_components": {"sandbox": score},
        }
    )
rows.sort(key=lambda item: item["score"], reverse=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR, "sandbox_retrieval.json").write_text(json.dumps({"results": rows[:top_k]}), encoding="utf-8")
print(json.dumps({"count": len(rows[:top_k])}))
""".strip()
    result = context.python_runner.run(
        code=code,
        inputs_json={"query": query, "top_k": top_k, "candidates": candidates},
    )
    if result.exit_code != 0:
        return []
    for artifact in result.artifacts:
        if artifact.name != "sandbox_retrieval.json":
            continue
        try:
            payload = json.loads(base64.b64decode(artifact.bytes_b64.encode("ascii")).decode("utf-8"))
        except Exception:
            return []
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            return []
        return _normalize_result_rows([dict(row) for row in rows], "sandbox")
    return []


@dataclass(slots=True)
class AgentExecutionContext:
    run_id: str
    artifacts_dir: Path
    search_backend: SearchBackend
    working_store: WorkingSetStore
    llm_client: Any | None = None
    python_runner: DockerPythonRunnerService | None = None
    runtime: Any | None = None
    state: Any | None = None
    event_callback: Callable[[dict[str, Any]], None] | None = None
    cancel_requested: Callable[[], bool] | None = None


class FunctionalToolAdapter(CapabilityToolAdapter):
    def __init__(
        self,
        *,
        tool_name: str,
        capability: str,
        provider: str,
        priority: int,
        run_fn: Callable[[dict[str, Any], dict[str, ToolExecutionResult], AgentExecutionContext], ToolExecutionResult],
        deterministic: bool = True,
        cost_class: str = "low",
        fallback_of: str | None = None,
    ) -> None:
        self._run_fn = run_fn
        self.spec = ToolSpec(
            tool_name=tool_name,
            provider=provider,
            capabilities=[capability],
            input_schema=SchemaDescriptor(name=f"{tool_name}_input", fields={"payload": "dict"}),
            output_schema=SchemaDescriptor(name=f"{tool_name}_output", fields={"payload": "dict"}),
            deterministic=deterministic,
            cost_class=cost_class,
            priority=priority,
            fallback_of=fallback_of,
        )

    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        return True, ["registered"]

    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        return self._run_fn(_payload_or_params(params), dependency_results, context)


def _load_spacy_model():
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    try:
        import spacy
    except Exception:
        _SPACY_NLP = False
        return None
    for name in ("en_core_web_sm", "en_core_web_lg"):
        try:
            _SPACY_NLP = spacy.load(name)
            return _SPACY_NLP
        except Exception:
            continue
    try:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _SPACY_NLP = nlp
        return nlp
    except Exception:
        _SPACY_NLP = False
        return None


def _tokenize(text: str) -> list[str]:
    return [token.group(0) for token in TOKEN_PATTERN.finditer(text)]


def _lemma(token: str) -> str:
    lowered = token.lower()
    if lowered.endswith("ies") and len(lowered) > 4:
        return lowered[:-3] + "y"
    if lowered.endswith("s") and len(lowered) > 3:
        return lowered[:-1]
    return lowered


def _heuristic_pos(token: str) -> str:
    lowered = token.lower()
    if lowered in NOUN_LEMMA_STOPWORDS:
        return "STOP"
    if lowered.endswith("ly"):
        return "ADV"
    if lowered.endswith(("ing", "ed")):
        return "VERB"
    if lowered.endswith(("ous", "ive", "al", "ful")):
        return "ADJ"
    if token[:1].isupper():
        return "PROPN"
    return "NOUN"


def _spacy_supports_pos(nlp: Any) -> bool:
    pipes = set(getattr(nlp, "pipe_names", []) or [])
    return bool(pipes.intersection({"tagger", "morphologizer"}))


def _spacy_noun_pipe_disabled(nlp: Any) -> list[str]:
    needed = {"tok2vec", "transformer", "tagger", "morphologizer", "attribute_ruler", "lemmatizer"}
    return [pipe for pipe in getattr(nlp, "pipe_names", []) if pipe not in needed]


def _iter_batches(items: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _doc_rows(dependency_results: dict[str, ToolExecutionResult]) -> list[dict[str, Any]]:
    for result in dependency_results.values():
        payload = result.payload
        if isinstance(payload, dict):
            if "documents" in payload:
                return [_coerce_text_document_row(dict(item)) for item in payload["documents"] if isinstance(item, dict)]
            if "results" in payload and payload["results"] and "text" in payload["results"][0]:
                return [_coerce_text_document_row(dict(item)) for item in payload["results"] if isinstance(item, dict)]
            if "rows" in payload and isinstance(payload["rows"], list) and payload["rows"]:
                text_rows = [
                    _coerce_text_document_row(dict(item))
                    for item in payload["rows"]
                    if isinstance(item, dict) and _row_has_text_payload(item)
                ]
                if text_rows:
                    return text_rows
    return []


def _search_rows(dependency_results: dict[str, ToolExecutionResult]) -> list[dict[str, Any]]:
    for result in dependency_results.values():
        payload = result.payload
        if isinstance(payload, dict) and "results" in payload:
            return list(payload["results"])
    return []


def _coerce_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not pd.notna(numeric):
        return 0.0
    return numeric


def _score_display(value: Any) -> str:
    numeric = _coerce_score(value)
    absolute = abs(numeric)
    if absolute >= 1:
        return f"{numeric:.3f}".rstrip("0").rstrip(".")
    if absolute >= 0.01:
        return f"{numeric:.4f}".rstrip("0").rstrip(".")
    if absolute > 0:
        return f"{numeric:.2e}"
    return "0"


def _time_bin(value: str, granularity: str | None = None) -> str:
    text = str(value).strip()
    if _is_placeholder_axis_label(text):
        return "unknown"
    mode = str(granularity or _default_time_granularity()).strip().lower() or "month"
    if mode in {"quarter", "q"}:
        quarter_match = re.match(r"^(\d{4})[-_/ ]?Q([1-4])$", text, flags=re.IGNORECASE)
        if quarter_match:
            return f"{quarter_match.group(1)}-Q{quarter_match.group(2)}"
        if len(text) >= 7 and text[4] == "-":
            try:
                month = int(text[5:7])
            except ValueError:
                month = 0
            if 1 <= month <= 12:
                return f"{text[:4]}-Q{((month - 1) // 3) + 1}"
        if re.match(r"^\d{4}", text):
            return text[:4]
        return "unknown"
    if mode in {"year", "annual", "annually"}:
        if re.match(r"^\d{4}", text):
            return text[:4]
        return "unknown"
    if mode == "day":
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            return text[:10]
        if len(text) >= 7 and text[4] == "-":
            return text[:7]
        if re.match(r"^\d{4}", text):
            return text[:4]
        return "unknown"
    if len(text) >= 7 and text[4] == "-":
        return text[:7]
    if mode == "month":
        if re.match(r"^\d{4}", text):
            return text[:4]
        return "unknown"
    if re.match(r"^\d{4}", text):
        return text[:4]
    return "unknown"


def _time_bin_fields(time_bin: str, granularity: str | None = None) -> dict[str, str]:
    rendered = str(time_bin or "unknown")
    fields = {
        "time_bin": rendered,
        "bucket": rendered,
        "month": rendered,
        "period": rendered,
        "time_period": rendered,
    }
    mode = str(granularity or _default_time_granularity()).strip().lower()
    if mode in {"quarter", "q"} or re.match(r"^\d{4}-Q[1-4]$", rendered):
        fields["quarter"] = rendered
    if mode in {"year", "annual", "annually"} or re.match(r"^\d{4}$", rendered):
        fields["year"] = rendered[:4]
    return fields


def _row_timestamp(row: dict[str, Any]) -> str:
    return str(row.get("date", row.get("published_at", "")))


def _row_analysis_text(row: dict[str, Any]) -> str:
    cleaned = str(row.get("cleaned_text", "")).strip()
    if cleaned:
        return cleaned
    text, _ = _clean_analysis_text(row.get("title", ""), row.get("text", ""))
    return text


def _infer_language(text: str) -> tuple[str, float]:
    lowered = text.lower()
    if not lowered.strip():
        return "unknown", 0.0
    counts = {
        lang: sum(1 for token in hints if f" {token} " in f" {lowered} ")
        for lang, hints in LANGUAGE_HINTS.items()
    }
    best_lang, best_count = max(counts.items(), key=lambda item: item[1])
    total = sum(counts.values())
    if best_count == 0:
        return "en", 0.2
    confidence = best_count / max(total, 1)
    return best_lang, round(confidence, 3)


def _link_entity_row(entity: str, label: str) -> dict[str, Any]:
    normalized = entity.strip()
    return {
        "entity": normalized,
        "label": label,
        "kb_id": "",
        "kb_url": "",
        "confidence": 0.25 if normalized else 0.0,
        "link_method": "string_canonicalization",
        "link_status": "unresolved",
    }


def _provider_order(capability: str, default: list[str]) -> list[str]:
    env_name = f"CORPUSAGENT2_PROVIDER_ORDER_{capability.upper()}".replace(".", "_")
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _metadata(provider: str, tool_name: str, **extra: Any) -> dict[str, Any]:
    payload = {"provider": provider, "tool_name": tool_name}
    payload.update(extra)
    return payload


def _provider_unavailable_result(capability: str, *, caveats: list[str] | None = None, **metadata: Any) -> ToolExecutionResult:
    reason = f"{capability}_provider_unavailable"
    return ToolExecutionResult(
        payload={"rows": [], **metadata},
        caveats=[
            *(caveats or []),
            (
                f"No provider-backed implementation was available for {capability}. "
                "Heuristic NLP fallbacks are disabled unless explicitly requested in provider_order."
            ),
        ],
        metadata={"no_data": True, "no_data_reason": reason, **metadata},
    )


def _no_input_documents_result(
    capability: str,
    *,
    rows_key: str = "rows",
    caveats: list[str] | None = None,
    **metadata: Any,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        payload={rows_key: [], **metadata},
        caveats=list(caveats or [f"No input documents were available for {capability}."]),
        metadata={"no_data": True, "no_data_reason": "no_input_documents", "provider": "", **metadata},
    )


def _sentence_embedding_model_id(context: AgentExecutionContext, params: dict[str, Any]) -> str:
    model_id = str(params.get("model_id", "")).strip()
    if model_id:
        return model_id
    if context.runtime is not None and getattr(context.runtime, "dense_model_id", ""):
        return str(context.runtime.dense_model_id)
    return dense_model_id_from_env()


def _encode_texts(
    texts: list[str],
    *,
    model_id: str,
    normalize: bool = True,
) -> tuple[Any, str]:
    model, resolved_device = _load_sentence_transformer(model_id=model_id, device=resolve_device("auto"))
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return embeddings, resolved_device


def _infer_market_ticker_from_text(text: str) -> str:
    lowered = str(text or "").lower()
    explicit_match = re.search(
        r"\b(?:ticker|symbol)\s*[:=]?\s*\$?([A-Z]{1,5}(?:\.[A-Z])?)\b",
        text,
    )
    if explicit_match:
        return explicit_match.group(1)
    ticker_match = re.search(r"(?<![A-Za-z0-9])\$([A-Z]{1,5}(?:\.[A-Z])?)\b", text)
    if ticker_match:
        return ticker_match.group(1)
    market_context_patterns = (
        r"\b([A-Z]{1,5}(?:\.[A-Z])?)\s+(?:stock|stocks|share|shares|equity|price|drawdown|drawdowns|returns?)\b",
        r"\b(?:stock|stocks|share|shares|equity|price|drawdown|drawdowns|returns?)\s+(?:of|for|in)?\s*([A-Z]{1,5}(?:\.[A-Z])?)\b",
    )
    for pattern in market_context_patterns:
        contextual_match = re.search(pattern, text)
        if contextual_match:
            candidate = contextual_match.group(1)
            if candidate.upper() not in {"A", "AN", "AND", "FOR", "IN", "OR", "THE", "US"}:
                return candidate
    if any(term in lowered for term in ("s&p 500", "s&p500", "sp500", "standard & poor", "standard and poor")):
        return "^GSPC"
    if any(term in lowered for term in ("crude oil", "oil price", "oil prices", "wti", "brent")):
        return "CL=F"
    if any(term in lowered for term in ("gasoline price", "gas prices", "gas price")):
        return "RB=F"
    return ""


def _yfinance_download_history(yf: Any, ticker: str, *, start: str, end: str, interval: str) -> Any:
    return yf.download(
        ticker,
        start=start or None,
        end=end or None,
        interval=interval or "1d",
        auto_adjust=False,
        progress=False,
    )


def _yfinance_equity_symbol_candidates(yf: Any, query: str) -> list[str]:
    search_factory = getattr(yf, "Search", None)
    if search_factory is None:
        return []
    try:
        search_result = search_factory(query, max_results=8)
    except Exception:
        return []
    quotes = getattr(search_result, "quotes", []) or []
    candidates: list[str] = []
    preferred_exchanges = {"NMS", "NYQ", "ASE", "PCX", "BTS", "NASDAQ", "NYSE"}
    ranked_quotes = sorted(
        (quote for quote in quotes if isinstance(quote, dict)),
        key=lambda quote: (
            str(quote.get("quoteType", "")).upper() == "EQUITY",
            str(quote.get("exchange", "")).upper() in preferred_exchanges,
            bool(str(quote.get("prevName", "")).strip()),
            float(quote.get("score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    for quote in ranked_quotes:
        if str(quote.get("quoteType", "")).upper() != "EQUITY":
            continue
        symbol = str(quote.get("symbol", "") or "").strip().upper()
        if not symbol:
            continue
        candidates.append(symbol)
    return list(dict.fromkeys(candidates))


def _load_stanza_pipeline(processors: str) -> Any | None:
    key = ("en", processors)
    if key in _STANZA_PIPELINES:
        return _STANZA_PIPELINES[key]
    if not _module_available("stanza"):
        return None
    try:
        import stanza

        pipeline = stanza.Pipeline(
            lang="en",
            processors=processors,
            use_gpu=resolve_device("auto") == "cuda",
            verbose=False,
        )
    except Exception:
        return None
    _STANZA_PIPELINES[key] = pipeline
    return pipeline


def _load_flair_object(kind: str) -> Any | None:
    if kind in _FLAIR_OBJECTS:
        return _FLAIR_OBJECTS[kind]
    if not _module_available("flair"):
        return None
    try:
        import torch
        import flair

        flair.device = torch.device("cuda" if resolve_device("auto") == "cuda" else "cpu")
        if kind == "sentiment":
            from flair.models import TextClassifier

            obj = TextClassifier.load("sentiment")
        elif kind == "ner":
            from flair.models import SequenceTagger

            obj = SequenceTagger.load("ner")
        elif kind == "pos":
            from flair.models import SequenceTagger

            obj = SequenceTagger.load("upos")
        else:
            return None
    except Exception:
        return None
    _FLAIR_OBJECTS[kind] = obj
    return obj


def _db_search(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    query = str(
        params.get("query")
        or getattr(context.state, "rewritten_question", "")
        or getattr(context.state, "question", "")
    ).strip()
    budget = infer_retrieval_budget(
        query,
        inputs=params,
        configured_mode=os.getenv("CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE", "hybrid"),
    )
    retrieval_strategy = budget.retrieval_strategy
    top_k = budget.top_k
    date_from = str(params.get("date_from", "")).strip()
    date_to = str(params.get("date_to", "")).strip()
    retrieval_mode = budget.retrieval_mode
    lexical_top_k = budget.lexical_top_k
    dense_top_k = budget.dense_top_k
    use_rerank = budget.use_rerank
    rerank_top_k = budget.rerank_top_k
    fusion_k = budget.fusion_k
    year_balance_mode = str(params.get("year_balance", "auto")).strip().lower() or "auto"
    allow_local_fallback = _env_flag("CORPUSAGENT2_ALLOW_LOCAL_FALLBACK", True)
    require_backend_services = _env_flag("CORPUSAGENT2_REQUIRE_BACKEND_SERVICES", False)
    if require_backend_services:
        allow_local_fallback = False
    caveats: list[str] = []
    rows: list[dict[str, Any]] = []
    primary_error: Exception | None = None
    sql_fallback_attempted = False
    years = _year_range(date_from, date_to)
    source_filters = _query_source_filters(query)
    backend_query = _query_without_field_filters(query) if source_filters else query
    use_year_balance = bool(years) and year_balance_mode not in {"0", "false", "no", "off"}
    if year_balance_mode == "auto":
        use_year_balance = len(years) >= 2
    _, sql_store_error = _queryable_sql_store(context)
    sql_store_available = not bool(sql_store_error)

    if retrieval_strategy == "exhaustive_analytic" and sql_store_available:
        rows, duplicates_removed = _prepare_result_rows(
            _sql_search_rows(
                query=query,
                top_k=0,
                date_from=date_from,
                date_to=date_to,
                context=context,
            ),
            top_k=0,
            retrieval_mode="sql",
        )
        if rows:
            caveats.append(
                f"Exhaustive analytical retrieval used full lexical Postgres materialization and returned {len(rows)} matching documents."
            )
            if duplicates_removed > 0:
                caveats.append(f"Suppressed {duplicates_removed} near-duplicate SQL retrieval hits.")
        else:
            caveats.append(
                _source_filtered_no_data_caveat(context, source_filters)
                if source_filters
                else "Exhaustive SQL retrieval did not find documents matching the main query entities."
            )
        return _search_result(
            context=context,
            query=query,
            retrieval_mode="sql",
            retrieval_strategy=retrieval_strategy,
            rows=rows,
            caveats=caveats,
        )
    if retrieval_strategy == "exhaustive_analytic" and not sql_store_available and context.runtime is not None:
        rows, duplicates_removed = _prepare_result_rows(
            _local_exhaustive_rows(
                query=query,
                top_k=0,
                date_from=date_from,
                date_to=date_to,
                context=context,
            ),
            top_k=0,
            retrieval_mode="local_exhaustive",
        )
        if rows:
            reason = f" ({sql_store_error})" if sql_store_error else ""
            caveats.append(
                "Exhaustive analytical retrieval used full local lexical materialization because the Postgres corpus store was unavailable"
                f"{reason}."
            )
            if duplicates_removed > 0:
                caveats.append(f"Suppressed {duplicates_removed} near-duplicate local exhaustive retrieval hits.")
            return _search_result(
                context=context,
                query=query,
                retrieval_mode="local_exhaustive",
                retrieval_strategy=retrieval_strategy,
                rows=rows,
                caveats=caveats,
            )
    if retrieval_strategy == "exhaustive_analytic" and not sql_store_available:
        if sql_store_error:
            caveats.append(
                f"Exhaustive Postgres materialization was unavailable ({sql_store_error}), so ranked retrieval was used instead."
            )
        else:
            caveats.append(
                "Exhaustive retrieval was requested, but the Postgres corpus store is unavailable, so ranked retrieval was used instead."
            )

    def _primary_search_once(window_from: str, window_to: str, limit: int) -> list[dict[str, Any]]:
        return context.search_backend.search(
            query=backend_query,
            top_k=limit,
            date_from=window_from,
            date_to=window_to,
            retrieval_mode=retrieval_mode,
            lexical_top_k=lexical_top_k,
            dense_top_k=dense_top_k,
            use_rerank=use_rerank,
            rerank_top_k=rerank_top_k,
            fusion_k=fusion_k,
        )

    def _balanced_candidates(search_fn, retrieval_label: str) -> tuple[list[dict[str, Any]], int]:
        if not use_year_balance:
            base_rows = search_fn(date_from, date_to, max(top_k * 3, 40))
            return _prepare_result_rows(base_rows, top_k=top_k, retrieval_mode=retrieval_label)
        per_year_limit = max(12, min(40, ((top_k + len(years) - 1) // max(len(years), 1)) * 3))
        combined: list[dict[str, Any]] = []
        global_rows = search_fn(date_from, date_to, max(top_k * 2, per_year_limit))
        combined.extend(global_rows)
        for year in years:
            combined.extend(search_fn(f"{year}-01-01", f"{year}-12-31", per_year_limit))
        prepared, duplicates_removed = _prepare_result_rows(
            combined,
            top_k=top_k,
            retrieval_mode=retrieval_label,
            years=years,
        )
        return prepared, duplicates_removed

    try:
        rows, duplicates_removed = _balanced_candidates(_primary_search_once, retrieval_mode)
        if source_filters:
            rows = [
                row for row in rows
                if _row_matches_source_filters(row.get("outlet", row.get("source", "")), source_filters)
            ]
        if use_year_balance and years:
            present_years = sorted({year for year in (_row_year(row) for row in rows) if year is not None})
            caveats.append(f"Year-balanced retrieval was applied across {', '.join(str(year) for year in years)}.")
            missing_years = [str(year) for year in years if year not in present_years]
            if missing_years:
                caveats.append(f"No strong retrieval hits were found for year buckets: {', '.join(missing_years)}.")
        if duplicates_removed > 0:
            caveats.append(f"Suppressed {duplicates_removed} near-duplicate or syndicated retrieval hits.")
    except Exception as exc:
        primary_error = exc
    weak_anchor_match = bool(rows) and not _rows_match_query_anchor_terms(rows, query)
    if not rows or weak_anchor_match:
        sql_rows: list[dict[str, Any]] = []
        sql_duplicates_removed = 0
        sql_error: Exception | None = None
        if sql_store_available:
            sql_fallback_attempted = True
            try:
                sql_rows, sql_duplicates_removed = _balanced_candidates(
                    lambda window_from, window_to, limit: _sql_search_rows(
                        query=query,
                        top_k=limit,
                        date_from=window_from,
                        date_to=window_to,
                        context=context,
                    ),
                    "sql",
                )
            except Exception as exc:
                sql_error = exc
        if sql_rows and (not rows or _rows_match_query_anchor_terms(sql_rows, query)):
            rows = sql_rows
            if primary_error is not None:
                caveats.append(f"Primary search backend failed and Postgres SQL retrieval was used instead: {primary_error}")
            elif weak_anchor_match:
                caveats.append("Hybrid retrieval returned off-topic documents, so Postgres SQL retrieval was used instead.")
            if sql_duplicates_removed > 0:
                caveats.append(f"Suppressed {sql_duplicates_removed} near-duplicate SQL retrieval hits.")
        elif weak_anchor_match:
            if sql_error is not None:
                caveats.append(f"Hybrid retrieval returned off-topic documents and SQL fallback was unavailable: {sql_error}")
            elif not sql_store_available and sql_store_error:
                caveats.append(f"Hybrid retrieval returned off-topic documents and SQL fallback was unavailable: {sql_store_error}")
            else:
                caveats.append("Hybrid retrieval returned documents that did not match the main query entities; off-topic hits were discarded.")
            rows = []
    if not rows:
        sandbox_rows = _sandbox_retrieval_rows(
            query=query,
            top_k=top_k,
            date_from=date_from,
            date_to=date_to,
            context=context,
        )
        if sandbox_rows:
            rows, sandbox_duplicates_removed = _prepare_result_rows(
                sandbox_rows,
                top_k=top_k,
                retrieval_mode="sandbox",
                years=years if use_year_balance else None,
            )
            caveats.append("Hybrid and SQL retrieval did not return usable evidence, so a bounded sandbox retrieval fallback was used.")
            if sandbox_duplicates_removed > 0:
                caveats.append(f"Suppressed {sandbox_duplicates_removed} near-duplicate sandbox retrieval hits.")
    if primary_error is not None and not rows:
        if sql_fallback_attempted and source_filters:
            caveats.append(
                "Primary search backend failed after source filters were stripped for fallback retrieval, "
                f"and SQL fallback found no matching documents: {primary_error}"
            )
            caveats.append(_source_filtered_no_data_caveat(context, source_filters))
            return ToolExecutionResult(
                payload={
                    "results": [],
                    "query": query,
                    "retrieval_mode": "sql",
                    "retrieval_strategy": retrieval_strategy,
                    "result_count": 0,
                    "document_count": 0,
                },
                evidence=[],
                caveats=caveats,
                metadata={
                    "no_data": True,
                    "no_data_reason": "source_filtered_sql_fallback_empty",
                    "primary_error": str(primary_error),
                },
            )
        if context.runtime is None or not allow_local_fallback:
            raise primary_error
        from .agent_backends import LocalSearchBackend

        local_rows = LocalSearchBackend(context.runtime).search(
            query=backend_query,
            top_k=max(top_k * 3, 40),
            date_from=date_from,
            date_to=date_to,
            retrieval_mode=retrieval_mode,
            lexical_top_k=lexical_top_k,
            dense_top_k=dense_top_k,
            use_rerank=use_rerank,
            rerank_top_k=rerank_top_k,
            fusion_k=fusion_k,
        )
        rows, local_duplicates_removed = _prepare_result_rows(
            local_rows,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            years=years if use_year_balance else None,
        )
        caveats.append(f"Primary search backend failed and local retrieval fallback was used: {primary_error}")
        if local_duplicates_removed > 0:
            caveats.append(f"Suppressed {local_duplicates_removed} near-duplicate local retrieval hits.")
    if rows and not _rows_match_query_anchor_terms(rows, query):
        caveats.append("Retrieved documents did not match the main query entities closely enough, so no evidence rows were kept.")
        rows = []
    return _search_result(
        context=context,
        query=query,
        retrieval_mode=retrieval_mode,
        retrieval_strategy=retrieval_strategy,
        rows=rows,
        caveats=caveats,
    )


def _sql_query_search(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    query = str(
        params.get("query")
        or getattr(context.state, "rewritten_question", "")
        or getattr(context.state, "question", "")
    ).strip()
    budget = infer_retrieval_budget(
        query,
        inputs=params,
        configured_mode=os.getenv("CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE", "hybrid"),
    )
    retrieval_strategy = budget.retrieval_strategy
    top_k = budget.top_k
    date_from = str(params.get("date_from", "")).strip()
    date_to = str(params.get("date_to", "")).strip()
    _, sql_store_error = _queryable_sql_store(context)
    sql_store_available = not bool(sql_store_error)
    if retrieval_strategy == "exhaustive_analytic":
        if sql_store_available:
            rows, duplicates_removed = _prepare_result_rows(
                _sql_search_rows(
                    query=query,
                    top_k=0,
                    date_from=date_from,
                    date_to=date_to,
                    context=context,
                ),
                top_k=0,
                retrieval_mode="sql",
            )
            source_filters = _query_source_filters(query)
            caveats = [] if rows else [
                _source_filtered_no_data_caveat(context, source_filters)
                if source_filters
                else "Exhaustive SQL retrieval did not find documents matching the main query entities."
            ]
            if rows:
                caveats.append(
                    f"Exhaustive analytical retrieval used full lexical Postgres materialization and returned {len(rows)} matching documents."
                )
            if duplicates_removed > 0:
                caveats.append(f"Suppressed {duplicates_removed} near-duplicate SQL retrieval hits.")
            return _search_result(
                context=context,
                query=query,
                retrieval_mode="sql",
                retrieval_strategy=retrieval_strategy,
                rows=rows,
                caveats=caveats,
            )
        rows, duplicates_removed = _prepare_result_rows(
            _local_exhaustive_rows(
                query=query,
                top_k=0,
                date_from=date_from,
                date_to=date_to,
                context=context,
            ),
            top_k=0,
            retrieval_mode="local_exhaustive",
        )
        caveats = []
        if rows:
            caveats.append(
                "Exhaustive analytical retrieval used full local lexical materialization because the Postgres corpus store was unavailable"
                f" ({sql_store_error})."
            )
        if duplicates_removed > 0:
            caveats.append(f"Suppressed {duplicates_removed} near-duplicate local exhaustive retrieval hits.")
        if not rows:
            caveats.append(
                f"Exhaustive SQL retrieval was unavailable ({sql_store_error}) and local lexical materialization did not find matching documents."
            )
        return _search_result(
            context=context,
            query=query,
            retrieval_mode="local_exhaustive",
            retrieval_strategy=retrieval_strategy,
            rows=rows,
            caveats=caveats,
        )
    years = _year_range(date_from, date_to)
    if years:
        combined: list[dict[str, Any]] = []
        per_year_limit = max(12, min(40, ((top_k + len(years) - 1) // max(len(years), 1)) * 3))
        combined.extend(
            _sql_search_rows(
                query=query,
                top_k=max(top_k * 2, per_year_limit),
                date_from=date_from,
                date_to=date_to,
                context=context,
            )
        )
        for year in years:
            combined.extend(
                _sql_search_rows(
                    query=query,
                    top_k=per_year_limit,
                    date_from=f"{year}-01-01",
                    date_to=f"{year}-12-31",
                    context=context,
                )
            )
        rows, duplicates_removed = _prepare_result_rows(combined, top_k=top_k, retrieval_mode="sql", years=years)
    else:
        rows, duplicates_removed = _prepare_result_rows(
            _sql_search_rows(
                query=query,
                top_k=max(top_k * 3, 40),
                date_from=date_from,
                date_to=date_to,
                context=context,
            ),
            top_k=top_k,
            retrieval_mode="sql",
        )
    caveats = [] if rows else ["Postgres SQL retrieval did not find documents matching the main query entities."]
    if duplicates_removed > 0:
        caveats.append(f"Suppressed {duplicates_removed} near-duplicate SQL retrieval hits.")
    return ToolExecutionResult(
        payload={
            "results": rows,
            "query": query,
            "retrieval_mode": "sql",
            "retrieval_strategy": retrieval_strategy,
            "result_count": len(rows),
        },
        evidence=list(rows),
        caveats=caveats,
    )


def _fetch_documents(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    explicit_doc_ids = [str(item) for item in params.get("doc_ids", []) if str(item).strip()]
    doc_ids = list(explicit_doc_ids)
    search_rows = _search_rows(deps)
    search_lookup = {
        str(row.get("doc_id", "")).strip(): row
        for row in search_rows
        if str(row.get("doc_id", "")).strip()
    }
    working_set_ref = _resolve_working_set_ref(params, deps)
    if working_set_ref and not explicit_doc_ids:
        doc_ids = []
    if not doc_ids and not working_set_ref:
        doc_ids = _working_set_doc_ids(deps)
    if not doc_ids and not working_set_ref:
        doc_ids = [str(row.get("doc_id", "")) for row in search_rows if str(row.get("doc_id", "")).strip()]
    batching = params.get("batching") if isinstance(params.get("batching"), dict) else {}
    explicit_limit = params.get("limit", params.get("batch_size", batching.get("batch_size")))
    try:
        fetch_limit = int(explicit_limit) if explicit_limit not in (None, "") else int(os.getenv("CORPUSAGENT2_WORKING_SET_FETCH_LIMIT", "1000"))
    except ValueError:
        fetch_limit = 1000
    fetch_limit = max(1, fetch_limit)
    doc_ids = _dedupe_doc_ids(doc_ids)
    if not doc_ids and not working_set_ref:
        return ToolExecutionResult(payload={"documents": []}, evidence=[])
    allow_local_fallback = _env_flag("CORPUSAGENT2_ALLOW_LOCAL_FALLBACK", True)
    require_backend_services = _env_flag("CORPUSAGENT2_REQUIRE_BACKEND_SERVICES", False)
    if require_backend_services:
        allow_local_fallback = False

    def _merge_document(row: dict[str, Any]) -> dict[str, Any]:
        doc_id = str(row.get("doc_id", "")).strip()
        merged = dict(search_lookup.get(doc_id, {}))
        merged.update(row)
        if "date" not in merged or not str(merged.get("date", "")).strip():
            merged["date"] = str(merged.get("published_at", merged.get("year", "")))
        if "outlet" not in merged or not str(merged.get("outlet", "")).strip():
            merged["outlet"] = str(merged.get("source", merged.get("source_domain", "")))
        merged["score"] = _coerce_score(merged.get("score", 0.0))
        merged["score_display"] = str(merged.get("score_display") or _score_display(merged["score"]))
        if "score_components" in merged and not isinstance(merged.get("score_components"), dict):
            merged.pop("score_components", None)
        return _with_cleaned_document_text(merged)

    caveats: list[str] = []
    total_available = len(doc_ids)
    if working_set_ref and not doc_ids:
        total_available = _count_working_set(context, working_set_ref, 0)
        fetcher = getattr(context.working_store, "fetch_working_set_documents", None)
        if callable(fetcher):
            try:
                rows = fetcher(context.run_id, working_set_ref, limit=fetch_limit, offset=0)
            except Exception as exc:
                if not allow_local_fallback:
                    raise
                rows = []
                caveats.append(f"Working-set document fetch failed and runtime fallback was used: {exc}")
        else:
            ids = _fetch_working_set_ids(context, working_set_ref, limit=fetch_limit)
            rows = context.working_store.fetch_documents(ids) if ids else []
    else:
        try:
            rows = context.working_store.fetch_documents(doc_ids[:fetch_limit] if len(doc_ids) > fetch_limit else doc_ids)
        except Exception as exc:
            if not allow_local_fallback:
                raise
            rows = []
            caveats.append(f"Working-set document fetch failed and runtime fallback was used: {exc}")
    if not rows and context.runtime is not None and allow_local_fallback:
        try:
            fallback_ids = doc_ids[:fetch_limit]
            if working_set_ref and not fallback_ids:
                fallback_ids = _fetch_working_set_ids(context, working_set_ref, limit=fetch_limit)
            df = context.runtime.load_docs(fallback_ids)
            rows = [
                _merge_document(
                    {
                        "doc_id": str(row.doc_id),
                        "title": str(getattr(row, "title", "")),
                        "text": str(getattr(row, "text", "")),
                        "published_at": str(getattr(row, "published_at", "")),
                        "date": str(getattr(row, "published_at", "")),
                        "outlet": str(getattr(row, "source", "")),
                        "source": str(getattr(row, "source", "")),
                    }
                )
                for row in df.itertuples(index=False)
            ]
        except Exception as exc:
            caveats.append(f"Runtime document lookup fallback failed: {exc}")
    else:
        rows = [_merge_document(dict(row)) for row in rows]
    if total_available > len(rows):
        caveats.append(
            f"Fetched {len(rows)} preview/batch documents from working set of {total_available}. "
            "Large-population analysis should consume working_set_ref in batches instead of treating this preview as the full corpus."
        )
    return ToolExecutionResult(
        payload={
            "documents": rows,
            "working_set_ref": working_set_ref,
            "document_count": total_available or len(rows),
            "returned_document_count": len(rows),
            "documents_truncated": total_available > len(rows),
        },
        evidence=[{"doc_id": row["doc_id"], "score": row.get("score", 0.0)} for row in rows],
        caveats=caveats,
    )


def _create_working_set(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    filters = dict(params.get("filter", {})) if isinstance(params.get("filter"), dict) else {}
    upstream_ref = str(params.get("working_set_ref", "") or _working_set_ref(deps)).strip()
    if upstream_ref and not filters:
        count = _count_working_set(context, upstream_ref, 0)
        preview_ids = _fetch_working_set_ids(context, upstream_ref, limit=_result_preview_limit())
        if context.state is not None:
            context.state.working_set_ref = upstream_ref
            context.state.working_set_count = count
            context.state.working_set_doc_ids = list(preview_ids)
        return ToolExecutionResult(
            payload={
                "working_set_ref": upstream_ref,
                "working_set_doc_ids": preview_ids,
                "document_count": count,
                "preview_count": len(preview_ids),
                "working_set_truncated": count > len(preview_ids),
            },
            caveats=[
                f"Working set '{upstream_ref}' contains {count} documents; payload includes only preview IDs."
            ]
            if count > len(preview_ids)
            else [],
        )
    rows = _text_rows(deps)
    if rows:
        context.working_store.record_documents(context.run_id, rows)
    doc_ids = _working_set_doc_ids(deps)
    dependency_rows = _dependency_rows(deps)
    if not doc_ids:
        filtered_rows = dependency_rows
        if filters.get("language_in"):
            allowed = {str(item).strip().lower() for item in filters.get("language_in", []) if str(item).strip()}
            filtered_rows = [
                row for row in filtered_rows if str(row.get("language", "")).strip().lower() in allowed
            ]
        doc_ids = [str(row.get("doc_id", "")) for row in filtered_rows if str(row.get("doc_id", "")).strip()]
    doc_ids = _dedupe_doc_ids(doc_ids or [str(row.get("doc_id", "")) for row in rows if str(row.get("doc_id", "")).strip()])
    materialization_rows_by_id: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate([*dependency_rows, *rows], start=1):
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id or doc_id in materialization_rows_by_id:
            continue
        materialization_rows_by_id[doc_id] = {
            **row,
            "doc_id": doc_id,
            "rank": int(row.get("rank", rank) or rank),
            "score": _coerce_score(row.get("score", row.get("_retrieval_score", 0.0))),
        }
    materialization_rows = [
        materialization_rows_by_id.get(doc_id, {"doc_id": doc_id, "rank": index, "score": 0.0})
        for index, doc_id in enumerate(doc_ids, start=1)
    ]
    label = ""
    materialized_count = 0
    if materialization_rows:
        label, materialized_count = _materialize_result_working_set(
            context,
            query=str(params.get("label") or params.get("name") or "working_set"),
            retrieval_mode="working_set",
            rows=materialization_rows,
        )
    if context.state is not None:
        context.state.working_set_doc_ids = list(doc_ids)
        if label:
            context.state.working_set_ref = label
            context.state.working_set_count = materialized_count or len(doc_ids)
    preview_limit = _result_preview_limit()
    preview_ids = list(doc_ids[:preview_limit])
    caveats = []
    if label and len(doc_ids) > len(preview_ids):
        caveats.append(
            f"Working set '{label}' contains {materialized_count or len(doc_ids)} documents; payload includes only preview IDs."
        )
    return ToolExecutionResult(
        payload={
            "working_set_ref": label,
            "working_set_doc_ids": preview_ids,
            "working_set_doc_ids_truncated": len(doc_ids) > len(preview_ids),
            "working_set_doc_ids_count": len(doc_ids),
            "document_count": len(doc_ids),
        },
        metadata={"working_set_ref": label} if label else {},
        caveats=caveats,
    )


STRUCTURED_FILTER_ALIASES = {
    "source": ("source", "outlet", "source_domain", "publisher"),
    "outlet": ("outlet", "source", "source_domain", "publisher"),
    "source_name": ("source", "outlet", "source_domain", "publisher"),
    "publisher": ("publisher", "source", "outlet", "source_domain"),
    "domain": ("source_domain", "source", "outlet"),
    "language": ("language", "lang"),
    "lang": ("lang", "language"),
    "published_at": ("published_at", "date"),
    "date": ("date", "published_at"),
}


def _normalise_structured_filter_key(key: str) -> tuple[str, str]:
    raw = str(key or "").strip()
    for suffix, mode in (
        ("_contains", "contains"),
        ("_in", "in"),
        ("_equals", "equals"),
        ("_eq", "equals"),
    ):
        if raw.endswith(suffix):
            return raw[: -len(suffix)], mode
    return raw, "contains"


def _filter_expected_values(value: Any, mode: str) -> list[str]:
    if isinstance(value, dict):
        for key in ("contains", "in", "equals", "eq", "value", "values"):
            if key in value:
                next_mode = "contains" if key == "contains" else mode
                return _filter_expected_values(value.get(key), next_mode)
        return []
    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            if isinstance(item, dict):
                values.extend(_filter_expected_values(item, mode))
                continue
            text = str(item).strip()
            if text:
                values.append(text)
        return values
    text = str(value).strip()
    return [text] if text else []


def _is_filter_control_metadata_key(key: str) -> bool:
    normalized = str(key or "").strip().lower()
    if normalized in {
        "source_node",
        "source_node_id",
        "source_ref",
        "source_result",
        "working_set_node",
        "working_set_source_node",
        "working_set_source_node_id",
    }:
        return True
    return normalized.endswith(("_source", "_source_node", "_source_node_id", "_source_ref", "_source_result"))


def _row_filter_actual_values(row: dict[str, Any], key: str) -> tuple[list[str], bool]:
    base_key, _ = _normalise_structured_filter_key(key)
    candidate_keys = STRUCTURED_FILTER_ALIASES.get(base_key, (base_key,))
    values: list[str] = []
    supported = False
    for candidate in candidate_keys:
        if candidate not in row:
            continue
        supported = True
        value = row.get(candidate)
        if isinstance(value, (list, tuple, set)):
            values.extend(str(item).strip() for item in value if str(item).strip())
        elif value not in (None, ""):
            values.append(str(value).strip())
    return values, supported


def _structured_filter_matches(row: dict[str, Any], filters: dict[str, Any]) -> tuple[bool, set[str]]:
    unsupported: set[str] = set()
    for raw_key, expected in filters.items():
        key, mode = _normalise_structured_filter_key(str(raw_key))
        actual_values, supported = _row_filter_actual_values(row, key)
        if isinstance(expected, dict) and "exists" in expected:
            should_exist = bool(expected.get("exists"))
            has_value = bool(actual_values)
            if should_exist != has_value:
                return False, unsupported
            continue
        expected_values = _filter_expected_values(expected, mode)
        if not expected_values:
            continue
        if not supported:
            unsupported.add(key)
            return False, unsupported
        actual_norm = [_normalize_source_filter(value) for value in actual_values]
        expected_norm = [_normalize_source_filter(value) for value in expected_values]
        if mode == "equals":
            matched = any(actual == expected_value for actual in actual_norm for expected_value in expected_norm)
        elif mode == "in":
            expected_set = set(expected_norm)
            matched = any(actual in expected_set for actual in actual_norm)
        else:
            matched = any(expected_value in actual for actual in actual_norm for expected_value in expected_norm)
        if not matched:
            return False, unsupported
    return True, unsupported


def _positive_int_param(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _working_set_sort_specs(params: dict[str, Any]) -> list[dict[str, str]]:
    raw = params.get("sort_by", params.get("order_by", []))
    if isinstance(raw, str):
        raw = [{"field": raw}]
    if isinstance(raw, dict):
        raw = [raw]
    specs: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return specs
    for item in raw:
        if isinstance(item, str):
            field = item
            order = "desc" if field.startswith("-") else "asc"
            field = field.lstrip("-")
        elif isinstance(item, dict):
            field = str(item.get("field") or item.get("key") or item.get("column") or "").strip()
            order = str(item.get("order") or item.get("direction") or "asc").strip().lower()
        else:
            continue
        if field:
            specs.append({"field": field, "order": "desc" if order.startswith("desc") else "asc"})
    return specs


def _working_set_sort_value(row: dict[str, Any], field: str) -> Any:
    normalized = field.strip().lower().lstrip("_").replace("-", "_")
    aliases = {
        "retrieval_score": ("score", "_retrieval_score", "retrieval_score"),
        "score": ("score", "_retrieval_score", "retrieval_score"),
        "rank": ("rank", "_rank", "retrieval_rank"),
        "date": ("published_at", "date"),
    }
    for candidate in aliases.get(normalized, (field, normalized)):
        if candidate not in row:
            continue
        value = row.get(candidate)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value).lower()
    return None


def _apply_working_set_sort(rows: list[dict[str, Any]], specs: list[dict[str, str]]) -> list[dict[str, Any]]:
    ordered = list(rows)
    for spec in reversed(specs):
        field = spec["field"]
        reverse = spec["order"] == "desc"
        present = [row for row in ordered if _working_set_sort_value(row, field) is not None]
        missing = [row for row in ordered if _working_set_sort_value(row, field) is None]
        present.sort(key=lambda row: _working_set_sort_value(row, field), reverse=reverse)
        ordered = present + missing
    return ordered


def _filter_working_set(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    params = _merged_payload_params(params)
    query = str(params.get("query", "") or "").strip()
    filters = params.get("filters", params.get("filter", {}))
    filters = dict(filters) if isinstance(filters, dict) else {}
    predicate = params.get("predicate", {})
    if isinstance(predicate, dict):
        filters = {**filters, **predicate}
    filters = {
        key: value
        for key, value in filters.items()
        if not _is_filter_control_metadata_key(str(key))
    }
    min_text_length = None
    for length_key in ("min_text_length", "minimum_text_length", "text_min_length"):
        if length_key in filters:
            min_text_length = _positive_int_param(filters.pop(length_key))
            break
    date_from = str(params.get("date_from", "") or "").strip()
    date_to = str(params.get("date_to", "") or "").strip()
    for date_key in ("published_at", "date"):
        date_filter = filters.get(date_key)
        if not isinstance(date_filter, dict):
            continue
        extracted_from = str(
            date_filter.get("gte")
            or date_filter.get("from")
            or date_filter.get("after")
            or date_filter.get("start")
            or ""
        ).strip()
        extracted_to = str(
            date_filter.get("lte")
            or date_filter.get("to")
            or date_filter.get("before")
            or date_filter.get("end")
            or ""
        ).strip()
        if extracted_from and not date_from:
            date_from = extracted_from
        if extracted_to and not date_to:
            date_to = extracted_to
        filters.pop(date_key, None)
        break
    annotation_doc_ids: set[str] | None = None
    annotation_filter_keys: set[str] = set()
    annotation_filter_caveats: list[str] = []
    for field_name, spec in list(filters.items()):
        if not isinstance(spec, dict) or not str(spec.get("source", "") or "").strip():
            continue
        source_name = str(spec.get("source", "") or "").strip()
        source_result = deps.get(source_name)
        if source_result is None:
            annotation_filter_caveats.append(f"Annotation filter '{field_name}' referenced missing source '{source_name}'.")
            annotation_filter_keys.add(str(field_name))
            annotation_doc_ids = set()
            continue
        source_payload = source_result.payload if isinstance(source_result.payload, dict) else {}
        source_rows = source_payload.get("rows", []) if isinstance(source_payload, dict) else []
        if not isinstance(source_rows, list):
            source_rows = []
        expected_values = {str(item).strip().lower() for item in _filter_expected_values(spec, "in")}
        expected_values.discard("")
        annotation_field = str(spec.get("field", field_name) or field_name)
        matched_doc_ids = {
            str(row.get("doc_id", "")).strip()
            for row in source_rows
            if str(row.get("doc_id", "")).strip()
            and (
                not expected_values
                or str(row.get(annotation_field, row.get(field_name, "")) or "").strip().lower() in expected_values
            )
        }
        annotation_doc_ids = matched_doc_ids if annotation_doc_ids is None else annotation_doc_ids.intersection(matched_doc_ids)
        annotation_filter_keys.add(str(field_name))
    document_filters = {key: value for key, value in filters.items() if str(key) not in annotation_filter_keys}
    limit = _positive_int_param(params.get("limit", params.get("top_k", params.get("max_documents"))))
    sort_specs = _working_set_sort_specs(params)
    upstream_ref = str(params.get("working_set_ref", "") or _resolve_working_set_ref(params, deps)).strip()
    if not upstream_ref:
        return ToolExecutionResult(
            payload={"results": [], "document_count": 0},
            caveats=["No upstream working_set_ref was available for working-set filtering."],
            metadata={"no_data": True, "no_data_reason": "missing_working_set_ref"},
        )
    if _cancel_requested(context):
        return ToolExecutionResult(
            payload={"results": [], "document_count": 0, "source_working_set_ref": upstream_ref, "cancelled": True},
            caveats=["Run abort was requested before working-set filtering started."],
            metadata={"cancelled": True, "no_data": True, "no_data_reason": "cancelled"},
        )
    if (
        not query
        and not document_filters
        and annotation_doc_ids is None
        and min_text_length is None
        and not date_from
        and not date_to
        and not limit
        and not sort_specs
    ):
        count = _count_working_set(context, upstream_ref, 0)
        preview_ids = _fetch_working_set_ids(context, upstream_ref, limit=_result_preview_limit())
        return ToolExecutionResult(
            payload={
                "results": [{"doc_id": doc_id} for doc_id in preview_ids],
                "working_set_ref": upstream_ref,
                "document_count": count,
                "preview_count": len(preview_ids),
                "results_truncated": count > len(preview_ids),
            },
            metadata={"working_set_ref": upstream_ref, "full_result_count": count},
        )

    try:
        batch_size = max(1, int(params.get("batch_size") or os.getenv("CORPUSAGENT2_WORKING_SET_FILTER_BATCH_SIZE", "5000")))
    except ValueError:
        batch_size = 5000
    total = _count_working_set(context, upstream_ref, 0)
    fetcher = getattr(context.working_store, "fetch_working_set_documents", None)
    if not callable(fetcher):
        return ToolExecutionResult(
            payload={"results": [], "document_count": 0, "source_working_set_ref": upstream_ref},
            caveats=["The configured working-set store cannot fetch working-set documents for filtering."],
            metadata={"no_data": True, "no_data_reason": "working_set_fetch_unavailable"},
        )

    filtered_rows: list[dict[str, Any]] = []
    unsupported_filter_keys: set[str] = set()
    missing_date_count = 0
    offset = 0
    cancelled = False
    while True:
        if _cancel_requested(context):
            cancelled = True
            break
        batch = fetcher(context.run_id, upstream_ref, limit=batch_size, offset=offset)
        if not batch:
            break
        for raw_row in batch:
            if _cancel_requested(context):
                cancelled = True
                break
            row = _with_cleaned_document_text(dict(raw_row))
            published_at = str(row.get("published_at", row.get("date", "")) or "")
            if (date_from or date_to) and not published_at:
                missing_date_count += 1
                continue
            if date_from and published_at and published_at < date_from:
                continue
            if date_to and published_at and published_at > date_to:
                continue
            doc_id = str(row.get("doc_id", "")).strip()
            if annotation_doc_ids is not None and doc_id not in annotation_doc_ids:
                continue
            if min_text_length is not None:
                text_for_length = _row_analysis_text(row)
                if len(text_for_length) < min_text_length:
                    continue
            if document_filters:
                filter_matched, unsupported = _structured_filter_matches(row, document_filters)
                unsupported_filter_keys.update(unsupported)
                if not filter_matched:
                    continue
            if not query or _row_matches_query_expression(row, query):
                filtered_rows.append(
                    {
                        "doc_id": doc_id,
                        "title": str(row.get("title", "")),
                        "snippet": str(row.get("text", row.get("cleaned_text", "")))[:360],
                        "outlet": str(row.get("outlet", row.get("source", ""))),
                        "source": str(row.get("source", row.get("outlet", ""))),
                        "date": published_at,
                        "score": _coerce_score(row.get("score", 0.0)),
                    }
                )
        if cancelled:
            break
        offset += len(batch)
        if total and offset >= total:
            break

    if cancelled:
        preview_limit = _result_preview_limit()
        preview_rows = filtered_rows[:preview_limit]
        return ToolExecutionResult(
            payload={
                "results": preview_rows,
                "query": query,
                "source_working_set_ref": upstream_ref,
                "retrieval_mode": "working_set_filter",
                "retrieval_strategy": "working_set_filter",
                "result_count": len(filtered_rows),
                "document_count": len(filtered_rows),
                "preview_count": len(preview_rows),
                "results_truncated": len(filtered_rows) > len(preview_rows),
                "cancelled": True,
            },
            evidence=preview_rows,
            caveats=["Run abort was requested; working-set filtering stopped early."],
            metadata={
                "source_working_set_ref": upstream_ref,
                "full_result_count": len(filtered_rows),
                "filtered_from_working_set": True,
                "payload_truncated": len(filtered_rows) > len(preview_rows),
                "cancelled": True,
                "no_data": not filtered_rows,
                "no_data_reason": "cancelled" if not filtered_rows else "",
            },
        )

    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in filtered_rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id or doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        row["rank"] = len(deduped) + 1
        deduped.append(row)
    matched_count_before_limit = len(deduped)
    if sort_specs:
        deduped = _apply_working_set_sort(deduped, sort_specs)
    if limit is not None:
        deduped = deduped[:limit]
    for rank, row in enumerate(deduped, start=1):
        row["rank"] = rank

    label, materialized_count = _materialize_result_working_set(
        context,
        query=f"{upstream_ref}\n{query}\nlimit={limit or ''}\nsort={json.dumps(sort_specs, sort_keys=True)}",
        retrieval_mode="working_set_filter",
        rows=deduped,
    )
    preview_limit = _result_preview_limit()
    preview_rows = deduped[:preview_limit]
    result_count = materialized_count or len(deduped)
    caveats = [
        (
            f"Filtered upstream working_set_ref='{upstream_ref}' with the requested query/metadata filters instead of running "
            f"another full-corpus retrieval; matched {matched_count_before_limit} of {total or offset} upstream documents."
        )
    ]
    if document_filters or annotation_filter_keys:
        caveats.append(
            "Applied structured working-set filters: "
            + json.dumps(filters, ensure_ascii=False, sort_keys=True)
            + "."
        )
    if annotation_filter_keys:
        caveats.append(
            "Resolved annotation-backed filters by doc_id from dependency rows: "
            + ", ".join(sorted(annotation_filter_keys))
            + f" ({len(annotation_doc_ids or set())} matching doc ids)."
        )
    if min_text_length is not None:
        caveats.append(f"Applied minimum text length filter: {min_text_length} characters.")
    if date_from or date_to:
        date_parts = []
        if date_from:
            date_parts.append(f"from {date_from}")
        if date_to:
            date_parts.append(f"to {date_to}")
        caveats.append("Applied date window filter: " + " ".join(date_parts) + ".")
    if missing_date_count:
        caveats.append(f"Skipped {missing_date_count} rows without date metadata while applying the date window filter.")
    caveats.extend(annotation_filter_caveats)
    if sort_specs:
        caveats.append("Applied working-set ordering: " + json.dumps(sort_specs, ensure_ascii=False, sort_keys=True) + ".")
    if limit is not None and matched_count_before_limit > len(deduped):
        caveats.append(f"Limited filtered working set to top {limit} of {matched_count_before_limit} matched documents.")
    if unsupported_filter_keys:
        caveats.append(
            "Requested working-set filters could not be evaluated from available document metadata: "
            + ", ".join(sorted(unsupported_filter_keys))
            + "."
        )
    if not deduped:
        caveats.append("Working-set filter found no documents matching the requested narrowing query.")
    payload = {
        "results": preview_rows,
        "query": query,
        "source_working_set_ref": upstream_ref,
        "working_set_ref": label,
        "retrieval_mode": "working_set_filter",
        "retrieval_strategy": "working_set_filter",
        "result_count": result_count,
        "document_count": result_count,
        "preview_count": len(preview_rows),
        "results_truncated": result_count > len(preview_rows),
    }
    return ToolExecutionResult(
        payload=payload,
        evidence=preview_rows,
        caveats=caveats,
        metadata={
            "working_set_ref": label,
            "source_working_set_ref": upstream_ref,
            "full_result_count": result_count,
            "full_match_count_before_limit": matched_count_before_limit,
            "annotation_filter_doc_count": len(annotation_doc_ids or set()) if annotation_doc_ids is not None else None,
            "min_text_length": min_text_length,
            "date_from": date_from,
            "date_to": date_to,
            "missing_date_count": missing_date_count,
            "filtered_from_working_set": True,
            "payload_truncated": result_count > len(preview_rows),
        },
    )


def _lang_id(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows, source_metadata, source_caveats = _analysis_document_rows_from_deps(deps, context)
    if not rows:
        return _no_input_documents_result("lang_id")
    detected = []
    providers = _provider_order("lang_id", ["langdetect"])
    used_provider = ""
    for row in rows:
        text = _row_analysis_text(row)
        language = "unknown"
        confidence = 0.0
        for provider in providers:
            try:
                if provider == "langdetect" and _module_available("langdetect"):
                    from langdetect import DetectorFactory, detect_langs

                    DetectorFactory.seed = 0
                    candidates = detect_langs(text) if text.strip() else []
                    if candidates:
                        language = str(candidates[0].lang or "unknown").lower()
                        confidence = round(float(candidates[0].prob or 0.0), 3)
                        used_provider = "langdetect"
                        break
                elif provider == "heuristic":
                    language, confidence = _infer_language(text)
                    used_provider = "heuristic"
                    break
            except Exception:
                continue
        detected.append(
            {
                "doc_id": str(row.get("doc_id", "")),
                "language": language,
                "confidence": confidence,
            }
        )
    if not detected:
        return _provider_unavailable_result("lang_id")
    caveats = list(source_caveats)
    if not used_provider:
        caveats.append("No configured language detection provider produced a result.")
    elif used_provider == "heuristic":
        caveats.append("Language detection used explicit lexical heuristic fallback.")
    return ToolExecutionResult(
        payload={"rows": detected},
        caveats=caveats,
        metadata={**source_metadata, **_metadata(used_provider, f"{used_provider}_language_detection")},
    )


def _clean_normalize(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows, source_metadata, source_caveats = _analysis_document_rows_from_deps(
        deps,
        context,
        max_documents=_working_set_analysis_max_documents(),
    )
    if not rows:
        return ToolExecutionResult(
            payload={"documents": [], "rows": [], **source_metadata},
            caveats=source_caveats or ["No input documents were available for clean_normalize."],
            metadata={"no_data": True, "no_data_reason": "no_input_documents", **source_metadata},
        )
    cleaned = []
    for row in rows:
        raw_body_text = str(row.get("text", row.get("cleaned_text", "")) or "").replace("\x00", " ")
        if _looks_machine_payload(raw_body_text):
            text, flags = _clean_analysis_text(row.get("title", ""), raw_body_text)
        else:
            limit = _analysis_text_char_limit()
            flags = {"machine_payload": False, "truncated": False}
            if len(raw_body_text) > limit * 2:
                flags["truncated"] = True
                raw_body_text = raw_body_text[: limit * 2]
            text = _collapse_text(raw_body_text)
            if len(text) > limit:
                flags["truncated"] = True
                text = text[:limit]
        cleaned.append(
            {
                "doc_id": row["doc_id"],
                "title": str(row.get("title", "")),
                "text": text,
                "cleaned_text": text,
                "published_at": str(row.get("published_at", row.get("date", ""))),
                "date": str(row.get("date", row.get("published_at", ""))),
                "source": str(row.get("source", row.get("outlet", ""))),
                "outlet": str(row.get("outlet", row.get("source", ""))),
                "text_is_machine_payload": bool(flags.get("machine_payload")),
                "analysis_text_truncated": bool(flags.get("truncated")),
            }
        )
    return ToolExecutionResult(
        payload={"documents": cleaned, "rows": cleaned, **source_metadata},
        caveats=source_caveats,
        metadata={"no_data": not cleaned, "no_data_reason": "" if cleaned else "no_input_documents", **source_metadata},
    )


def _tokenize_docs(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = _doc_rows(deps)
    if not rows:
        return _no_input_documents_result("tokenize")
    providers = _provider_order("tokenize", ["spacy", "stanza", "nltk", "regex"])
    output = []
    used_provider = ""
    for row in rows:
        text = _row_analysis_text(row)
        tokens: list[str] | None = None
        for provider in providers:
            try:
                if provider == "spacy":
                    nlp = _load_spacy_model()
                    if nlp is None:
                        continue
                    tokens = [token.text for token in nlp.make_doc(text)]
                elif provider == "stanza":
                    pipeline = _load_stanza_pipeline("tokenize")
                    if pipeline is None:
                        continue
                    doc = pipeline(text)
                    tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
                elif provider == "nltk" and _module_available("nltk"):
                    import nltk

                    tokens = nltk.word_tokenize(text)
                elif provider in {"regex", "heuristic"}:
                    tokens = _tokenize(text)
                if tokens is not None:
                    used_provider = provider
                    break
            except Exception:
                tokens = None
        output.append({"doc_id": row["doc_id"], "tokens": tokens or _tokenize(text)})
    return ToolExecutionResult(
        payload={"rows": output},
        metadata=_metadata(used_provider, f"{used_provider}_tokenize"),
    )


def _sentence_split_docs(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = _doc_rows(deps)
    if not rows:
        return _no_input_documents_result("sentence_split")
    providers = _provider_order("sentence_split", ["spacy", "stanza", "nltk"])
    output = []
    used_provider = ""
    for row in rows:
        text = _row_analysis_text(row)
        sentences: list[str] | None = None
        for provider in providers:
            try:
                if provider == "spacy":
                    nlp = _load_spacy_model()
                    if nlp is None:
                        continue
                    doc = nlp(text)
                    if not list(doc.sents):
                        continue
                    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                elif provider == "stanza":
                    pipeline = _load_stanza_pipeline("tokenize")
                    if pipeline is None:
                        continue
                    doc = pipeline(text)
                    sentences = [
                        " ".join(token.text for token in sentence.tokens).strip()
                        for sentence in doc.sentences
                        if sentence.tokens
                    ]
                elif provider == "nltk" and _module_available("nltk"):
                    import nltk

                    sentences = [item.strip() for item in nltk.sent_tokenize(text) if item.strip()]
                elif provider in {"heuristic", "regex"}:
                    sentences = simple_sentence_split(text)
                if sentences is not None:
                    used_provider = provider
                    break
            except Exception:
                sentences = None
        sentence_rows = sentences or []
        output_row = {
            key: row.get(key)
            for key in ("doc_id", "title", "date", "published_at", "outlet", "source", "rank", "score")
            if key in row
        }
        output_row["sentences"] = sentence_rows
        output_row["text"] = " ".join(sentence_rows)
        output.append(output_row)
    if not any(row.get("sentences") for row in output):
        return _provider_unavailable_result("sentence_split")
    return ToolExecutionResult(
        payload={"rows": output},
        metadata=_metadata(used_provider, f"{used_provider}_sentence_split"),
    )


def _pos_morph(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = _doc_rows(deps)
    if not rows:
        return _no_input_documents_result("pos_morph")
    output: list[dict[str, Any]] = []
    providers = _provider_order("pos_morph", ["spacy", "stanza", "flair", "nltk"])
    used_provider = ""
    for provider in providers:
        try:
            if provider == "spacy":
                nlp = _load_spacy_model()
                if nlp is None or "tagger" not in getattr(nlp, "pipe_names", []):
                    continue
                docs = nlp.pipe(
                    [str(row.get("text") or row.get("cleaned_text") or row.get("body") or _row_analysis_text(row)) for row in rows],
                    batch_size=16,
                )
                for row, doc in zip(rows, docs, strict=False):
                    for token in doc:
                        output.append(
                            {
                                "doc_id": str(row.get("doc_id", "")),
                                "token": token.text,
                                "lemma": token.lemma_.lower() or _lemma(token.text),
                                "pos": token.pos_ or _heuristic_pos(token.text),
                                "morph": str(token.morph),
                                "outlet": str(row.get("outlet", row.get("source", ""))),
                                "time_bin": _time_bin(_row_timestamp(row)),
                            }
                        )
                used_provider = "spacy"
                provider_ran = True
                break
            if provider == "stanza":
                pipeline = _load_stanza_pipeline("tokenize,pos,lemma,depparse")
                if pipeline is None:
                    continue
                for row in rows:
                    doc = pipeline(str(row.get("text") or row.get("cleaned_text") or row.get("body") or _row_analysis_text(row)))
                    for sentence in doc.sentences:
                        for word in sentence.words:
                            output.append(
                                {
                                    "doc_id": str(row.get("doc_id", "")),
                                    "token": word.text,
                                    "lemma": (word.lemma or _lemma(word.text)).lower(),
                                    "pos": word.upos or _heuristic_pos(word.text),
                                    "morph": word.feats or "",
                                    "outlet": str(row.get("outlet", row.get("source", ""))),
                                    "time_bin": _time_bin(_row_timestamp(row)),
                                }
                            )
                used_provider = "stanza"
                provider_ran = True
                break
            if provider == "flair":
                tagger = _load_flair_object("pos")
                if tagger is None:
                    continue
                from flair.data import Sentence

                for row in rows:
                    sentence = Sentence(_row_analysis_text(row))
                    tagger.predict(sentence)
                    for token in sentence.tokens:
                        output.append(
                            {
                                "doc_id": str(row.get("doc_id", "")),
                                "token": token.text,
                                "lemma": _lemma(token.text),
                                "pos": token.get_label("upos").value if token.labels else _heuristic_pos(token.text),
                                "morph": "",
                                "outlet": str(row.get("outlet", row.get("source", ""))),
                                "time_bin": _time_bin(_row_timestamp(row)),
                            }
                        )
                used_provider = "flair"
                provider_ran = True
                break
            if provider == "nltk" and _module_available("nltk"):
                import nltk

                for row in rows:
                    tagged = nltk.pos_tag(nltk.word_tokenize(_row_analysis_text(row)))
                    for token, tag in tagged:
                        output.append(
                            {
                                "doc_id": str(row.get("doc_id", "")),
                                "token": token,
                                "lemma": _lemma(token),
                                "pos": tag,
                                "morph": "",
                                "outlet": str(row.get("outlet", row.get("source", ""))),
                                "time_bin": _time_bin(_row_timestamp(row)),
                            }
                        )
                used_provider = "nltk"
                break
        except Exception:
            output = []
            continue

    if not output and "heuristic" in providers:
        for row in rows:
            for token in _tokenize(_row_analysis_text(row)):
                output.append(
                    {
                        "doc_id": str(row.get("doc_id", "")),
                        "token": token,
                        "lemma": _lemma(token),
                        "pos": _heuristic_pos(token),
                        "morph": "",
                        "outlet": str(row.get("outlet", row.get("source", ""))),
                        "time_bin": _time_bin(_row_timestamp(row)),
                    }
                )
        if output:
            used_provider = "heuristic"
    if not output:
        return _provider_unavailable_result("pos_morph")
    return ToolExecutionResult(
        payload={"rows": output},
        metadata=_metadata(used_provider, f"{used_provider}_pos_morph"),
    )


def _lemmatize_docs(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = _doc_rows(deps)
    if not rows:
        return _no_input_documents_result("lemmatize")
    providers = _provider_order("lemmatize", ["spacy", "stanza", "textblob"])
    output = []
    used_provider = ""
    for provider in providers:
        try:
            if provider == "spacy":
                nlp = _load_spacy_model()
                if nlp is None:
                    continue
                docs = nlp.pipe([_row_analysis_text(row) for row in rows], batch_size=16)
                output = [
                    {"doc_id": row["doc_id"], "lemmas": [(token.lemma_ or _lemma(token.text)).lower() for token in doc]}
                    for row, doc in zip(rows, docs, strict=False)
                ]
                used_provider = "spacy"
                break
            if provider == "stanza":
                pipeline = _load_stanza_pipeline("tokenize,lemma")
                if pipeline is None:
                    continue
                output = []
                for row in rows:
                    doc = pipeline(_row_analysis_text(row))
                    output.append(
                        {
                            "doc_id": row["doc_id"],
                            "lemmas": [
                                (word.lemma or _lemma(word.text)).lower()
                                for sentence in doc.sentences
                                for word in sentence.words
                            ],
                        }
                    )
                used_provider = "stanza"
                break
            if provider == "textblob" and _module_available("textblob"):
                from textblob import TextBlob

                output = []
                for row in rows:
                    blob = TextBlob(_row_analysis_text(row))
                    output.append(
                        {
                            "doc_id": row["doc_id"],
                            "lemmas": [word.lemmatize().lower() for word in blob.words],
                        }
                    )
                used_provider = "textblob"
                break
        except Exception:
            output = []
            continue
    if not output and "heuristic" in providers:
        output = [
            {"doc_id": row["doc_id"], "lemmas": [_lemma(token) for token in _tokenize(_row_analysis_text(row))]}
            for row in rows
        ]
        if output:
            used_provider = "heuristic"
    if not output:
        return _provider_unavailable_result("lemmatize")
    return ToolExecutionResult(
        payload={"rows": output},
        metadata=_metadata(used_provider, f"{used_provider}_lemmatize"),
    )


def _dependency_parse(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = _doc_rows(deps)
    if not rows:
        return _no_input_documents_result("dependency_parse")
    providers = _provider_order("dependency_parse", ["spacy", "stanza"])
    parsed: list[dict[str, Any]] = []
    used_provider = ""
    for provider in providers:
        try:
            if provider == "spacy":
                nlp = _load_spacy_model()
                if nlp is None or "parser" not in getattr(nlp, "pipe_names", []):
                    continue
                docs = nlp.pipe([_row_analysis_text(row) for row in rows], batch_size=16)
                for row, doc in zip(rows, docs, strict=False):
                    deps_rows = []
                    for sent in doc.sents:
                        sentence_text = sent.text.strip()
                        for token in sent:
                            deps_rows.append(
                                {
                                    "token": token.text,
                                    "child": token.text,
                                    "child_i": int(token.i),
                                    "head": token.head.text if token.head is not None else "ROOT",
                                    "head_i": int(token.head.i) if token.head is not None else -1,
                                    "dep": token.dep_,
                                    "pos": token.pos_,
                                    "lemma": (token.lemma_ or token.text).lower(),
                                    "sentence": sentence_text,
                                    "is_root": bool(token.head is token),
                                }
                            )
                    parsed.append({"doc_id": row["doc_id"], "dependencies": deps_rows, "provider": provider})
                used_provider = provider
                break
            if provider == "stanza":
                pipeline = _load_stanza_pipeline("tokenize,pos,lemma,depparse")
                if pipeline is None:
                    continue
                for row in rows:
                    doc = pipeline(_row_analysis_text(row))
                    deps_rows = []
                    for sentence in doc.sentences:
                        words = list(sentence.words)
                        sentence_text = " ".join(word.text for word in words).strip()
                        for word in words:
                            head_index = int(word.head or 0)
                            head_text = "ROOT" if head_index <= 0 else words[head_index - 1].text
                            deps_rows.append(
                                {
                                    "token": word.text,
                                    "child": word.text,
                                    "child_i": int(word.id),
                                    "head": head_text,
                                    "head_i": head_index,
                                    "dep": word.deprel or "",
                                    "pos": word.upos or "",
                                    "lemma": (word.lemma or word.text).lower(),
                                    "sentence": sentence_text,
                                    "is_root": head_index <= 0,
                                }
                            )
                    parsed.append({"doc_id": row["doc_id"], "dependencies": deps_rows, "provider": provider})
                used_provider = provider
                break
        except Exception:
            parsed = []
            continue
    if not parsed:
        return _provider_unavailable_result("dependency_parse")
    return ToolExecutionResult(payload={"rows": parsed}, metadata=_metadata(used_provider, f"{used_provider}_dependency_parse"))


def _noun_chunks(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    doc_rows = _doc_rows(deps)
    has_any_input_rows = bool(doc_rows)
    providers = _provider_order("noun_chunks", ["spacy"])
    chunks_by_doc: defaultdict[str, list[str]] = defaultdict(list)
    used_provider = ""
    if doc_rows:
        for provider in providers:
            try:
                if provider == "spacy":
                    nlp = _load_spacy_model()
                    if nlp is None or "parser" not in getattr(nlp, "pipe_names", []):
                        continue
                    docs = nlp.pipe([_row_analysis_text(row) for row in doc_rows], batch_size=16)
                    for row, doc in zip(doc_rows, docs, strict=False):
                        doc_id = str(row.get("doc_id", ""))
                        for chunk in doc.noun_chunks:
                            value = _canonical_entity(chunk.text)
                            if value and _valid_series_surface(value):
                                chunks_by_doc[doc_id].append(value)
                    used_provider = provider
                    break
            except Exception:
                chunks_by_doc.clear()
                continue
    if chunks_by_doc:
        return ToolExecutionResult(
            payload={"rows": [{"doc_id": doc_id, "noun_chunks": values} for doc_id, values in chunks_by_doc.items()]},
            metadata=_metadata(used_provider, f"{used_provider}_noun_chunks"),
        )
    pos_rows = []
    pos_provider = ""
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload and payload["rows"] and "pos" in payload["rows"][0]:
            pos_rows = payload["rows"]
            pos_provider = str(result.metadata.get("provider", "") if isinstance(result.metadata, dict) else "")
            break
    if not pos_rows:
        if not has_any_input_rows:
            return _no_input_documents_result("noun_chunks")
        return _provider_unavailable_result("noun_chunks")
    if "heuristic" in pos_provider and "heuristic" not in providers and "pos_sequence" not in providers:
        return _provider_unavailable_result(
            "noun_chunks",
            caveats=["Only heuristic POS rows were available, so noun chunk fallback was not used."],
            upstream_provider=pos_provider,
        )
    chunks: defaultdict[str, list[str]] = defaultdict(list)
    current: list[str] = []
    current_doc = ""
    for row in pos_rows:
        doc_id = str(row.get("doc_id", ""))
        if current and doc_id != current_doc:
            chunks[current_doc].append(" ".join(current))
            current = []
        current_doc = doc_id
        if str(row.get("pos", "")) in {"NOUN", "PROPN", "ADJ"}:
            current.append(str(row.get("lemma", row.get("token", ""))))
        elif current:
            chunks[current_doc].append(" ".join(current))
            current = []
    if current:
        chunks[current_doc].append(" ".join(current))
    if not chunks:
        return _provider_unavailable_result("noun_chunks", upstream_provider=pos_provider)
    return ToolExecutionResult(
        payload={"rows": [{"doc_id": doc_id, "noun_chunks": values} for doc_id, values in chunks.items()]},
        caveats=["Used POS-sequence fallback for noun chunks because parser-backed noun chunks were unavailable."],
        metadata=_metadata("pos_sequence", "pos_sequence_noun_chunks", upstream_provider=pos_provider),
    )


def _ner(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows, source_metadata, source_caveats = _analysis_document_rows_from_deps(
        deps,
        context,
        max_documents=_entity_analysis_max_documents(),
    )
    if not rows:
        return _no_input_documents_result("ner", caveats=source_caveats, **source_metadata)
    entities: list[dict[str, Any]] = []
    cancelled = _cancel_requested(context)
    providers = _provider_order("ner", ["spacy", "stanza", "flair"])
    provider_limit = _entity_provider_max_documents()
    provider_order_explicit = os.getenv("CORPUSAGENT2_PROVIDER_ORDER_NER") is not None
    working_set_count = int(source_metadata.get("working_set_document_count") or source_metadata.get("analyzed_document_count") or len(rows))
    if provider_limit is not None and working_set_count > provider_limit and not provider_order_explicit:
        source_caveats.append(
            f"Provider NER is running despite large working set: working_set_document_count {working_set_count} "
            f"exceeds CORPUSAGENT2_ENTITY_PROVIDER_MAX_DOCS={provider_limit}. "
            "Heuristic regex NER is only used when explicitly requested via CORPUSAGENT2_PROVIDER_ORDER_NER."
        )
    used_provider = ""
    provider_ran = False

    def finish() -> ToolExecutionResult:
        caveats = list(source_caveats)
        metadata = {**_metadata(used_provider, f"{used_provider}_ner"), **source_metadata}
        if cancelled:
            caveats.append("Run abort was requested; NER stopped before all documents were processed.")
            metadata.update(
                {
                    "cancelled": True,
                    "no_data": not entities,
                    "no_data_reason": "cancelled" if not entities else "",
                }
            )
        return ToolExecutionResult(
            payload={"rows": entities},
            caveats=caveats,
            metadata=metadata,
        )

    if cancelled:
        return finish()

    for provider in providers:
        try:
            if provider == "spacy":
                nlp = _load_spacy_model()
                if nlp is None or "ner" not in getattr(nlp, "pipe_names", []):
                    continue
                docs = nlp.pipe([_row_analysis_text(row) for row in rows], batch_size=16)
                for row, doc in zip(rows, docs, strict=False):
                    if _cancel_requested(context):
                        cancelled = True
                        return finish()
                    for ent in doc.ents:
                        entities.append(
                            {
                                "doc_id": str(row.get("doc_id", "")),
                                "entity": ent.text.strip(),
                                "entity_text": ent.text.strip(),
                                "label": ent.label_,
                                "outlet": str(row.get("outlet", row.get("source", ""))),
                                "time_bin": _time_bin(_row_timestamp(row)),
                                "published_at": _row_timestamp(row),
                            }
                        )
                used_provider = "spacy"
                provider_ran = True
                break
            if provider == "stanza":
                pipeline = _load_stanza_pipeline("tokenize,ner")
                if pipeline is None:
                    continue
                for row in rows:
                    if _cancel_requested(context):
                        cancelled = True
                        return finish()
                    doc = pipeline(_row_analysis_text(row))
                    for ent in getattr(doc, "ents", []):
                        entities.append(
                            {
                                "doc_id": str(row.get("doc_id", "")),
                                "entity": ent.text.strip(),
                                "entity_text": ent.text.strip(),
                                "label": ent.type,
                                "outlet": str(row.get("outlet", row.get("source", ""))),
                                "time_bin": _time_bin(_row_timestamp(row)),
                                "published_at": _row_timestamp(row),
                            }
                        )
                used_provider = "stanza"
                provider_ran = True
                break
            if provider == "flair":
                tagger = _load_flair_object("ner")
                if tagger is None:
                    continue
                from flair.data import Sentence

                for row in rows:
                    if _cancel_requested(context):
                        cancelled = True
                        return finish()
                    sentence = Sentence(_row_analysis_text(row))
                    tagger.predict(sentence)
                    for entity in sentence.get_spans("ner"):
                        entities.append(
                            {
                                "doc_id": str(row.get("doc_id", "")),
                                "entity": entity.text.strip(),
                                "entity_text": entity.text.strip(),
                                "label": entity.get_label("ner").value,
                                "outlet": str(row.get("outlet", row.get("source", ""))),
                                "time_bin": _time_bin(_row_timestamp(row)),
                                "published_at": _row_timestamp(row),
                            }
                        )
                used_provider = "flair"
                provider_ran = True
                break
        except Exception:
            entities = []
            continue

    if not entities and "regex" in providers:
        for row in rows:
            if _cancel_requested(context):
                cancelled = True
                return finish()
            for match in ENTITY_PATTERN.finditer(_row_analysis_text(row)):
                entities.append(
                    {
                        "doc_id": str(row.get("doc_id", "")),
                        "entity": match.group(0).strip(),
                        "entity_text": match.group(0).strip(),
                        "label": "ENTITY",
                        "outlet": str(row.get("outlet", row.get("source", ""))),
                        "time_bin": _time_bin(_row_timestamp(row)),
                        "published_at": _row_timestamp(row),
                    }
                )
        if entities:
            used_provider = "regex"
            provider_ran = True
    if not entities and provider_ran:
        return ToolExecutionResult(
            payload={"rows": [], **source_metadata},
            caveats=source_caveats,
            metadata={
                **_metadata(used_provider, f"{used_provider}_ner"),
                "no_data": True,
                "no_data_reason": "no_entities_extracted",
                **source_metadata,
            },
        )
    if not entities:
        return _provider_unavailable_result("ner", caveats=source_caveats, **source_metadata)
    return finish()


def _entity_link(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    target_groups = _target_alias_groups(params)
    entity_rows = []
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload and payload["rows"]:
            first = payload["rows"][0]
            if "entity" in first:
                entity_rows = list(payload["rows"])
                break
    if not entity_rows:
        ner_result = _ner(params, deps, context)
        entity_rows = ner_result.payload["rows"]
        if not entity_rows:
            upstream_reason = ""
            if isinstance(ner_result.metadata, dict):
                upstream_reason = str(ner_result.metadata.get("no_data_reason", ""))
            if upstream_reason == "no_input_documents":
                return _no_input_documents_result("entity_link", caveats=ner_result.caveats)
            return ToolExecutionResult(
                payload={"rows": []},
                caveats=list(ner_result.caveats or ["No entities were available to link."]),
                metadata={
                    "no_data": True,
                    "no_data_reason": upstream_reason or "no_entities_to_link",
                    "provider": "string_canonicalization",
                },
            )
    linked = []
    for row in entity_rows:
        link_payload = _link_entity_row(str(row.get("entity", "")), str(row.get("label", "")))
        canonical = str(link_payload.get("entity", "")).strip()
        target_canonical, matched_alias = _match_target_alias(
            " ".join(
                str(row.get(field, "") or "")
                for field in ("entity", "entity_text", "canonical_entity", "linked_entity")
            ),
            target_groups,
        )
        if target_canonical:
            canonical = target_canonical
        linked.append(
            {
                "doc_id": str(row.get("doc_id", "")),
                **link_payload,
                "canonical_entity": canonical,
                "linked_entity": canonical,
                "target_entity": canonical if target_canonical else str(row.get("target_entity", "")),
                "series_name": canonical,
                "matched_alias": matched_alias,
                "entity_text": str(row.get("entity_text", row.get("entity", ""))),
                "outlet": str(row.get("outlet", "")),
                "time_bin": str(row.get("time_bin", "")),
                "published_at": str(row.get("published_at", row.get("date", ""))),
            }
        )
    return ToolExecutionResult(
        payload={"rows": linked},
        caveats=[
            "Entity linking currently canonicalizes entity strings only; no external knowledge-base resolver is configured."
        ],
        metadata=_metadata("string_canonicalization", "entity_string_canonicalization"),
    )


def _extract_keyterms(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows, source_metadata, source_caveats = _analysis_document_rows_from_deps(
        deps,
        context,
        max_documents=_working_set_analysis_max_documents(),
    )
    requested_top_k = _int_param(params, "top_k", "limit", default=25, maximum=500)
    group_by = str(params.get("group_by", params.get("group_field", "")) or "").strip()

    def _keyterms_for_bucket(bucket_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        joined = "\n".join(_row_analysis_text(row) for row in bucket_rows)
        bucket_keyterms = []
        seen: set[str] = set()
        for term, score in textrank_keywords(joined, top_k=max(requested_top_k * 4, 25)):
            normalized = str(term or "").strip().lower()
            if normalized in seen or not _valid_noun_lemma(normalized, min_length=3):
                continue
            seen.add(normalized)
            bucket_keyterms.append({"term": normalized, "score": float(score)})
            if len(bucket_keyterms) >= requested_top_k:
                break
        return bucket_keyterms

    if group_by:
        buckets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            group_value = str(_first_nonempty_field(row, [group_by, "outlet" if group_by == "source" else "", "source" if group_by == "outlet" else ""]) or "").strip()
            if not group_value:
                group_value = "__unknown__"
            buckets[group_value].append(row)
        grouped_rows: list[dict[str, Any]] = []
        for group_value, bucket_rows in sorted(buckets.items(), key=lambda item: item[0].lower()):
            for rank, item in enumerate(_keyterms_for_bucket(bucket_rows), start=1):
                grouped_rows.append(
                    {
                        **item,
                        "rank": rank,
                        group_by: group_value,
                        "source": group_value if group_by == "source" else group_value if group_by == "outlet" else str(bucket_rows[0].get("source", "")),
                        "outlet": group_value if group_by == "outlet" else str(bucket_rows[0].get("outlet", bucket_rows[0].get("source", ""))),
                        "document_count": len(bucket_rows),
                    }
                )
        return ToolExecutionResult(
            payload={"rows": grouped_rows, **source_metadata},
            caveats=source_caveats,
            metadata={"no_data": not grouped_rows, "group_by": group_by, **source_metadata},
        )

    keyterms = []
    for rank, item in enumerate(_keyterms_for_bucket(rows), start=1):
        keyterms.append({**item, "rank": rank, "document_count": len(rows)})
    return ToolExecutionResult(
        payload={"rows": keyterms, **source_metadata},
        caveats=source_caveats,
        metadata={"no_data": not keyterms, **source_metadata},
    )


SVO_VERB_LEMMAS = {
    "arrest", "arrested", "attack", "attacked", "beat", "beaten", "block", "blocked", "call", "called",
    "charge", "charged", "clash", "clashed", "clear", "cleared", "criticize", "criticized", "damage", "damaged",
    "demand", "demanded", "detain", "detained", "fire", "fired", "force", "forced", "grapple", "grappled",
    "haul", "hauled", "hit", "hold", "held", "injure", "injured", "kill", "killed", "march", "marched",
    "meet", "met", "name", "named", "organize", "organized", "oppose", "opposed", "push", "pushed",
    "release", "released", "remove", "removed", "resist", "resisted", "say", "said", "shoot", "shot",
    "strike", "struck", "support", "supported", "target", "targeted", "throw", "threw", "use", "used",
    "vandalize", "vandalized", "warn", "warned",
}
SVO_AUXILIARIES = {
    "am", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "will", "would",
    "can", "could", "may", "might", "must", "shall", "should",
}
SVO_PHRASE_BREAKERS = SERIES_SURFACE_STOPWORDS | SVO_AUXILIARIES | {
    "ed", "edt", "gmt", "pst", "pm", "am", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
}
SVO_ROLE_GROUP_PATTERNS = (
    (
        "police",
        re.compile(
            r"\b(?:police|officers?|cops?|law enforcement|sheriff(?:s)?|troopers?|nypd|lapd|national guard|guards?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "protesters",
        re.compile(
            r"\b(?:protesters?|protestors?|demonstrators?|activists?|marchers?|rioters?|crowd|crowds|campaigners?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "politicians",
        re.compile(
            r"\b(?:president|mayor|governor|senators?|lawmakers?|congress(?:man|woman|person|people)?|politicians?|trump|biden|democrats?|republicans?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "institutions",
        re.compile(
            r"\b(?:government|court|department|agency|administration|parliament|congress|white house|city council|ice|fbi|justice department|prosecutors?|authorities|officials?)\b",
            re.IGNORECASE,
        ),
    ),
)


def _svo_role_group(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "other"
    for label, pattern in SVO_ROLE_GROUP_PATTERNS:
        if pattern.search(text):
            return label
    return "other"


def _clean_svo_phrase(tokens: list[str]) -> str:
    cleaned: list[str] = []
    for token in tokens:
        value = str(token or "").strip(" \t\r\n\"',;:()[]{}")
        if not value:
            continue
        lowered = value.lower()
        if lowered in SVO_PHRASE_BREAKERS:
            if cleaned:
                break
            continue
        if lowered in SVO_VERB_LEMMAS and cleaned:
            break
        cleaned.append(value)
        if len(cleaned) >= 5:
            break
    phrase = _canonical_entity(" ".join(cleaned))
    if not phrase or not _valid_series_surface(phrase):
        return ""
    return phrase


def _svo_phrase_before(tokens: list[str], verb_index: int) -> str:
    chunk: list[str] = []
    for token in reversed(tokens[:verb_index]):
        lowered = str(token or "").lower()
        if lowered in SVO_PHRASE_BREAKERS or lowered in SVO_VERB_LEMMAS:
            if chunk:
                break
            continue
        chunk.append(str(token))
        if len(chunk) >= 5:
            break
    return _clean_svo_phrase(list(reversed(chunk)))


def _svo_phrase_after(tokens: list[str], start_index: int) -> str:
    return _clean_svo_phrase([str(token) for token in tokens[start_index: start_index + 8]])


def _svo_by_agent_phrase(tokens: list[str], verb_index: int) -> str:
    for idx in range(verb_index + 1, min(len(tokens) - 1, verb_index + 10)):
        if str(tokens[idx]).lower() == "by":
            return _svo_phrase_after(tokens, idx + 1)
    return ""


def _svo_is_passive(tokens: list[str], verb_index: int) -> bool:
    previous = {str(token).lower() for token in tokens[max(0, verb_index - 3): verb_index]}
    return bool(previous & {"was", "were", "is", "are", "been", "being"})


def _svo_verb_index(tokens: list[str]) -> int:
    for idx in range(1, max(1, len(tokens) - 1)):
        lowered = str(tokens[idx]).lower()
        if lowered in SVO_AUXILIARIES or lowered in SVO_PHRASE_BREAKERS:
            continue
        if lowered in SVO_VERB_LEMMAS:
            return idx
        if lowered.endswith(("ed", "ing")) and _svo_phrase_before(tokens, idx):
            return idx
    return -1


def _extract_svo_triples(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows, source_metadata, source_caveats = _analysis_document_rows_from_deps(
        deps,
        context,
        max_documents=_working_set_analysis_max_documents(),
    )
    if not rows:
        return _no_input_documents_result("extract_svo_triples", caveats=source_caveats, **source_metadata)
    triples: list[dict[str, Any]] = []
    seen_triples: set[tuple[str, str, str, str, str, str]] = set()
    skipped_sentences = 0
    max_sentences = _int_param(params, "max_sentences_per_doc", default=12, minimum=1, maximum=100)
    providers = _provider_order("extract_svo_triples", ["spacy", "stanza"])
    used_provider = ""
    provider_ran = False

    def add_triple(
        row: dict[str, Any],
        *,
        sentence: str,
        subject: str,
        verb: str,
        obj: str,
        passive: bool,
        by_agent: str = "",
        provider: str,
    ) -> None:
        subject = _canonical_entity(subject)
        obj = _canonical_entity(obj)
        by_agent = _canonical_entity(by_agent)
        if not subject or not _valid_series_surface(subject):
            return
        semantic_actor = by_agent if passive and by_agent else subject
        semantic_target = subject if passive else obj
        if not semantic_target:
            semantic_target = obj or subject
        if not semantic_actor or not semantic_target:
            return
        dedupe_key = (
            str(row.get("doc_id", "") or ""),
            sentence,
            subject,
            str(verb or "").lower(),
            obj,
            semantic_actor,
        )
        if dedupe_key in seen_triples:
            return
        seen_triples.add(dedupe_key)
        triples.append(
            {
                "doc_id": str(row.get("doc_id", "") or ""),
                "published_at": row.get("published_at") or row.get("date") or "",
                "date": row.get("date") or row.get("published_at") or "",
                "source": row.get("source") or row.get("outlet") or "",
                "sentence": sentence,
                "subject": subject,
                "verb": str(verb or "").lower(),
                "object": obj,
                "voice": "passive" if passive else "active",
                "semantic_actor": semantic_actor,
                "semantic_target": semantic_target,
                "subject_group": _svo_role_group(subject),
                "object_group": _svo_role_group(obj),
                "actor_group": _svo_role_group(semantic_actor),
                "target_group": _svo_role_group(semantic_target),
                "mention_count": 1,
                "count": 1,
                "provider": provider,
            }
        )

    def spacy_phrase(token: Any) -> str:
        left = min(item.i for item in token.subtree)
        right = max(item.i for item in token.subtree)
        doc = token.doc
        return doc[left : right + 1].text

    def spacy_conjuncts(token: Any) -> list[Any]:
        items = [token]
        items.extend(child for child in token.children if child.dep_ == "conj")
        return items

    for provider in providers:
        try:
            if provider == "spacy":
                nlp = _load_spacy_model()
                if nlp is None or "parser" not in getattr(nlp, "pipe_names", []):
                    continue
                provider_ran = True
                used_provider = "spacy"
                docs = nlp.pipe([_row_analysis_text(row) for row in rows], batch_size=16)
                for row, doc in zip(rows, docs, strict=False):
                    for sent in list(doc.sents)[:max_sentences]:
                        sentence_text = sent.text.strip()
                        sentence_matched = False
                        for verb_token in sent:
                            if verb_token.pos_ not in {"VERB", "AUX"} or verb_token.dep_ in {"aux", "auxpass"}:
                                continue
                            subjects = [
                                item
                                for child in verb_token.children
                                if child.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}
                                for item in spacy_conjuncts(child)
                            ]
                            objects = [
                                item
                                for child in verb_token.children
                                if child.dep_ in {"dobj", "obj", "iobj", "attr", "oprd", "dative"}
                                for item in spacy_conjuncts(child)
                            ]
                            if not objects:
                                objects.extend(
                                    item
                                    for prep in verb_token.children
                                    if prep.dep_ == "prep"
                                    for child in prep.children
                                    if child.dep_ in {"pobj", "obj"} and str(prep.text).lower() not in {"in", "on", "at", "near", "outside", "inside", "during", "before", "after"}
                                    for item in spacy_conjuncts(child)
                                )
                            agents = [
                                item
                                for child in verb_token.children
                                if child.dep_ == "agent"
                                for pobj in child.children
                                if pobj.dep_ in {"pobj", "obj"}
                                for item in spacy_conjuncts(pobj)
                            ]
                            passive = any(child.dep_ == "auxpass" for child in verb_token.children) or any(
                                subject.dep_ in {"nsubjpass", "csubjpass"} for subject in subjects
                            )
                            if not subjects:
                                continue
                            if not objects and not agents and not passive:
                                continue
                            for subject in subjects:
                                selected_objects = objects or agents or [subject]
                                for obj_token in selected_objects:
                                    add_triple(
                                        row,
                                        sentence=sentence_text,
                                        subject=spacy_phrase(subject),
                                        verb=verb_token.lemma_ or verb_token.text,
                                        obj=spacy_phrase(obj_token),
                                        passive=passive,
                                        by_agent=spacy_phrase(agents[0]) if agents else "",
                                        provider="spacy",
                                    )
                                    sentence_matched = True
                        if not sentence_matched:
                            skipped_sentences += 1
                if triples:
                    break
            elif provider == "stanza":
                pipeline = _load_stanza_pipeline("tokenize,pos,lemma,depparse")
                if pipeline is None:
                    continue
                provider_ran = True
                used_provider = "stanza"
                for row in rows:
                    doc = pipeline(_row_analysis_text(row))
                    for sentence in doc.sentences[:max_sentences]:
                        words = list(sentence.words)
                        by_id = {int(word.id): word for word in words}
                        sentence_text = " ".join(word.text for word in words).strip()
                        sentence_matched = False
                        for word in words:
                            if str(word.upos or "") not in {"VERB", "AUX"}:
                                continue
                            children = [candidate for candidate in words if int(candidate.head or 0) == int(word.id)]
                            subjects = [child for child in children if str(child.deprel or "") in {"nsubj", "nsubj:pass", "csubj", "csubj:pass"}]
                            objects = [child for child in children if str(child.deprel or "") in {"obj", "iobj", "obl"}]
                            agents = [
                                child
                                for child in children
                                if str(child.deprel or "") in {"obl:agent", "agent"} or str(child.text).lower() == "by"
                            ]
                            passive = any(":pass" in str(child.deprel or "") for child in subjects + children)
                            if not subjects:
                                continue
                            selected_objects = objects or agents
                            if not selected_objects and not passive:
                                continue
                            for subject in subjects:
                                for obj_word in selected_objects or [subject]:
                                    add_triple(
                                        row,
                                        sentence=sentence_text,
                                        subject=subject.text,
                                        verb=word.lemma or word.text,
                                        obj=obj_word.text,
                                        passive=passive,
                                        by_agent=agents[0].text if agents else "",
                                        provider="stanza",
                                    )
                                    sentence_matched = True
                        if not sentence_matched:
                            skipped_sentences += 1
                if triples:
                    break
        except Exception:
            triples = []
            continue

    if not triples and "heuristic" in providers:
        provider_ran = True
        used_provider = "heuristic"
        for row in rows:
            doc_id = str(row.get("doc_id", "") or "")
            analysis_text = str(row.get("text") or row.get("cleaned_text") or row.get("body") or _row_analysis_text(row))
            for sentence in simple_sentence_split(analysis_text)[:max_sentences]:
                tokens = _tokenize(sentence)
                if len(tokens) < 3:
                    skipped_sentences += 1
                    continue
                verb_index = _svo_verb_index(tokens)
                if verb_index <= 0:
                    skipped_sentences += 1
                    continue
                subject = _svo_phrase_before(tokens, verb_index)
                obj = _svo_phrase_after(tokens, verb_index + 1)
                if not subject:
                    skipped_sentences += 1
                    continue
                by_agent = _svo_by_agent_phrase(tokens, verb_index)
                add_triple(
                    row,
                    sentence=sentence,
                    subject=subject,
                    verb=str(tokens[verb_index]).lower(),
                    obj=obj,
                    passive=_svo_is_passive(tokens, verb_index),
                    by_agent=by_agent,
                    provider="heuristic",
                )
    caveats = list(source_caveats)
    if skipped_sentences:
        caveats.append(f"Skipped {skipped_sentences} sentences without a usable subject-verb-object pattern.")
    if not triples:
        if provider_ran:
            return ToolExecutionResult(
                payload={"rows": [], "skipped_sentence_count": skipped_sentences, **source_metadata},
                caveats=caveats,
                metadata={
                    "provider": used_provider,
                    "no_data": True,
                    "no_data_reason": "no_svo_triples_extracted",
                    **source_metadata,
                },
            )
        return _provider_unavailable_result("extract_svo_triples", caveats=caveats, **source_metadata)
    return ToolExecutionResult(
        payload={"rows": triples, "skipped_sentence_count": skipped_sentences, **source_metadata},
        caveats=caveats,
        metadata={"provider": used_provider or str(triples[0].get("provider", "")), "no_data": False, "no_data_reason": "", **source_metadata},
    )


def _topic_model(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows, source_metadata, source_caveats = _analysis_document_rows_from_deps(
        deps,
        context,
        max_documents=_topic_model_analysis_max_documents(),
    )
    if not rows:
        return _no_input_documents_result("topic_model", caveats=source_caveats, **source_metadata)
    providers = _provider_order("topic_model", ["textacy", "gensim"])
    payload: list[dict[str, Any]] = []
    used_provider = ""
    texts = [_row_analysis_text(row) for row in rows]
    num_topics = int(params.get("num_topics", 4))
    granularity = str(params.get("granularity", _default_time_granularity())).strip().lower() or "month"
    topics_per_bin = max(int(params.get("topics_per_bin", 1)), 1)
    bucket_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket_rows[_time_bin(_row_timestamp(row), granularity)].append(row)

    for provider in providers:
        try:
            if provider == "textacy" and _module_available("textacy"):
                import textacy.preprocessing as tprep
                from sklearn.decomposition import NMF
                from sklearn.feature_extraction.text import CountVectorizer

                topic_counter = 1
                for time_bin, bucket in sorted(bucket_rows.items()):
                    cleaned = [
                        tprep.normalize.whitespace(tprep.remove.punctuation(_row_analysis_text(item).lower()))
                        for item in bucket
                        if _row_analysis_text(item).strip()
                    ]
                    if not cleaned:
                        continue
                    vectorizer = CountVectorizer(stop_words="english", max_features=1000)
                    matrix = vectorizer.fit_transform(cleaned)
                    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
                        continue
                    n_components = min(num_topics, topics_per_bin, max(1, matrix.shape[0]), max(1, matrix.shape[1]))
                    model = NMF(n_components=n_components, init="nndsvda", random_state=42)
                    weights = model.fit_transform(matrix)
                    vocab = vectorizer.get_feature_names_out()
                    for idx, component in enumerate(model.components_, start=1):
                        top_indices = component.argsort()[::-1][:20]
                        raw_terms = [str(vocab[item]) for item in top_indices]
                        cleaned_terms = clean_topic_terms(raw_terms, max_count=10, min_length=3)
                        payload.append(
                            {
                                "topic_id": topic_counter,
                                "time_bin": time_bin,
                                "top_terms": cleaned_terms,
                                "weight": float(weights[:, idx - 1].sum()),
                            }
                        )
                        topic_counter += 1
                used_provider = "textacy"
                break
            if provider == "gensim" and _module_available("gensim"):
                from gensim import corpora
                from gensim.models import LdaModel

                topic_counter = 1
                for time_bin, bucket in sorted(bucket_rows.items()):
                    tokenized = [
                        [
                            token.lower()
                            for token in _tokenize(_row_analysis_text(item))
                            if token.lower() not in TOPIC_LABEL_STOPWORDS and len(token) >= 3
                        ]
                        for item in bucket
                    ]
                    dictionary = corpora.Dictionary(tokenized)
                    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
                    if not corpus or len(dictionary) == 0:
                        continue
                    n_topics = min(num_topics, topics_per_bin, max(1, len(dictionary)))
                    model = LdaModel(
                        corpus=corpus,
                        id2word=dictionary,
                        num_topics=n_topics,
                        random_state=42,
                        iterations=50,
                        passes=5,
                    )
                    for topic_id in range(model.num_topics):
                        raw_terms = [term for term, _ in model.show_topic(topic_id, topn=20)]
                        cleaned_terms = clean_topic_terms(raw_terms, max_count=10, min_length=3)
                        payload.append(
                            {
                                "topic_id": topic_counter,
                                "time_bin": time_bin,
                                "top_terms": cleaned_terms,
                                "weight": float(sum(weight for _, weight in model.get_topic_terms(topic_id, topn=20))),
                            }
                        )
                        topic_counter += 1
                used_provider = "gensim"
                break
        except Exception:
            payload = []
            continue

    if not payload and "heuristic" in providers:
        grouped: defaultdict[str, Counter] = defaultdict(Counter)
        for row in rows:
            time_bin = _time_bin(_row_timestamp(row), granularity)
            for token in _tokenize(_row_analysis_text(row).lower()):
                if token in TOPIC_LABEL_STOPWORDS or len(token) < 3:
                    continue
                grouped[time_bin][token] += 1
        for idx, (time_bin, counts) in enumerate(sorted(grouped.items()), start=1):
            raw_terms = [term for term, _ in counts.most_common(20)]
            cleaned = clean_topic_terms(raw_terms, max_count=10, min_length=3)
            if not cleaned:
                continue
            payload.append(
                {
                    "topic_id": idx,
                    "time_bin": time_bin,
                    "top_terms": cleaned,
                    "weight": float(sum(counts.values())),
                }
            )
        if payload:
            used_provider = "heuristic"
    if not payload:
        return _provider_unavailable_result("topic_model", caveats=source_caveats, **source_metadata)
    return ToolExecutionResult(
        payload={"rows": payload, **source_metadata},
        caveats=source_caveats,
        metadata={**_metadata(used_provider, f"{used_provider}_topic_model"), "no_data": not payload, **source_metadata},
    )


def _readability_stats(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = []
    for doc in _doc_rows(deps):
        text = _row_analysis_text(doc)
        sentences = simple_sentence_split(text)
        tokens = _tokenize(text)
        avg_sentence_len = len(tokens) / max(len(sentences), 1)
        avg_word_len = sum(len(token) for token in tokens) / max(len(tokens), 1)
        rows.append({"doc_id": doc["doc_id"], "avg_sentence_len": avg_sentence_len, "avg_word_len": avg_word_len})
    return ToolExecutionResult(payload={"rows": rows})


def _lexical_diversity(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = []
    for doc in _doc_rows(deps):
        tokens = [token.lower() for token in _tokenize(_row_analysis_text(doc))]
        rows.append({"doc_id": doc["doc_id"], "type_token_ratio": len(set(tokens)) / max(len(tokens), 1)})
    return ToolExecutionResult(payload={"rows": rows})


def _extract_ngrams(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    params = _payload_or_params(params)
    raw_n = params.get("n", 2)
    raw_values = raw_n if isinstance(raw_n, list) else [raw_n]
    n_values: list[int] = []
    for value in raw_values:
        try:
            n_value = int(value)
        except (TypeError, ValueError):
            continue
        if n_value >= 1:
            n_values.append(n_value)
    if not n_values:
        n_values = [2]
    n_values = list(dict.fromkeys(n_values))[:4]
    counts: Counter = Counter()
    for doc in _doc_rows(deps):
        tokens = [token.lower() for token in _tokenize(_row_analysis_text(doc)) if token.lower() not in STOPWORDS]
        for n in n_values:
            for idx in range(len(tokens) - n + 1):
                counts[(n, " ".join(tokens[idx : idx + n]))] += 1
    return ToolExecutionResult(
        payload={
            "rows": [
                {"n": int(n), "ngram": ngram, "count": int(count)}
                for (n, ngram), count in counts.most_common(25)
            ]
        }
    )


def _extract_acronyms(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    pattern = re.compile(r"\b[A-Z]{2,8}\b")
    counts: Counter = Counter()
    for doc in _doc_rows(deps):
        counts.update(match.group(0) for match in pattern.finditer(_row_analysis_text(doc)))
    return ToolExecutionResult(
        payload={"rows": [{"acronym": item, "count": int(count)} for item, count in counts.most_common(20)]}
    )


def _raw_list_param(params: dict[str, Any], *names: str) -> list[str]:
    values: list[str] = []
    for name in names:
        raw = params.get(name)
        if raw in (None, ""):
            continue
        if isinstance(raw, str):
            quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', raw)
            for left, right in quoted:
                value = (left or right).strip()
                if value:
                    values.append(value)
            if name == "query_focus":
                values.extend(re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", raw))
                continue
            values.extend(token for token in re.split(r"[,;/|]|\bOR\b|\bAND\b", raw, flags=re.IGNORECASE) if token.strip())
        elif isinstance(raw, list):
            values.extend(str(item).strip() for item in raw if str(item).strip())
    return list(dict.fromkeys(_collapse_text(value) for value in values if _collapse_text(value)))


def _focus_terms(params: dict[str, Any]) -> list[str]:
    terms = _raw_list_param(
        params,
        "focus_terms",
        "claim_focus_terms",
        "context_keywords",
        "keywords",
        "claim_keywords",
        "query_focus",
    )
    cleaned: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not term:
            continue
        if len(term) > 80:
            term_tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", term) if token.lower() not in STOPWORDS]
            candidates = term_tokens
        else:
            candidates = [term]
        for candidate in candidates:
            normalized = candidate.strip(" \"'.,;:()[]{}")
            lowered = normalized.lower()
            if len(normalized) < 3 or lowered in STOPWORDS or lowered in TEMPLATE_TERM_STOPWORDS:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(normalized)
    return cleaned


def _context_window_for_alias(text: str, aliases: list[str], *, window_chars: int) -> tuple[str, str]:
    haystack = str(text or "")
    best_alias = ""
    best_start: int | None = None
    for alias in sorted((alias for alias in aliases if alias), key=len, reverse=True):
        match = re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", haystack, flags=re.IGNORECASE)
        if match:
            best_alias = alias
            best_start = match.start()
            break
    if best_start is None:
        return "", ""
    start = max(0, best_start - window_chars)
    end = min(len(haystack), best_start + len(best_alias) + window_chars)
    return _collapse_text(haystack[start:end]), best_alias


def _targeted_text_units(
    docs: list[dict[str, Any]],
    params: dict[str, Any],
    context: AgentExecutionContext,
) -> list[dict[str, Any]]:
    target_groups = _target_alias_groups(params, context)
    if not target_groups:
        return [
            {
                "doc": doc,
                "text": _row_analysis_text(doc),
                "target_entity": str(doc.get("target_entity", doc.get("entity_label", "")) or ""),
                "matched_alias": str(doc.get("matched_alias", "") or ""),
                "matched_focus_terms": [],
            }
            for doc in docs
        ]
    focus_terms = _focus_terms(params)
    window_chars = _int_param(params, "window_chars", "context_window_chars", default=500, maximum=5000)
    units: list[dict[str, Any]] = []
    for doc in docs:
        full_text = _row_analysis_text(doc)
        lowered_full = full_text.lower()
        for canonical, aliases in target_groups:
            window, matched_alias = _context_window_for_alias(full_text, aliases, window_chars=window_chars)
            if not window:
                continue
            lowered_window = window.lower()
            matched_focus = [
                term
                for term in focus_terms
                if term.lower() in lowered_window or term.lower() in lowered_full
            ]
            if focus_terms and not matched_focus:
                continue
            units.append(
                {
                    "doc": doc,
                    "text": window,
                    "target_entity": canonical,
                    "entity_label": canonical,
                    "series_name": canonical,
                    "matched_alias": matched_alias,
                    "matched_focus_terms": matched_focus,
                }
            )
    if not units:
        return [
            {
                "doc": doc,
                "text": _row_analysis_text(doc),
                "target_entity": str(doc.get("target_entity", doc.get("entity_label", "")) or ""),
                "matched_alias": str(doc.get("matched_alias", "") or ""),
                "matched_focus_terms": [],
                "target_context_fallback": True,
            }
            for doc in docs
        ]
    return units


def _sentiment(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = []
    docs, source_metadata, source_caveats = _analysis_document_rows_from_deps(
        deps,
        context,
        max_documents=_sentiment_analysis_max_documents(),
    )
    if not docs:
        return _no_input_documents_result("sentiment", caveats=source_caveats, **source_metadata)
    units = _targeted_text_units(docs, params, context)
    if any(unit.get("target_context_fallback") for unit in units):
        source_caveats.append(
            "No target-local context windows matched the inferred aliases, so sentiment fell back to full document text."
        )
    providers = _provider_order("sentiment", ["flair", "textblob"])
    used_provider = ""
    cancelled = _cancel_requested(context)

    def finish() -> ToolExecutionResult:
        caveats = list(source_caveats)
        metadata = {**_metadata(used_provider, f"{used_provider}_sentiment"), "no_data": not rows, **source_metadata}
        if cancelled:
            caveats.append("Run abort was requested; sentiment analysis stopped before all documents were processed.")
            metadata.update(
                {
                    "cancelled": True,
                    "no_data": not rows,
                    "no_data_reason": "cancelled" if not rows else "",
                }
            )
        return ToolExecutionResult(
            payload={"rows": rows, **source_metadata},
            caveats=caveats,
            metadata=metadata,
        )

    if cancelled:
        return finish()

    for provider in providers:
        try:
            if provider == "flair":
                classifier = _load_flair_object("sentiment")
                if classifier is None:
                    continue
                from flair.data import Sentence

                rows = []
                for unit in units:
                    if _cancel_requested(context):
                        cancelled = True
                        return finish()
                    doc = unit["doc"]
                    sentence = Sentence(str(unit.get("text", ""))[:1500])
                    classifier.predict(sentence)
                    label = sentence.labels[0].value.lower() if sentence.labels else "neutral"
                    confidence = float(sentence.labels[0].score) if sentence.labels else 0.0
                    numeric_score = confidence if label == "positive" else -confidence if label == "negative" else 0.0
                    rows.append(
                        {
                            "doc_id": doc["doc_id"],
                            "score": numeric_score,
                            "sentiment_score": numeric_score,
                            "label": label,
                            "time_bin": _time_bin(_row_timestamp(doc)),
                            "published_at": _row_timestamp(doc),
                            "entity_label": unit.get("entity_label", doc.get("entity_label", doc.get("target_entity", ""))),
                            "target_entity": unit.get("target_entity", doc.get("target_entity", "")),
                            "series_name": unit.get("series_name", doc.get("series_name", "")),
                            "matched_alias": unit.get("matched_alias", doc.get("matched_alias", "")),
                            "matched_focus_terms": unit.get("matched_focus_terms", doc.get("matched_focus_terms", [])),
                            **_series_identity_fields(doc),
                        }
                    )
                used_provider = "flair"
                break
            if provider == "textblob" and _module_available("textblob"):
                from textblob import TextBlob

                rows = []
                for unit in units:
                    if _cancel_requested(context):
                        cancelled = True
                        return finish()
                    doc = unit["doc"]
                    polarity = float(TextBlob(str(unit.get("text", ""))).sentiment.polarity)
                    label = "positive" if polarity > 0.05 else "negative" if polarity < -0.05 else "neutral"
                    rows.append(
                        {
                            "doc_id": doc["doc_id"],
                            "score": polarity,
                            "sentiment_score": polarity,
                            "label": label,
                            "time_bin": _time_bin(_row_timestamp(doc)),
                            "published_at": _row_timestamp(doc),
                            "entity_label": unit.get("entity_label", doc.get("entity_label", doc.get("target_entity", ""))),
                            "target_entity": unit.get("target_entity", doc.get("target_entity", "")),
                            "series_name": unit.get("series_name", doc.get("series_name", "")),
                            "matched_alias": unit.get("matched_alias", doc.get("matched_alias", "")),
                            "matched_focus_terms": unit.get("matched_focus_terms", doc.get("matched_focus_terms", [])),
                            **_series_identity_fields(doc),
                        }
                    )
                used_provider = "textblob"
                break
        except Exception:
            rows = []
            continue

    if not rows and "heuristic" in providers:
        for unit in units:
            if _cancel_requested(context):
                cancelled = True
                return finish()
            doc = unit["doc"]
            tokens = [token.lower() for token in _tokenize(str(unit.get("text", "")))]
            score = sum(1 for token in tokens if token in POSITIVE_WORDS) - sum(
                1 for token in tokens if token in NEGATIVE_WORDS
            )
            rows.append(
                {
                    "doc_id": doc["doc_id"],
                    "score": float(score),
                    "sentiment_score": float(score),
                    "label": "positive" if score > 0 else "negative" if score < 0 else "neutral",
                    "time_bin": _time_bin(_row_timestamp(doc)),
                    "published_at": _row_timestamp(doc),
                    "entity_label": unit.get("entity_label", doc.get("entity_label", doc.get("target_entity", ""))),
                    "target_entity": unit.get("target_entity", doc.get("target_entity", "")),
                    "series_name": unit.get("series_name", doc.get("series_name", "")),
                    "matched_alias": unit.get("matched_alias", doc.get("matched_alias", "")),
                    "matched_focus_terms": unit.get("matched_focus_terms", doc.get("matched_focus_terms", [])),
                    **_series_identity_fields(doc),
                }
            )
        if rows:
            used_provider = "heuristic"
    if not rows:
        return _provider_unavailable_result("sentiment", caveats=source_caveats, **source_metadata)
    return finish()


def _text_classify(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    candidate_labels = params.get("candidate_labels", params.get("labels", params.get("classes", [])))
    if isinstance(candidate_labels, str):
        candidate_labels = [item.strip() for item in re.split(r"[,|]", candidate_labels) if item.strip()]
    candidate_labels = [str(label).strip() for label in candidate_labels if str(label).strip()]
    if candidate_labels:
        docs = _doc_rows(deps)
        texts = [_row_analysis_text(doc) for doc in docs]
        if not docs or not texts:
            return ToolExecutionResult(payload={"rows": []}, metadata={"no_data": True, "no_data_reason": "no_documents"})
        model_id = _sentence_embedding_model_id(context, params)
        embeddings, resolved_device = _encode_texts(candidate_labels + texts, model_id=model_id, normalize=True)
        label_vectors = embeddings[: len(candidate_labels)]
        doc_vectors = embeddings[len(candidate_labels) :]
        rows = []
        for doc, vector in zip(docs, doc_vectors, strict=False):
            scores = [float(vector @ label_vector) for label_vector in label_vectors]
            shifted = [max(score, -1.0) + 1.0 for score in scores]
            total = sum(shifted) or 1.0
            probs = [round(value / total, 6) for value in shifted]
            ranked = sorted(zip(candidate_labels, probs, scores, strict=False), key=lambda item: item[1], reverse=True)
            rows.append(
                {
                    "doc_id": str(doc.get("doc_id", "")),
                    "labels": [label for label, _, _ in ranked],
                    "probs": [prob for _, prob, _ in ranked],
                    "scores": [round(score, 6) for _, _, score in ranked],
                    "label": ranked[0][0] if ranked else "",
                    "confidence": ranked[0][1] if ranked else 0.0,
                }
            )
        return ToolExecutionResult(
            payload={"rows": rows},
            metadata=_metadata(
                "sentence_transformer",
                "embedding_zero_shot_text_classification",
                device=resolved_device,
                classification_mode="zero_shot_embedding",
            ),
        )
    sentiment_result = _sentiment(params, deps, context)
    sentiment_rows = sentiment_result.payload["rows"]
    return ToolExecutionResult(
        payload={
            "rows": [
                {"doc_id": row["doc_id"], "labels": [row["label"]], "probs": [abs(float(row["score"]))], "label": row["label"]}
                for row in sentiment_rows
            ]
        },
        caveats=["No candidate labels were supplied, so text_classify returned sentiment labels from the sentiment provider."],
        metadata={**sentiment_result.metadata, "classification_mode": "sentiment_proxy"},
    )


def _word_embeddings(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    vocab = sorted(
        {
            token.lower()
            for doc in _doc_rows(deps)
            for token in _tokenize(_row_analysis_text(doc))
            if token.lower() not in STOPWORDS
        }
    )[:256]
    if not vocab:
        return ToolExecutionResult(payload={"rows": []})
    model_id = _sentence_embedding_model_id(context, params)
    embeddings, resolved_device = _encode_texts(vocab, model_id=model_id, normalize=True)
    rows = []
    for token, vector in zip(vocab, embeddings, strict=False):
        preview = [round(float(value), 6) for value in vector[:8]]
        rows.append(
            {
                "token": token,
                "vector_ref": f"embed:{sha256(token.encode()).hexdigest()[:12]}",
                "vector_preview": preview,
                "embedding_dim": int(len(vector)),
            }
        )
    return ToolExecutionResult(
        payload={"rows": rows},
        metadata=_metadata("sentence-transformers", "sentence_transformer_word_embeddings", model_id=model_id, device=resolved_device),
    )


def _doc_embeddings(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    docs = _doc_rows(deps)
    if not docs:
        return ToolExecutionResult(payload={"rows": []})
    texts = [_row_analysis_text(doc) for doc in docs]
    model_id = _sentence_embedding_model_id(context, params)
    embeddings, resolved_device = _encode_texts(texts, model_id=model_id, normalize=True)
    rows = []
    for doc, vector in zip(docs, embeddings, strict=False):
        rows.append(
            {
                "doc_id": str(doc["doc_id"]),
                "vector_ref": f"embed:{sha256(str(doc.get('doc_id', '')).encode()).hexdigest()[:12]}",
                "vector_preview": [round(float(value), 6) for value in vector[:8]],
                "embedding_dim": int(len(vector)),
            }
        )
    return ToolExecutionResult(
        payload={"rows": rows},
        metadata=_metadata("sentence-transformers", "sentence_transformer_doc_embeddings", model_id=model_id, device=resolved_device),
    )


def _similarity_index(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    docs = _doc_rows(deps)
    if len(docs) < 2:
        return ToolExecutionResult(payload={"rows": []})
    model_id = _sentence_embedding_model_id(context, params)
    texts = [_row_analysis_text(doc) for doc in docs]
    embeddings, resolved_device = _encode_texts(texts, model_id=model_id, normalize=True)
    rows = []
    for idx, left_doc in enumerate(docs):
        left_vector = embeddings[idx]
        for right_idx in range(idx + 1, min(len(docs), idx + 8)):
            score = float(left_vector @ embeddings[right_idx])
            rows.append(
                {
                    "left_doc_id": str(left_doc["doc_id"]),
                    "right_doc_id": str(docs[right_idx]["doc_id"]),
                    "score": round(score, 4),
                }
            )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return ToolExecutionResult(
        payload={"rows": rows[:25]},
        metadata=_metadata("sentence-transformers", "sentence_transformer_similarity_index", model_id=model_id, device=resolved_device),
    )


def _similarity_pairwise(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    docs = _doc_rows(deps)
    query = str(params.get("query", getattr(context.state, "rewritten_question", "") or getattr(context.state, "question", ""))).strip()
    rows = []
    model_id = _sentence_embedding_model_id(context, params)
    if query:
        texts = [query] + [_row_analysis_text(doc) for doc in docs]
        embeddings, resolved_device = _encode_texts(texts, model_id=model_id, normalize=True)
        query_vector = embeddings[0]
        for idx, doc in enumerate(docs, start=1):
            rows.append(
                {
                    "left_id": "__query__",
                    "right_id": str(doc.get("doc_id", "")),
                    "score": round(float(query_vector @ embeddings[idx]), 4),
                }
            )
    else:
        texts = [_row_analysis_text(doc) for doc in docs]
        embeddings, resolved_device = _encode_texts(texts, model_id=model_id, normalize=True)
        for idx, left_doc in enumerate(docs):
            left_vector = embeddings[idx]
            for right_idx in range(idx + 1, min(len(docs), idx + 8)):
                rows.append(
                    {
                        "left_id": str(left_doc.get("doc_id", "")),
                        "right_id": str(docs[right_idx].get("doc_id", "")),
                        "score": round(float(left_vector @ embeddings[right_idx]), 4),
                    }
                )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return ToolExecutionResult(
        payload={"rows": rows[:25]},
        metadata=_metadata("sentence-transformers", "sentence_transformer_similarity_pairwise", model_id=model_id, device=resolved_device),
    )


def _metric_name_list(params: dict[str, Any]) -> list[str]:
    raw_metrics = params.get("metrics", [])
    if isinstance(raw_metrics, str):
        raw_metrics = [raw_metrics]
    if not isinstance(raw_metrics, list):
        return []
    names: list[str] = []
    for metric in raw_metrics:
        if isinstance(metric, str):
            name = metric.strip()
        elif isinstance(metric, dict):
            name = str(metric.get("name", metric.get("metric", "")) or "").strip()
        else:
            name = ""
        if name:
            names.append(name)
    return list(dict.fromkeys(names))


def _time_series_source_name(params: dict[str, Any]) -> str:
    return str(
        params.get(
            "metrics_source",
            params.get(
                "entities_source",
                params.get(
                    "annotations_source",
                    params.get(
                        "documents_node",
                        params.get(
                            "documents_source",
                            params.get("source", params.get("source_node", params.get("source_node_id", ""))),
                        ),
                    ),
                ),
            ),
        )
        or ""
    ).strip()


def _series_dicts(params: dict[str, Any]) -> list[dict[str, Any]]:
    raw_series = params.get("series_definitions", params.get("series", []))
    if isinstance(raw_series, dict):
        raw_series = [raw_series]
    if not isinstance(raw_series, list):
        return []
    return [dict(item) for item in raw_series if isinstance(item, dict)]


def _series_display_name(item: dict[str, Any]) -> str:
    raw_entity_terms = item.get("entity_terms", item.get("terms", item.get("match_terms", [])))
    if isinstance(raw_entity_terms, str):
        entity_terms = [raw_entity_terms]
    elif isinstance(raw_entity_terms, list):
        entity_terms = [str(term).strip() for term in raw_entity_terms if str(term).strip()]
    else:
        entity_terms = []
    explicit = str(item.get("series_name", item.get("label", item.get("entity", ""))) or "").strip()
    if explicit:
        return explicit
    raw_name = str(item.get("name", "") or "").strip()
    if entity_terms and (not raw_name or "_" in raw_name or raw_name.lower().endswith(("_docs", "_documents"))):
        return entity_terms[0]
    return raw_name or (entity_terms[0] if entity_terms else "")


def _series_match_terms(item: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for key in ("aliases", "match_terms", "terms", "entity_terms"):
        raw = item.get(key)
        if isinstance(raw, str):
            terms.append(raw)
        elif isinstance(raw, list):
            terms.extend(str(term).strip() for term in raw if str(term).strip())
    display = _series_display_name(item)
    if display:
        terms.append(display)
        terms.extend(_target_aliases_for_name(display))
    return list(dict.fromkeys(term for term in terms if term))


def _time_series_rows_from_document_series(
    params: dict[str, Any],
    deps: dict[str, ToolExecutionResult],
    context: AgentExecutionContext,
    *,
    granularity: str,
) -> tuple[list[dict[str, Any]], int] | None:
    series_items = _series_dicts(params)
    metric_names = {name.lower() for name in _metric_name_list(params)}
    if not series_items or (metric_names and not metric_names.intersection({"document_count", "doc_count", "count"})):
        return None
    source = _time_series_source_name(params)
    docs = _time_series_source_rows(params, deps, context, source)
    docs = [_coerce_text_document_row(dict(doc)) for doc in docs if isinstance(doc, dict) and _row_has_text_payload(doc)]
    if not docs:
        return [], 0
    values: Counter[tuple[str, str]] = Counter()
    doc_sets: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    period_doc_sets: defaultdict[str, set[str]] = defaultdict(set)
    period_doc_counts: Counter[str] = Counter()
    for doc in docs:
        text = _row_analysis_text(doc)
        haystack = text.lower()
        timestamp = str(_first_nonempty_field(doc, ["published_at", "date", "time_bin", "month", "period", "time_period"]))
        time_bin = _time_bin(timestamp, granularity)
        doc_id = str(doc.get("doc_id", "") or "").strip()
        if doc_id:
            period_doc_sets[time_bin].add(doc_id)
        else:
            period_doc_counts[time_bin] += 1
        for item in series_items:
            series_name = _series_display_name(item)
            aliases = _series_match_terms(item)
            keyword_terms = item.get("keyword_terms", item.get("context_keywords", item.get("keywords", [])))
            if isinstance(keyword_terms, str):
                keywords = [keyword_terms]
            elif isinstance(keyword_terms, list):
                keywords = [str(term).strip() for term in keyword_terms if str(term).strip()]
            else:
                keywords = []
            if not series_name or not any(_alias_in_text(text, alias) for alias in aliases):
                continue
            if keywords and not any(keyword.lower() in haystack for keyword in keywords):
                continue
            values[(series_name, time_bin)] += 1
            if doc_id:
                doc_sets[(series_name, time_bin)].add(doc_id)
    rows = []
    for (series_name, time_bin), count in sorted(values.items()):
        document_count = len(doc_sets.get((series_name, time_bin), set())) or int(count)
        period_document_count = len(period_doc_sets.get(time_bin, set())) + int(period_doc_counts.get(time_bin, 0))
        document_share = document_count / max(period_document_count, 1)
        rows.append(
            {
                "series_name": series_name,
                "target_entity": series_name,
                "entity": series_name,
                "canonical_entity": series_name,
                "entity_label": series_name,
                **_time_bin_fields(time_bin, granularity),
                "document_count": document_count,
                "doc_count": document_count,
                "count": document_count,
                "period_document_count": period_document_count,
                "share_of_documents": round(document_share, 6),
                "document_share": round(document_share, 6),
                "percent_of_documents": round(document_share * 100.0, 4),
            }
        )
    return rows, 0


def _time_series_rows_from_named_metrics(
    params: dict[str, Any],
    deps: dict[str, ToolExecutionResult],
    context: AgentExecutionContext,
    *,
    granularity: str,
) -> tuple[list[dict[str, Any]], int] | None:
    raw_metrics = params.get("metrics", [])
    if isinstance(raw_metrics, list) and any(isinstance(metric, dict) for metric in raw_metrics):
        return None
    metric_names = _metric_name_list(params)
    supported = {
        "average_sentiment",
        "avg_sentiment",
        "mean_sentiment",
        "average_claim_strength",
        "avg_claim_strength",
        "mean_claim_strength",
        "document_count",
        "doc_count",
        "claim_count",
        "count",
        "mention_count",
        "document_frequency",
        "doc_frequency",
        "share_of_documents",
        "share_of_docs",
        "doc_share",
        "average_score",
        "avg_score",
        "mean_score",
    }
    normalized_metrics = [name.lower() for name in metric_names if name.lower() in supported]
    if not normalized_metrics:
        return None
    source = _time_series_source_name(params)
    source_rows = _time_series_source_rows(params, deps, context, source)
    if not source_rows:
        return [], 0
    raw_group_by = params.get("group_by", params.get("series_key", ""))
    group_candidates = [str(item) for item in raw_group_by] if isinstance(raw_group_by, list) else [str(raw_group_by or "")]
    group_candidates.extend(
        [
            "target_label",
            "series_name",
            "target_entity",
            "entity_label",
            "canonical_entity",
            "linked_entity",
            "actor",
            "attributed_actor",
            "entity_text",
            "entity",
        ]
    )
    raw_entity_types = params.get("entity_types") or params.get("include_entity_types") or params.get("allowed_entity_labels") or []
    if isinstance(raw_entity_types, str):
        allowed_labels = {raw_entity_types.upper()}
    elif isinstance(raw_entity_types, list):
        allowed_labels = {str(item).upper() for item in raw_entity_types if str(item).strip()}
    else:
        allowed_labels = set()
    ignored_labels = {"DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT"}
    label_fields = [
        str(params.get("entity_label_field", "") or "").strip(),
        "label",
        "entity_label",
        "entity_type",
        "type",
    ]
    date_field = str(params.get("date_field", params.get("time_field", params.get("datetime_field", ""))) or "")
    fallback_timestamp = str(
        params.get("fallback_time_bin")
        or params.get("default_time_bin")
        or params.get("fallback_date")
        or ""
    ).strip()
    values: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    doc_sets: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    period_doc_sets: defaultdict[str, set[str]] = defaultdict(set)
    row_counts: Counter[tuple[str, str]] = Counter()
    skipped = 0
    for row in source_rows:
        label = str(_first_nonempty_field(row, label_fields) or "").upper()
        if allowed_labels and label and label not in allowed_labels:
            skipped += 1
            continue
        if not allowed_labels and label in ignored_labels:
            skipped += 1
            continue
        entity = _canonical_entity(_first_nonempty_field(row, group_candidates))
        if not entity and row.get("topic_id") is not None:
            entity = f"topic_{row.get('topic_id')}"
        if not entity:
            entity = "__all__"
        if entity not in {"__all__"} and not entity.startswith("topic_") and not _valid_series_surface(entity):
            skipped += 1
            continue
        timestamp = str(_first_nonempty_field(row, [date_field, "month", "period", "time_period", "date", "published_at", "time_bin"]))
        if not timestamp.strip() and fallback_timestamp:
            timestamp = fallback_timestamp
        time_bin = _time_bin(timestamp, granularity)
        key = (entity, time_bin)
        row_counts[key] += 1
        doc_id = str(row.get("doc_id", "") or "").strip()
        if doc_id:
            doc_sets[key].add(doc_id)
            period_doc_sets[time_bin].add(doc_id)
        for metric in normalized_metrics:
            if metric in {
                "document_count",
                "doc_count",
                "claim_count",
                "count",
                "mention_count",
                "document_frequency",
                "doc_frequency",
                "share_of_documents",
                "share_of_docs",
                "doc_share",
            }:
                continue
            if metric in {"average_sentiment", "avg_sentiment", "mean_sentiment"}:
                value = _plot_float(row.get("sentiment_score", row.get("score")))
                output_metric = "average_sentiment"
            elif metric in {"average_claim_strength", "avg_claim_strength", "mean_claim_strength"}:
                value = _plot_float(row.get("claim_strength_score", row.get("score")))
                output_metric = "average_claim_strength"
            else:
                value = _plot_float(row.get("score", row.get("value")))
                output_metric = "average_score"
            if value is not None:
                values[(entity, time_bin, output_metric)].append(value)
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for key, row_count in row_counts.items():
        entity, time_bin = key
        document_count = len(doc_sets.get(key, set())) or int(row_count)
        rows_by_key[key] = {
            "entity": entity,
            "canonical_entity": entity,
            "actor": entity,
            "entity_label": entity,
            "target_label": entity,
            "series_name": entity,
            "target_entity": entity,
            **_time_bin_fields(time_bin, granularity),
        }
        if any(metric in normalized_metrics for metric in ("document_count", "doc_count")):
            rows_by_key[key]["document_count"] = document_count
            rows_by_key[key]["doc_count"] = document_count
        if "claim_count" in normalized_metrics:
            rows_by_key[key]["claim_count"] = int(row_count)
        if "count" in normalized_metrics:
            rows_by_key[key]["count"] = int(row_count)
        if "mention_count" in normalized_metrics:
            rows_by_key[key]["mention_count"] = int(row_count)
            rows_by_key[key].setdefault("count", int(row_count))
        if any(metric in normalized_metrics for metric in ("document_frequency", "doc_frequency")):
            rows_by_key[key]["document_frequency"] = document_count
            rows_by_key[key]["doc_frequency"] = document_count
        if any(metric in normalized_metrics for metric in ("share_of_documents", "share_of_docs", "doc_share")):
            period_document_count = len(period_doc_sets.get(time_bin, set())) or document_count
            share = round(document_count / max(period_document_count, 1), 6)
            rows_by_key[key]["share_of_documents"] = share
            rows_by_key[key]["share_of_docs"] = share
            rows_by_key[key]["doc_share"] = share
    for (entity, time_bin, metric), metric_values in values.items():
        target = rows_by_key.setdefault(
            (entity, time_bin),
            {
                "entity": entity,
                "canonical_entity": entity,
                "actor": entity,
                "entity_label": entity,
                "target_label": entity,
                "series_name": entity,
                "target_entity": entity,
                **_time_bin_fields(time_bin, granularity),
            },
        )
        target[metric] = round(sum(metric_values) / max(len(metric_values), 1), 6)
    rows = list(rows_by_key.values())
    for row in rows:
        if "count" not in row:
            row["count"] = row.get("document_count", row.get("claim_count", 1))
    rows.sort(key=lambda row: (str(row.get("time_bin", "")), str(row.get("series_name", ""))))
    return rows, skipped


def _time_series_aggregate(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    params = _payload_or_params(params)
    granularity = str(
        params.get(
            "frequency",
            params.get(
                "bucket_granularity",
                params.get("granularity", params.get("time_granularity", params.get("interval", _default_time_granularity()))),
            ),
        )
    ).strip().lower() or "month"
    metric_series = _time_series_rows_from_metric_specs(params, deps, context, granularity=granularity)
    if metric_series is not None:
        rows, skipped_rows = metric_series
        caveats = []
        if skipped_rows:
            caveats.append(f"Skipped {skipped_rows} metric rows that could not be assigned to a requested series or numeric metric.")
        return ToolExecutionResult(payload={"rows": rows, "skipped_row_count": skipped_rows}, caveats=caveats)
    document_series = _time_series_rows_from_document_series(params, deps, context, granularity=granularity)
    if document_series is not None:
        rows, skipped_rows = document_series
        return ToolExecutionResult(payload={"rows": rows, "skipped_row_count": skipped_rows})
    named_metric_series = _time_series_rows_from_named_metrics(params, deps, context, granularity=granularity)
    if named_metric_series is not None:
        rows, skipped_rows = named_metric_series
        caveats = []
        if skipped_rows:
            caveats.append(f"Skipped {skipped_rows} time-series rows with unsupported entity labels or unusable series names.")
        return ToolExecutionResult(payload={"rows": rows, "skipped_row_count": skipped_rows}, caveats=caveats)
    preferred_source = _time_series_source_name(params)
    source_rows = _time_series_source_rows(params, deps, context, preferred_source) if preferred_source else []
    if not source_rows:
        for result in deps.values():
            payload = result.payload
            if isinstance(payload, dict) and "rows" in payload:
                source_rows = list(payload["rows"])
                break
    raw_group_by = params.get("group_by", params.get("series_key", ""))
    if isinstance(raw_group_by, list):
        group_candidates = [str(item) for item in raw_group_by]
    else:
        group_candidates = [str(raw_group_by or "")]
    fallback_group = str(params.get("fallback_group_by", params.get("fallback_series_key", "")) or "")
    group_candidates.extend([fallback_group, "series_name", "target_entity", "entity_label", "canonical_entity", "linked_entity", "actor", "attributed_actor", "entity_text", "entity", "label", "term"])
    date_field = str(params.get("date_field", params.get("time_field", params.get("datetime_field", ""))) or "")
    fallback_timestamp = str(
        params.get("fallback_time_bin")
        or params.get("default_time_bin")
        or params.get("fallback_date")
        or ""
    ).strip()
    value_field = str(params.get("value_field", params.get("metric", "")) or "")
    metric_rows = params.get("metrics")
    if isinstance(metric_rows, list) and metric_rows:
        first_metric = metric_rows[0] if isinstance(metric_rows[0], dict) else {}
        if not value_field:
            value_field = str(first_metric.get("field", first_metric.get("name", first_metric.get("metric", ""))) or "")
    raw_entity_types = params.get("entity_types") or params.get("include_entity_types") or []
    if isinstance(raw_entity_types, str):
        allowed_labels = {raw_entity_types.upper()}
    elif isinstance(raw_entity_types, list):
        allowed_labels = {str(item).upper() for item in raw_entity_types if str(item).strip()}
    else:
        allowed_labels = set()
    ignored_labels = {"DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT"}
    grouped: defaultdict[tuple[str, str], float] = defaultdict(float)
    grouped_value_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    period_docs: defaultdict[str, set[str]] = defaultdict(set)
    skipped_rows = 0
    top_n = _int_param(params, "top_n", "top_k", "limit", default=100, maximum=5000)
    aggregation = str(params.get("aggregation", params.get("agg", "")) or "").strip().lower()
    for row in source_rows:
        label = str(row.get("label", row.get("entity_type", "")) or "").upper()
        if allowed_labels and label and label not in allowed_labels:
            skipped_rows += 1
            continue
        if not allowed_labels and label in ignored_labels:
            skipped_rows += 1
            continue
        entity = str(_first_nonempty_field(row, group_candidates)).strip()
        if not entity and row.get("topic_id") is not None:
            entity = f"topic_{row.get('topic_id')}"
        if not entity:
            entity = str(row.get("doc_id") or "__all__")
        entity = _canonical_entity(entity)
        if entity not in {"__all__"} and not entity.startswith("topic_") and not _valid_series_surface(entity):
            skipped_rows += 1
            continue
        timestamp = str(_first_nonempty_field(row, [date_field, "month", "period", "time_period", "date", "published_at", "time_bin"]))
        if not timestamp.strip() and fallback_timestamp:
            timestamp = fallback_timestamp
        time_bin = _time_bin(timestamp, granularity)
        value = row.get(value_field) if value_field else None
        if value in (None, ""):
            value = row.get("count", row.get("mention_count", row.get("document_frequency", row.get("weight", row.get("score", 1)))))
        numeric_value = _plot_float(value)
        if numeric_value is None:
            numeric_value = 1.0
        grouped[(entity, time_bin)] += float(numeric_value)
        grouped_value_counts[(entity, time_bin)] += 1
        doc_id = str(row.get("doc_id", "")).strip()
        if doc_id:
            period_docs[time_bin].add(doc_id)
    entity_totals: Counter[str] = Counter()
    for (entity, _period_key), count in grouped.items():
        entity_totals[entity] += float(count)
    retained_entities = {
        entity
        for entity, _ in sorted(entity_totals.items(), key=lambda item: (item[1], item[0].lower()), reverse=True)[:top_n]
    }
    rows = []
    normalize_by_docs = str(params.get("normalize_by", "")).strip().lower() in {"documents_per_period", "docs_per_period", "document_count"}
    for (entity, time_bin), count in sorted(grouped.items()):
        if entity not in retained_entities:
            continue
        value_count = max(grouped_value_counts.get((entity, time_bin), 1), 1)
        aggregate_value = count / value_count if aggregation in {"mean", "avg", "average"} else count
        period_doc_count = len(period_docs.get(time_bin, set()))
        normalized = aggregate_value / max(period_doc_count, 1) if normalize_by_docs else aggregate_value
        rendered_value = int(round(aggregate_value)) if float(aggregate_value).is_integer() else round(aggregate_value, 6)
        rendered_count = int(round(count)) if float(count).is_integer() else round(count, 6)
        row = {
            "entity": entity,
            "canonical_entity": entity,
            "actor": entity,
            "entity_label": entity,
            "series_name": entity,
            **_time_bin_fields(time_bin, granularity),
            "count": rendered_count,
            "mention_count": rendered_count,
            "mention_count_normalized": round(normalized, 6),
            "normalized_value": round(normalized, 6),
        }
        if value_field:
            row[value_field] = rendered_value
            if aggregation in {"mean", "avg", "average"}:
                row[f"mean_{value_field}"] = rendered_value
        rows.append(row)
    rows.sort(key=lambda row: (str(row.get("time_bin", "")), -float(row.get("count", 0.0)), str(row.get("entity", ""))))
    caveats = []
    if skipped_rows:
        caveats.append(f"Skipped {skipped_rows} time-series rows with unsupported entity labels or unusable series names.")
    return ToolExecutionResult(payload={"rows": rows, "skipped_row_count": skipped_rows}, caveats=caveats)


def _change_point_detect(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    series = []
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload:
            series = list(payload["rows"])
            break
    by_entity: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
    skipped_rows = 0
    for row in series:
        entity = str(row.get("entity", "__all__"))
        time_bin = str(row.get("time_bin", "unknown"))
        if _is_placeholder_axis_label(entity) or _is_placeholder_axis_label(time_bin):
            skipped_rows += 1
            continue
        by_entity[entity].append((time_bin, float(row.get("count", row.get("score", 0.0)))))
    changes = []
    for entity, items in by_entity.items():
        ordered = sorted(items)
        values = [value for _, value in ordered]
        if len(values) < 2:
            continue
        avg_delta = sum(abs(values[idx] - values[idx - 1]) for idx in range(1, len(values))) / max(len(values) - 1, 1)
        threshold = max(avg_delta * 1.5, 1.0)
        for idx in range(1, len(values)):
            delta = values[idx] - values[idx - 1]
            if abs(delta) >= threshold:
                changes.append({"entity": entity, "time_bin": ordered[idx][0], "delta": delta})
    caveats = []
    if skipped_rows:
        caveats.append(f"Skipped {skipped_rows} rows with missing or placeholder change-point labels.")
    return ToolExecutionResult(payload={"rows": changes, "skipped_row_count": skipped_rows}, caveats=caveats)


def _burst_detect(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    series = _time_series_aggregate(params, deps, context).payload["rows"]
    by_entity: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
    skipped_rows = 0
    for row in series:
        entity = str(row.get("entity", "__all__"))
        time_bin = str(row.get("time_bin", "unknown"))
        if _is_placeholder_axis_label(entity) or _is_placeholder_axis_label(time_bin):
            skipped_rows += 1
            continue
        by_entity[entity].append((time_bin, float(row.get("count", 0))))
    bursts = []
    for entity, items in by_entity.items():
        values = [value for _, value in items]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        std = variance ** 0.5
        threshold = mean + max(std, 1.0)
        for time_bin, value in items:
            if value >= threshold:
                bursts.append(
                    {
                        "entity": entity,
                        "time_bin": time_bin,
                        "burst_level": 1,
                        "intensity": value,
                        "start": time_bin,
                        "end": time_bin,
                    }
                )
    caveats = []
    if skipped_rows:
        caveats.append(f"Skipped {skipped_rows} rows with missing or placeholder burst labels.")
    return ToolExecutionResult(payload={"rows": bursts, "skipped_row_count": skipped_rows}, caveats=caveats)


def _claim_span_extract(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    target_groups = _target_alias_groups(params, context)
    focus_terms = _focus_terms(params)
    fallback_keywords = sorted(CLAIM_KEYWORDS)
    rows = []
    for doc in _doc_rows(deps):
        raw_sentences = doc.get("sentences")
        if isinstance(raw_sentences, list) and raw_sentences:
            sentences = [str(sentence).strip() for sentence in raw_sentences if str(sentence).strip()]
        else:
            sentences = simple_sentence_split(_row_analysis_text(doc))
        for sentence in sentences:
            lowered = sentence.lower()
            target_entity, matched_alias = _match_target_alias(sentence, target_groups)
            if target_groups and not target_entity:
                continue
            matched_claim_keywords = [keyword for keyword in fallback_keywords if keyword in lowered]
            matched_focus_terms = [term for term in focus_terms if term.lower() in lowered]
            if focus_terms:
                if not matched_focus_terms:
                    continue
            elif not matched_claim_keywords:
                continue
            rows.append(
                {
                    "doc_id": str(doc.get("doc_id", "")),
                    "outlet": str(doc.get("outlet", doc.get("source", ""))),
                    "date": str(doc.get("date", doc.get("published_at", ""))),
                    "published_at": str(doc.get("published_at", doc.get("date", ""))),
                    "time_bin": _time_bin(str(doc.get("date", doc.get("published_at", "")))),
                    "excerpt": sentence[:320],
                    "claim_span": sentence,
                    "text": sentence,
                    "target_entity": target_entity,
                    "entity_label": target_entity,
                    "series_name": target_entity,
                    "matched_alias": matched_alias,
                    "matched_keywords": list(dict.fromkeys([*matched_claim_keywords, *matched_focus_terms])),
                    "matched_focus_terms": matched_focus_terms,
                }
            )
    return ToolExecutionResult(payload={"rows": rows})


def _claim_strength_score(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    spans = []
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload:
            spans = list(payload["rows"])
            break
    scored = []
    for row in spans:
        excerpt = str(row.get("claim_span", row.get("excerpt", ""))).lower()
        score = 0.3
        matched = {str(item).lower() for item in row.get("matched_keywords", []) if str(item).strip()}
        focus_matched = {str(item).lower() for item in row.get("matched_focus_terms", []) if str(item).strip()}
        if "imminent" in matched or "likely" in matched:
            score += 0.35
        if matched.intersection({"predict", "predicted", "prediction", "forecast", "forecasted", "forecasts", "foresaw", "foresee"}):
            score += 0.25
        if matched.intersection({"warn", "warned", "warning", "anticipate", "anticipated", "anticipates", "expect", "expected", "expects"}):
            score += 0.15
        if focus_matched.intersection({"value", "worth", "quality", "performance", "legacy", "talent", "form"}):
            score += 0.15
        if focus_matched.intersection({"best", "greatest", "elite", "legend", "icon", "star", "dominance", "peak", "decline"}):
            score += 0.2
        claim_strength = round(min(score, 1.0), 3)
        scored.append({**row, "score": claim_strength, "claim_strength_score": claim_strength})
    scored.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return ToolExecutionResult(payload={"rows": scored})


def _quote_extract(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = []
    document_lookup = {str(doc.get("doc_id", "")): doc for doc in _doc_rows(deps) if str(doc.get("doc_id", ""))}
    for doc in _doc_rows(deps):
        text = _row_analysis_text(doc)
        for match in QUOTE_PATTERN.finditer(text):
            doc_id = str(doc.get("doc_id", ""))
            source_doc = document_lookup.get(doc_id, doc)
            rows.append(
                {
                    "doc_id": doc_id,
                    "quote": match.group(1),
                    "text": text[:500],
                    "outlet": str(source_doc.get("outlet", source_doc.get("source", ""))),
                    "time_bin": _time_bin(_row_timestamp(source_doc)),
                    "published_at": _row_timestamp(source_doc),
                }
            )
    return ToolExecutionResult(payload={"rows": rows})


def _quote_attribute(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = []
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload:
            rows = list(payload["rows"])
            break
    attributed = []
    for row in rows:
        text = str(row.get("text", ""))
        speaker_match = SPEAKER_PATTERN.search(text)
        speaker = speaker_match.group(1) if speaker_match else str(row.get("speaker", "unknown") or "unknown")
        attributed.append({**row, "speaker": speaker, "attributed_actor": speaker})
    return ToolExecutionResult(payload={"rows": attributed})


NOUN_FREQUENCY_TASK_ALIASES = {
    "aggregate_noun_frequencies",
    "aggregate_noun_frequency",
    "aggregate_noun_lemma_distribution",
    "aggregate_noun_lemmas",
    "aggregate_token_frequencies",
    "frequency_distribution",
    "noun_frequencies",
    "noun_frequency",
    "noun_frequency_distribution",
    "noun_lemma_distribution",
    "token_frequency_distribution",
}


ENTITY_FREQUENCY_TASK_ALIASES = {
    "actor_frequency",
    "actor_distribution",
    "actor_prominence",
    "aggregate_named_entity_prominence_overall",
    "aggregate_named_entity_time_series",
    "document_frequency_by_entity",
    "entity_distribution",
    "entity_frequency",
    "entity_frequency_distribution",
    "entity_prominence",
    "named_entity_distribution",
    "named_entity_frequency",
    "named_entity_frequency_distribution",
    "named_entity_prominence",
    "quote_attribution_frequency",
}


def _task_name(params: dict[str, Any]) -> str:
    for key in ("task", "task_name", "operation", "purpose"):
        value = str(params.get(key, "") or "").strip()
        if value:
            return value.lower()
    return ""


def _analysis_params(params: dict[str, Any]) -> dict[str, Any]:
    payload = _payload_or_params(params)
    merged = dict(payload)
    nested = merged.get("params")
    if isinstance(nested, dict):
        for key, value in nested.items():
            merged.setdefault(key, value)
    return merged


def _is_noun_frequency_task(task: str, params: dict[str, Any]) -> bool:
    if task in NOUN_FREQUENCY_TASK_ALIASES:
        return True
    filters = params.get("filters")
    upos_values: set[str] = set()
    if isinstance(filters, dict):
        raw_upos = filters.get("upos", filters.get("pos", []))
        if isinstance(raw_upos, str):
            upos_values.add(raw_upos.upper())
        elif isinstance(raw_upos, list):
            upos_values.update(str(item).upper() for item in raw_upos)
    return bool("frequency" in task and ("noun" in task or "NOUN" in upos_values or "PROPN" in upos_values))


def _is_entity_frequency_task(task: str, params: dict[str, Any]) -> bool:
    if _is_actor_prominence_merge_task(task):
        return False
    if task in ENTITY_FREQUENCY_TASK_ALIASES:
        return True
    if any(key in params for key in ("entity_source", "entity_field", "actor_field", "entities_node")):
        return True
    if "aggregate" in task and ("entity" in task or "actor" in task):
        return True
    return bool(
        ("entity" in task or "actor" in task or "attribution" in task)
        and ("frequency" in task or "prominence" in task or "distribution" in task)
    )


def _is_actor_prominence_merge_task(task: str) -> bool:
    return "merge" in task and ("actor" in task or "entity" in task or "prominence" in task)


def _int_param(params: dict[str, Any], *names: str, default: int, minimum: int = 1, maximum: int = 1000) -> int:
    for name in names:
        if params.get(name) in (None, ""):
            continue
        try:
            value = int(params.get(name) or default)
        except (TypeError, ValueError):
            continue
        return max(minimum, min(maximum, value))
    return max(minimum, min(maximum, default))


def _payload_rows_from_all_dependencies(deps: dict[str, ToolExecutionResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in deps.values():
        payload = result.payload if isinstance(result.payload, dict) else {}
        for key in ("rows", "documents", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                rows.extend(dict(item) for item in value if isinstance(item, dict))
    return rows


def _first_nonempty_field(row: dict[str, Any], fields: Iterable[str]) -> Any:
    for field in fields:
        if not field:
            continue
        value = row.get(field)
        if value not in (None, "") and str(value).strip():
            return value
    return ""


def _target_aliases_for_name(name: str) -> list[str]:
    canonical = _canonical_entity(name)
    aliases = [canonical]
    tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z'.-]*", canonical) if token]
    if len(tokens) >= 2:
        last = tokens[-1].strip(".'-")
        if len(last) >= 3 and last.lower() not in ENTITY_SURFACE_STOPWORDS and last.lower() not in {"president", "minister", "chancellor"}:
            aliases.append(last)
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _entity_like_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"\b(?:[A-Z][A-Za-z'.-]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z'.-]+|[A-Z]{2,})){1,5}\b",
        str(text or ""),
    ):
        phrase = _canonical_entity(match.group(0))
        lowered = phrase.lower()
        if lowered in seen or lowered in ENTITY_SURFACE_STOPWORDS:
            continue
        if not _valid_entity_surface(phrase):
            continue
        seen.add(lowered)
        phrases.append(phrase)
    return phrases


def _comparison_side_phrases(text: str) -> list[str]:
    phrase = r"(?:[A-Z][A-Za-z'.-]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z'.-]+|[A-Z]{2,})){0,4}"
    pattern = re.compile(
        rf"\b(?P<left>{phrase})\s+(?:vs\.?|versus|compared\s+(?:with|to)|against|rather\s+than)\s+(?P<right>{phrase})\b",
    )
    phrases: list[str] = []
    for match in pattern.finditer(str(text or "")):
        for side in ("left", "right"):
            candidate = _canonical_entity(match.group(side))
            if candidate and _valid_entity_surface(candidate):
                phrases.append(candidate)
    return list(dict.fromkeys(phrases))


def _inferred_comparison_targets(params: dict[str, Any], context: AgentExecutionContext | None = None) -> list[tuple[str, list[str]]]:
    texts = [
        str(params.get("question", "") or ""),
        str(params.get("query", "") or ""),
    ]
    focus_text = str(params.get("query_focus", "") or "")
    if context is not None:
        texts.extend(
            [
                str(getattr(context.state, "rewritten_question", "") or ""),
                str(getattr(context.state, "question", "") or ""),
            ]
        )
    combined = " ".join(text for text in texts if text.strip())
    if not combined.strip():
        return []
    has_comparison_signal = bool(
        re.search(r"\b(vs\.?|versus|compared\s+(?:with|to)|between|rather\s+than|against)\b", combined, flags=re.IGNORECASE)
    )
    quoted = []
    for left, right in re.findall(r'"([^"]{2,80})"|\'([^\']{2,80})\'', f"{combined} {focus_text}"):
        value = (left or right).strip()
        if value:
            quoted.append(value)
    candidates = quoted + (_comparison_side_phrases(combined) + _entity_like_phrases(combined) if has_comparison_signal else [])
    groups: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for candidate in candidates:
        canonical = _canonical_entity(candidate)
        lowered = canonical.lower()
        if lowered in seen or not _valid_entity_surface(canonical):
            continue
        seen.add(lowered)
        groups.append((canonical, _target_aliases_for_name(canonical)))
    return groups[:6]


def _target_alias_groups(params: dict[str, Any], context: AgentExecutionContext | None = None) -> list[tuple[str, list[str]]]:
    raw_targets = params.get("targets", params.get("target_entities", params.get("entities", [])))
    if isinstance(raw_targets, dict):
        raw_targets = [raw_targets]
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    groups: list[tuple[str, list[str]]] = []
    if not isinstance(raw_targets, list):
        return _inferred_comparison_targets(params, context)
    for item in raw_targets:
        canonical = ""
        aliases: list[str] = []
        if isinstance(item, dict):
            canonical = str(item.get("canonical", item.get("entity", item.get("label", item.get("name", "")))) or "").strip()
            raw_aliases = item.get("aliases", item.get("match_terms", item.get("terms", [])))
            if isinstance(raw_aliases, str):
                raw_aliases = [raw_aliases]
            if isinstance(raw_aliases, list):
                aliases.extend(str(alias).strip() for alias in raw_aliases if str(alias).strip())
        else:
            canonical = str(item or "").strip()
        if canonical:
            aliases = [*aliases, *_target_aliases_for_name(canonical)]
        normalized_aliases = list(dict.fromkeys(alias for alias in aliases if alias))
        if canonical and normalized_aliases:
            groups.append((canonical, normalized_aliases))
    return groups or _inferred_comparison_targets(params, context)


def _alias_in_text(text: str, alias: str) -> bool:
    if not text or not alias:
        return False
    pattern = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _match_target_alias(text: Any, groups: list[tuple[str, list[str]]]) -> tuple[str, str]:
    haystack = str(text or "")
    for canonical, aliases in groups:
        for alias in aliases:
            if _alias_in_text(haystack, alias):
                return canonical, alias
    return "", ""


def _series_identity_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        field: row[field]
        for field in ("target_entity", "entity_label", "series_name", "entity", "canonical_entity", "actor")
        if field in row and str(row.get(field, "")).strip()
    }


def _canonical_entity(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n\"',;:")
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\s+(today|yesterday|tomorrow|tonight|said|says|say)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text[:160]


def _entity_row_time_bin(row: dict[str, Any], timestamp_field: str = "") -> str:
    raw = _first_nonempty_field(
        row,
        [timestamp_field, "month", "period", "time_bin", "date", "published_at", "timestamp"],
    )
    return _time_bin(str(raw))


def _valid_entity_surface(entity: str) -> bool:
    if not entity or len(entity) < 2:
        return False
    lowered = entity.lower()
    if lowered in ENTITY_SURFACE_STOPWORDS:
        return False
    if re.fullmatch(r"[0-9a-f]{24,}", lowered):
        return False
    if re.fullmatch(r"[\d\s,./:-]+", entity):
        return False
    if re.fullmatch(r"[$€£]?\s*\d+(?:[.,]\d+)?\s*(?:k|m|bn|billion|million)?", lowered):
        return False
    if re.fullmatch(r"[a-z]+", entity):
        return False
    return bool(re.search(r"[A-Za-z]", entity))


def _valid_series_surface(series: str) -> bool:
    if not series or len(series) < 2:
        return False
    lowered = series.lower()
    if lowered in SERIES_SURFACE_STOPWORDS:
        return False
    if _valid_entity_surface(series):
        return True
    if re.fullmatch(r"[0-9a-f]{24,}", lowered):
        return False
    if re.fullmatch(r"[\d\s,./:-]+", series):
        return False
    return bool(re.search(r"[A-Za-z]", series))


def _dependency_rows_for_source(deps: dict[str, ToolExecutionResult], source: str) -> list[dict[str, Any]]:
    if source and source in deps:
        payload = deps[source].payload if isinstance(deps[source].payload, dict) else {}
        for key in ("rows", "documents", "results"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [dict(item) for item in rows if isinstance(item, dict)]
    return _payload_rows_from_all_dependencies(deps)


def _time_series_source_rows(
    params: dict[str, Any],
    deps: dict[str, ToolExecutionResult],
    context: AgentExecutionContext,
    source: str = "",
) -> list[dict[str, Any]]:
    rows = _dependency_rows_for_source(deps, source) if source else _payload_rows_from_all_dependencies(deps)
    if rows:
        return rows
    working_set_ref = _resolve_working_set_ref(params, deps)
    if not working_set_ref:
        return []
    max_documents = _working_set_analysis_max_documents()
    documents, _metadata, _caveats = _analysis_document_rows_from_deps(
        deps,
        context,
        max_documents=max_documents,
    )
    return documents


def _series_definitions(params: dict[str, Any]) -> list[tuple[str, list[str]]]:
    raw_series = params.get("series_definitions", params.get("series", []))
    if isinstance(raw_series, dict):
        raw_series = [raw_series]
    if isinstance(raw_series, str):
        raw_series = [raw_series]
    groups: list[tuple[str, list[str]]] = []
    if isinstance(raw_series, list):
        for item in raw_series:
            if isinstance(item, dict):
                canonical = str(
                    item.get(
                        "series_name",
                        item.get("name", item.get("label", item.get("entity", item.get("canonical", "")))),
                    )
                    or ""
                ).strip()
                aliases = [canonical]
                raw_aliases = item.get("aliases", item.get("match_terms", item.get("terms", item.get("entity_terms", []))))
                if isinstance(raw_aliases, str):
                    raw_aliases = [raw_aliases]
                if isinstance(raw_aliases, list):
                    aliases.extend(str(alias).strip() for alias in raw_aliases if str(alias).strip())
                entity = str(item.get("entity", "") or "").strip()
                if entity:
                    aliases.append(entity)
            else:
                canonical = str(item or "").strip()
                aliases = [canonical]
            aliases = list(dict.fromkeys(alias for alias in aliases if alias))
            if canonical and aliases:
                groups.append((canonical, aliases))
    if groups:
        return groups
    return _target_alias_groups(params)


def _row_series_name(row: dict[str, Any], groups: list[tuple[str, list[str]]]) -> str:
    series_fields = [
        "series_name",
        "target_entity",
        "entity_label",
        "canonical_entity",
        "linked_entity",
        "entity",
        "entity_text",
        "actor",
        "attributed_actor",
        "speaker",
        "label",
        "term",
    ]
    for field in series_fields:
        value = str(row.get(field, "") or "").strip()
        if not value:
            continue
        if groups:
            matched, _alias = _match_target_alias(value, groups)
            if matched:
                return matched
        else:
            canonical = _canonical_entity(value)
            if canonical and _valid_entity_surface(canonical):
                return canonical
    if groups:
        text = " ".join(str(row.get(field, "") or "") for field in ("claim_span", "excerpt", "text", "title"))
        matched, _alias = _match_target_alias(text, groups)
        if matched:
            return matched
    return ""


def _metric_value(row: dict[str, Any], metric: dict[str, Any]) -> float | None:
    aggregation = str(metric.get("aggregation", metric.get("agg", "")) or "").strip().lower()
    if aggregation == "count":
        return 1.0
    metric_name = str(metric.get("name") or metric.get("as") or metric.get("output_field") or "").strip()
    field_candidates = [
        str(metric.get("field", "") or "").strip(),
        str(metric.get("value_field", "") or "").strip(),
        str(metric.get("metric", "") or "").strip(),
        metric_name,
        "score",
        "value",
        "count",
        "mention_count",
        "document_frequency",
        "weight",
    ]
    for field in field_candidates:
        if not field:
            continue
        value = _plot_float(row.get(field))
        if value is not None:
            return value
    return None


def _time_series_rows_from_metric_specs(
    params: dict[str, Any],
    deps: dict[str, ToolExecutionResult],
    context: AgentExecutionContext,
    *,
    granularity: str,
) -> tuple[list[dict[str, Any]], int] | None:
    raw_metrics = params.get("metrics")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        return None
    metrics = [
        metric
        for metric in raw_metrics
        if isinstance(metric, dict)
        and str(metric.get("name") or metric.get("as") or metric.get("output_field") or metric.get("field") or "").strip()
    ]
    if not metrics:
        return None
    groups = _series_definitions(params)
    values: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    aggregations: dict[str, str] = {}
    doc_sets: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    skipped = 0
    for metric in metrics:
        metric_name = str(metric.get("name") or metric.get("as") or metric.get("output_field") or metric.get("field") or "").strip()
        aggregations[metric_name] = str(metric.get("aggregation", metric.get("agg", "sum")) or "sum").strip().lower()
        source = str(metric.get("source", metric.get("source_node", metric.get("source_node_id", ""))) or "").strip()
        for row in _time_series_source_rows(params, deps, context, source):
            series = _row_series_name(row, groups) if groups else "__all__"
            if not series:
                skipped += 1
                continue
            timestamp = str(_first_nonempty_field(row, ["published_at", "date", "time_bin", "month", "period", "time_period"]))
            time_bin = _time_bin(timestamp, granularity)
            value = _metric_value(row, metric)
            if value is None:
                skipped += 1
                continue
            values[(series, time_bin, metric_name)].append(value)
            doc_id = str(row.get("doc_id", "") or "").strip()
            if doc_id:
                doc_sets[(series, time_bin)].add(doc_id)
    if not values:
        return [], skipped
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for (series, time_bin, metric_name), metric_values in values.items():
        target = rows_by_key.setdefault(
            (series, time_bin),
            {
                "series_name": series,
                "target_entity": series,
                "entity": series,
                "canonical_entity": series,
                **_time_bin_fields(time_bin, granularity),
            },
        )
        aggregation = aggregations.get(metric_name, "sum")
        if aggregation in {"mean", "avg", "average"}:
            target[metric_name] = round(sum(metric_values) / max(len(metric_values), 1), 6)
        elif aggregation == "max":
            target[metric_name] = max(metric_values)
        elif aggregation == "min":
            target[metric_name] = min(metric_values)
        else:
            total = sum(metric_values)
            target[metric_name] = int(total) if float(total).is_integer() else round(total, 6)
        target["document_frequency"] = len(doc_sets.get((series, time_bin), set()))
    rows = list(rows_by_key.values())
    for row in rows:
        if "mention_count" in row:
            row["count"] = row["mention_count"]
    rows.sort(key=lambda row: (str(row.get("time_bin", "")), str(row.get("series_name", ""))))
    return rows, skipped


def _entity_frequency_rows(params: dict[str, Any], deps: dict[str, ToolExecutionResult], *, task: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    all_rows = _payload_rows_from_all_dependencies(deps)
    mention_rows = [row for row in all_rows if str(row.get("doc_id", "")).strip()]
    source_rows = mention_rows if any("entity" in row or "speaker" in row or "attributed_actor" in row for row in mention_rows) else all_rows
    requested_entity_field = str(params.get("entity_field", params.get("actor_field", "")) or "").strip()
    fallback_entity_field = str(params.get("fallback_entity_field", params.get("fallback_series_key", "")) or "").strip()
    entity_fields = [
        requested_entity_field,
        fallback_entity_field,
        str(params.get("linked_name_field", "") or "").strip(),
        str(params.get("linked_id_field", "") or "").strip(),
        str(params.get("entity_text_field", "") or "").strip(),
        "canonical_entity",
        "entity_canonical",
        "linked_entity_name",
        "linked_entity",
        "linked_entity_id",
        "entity_text",
        "attributed_actor",
        "actor",
        "speaker",
        "entity",
        "name",
        "label",
    ]
    timestamp_field = str(params.get("timestamp_field", params.get("published_at_field", params.get("time_field", ""))) or "").strip()
    raw_entity_types = params.get("entity_types") or params.get("include_entity_types") or params.get("allowed_entity_labels") or []
    if isinstance(raw_entity_types, str):
        allowed_labels = {raw_entity_types.upper()}
    elif isinstance(raw_entity_types, list):
        allowed_labels = {str(item).upper() for item in raw_entity_types if str(item).strip()}
    else:
        allowed_labels = set()
    ignored_labels = {"DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT"}
    group_by_time = bool(params.get("group_by_time") or "time_series" in task or "over_time" in task or "monthly" in task)
    top_n = _int_param(params, "top_n", "top_k", "limit", default=100, maximum=5000)
    min_df = _int_param(params, "min_document_frequency", "min_doc_frequency", default=1, maximum=100000)
    quote_mode = "quote" in task or str(params.get("actor_field", "")).strip() in {"attributed_actor", "speaker"}
    mention_counts: Counter[tuple[str, str]] = Counter()
    quote_counts: Counter[tuple[str, str]] = Counter()
    doc_sets: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    total_docs_by_time: defaultdict[str, set[str]] = defaultdict(set)
    total_mentions = 0
    skipped = 0
    skipped_time = 0
    label_fields = [
        str(params.get("entity_label_field", "") or "").strip(),
        "label",
        "entity_label",
        "entity_type",
        "type",
    ]
    for row in source_rows:
        label = str(_first_nonempty_field(row, label_fields) or "").upper()
        if allowed_labels and label and label not in allowed_labels:
            skipped += 1
            continue
        if not allowed_labels and label in ignored_labels:
            skipped += 1
            continue
        entity = _canonical_entity(_first_nonempty_field(row, entity_fields))
        if not _valid_entity_surface(entity):
            skipped += 1
            continue
        time_bin = _entity_row_time_bin(row, timestamp_field)
        time_key = time_bin if group_by_time else "__overall__"
        if group_by_time and _is_placeholder_axis_label(time_key):
            skipped_time += 1
            continue
        doc_id = str(row.get("doc_id", "")).strip()
        key = (entity, time_key)
        mention_counts[key] += 1
        total_mentions += 1
        if quote_mode or row.get("quote") not in (None, ""):
            quote_counts[key] += 1
        if doc_id:
            doc_sets[key].add(doc_id)
            total_docs_by_time[time_key].add(doc_id)
    overall_doc_sets: defaultdict[str, set[str]] = defaultdict(set)
    overall_mentions: Counter[str] = Counter()
    for (entity, time_key), count in mention_counts.items():
        overall_mentions[entity] += count
        overall_doc_sets[entity].update(doc_sets.get((entity, time_key), set()))
    eligible_entities = {
        entity
        for entity, docs in overall_doc_sets.items()
        if len(docs) >= min_df or not any(doc_sets.values())
    }
    if not eligible_entities and min_df > 1:
        eligible_entities = set(overall_mentions)
        min_df = 1
    ranked_entities = [
        entity
        for entity, _ in sorted(
            overall_mentions.items(),
            key=lambda item: (len(overall_doc_sets.get(item[0], set())), item[1], item[0].lower()),
            reverse=True,
        )
        if entity in eligible_entities
    ][:top_n]
    ranked_set = set(ranked_entities)
    rows: list[dict[str, Any]] = []
    total_doc_count = len(set().union(*total_docs_by_time.values())) if total_docs_by_time else 0
    for (entity, time_key), mention_count in sorted(
        mention_counts.items(),
        key=lambda item: (item[0][1], item[0][0].lower()),
    ):
        if entity not in ranked_set:
            continue
        docs = doc_sets.get((entity, time_key), set())
        period_doc_count = len(total_docs_by_time.get(time_key, set())) if group_by_time else total_doc_count
        doc_frequency = len(docs)
        quote_count = quote_counts.get((entity, time_key), 0)
        prominence_score = doc_frequency + (0.25 * mention_count) + (2.0 * quote_count)
        row = {
            "entity": entity,
            "canonical_entity": entity,
            "entity_text": entity,
            "linked_entity": entity,
            "actor": entity,
            "attributed_actor": entity,
            "mention_count": int(mention_count),
            "count": int(mention_count),
            "document_frequency": int(doc_frequency),
            "doc_frequency": int(doc_frequency),
            "quote_count": int(quote_count),
            "prominence_score": round(prominence_score, 6),
            "share_of_mentions": round(mention_count / max(total_mentions, 1), 6),
            "share_of_docs": round(doc_frequency / max(period_doc_count, 1), 6),
        }
        row["mention_share"] = row["share_of_mentions"]
        row["doc_share"] = row["share_of_docs"]
        row["share_of_documents"] = row["share_of_docs"]
        row["composite_prominence"] = row["prominence_score"]
        row["mention_count_normalized"] = row["share_of_docs"]
        if group_by_time:
            row.update({"time_bin": time_key, "month": time_key, "period": time_key, "time_period": time_key})
        rows.append(row)
    if not group_by_time:
        rows.sort(key=lambda row: (float(row.get("document_frequency", 0)), float(row.get("mention_count", 0))), reverse=True)
        rows = rows[:top_n]
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
    else:
        rows.sort(key=lambda row: (str(row.get("time_bin", "")), -float(row.get("prominence_score", 0)), str(row.get("entity", ""))))
    caveats = []
    if skipped:
        caveats.append(f"Skipped {skipped} entity/actor rows with unsupported labels or unusable names.")
    if skipped_time:
        caveats.append(f"Skipped {skipped_time} entity/actor rows with missing or placeholder time values.")
    metadata = {
        "provider": "analytics_entity_frequency",
        "task": task,
        "input_row_count": len(source_rows),
        "entity_count": len(ranked_entities),
        "group_by_time": group_by_time,
        "min_document_frequency": min_df,
    }
    return rows, metadata, caveats


def _merge_actor_prominence_rows(params: dict[str, Any], deps: dict[str, ToolExecutionResult]) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    aggregate_rows = _payload_rows_from_all_dependencies(deps)
    combined: dict[tuple[str, str], dict[str, Any]] = {}
    doc_sets: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in aggregate_rows:
        actor = _canonical_entity(_first_nonempty_field(row, ["actor", "canonical_entity", "linked_entity", "attributed_actor", "entity_text", "entity", "speaker"]))
        if not _valid_entity_surface(actor):
            continue
        month = _entity_row_time_bin(row)
        key = (actor, month)
        target = combined.setdefault(
            key,
            {
                "actor": actor,
                "entity": actor,
                "canonical_entity": actor,
                "entity_text": actor,
                "linked_entity": actor,
                "attributed_actor": actor,
                "month": month,
                "time_bin": month,
                "period": month,
                "time_period": month,
            },
        )
        saw_numeric = False
        for field in ("document_frequency", "doc_frequency", "mention_count", "quote_count", "count"):
            value = _plot_float(row.get(field))
            if value is not None:
                canonical = "document_frequency" if field == "doc_frequency" else field
                target[canonical] = float(target.get(canonical, 0.0) or 0.0) + value
                saw_numeric = True
        if not saw_numeric and any(name in row for name in ("entity", "canonical_entity", "linked_entity", "speaker", "attributed_actor")):
            target["mention_count"] = float(target.get("mention_count", 0.0) or 0.0) + 1.0
        if row.get("quote") not in (None, ""):
            target["quote_count"] = float(target.get("quote_count", 0.0) or 0.0) + 1.0
        doc_id = str(row.get("doc_id", "")).strip()
        if doc_id:
            doc_sets[key].add(doc_id)
    rows = []
    for key, row in combined.items():
        if doc_sets.get(key):
            row["document_frequency"] = max(float(row.get("document_frequency", 0.0) or 0.0), float(len(doc_sets[key])))
        doc_frequency = float(row.get("document_frequency", 0.0) or 0.0)
        mention_count = float(row.get("mention_count", row.get("count", 0.0)) or 0.0)
        quote_count = float(row.get("quote_count", 0.0) or 0.0)
        row["doc_mentions"] = doc_frequency
        row["mentioned_in_doc"] = 1.0 if doc_frequency > 0 else 0.0
        row["quoted_in_doc"] = 1.0 if quote_count > 0 else 0.0
        row["count"] = mention_count
        row["prominence_score"] = round(doc_frequency + (0.25 * mention_count) + (2.0 * quote_count), 6)
        rows.append(row)
    totals_by_month: defaultdict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        month = str(row.get("month", "unknown"))
        totals_by_month[month]["mention_count"] += float(row.get("mention_count", row.get("count", 0.0)) or 0.0)
        totals_by_month[month]["doc_mentions"] += float(row.get("doc_mentions", row.get("document_frequency", 0.0)) or 0.0)
        totals_by_month[month]["quote_count"] += float(row.get("quote_count", 0.0) or 0.0)
    for row in rows:
        month = str(row.get("month", "unknown"))
        totals = totals_by_month[month]
        mention_count = float(row.get("mention_count", row.get("count", 0.0)) or 0.0)
        doc_mentions = float(row.get("doc_mentions", row.get("document_frequency", 0.0)) or 0.0)
        quote_count = float(row.get("quote_count", 0.0) or 0.0)
        mention_share = mention_count / max(totals["mention_count"], 1.0)
        doc_share = doc_mentions / max(totals["doc_mentions"], 1.0)
        quote_share = quote_count / max(totals["quote_count"], 1.0)
        row["mention_share"] = round(mention_share, 6)
        row["share_of_mentions"] = row["mention_share"]
        row["doc_share"] = round(doc_share, 6)
        row["share_of_documents"] = row["doc_share"]
        row["quote_share"] = round(quote_share, 6)
        row["composite_prominence"] = round((0.45 * doc_share) + (0.35 * mention_share) + (0.20 * quote_share), 6)
        row["mention_count_normalized"] = row["mention_share"]
    rows.sort(key=lambda row: (str(row.get("month", "")), -float(row.get("prominence_score", 0.0)), str(row.get("actor", ""))))
    return rows, {"provider": "analytics_actor_prominence_merge", "input_row_count": len(aggregate_rows)}, []


def _is_syntax_role_evidence_task(task: str, params: dict[str, Any]) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(task or "").strip().lower()).strip("_")
    if normalized in {"syntax_role_evidence", "svo_evidence", "subject_verb_object_evidence"}:
        return True
    requested = " ".join(str(value or "") for value in (task, params.get("task_name"), params.get("description")))
    requested = requested.lower()
    return bool(("evidence" in requested or "examples" in requested) and ("svo" in requested or "subject-verb-object" in requested))


def _row_has_syntax_role_shape(row: dict[str, Any]) -> bool:
    explicit_fields = ("subject", "verb", "object", "semantic_actor", "semantic_target")
    if any(str(row.get(field, "")).strip() for field in explicit_fields):
        return True
    return bool(
        str(row.get("actor_group", "")).strip()
        and str(row.get("target_group", "")).strip()
        and str(row.get("sentence", row.get("excerpt", row.get("text", "")))).strip()
    )


def _syntax_role_evidence_rows(params: dict[str, Any], deps: dict[str, ToolExecutionResult]) -> list[dict[str, Any]]:
    limit = _int_param(params, "top_k", "limit", "max_rows", default=100, maximum=1000)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    for result in deps.values():
        payload = result.payload if isinstance(result.payload, dict) else {}
        upstream_rows = payload.get("rows")
        if not isinstance(upstream_rows, list):
            continue
        for item in upstream_rows:
            if not isinstance(item, dict) or not _row_has_syntax_role_shape(item):
                continue
            doc_id = str(item.get("doc_id", "") or "").strip()
            subject = str(item.get("subject", item.get("semantic_actor", "")) or "").strip()
            verb = str(item.get("verb", item.get("predicate", "")) or "").strip()
            obj = str(item.get("object", item.get("semantic_target", "")) or "").strip()
            actor = str(item.get("semantic_actor", subject) or subject).strip()
            target = str(item.get("semantic_target", obj) or obj or subject).strip()
            sentence = str(item.get("sentence", item.get("excerpt", item.get("text", ""))) or "").strip()
            if not sentence and (actor or verb or target):
                sentence = " ".join(part for part in (actor, verb, target) if part).strip()
            if not sentence:
                continue
            dedupe_key = (doc_id, sentence, subject, verb, obj, actor, target)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            score = _coerce_score(item.get("score", item.get("mention_count", item.get("count", 1.0))))
            if score <= 0:
                score = 1.0
            role_pattern = " -> ".join(part for part in (actor or subject, verb or "acts on", target or obj) if part)
            rows.append(
                {
                    "doc_id": doc_id,
                    "outlet": str(item.get("outlet", item.get("source", "")) or ""),
                    "date": str(item.get("date", item.get("published_at", item.get("time_bin", ""))) or ""),
                    "published_at": str(item.get("published_at", item.get("date", "")) or ""),
                    "excerpt": sentence[:320],
                    "score": score,
                    "score_display": _score_display(score),
                    "subject": subject,
                    "verb": verb,
                    "object": obj,
                    "semantic_actor": actor,
                    "semantic_target": target,
                    "actor_group": str(item.get("actor_group", "") or ""),
                    "target_group": str(item.get("target_group", "") or ""),
                    "voice": str(item.get("voice", "") or ""),
                    "role_pattern": role_pattern,
                    "provider": str(item.get("provider", "") or ""),
                }
            )
    rows.sort(key=lambda row: (-_coerce_score(row.get("score", 0.0)), str(row.get("date", "")), str(row.get("role_pattern", ""))))
    for rank, row in enumerate(rows[:limit], start=1):
        row["rank"] = rank
    return rows[:limit]


def _build_evidence_table(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    params = _analysis_params(params)
    task = _task_name(params)
    if _is_noun_frequency_task(task, params):
        top_k = _int_param(params, "top_k", "limit", default=100, maximum=5000)
        documents = _text_rows(deps)
        working_set_ref = _resolve_working_set_ref(params, deps)
        documents_truncated = _dependency_payload_flag(deps, "documents_truncated") or _dependency_payload_flag(deps, "working_set_truncated")
        full_document_count = _dependency_payload_int(deps, "document_count", len(documents))
        if working_set_ref and (documents_truncated or not documents):
            noun_rows, full_metadata = _noun_frequency_rows_from_working_set(
                context,
                working_set_ref,
                top_k=top_k,
            )
            caveats = [
                (
                    "Noun distribution was computed by streaming the full working_set_ref in batches "
                    "instead of relying on preview-only or ID-only upstream rows."
                )
            ]
            if full_document_count and full_metadata.get("analyzed_document_count") != full_document_count:
                caveats.append(
                    f"Expected {full_document_count} working-set documents but analyzed {full_metadata.get('analyzed_document_count', 0)}."
                )
            if full_metadata.get("analysis_document_limit") and full_document_count > int(full_metadata.get("analysis_document_limit") or 0):
                caveats.append(
                    f"Working-set linguistic aggregation was capped at {full_metadata.get('analysis_document_limit')} documents; "
                    "set CORPUSAGENT2_WORKING_SET_ANALYSIS_MAX_DOCS=-1 for an uncapped offline run."
                )
            if full_metadata.get("provider_fallback_reason"):
                caveats.append(
                    "Provider POS tagging was skipped for this large working set: "
                    f"{full_metadata.get('provider_fallback_reason')}."
                )
            if not noun_rows:
                caveats.append("No noun distribution rows were produced from the full working set.")
            return ToolExecutionResult(
                payload={"rows": noun_rows, **full_metadata},
                evidence=[],
                caveats=caveats,
                metadata={"no_data": not noun_rows, "task": task, **full_metadata},
            )
        pos_rows = []
        for result in deps.values():
            payload = result.payload if isinstance(result.payload, dict) else {}
            rows = payload.get("rows")
            if not isinstance(rows, list) or not rows:
                continue
            first = rows[0]
            if isinstance(first, dict) and ("pos" in first or "lemma" in first):
                pos_rows = [dict(item) for item in rows if isinstance(item, dict)]
                break
        if not pos_rows and documents:
            noun_rows, noun_metadata = _noun_frequency_rows_from_documents(
                documents,
                top_k=top_k,
                full_working_set=False,
                working_set_ref=working_set_ref,
            )
            caveats = [] if noun_rows else ["No noun distribution rows were produced from the upstream documents."]
            if noun_rows:
                noun_provider = str(noun_metadata.get("provider") or "").strip()
                if noun_provider.startswith("heuristic"):
                    caveats.append(
                        "No POS rows were provided, so noun distribution was computed with explicit heuristic token/POS fallback over fetched documents."
                    )
                else:
                    caveats.append(
                        "No POS rows were provided, so noun distribution was computed directly "
                        f"with provider-backed {noun_provider or 'NLP'} over fetched documents."
                    )
            if documents_truncated:
                caveats.append(
                    "Noun distribution is preview-only because upstream documents were truncated and no batch working_set_ref was available."
                )
            return ToolExecutionResult(
                payload={"rows": noun_rows, "source_document_count": full_document_count, **noun_metadata},
                evidence=[],
                caveats=caveats,
                metadata={
                    "no_data": not noun_rows,
                    "task": task,
                    "preview_only": bool(documents_truncated),
                    **noun_metadata,
                },
            )
        noun_rows = _noun_frequency_rows(documents, pos_rows, top_k=top_k)
        caveats = [] if noun_rows else ["No noun distribution rows were produced from the upstream documents and POS rows."]
        if documents_truncated:
            caveats.append(
                "Noun distribution is preview-only because upstream documents were truncated and no batch working_set_ref was available."
            )
        return ToolExecutionResult(
            payload={"rows": noun_rows, "analyzed_document_count": len(documents), "source_document_count": full_document_count},
            evidence=[],
            caveats=caveats,
            metadata={"no_data": not noun_rows, "task": task, "preview_only": bool(documents_truncated)},
        )
    if task == "summary_stats":
        documents = _text_rows(deps)
        upstream_rows = []
        matched_document_count: int | None = None
        for result in deps.values():
            payload = result.payload if isinstance(result.payload, dict) else {}
            if payload.get("analyzed_document_count") not in (None, ""):
                try:
                    matched_document_count = int(payload.get("analyzed_document_count") or 0)
                except (TypeError, ValueError):
                    matched_document_count = matched_document_count
            rows = payload.get("rows")
            if not isinstance(rows, list) or not rows:
                continue
            first = rows[0]
            if isinstance(first, dict) and "lemma" in first and "count" in first:
                upstream_rows = [dict(item) for item in rows if isinstance(item, dict)]
                break
        summary_rows = _summary_stat_rows(documents, upstream_rows, matched_document_count=matched_document_count)
        caveats = [] if summary_rows else ["No summary statistics rows were produced from the upstream aggregation."]
        return ToolExecutionResult(
            payload={"rows": summary_rows},
            evidence=[],
            caveats=caveats,
            metadata={"no_data": not summary_rows, "task": task},
        )
    if _is_entity_frequency_task(task, params):
        entity_rows, entity_metadata, caveats = _entity_frequency_rows(params, deps, task=task)
        if not entity_rows:
            caveats = [*caveats, "No entity or actor frequency rows were produced from upstream rows."]
        return ToolExecutionResult(
            payload={"rows": entity_rows, **entity_metadata},
            evidence=[],
            caveats=caveats,
            metadata={"no_data": not entity_rows, **entity_metadata},
        )
    if _is_syntax_role_evidence_task(task, params):
        evidence_rows = _syntax_role_evidence_rows(params, deps)
        caveats = [] if evidence_rows else ["No subject-verb-object rows were available for syntax role evidence."]
        metadata = {
            "no_data": not evidence_rows,
            "task": task,
            "provider": "analytics_syntax_role_evidence",
            "input_dependency_count": len(deps),
            "evidence_row_count": len(evidence_rows),
        }
        if not evidence_rows:
            metadata["no_data_reason"] = "no_svo_triples"
        return ToolExecutionResult(
            payload={"rows": evidence_rows},
            evidence=evidence_rows,
            caveats=caveats,
            metadata=metadata,
        )
    rows = []
    search_score_lookup = {
        str(row.get("doc_id", "")).strip(): _coerce_score(row.get("score", 0.0))
        for row in _search_rows(deps)
        if str(row.get("doc_id", "")).strip()
    }
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload:
            rows = list(payload["rows"])
    evidence_map: dict[str, EvidenceRow] = {}
    for row in rows:
        doc_id = str(row.get("doc_id", ""))
        if not doc_id:
            continue
        score = _coerce_score(row.get("score", search_score_lookup.get(doc_id, 0.0)))
        if score == 0.0 and doc_id in search_score_lookup:
            score = search_score_lookup[doc_id]
        current = evidence_map.get(doc_id)
        candidate = EvidenceRow(
            doc_id=doc_id,
            outlet=str(row.get("outlet", "")),
            date=str(row.get("date", "")),
            excerpt=str(row.get("excerpt", row.get("quote", "")))[:320],
            score=score,
        )
        if current is None or candidate.score > current.score:
            evidence_map[doc_id] = candidate
    evidence = []
    for rank, item in enumerate(sorted(evidence_map.values(), key=lambda item: item.score, reverse=True), start=1):
        payload = item.to_dict()
        payload["score_display"] = _score_display(item.score)
        payload["rank"] = rank
        evidence.append(payload)
    return ToolExecutionResult(payload={"rows": evidence}, evidence=evidence)


def _fetch_yfinance_series_rows(
    *,
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> list[dict[str, Any]]:
    normalized_ticker = ticker.strip().upper()
    cache_key = (normalized_ticker, start.strip(), end.strip(), interval.strip())
    if cache_key in _YFINANCE_SERIES_CACHE:
        return [dict(row) for row in _YFINANCE_SERIES_CACHE[cache_key]]
    if not _module_available("yfinance"):
        raise RuntimeError("yfinance is not installed.")
    import yfinance as yf

    requested_ticker = normalized_ticker
    history = _yfinance_download_history(yf, normalized_ticker, start=start, end=end, interval=interval)
    if history is None or history.empty:
        for candidate in _yfinance_equity_symbol_candidates(yf, normalized_ticker):
            if candidate == normalized_ticker:
                continue
            candidate_history = _yfinance_download_history(yf, candidate, start=start, end=end, interval=interval)
            if candidate_history is not None and not candidate_history.empty:
                normalized_ticker = candidate
                history = candidate_history
                break
    if history is None or history.empty:
        _YFINANCE_SERIES_CACHE[cache_key] = []
        return []
    if hasattr(history.columns, "levels"):
        history.columns = history.columns.get_level_values(0)
    frame = history.reset_index()
    date_column = "Date" if "Date" in frame.columns else frame.columns[0]
    rows: list[dict[str, Any]] = []
    previous_close: float | None = None
    normalized_interval = str(interval or "1d").strip().lower()
    series_granularity = "day" if normalized_interval in {"1d", "1h", "60m", "30m", "15m", "5m", "2m", "1m"} else "month"
    for row in frame.itertuples(index=False):
        record = row._asdict()
        raw_date = record.get(date_column)
        date_text = str(raw_date)[:10]
        close_value = float(record.get("Close", 0.0) or 0.0)
        open_value = float(record.get("Open", 0.0) or 0.0)
        high_value = float(record.get("High", 0.0) or 0.0)
        low_value = float(record.get("Low", 0.0) or 0.0)
        volume_value = float(record.get("Volume", 0.0) or 0.0)
        daily_return = 0.0 if previous_close in (None, 0.0) else (close_value - previous_close) / previous_close
        drawdown = 0.0 if previous_close in (None, 0.0) else min(daily_return, 0.0)
        rows.append(
            {
                "ticker": normalized_ticker,
                "requested_ticker": requested_ticker,
                "ticker_resolution": "yfinance_search_fallback" if normalized_ticker != requested_ticker else "exact",
                "date": date_text,
                "time_bin": _time_bin(date_text, series_granularity),
                "market_open": open_value,
                "market_high": high_value,
                "market_low": low_value,
                "market_close": close_value,
                "market_volume": volume_value,
                "market_return": round(daily_return, 6),
                "market_drawdown": round(drawdown, 6),
            }
        )
        previous_close = close_value
    _YFINANCE_SERIES_CACHE[cache_key] = [dict(row) for row in rows]
    return rows


def _infer_external_series_bounds(left_rows: list[dict[str, Any]], key: str) -> tuple[str, str]:
    values = []
    for row in left_rows:
        raw = str(
            _first_nonempty_field(
                row,
                [key, "time_bin", "date", "published_at", "month", "period", "time_period"],
            )
        ).strip()
        if not raw or raw.lower() in {"unknown", "unkn", "nan", "none", "nat"}:
            continue
        if re.fullmatch(r"\d{4}(-\d{2})?(-\d{2})?", raw):
            values.append(raw)
    if not values:
        return "", ""
    first = min(values)
    last = max(values)

    def _period_start(value: str) -> str:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return value
        if re.fullmatch(r"\d{4}-\d{2}", value):
            return f"{value}-01"
        if re.fullmatch(r"\d{4}", value):
            return f"{value}-01-01"
        return ""

    def _exclusive_period_end(value: str) -> str:
        start_text = _period_start(value)
        if not start_text:
            return ""
        start_date = pd.to_datetime(start_text, errors="coerce")
        if pd.isna(start_date):
            return ""
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            end_date = start_date + pd.Timedelta(days=1)
        elif re.fullmatch(r"\d{4}-\d{2}", value):
            end_date = start_date + pd.DateOffset(months=1)
        else:
            end_date = start_date + pd.DateOffset(years=1)
        return str(end_date.date())

    return _period_start(first), _exclusive_period_end(last)


def _infer_join_time_granularity(rows: list[dict[str, Any]], key: str) -> str:
    values = [
        str(
            _first_nonempty_field(
                row,
                [key, "time_bin", "month", "period", "time_period", "date", "published_at"],
            )
        ).strip()
        for row in rows
        if isinstance(row, dict)
    ]
    values = [value for value in values if value and not _is_placeholder_axis_label(value)]
    if not values:
        return ""
    if all(re.fullmatch(r"\d{4}", value) for value in values):
        return "year"
    if all(re.fullmatch(r"\d{4}-\d{2}", value) for value in values):
        return "month"
    if all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in values):
        return "day"
    if any(re.fullmatch(r"\d{4}-\d{2}", value) for value in values):
        return "month"
    return ""


def _align_external_series_join_key(
    right_df: pd.DataFrame,
    *,
    right_key: str,
    target_granularity: str,
) -> tuple[pd.DataFrame, bool]:
    normalized_granularity = str(target_granularity or "").strip().lower()
    if normalized_granularity not in {"day", "month", "year"} or right_df.empty:
        return right_df, False
    if right_key not in right_df.columns:
        return right_df, False
    source_column = "date" if "date" in right_df.columns else right_key
    aligned = right_df.copy()
    original_values = aligned[right_key].astype(str).tolist()
    aligned[right_key] = aligned[source_column].astype(str).map(lambda value: _time_bin(value, normalized_granularity))
    return aligned, aligned[right_key].astype(str).tolist() != original_values


def _join_external_series(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    params = _merged_payload_params(params)
    external_spec = params.get("external_series") if isinstance(params.get("external_series"), dict) else {}
    date_range = params.get("date_range") if isinstance(params.get("date_range"), dict) else {}
    external_rows = list(params.get("series_rows") or external_spec.get("series_rows") or external_spec.get("rows") or [])
    ticker = str(
        params.get("ticker")
        or params.get("symbol")
        or external_spec.get("ticker")
        or external_spec.get("symbol")
        or ""
    ).strip() or _infer_market_ticker_from_text(
        str(getattr(context.state, "rewritten_question", "") or getattr(context.state, "question", ""))
    )
    start = str(
        params.get("start")
        or params.get("date_from")
        or date_range.get("start")
        or date_range.get("date_from")
        or external_spec.get("start")
        or external_spec.get("start_date")
        or ""
    ).strip()
    end = str(
        params.get("end")
        or params.get("date_to")
        or date_range.get("end")
        or date_range.get("date_to")
        or external_spec.get("end")
        or external_spec.get("end_date")
        or ""
    ).strip()
    requested_join_granularity = str(params.get("join_granularity") or params.get("granularity") or "").strip().lower()
    interval = str(params.get("interval") or external_spec.get("interval") or external_spec.get("frequency") or "").strip()
    interval_aliases = {"daily": "1d", "day": "1d", "weekly": "1wk", "week": "1wk", "monthly": "1mo", "month": "1mo"}
    if not interval and requested_join_granularity in interval_aliases:
        interval = requested_join_granularity
    if not interval:
        interval = "1d"
    interval = interval_aliases.get(interval.lower(), interval)
    left_rows: list[dict[str, Any]] = []
    bound_rows: list[dict[str, Any]] = []
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload:
            left_rows = list(payload["rows"])
            if left_rows:
                break
        if isinstance(payload, dict):
            for key in ("documents", "results"):
                values = payload.get(key)
                if isinstance(values, list) and values and not bound_rows:
                    bound_rows = [dict(item) for item in values if isinstance(item, dict)]
                    break

    caveats: list[str] = []
    provider = "analytics"
    left_key = str(params.get("left_key", "time_bin"))
    right_key = str(params.get("right_key", left_key))
    if ticker and (not start or not end):
        inferred_start, inferred_end = _infer_external_series_bounds(left_rows or bound_rows, left_key)
        if inferred_start and not start:
            start = inferred_start
        if inferred_end and not end:
            end = inferred_end
        if inferred_start or inferred_end:
            caveats.append("Inferred external series date range from the internal corpus time bins.")
    if not external_rows and ticker:
        try:
            external_rows = _fetch_yfinance_series_rows(
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
            )
            provider = "yfinance"
            if external_rows:
                resolved_ticker = str(external_rows[0].get("ticker", "") or "").strip()
                requested_ticker = str(external_rows[0].get("requested_ticker", "") or ticker).strip()
                if resolved_ticker and requested_ticker and resolved_ticker != requested_ticker:
                    caveats.append(
                        f"Resolved requested market ticker '{requested_ticker}' to yfinance symbol '{resolved_ticker}' after the requested symbol returned no rows."
                    )
        except Exception as exc:
            caveats.append(f"External market series fetch failed for ticker '{ticker}': {exc}")
    if not left_rows:
        if external_rows:
            caveats.append(
                "No internal aggregate rows were available to join, so the external series is returned as a standalone time series."
            )
            return ToolExecutionResult(
                payload={"rows": external_rows},
                caveats=caveats,
                metadata=_metadata(provider, f"{provider}_external_series", ticker=ticker, interval=interval),
            )
        return ToolExecutionResult(
            payload={"rows": []},
            caveats=caveats or ["No internal rows or external series rows were available."],
            metadata={"no_data": True, "ticker": ticker, "interval": interval},
        )

    left_df = pd.DataFrame(left_rows)
    right_df = pd.DataFrame(external_rows)
    if right_df.empty:
        if ticker and not caveats:
            caveats.append(f"No external market series rows were returned for ticker '{ticker}'.")
        elif not caveats:
            caveats.append("No external series rows were provided.")
        return ToolExecutionResult(
            payload={"rows": left_rows},
            caveats=caveats,
            metadata=_metadata(provider, f"{provider}_join_external_series", ticker=ticker),
        )

    if left_key not in left_df.columns:
        if "date" in left_df.columns:
            left_df[left_key] = left_df["date"].astype(str).map(_time_bin)
        else:
            return ToolExecutionResult(payload={"rows": left_rows}, caveats=[f"Join key '{left_key}' is missing from internal rows."])
    if right_key not in right_df.columns:
        return ToolExecutionResult(payload={"rows": left_rows}, caveats=[f"Join key '{right_key}' is missing from external series rows."])
    target_granularity = _infer_join_time_granularity(left_rows, left_key)
    if not target_granularity and requested_join_granularity in {"day", "month", "year"}:
        target_granularity = requested_join_granularity
    right_df, aligned_external_key = _align_external_series_join_key(
        right_df,
        right_key=right_key,
        target_granularity=target_granularity,
    )
    if aligned_external_key:
        caveats.append(f"Aligned external series to {target_granularity} time bins before joining with corpus rows.")
    if right_df[right_key].duplicated().any():
        aggregations: dict[str, str] = {}
        for column, method in (
            ("ticker", "first"),
            ("date", "last"),
            ("market_open", "first"),
            ("market_high", "max"),
            ("market_low", "min"),
            ("market_close", "last"),
            ("market_volume", "sum"),
            ("market_drawdown", "min"),
        ):
            if column in right_df.columns:
                aggregations[column] = method
        right_df = right_df.sort_values([right_key, "date"] if "date" in right_df.columns else [right_key])
        if aggregations:
            grouped_right = right_df.groupby(right_key, as_index=False).agg(aggregations)
            if {"market_open", "market_close"}.issubset(grouped_right.columns):
                grouped_right["market_return"] = grouped_right.apply(
                    lambda row: round((float(row["market_close"]) - float(row["market_open"])) / float(row["market_open"]), 6)
                    if float(row.get("market_open") or 0.0) else 0.0,
                    axis=1,
                )
            elif "market_return" in right_df.columns:
                returns = right_df.groupby(right_key, as_index=False)["market_return"].sum()
                grouped_right = grouped_right.merge(returns, on=right_key, how="left")
            right_df = grouped_right
            caveats.append(f"Aggregated external series to one row per '{right_key}' before joining with corpus rows.")
        else:
            right_df = right_df.drop_duplicates(subset=[right_key], keep="last")
            caveats.append(f"Deduplicated external series to one row per '{right_key}' before joining with corpus rows.")

    merged = left_df.merge(
        right_df,
        how=str(params.get("how") or params.get("join_type") or "left"),
        left_on=left_key,
        right_on=right_key,
        suffixes=("", "_external"),
    )
    return ToolExecutionResult(
        payload={"rows": merged.fillna("").to_dict(orient="records")},
        caveats=caveats,
        metadata=_metadata(provider, f"{provider}_join_external_series", ticker=ticker, interval=interval),
    )


PLOT_FIELD_ALIASES = {
    "noun_lemma": ("lemma", "term", "token"),
    "noun": ("lemma", "term", "token"),
    "label": ("label", "name", "lemma", "entity", "term", "source", "outlet", "time_bin", "doc_id"),
    "name": ("name", "label", "entity", "term", "lemma"),
    "x": ("time_bin", "date", "label", "name", "lemma", "entity", "term", "source", "doc_id"),
    "category": ("label", "name", "lemma", "entity", "term", "source", "outlet"),
    "date": ("date", "published_at", "time_bin"),
    "published_at": ("published_at", "date", "time_bin", "month", "period", "time_period"),
    "published_at_month": ("time_bin", "month", "period", "time_period", "published_at", "date"),
    "published_month": ("time_bin", "month", "period", "time_period", "published_at", "date"),
    "date_month": ("time_bin", "month", "period", "time_period", "published_at", "date"),
    "time": ("time_bin", "period", "month", "date", "published_at"),
    "time_bucket": ("time_bin", "period", "month", "date", "published_at"),
    "bucket": ("bucket", "time_bin", "period", "month", "date", "published_at"),
    "period": ("period", "time_bin", "month", "date", "published_at"),
    "quarter": ("quarter", "time_bin", "period", "time_period", "month", "date", "published_at"),
    "month": ("month", "time_bin", "period", "date", "published_at"),
    "time_period": ("time_period", "time_bin", "period", "month", "date", "published_at"),
    "year": ("year", "time_bin", "period", "date", "published_at"),
    "outlet": ("outlet", "source"),
    "source": ("source", "outlet"),
    "actor": ("actor", "canonical_actor", "linked_entity", "attributed_actor", "entity", "speaker"),
    "canonical_actor": ("canonical_actor", "actor", "canonical_entity", "linked_entity", "attributed_actor", "entity", "speaker"),
    "canonical_entity": ("canonical_entity", "linked_entity", "entity", "actor", "entity_text"),
    "entity_canonical": ("canonical_entity", "entity_canonical", "linked_entity", "entity", "actor", "entity_text"),
    "entity_name": ("canonical_entity", "linked_entity", "entity", "actor", "entity_text", "name"),
    "linked_entity_name": ("linked_entity_name", "canonical_entity", "linked_entity", "entity", "actor", "entity_text"),
    "linked_entity_id": ("linked_entity_id", "linked_entity", "canonical_entity", "entity"),
    "entity_text": ("entity_text", "canonical_entity", "entity", "linked_entity", "actor"),
    "linked_entity": ("linked_entity", "entity", "actor", "attributed_actor", "speaker"),
    "attributed_actor": ("attributed_actor", "speaker", "actor", "entity", "linked_entity"),
    "speaker": ("speaker", "attributed_actor", "actor", "entity"),
    "series": ("series_name", "target_entity", "entity", "actor", "linked_entity", "attributed_actor", "speaker", "label", "term"),
    "series_name": ("series_name", "target_entity", "entity", "actor", "linked_entity", "label", "term"),
    "token": ("lemma", "term"),
    "term": ("lemma", "token"),
    "value": ("value", "count", "score", "weight", "mean", "intensity"),
    "y": ("count", "value", "score", "weight", "mean", "intensity"),
    "token_count": ("count",),
    "noun_count": ("count",),
    "frequency": ("count", "relative_frequency", "frequency", "freq"),
    "freq": ("count", "relative_frequency", "frequency"),
    "document_count": ("document_frequency", "count"),
    "document_frequency": ("document_frequency", "count", "mention_count", "doc_mentions"),
    "doc_count": ("doc_count", "document_count", "document_frequency", "count"),
    "doc_frequency": ("document_frequency", "count"),
    "df": ("document_frequency", "count"),
    "mentions": ("count", "document_frequency"),
    "mention_count": ("mention_count", "count", "document_frequency"),
    "entity_count": ("count", "document_frequency"),
    "prominence": ("prominence_score", "document_frequency", "mention_count", "count"),
    "prominence_score": ("prominence_score", "document_frequency", "mention_count", "count"),
    "composite_prominence": ("composite_prominence", "prominence_score", "document_frequency", "mention_count", "count"),
    "doc_share": ("doc_share", "share_of_docs", "document_frequency", "count"),
    "share_of_documents": ("share_of_documents", "share_of_docs", "doc_share", "document_frequency", "count"),
    "share_of_climate_docs": ("share_of_climate_docs", "share_of_documents", "doc_share", "mention_share", "share_of_mentions", "mention_count_normalized", "normalized_value", "count"),
    "document_share": ("share_of_documents", "share_of_docs", "doc_share", "document_frequency", "count"),
    "normalized_document_frequency": ("normalized_document_frequency", "normalized_value", "mention_count_normalized", "document_frequency", "count"),
    "document_frequency_normalized": ("document_frequency_normalized", "normalized_value", "mention_count_normalized", "document_frequency", "count"),
    "share_of_mentions": ("share_of_mentions", "mention_share", "mention_count", "count"),
    "mention_share": ("mention_share", "share_of_mentions", "mention_count", "count"),
    "avg_claim_sentiment": ("avg_claim_sentiment", "mean_sentiment_score", "sentiment_score", "average_sentiment", "mean_sentiment", "score", "mean"),
    "mean_sentiment_score": ("mean_sentiment_score", "avg_claim_sentiment", "sentiment_score", "average_sentiment", "score", "mean"),
    "avg_claim_strength": ("avg_claim_strength", "mean_claim_strength_score", "claim_strength_score", "average_strength", "mean_strength", "score", "mean"),
    "mean_claim_strength_score": ("mean_claim_strength_score", "avg_claim_strength", "claim_strength_score", "average_strength", "score", "mean"),
    "share_of_monthly_mentions": ("mention_share", "share_of_mentions", "normalized_value", "mention_count_normalized", "count"),
    "share_of_monthly_entity_mentions": ("mention_share", "share_of_mentions", "normalized_value", "mention_count_normalized", "count"),
    "monthly_mention_share": ("mention_share", "share_of_mentions", "normalized_value", "mention_count_normalized", "count"),
    "monthly_entity_mention_share": ("mention_share", "share_of_mentions", "normalized_value", "mention_count_normalized", "count"),
    "drawdown": ("drawdown", "market_drawdown", "max_drawdown", "minimum_drawdown"),
    "market_drawdown": ("market_drawdown", "drawdown", "max_drawdown", "minimum_drawdown"),
    "market_return": ("market_return", "return", "daily_return", "monthly_return"),
    "market_close": ("market_close", "close", "price", "value"),
    "quote_share": ("quote_share", "quote_count", "count"),
    "mention_count_normalized": ("mention_count_normalized", "share_of_docs", "mention_share", "count"),
    "normalized_value": ("normalized_value", "mention_count_normalized", "share_of_docs", "mention_share", "doc_share", "count", "value"),
    "quote_count": ("quote_count", "count"),
    "topic_weight": ("weight",),
    "sentiment": ("mean", "score", "value"),
}
PLOT_X_FIELD_PRIORITY = (
    "time_bin",
    "date",
    "published_at",
    "label",
    "name",
    "lemma",
    "term",
    "entity",
    "source",
    "outlet",
    "topic_id",
    "doc_id",
)
PLOT_Y_FIELD_PRIORITY = (
    "count",
    "value",
    "score",
    "weight",
    "mean",
    "intensity",
    "relative_frequency",
    "document_frequency",
    "normalized_value",
    "mention_count_normalized",
    "doc_share",
    "mention_share",
    "market_drawdown",
    "market_return",
    "market_close",
    "frequency",
)
PLOT_NUMERIC_FIELD_EXCLUDES = {"rank", "id", "doc_id", "topic_id", "year", "month", "day"}
PLOT_NULL_AXIS_LABELS = NULL_AXIS_LABELS
PLOT_TIME_AXIS_FIELDS = {"time_bin", "month", "quarter", "period", "date", "published_at", "year", "time_period", "bucket"}
PLOT_TIME_VALUE_PATTERN = re.compile(
    r"^\d{4}(?:[-/](?:Q[1-4]|\d{1,2})(?:[-/]\d{1,2})?)?(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$",
    re.IGNORECASE,
)


def _normalize_plot_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _ordered_plot_keys(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            text = str(key)
            if text not in seen:
                seen.add(text)
                keys.append(text)
    return keys


def _plot_field_coverage(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for row in rows if str(row.get(field, "")).strip() != "")


def _plot_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if not text:
        return None
    is_percent = text.endswith("%")
    text = text.rstrip("%").replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number / 100.0 if is_percent else number


def _plot_field_has_numeric_values(rows: list[dict[str, Any]], field: str) -> bool:
    return any(_plot_float(row.get(field)) is not None for row in rows)


def _plot_field_has_nonzero_numeric_values(rows: list[dict[str, Any]], field: str) -> bool:
    return any((number := _plot_float(row.get(field))) is not None and abs(number) > 1e-12 for row in rows)


def _semantic_plot_field_candidates(keys: list[str], requested_normalized: str) -> list[str]:
    if not requested_normalized:
        return []
    semantic_terms = (
        ("return", ("return", "pctchange", "percentchange", "dailychange")),
        ("sentiment", ("sentiment", "tone", "polarity")),
        ("volatility", ("volatility", "drawdown", "variance", "risk")),
        ("price", ("price", "close", "open", "high", "low")),
        ("count", ("count", "frequency", "mentions", "documents")),
        ("share", ("share", "ratio", "percent", "percentage")),
    )
    requested_terms = [
        target
        for target, synonyms in semantic_terms
        if target in requested_normalized or any(synonym in requested_normalized for synonym in synonyms)
    ]
    if not requested_terms:
        return []
    candidates: list[str] = []
    for key in keys:
        normalized_key = _normalize_plot_field_name(key)
        if any(term in normalized_key for term in requested_terms) and key not in candidates:
            candidates.append(key)
    return candidates


def _plot_axis_label_is_usable(value: Any, *, time_axis_like: bool = False) -> bool:
    text = str(value if value is not None else "").strip()
    if text.lower() in PLOT_NULL_AXIS_LABELS:
        return False
    if time_axis_like and text.strip("_").lower() in PLOT_NULL_AXIS_LABELS:
        return False
    return True


def _plot_field_looks_time_like(rows: list[dict[str, Any]], field: str) -> bool:
    if not field:
        return False
    normalized = str(field).strip().lower()
    if normalized in PLOT_TIME_AXIS_FIELDS or any(token in normalized for token in ("time", "date", "month", "quarter", "period", "year")):
        return True
    sampled = 0
    matched = 0
    for row in rows:
        value = row.get(field)
        if not _plot_axis_label_is_usable(value, time_axis_like=True):
            continue
        sampled += 1
        if PLOT_TIME_VALUE_PATTERN.match(str(value).strip()):
            matched += 1
        if sampled >= 50:
            break
    return bool(sampled and matched / sampled >= 0.8)


def _distinct_usable_plot_values(rows: list[dict[str, Any]], field: str, *, time_axis_like: bool = False) -> set[str]:
    values: set[str] = set()
    for row in rows:
        value = row.get(field)
        if not _plot_axis_label_is_usable(value, time_axis_like=time_axis_like):
            continue
        values.add(str(value).strip())
        if len(values) > 1:
            break
    return values


def _inferred_plot_x_priority(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    base_priority = tuple(PLOT_X_FIELD_PRIORITY)
    if not rows:
        return base_priority
    keys = _ordered_plot_keys(rows)
    time_like_fields = [key for key in keys if _plot_field_looks_time_like(rows, key)]
    has_single_time_axis = any(
        _plot_field_coverage(rows, field) > 0
        and len(_distinct_usable_plot_values(rows, field, time_axis_like=True)) <= 1
        for field in time_like_fields
    )
    if not has_single_time_axis:
        return base_priority
    categorical_candidates = [
        field
        for field in base_priority
        if field not in PLOT_TIME_AXIS_FIELDS
        and (resolved := _resolve_existing_plot_field(rows, field, require_numeric=False))
        and not _plot_field_looks_time_like(rows, resolved)
        and len(_distinct_usable_plot_values(rows, resolved)) > 1
    ]
    if not categorical_candidates:
        return base_priority
    remaining = [field for field in base_priority if field not in categorical_candidates]
    return tuple(categorical_candidates + remaining)


def _infer_plot_series_field(rows: list[dict[str, Any]]) -> str:
    for candidate in PLOT_FIELD_ALIASES["series"]:
        resolved = _resolve_existing_plot_field(rows, candidate, require_numeric=False)
        if resolved and _plot_field_coverage(rows, resolved) > 0:
            return resolved
    return ""


def _resolve_existing_plot_field(
    rows: list[dict[str, Any]],
    field: str,
    *,
    require_numeric: bool,
) -> str | None:
    if not field:
        return None
    keys = _ordered_plot_keys(rows)
    normalized = _normalize_plot_field_name(field)
    candidates: list[str] = []
    if field in keys:
        candidates.append(field)
    candidates.extend(key for key in keys if _normalize_plot_field_name(key) == normalized and key not in candidates)
    alias_candidates = list(PLOT_FIELD_ALIASES.get(str(field).strip().lower(), ()))
    if not alias_candidates:
        for alias_key, aliases in PLOT_FIELD_ALIASES.items():
            if _normalize_plot_field_name(alias_key) == normalized:
                alias_candidates.extend(aliases)
                break
    candidates.extend(candidate for candidate in alias_candidates if candidate not in candidates)
    candidates.extend(
        key
        for key in keys
        for alias in alias_candidates
        if _normalize_plot_field_name(key) == _normalize_plot_field_name(alias) and key not in candidates
    )
    candidates.extend(
        candidate
        for candidate in _semantic_plot_field_candidates(keys, normalized)
        if candidate not in candidates
    )
    valid_candidates: list[str] = []
    for candidate in candidates:
        if _plot_field_coverage(rows, candidate) <= 0:
            continue
        if require_numeric and not _plot_field_has_numeric_values(rows, candidate):
            continue
        valid_candidates.append(candidate)
    if require_numeric:
        for candidate in valid_candidates:
            if _plot_field_has_nonzero_numeric_values(rows, candidate):
                return candidate
    if valid_candidates:
        return valid_candidates[0]
    return None


def _resolve_plot_field(
    rows: list[dict[str, Any]],
    requested: str,
    axis_name: str,
    *,
    require_numeric: bool = False,
    allow_infer: bool = False,
) -> tuple[str, str | None]:
    field = str(requested or "").strip()
    resolved = _resolve_existing_plot_field(rows, field, require_numeric=require_numeric)
    if resolved:
        if field and resolved != field:
            return resolved, f"Resolved plot {axis_name} field '{field}' to upstream field '{resolved}'."
        return resolved, None
    if field or not allow_infer:
        return field, None
    priority = PLOT_Y_FIELD_PRIORITY if require_numeric else _inferred_plot_x_priority(rows)
    for candidate in priority:
        resolved = _resolve_existing_plot_field(rows, candidate, require_numeric=require_numeric)
        if resolved:
            return resolved, f"Inferred plot {axis_name} field '{resolved}' from upstream rows."
    if require_numeric:
        for key in _ordered_plot_keys(rows):
            if key in PLOT_NUMERIC_FIELD_EXCLUDES:
                continue
            if _plot_field_has_numeric_values(rows, key):
                return key, f"Inferred plot {axis_name} field '{key}' from numeric upstream values."
    else:
        for key in _ordered_plot_keys(rows):
            if _plot_field_coverage(rows, key) > 0:
                return key, f"Inferred plot {axis_name} field '{key}' from upstream rows."
    return "", None


def _plot_label(value: Any, *, width: int = 48) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return "row"
    shortened = textwrap.shorten(text, width=width, placeholder="...")
    if shortened == "...":
        return text[: max(1, width - 3)] + "..."
    return shortened


def _plot_value_label(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:,.0f}"
    if abs_value >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs_value >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def _prepare_plot_points(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    points: list[dict[str, Any]] = []
    skipped = 0
    for index, row in enumerate(rows):
        value = _plot_float(row.get(y_key)) if y_key else None
        if value is None:
            skipped += 1
            continue
        raw_label = row.get(x_key) if x_key else None
        if raw_label is None or str(raw_label).strip() == "":
            raw_label = f"row {index + 1}"
        label = _plot_label(raw_label)
        points.append({"label": label, "value": value, "row": row})
        if len(points) >= limit:
            break
    return points, skipped


def _time_series_plot_series(row: dict[str, Any], series_key: str) -> str:
    if series_key:
        label = str(row.get(series_key, "")).strip()
        if _plot_axis_label_is_usable(label):
            return _plot_label(label, width=34)
    for fallback in (
        "canonical_entity",
        "linked_entity",
        "entity",
        "actor",
        "target_entity",
        "target_label",
        "series_name",
        "label",
        "term",
    ):
        label = str(row.get(fallback, "")).strip()
        if _plot_axis_label_is_usable(label):
            return _plot_label(label, width=34)
    return "All rows"


def _prepare_time_series_plot_data(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    series_key: str,
    series_limit: int,
) -> tuple[list[dict[str, Any]], list[tuple[str, list[tuple[str, float, dict[str, Any]]]]], list[str], int]:
    series_by_label: defaultdict[str, list[tuple[str, float, dict[str, Any]]]] = defaultdict(list)
    skipped = 0
    for row in rows:
        time_value = str(row.get(x_key, row.get("time_bin", ""))).strip()
        if not _plot_axis_label_is_usable(time_value, time_axis_like=True):
            skipped += 1
            continue
        value = _plot_float(row.get(y_key, row.get("count", row.get("score", row.get("weight", row.get("intensity", 0.0))))))
        if value is None:
            skipped += 1
            continue
        series_by_label[_time_series_plot_series(row, series_key)].append((time_value, value, row))
    top_series = sorted(
        series_by_label.items(),
        key=lambda entry: sum(value for _, value, _ in entry[1]),
        reverse=True,
    )[: max(1, series_limit)]
    all_bins = sorted({time_bin for _, points in top_series for time_bin, _, _ in points})
    plotted_rows: list[dict[str, Any]] = []
    for series_label, points in top_series:
        for time_bin, value, row in sorted(points, key=lambda item: item[0]):
            plotted_row = dict(row)
            plotted_row["_plot_series"] = series_label
            plotted_row["_plot_time_bin"] = time_bin
            plotted_row["_plot_value"] = value
            plotted_rows.append(plotted_row)
    return plotted_rows, top_series, all_bins, skipped


def _svg_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_svg_plot_fallback(
    *,
    params: dict[str, Any],
    rows: list[dict[str, Any]],
    target: Path,
    plot_name: str,
) -> Path:
    target = target.with_suffix(".svg")
    x_key = str(params.get("x", "")).strip()
    y_key = str(params.get("y", "")).strip()
    limit = _int_param(params, "limit", "top_k", default=16, maximum=75)
    points, _ = _prepare_plot_points(rows, x_key=x_key, y_key=y_key, limit=limit)
    labels = [point["label"] for point in points]
    values = [float(point["value"]) for point in points]
    max_value = max((abs(value) for value in values), default=1.0) or 1.0
    width = 960
    height = max(420, 110 + (len(points) * 30))
    left = 230
    top = 72
    row_height = 28
    bar_width = 600
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{left}" y="38" font-family="Verdana, sans-serif" font-size="20" font-weight="700" fill="#17312d">{_svg_escape(plot_name)}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values, strict=False)):
        y = top + (index * row_height)
        normalized_width = int((abs(value) / max_value) * bar_width)
        color = "#0f766e" if value >= 0 else "#be123c"
        lines.append(
            f'<text x="16" y="{y + 16}" font-family="Verdana, sans-serif" font-size="12" fill="#17312d">{_svg_escape(label[:34])}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{y}" width="{normalized_width}" height="16" rx="3" fill="{color}" opacity="0.82"/>'
        )
        lines.append(
            f'<text x="{left + normalized_width + 8}" y="{y + 13}" font-family="Verdana, sans-serif" font-size="11" fill="#17312d">{_plot_value_label(value)}</text>'
        )
    lines.append("</svg>")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _image_has_visual_content(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            extrema = image.convert("RGB").getextrema()
    except Exception:
        return path.exists() and path.stat().st_size > 2048
    return any((high - low) > 2 for low, high in extrema)


def _plot_artifact(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = []
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload:
            rows = list(payload["rows"])
            if rows:
                break
    if not rows:
        return ToolExecutionResult(
            payload={"rows": []},
            caveats=["No rows available for plotting."],
            metadata={"no_data": True, "no_data_reason": "No rows available for plotting."},
        )
    limit = _int_param(params, "limit", "top_k", default=16, maximum=100)
    structured_rows = [dict(item) for item in rows if isinstance(item, dict)]
    if not structured_rows:
        return ToolExecutionResult(
            payload={"rows": []},
            caveats=["No structured rows available for plotting."],
            metadata={"no_data": True, "no_data_reason": "No structured rows available for plotting."},
        )
    requested_x_key = str(params.get("x", "")).strip()
    requested_y_key = str(params.get("y", "")).strip()
    x_key, x_caveat = _resolve_plot_field(
        structured_rows,
        requested_x_key,
        "x",
        allow_infer=not requested_x_key,
    )
    y_key, y_caveat = _resolve_plot_field(
        structured_rows,
        requested_y_key,
        "y",
        require_numeric=True,
        allow_infer=not requested_y_key,
    )
    requested_series_key = str(params.get("series", params.get("series_key", "")) or "").strip()
    series_key = ""
    series_caveat = None
    if requested_series_key:
        series_key, series_caveat = _resolve_plot_field(
            structured_rows,
            requested_series_key,
            "series",
            allow_infer=False,
        )
        if series_key and _plot_field_coverage(structured_rows, series_key) <= 0:
            series_caveat = f"Plot requested series='{requested_series_key}', but upstream rows do not contain a compatible field; falling back to entity-like fields."
            series_key = ""
    plot_field_caveats = [caveat for caveat in (x_caveat, y_caveat, series_caveat) if caveat]
    if requested_x_key and _plot_field_coverage(structured_rows, x_key) <= 0:
        return ToolExecutionResult(
            payload={"rows": []},
            caveats=[f"Plot requested x='{requested_x_key}', but upstream rows do not contain a compatible field."],
            metadata={"no_data": True, "no_data_reason": f"Missing plot x field: {requested_x_key}"},
        )
    if requested_y_key and (
        _plot_field_coverage(structured_rows, y_key) <= 0 or not _plot_field_has_numeric_values(structured_rows, y_key)
    ):
        return ToolExecutionResult(
            payload={"rows": []},
            caveats=[f"Plot requested y='{requested_y_key}', but upstream rows do not contain compatible numeric values."],
            metadata={"no_data": True, "no_data_reason": f"Missing plot y field: {requested_y_key}"},
        )
    if not y_key:
        return ToolExecutionResult(
            payload={"rows": []},
            caveats=["No numeric field was available for plotting."],
            metadata={"no_data": True, "no_data_reason": "No numeric plot field available."},
        )
    requested_plot_type = str(params.get("plot_type", "") or "").strip().lower()
    scatter_like = requested_plot_type in {"scatter", "scatterplot", "point", "points"}
    time_axis_like = (
        requested_plot_type in {"line", "time_series", "timeseries"}
        or _plot_field_looks_time_like(structured_rows, x_key)
    )
    if not series_key and time_axis_like:
        series_key = _infer_plot_series_field(structured_rows)
    axis_filtered_rows = [
        row
        for row in structured_rows
        if _plot_axis_label_is_usable(row.get(x_key) if x_key else None, time_axis_like=time_axis_like)
    ]
    skipped_axis_rows = len(structured_rows) - len(axis_filtered_rows)
    if not axis_filtered_rows:
        return ToolExecutionResult(
            payload={"rows": []},
            caveats=[f"Plot field '{x_key}' did not contain usable axis labels."],
            metadata={"no_data": True, "no_data_reason": f"Missing usable plot x values: {x_key}"},
        )
    topic_like_source = bool(axis_filtered_rows and "topic_id" in axis_filtered_rows[0] and any(item.get("top_terms") for item in axis_filtered_rows))
    market_overlay_like_source = bool(
        axis_filtered_rows and "market_close" in axis_filtered_rows[0] and any("count" in item or "score" in item for item in axis_filtered_rows)
    )
    time_series_like = bool(
        axis_filtered_rows
        and x_key
        and not topic_like_source
        and not market_overlay_like_source
        and (requested_plot_type in {"line", "time_series", "timeseries"} or (series_key and time_axis_like))
    )
    points: list[dict[str, Any]] = []
    skipped_points = 0
    time_series_data: tuple[list[dict[str, Any]], list[tuple[str, list[tuple[str, float, dict[str, Any]]]]], list[str], int] | None = None
    if scatter_like:
        first = []
        for row in axis_filtered_rows:
            x_value = _plot_float(row.get(x_key)) if x_key else None
            y_value = _plot_float(row.get(y_key)) if y_key else None
            if x_value is None or y_value is None:
                skipped_points += 1
                continue
            plotted_row = dict(row)
            plotted_row["_plot_x"] = x_value
            plotted_row["_plot_y"] = y_value
            first.append(plotted_row)
            if len(first) >= limit:
                break
    elif time_series_like:
        series_limit = _int_param(params, "series_top_k", "max_series", default=5, maximum=12)
        time_series_data = _prepare_time_series_plot_data(
            axis_filtered_rows,
            x_key=x_key,
            y_key=y_key,
            series_key=series_key,
            series_limit=series_limit,
        )
        first, _, _, skipped_points = time_series_data
    else:
        points, skipped_points = _prepare_plot_points(axis_filtered_rows, x_key=x_key, y_key=y_key, limit=limit)
        first = [dict(point["row"]) for point in points]
    if not first:
        return ToolExecutionResult(
            payload={"rows": []},
            caveats=[f"Plot field '{y_key}' did not contain numeric values in the selected rows."],
            metadata={"no_data": True, "no_data_reason": f"Missing usable plot y values: {y_key}"},
        )
    if skipped_axis_rows:
        plot_field_caveats.append(f"Skipped {skipped_axis_rows} row(s) with empty or placeholder labels for '{x_key}'.")
    if skipped_points:
        if time_series_like:
            plot_field_caveats.append(f"Skipped {skipped_points} row(s) without usable time-series values.")
        else:
            plot_field_caveats.append(f"Skipped {skipped_points} row(s) without numeric values for '{y_key}'.")
    total_skipped_rows = skipped_axis_rows + skipped_points
    plot_kind = (
        "scatter"
        if scatter_like
        else "topic_bar"
        if topic_like_source
        else "market_overlay"
        if market_overlay_like_source
        else "time_series"
        if time_series_like
        else "bar"
    )
    plot_dir = context.artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    raw_plot_name = str(params.get("plot_name") or params.get("title") or "plot").strip() or "plot"
    plot_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_plot_name).strip("._")[:80] or "plot"
    target = plot_dir / f"{plot_slug}.png"
    plot_name = str(params.get("title") or raw_plot_name).replace("_", " ").title()
    try:
        with MATPLOTLIB_PLOT_LOCK:
            import matplotlib

            matplotlib.use("Agg", force=True)
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
    except Exception as exc:
        fallback_params = {**params, "x": x_key, "y": y_key}
        fallback_target = _write_svg_plot_fallback(params=fallback_params, rows=axis_filtered_rows, target=target, plot_name=plot_name)
        return ToolExecutionResult(
            payload={
                "artifact_path": str(fallback_target),
                "rows": first,
                "plot_name": plot_name,
                "resolved_x": x_key,
                "resolved_y": y_key,
                "resolved_series": series_key,
                "plot_kind": plot_kind,
                "plotted_row_count": len(first),
                "skipped_row_count": total_skipped_rows,
            },
            artifacts=[str(fallback_target)],
            caveats=[*plot_field_caveats, f"matplotlib unavailable; generated SVG fallback plot instead: {exc}"],
            metadata={
                "no_data": False,
                "fallback": "svg",
                "reason": "matplotlib_unavailable",
                "resolved_x": x_key,
                "resolved_y": y_key,
                "resolved_series": series_key,
                "plot_kind": plot_kind,
                "plotted_row_count": len(first),
                "skipped_row_count": total_skipped_rows,
            },
        )
    figure_height = 6.6 if time_series_like else max(5.6, min(14.0, 2.7 + (0.34 * len(first))))
    figure = Figure(figsize=(13.2, figure_height), dpi=180)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.grid(axis="x", color="#d9d2c3", linewidth=0.8, alpha=0.55)
    axis.grid(axis="y", visible=False)
    axis.tick_params(axis="both", labelsize=8.5, colors="#17312d")
    topic_like = bool(first and "topic_id" in first[0] and any(item.get("top_terms") for item in first))
    market_overlay_like = bool(first and "market_close" in first[0] and any("count" in item or "score" in item for item in first))

    if scatter_like:
        x_values = [float(item["_plot_x"]) for item in first]
        y_values = [float(item["_plot_y"]) for item in first]
        axis.scatter(x_values, y_values, s=44, color="#0f766e", edgecolors="#17312d", linewidths=0.45, alpha=0.82)
        if len(x_values) >= 2 and len(set(x_values)) > 1:
            try:
                import numpy as np

                slope, intercept = np.polyfit(x_values, y_values, 1)
                line_x = np.array([min(x_values), max(x_values)])
                axis.plot(line_x, slope * line_x + intercept, color="#b45309", linewidth=2.0, alpha=0.85, label="Linear fit")
                axis.legend(loc="best", fontsize=8, frameon=True)
            except Exception:
                pass
        axis.axhline(0, color="#4b5563", linewidth=0.8, alpha=0.45)
        axis.axvline(0, color="#4b5563", linewidth=0.8, alpha=0.35)
        axis.set_xlabel(x_key.replace("_", " ").title() or "X")
        axis.set_ylabel(y_key.replace("_", " ").title() or "Y")
    elif topic_like:
        labels = []
        values = []
        for index, item in enumerate(first):
            top_terms = clean_topic_terms(item.get("top_terms", []), max_count=4)
            label = f"Topic {item.get('topic_id', index + 1)}"
            if top_terms:
                label = f"{label}: {', '.join(top_terms)}"
            labels.append(label)
            values.append(_plot_float(item.get("weight")) or 0.0)
        colors = ["#0f766e", "#138f7a", "#16a085", "#54b499", "#7ec9b4", "#a5ded0"]
        axis.barh(labels[::-1], values[::-1], color=colors[: len(values)][::-1], edgecolor="#17312d", linewidth=0.7)
        axis.set_xlabel("Topic weight")
        max_value = max(values) if values else 0.0
        for idx, value in enumerate(values[::-1]):
            axis.text(value + max(max_value * 0.015, 0.02), idx, f"{value:.2f}", va="center", fontsize=8, color="#17312d")
    elif market_overlay_like:
        ordered = sorted(first, key=lambda item: str(item.get("time_bin", "unknown")))
        x_labels = [str(item.get("time_bin", "unknown")) for item in ordered]
        x_values = list(range(len(x_labels)))
        signal_values = [
            _plot_float(item.get("count", item.get("score", item.get("weight", 0.0)))) or 0.0
            for item in ordered
        ]
        market_values = [_plot_float(item.get("market_close")) or 0.0 for item in ordered]
        axis.bar(x_values, signal_values, color="#0f766e", alpha=0.78, width=0.65, label="Corpus signal")
        axis.set_ylabel("Corpus signal", color="#0f766e")
        axis.tick_params(axis="y", colors="#0f766e")
        axis.set_xticks(x_values, x_labels, rotation=35, ha="right")
        market_axis = axis.twinx()
        market_axis.plot(x_values, market_values, color="#b45309", linewidth=2.4, marker="o", label="Market close")
        market_axis.set_ylabel("Market close", color="#b45309")
        market_axis.tick_params(axis="y", colors="#b45309")
        handles_1, labels_1 = axis.get_legend_handles_labels()
        handles_2, labels_2 = market_axis.get_legend_handles_labels()
        if handles_1 or handles_2:
            axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best", fontsize=8, frameon=True)
    elif time_series_like:
        _, top_entities, all_bins, _ = time_series_data or ([], [], [], 0)
        x_lookup = {time_bin: index for index, time_bin in enumerate(all_bins)}
        x_values = list(range(len(all_bins)))
        palette = ["#0f766e", "#b45309", "#7c3aed", "#be123c", "#2563eb", "#0f4c81", "#7c2d12", "#166534"]
        for index, (entity, points) in enumerate(top_entities):
            values_by_bin: defaultdict[str, float] = defaultdict(float)
            for time_bin, value, _ in points:
                values_by_bin[time_bin] += value
            y_values = [values_by_bin.get(time_bin, 0.0) for time_bin in all_bins]
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2.4,
                color=palette[index % len(palette)],
                label=entity,
            )
            axis.fill_between(x_values, y_values, alpha=0.08, color=palette[index % len(palette)])
        if top_entities and all_bins:
            step = max(1, len(all_bins) // 12)
            tick_values = [x_lookup[time_bin] for idx, time_bin in enumerate(all_bins) if idx % step == 0]
            tick_labels = [time_bin for idx, time_bin in enumerate(all_bins) if idx % step == 0]
            axis.set_xticks(tick_values, tick_labels, rotation=35, ha="right")
        axis.set_ylabel(y_key.replace("_", " ").title() or "Signal")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=True)
    else:
        labels = [point["label"] for point in points]
        values = [float(point["value"]) for point in points]
        colors = ["#0f766e" if value >= 0 else "#be123c" for value in values]
        needs_horizontal = len(labels) > 8 or max((len(label) for label in labels), default=0) > 18
        if needs_horizontal:
            plotted_labels = labels[::-1]
            plotted_values = values[::-1]
            plotted_colors = colors[::-1]
            bars = axis.barh(plotted_labels, plotted_values, color=plotted_colors, edgecolor="#17312d", linewidth=0.5)
            max_abs = max((abs(value) for value in plotted_values), default=1.0) or 1.0
            for bar, value in zip(bars, plotted_values, strict=False):
                x_offset = max_abs * 0.012
                axis.text(
                    value + (x_offset if value >= 0 else -x_offset),
                    bar.get_y() + bar.get_height() / 2,
                    _plot_value_label(value),
                    ha="left" if value >= 0 else "right",
                    va="center",
                    fontsize=8,
                )
            axis.set_xlabel(y_key.replace("_", " ").title() or "Value")
        else:
            bars = axis.bar(labels, values, color=colors, edgecolor="#17312d", linewidth=0.5)
            for bar, value in zip(bars, values, strict=False):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    _plot_value_label(value),
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                )
            axis.tick_params(axis="x", rotation=25)
            axis.set_ylabel(y_key.replace("_", " ").title() or "Value")

    axis.set_title(plot_name, fontsize=15, fontweight="bold")
    axis.set_facecolor("#fffaf2")
    figure.patch.set_facecolor("#fffdf8")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    with MATPLOTLIB_PLOT_LOCK:
        figure.tight_layout()
        figure.savefig(target)
    if not _image_has_visual_content(target):
        try:
            target.unlink(missing_ok=True)
        except Exception:
            pass
        return ToolExecutionResult(
            payload={"rows": []},
            caveats=[*plot_field_caveats, "Generated plot image had no visible content, so it was discarded."],
            metadata={"no_data": True, "no_data_reason": "Blank plot image generated."},
        )
    return ToolExecutionResult(
        payload={
            "artifact_path": str(target),
            "rows": first,
            "plot_name": plot_name,
            "resolved_x": x_key,
            "resolved_y": y_key,
            "resolved_series": series_key,
            "plot_kind": plot_kind,
            "plotted_row_count": len(first),
            "skipped_row_count": total_skipped_rows,
        },
        artifacts=[str(target)],
        caveats=plot_field_caveats,
        metadata={
            "no_data": False,
            "resolved_x": x_key,
            "resolved_y": y_key,
            "resolved_series": series_key,
            "plot_kind": plot_kind,
            "plotted_row_count": len(first),
            "skipped_row_count": total_skipped_rows,
        },
    )


def _plot_artifact_safe(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    """Wrap _plot_artifact so rendering failures degrade gracefully.

    Plot generation is the most failure-prone step in many runs (matplotlib
    state, weird input distributions, Windows path quirks). Instead of letting
    a renderer exception kill the node and cascade to downstream consumers,
    we catch it and return the upstream rows with plot_skipped=True so
    non-plot downstream nodes can still process the data.
    """
    fallback_rows: list[dict[str, Any]] = []
    try:
        for result in deps.values():
            payload = result.payload
            if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
                candidate = [dict(item) for item in payload["rows"] if isinstance(item, dict)]
                if candidate:
                    fallback_rows = candidate
                    break
    except Exception:
        fallback_rows = []

    try:
        return _plot_artifact(params, deps, context)
    except Exception as exc:
        import traceback as _plot_tb

        reason = f"{exc.__class__.__name__}: {exc}"
        return ToolExecutionResult(
            payload={
                "rows": fallback_rows,
                "plot_skipped": True,
                "plot_reason": reason,
            },
            caveats=[f"plot_artifact rendering failed: {reason}. Downstream rows preserved."],
            metadata={
                "no_data": False,
                "plot_skipped": True,
                "plot_reason": reason,
                "plot_traceback": _plot_tb.format_exc(),
                "fallback": "rows_only",
                "degraded": True,
            },
        )


def _looks_like_python_code(code: str) -> bool:
    text = str(code or "").strip()
    if not text:
        return False
    try:
        compile(text, "<python_runner>", "exec")
    except SyntaxError:
        return False
    return True


def _is_numeric_correlation_task(task: str, params: dict[str, Any]) -> bool:
    combined = " ".join(
        str(value)
        for value in (
            task,
            params.get("task", ""),
            params.get("description", ""),
            params.get("question", ""),
        )
        if value
    ).lower()
    return any(term in combined for term in ("correlation", "correlat", "association", "co-movement", "relationship"))


def _numeric_series_field(rows: list[dict[str, Any]], explicit: Any, preferred_terms: tuple[str, ...]) -> str:
    if explicit:
        field = str(explicit).strip()
        if field and any(field in row for row in rows):
            return field
    columns = sorted({key for row in rows for key in row})
    numeric_columns: list[str] = []
    for column in columns:
        values = [row.get(column) for row in rows if row.get(column) not in (None, "")]
        if not values:
            continue
        numeric_count = 0
        for value in values:
            try:
                float(value)
                numeric_count += 1
            except (TypeError, ValueError):
                pass
        if numeric_count >= max(2, min(len(values), 3)):
            numeric_columns.append(column)
    for term in preferred_terms:
        for column in numeric_columns:
            if term in column.lower():
                return column
    low_value_columns = {
        "rank",
        "document_count",
        "doc_count",
        "count",
        "mention_count",
        "document_frequency",
        "market_volume",
    }
    for column in numeric_columns:
        if column.lower() not in low_value_columns:
            return column
    return numeric_columns[0] if numeric_columns else ""


def _native_numeric_correlation_rows(
    params: dict[str, Any],
    deps: dict[str, ToolExecutionResult],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    rows = _payload_rows_from_all_dependencies(deps)
    if not rows:
        return [], {"analysis": "numeric_correlation"}, ["No rows were available for native numeric correlation."]
    x_field = _numeric_series_field(
        rows,
        params.get("x_field") or params.get("left_value_field") or params.get("sentiment_field"),
        ("sentiment", "tone", "score"),
    )
    y_field = _numeric_series_field(
        rows,
        params.get("y_field") or params.get("right_value_field") or params.get("market_field") or params.get("return_field"),
        ("market_return", "return", "drawdown", "volatility", "market_close", "close"),
    )
    if not x_field or not y_field or x_field == y_field:
        return [], {"analysis": "numeric_correlation"}, ["Could not infer two distinct numeric fields for correlation."]
    frame = pd.DataFrame(rows)
    if x_field not in frame.columns or y_field not in frame.columns:
        return [], {"analysis": "numeric_correlation", "x_field": x_field, "y_field": y_field}, [
            "Requested correlation fields were not present in the dependency rows."
        ]
    pair_frame = frame[[x_field, y_field]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair_frame) < 2:
        return [], {"analysis": "numeric_correlation", "x_field": x_field, "y_field": y_field}, [
            "Fewer than two aligned numeric observations were available for correlation."
        ]
    pearson = float(pair_frame[x_field].corr(pair_frame[y_field], method="pearson"))
    spearman = float(pair_frame[x_field].corr(pair_frame[y_field], method="spearman"))
    row = {
        "analysis": "numeric_correlation",
        "x_field": x_field,
        "y_field": y_field,
        "paired_observation_count": int(len(pair_frame)),
        "pearson_correlation": round(pearson, 6) if math.isfinite(pearson) else None,
        "spearman_correlation": round(spearman, 6) if math.isfinite(spearman) else None,
    }
    return [row], {"analysis": "numeric_correlation", "x_field": x_field, "y_field": y_field}, []


def _python_runner(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    params = _analysis_params(params)
    task = _task_name(params)
    if _is_actor_prominence_merge_task(task):
        merged_rows, merge_metadata, merge_caveats = _merge_actor_prominence_rows(params, deps)
        return ToolExecutionResult(
            payload={"rows": merged_rows, **merge_metadata, "exit_code": 0, "stdout": "", "stderr": ""},
            caveats=merge_caveats if merged_rows else [*merge_caveats, "No actor prominence rows were available to merge."],
            metadata={"no_data": not merged_rows, **merge_metadata},
        )
    if _is_entity_frequency_task(task, params):
        entity_rows, entity_metadata, caveats = _entity_frequency_rows(params, deps, task=task)
        if not entity_rows:
            caveats = [*caveats, "No entity or actor rows were available for native python_runner aggregation."]
        return ToolExecutionResult(
            payload={"rows": entity_rows, **entity_metadata, "exit_code": 0, "stdout": "", "stderr": ""},
            caveats=caveats,
            metadata={"no_data": not entity_rows, **entity_metadata},
        )
    code = str(params.get("code", "")).strip()
    if not _looks_like_python_code(code):
        if _is_numeric_correlation_task(task, params):
            correlation_rows, correlation_metadata, correlation_caveats = _native_numeric_correlation_rows(params, deps)
            if correlation_rows:
                return ToolExecutionResult(
                    payload={"rows": correlation_rows, **correlation_metadata, "exit_code": 0, "stdout": "", "stderr": ""},
                    caveats=correlation_caveats,
                    metadata={"no_data": False, **correlation_metadata},
                )
            return ToolExecutionResult(
                payload={"rows": [], **correlation_metadata, "exit_code": 0, "stdout": "", "stderr": ""},
                caveats=correlation_caveats,
                metadata={"no_data": True, "no_data_reason": "numeric_correlation_unavailable", **correlation_metadata},
            )
        if any(
            str(row.get("doc_id", "")).strip()
            and any(name in row for name in ("entity", "canonical_entity", "linked_entity", "speaker", "attributed_actor"))
            for row in _payload_rows_from_all_dependencies(deps)
        ):
            native_params = {
                **params,
                "task": "actor_prominence",
                "group_by_time": params.get("group_by_time", "month"),
                "entity_types": params.get("entity_types", ["PERSON", "ORG"]),
                "top_n": params.get("top_n", params.get("limit", 100)),
            }
            entity_rows, entity_metadata, entity_caveats = _entity_frequency_rows(native_params, deps, task="actor_prominence")
            if entity_rows:
                return ToolExecutionResult(
                    payload={"rows": entity_rows, **entity_metadata, "exit_code": 0, "stdout": "", "stderr": ""},
                    caveats=entity_caveats,
                    metadata={"no_data": False, **entity_metadata},
                )
        merged_rows, merge_metadata, merge_caveats = _merge_actor_prominence_rows(params, deps)
        if merged_rows:
            return ToolExecutionResult(
                payload={"rows": merged_rows, **merge_metadata, "exit_code": 0, "stdout": "", "stderr": ""},
                caveats=merge_caveats,
                metadata={"no_data": False, **merge_metadata},
            )
        return ToolExecutionResult(
            payload={"rows": [], "exit_code": 0, "stdout": "", "stderr": ""},
            caveats=["python_runner received instructions that were not valid Python code; no sandbox execution was attempted."],
            metadata={"no_data": True, "no_data_reason": "python_runner_non_python_code"},
        )
    if context.python_runner is None:
        raise RuntimeError("python_runner is unavailable")
    inputs_json = dict(params.get("inputs_json", {}))
    if not inputs_json:
        for key, result in deps.items():
            inputs_json[key] = result.payload
    result = context.python_runner.run(code=code, inputs_json=inputs_json)
    import base64

    artifacts = []
    structured_payload: dict[str, Any] | None = None
    for artifact in result.artifacts:
        artifact_path = context.artifacts_dir / "python_runner" / artifact.name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        decoded_bytes = base64.b64decode(artifact.bytes_b64.encode("ascii"))
        artifact_path.write_bytes(decoded_bytes)
        if artifact.mime == "application/json":
            try:
                parsed = json.loads(decoded_bytes.decode("utf-8"))
            except Exception:
                parsed = None
            if structured_payload is None and isinstance(parsed, dict):
                structured_payload = dict(parsed)
        artifacts.append(str(artifact_path))
    payload = result.to_dict()
    if structured_payload is not None:
        payload = dict(structured_payload)
        payload.setdefault("stdout", result.stdout)
        payload.setdefault("stderr", result.stderr)
        payload.setdefault("exit_code", int(result.exit_code))
        payload.setdefault("artifacts", [item.to_dict() for item in result.artifacts])
    return ToolExecutionResult(
        payload=payload,
        artifacts=artifacts,
        caveats=[] if result.exit_code == 0 else ["Python fallback returned non-zero exit code."],
    )


def build_agent_registry() -> ToolRegistry:
    registry = ToolRegistry()
    entries = [
        ("opensearch_db_search", "db_search", "backend", 100, _db_search),
        ("postgres_sql_search", "sql_query_search", "backend", 99, _sql_query_search),
        ("postgres_fetch_documents", "fetch_documents", "backend", 99, _fetch_documents),
        ("working_set_store", "create_working_set", "backend", 98, _create_working_set),
        ("working_set_filter", "filter_working_set", "backend", 97, _filter_working_set),
        ("lang_id", "lang_id", "textacy", 91, _lang_id),
        ("clean_normalize", "clean_normalize", "textacy", 90, _clean_normalize),
        ("tokenize", "tokenize", "spacy", 89, _tokenize_docs),
        ("sentence_split", "sentence_split", "spacy", 88, _sentence_split_docs),
        ("mwt_expand", "mwt_expand", "stanza", 87, _tokenize_docs),
        ("pos_morph", "pos_morph", "spacy", 86, _pos_morph),
        ("lemmatize", "lemmatize", "spacy", 85, _lemmatize_docs),
        ("dependency_parse", "dependency_parse", "spacy", 84, _dependency_parse),
        ("noun_chunks", "noun_chunks", "spacy", 83, _noun_chunks),
        ("ner", "ner", "spacy", 82, _ner),
        ("entity_link", "entity_link", "spacy", 81, _entity_link),
        ("extract_keyterms", "extract_keyterms", "textacy", 80, _extract_keyterms),
        ("extract_svo_triples", "extract_svo_triples", "textacy", 79, _extract_svo_triples),
        ("topic_model", "topic_model", "textacy", 78, _topic_model),
        ("readability_stats", "readability_stats", "textacy", 77, _readability_stats),
        ("lexical_diversity", "lexical_diversity", "textacy", 76, _lexical_diversity),
        ("extract_ngrams", "extract_ngrams", "textacy", 75, _extract_ngrams),
        ("extract_acronyms", "extract_acronyms", "textacy", 74, _extract_acronyms),
        ("sentiment", "sentiment", "flair", 73, _sentiment),
        ("text_classify", "text_classify", "flair", 72, _text_classify),
        ("word_embeddings", "word_embeddings", "gensim", 71, _word_embeddings),
        ("doc_embeddings", "doc_embeddings", "gensim", 70, _doc_embeddings),
        ("similarity_pairwise", "similarity_pairwise", "spacy", 69, _similarity_pairwise),
        ("similarity_index", "similarity_index", "gensim", 68, _similarity_index),
        ("time_series_aggregate", "time_series_aggregate", "analytics", 67, _time_series_aggregate),
        ("change_point_detect", "change_point_detect", "analytics", 66, _change_point_detect),
        ("burst_detect", "burst_detect", "analytics", 65, _burst_detect),
        ("claim_span_extract", "claim_span_extract", "analytics", 64, _claim_span_extract),
        ("claim_strength_score", "claim_strength_score", "analytics", 63, _claim_strength_score),
        ("quote_extract", "quote_extract", "analytics", 62, _quote_extract),
        ("quote_attribute", "quote_attribute", "analytics", 61, _quote_attribute),
        ("build_evidence_table", "build_evidence_table", "analytics", 60, _build_evidence_table),
        ("join_external_series", "join_external_series", "analytics", 59, _join_external_series),
        ("plot_artifact", "plot_artifact", "matplotlib", 58, _plot_artifact_safe),
        ("python_runner", "python_runner", "sandbox", 57, _python_runner),
    ]
    for tool_name, capability, provider, priority, fn in entries:
        registry.register(
            FunctionalToolAdapter(
                tool_name=tool_name,
                capability=capability,
                provider=provider,
                priority=priority,
                run_fn=fn,
            )
        )
    return registry
