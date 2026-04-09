from __future__ import annotations

from collections import Counter, defaultdict
import base64
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
from typing import Any, Callable
from urllib.parse import quote

import pandas as pd

from .agent_backends import PostgresWorkingSetStore, SearchBackend, WorkingSetStore
from .agent_models import EvidenceRow
from .analysis_tools import textrank_keywords
from .io_utils import sentence_split as simple_sentence_split
from .python_runner_service import DockerPythonRunnerService
from .retrieval import _load_sentence_transformer
from .seed import resolve_device
from .tool_registry import CapabilityToolAdapter, SchemaDescriptor, ToolExecutionResult, ToolRegistry, ToolSpec


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']+")
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
QUOTE_PATTERN = re.compile(r'["“](.*?)["”]')
SPEAKER_PATTERN = re.compile(
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(said|says|warned|argued|claimed|according to)",
    re.IGNORECASE,
)

STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "have", "were", "their", "about", "into", "there",
    "would", "could", "should", "while", "after", "before", "where", "which", "what", "when", "been", "being",
    "over", "under", "between", "across", "relationship", "systematic", "movements", "movement",
    "coverage", "media", "sentiment", "price", "prices", "analysis",
}
POSITIVE_WORDS = {"good", "strong", "gain", "improve", "success", "positive", "optimistic"}
NEGATIVE_WORDS = {"bad", "weak", "loss", "drop", "risk", "fear", "negative", "warn", "crisis"}
CLAIM_KEYWORDS = {
    "predict", "predicted", "prediction", "warn", "warned", "warning", "imminent",
    "within days", "could invade", "will invade", "invasion",
}
LANGUAGE_HINTS = {
    "de": {"und", "der", "die", "das", "nicht", "mit", "ist", "ein"},
    "fr": {"le", "la", "les", "des", "une", "est", "avec", "dans"},
    "it": {"il", "lo", "la", "gli", "che", "con", "per", "non"},
    "en": {"the", "and", "with", "from", "that", "this", "have", "will"},
}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_time_granularity() -> str:
    return (os.getenv("CORPUSAGENT2_TIME_GRANULARITY", "month").strip().lower() or "month")


QUERY_ANCHOR_STOPWORDS = STOPWORDS.union(
    {
        "how",
        "did",
        "does",
        "what",
        "which",
        "who",
        "why",
        "around",
        "during",
        "toward",
        "towards",
        "shift",
        "shifted",
        "framing",
        "frame",
        "growth",
        "innovation",
        "privacy",
        "regulation",
        "drawdown",
        "drawdowns",
        "stock",
        "stocks",
        "overall",
        "monthly",
        "weekly",
        "daily",
        "yearly",
    }
)


def _query_anchor_terms(query: str) -> list[str]:
    text = str(query or "").strip()
    if not text:
        return []
    preferred: list[str] = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Z]?[A-Za-z0-9]+)?\b", text):
        token = match.group(0).strip()
        lowered = token.lower()
        if lowered in QUERY_ANCHOR_STOPWORDS or len(lowered) < 3:
            continue
        if token not in preferred:
            preferred.append(token)
        if "-" in token:
            for part in token.split("-"):
                part = part.strip()
                lowered_part = part.lower()
                if len(lowered_part) < 3 or lowered_part in QUERY_ANCHOR_STOPWORDS:
                    continue
                if part not in preferred:
                    preferred.append(part)
    if preferred:
        return preferred[:8]
    fallback: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\\-]+", text):
        lowered = token.lower()
        if len(lowered) < 4 or lowered in QUERY_ANCHOR_STOPWORDS:
            continue
        if token not in fallback:
            fallback.append(token)
    return fallback[:8]


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


def _normalize_result_rows(rows: list[dict[str, Any]], retrieval_mode: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda item: _coerce_score(item.get("score", 0.0)), reverse=True)
    max_score = max((_coerce_score(item.get("score", 0.0)) for item in ordered), default=0.0)
    normalized: list[dict[str, Any]] = []
    for rank, row in enumerate(ordered, start=1):
        copy = dict(row)
        score = _coerce_score(copy.get("score", 0.0))
        copy["score"] = score
        copy["rank"] = rank
        copy["retrieval_mode"] = str(copy.get("retrieval_mode", retrieval_mode) or retrieval_mode)
        copy["score_display"] = str(copy.get("score_display") or _score_display(score / max_score if max_score > 0 else score))
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
    duplicates_removed = 0
    for row in ordered:
        duplicate_of: dict[str, Any] | None = None
        for candidate in kept:
            if _rows_are_near_duplicates(row, candidate):
                duplicate_of = candidate
                break
        if duplicate_of is None:
            copy = dict(row)
            copy["duplicate_cluster_size"] = 1
            kept.append(copy)
            continue
        duplicate_of["duplicate_cluster_size"] = int(duplicate_of.get("duplicate_cluster_size", 1)) + 1
        duplicates_removed += 1
    return kept, duplicates_removed


def _round_robin_year_balance(rows: list[dict[str, Any]], years: list[int], top_k: int) -> list[dict[str, Any]]:
    if not rows or not years or top_k <= 0:
        return rows[:top_k]
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
    selected = _round_robin_year_balance(deduped, list(years or []), top_k) if years else deduped[:top_k]
    return _normalize_result_rows(selected[:top_k], retrieval_mode), duplicates_removed

_SPACY_NLP = None
_STANZA_PIPELINES: dict[tuple[str, str], Any] = {}
_FLAIR_OBJECTS: dict[str, Any] = {}
_YFINANCE_SERIES_CACHE: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
MARKET_TICKER_ALIASES = {
    "facebook": "META",
    "meta": "META",
    "amazon": "AMZN",
    "aws": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "microsoft": "MSFT",
    "apple": "AAPL",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "uber": "UBER",
    "paypal": "PYPL",
}


def _sql_fallback_store(context: "AgentExecutionContext") -> PostgresWorkingSetStore | None:
    if isinstance(context.working_store, PostgresWorkingSetStore):
        return context.working_store
    dsn = os.getenv("CORPUSAGENT2_PG_DSN", "").strip()
    if not dsn:
        return None
    table = os.getenv("CORPUSAGENT2_PG_TABLE", "article_corpus").strip() or "article_corpus"
    return PostgresWorkingSetStore(dsn=dsn, documents_table=table)


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
    vector_expr = (
        f"setweight(to_tsvector('simple', {title_expr}), 'A') || "
        f"setweight(to_tsvector('simple', {text_expr}), 'B')"
    )
    date_clauses, date_params = _sql_date_filters(columns["date"], date_from, date_to)
    query_text = " ".join(tokens)
    sql = (
        f"SELECT "
        f"{columns['doc_id']}::text AS doc_id, "
        f"{title_expr} AS title, "
        f"SUBSTRING({text_expr} FROM 1 FOR 360) AS snippet, "
        f"{source_expr} AS outlet, "
        f"{date_expr} AS date, "
        f"ts_rank_cd({vector_expr}, websearch_to_tsquery('simple', %s)) AS score "
        f"FROM {table_name} "
        f"WHERE {vector_expr} @@ websearch_to_tsquery('simple', %s)"
    )
    if date_clauses:
        sql += f" AND {' AND '.join(date_clauses)}"
    sql += " ORDER BY score DESC LIMIT %s"
    params: list[Any] = [query_text, query_text, *date_params, int(top_k)]
    try:
        with store._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()
    except Exception:
        like_clauses: list[str] = []
        like_score_parts: list[str] = []
        like_params: list[Any] = []
        for token in tokens:
            needle = f"%{token.lower()}%"
            title_match = f"LOWER({title_expr}) LIKE %s"
            text_match = f"LOWER({text_expr}) LIKE %s"
            like_clauses.append(f"({title_match} OR {text_match})")
            like_score_parts.append(
                f"(CASE WHEN {title_match} THEN 2 ELSE 0 END + CASE WHEN {text_match} THEN 1 ELSE 0 END)"
            )
            like_params.extend([needle, needle, needle, needle])
        fallback_sql = (
            f"SELECT "
            f"{columns['doc_id']}::text AS doc_id, "
            f"{title_expr} AS title, "
            f"SUBSTRING({text_expr} FROM 1 FOR 360) AS snippet, "
            f"{source_expr} AS outlet, "
            f"{date_expr} AS date, "
            f"({' + '.join(like_score_parts)})::float AS score "
            f"FROM {table_name} "
            f"WHERE ({' OR '.join(like_clauses)})"
        )
        if date_clauses:
            fallback_sql += f" AND {' AND '.join(date_clauses)}"
        fallback_sql += " ORDER BY score DESC LIMIT %s"
        with store._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(fallback_sql, tuple(like_params + date_params + [int(top_k)]))
                rows = cursor.fetchall()
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
        return self._run_fn(params, dependency_results, context)


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
    if lowered in STOPWORDS:
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


def _doc_rows(dependency_results: dict[str, ToolExecutionResult]) -> list[dict[str, Any]]:
    for result in dependency_results.values():
        payload = result.payload
        if isinstance(payload, dict):
            if "documents" in payload:
                return list(payload["documents"])
            if "results" in payload and payload["results"] and "text" in payload["results"][0]:
                return list(payload["results"])
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
    text = str(value)
    mode = str(granularity or _default_time_granularity()).strip().lower() or "month"
    if mode == "day":
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            return text[:10]
        if len(text) >= 7 and text[4] == "-":
            return text[:7]
        if len(text) >= 4:
            return text[:4]
        return "unknown"
    if len(text) >= 7 and text[4] == "-":
        return text[:7]
    if mode == "month":
        if len(text) >= 4:
            return text[:4]
        return "unknown"
    if len(text) >= 4:
        return text[:4]
    return "unknown"


def _row_timestamp(row: dict[str, Any]) -> str:
    return str(row.get("date", row.get("published_at", "")))


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
    slug = quote(normalized.replace(" ", "_"))
    kb_id = f"kb:{slug.lower()}"
    url = f"https://www.wikidata.org/wiki/Special:EntityPage/{slug}"
    return {
        "entity": normalized,
        "label": label,
        "kb_id": kb_id,
        "kb_url": url,
        "confidence": 0.35 if normalized else 0.0,
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


def _sentence_embedding_model_id(context: AgentExecutionContext, params: dict[str, Any]) -> str:
    model_id = str(params.get("model_id", "")).strip()
    if model_id:
        return model_id
    if context.runtime is not None and getattr(context.runtime, "dense_model_id", ""):
        return str(context.runtime.dense_model_id)
    return "intfloat/e5-base-v2"


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
    lowered = text.lower()
    for alias, ticker in MARKET_TICKER_ALIASES.items():
        if alias in lowered:
            return ticker
    ticker_match = re.search(r"\b[A-Z]{1,5}\b", text)
    if ticker_match:
        return ticker_match.group(0)
    return ""


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
    top_k = int(params.get("top_k", 20))
    date_from = str(params.get("date_from", "")).strip()
    date_to = str(params.get("date_to", "")).strip()
    retrieval_mode = str(params.get("retrieval_mode", os.getenv("CORPUSAGENT2_DEFAULT_RETRIEVAL_MODE", "hybrid"))).strip().lower() or "hybrid"
    lexical_top_k = int(params.get("lexical_top_k", top_k * 5 or 50))
    dense_top_k = int(params.get("dense_top_k", top_k * 5 or 50))
    use_rerank = str(params.get("use_rerank", os.getenv("CORPUSAGENT2_RETRIEVAL_USE_RERANK", "true"))).strip().lower() not in {"0", "false", "no", "off"}
    rerank_top_k = int(params.get("rerank_top_k", os.getenv("CORPUSAGENT2_RETRIEVAL_RERANK_TOP_K", "25")))
    fusion_k = int(params.get("fusion_k", os.getenv("CORPUSAGENT2_RETRIEVAL_FUSION_K", "60")))
    year_balance_mode = str(params.get("year_balance", "auto")).strip().lower() or "auto"
    allow_local_fallback = _env_flag("CORPUSAGENT2_ALLOW_LOCAL_FALLBACK", True)
    require_backend_services = _env_flag("CORPUSAGENT2_REQUIRE_BACKEND_SERVICES", False)
    if require_backend_services:
        allow_local_fallback = False
    caveats: list[str] = []
    rows: list[dict[str, Any]] = []
    primary_error: Exception | None = None
    years = _year_range(date_from, date_to)
    use_year_balance = bool(years) and year_balance_mode not in {"0", "false", "no", "off"}
    if year_balance_mode == "auto":
        use_year_balance = len(years) >= 2

    def _primary_search_once(window_from: str, window_to: str, limit: int) -> list[dict[str, Any]]:
        return context.search_backend.search(
            query=query,
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
        if sql_rows and (not rows or _rows_match_query_anchor_terms(sql_rows, query)):
            rows = sql_rows
            if primary_error is not None:
                caveats.append(f"Primary search backend failed and Postgres SQL retrieval was used instead: {primary_error}")
            elif weak_anchor_match:
                caveats.append("Hybrid retrieval returned off-topic documents, so Postgres SQL retrieval was used instead.")
            if sql_duplicates_removed > 0:
                caveats.append(f"Suppressed {sql_duplicates_removed} near-duplicate SQL retrieval hits.")
        elif weak_anchor_match:
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
        if context.runtime is None or not allow_local_fallback:
            raise primary_error
        from .agent_backends import LocalSearchBackend

        local_rows = LocalSearchBackend(context.runtime).search(
            query=query,
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
    return ToolExecutionResult(
        payload={"results": rows, "query": query, "retrieval_mode": retrieval_mode},
        evidence=list(rows),
        caveats=caveats,
    )


def _sql_query_search(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    query = str(
        params.get("query")
        or getattr(context.state, "rewritten_question", "")
        or getattr(context.state, "question", "")
    ).strip()
    top_k = int(params.get("top_k", 20))
    date_from = str(params.get("date_from", "")).strip()
    date_to = str(params.get("date_to", "")).strip()
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
        payload={"results": rows, "query": query, "retrieval_mode": "sql"},
        evidence=list(rows),
        caveats=caveats,
    )


def _fetch_documents(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    doc_ids = [str(item) for item in params.get("doc_ids", []) if str(item).strip()]
    search_rows = _search_rows(deps)
    search_lookup = {
        str(row.get("doc_id", "")).strip(): row
        for row in search_rows
        if str(row.get("doc_id", "")).strip()
    }
    if not doc_ids:
        doc_ids = [str(row.get("doc_id", "")) for row in search_rows if str(row.get("doc_id", "")).strip()]
    if not doc_ids:
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
        return merged

    caveats: list[str] = []
    try:
        rows = context.working_store.fetch_documents(doc_ids)
    except Exception as exc:
        if not allow_local_fallback:
            raise
        rows = []
        caveats.append(f"Working-set document fetch failed and runtime fallback was used: {exc}")
    if not rows and context.runtime is not None and allow_local_fallback:
        try:
            df = context.runtime.load_docs(doc_ids)
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
    return ToolExecutionResult(
        payload={"documents": rows},
        evidence=[{"doc_id": row["doc_id"], "score": row.get("score", 0.0)} for row in rows],
        caveats=caveats,
    )


def _create_working_set(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = _doc_rows(deps)
    context.working_store.record_documents(context.run_id, rows)
    if context.state is not None:
        context.state.working_set_doc_ids = [str(row.get("doc_id", "")) for row in rows]
    return ToolExecutionResult(
        payload={
            "working_set_doc_ids": [str(row.get("doc_id", "")) for row in rows],
            "document_count": len(rows),
        }
    )


def _lang_id(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = _doc_rows(deps)
    detected = []
    for row in rows:
        language, confidence = _infer_language(str(row.get("text", "")))
        detected.append(
            {
                "doc_id": str(row.get("doc_id", "")),
                "language": language,
                "confidence": confidence,
            }
        )
    return ToolExecutionResult(
        payload={"rows": detected},
        caveats=["Prototype language detection uses lightweight lexical heuristics when no dedicated model is installed."],
    )


def _clean_normalize(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = _doc_rows(deps)
    cleaned = [
        {
            "doc_id": row["doc_id"],
            "cleaned_text": " ".join(str(row.get("text", "")).split()).strip(),
        }
        for row in rows
    ]
    return ToolExecutionResult(payload={"rows": cleaned})


def _tokenize_docs(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = _doc_rows(deps)
    providers = _provider_order("tokenize", ["spacy", "stanza", "nltk", "regex"])
    output = []
    used_provider = "regex"
    for row in rows:
        text = str(row.get("text", ""))
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
    providers = _provider_order("sentence_split", ["spacy", "stanza", "nltk", "heuristic"])
    output = []
    used_provider = "heuristic"
    for row in rows:
        text = str(row.get("text", ""))
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
        output.append({"doc_id": row["doc_id"], "sentences": sentences or simple_sentence_split(text)})
    return ToolExecutionResult(
        payload={"rows": output},
        metadata=_metadata(used_provider, f"{used_provider}_sentence_split"),
    )


def _pos_morph(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = _doc_rows(deps)
    output: list[dict[str, Any]] = []
    providers = _provider_order("pos_morph", ["spacy", "stanza", "flair", "nltk", "heuristic"])
    used_provider = "heuristic"
    for provider in providers:
        try:
            if provider == "spacy":
                nlp = _load_spacy_model()
                if nlp is None or "tagger" not in getattr(nlp, "pipe_names", []):
                    continue
                docs = nlp.pipe([str(row.get("text", "")) for row in rows], batch_size=16)
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
                break
            if provider == "stanza":
                pipeline = _load_stanza_pipeline("tokenize,pos,lemma,depparse")
                if pipeline is None:
                    continue
                for row in rows:
                    doc = pipeline(str(row.get("text", "")))
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
                break
            if provider == "flair":
                tagger = _load_flair_object("pos")
                if tagger is None:
                    continue
                from flair.data import Sentence

                for row in rows:
                    sentence = Sentence(str(row.get("text", "")))
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
                break
            if provider == "nltk" and _module_available("nltk"):
                import nltk

                for row in rows:
                    tagged = nltk.pos_tag(nltk.word_tokenize(str(row.get("text", ""))))
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

    if not output:
        for row in rows:
            for token in _tokenize(str(row.get("text", ""))):
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
    return ToolExecutionResult(
        payload={"rows": output},
        metadata=_metadata(used_provider, f"{used_provider}_pos_morph"),
    )


def _lemmatize_docs(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = _doc_rows(deps)
    providers = _provider_order("lemmatize", ["spacy", "stanza", "textblob", "heuristic"])
    output = []
    used_provider = "heuristic"
    for provider in providers:
        try:
            if provider == "spacy":
                nlp = _load_spacy_model()
                if nlp is None:
                    continue
                docs = nlp.pipe([str(row.get("text", "")) for row in rows], batch_size=16)
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
                    doc = pipeline(str(row.get("text", "")))
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
                    blob = TextBlob(str(row.get("text", "")))
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
    if not output:
        output = [
            {"doc_id": row["doc_id"], "lemmas": [_lemma(token) for token in _tokenize(str(row.get("text", "")))]}
            for row in rows
        ]
    return ToolExecutionResult(
        payload={"rows": output},
        metadata=_metadata(used_provider, f"{used_provider}_lemmatize"),
    )


def _dependency_parse(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = _doc_rows(deps)
    parsed = []
    for row in rows:
        sentences = simple_sentence_split(str(row.get("text", "")))[:8]
        deps_rows = []
        for sentence in sentences:
            tokens = _tokenize(sentence)
            for idx, token in enumerate(tokens[1:], start=1):
                deps_rows.append({"head": tokens[idx - 1], "child": token, "dep": "next"})
        parsed.append({"doc_id": row["doc_id"], "dependencies": deps_rows})
    return ToolExecutionResult(payload={"rows": parsed})


def _noun_chunks(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    pos_rows = []
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload and payload["rows"] and "pos" in payload["rows"][0]:
            pos_rows = payload["rows"]
            break
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
    return ToolExecutionResult(
        payload={"rows": [{"doc_id": doc_id, "noun_chunks": values} for doc_id, values in chunks.items()]}
    )


def _ner(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = _doc_rows(deps)
    entities: list[dict[str, Any]] = []
    providers = _provider_order("ner", ["spacy", "stanza", "flair", "regex"])
    used_provider = "regex"
    for provider in providers:
        try:
            if provider == "spacy":
                nlp = _load_spacy_model()
                if nlp is None or "ner" not in getattr(nlp, "pipe_names", []):
                    continue
                docs = nlp.pipe([str(row.get("text", "")) for row in rows], batch_size=16)
                for row, doc in zip(rows, docs, strict=False):
                    for ent in doc.ents:
                        entities.append(
                            {
                                "doc_id": str(row.get("doc_id", "")),
                                "entity": ent.text.strip(),
                                "label": ent.label_,
                                "outlet": str(row.get("outlet", row.get("source", ""))),
                                "time_bin": _time_bin(_row_timestamp(row)),
                            }
                        )
                used_provider = "spacy"
                break
            if provider == "stanza":
                pipeline = _load_stanza_pipeline("tokenize,ner")
                if pipeline is None:
                    continue
                for row in rows:
                    doc = pipeline(str(row.get("text", "")))
                    for ent in getattr(doc, "ents", []):
                        entities.append(
                            {
                                "doc_id": str(row.get("doc_id", "")),
                                "entity": ent.text.strip(),
                                "label": ent.type,
                                "outlet": str(row.get("outlet", row.get("source", ""))),
                                "time_bin": _time_bin(_row_timestamp(row)),
                            }
                        )
                used_provider = "stanza"
                break
            if provider == "flair":
                tagger = _load_flair_object("ner")
                if tagger is None:
                    continue
                from flair.data import Sentence

                for row in rows:
                    sentence = Sentence(str(row.get("text", "")))
                    tagger.predict(sentence)
                    for entity in sentence.get_spans("ner"):
                        entities.append(
                            {
                                "doc_id": str(row.get("doc_id", "")),
                                "entity": entity.text.strip(),
                                "label": entity.get_label("ner").value,
                                "outlet": str(row.get("outlet", row.get("source", ""))),
                                "time_bin": _time_bin(_row_timestamp(row)),
                            }
                        )
                used_provider = "flair"
                break
        except Exception:
            entities = []
            continue

    if not entities:
        for row in rows:
            for match in ENTITY_PATTERN.finditer(str(row.get("text", ""))):
                entities.append(
                    {
                        "doc_id": str(row.get("doc_id", "")),
                        "entity": match.group(0).strip(),
                        "label": "ENTITY",
                        "outlet": str(row.get("outlet", row.get("source", ""))),
                        "time_bin": _time_bin(_row_timestamp(row)),
                    }
                )
    return ToolExecutionResult(
        payload={"rows": entities},
        metadata=_metadata(used_provider, f"{used_provider}_ner"),
    )


def _entity_link(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    entity_rows = []
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload and payload["rows"]:
            first = payload["rows"][0]
            if "entity" in first:
                entity_rows = list(payload["rows"])
                break
    if not entity_rows:
        entity_rows = _ner(params, deps, context).payload["rows"]
    linked = []
    for row in entity_rows:
        link_payload = _link_entity_row(str(row.get("entity", "")), str(row.get("label", "")))
        linked.append(
            {
                "doc_id": str(row.get("doc_id", "")),
                **link_payload,
                "outlet": str(row.get("outlet", "")),
                "time_bin": str(row.get("time_bin", "")),
            }
        )
    return ToolExecutionResult(
        payload={"rows": linked},
        caveats=["Entity linking is optional and currently uses a deterministic URI placeholder scheme unless a knowledge base is integrated."],
    )


def _extract_keyterms(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    joined = "\n".join(str(row.get("text", "")) for row in _doc_rows(deps))
    keyterms = [{"term": term, "score": float(score)} for term, score in textrank_keywords(joined, top_k=25)]
    return ToolExecutionResult(payload={"rows": keyterms})


def _extract_svo_triples(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    triples = []
    for row in _doc_rows(deps):
        for sentence in simple_sentence_split(str(row.get("text", "")))[:6]:
            tokens = _tokenize(sentence)
            if len(tokens) >= 3:
                triples.append(
                    {
                        "doc_id": row["doc_id"],
                        "subject": tokens[0],
                        "verb": tokens[1],
                        "object": " ".join(tokens[2:5]),
                    }
                )
    return ToolExecutionResult(payload={"rows": triples})


def _topic_model(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = _doc_rows(deps)
    providers = _provider_order("topic_model", ["textacy", "gensim", "heuristic"])
    payload: list[dict[str, Any]] = []
    used_provider = "heuristic"
    texts = [str(row.get("text", "")) for row in rows]
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
                        tprep.normalize.whitespace(tprep.remove.punctuation(str(item.get("text", "")).lower()))
                        for item in bucket
                        if str(item.get("text", "")).strip()
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
                        top_indices = component.argsort()[::-1][:10]
                        payload.append(
                            {
                                "topic_id": topic_counter,
                                "time_bin": time_bin,
                                "top_terms": [str(vocab[item]) for item in top_indices],
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
                        [token.lower() for token in _tokenize(str(item.get("text", ""))) if token.lower() not in STOPWORDS]
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
                        payload.append(
                            {
                                "topic_id": topic_counter,
                                "time_bin": time_bin,
                                "top_terms": [term for term, _ in model.show_topic(topic_id, topn=10)],
                                "weight": float(sum(weight for _, weight in model.get_topic_terms(topic_id, topn=20))),
                            }
                        )
                        topic_counter += 1
                used_provider = "gensim"
                break
        except Exception:
            payload = []
            continue

    if not payload:
        grouped: defaultdict[str, Counter] = defaultdict(Counter)
        for row in rows:
            time_bin = _time_bin(_row_timestamp(row), granularity)
            for token in _tokenize(str(row.get("text", "")).lower()):
                if token in STOPWORDS:
                    continue
                grouped[time_bin][token] += 1
        for idx, (time_bin, counts) in enumerate(sorted(grouped.items()), start=1):
            payload.append(
                {
                    "topic_id": idx,
                    "time_bin": time_bin,
                    "top_terms": [term for term, _ in counts.most_common(10)],
                    "weight": float(sum(counts.values())),
                }
            )
    return ToolExecutionResult(
        payload={"rows": payload},
        metadata=_metadata(used_provider, f"{used_provider}_topic_model"),
    )


def _readability_stats(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = []
    for doc in _doc_rows(deps):
        text = str(doc.get("text", ""))
        sentences = simple_sentence_split(text)
        tokens = _tokenize(text)
        avg_sentence_len = len(tokens) / max(len(sentences), 1)
        avg_word_len = sum(len(token) for token in tokens) / max(len(tokens), 1)
        rows.append({"doc_id": doc["doc_id"], "avg_sentence_len": avg_sentence_len, "avg_word_len": avg_word_len})
    return ToolExecutionResult(payload={"rows": rows})


def _lexical_diversity(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = []
    for doc in _doc_rows(deps):
        tokens = [token.lower() for token in _tokenize(str(doc.get("text", "")))]
        rows.append({"doc_id": doc["doc_id"], "type_token_ratio": len(set(tokens)) / max(len(tokens), 1)})
    return ToolExecutionResult(payload={"rows": rows})


def _extract_ngrams(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    n = int(params.get("n", 2))
    counts: Counter = Counter()
    for doc in _doc_rows(deps):
        tokens = [token.lower() for token in _tokenize(str(doc.get("text", ""))) if token.lower() not in STOPWORDS]
        for idx in range(len(tokens) - n + 1):
            counts[" ".join(tokens[idx : idx + n])] += 1
    return ToolExecutionResult(
        payload={"rows": [{"ngram": ngram, "count": int(count)} for ngram, count in counts.most_common(25)]}
    )


def _extract_acronyms(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    pattern = re.compile(r"\b[A-Z]{2,8}\b")
    counts: Counter = Counter()
    for doc in _doc_rows(deps):
        counts.update(match.group(0) for match in pattern.finditer(str(doc.get("text", ""))))
    return ToolExecutionResult(
        payload={"rows": [{"acronym": item, "count": int(count)} for item, count in counts.most_common(20)]}
    )


def _sentiment(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = []
    docs = _doc_rows(deps)
    providers = _provider_order("sentiment", ["flair", "textblob", "heuristic"])
    used_provider = "heuristic"
    for provider in providers:
        try:
            if provider == "flair":
                classifier = _load_flair_object("sentiment")
                if classifier is None:
                    continue
                from flair.data import Sentence

                rows = []
                for doc in docs:
                    sentence = Sentence(str(doc.get("text", ""))[:1500])
                    classifier.predict(sentence)
                    label = sentence.labels[0].value.lower() if sentence.labels else "neutral"
                    confidence = float(sentence.labels[0].score) if sentence.labels else 0.0
                    rows.append(
                        {
                            "doc_id": doc["doc_id"],
                            "score": confidence if label == "positive" else -confidence if label == "negative" else 0.0,
                            "label": label,
                            "time_bin": _time_bin(_row_timestamp(doc)),
                        }
                    )
                used_provider = "flair"
                break
            if provider == "textblob" and _module_available("textblob"):
                from textblob import TextBlob

                rows = []
                for doc in docs:
                    polarity = float(TextBlob(str(doc.get("text", ""))).sentiment.polarity)
                    label = "positive" if polarity > 0.05 else "negative" if polarity < -0.05 else "neutral"
                    rows.append(
                        {
                            "doc_id": doc["doc_id"],
                            "score": polarity,
                            "label": label,
                            "time_bin": _time_bin(_row_timestamp(doc)),
                        }
                    )
                used_provider = "textblob"
                break
        except Exception:
            rows = []
            continue

    if not rows:
        for doc in docs:
            tokens = [token.lower() for token in _tokenize(str(doc.get("text", "")))]
            score = sum(1 for token in tokens if token in POSITIVE_WORDS) - sum(
                1 for token in tokens if token in NEGATIVE_WORDS
            )
            rows.append(
                {
                    "doc_id": doc["doc_id"],
                    "score": float(score),
                    "label": "positive" if score > 0 else "negative" if score < 0 else "neutral",
                    "time_bin": _time_bin(_row_timestamp(doc)),
                }
            )
    return ToolExecutionResult(
        payload={"rows": rows},
        metadata=_metadata(used_provider, f"{used_provider}_sentiment"),
    )


def _text_classify(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    sentiment_rows = _sentiment(params, deps, context).payload["rows"]
    return ToolExecutionResult(
        payload={
            "rows": [
                {"doc_id": row["doc_id"], "labels": [row["label"]], "probs": [abs(float(row["score"]))]}
                for row in sentiment_rows
            ]
        }
    )


def _word_embeddings(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    vocab = sorted(
        {
            token.lower()
            for doc in _doc_rows(deps)
            for token in _tokenize(str(doc.get("text", "")))
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
    texts = [f"{str(doc.get('title', ''))} {str(doc.get('text', ''))}".strip() for doc in docs]
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
    texts = [f"{str(doc.get('title', ''))} {str(doc.get('text', ''))}".strip() for doc in docs]
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
        texts = [query] + [f"{str(doc.get('title', ''))} {str(doc.get('text', ''))}".strip() for doc in docs]
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
        texts = [f"{str(doc.get('title', ''))} {str(doc.get('text', ''))}".strip() for doc in docs]
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


def _time_series_aggregate(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    source_rows = []
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload:
            source_rows = list(payload["rows"])
            break
    granularity = str(params.get("granularity", _default_time_granularity())).strip().lower() or "month"
    grouped: defaultdict[tuple[str, str], int] = defaultdict(int)
    for row in source_rows:
        entity = str(
            row.get("entity")
            or row.get("label")
            or row.get("term")
            or (f"topic_{row.get('topic_id')}" if row.get("topic_id") is not None else "")
            or row.get("doc_id")
            or "__all__"
        )
        timestamp = str(row.get("date") or row.get("published_at") or row.get("time_bin") or "")
        time_bin = _time_bin(timestamp, granularity)
        value = row.get("count", row.get("weight", row.get("score", 1)))
        grouped[(entity, time_bin)] += int(round(float(value)))
    rows = [{"entity": entity, "time_bin": time_bin, "count": count} for (entity, time_bin), count in sorted(grouped.items())]
    return ToolExecutionResult(payload={"rows": rows})


def _change_point_detect(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    series = []
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload:
            series = list(payload["rows"])
            break
    by_entity: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in series:
        by_entity[str(row.get("entity", "__all__"))].append(
            (str(row.get("time_bin", "unknown")), float(row.get("count", row.get("score", 0.0))))
        )
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
    return ToolExecutionResult(payload={"rows": changes})


def _burst_detect(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    series = _time_series_aggregate(params, deps, context).payload["rows"]
    by_entity: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in series:
        by_entity[str(row.get("entity", "__all__"))].append((str(row.get("time_bin", "unknown")), float(row.get("count", 0))))
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
    return ToolExecutionResult(payload={"rows": bursts})


def _claim_span_extract(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = []
    for doc in _doc_rows(deps):
        for sentence in simple_sentence_split(str(doc.get("text", ""))):
            lowered = sentence.lower()
            matched = [keyword for keyword in CLAIM_KEYWORDS if keyword in lowered]
            if not matched:
                continue
            rows.append(
                {
                    "doc_id": str(doc.get("doc_id", "")),
                    "outlet": str(doc.get("outlet", doc.get("source", ""))),
                    "date": str(doc.get("date", doc.get("published_at", ""))),
                    "excerpt": sentence[:320],
                    "matched_keywords": matched,
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
        excerpt = str(row.get("excerpt", "")).lower()
        score = 0.3
        if "imminent" in excerpt or "within days" in excerpt:
            score += 0.35
        if "will invade" in excerpt or "could invade" in excerpt:
            score += 0.25
        if "warn" in excerpt or "prediction" in excerpt or "predicted" in excerpt:
            score += 0.15
        scored.append({**row, "score": round(min(score, 1.0), 3)})
    scored.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return ToolExecutionResult(payload={"rows": scored})


def _quote_extract(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    rows = []
    for doc in _doc_rows(deps):
        for match in QUOTE_PATTERN.finditer(str(doc.get("text", ""))):
            rows.append({"doc_id": doc["doc_id"], "quote": match.group(1), "text": str(doc.get("text", ""))[:500]})
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
        attributed.append({**row, "speaker": speaker_match.group(1) if speaker_match else "unknown"})
    return ToolExecutionResult(payload={"rows": attributed})


def _build_evidence_table(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
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

    history = yf.download(
        normalized_ticker,
        start=start or None,
        end=end or None,
        interval=interval or "1d",
        auto_adjust=False,
        progress=False,
    )
    if history is None or history.empty:
        _YFINANCE_SERIES_CACHE[cache_key] = []
        return []
    if hasattr(history.columns, "levels"):
        history.columns = history.columns.get_level_values(0)
    frame = history.reset_index()
    date_column = "Date" if "Date" in frame.columns else frame.columns[0]
    rows: list[dict[str, Any]] = []
    previous_close: float | None = None
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
                "date": date_text,
                "time_bin": _time_bin(date_text),
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


def _join_external_series(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    external_rows = list(params.get("series_rows", []))
    ticker = str(params.get("ticker", "")).strip() or _infer_market_ticker_from_text(
        str(getattr(context.state, "rewritten_question", "") or getattr(context.state, "question", ""))
    )
    start = str(params.get("start", params.get("date_from", ""))).strip()
    end = str(params.get("end", params.get("date_to", ""))).strip()
    interval = str(params.get("interval", "1d")).strip() or "1d"
    left_rows: list[dict[str, Any]] = []
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload:
            left_rows = list(payload["rows"])
            if left_rows:
                break
    if not left_rows:
        return ToolExecutionResult(payload={"rows": []}, caveats=["No internal rows available to join with external series."])

    caveats: list[str] = []
    provider = "analytics"
    if not external_rows and ticker:
        try:
            external_rows = _fetch_yfinance_series_rows(
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
            )
            provider = "yfinance"
        except Exception as exc:
            caveats.append(f"External market series fetch failed for ticker '{ticker}': {exc}")

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

    left_key = str(params.get("left_key", "time_bin"))
    right_key = str(params.get("right_key", left_key))
    if left_key not in left_df.columns:
        if "date" in left_df.columns:
            left_df[left_key] = left_df["date"].astype(str).map(_time_bin)
        else:
            return ToolExecutionResult(payload={"rows": left_rows}, caveats=[f"Join key '{left_key}' is missing from internal rows."])
    if right_key not in right_df.columns:
        return ToolExecutionResult(payload={"rows": left_rows}, caveats=[f"Join key '{right_key}' is missing from external series rows."])

    merged = left_df.merge(
        right_df,
        how=str(params.get("how", "left")),
        left_on=left_key,
        right_on=right_key,
        suffixes=("", "_external"),
    )
    return ToolExecutionResult(
        payload={"rows": merged.fillna("").to_dict(orient="records")},
        caveats=caveats,
        metadata=_metadata(provider, f"{provider}_join_external_series", ticker=ticker, interval=interval),
    )


def _plot_artifact(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        return ToolExecutionResult(payload={"rows": []}, caveats=[f"matplotlib unavailable: {exc}"])
    rows = []
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload:
            rows = list(payload["rows"])
            if rows:
                break
    if not rows:
        return ToolExecutionResult(payload={"rows": []}, caveats=["No rows available for plotting."])
    plot_dir = context.artifacts_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    target = plot_dir / f"{params.get('plot_name', 'plot')}.png"
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(12.5, 6.6), dpi=180)
    plot_name = str(params.get("plot_name", "plot")).replace("_", " ").title()
    first = rows[:16]
    unique_time_bins = {
        str(item.get("time_bin", "unknown"))
        for item in first
        if item.get("time_bin") is not None
    }
    topic_like = bool(first and "topic_id" in first[0] and any(item.get("top_terms") for item in first))
    market_overlay_like = bool(first and "market_close" in first[0] and any("count" in item or "score" in item for item in first))
    time_series_like = bool(first and "time_bin" in first[0] and len(unique_time_bins) > 1)

    if topic_like:
        labels = []
        values = []
        for index, item in enumerate(first):
            top_terms = [str(term) for term in item.get("top_terms", [])[:4] if str(term).strip()]
            label = f"Topic {item.get('topic_id', index + 1)}"
            if top_terms:
                label = f"{label}: {', '.join(top_terms)}"
            labels.append(label)
            values.append(float(item.get("weight", 0.0)))
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
        signal_values = [float(item.get("count", item.get("score", item.get("weight", 0.0)))) for item in ordered]
        market_values = [float(item.get("market_close", 0.0)) for item in ordered]
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
        series_by_entity: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
        for item in rows:
            entity = str(item.get("entity", item.get("label", item.get("term", "__all__"))))
            value = float(item.get("count", item.get("score", item.get("weight", item.get("intensity", 0.0)))))
            series_by_entity[entity].append((str(item.get("time_bin", "unknown")), value))
        top_entities = sorted(
            series_by_entity.items(),
            key=lambda entry: sum(value for _, value in entry[1]),
            reverse=True,
        )[:5]
        palette = ["#0f766e", "#b45309", "#7c3aed", "#be123c", "#2563eb"]
        for index, (entity, points) in enumerate(top_entities):
            ordered = sorted(points)
            x_labels = [item[0] for item in ordered]
            x_values = list(range(len(x_labels)))
            y_values = [item[1] for item in ordered]
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2.4,
                color=palette[index % len(palette)],
                label=entity,
            )
            axis.fill_between(x_values, y_values, alpha=0.08, color=palette[index % len(palette)])
        if top_entities:
            axis.set_xticks(list(range(len(x_labels))), x_labels, rotation=35, ha="right")
        axis.set_ylabel("Signal")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best", fontsize=8, frameon=True)
    else:
        labels = [
            str(item.get("entity", item.get("term", item.get("doc_id", item.get("time_bin", "row")))))
            for item in first
        ]
        values = [
            float(item.get("count", item.get("score", item.get("weight", item.get("intensity", 0.0)))))
            for item in first
        ]
        colors = ["#0f766e" if value >= 0 else "#be123c" for value in values]
        bars = axis.bar(labels, values, color=colors, edgecolor="#17312d", linewidth=0.5)
        for bar, value in zip(bars, values, strict=False):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
        axis.tick_params(axis="x", rotation=35)

    axis.set_title(plot_name, fontsize=15, fontweight="bold")
    axis.set_facecolor("#fffaf2")
    figure.patch.set_facecolor("#fffdf8")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return ToolExecutionResult(
        payload={"artifact_path": str(target), "rows": first, "plot_name": plot_name},
        artifacts=[str(target)],
    )


def _python_runner(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    if context.python_runner is None:
        raise RuntimeError("python_runner is unavailable")
    code = str(params.get("code", "")).strip()
    inputs_json = dict(params.get("inputs_json", {}))
    if not inputs_json:
        for key, result in deps.items():
            inputs_json[key] = result.payload
    result = context.python_runner.run(code=code, inputs_json=inputs_json)
    import base64

    artifacts = []
    for artifact in result.artifacts:
        artifact_path = context.artifacts_dir / "python_runner" / artifact.name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(base64.b64decode(artifact.bytes_b64.encode("ascii")))
        artifacts.append(str(artifact_path))
    return ToolExecutionResult(
        payload=result.to_dict(),
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
        ("plot_artifact", "plot_artifact", "matplotlib", 58, _plot_artifact),
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
