from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
import re
from typing import Any, Callable
from urllib.parse import quote

import pandas as pd

from .agent_backends import SearchBackend, WorkingSetStore
from .agent_models import EvidenceRow
from .analysis_tools import textrank_keywords
from .io_utils import sentence_split as simple_sentence_split
from .python_runner_service import DockerPythonRunnerService
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
    "over", "under",
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

_SPACY_NLP = None
_STANZA_PIPELINES: dict[tuple[str, str], Any] = {}
_FLAIR_OBJECTS: dict[str, Any] = {}


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


def _time_bin(value: str) -> str:
    text = str(value)
    if len(text) >= 7 and text[4] == "-":
        return text[:7]
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


def _load_stanza_pipeline(processors: str) -> Any | None:
    key = ("en", processors)
    if key in _STANZA_PIPELINES:
        return _STANZA_PIPELINES[key]
    if not _module_available("stanza"):
        return None
    try:
        import stanza

        pipeline = stanza.Pipeline(lang="en", processors=processors, use_gpu=False, verbose=False)
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
    try:
        rows = context.search_backend.search(
            query=query,
            top_k=top_k,
            date_from=date_from,
            date_to=date_to,
        )
    except Exception as exc:
        if context.runtime is None:
            raise
        from .agent_backends import LocalSearchBackend

        rows = LocalSearchBackend(context.runtime).search(
            query=query,
            top_k=top_k,
            date_from=date_from,
            date_to=date_to,
        )
        return ToolExecutionResult(
            payload={"results": rows, "query": query},
            evidence=list(rows),
            caveats=[f"Primary search backend failed and local retrieval fallback was used: {exc}"],
        )
    return ToolExecutionResult(payload={"results": rows, "query": query}, evidence=list(rows))


def _fetch_documents(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    doc_ids = [str(item) for item in params.get("doc_ids", []) if str(item).strip()]
    if not doc_ids:
        doc_ids = [str(row.get("doc_id", "")) for row in _search_rows(deps) if str(row.get("doc_id", "")).strip()]
    rows = context.working_store.fetch_documents(doc_ids)
    if not rows and context.runtime is not None:
        df = context.runtime.load_docs(doc_ids)
        rows = [
            {
                "doc_id": str(row.doc_id),
                "title": str(getattr(row, "title", "")),
                "text": str(getattr(row, "text", "")),
                "published_at": str(getattr(row, "published_at", "")),
                "date": str(getattr(row, "published_at", "")),
                "outlet": str(getattr(row, "source", "")),
                "source": str(getattr(row, "source", "")),
            }
            for row in df.itertuples(index=False)
        ]
    return ToolExecutionResult(
        payload={"documents": rows},
        evidence=[{"doc_id": row["doc_id"]} for row in rows],
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

    for provider in providers:
        try:
            if provider == "textacy" and _module_available("textacy"):
                import textacy.preprocessing as tprep
                from sklearn.decomposition import NMF
                from sklearn.feature_extraction.text import CountVectorizer

                cleaned = [
                    tprep.normalize.whitespace(tprep.remove.punctuation(text.lower()))
                    for text in texts
                ]
                vectorizer = CountVectorizer(stop_words="english", max_features=1000)
                matrix = vectorizer.fit_transform(cleaned)
                if matrix.shape[0] == 0 or matrix.shape[1] == 0:
                    continue
                model = NMF(n_components=min(num_topics, max(1, matrix.shape[0])), init="nndsvda", random_state=42)
                weights = model.fit_transform(matrix)
                vocab = vectorizer.get_feature_names_out()
                for idx, component in enumerate(model.components_, start=1):
                    top_indices = component.argsort()[::-1][:10]
                    payload.append(
                        {
                            "topic_id": idx,
                            "time_bin": "all",
                            "top_terms": [str(vocab[item]) for item in top_indices],
                            "weight": float(weights[:, idx - 1].sum()),
                        }
                    )
                used_provider = "textacy"
                break
            if provider == "gensim" and _module_available("gensim"):
                from gensim import corpora
                from gensim.models import LdaModel

                tokenized = [
                    [token.lower() for token in _tokenize(text) if token.lower() not in STOPWORDS]
                    for text in texts
                ]
                dictionary = corpora.Dictionary(tokenized)
                corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
                if not corpus or len(dictionary) == 0:
                    continue
                model = LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=min(num_topics, max(1, len(dictionary))),
                    random_state=42,
                    iterations=50,
                    passes=5,
                )
                for topic_id in range(model.num_topics):
                    payload.append(
                        {
                            "topic_id": topic_id + 1,
                            "time_bin": "all",
                            "top_terms": [term for term, _ in model.show_topic(topic_id, topn=10)],
                            "weight": float(sum(weight for _, weight in model.get_topic_terms(topic_id, topn=20))),
                        }
                    )
                used_provider = "gensim"
                break
        except Exception:
            payload = []
            continue

    if not payload:
        grouped: defaultdict[str, Counter] = defaultdict(Counter)
        for row in rows:
            time_bin = _time_bin(_row_timestamp(row))
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
    return ToolExecutionResult(
        payload={
            "rows": [
                {"token": token, "vector_ref": f"hash:{sha256(token.encode()).hexdigest()[:12]}"}
                for token in vocab
            ]
        },
        caveats=["Using lightweight placeholder vector references for prototype runtime."],
    )


def _doc_embeddings(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    return ToolExecutionResult(
        payload={
            "rows": [
                {
                    "doc_id": doc["doc_id"],
                    "vector_ref": f"hash:{sha256(str(doc.get('text', '')).encode()).hexdigest()[:12]}",
                }
                for doc in _doc_rows(deps)
            ]
        },
        caveats=["Using lightweight placeholder document vector references for prototype runtime."],
    )


def _similarity_index(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    docs = _doc_rows(deps)
    tokens_by_doc = {
        str(doc["doc_id"]): set(token.lower() for token in _tokenize(str(doc.get("text", ""))))
        for doc in docs
    }
    rows = []
    doc_ids = list(tokens_by_doc.keys())
    for idx, left in enumerate(doc_ids):
        for right in doc_ids[idx + 1 : idx + 4]:
            overlap = len(tokens_by_doc[left].intersection(tokens_by_doc[right]))
            denom = max(len(tokens_by_doc[left].union(tokens_by_doc[right])), 1)
            rows.append({"left_doc_id": left, "right_doc_id": right, "score": overlap / denom})
    return ToolExecutionResult(payload={"rows": sorted(rows, key=lambda item: item["score"], reverse=True)[:20]})


def _similarity_pairwise(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    docs = _doc_rows(deps)
    query = str(params.get("query", getattr(context.state, "rewritten_question", "") or getattr(context.state, "question", ""))).strip()
    query_tokens = set(token.lower() for token in _tokenize(query) if token.lower() not in STOPWORDS)
    rows = []
    if query_tokens:
        for doc in docs:
            doc_tokens = set(token.lower() for token in _tokenize(str(doc.get("text", ""))) if token.lower() not in STOPWORDS)
            overlap = len(query_tokens.intersection(doc_tokens))
            denom = max(len(query_tokens.union(doc_tokens)), 1)
            rows.append(
                {
                    "left_id": "__query__",
                    "right_id": str(doc.get("doc_id", "")),
                    "score": round(overlap / denom, 4),
                }
            )
    else:
        tokens_by_doc = {
            str(doc["doc_id"]): set(token.lower() for token in _tokenize(str(doc.get("text", ""))) if token.lower() not in STOPWORDS)
            for doc in docs
        }
        doc_ids = list(tokens_by_doc.keys())
        for idx, left in enumerate(doc_ids):
            for right in doc_ids[idx + 1 : idx + 8]:
                overlap = len(tokens_by_doc[left].intersection(tokens_by_doc[right]))
                denom = max(len(tokens_by_doc[left].union(tokens_by_doc[right])), 1)
                rows.append({"left_id": left, "right_id": right, "score": round(overlap / denom, 4)})
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return ToolExecutionResult(payload={"rows": rows[:25]})


def _time_series_aggregate(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    source_rows = []
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload:
            source_rows = list(payload["rows"])
            break
    grouped: defaultdict[tuple[str, str], int] = defaultdict(int)
    for row in source_rows:
        entity = str(row.get("entity", row.get("label", row.get("doc_id", "__all__"))))
        time_bin = str(row.get("time_bin", "unknown"))
        grouped[(entity, time_bin)] += int(row.get("count", 1))
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
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload:
            rows = list(payload["rows"])
    evidence_map: dict[str, EvidenceRow] = {}
    for row in rows:
        doc_id = str(row.get("doc_id", ""))
        if not doc_id:
            continue
        score = float(row.get("score", 0.0))
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
    evidence = [item.to_dict() for item in sorted(evidence_map.values(), key=lambda item: item.score, reverse=True)]
    return ToolExecutionResult(payload={"rows": evidence}, evidence=evidence)


def _join_external_series(params: dict[str, Any], deps: dict[str, ToolExecutionResult], context: AgentExecutionContext) -> ToolExecutionResult:
    external_rows = list(params.get("series_rows", []))
    left_rows: list[dict[str, Any]] = []
    for result in deps.values():
        payload = result.payload
        if isinstance(payload, dict) and "rows" in payload:
            left_rows = list(payload["rows"])
            if left_rows:
                break
    if not left_rows:
        return ToolExecutionResult(payload={"rows": []}, caveats=["No internal rows available to join with external series."])

    left_df = pd.DataFrame(left_rows)
    right_df = pd.DataFrame(external_rows)
    if right_df.empty:
        return ToolExecutionResult(payload={"rows": left_rows}, caveats=["No external series rows were provided."])

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
    return ToolExecutionResult(payload={"rows": merged.fillna("").to_dict(orient="records")})


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
    first = rows[:10]
    labels = [
        str(item.get("entity", item.get("term", item.get("doc_id", item.get("time_bin", "row")))))
        for item in first
    ]
    values = [
        float(item.get("count", item.get("score", item.get("weight", item.get("intensity", 0.0)))))
        for item in first
    ]
    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return ToolExecutionResult(payload={"artifact_path": str(target), "rows": first}, artifacts=[str(target)])


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
