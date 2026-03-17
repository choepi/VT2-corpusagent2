from __future__ import annotations

from collections import Counter, defaultdict
import re
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from .io_utils import ensure_absolute, ensure_exists, read_documents, write_json
from .seed import hf_pipeline_device_arg, resolve_device, runtime_device_report, set_global_seed
from .temporal import extract_time_bin, normalize_granularity


TOKEN_PATTERN = re.compile(r"[A-Za-z]{3,}")
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def textrank_keywords(text: str, top_k: int = 25, window_size: int = 4) -> list[tuple[str, float]]:
    tokens = tokenize(text)
    if len(tokens) < 3:
        return []

    graph = nx.Graph()
    for idx, token in enumerate(tokens):
        graph.add_node(token)
        right = min(len(tokens), idx + window_size)
        for jdx in range(idx + 1, right):
            other = tokens[jdx]
            if token == other:
                continue
            weight = graph[token][other]["weight"] + 1.0 if graph.has_edge(token, other) else 1.0
            graph.add_edge(token, other, weight=weight)

    if graph.number_of_nodes() == 0:
        return []

    scores = nx.pagerank(graph, weight="weight")
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


def load_spacy_ner_model(model_name: str) -> tuple[Any | None, str]:
    try:
        import spacy
    except Exception:
        return None, "regex_fallback"

    candidates = [model_name, "en_core_web_lg", "en_core_web_sm"]
    for candidate in candidates:
        try:
            nlp = spacy.load(candidate)
            return nlp, candidate
        except Exception:
            continue

    try:
        from spacy.cli import download

        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        return nlp, "en_core_web_sm"
    except BaseException:
        return None, "regex_fallback"


def maybe_sample(df: pd.DataFrame, max_docs: int | None, seed: int) -> pd.DataFrame:
    if max_docs is None:
        return df
    if max_docs <= 0:
        return df.head(0).copy()
    if df.shape[0] <= max_docs:
        return df
    return df.sample(n=max_docs, random_state=seed).reset_index(drop=True)


def run_ner(
    df: pd.DataFrame,
    model_name: str,
    granularity: str = "year",
    max_docs: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    granularity = normalize_granularity(granularity)
    df = maybe_sample(df=df, max_docs=max_docs, seed=seed)
    counts: Counter = Counter()
    doc_freq: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    cooccurrence: Counter = Counter()
    nlp, model_used = load_spacy_ner_model(model_name=model_name)

    if nlp is not None:
        docs = nlp.pipe(df["text"].astype(str).tolist(), batch_size=32)
        row_doc_iter = zip(df.itertuples(index=False), docs, strict=False)
        for row, doc in row_doc_iter:
            time_bin = extract_time_bin(str(row.published_at), granularity=granularity)
            entities = [ent.text.strip() for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "EVENT"}]
            unique_entities = sorted(set(entity for entity in entities if entity))

            for entity in entities:
                counts[(entity, time_bin)] += 1
                doc_freq[(entity, time_bin)].add(str(row.doc_id))

            for idx, left in enumerate(unique_entities):
                for right in unique_entities[idx + 1 :]:
                    cooccurrence[(left, right)] += 1
    else:
        for row in df.itertuples(index=False):
            time_bin = extract_time_bin(str(row.published_at), granularity=granularity)
            text = str(row.text)
            entities = [match.group(0).strip() for match in ENTITY_PATTERN.finditer(text)]
            unique_entities = sorted(set(entity for entity in entities if entity))

            for entity in entities:
                counts[(entity, time_bin)] += 1
                doc_freq[(entity, time_bin)].add(str(row.doc_id))

            for idx, left in enumerate(unique_entities):
                for right in unique_entities[idx + 1 :]:
                    cooccurrence[(left, right)] += 1

    top_neighbor_by_entity: dict[str, str] = {}
    for (left, right), _weight in cooccurrence.items():
        if left not in top_neighbor_by_entity:
            top_neighbor_by_entity[left] = right
        if right not in top_neighbor_by_entity:
            top_neighbor_by_entity[right] = left

    rows = []
    for (entity, time_bin), count in counts.items():
        rows.append(
            {
                "entity": entity,
                "time_bin": time_bin,
                "count": int(count),
                "doc_freq": int(len(doc_freq[(entity, time_bin)])),
                "top_cooccurring": top_neighbor_by_entity.get(entity, ""),
                "confidence_stats": "not_available",
                "model_id": model_used,
            }
        )

    return pd.DataFrame(rows)


def run_sentiment(
    df: pd.DataFrame,
    model_id: str,
    granularity: str = "year",
    preferred_device: str | None = None,
    batch_size: int = 16,
    max_docs: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    granularity = normalize_granularity(granularity)
    df = maybe_sample(df=df, max_docs=max_docs, seed=seed)
    from transformers import pipeline

    selected_device = resolve_device(preferred_device)
    pipeline_device = hf_pipeline_device_arg(selected_device)

    try:
        pipe = pipeline("text-classification", model=model_id, tokenizer=model_id, device=pipeline_device)
    except Exception:
        selected_device = "cpu"
        pipe = pipeline("text-classification", model=model_id, tokenizer=model_id, device=-1)

    rows = []
    time_bins = [extract_time_bin(value, granularity=granularity) for value in df["published_at"].astype(str).tolist()]
    texts = [str(text)[:1200] for text in df["text"].astype(str).tolist()]

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_bins = time_bins[start : start + batch_size]
        try:
            preds = pipe(batch_texts, truncation=True, batch_size=batch_size)
        except Exception:
            if selected_device != "cpu":
                selected_device = "cpu"
                pipe = pipeline("text-classification", model=model_id, tokenizer=model_id, device=-1)
                preds = pipe(batch_texts, truncation=True, batch_size=batch_size)
            else:
                preds = [{"label": "neutral", "score": 0.0} for _ in batch_texts]

        for pred, time_bin in zip(preds, batch_bins, strict=False):
            label = str(pred.get("label", "")).lower()
            if "neg" in label:
                score = -1.0
            elif "pos" in label:
                score = 1.0
            else:
                score = 0.0

            rows.append(
                {
                    "entity": "__all__",
                    "time_bin": time_bin,
                    "score": score,
                    "model_id": model_id,
                }
            )

    sentiment_df = pd.DataFrame(rows)
    grouped = (
        sentiment_df.groupby(["entity", "time_bin", "model_id"], as_index=False)
        .agg(mean=("score", "mean"), std=("score", "std"), n_docs=("score", "size"))
        .fillna(0.0)
    )
    return grouped, selected_device


def run_topics_over_time(
    df: pd.DataFrame,
    granularity: str = "year",
    max_docs: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    granularity = normalize_granularity(granularity)
    df = maybe_sample(df=df, max_docs=max_docs, seed=seed)
    if df.empty:
        return pd.DataFrame(columns=["topic_id", "time_bin", "weight", "top_terms", "coherence_proxy"])
    from bertopic import BERTopic

    texts = df["text"].astype(str).tolist()
    topic_model = BERTopic(calculate_probabilities=False, verbose=False)
    topics, _ = topic_model.fit_transform(texts)

    topic_df = pd.DataFrame(
        {
            "topic_id": topics,
            "time_bin": df["published_at"].map(lambda value: extract_time_bin(value, granularity=granularity)),
        }
    )

    totals = topic_df.groupby("time_bin").size().rename("total")
    grouped = topic_df.groupby(["topic_id", "time_bin"]).size().rename("count").reset_index()
    grouped = grouped.merge(totals, on="time_bin", how="left")
    grouped["weight"] = grouped["count"] / grouped["total"].clip(lower=1)

    rows = []
    for row in grouped.itertuples(index=False):
        topic_terms = topic_model.get_topic(int(row.topic_id)) or []
        top_terms = [term for term, _ in topic_terms[:10]]
        coherence_proxy = float(sum(score for _, score in topic_terms[:10]) / max(len(topic_terms[:10]), 1))
        rows.append(
            {
                "topic_id": int(row.topic_id),
                "time_bin": str(row.time_bin),
                "weight": float(row.weight),
                "top_terms": ", ".join(top_terms),
                "coherence_proxy": coherence_proxy,
            }
        )

    return pd.DataFrame(rows)


def run_burst_detection(entity_trend_df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    rows = []
    if entity_trend_df.empty:
        return pd.DataFrame(rows)

    for entity, subset in entity_trend_df.groupby("entity"):
        if subset.shape[0] < 3:
            continue

        subset_sorted = subset.sort_values("time_bin")
        values = subset_sorted["count"].astype(float)
        mean = float(values.mean())
        std = float(values.std())
        if std == 0.0:
            continue

        z_scores = (values - mean) / std
        active_start = None
        active_values: list[float] = []
        time_bins = subset_sorted["time_bin"].tolist()

        for idx, z_value in enumerate(z_scores.tolist()):
            if z_value >= z_threshold:
                if active_start is None:
                    active_start = time_bins[idx]
                active_values.append(z_value)
            elif active_start is not None:
                rows.append(
                    {
                        "entity_or_term": entity,
                        "burst_level": "high",
                        "start": active_start,
                        "end": time_bins[idx - 1],
                        "intensity": float(sum(active_values) / len(active_values)),
                    }
                )
                active_start = None
                active_values = []

        if active_start is not None:
            rows.append(
                {
                    "entity_or_term": entity,
                    "burst_level": "high",
                    "start": active_start,
                    "end": time_bins[-1],
                    "intensity": float(sum(active_values) / len(active_values)),
                }
            )

    return pd.DataFrame(rows)


def run_keyphrases(
    df: pd.DataFrame,
    granularity: str = "year",
    top_k: int = 25,
    max_docs_per_bin: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    granularity = normalize_granularity(granularity)
    rows = []
    for time_bin, subset in df.groupby(df["published_at"].map(lambda value: extract_time_bin(value, granularity=granularity))):
        if subset.shape[0] > max_docs_per_bin:
            subset = subset.sample(n=max_docs_per_bin, random_state=seed)
        combined_text = " ".join(subset["text"].astype(str).tolist())
        ranked = textrank_keywords(combined_text, top_k=top_k)
        docs = subset["text"].astype(str).tolist()

        for phrase, score in ranked:
            doc_freq = sum(1 for text in docs if phrase in text.lower())
            rows.append(
                {
                    "phrase": phrase,
                    "time_bin": str(time_bin),
                    "score": float(score),
                    "doc_freq": int(doc_freq),
                    "method": "textrank",
                }
            )

    return pd.DataFrame(rows)


def build_nlp_outputs(
    documents_path: Path,
    output_dir: Path,
    mode: str = "full",
    seed: int = 42,
    granularity: str = "year",
    ner_model_id: str = "en_core_web_trf",
    sentiment_model_id: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    sentiment_device: str = "cpu",
    debug_max_docs: int = 5_000,
    full_max_docs: int = 624_095,
    ner_max_docs_full: int = 30_000,
    sentiment_max_docs_full: int = 30_000,
    topics_max_docs_full: int = 20_000,
    keyphrases_max_docs_per_bin: int = 2_500,
) -> dict[str, Any]:
    granularity = normalize_granularity(granularity)
    ensure_absolute(documents_path, "DOCUMENTS_PATH")
    ensure_absolute(output_dir, "OUTPUT_DIR")
    ensure_exists(documents_path, "DOCUMENTS_PATH")

    set_global_seed(seed)

    df = read_documents(documents_path)
    if mode == "debug":
        df = df.head(debug_max_docs).copy()
    else:
        df = df.head(full_max_docs).copy()

    if df.empty:
        raise RuntimeError("Input documents are empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"

    errors: dict[str, str] = {}
    tool_rows: dict[str, int] = {}

    try:
        entity_trend_df = run_ner(
            df=df,
            model_name=ner_model_id,
            granularity=granularity,
            max_docs=debug_max_docs if mode == "debug" else ner_max_docs_full,
            seed=seed,
        )
    except Exception as exc:
        entity_trend_df = pd.DataFrame(
            columns=["entity", "time_bin", "count", "doc_freq", "top_cooccurring", "confidence_stats", "model_id"]
        )
        errors["entity_trend"] = str(exc)
    entity_path = output_dir / "entity_trend.parquet"
    entity_trend_df.to_parquet(entity_path, index=False)
    tool_rows["entity_trend"] = int(entity_trend_df.shape[0])

    try:
        sentiment_df, sentiment_device_used = run_sentiment(
            df=df,
            model_id=sentiment_model_id,
            granularity=granularity,
            preferred_device=sentiment_device,
            max_docs=debug_max_docs if mode == "debug" else sentiment_max_docs_full,
            seed=seed,
        )
    except Exception as exc:
        sentiment_df = pd.DataFrame(columns=["entity", "time_bin", "model_id", "mean", "std", "n_docs"])
        sentiment_device_used = "cpu"
        errors["sentiment_series"] = str(exc)
    sentiment_path = output_dir / "sentiment_series.parquet"
    sentiment_df.to_parquet(sentiment_path, index=False)
    tool_rows["sentiment_series"] = int(sentiment_df.shape[0])

    try:
        topics_df = run_topics_over_time(
            df=df,
            granularity=granularity,
            max_docs=debug_max_docs if mode == "debug" else topics_max_docs_full,
            seed=seed,
        )
    except Exception as exc:
        topics_df = pd.DataFrame(columns=["topic_id", "time_bin", "weight", "top_terms", "coherence_proxy"])
        errors["topics_over_time"] = str(exc)
    topics_path = output_dir / "topics_over_time.parquet"
    topics_df.to_parquet(topics_path, index=False)
    tool_rows["topics_over_time"] = int(topics_df.shape[0])

    try:
        burst_df = run_burst_detection(entity_trend_df=entity_trend_df)
    except Exception as exc:
        burst_df = pd.DataFrame(columns=["entity_or_term", "burst_level", "start", "end", "intensity"])
        errors["burst_events"] = str(exc)
    burst_path = output_dir / "burst_events.parquet"
    burst_df.to_parquet(burst_path, index=False)
    tool_rows["burst_events"] = int(burst_df.shape[0])

    try:
        keyphrases_df = run_keyphrases(
            df=df,
            granularity=granularity,
            max_docs_per_bin=keyphrases_max_docs_per_bin,
            seed=seed,
        )
    except Exception as exc:
        keyphrases_df = pd.DataFrame(columns=["phrase", "time_bin", "score", "doc_freq", "method"])
        errors["keyphrases"] = str(exc)
    keyphrase_path = output_dir / "keyphrases.parquet"
    keyphrases_df.to_parquet(keyphrase_path, index=False)
    tool_rows["keyphrases"] = int(keyphrases_df.shape[0])

    summary = {
        "mode": mode,
        "seed": seed,
        "time_granularity": granularity,
        "documents_processed": int(df.shape[0]),
        "sentiment_device": sentiment_device_used,
        "device_report": runtime_device_report(),
        "tool_rows": tool_rows,
        "errors": errors,
        "outputs": {
            "entity_trend": str(entity_path),
            "sentiment_series": str(sentiment_path),
            "topics_over_time": str(topics_path),
            "burst_events": str(burst_path),
            "keyphrases": str(keyphrase_path),
        },
    }
    write_json(summary_path, summary)
    return summary
