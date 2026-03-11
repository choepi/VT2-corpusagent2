from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.io_utils import ensure_absolute, ensure_exists, read_documents, write_json
from corpusagent2.seed import set_global_seed


TOKEN_PATTERN = re.compile(r"[A-Za-z]{3,}")
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")


def extract_time_bin(value: str) -> str:
    value = str(value).strip()
    if not value:
        return "unknown"
    if len(value) >= 4 and value[:4].isdigit():
        return value[:4]
    return "unknown"


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


def run_ner(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    counts: Counter = Counter()
    doc_freq: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    cooccurrence: Counter = Counter()

    try:
        import spacy
    except Exception:
        spacy = None

    nlp = None
    model_used = "regex_fallback"
    if spacy is not None:
        for candidate in [model_name, "en_core_web_sm"]:
            try:
                nlp = spacy.load(candidate)
                model_used = candidate
                break
            except OSError:
                continue

    if nlp is not None:
        docs = nlp.pipe(df["text"].astype(str).tolist(), batch_size=32)
        row_doc_iter = zip(df.itertuples(index=False), docs, strict=False)
        for row, doc in row_doc_iter:
            time_bin = extract_time_bin(row.published_at)
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
            time_bin = extract_time_bin(row.published_at)
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


def run_sentiment(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    from transformers import pipeline

    pipe = pipeline("text-classification", model=model_id, tokenizer=model_id)

    rows = []
    for row in df.itertuples(index=False):
        text = str(row.text)[:1200]
        pred = pipe(text, truncation=True)[0]
        label = str(pred["label"]).lower()

        if "neg" in label:
            score = -1.0
        elif "pos" in label:
            score = 1.0
        else:
            score = 0.0

        rows.append(
            {
                "entity": "__all__",
                "time_bin": extract_time_bin(row.published_at),
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
    return grouped


def run_topics_over_time(df: pd.DataFrame) -> pd.DataFrame:
    from bertopic import BERTopic

    texts = df["text"].astype(str).tolist()
    topic_model = BERTopic(calculate_probabilities=False, verbose=False)
    topics, _ = topic_model.fit_transform(texts)

    topic_df = pd.DataFrame(
        {
            "topic_id": topics,
            "time_bin": df["published_at"].map(extract_time_bin),
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


def run_keyphrases(df: pd.DataFrame, top_k: int = 25) -> pd.DataFrame:
    rows = []
    for time_bin, subset in df.groupby(df["published_at"].map(extract_time_bin)):
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


if __name__ == "__main__":
    MODE = "debug"  # "debug" or "full"
    SEED = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DOCUMENTS_PATH = (PROJECT_ROOT / "data" / "processed" / "documents.parquet").resolve()
    OUTPUT_DIR = (PROJECT_ROOT / "outputs" / "nlp_tools").resolve()
    SUMMARY_PATH = (OUTPUT_DIR / "summary.json").resolve()

    DEBUG_MAX_DOCS = 5000
    NER_MODEL_ID = "en_core_web_trf"
    SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    ensure_absolute(DOCUMENTS_PATH, "DOCUMENTS_PATH")
    ensure_absolute(OUTPUT_DIR, "OUTPUT_DIR")
    ensure_exists(DOCUMENTS_PATH, "DOCUMENTS_PATH")

    set_global_seed(SEED)

    df = read_documents(DOCUMENTS_PATH)
    if MODE == "debug":
        df = df.head(DEBUG_MAX_DOCS).copy()

    if df.empty:
        raise RuntimeError("Input documents are empty")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entity_trend_df = run_ner(df=df, model_name=NER_MODEL_ID)
    entity_path = OUTPUT_DIR / "entity_trend.parquet"
    entity_trend_df.to_parquet(entity_path, index=False)

    sentiment_df = run_sentiment(df=df, model_id=SENTIMENT_MODEL_ID)
    sentiment_path = OUTPUT_DIR / "sentiment_series.parquet"
    sentiment_df.to_parquet(sentiment_path, index=False)

    topics_df = run_topics_over_time(df=df)
    topics_path = OUTPUT_DIR / "topics_over_time.parquet"
    topics_df.to_parquet(topics_path, index=False)

    burst_df = run_burst_detection(entity_trend_df=entity_trend_df)
    burst_path = OUTPUT_DIR / "burst_events.parquet"
    burst_df.to_parquet(burst_path, index=False)

    keyphrases_df = run_keyphrases(df=df)
    keyphrase_path = OUTPUT_DIR / "keyphrases.parquet"
    keyphrases_df.to_parquet(keyphrase_path, index=False)

    summary = {
        "mode": MODE,
        "seed": SEED,
        "documents_processed": int(df.shape[0]),
        "outputs": {
            "entity_trend": str(entity_path),
            "sentiment_series": str(sentiment_path),
            "topics_over_time": str(topics_path),
            "burst_events": str(burst_path),
            "keyphrases": str(keyphrase_path),
        },
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Wrote NLP outputs to: {OUTPUT_DIR}")
    print(f"Summary: {SUMMARY_PATH}")
