from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.analysis_tools import ENTITY_PATTERN, TOKEN_PATTERN  # noqa: E402
from corpusagent2.io_utils import write_json  # noqa: E402


POSITIVE_WORDS = {
    "good", "strong", "gain", "improve", "success", "positive", "optimistic", "support", "confidence",
    "win", "benefit", "stable", "growth", "calm", "approval",
}
NEGATIVE_WORDS = {
    "bad", "weak", "loss", "drop", "risk", "fear", "negative", "warn", "crisis", "attack", "scandal",
    "controversy", "conflict", "chaos", "decline", "critic", "accuse",
}
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "have", "were", "been", "will", "would", "about",
    "their", "they", "them", "said", "says", "into", "than", "then", "after", "before", "over", "under",
    "more", "most", "some", "such", "much", "many", "also", "could", "should", "news", "media", "coverage",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an in-depth corpus analysis for a query slice.")
    parser.add_argument("--documents-path", type=Path, default=REPO_ROOT / "data" / "processed" / "documents.parquet")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "corpus_deep_analysis" / "query_slice")
    parser.add_argument("--query", required=True, help="Regex used to select relevant documents.")
    parser.add_argument("--date-from", default="", help="Inclusive lower date bound, e.g. 2017-01-01")
    parser.add_argument("--date-to", default="", help="Inclusive upper date bound, e.g. 2019-12-31")
    parser.add_argument("--sample-size", type=int, default=12000, help="Maximum documents used for topic modeling.")
    parser.add_argument("--max-topics", type=int, default=6)
    return parser.parse_args()


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text))]


def _sentiment_score(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    pos = sum(1 for token in tokens if token in POSITIVE_WORDS)
    neg = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    return (pos - neg) / math.sqrt(len(tokens))


def _entity_candidates(text: str) -> list[str]:
    return [match.group(0).strip() for match in ENTITY_PATTERN.finditer(str(text))]


def _load_matching_documents(path: Path, query_pattern: re.Pattern[str], date_from: str, date_to: str) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    rows: list[pd.DataFrame] = []
    for batch in parquet.iter_batches(batch_size=4096, columns=["doc_id", "title", "text", "published_at", "source"]):
        frame = batch.to_pandas()
        combined = (frame["title"].fillna("") + "\n" + frame["text"].fillna("")).astype(str)
        mask = combined.str.contains(query_pattern, na=False)
        if date_from:
            mask &= frame["published_at"].astype(str) >= date_from
        if date_to:
            mask &= frame["published_at"].astype(str) <= date_to
        if bool(mask.any()):
            rows.append(frame.loc[mask].copy())
    if not rows:
        return pd.DataFrame(columns=["doc_id", "title", "text", "published_at", "source"])
    df = pd.concat(rows, ignore_index=True)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=False)
    df = df.dropna(subset=["published_at"]).copy()
    df["month"] = df["published_at"].dt.to_period("M").astype(str)
    df["source"] = df["source"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    return df.sort_values("published_at").reset_index(drop=True)


def _series_rows(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("month", as_index=False)
        .agg(
            doc_count=("doc_id", "size"),
            mean_sentiment=("sentiment_score", "mean"),
            median_sentiment=("sentiment_score", "median"),
            mean_doc_length=("token_count", "mean"),
            mean_title_length=("title_token_count", "mean"),
        )
    )
    return summary.sort_values("month").reset_index(drop=True)


def _source_rows(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("source", as_index=False)
        .agg(
            doc_count=("doc_id", "size"),
            mean_sentiment=("sentiment_score", "mean"),
            mean_doc_length=("token_count", "mean"),
        )
        .sort_values(["doc_count", "mean_sentiment"], ascending=[False, False])
    )
    return summary.reset_index(drop=True)


def _term_frequency_rows(texts: list[str], ngram_range: tuple[int, int], top_k: int) -> pd.DataFrame:
    vectorizer = CountVectorizer(stop_words="english", min_df=3, max_features=20000, ngram_range=ngram_range)
    matrix = vectorizer.fit_transform(texts)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    vocab = vectorizer.get_feature_names_out()
    order = np.argsort(counts)[::-1][:top_k]
    return pd.DataFrame({"term": vocab[order], "count": counts[order].astype(int)})


def _distinctive_terms(left_texts: list[str], right_texts: list[str], top_k: int = 30) -> pd.DataFrame:
    vectorizer = CountVectorizer(stop_words="english", min_df=3, max_features=25000)
    matrix = vectorizer.fit_transform(left_texts + right_texts)
    vocab = vectorizer.get_feature_names_out()
    left_counts = np.asarray(matrix[: len(left_texts)].sum(axis=0)).ravel() + 0.5
    right_counts = np.asarray(matrix[len(left_texts) :].sum(axis=0)).ravel() + 0.5
    left_total = float(left_counts.sum())
    right_total = float(right_counts.sum())
    score = np.log(left_counts / left_total) - np.log(right_counts / right_total)
    order = np.argsort(np.abs(score))[::-1][:top_k]
    return pd.DataFrame(
        {
            "term": vocab[order],
            "log_odds": score[order],
            "left_count": left_counts[order].astype(int),
            "right_count": right_counts[order].astype(int),
            "direction": np.where(score[order] > 0, "left", "right"),
        }
    )


def _topic_rows(texts: list[str], months: list[str], max_topics: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    vectorizer = TfidfVectorizer(stop_words="english", min_df=5, max_features=12000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[0] < 10 or matrix.shape[1] < 20:
        return pd.DataFrame(columns=["topic_id", "term", "weight"]), pd.DataFrame(columns=["month", "topic_id", "doc_share"])
    n_topics = max(2, min(max_topics, matrix.shape[0] // 20, 8))
    model = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=400)
    doc_topics = model.fit_transform(matrix)
    vocab = vectorizer.get_feature_names_out()

    topic_rows: list[dict[str, object]] = []
    for topic_id, component in enumerate(model.components_, start=1):
        top_indices = component.argsort()[::-1][:12]
        for idx in top_indices:
            topic_rows.append({"topic_id": topic_id, "term": str(vocab[idx]), "weight": float(component[idx])})

    dominant = doc_topics.argmax(axis=1) + 1
    month_topic = pd.DataFrame({"month": months, "topic_id": dominant})
    share_rows = (
        month_topic.groupby(["month", "topic_id"]).size().rename("doc_count").reset_index()
        .merge(month_topic.groupby("month").size().rename("month_total").reset_index(), on="month", how="left")
    )
    share_rows["doc_share"] = share_rows["doc_count"] / share_rows["month_total"].clip(lower=1)
    return pd.DataFrame(topic_rows), share_rows[["month", "topic_id", "doc_share"]].sort_values(["month", "topic_id"])


def _entity_rows(df: pd.DataFrame) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for text in df["text"].tolist():
        counts.update(_entity_candidates(text))
    rows = [{"entity": entity, "count": int(count)} for entity, count in counts.most_common(50) if entity]
    return pd.DataFrame(rows)


def _write_markdown_report(
    output_path: Path,
    *,
    query: str,
    total_docs: int,
    total_sources: int,
    monthly: pd.DataFrame,
    sources: pd.DataFrame,
    terms: pd.DataFrame,
    bigrams: pd.DataFrame,
    distinct_terms: pd.DataFrame,
    entities: pd.DataFrame,
) -> None:
    peak_row = monthly.sort_values("doc_count", ascending=False).head(1)
    peak_text = "n/a"
    if not peak_row.empty:
        peak = peak_row.iloc[0]
        peak_text = f"{peak['month']} ({int(peak['doc_count'])} docs)"
    lines = [
        "# Deep Corpus Analysis",
        "",
        f"- Query regex: `{query}`",
        f"- Matched documents: `{total_docs}`",
        f"- Unique sources: `{total_sources}`",
        f"- Peak month: `{peak_text}`",
        "",
        "## Monthly Pattern",
        "",
    ]
    if not monthly.empty:
        lines.extend(
            f"- `{row.month}`: {int(row.doc_count)} docs, mean sentiment {row.mean_sentiment:.3f}"
            for row in monthly.head(12).itertuples(index=False)
        )
    lines.extend(["", "## Top Sources", ""])
    if not sources.empty:
        lines.extend(
            f"- `{row.source}`: {int(row.doc_count)} docs, mean sentiment {row.mean_sentiment:.3f}"
            for row in sources.head(10).itertuples(index=False)
        )
    lines.extend(["", "## Dominant Terms", ""])
    if not terms.empty:
        lines.append("- Unigrams: " + ", ".join(terms["term"].head(15).tolist()))
    if not bigrams.empty:
        lines.append("- Bigrams: " + ", ".join(bigrams["term"].head(15).tolist()))
    lines.extend(["", "## Distinctive Temporal Terms", ""])
    if not distinct_terms.empty:
        lines.extend(
            f"- `{row.term}` -> {row.direction} window (log-odds {row.log_odds:.3f})"
            for row in distinct_terms.head(15).itertuples(index=False)
        )
    lines.extend(["", "## Frequent Entities", ""])
    if not entities.empty:
        lines.extend(f"- `{row.entity}`: {int(row.count)}" for row in entities.head(15).itertuples(index=False))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(args.query, flags=re.IGNORECASE)
    df = _load_matching_documents(args.documents_path, pattern, args.date_from, args.date_to)
    if df.empty:
        raise SystemExit(f"No documents matched query regex: {args.query}")

    df["token_count"] = df["text"].map(lambda value: len(_tokenize(value)))
    df["title_token_count"] = df["title"].map(lambda value: len(_tokenize(value)))
    df["sentiment_score"] = df["text"].map(_sentiment_score)

    monthly = _series_rows(df)
    sources = _source_rows(df)
    top_unigrams = _term_frequency_rows(df["text"].tolist(), (1, 1), top_k=50)
    top_bigrams = _term_frequency_rows(df["text"].tolist(), (2, 2), top_k=50)
    entities = _entity_rows(df)

    midpoint = max(len(df) // 2, 1)
    temporal_terms = _distinctive_terms(df["text"].iloc[:midpoint].tolist(), df["text"].iloc[midpoint:].tolist())

    sampled = df.sample(n=min(args.sample_size, len(df)), random_state=42).sort_values("published_at")
    topic_terms, topic_shares = _topic_rows(sampled["text"].tolist(), sampled["month"].tolist(), args.max_topics)

    summary = {
        "query": args.query,
        "documents_path": str(args.documents_path),
        "matched_documents": int(df.shape[0]),
        "unique_sources": int(df["source"].nunique()),
        "date_min": str(df["published_at"].min()),
        "date_max": str(df["published_at"].max()),
        "mean_sentiment": float(df["sentiment_score"].mean()),
        "median_sentiment": float(df["sentiment_score"].median()),
        "mean_doc_length": float(df["token_count"].mean()),
        "top_sources": sources.head(10).to_dict(orient="records"),
    }

    monthly.to_csv(args.output_dir / "monthly_series.csv", index=False)
    sources.to_csv(args.output_dir / "source_summary.csv", index=False)
    top_unigrams.to_csv(args.output_dir / "top_unigrams.csv", index=False)
    top_bigrams.to_csv(args.output_dir / "top_bigrams.csv", index=False)
    temporal_terms.to_csv(args.output_dir / "distinctive_terms_early_vs_late.csv", index=False)
    entities.to_csv(args.output_dir / "top_entities.csv", index=False)
    topic_terms.to_csv(args.output_dir / "topic_terms.csv", index=False)
    topic_shares.to_csv(args.output_dir / "topic_shares_by_month.csv", index=False)
    write_json(args.output_dir / "summary.json", summary)
    _write_markdown_report(
        args.output_dir / "report.md",
        query=args.query,
        total_docs=int(df.shape[0]),
        total_sources=int(df["source"].nunique()),
        monthly=monthly,
        sources=sources,
        terms=top_unigrams,
        bigrams=top_bigrams,
        distinct_terms=temporal_terms,
        entities=entities,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "matched_documents": int(df.shape[0])}, ensure_ascii=True))


if __name__ == "__main__":
    main()
