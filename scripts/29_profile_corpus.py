from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.analysis_tools import ENTITY_PATTERN, TOKEN_PATTERN  # noqa: E402
from corpusagent2.io_utils import write_json  # noqa: E402


STOPWORDS = set(ENGLISH_STOP_WORDS).union(
    {
        "said", "says", "news", "media", "coverage", "reuters", "update", "breaking",
        "http", "https", "www", "com",
    }
)

LANGUAGE_HINTS = {
    "en": {"the", "and", "with", "from", "that", "this", "have", "will"},
    "de": {"und", "der", "die", "das", "nicht", "mit", "ist", "ein"},
    "fr": {"le", "la", "les", "des", "une", "est", "avec", "dans"},
    "it": {"il", "lo", "gli", "che", "con", "per", "non"},
    "es": {"que", "los", "las", "con", "para", "una", "del", "por"},
}

DEFAULT_PROBES: dict[str, str] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the full corpus so query coverage and bias are visible.")
    parser.add_argument("--documents-path", type=Path, default=REPO_ROOT / "data" / "processed" / "documents.parquet")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "corpus_profile")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument(
        "--probe",
        action="append",
        default=[],
        help="Benchmark probe in the form name=regex. Can be provided multiple times.",
    )
    return parser.parse_args()


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text))]


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokenize(text) if token not in STOPWORDS]


def _normalize_title(title: str) -> str:
    tokens = _content_tokens(title)
    return " ".join(tokens[:20])


def _source_name(raw: str) -> str:
    source = str(raw or "").strip()
    return source or "<missing>"


def _estimate_language(text: str) -> str:
    lowered = f" {' '.join(_tokenize(text[:1500]))} "
    if not lowered.strip():
        return "unknown"
    counts = {
        lang: sum(1 for token in hints if f" {token} " in lowered)
        for lang, hints in LANGUAGE_HINTS.items()
    }
    best_lang, best_count = max(counts.items(), key=lambda item: item[1])
    return best_lang if best_count > 0 else "unknown"


def _as_year(value: str) -> str:
    text = str(value or "")
    if len(text) >= 4 and text[:4].isdigit():
        return text[:4]
    return "<missing>"


def _as_month(value: str) -> str:
    text = str(value or "")
    if len(text) >= 7 and text[4] == "-" and text[5:7].isdigit():
        return text[:7]
    return "<missing>"


def _looks_json_like(text: str) -> bool:
    head = str(text or "")[:1200]
    if not head.strip():
        return False
    structural_hits = sum(
        token in head
        for token in ('"url"', '"title"', '"content_type"', '"thumbnail_url"', '"published_at"', '"author_name"')
    )
    brace_count = head.count("{") + head.count("}")
    quote_colon_count = head.count('":"')
    return structural_hits >= 2 or brace_count >= 8 or quote_colon_count >= 6


def _counter_rows(counter: Counter[str], *, value_name: str, top_k: int) -> pd.DataFrame:
    rows = [{value_name: key, "count": int(value)} for key, value in counter.most_common(top_k)]
    return pd.DataFrame(rows)


def _quantile_summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    array = np.asarray(values, dtype=np.int32)
    return {
        "mean": round(float(array.mean()), 3),
        "median": round(float(np.median(array)), 3),
        "p90": round(float(np.percentile(array, 90)), 3),
        "p95": round(float(np.percentile(array, 95)), 3),
        "p99": round(float(np.percentile(array, 99)), 3),
        "max": float(array.max()),
    }


def _compile_probes(extra_probes: list[str]) -> dict[str, Any]:
    probes = dict(DEFAULT_PROBES)
    for raw in extra_probes:
        name, _, pattern = str(raw).partition("=")
        if not name.strip() or not pattern.strip():
            raise ValueError(f"Invalid --probe value: {raw!r}. Expected name=regex.")
        probes[name.strip()] = pattern.strip()
    return {name: re.compile(pattern, flags=re.IGNORECASE) for name, pattern in probes.items()}


def _write_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _top_source_rows(source_counter: Counter[str], total_docs: int, top_k: int) -> pd.DataFrame:
    rows = []
    for source, count in source_counter.most_common(top_k):
        rows.append(
            {
                "source": source,
                "count": int(count),
                "share_of_corpus": round(count / max(total_docs, 1), 6),
            }
        )
    return pd.DataFrame(rows)


def _markdown_report(
    output_path: Path,
    *,
    summary: dict[str, Any],
    top_sources: pd.DataFrame,
    top_tokens: pd.DataFrame,
    top_bigrams: pd.DataFrame,
    probe_rows: pd.DataFrame,
    duplicate_titles: pd.DataFrame,
) -> None:
    dominant_source = "n/a"
    if not top_sources.empty:
        row = top_sources.iloc[0]
        dominant_source = f"{row['source']} ({int(row['count'])} docs, share {float(row['share_of_corpus']):.3f})"
    lines = [
        "# Corpus Profile",
        "",
        f"- Documents: `{summary['document_count']}`",
        f"- Unique sources: `{summary['unique_sources']}`",
        f"- Date range: `{summary['date_min']}` to `{summary['date_max']}`",
        f"- Dominant source: `{dominant_source}`",
        f"- Duplicate-normalized titles (>1 hit): `{summary['duplicate_title_clusters']}`",
        f"- Estimated English share: `{summary['language_estimate_counts'].get('en', 0)}` docs",
        f"- JSON-like text payloads: `{summary['json_like_text_count']}` docs",
        "",
        "## Coverage Checks",
        "",
    ]
    if probe_rows.empty:
        lines.append("- No coverage probes configured. Pass `--probe name=regex` to measure topic coverage.")
    else:
        lines.extend(
            f"- `{row.probe}`: {int(row.doc_count)} docs"
            for row in probe_rows.sort_values("doc_count", ascending=False).itertuples(index=False)
        )
    lines.extend(["", "## Lexical Shape", ""])
    if not top_tokens.empty:
        lines.append("- Top content tokens: " + ", ".join(top_tokens["token"].head(20).tolist()))
    if not top_bigrams.empty:
        lines.append("- Top content bigrams: " + ", ".join(top_bigrams["bigram"].head(20).tolist()))
    lines.extend(["", "## Corpus Risks", ""])
    lines.append("- High source concentration or duplicate-title concentration means some question families may look richer than they really are.")
    lines.append("- Probe counts show whether a topic is present at all; they do not guarantee the topic is well-covered across years or outlets.")
    if not duplicate_titles.empty:
        lines.append(f"- Most repeated normalized title: `{duplicate_titles.iloc[0]['sample_title']}` ({int(duplicate_titles.iloc[0]['count'])} variants/docs).")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probes = _compile_probes(args.probe)

    parquet = pq.ParquetFile(args.documents_path)
    columns = ["doc_id", "title", "text", "published_at", "source"]

    total_docs = 0
    unique_doc_ids: set[str] = set()
    missing_title_count = 0
    missing_text_count = 0
    missing_source_count = 0
    missing_date_count = 0
    json_like_text_count = 0
    min_date = ""
    max_date = ""

    source_counter: Counter[str] = Counter()
    year_counter: Counter[str] = Counter()
    month_counter: Counter[str] = Counter()
    language_counter: Counter[str] = Counter()
    token_counter: Counter[str] = Counter()
    title_token_counter: Counter[str] = Counter()
    bigram_counter: Counter[str] = Counter()
    entity_counter: Counter[str] = Counter()
    duplicate_title_counter: Counter[str] = Counter()
    duplicate_title_examples: dict[str, str] = {}
    doc_lengths: list[int] = []
    title_lengths: list[int] = []

    probe_source_counter: dict[str, Counter[str]] = {name: Counter() for name in probes}
    probe_year_counter: defaultdict[tuple[str, str], int] = defaultdict(int)
    probe_stats: dict[str, dict[str, Any]] = {
        name: {
            "doc_count": 0,
            "title_hit_count": 0,
            "first_date": "",
            "last_date": "",
            "examples": [],
        }
        for name in probes
    }

    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
        frame = batch.to_pandas()
        frame["doc_id"] = frame["doc_id"].fillna("").astype(str)
        frame["title"] = frame["title"].fillna("").astype(str)
        frame["text"] = frame["text"].fillna("").astype(str)
        frame["published_at"] = frame["published_at"].fillna("").astype(str)
        frame["source"] = frame["source"].fillna("").astype(str)

        for row in frame.itertuples(index=False):
            total_docs += 1
            doc_id = str(row.doc_id)
            title = str(row.title)
            text = str(row.text)
            published_at = str(row.published_at)
            source = _source_name(row.source)
            combined = f"{title}\n{text}"

            unique_doc_ids.add(doc_id)
            missing_title_count += int(not title.strip())
            missing_text_count += int(not text.strip())
            missing_source_count += int(source == "<missing>")
            missing_date_count += int(not published_at.strip())
            json_like_text_count += int(_looks_json_like(text))
            if published_at:
                min_date = published_at if not min_date or published_at < min_date else min_date
                max_date = published_at if not max_date or published_at > max_date else max_date

            source_counter[source] += 1
            year = _as_year(published_at)
            month = _as_month(published_at)
            year_counter[year] += 1
            month_counter[month] += 1

            doc_tokens = _content_tokens(text)
            title_tokens = _content_tokens(title)
            doc_lengths.append(len(doc_tokens))
            title_lengths.append(len(title_tokens))
            token_counter.update(doc_tokens)
            title_token_counter.update(title_tokens)
            bigram_counter.update(
                f"{left} {right}"
                for left, right in zip(doc_tokens, doc_tokens[1:], strict=False)
            )
            entity_counter.update(match.group(0).strip() for match in ENTITY_PATTERN.finditer(f"{title}\n{text[:800]}"))

            language_counter[_estimate_language(f"{title}\n{text[:1200]}")] += 1

            normalized_title = _normalize_title(title)
            if normalized_title:
                duplicate_title_counter[normalized_title] += 1
                duplicate_title_examples.setdefault(normalized_title, title.strip())

            for probe_name, pattern in probes.items():
                if not pattern.search(combined):
                    continue
                probe_stats[probe_name]["doc_count"] += 1
                if pattern.search(title):
                    probe_stats[probe_name]["title_hit_count"] += 1
                probe_source_counter[probe_name][source] += 1
                probe_year_counter[(probe_name, year)] += 1
                current_first = str(probe_stats[probe_name]["first_date"])
                current_last = str(probe_stats[probe_name]["last_date"])
                if published_at and (not current_first or published_at < current_first):
                    probe_stats[probe_name]["first_date"] = published_at
                if published_at and (not current_last or published_at > current_last):
                    probe_stats[probe_name]["last_date"] = published_at
                examples = probe_stats[probe_name]["examples"]
                if len(examples) < 5:
                    examples.append(
                        {
                            "doc_id": doc_id,
                            "title": title[:220],
                            "source": source,
                            "published_at": published_at,
                        }
                    )

    top_sources = _top_source_rows(source_counter, total_docs, args.top_k)
    yearly_counts = _counter_rows(year_counter, value_name="year", top_k=max(len(year_counter), 1))
    monthly_counts = _counter_rows(month_counter, value_name="month", top_k=max(len(month_counter), 1))
    top_tokens = _counter_rows(token_counter, value_name="token", top_k=args.top_k)
    top_title_tokens = _counter_rows(title_token_counter, value_name="token", top_k=args.top_k)
    top_bigrams = _counter_rows(bigram_counter, value_name="bigram", top_k=args.top_k)
    top_entities = _counter_rows(entity_counter, value_name="entity", top_k=args.top_k)
    language_estimates = _counter_rows(language_counter, value_name="language", top_k=max(len(language_counter), 1))

    duplicate_title_rows = [
        {
            "normalized_title": key,
            "count": int(count),
            "sample_title": duplicate_title_examples.get(key, ""),
        }
        for key, count in duplicate_title_counter.items()
        if count > 1
    ]
    duplicate_titles = pd.DataFrame(duplicate_title_rows)
    if not duplicate_titles.empty:
        duplicate_titles = duplicate_titles.sort_values(["count", "sample_title"], ascending=[False, True]).reset_index(drop=True)
    else:
        duplicate_titles = pd.DataFrame(columns=["normalized_title", "count", "sample_title"])

    probe_rows = []
    probe_examples = {}
    for probe_name, stats in probe_stats.items():
        sources = probe_source_counter[probe_name].most_common(5)
        probe_rows.append(
            {
                "probe": probe_name,
                "doc_count": int(stats["doc_count"]),
                "share_of_corpus": round(int(stats["doc_count"]) / max(total_docs, 1), 6),
                "title_hit_count": int(stats["title_hit_count"]),
                "first_date": str(stats["first_date"] or ""),
                "last_date": str(stats["last_date"] or ""),
                "top_sources": ", ".join(f"{source}:{count}" for source, count in sources),
            }
        )
        probe_examples[probe_name] = list(stats["examples"])
    probe_hits = pd.DataFrame(probe_rows).sort_values(["doc_count", "probe"], ascending=[False, True])
    probe_yearly = pd.DataFrame(
        [
            {"probe": probe, "year": year, "count": int(count)}
            for (probe, year), count in sorted(probe_year_counter.items())
        ]
    )

    source_counts = list(source_counter.values())
    top_10_source_share = round(sum(count for _, count in source_counter.most_common(10)) / max(total_docs, 1), 6)
    source_hhi = round(sum((count / max(total_docs, 1)) ** 2 for count in source_counts), 6)
    summary = {
        "document_count": int(total_docs),
        "unique_doc_ids": int(len(unique_doc_ids)),
        "duplicate_doc_ids": int(total_docs - len(unique_doc_ids)),
        "unique_sources": int(len(source_counter)),
        "missing_title_count": int(missing_title_count),
        "missing_text_count": int(missing_text_count),
        "missing_source_count": int(missing_source_count),
        "missing_date_count": int(missing_date_count),
        "json_like_text_count": int(json_like_text_count),
        "date_min": min_date,
        "date_max": max_date,
        "content_token_total": int(sum(token_counter.values())),
        "unique_content_tokens": int(len(token_counter)),
        "doc_length_tokens": _quantile_summary(doc_lengths),
        "title_length_tokens": _quantile_summary(title_lengths),
        "top_10_source_share": top_10_source_share,
        "source_hhi": source_hhi,
        "duplicate_title_clusters": int((duplicate_titles["count"] > 1).sum()) if not duplicate_titles.empty else 0,
        "language_estimate_counts": {str(row["language"]): int(row["count"]) for row in language_estimates.to_dict(orient="records")},
    }

    write_json(args.output_dir / "summary.json", summary)
    _write_dataframe(top_sources, args.output_dir / "top_sources.csv")
    _write_dataframe(yearly_counts, args.output_dir / "yearly_counts.csv")
    _write_dataframe(monthly_counts, args.output_dir / "monthly_counts.csv")
    _write_dataframe(top_tokens, args.output_dir / "top_tokens.csv")
    _write_dataframe(top_title_tokens, args.output_dir / "top_title_tokens.csv")
    _write_dataframe(top_bigrams, args.output_dir / "top_bigrams.csv")
    _write_dataframe(top_entities, args.output_dir / "top_named_spans.csv")
    _write_dataframe(language_estimates, args.output_dir / "language_estimates.csv")
    _write_dataframe(duplicate_titles.head(args.top_k), args.output_dir / "duplicate_titles.csv")
    _write_dataframe(probe_hits, args.output_dir / "benchmark_topic_hits.csv")
    _write_dataframe(probe_yearly, args.output_dir / "benchmark_topic_yearly.csv")
    (args.output_dir / "benchmark_topic_examples.json").write_text(
        json.dumps(probe_examples, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _markdown_report(
        args.output_dir / "report.md",
        summary=summary,
        top_sources=top_sources,
        top_tokens=top_tokens,
        top_bigrams=top_bigrams,
        probe_rows=probe_hits,
        duplicate_titles=duplicate_titles,
    )
    print(f"Wrote corpus profile to {args.output_dir}")


if __name__ == "__main__":
    main()
