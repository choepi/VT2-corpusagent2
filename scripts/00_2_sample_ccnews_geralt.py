"""Stratified sampler for Geralt-Targaryen/CC-News (the paper's dataset).

The dataset has 143 YEAR-SORTED parquet shards (~937k rows each, ~134M total,
2016-2021). Its schema is quirky:
    title, text, source_domain (the OUTLET), source (the YEAR), authors
This script streams a chosen set of shards (spread across years), takes a capped
number of rows from each, and writes normalized JSONL.gz that the existing
pipeline (01_prepare_dataset.py) reads directly:
    title -> title, text -> text, source_domain -> source, source(year) -> published_at

published_at is written as "<year>-01-01" so downstream date parsing works; the
real granularity is YEAR (the source has no month/day).

Config via env:
    CORPUSAGENT2_SAMPLE_SHARDS         comma list of shard numbers (default spread)
    CORPUSAGENT2_SAMPLE_ROWS_PER_SHARD rows to take per shard (default 250000)
    CORPUSAGENT2_SAMPLE_OUT_DIR        output dir (default data/raw/incoming)
"""
from __future__ import annotations

import gzip
import json
import os
from collections import Counter
from pathlib import Path

from datasets import load_dataset

REPO = Path(__file__).resolve().parents[1]
N_SHARDS = 143


def _norm(rec: dict) -> dict | None:
    title = str(rec.get("title") or "").strip()
    text = str(rec.get("text") or "").strip()
    if not text:
        return None
    domain = str(rec.get("source_domain") or "").strip()
    year = str(rec.get("source") or "").strip()
    published_at = f"{year}-01-01" if year[:4].isdigit() else ""
    return {
        "title": title,
        "text": text,
        "source": domain,        # 01_prepare maps "source" -> source column
        "published_at": published_at,
        "authors": rec.get("authors") or [],
        "year": year,
    }


def main() -> None:
    shards_env = os.getenv("CORPUSAGENT2_SAMPLE_SHARDS", "").strip()
    if shards_env:
        shards = [int(x) for x in shards_env.split(",") if x.strip()]
    else:
        # Spread across the year-sorted shard range to capture 2016-2021.
        shards = [8, 25, 45, 60, 70, 90, 120, 140]
    rows_per_shard = int(os.getenv("CORPUSAGENT2_SAMPLE_ROWS_PER_SHARD", "250000"))
    out_dir = Path(os.getenv("CORPUSAGENT2_SAMPLE_OUT_DIR", str(REPO / "data" / "raw" / "incoming"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    year_counts: Counter = Counter()
    for n in shards:
        name = f"ccnews-{n:05d}-of-{N_SHARDS:05d}.parquet"
        out_path = out_dir / f"ccnews_geralt_shard{n:05d}.jsonl.gz"
        print(f"[shard {n}] streaming {name} -> {out_path.name}", flush=True)
        ds = load_dataset("Geralt-Targaryen/CC-News", split="train", data_files=name, streaming=True)
        written = 0
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            for rec in ds:
                norm = _norm(rec)
                if norm is None:
                    continue
                year_counts[norm["year"]] += 1
                fh.write(json.dumps(norm, ensure_ascii=False) + "\n")
                written += 1
                total += 1
                if written >= rows_per_shard:
                    break
        print(f"[shard {n}] wrote {written} rows", flush=True)
    print(f"DONE total={total} year_distribution={dict(sorted(year_counts.items()))}", flush=True)


if __name__ == "__main__":
    main()
