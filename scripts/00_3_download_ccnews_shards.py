"""Robust shard downloader for Geralt-Targaryen/CC-News (replaces streaming for
full-shard sampling, which hits HF CDN read timeouts).

Downloads each chosen parquet shard via hf_hub_download (resumable), converts it
to normalized JSONL.gz that 01_prepare_dataset.py reads, then deletes the parquet
to save disk. Resumable: shards whose output already exists are skipped.

Schema normalization (this dataset is quirky): title->title, text->text,
source_domain->source (outlet), source(=YEAR)->published_at as "<year>-01-01".

Config via env:
    CORPUSAGENT2_SAMPLE_SHARDS          comma list of shard numbers (required)
    CORPUSAGENT2_SAMPLE_ROWS_PER_SHARD  cap per shard (default: whole shard)
    CORPUSAGENT2_SAMPLE_OUT_DIR         output dir (default data/raw/incoming)
"""
from __future__ import annotations

import gzip
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[1]
REPO_ID = "Geralt-Targaryen/CC-News"
N_SHARDS = 143
COLS = ["title", "text", "source_domain", "source", "authors"]


def _norm(rec: dict) -> dict | None:
    text = str(rec.get("text") or "").strip()
    if not text:
        return None
    year = str(rec.get("source") or "").strip()
    return {
        "title": str(rec.get("title") or "").strip(),
        "text": text,
        "source": str(rec.get("source_domain") or "").strip(),
        "published_at": f"{year}-01-01" if year[:4].isdigit() else "",
        "authors": rec.get("authors") or [],
        "year": year,
    }


def main() -> None:
    shards_env = os.getenv("CORPUSAGENT2_SAMPLE_SHARDS", "").strip()
    if not shards_env:
        raise SystemExit("Set CORPUSAGENT2_SAMPLE_SHARDS=comma,list,of,shard,numbers")
    shards = [int(x) for x in shards_env.split(",") if x.strip()]
    cap = int(os.getenv("CORPUSAGENT2_SAMPLE_ROWS_PER_SHARD", "0") or "0")  # 0 = whole shard
    out_dir = Path(os.getenv("CORPUSAGENT2_SAMPLE_OUT_DIR", str(REPO / "data" / "raw" / "incoming"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    years: Counter = Counter()
    for n in shards:
        out_path = out_dir / f"ccnews_geralt_shard{n:05d}.jsonl.gz"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[shard {n}] already present, skipping", flush=True)
            continue
        name = f"ccnews-{n:05d}-of-{N_SHARDS:05d}.parquet"
        print(f"[shard {n}] downloading {name} ...", flush=True)
        local = hf_hub_download(repo_id=REPO_ID, filename=name, repo_type="dataset")
        written = 0
        tmp_path = out_path.with_suffix(".jsonl.gz.part")
        with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=20000, columns=COLS):
                for rec in batch.to_pylist():
                    norm = _norm(rec)
                    if norm is None:
                        continue
                    years[norm["year"]] += 1
                    fh.write(json.dumps(norm, ensure_ascii=False) + "\n")
                    written += 1
                    grand_total += 1
                    if cap and written >= cap:
                        break
                if cap and written >= cap:
                    break
        tmp_path.rename(out_path)
        try:
            os.remove(local)
        except OSError:
            pass
        print(f"[shard {n}] wrote {written} rows", flush=True)
    print(f"DONE total_new={grand_total} year_distribution={dict(sorted(years.items()))}", flush=True)


if __name__ == "__main__":
    main()
