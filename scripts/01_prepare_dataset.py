from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.io_utils import ensure_absolute, ensure_exists, write_json
from corpusagent2.seed import resolve_run_mode, set_global_seed


def extract_value(payload: dict, keys: list[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_doc_id(title: str, text: str, published_at: str, fallback_id: str = "") -> str:
    if fallback_id:
        return fallback_id
    digest = hashlib.sha256(f"{title}|{text}|{published_at}".encode("utf-8")).hexdigest()
    return digest


def iter_records(file_path: Path):
    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def normalize_row(payload: dict) -> dict:
    title = extract_value(payload, ["title", "headline"])
    text = extract_value(payload, ["text", "body", "content"])
    published_at = extract_value(payload, ["published_at", "date", "datetime", "timestamp", "year"])
    source = extract_value(payload, ["source", "domain", "publisher", "url"])
    raw_id = extract_value(payload, ["id", "doc_id", "_id"])

    doc_id = build_doc_id(title=title, text=text, published_at=published_at, fallback_id=raw_id)

    return {
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "published_at": published_at,
        "source": source,
    }


if __name__ == "__main__":
    MODE = resolve_run_mode("full")
    SEED = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_ROOT = (PROJECT_ROOT / "data" / "raw" / "ccnews_staged").resolve()
    PROCESSED_PATH = (PROJECT_ROOT / "data" / "processed" / "documents.parquet").resolve()
    SUMMARY_PATH = (PROJECT_ROOT / "outputs" / "prepare_dataset_summary.json").resolve()

    DEBUG_MAX_DOCS = 50_000

    ensure_absolute(RAW_ROOT, "RAW_ROOT")
    ensure_absolute(PROCESSED_PATH, "PROCESSED_PATH")
    ensure_absolute(SUMMARY_PATH, "SUMMARY_PATH")
    ensure_exists(RAW_ROOT, "RAW_ROOT")

    set_global_seed(SEED)

    raw_files = sorted(list(RAW_ROOT.rglob("*.jsonl")) + list(RAW_ROOT.rglob("*.jsonl.gz")))
    if not raw_files:
        raise RuntimeError(f"No staged raw files found under: {RAW_ROOT}")

    rows: list[dict] = []
    for raw_file in raw_files:
        for payload in iter_records(raw_file):
            row = normalize_row(payload)
            if not row["text"]:
                continue
            rows.append(row)
            if MODE == "debug" and len(rows) >= DEBUG_MAX_DOCS:
                break
        if MODE == "debug" and len(rows) >= DEBUG_MAX_DOCS:
            break

    if not rows:
        raise RuntimeError("No usable rows found after preprocessing")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)

    summary = {
        "mode": MODE,
        "seed": SEED,
        "raw_files_used": len(raw_files),
        "documents_written": int(df.shape[0]),
        "output_parquet": str(PROCESSED_PATH),
        "columns": list(df.columns),
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Prepared {df.shape[0]} documents")
    print(f"Parquet: {PROCESSED_PATH}")
    print(f"Summary: {SUMMARY_PATH}")

