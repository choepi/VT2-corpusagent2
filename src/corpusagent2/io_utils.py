from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_exists(path: Path, path_label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path_label} does not exist: {path}")


def ensure_absolute(path: Path, path_label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{path_label} must be absolute: {path}")


def read_json(path: Path) -> dict:
    ensure_exists(path, "JSON file")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    ensure_exists(path, "JSONL file")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def read_documents(path: Path) -> pd.DataFrame:
    ensure_exists(path, "documents parquet")
    df = pd.read_parquet(path)
    expected_columns = {"doc_id", "title", "text", "published_at"}
    missing = expected_columns.difference(df.columns)
    if missing:
        raise ValueError(f"documents parquet missing required columns: {sorted(missing)}")
    return df


def sentence_split(text: str) -> list[str]:
    """Simple sentence splitter to avoid heavy dependencies for evaluation paths."""
    chunks = [part.strip() for part in text.replace("\n", " ").split(".")]
    return [chunk for chunk in chunks if chunk]
