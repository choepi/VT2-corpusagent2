from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests
import urllib3

import sys
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.agent_backends import OpenSearchConfig
from corpusagent2.app_config import load_project_configuration
from corpusagent2.io_utils import write_json


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


PROJECT_ROOT = REPO_ROOT
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "documents.parquet"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "opensearch" / "bulk_index_summary.json"


def _batch_size_from_env() -> int:
    raw = os.getenv("CORPUSAGENT2_OPENSEARCH_BULK_BATCH_SIZE", "2000").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"CORPUSAGENT2_OPENSEARCH_BULK_BATCH_SIZE must be an integer, got: {raw}"
        ) from exc
    if value <= 0:
        raise ValueError("CORPUSAGENT2_OPENSEARCH_BULK_BATCH_SIZE must be > 0")
    return value


def _auth(config: OpenSearchConfig) -> tuple[str, str] | None:
    if config.username or config.password:
        return (config.username, config.password)
    return None


def _ensure_index(config: OpenSearchConfig) -> None:
    mapping = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "id": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
                "body": {"type": "text"},
                "content": {"type": "text"},
                "published_at": {
                    "type": "date",
                    "format": "strict_date_optional_time||epoch_millis||yyyy-MM-dd HH:mm:ss||yyyy-MM-dd",
                },
                "source": {"type": "keyword"},
                "source_domain": {"type": "keyword"},
            }
        },
    }
    response = requests.put(
        f"{config.base_url.rstrip('/')}/{config.index_name}",
        auth=_auth(config),
        json=mapping,
        verify=config.verify_ssl,
        timeout=max(config.timeout_s, 120.0),
    )
    if response.status_code not in {200, 201}:
        payload = response.text
        if '"resource_already_exists_exception"' not in payload:
            response.raise_for_status()


def _bulk_index_batch(records: dict[str, list[object]], *, config: OpenSearchConfig) -> int:
    lines: list[str] = []
    row_count = len(records["doc_id"])
    for index in range(row_count):
        doc_id = str(records["doc_id"][index] or "").strip()
        if not doc_id:
            continue
        title = str(records["title"][index] or "")
        text = str(records["text"][index] or "")
        published_at = str(records["published_at"][index] or "")
        source = str(records["source"][index] or "")
        lines.append(json.dumps({"index": {"_id": doc_id}}))
        source_payload = {
            "doc_id": doc_id,
            "id": doc_id,
            "title": title,
            "text": text,
            "body": text,
            "content": text,
            "source": source,
            "source_domain": source,
        }
        if published_at:
            source_payload["published_at"] = published_at
        lines.append(json.dumps(source_payload, ensure_ascii=True))
    if not lines:
        return 0
    response = requests.post(
        f"{config.base_url.rstrip('/')}/{config.index_name}/_bulk",
        auth=_auth(config),
        data="\n".join(lines) + "\n",
        headers={"Content-Type": "application/x-ndjson"},
        verify=config.verify_ssl,
        timeout=max(config.timeout_s, 300.0),
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        for item in payload.get("items", []):
            operation = item.get("index", {})
            if operation.get("status", 200) >= 300:
                raise RuntimeError(f"OpenSearch bulk indexing returned item-level errors: {operation}")
        raise RuntimeError("OpenSearch bulk indexing returned item-level errors.")
    return row_count


def main() -> None:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")

    load_project_configuration(PROJECT_ROOT)
    config = OpenSearchConfig.from_env()
    batch_size = _batch_size_from_env()

    _ensure_index(config)

    parquet = pq.ParquetFile(PARQUET_PATH)
    total_rows = parquet.metadata.num_rows
    processed = 0
    start = time.time()

    for batch_index, batch in enumerate(parquet.iter_batches(batch_size=batch_size), start=1):
        records = batch.to_pydict()
        processed += _bulk_index_batch(records, config=config)
        if batch_index % 20 == 0 or processed >= total_rows:
            elapsed = max(time.time() - start, 1e-6)
            rate = processed / elapsed
            print(
                f"processed={processed}/{total_rows} "
                f"({processed / total_rows:.1%}) rate={rate:,.0f} docs/s elapsed={elapsed / 60:.1f}m"
            )

    refresh = requests.post(
        f"{config.base_url.rstrip('/')}/{config.index_name}/_refresh",
        auth=_auth(config),
        verify=config.verify_ssl,
        timeout=max(config.timeout_s, 120.0),
    )
    refresh.raise_for_status()
    count = requests.get(
        f"{config.base_url.rstrip('/')}/{config.index_name}/_count",
        auth=_auth(config),
        verify=config.verify_ssl,
        timeout=max(config.timeout_s, 120.0),
    )
    count.raise_for_status()
    final_count = int(count.json().get("count", 0) or 0)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        SUMMARY_PATH,
        {
            "parquet_path": str(PARQUET_PATH),
            "index_name": config.index_name,
            "base_url": config.base_url,
            "batch_size": batch_size,
            "rows_processed": processed,
            "final_count": final_count,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"final_count={final_count}")
    print(f"summary={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
