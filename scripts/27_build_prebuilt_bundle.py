from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
PREBUILT_ROOT = REPO_ROOT / "outputs" / "prebuilt"
DEFAULT_BUNDLE_PATH = PREBUILT_ROOT / "corpusagent2_prebuilt_bundle.zip"
DEFAULT_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "id": ("id", "doc_id", "_id", "article_id", "new_article_id", "meta.article_id"),
    "title": ("title", "headline"),
    "text": ("text", "body", "content", "plain_text", "description", "summary"),
    "published_at": ("published_at", "published_date", "date", "datetime", "timestamp", "year", "crawl_date", "meta.date"),
    "source": ("source", "domain", "publisher", "sitename", "meta.outlet", "url", "requested_url", "responded_url"),
}


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"[run] {' '.join(command)}")
    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)


def _slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    normalized = "_".join(part for part in cleaned.split("_") if part)
    return normalized or "dataset"


def _write_jsonl_gz(records: Iterable[dict], target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True, default=str))
            handle.write("\n")
            count += 1
    return count


def _nested_value(payload: dict, dotted_key: str):
    current = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _first_value(payload: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _stringify(_nested_value(payload, key))
        if value:
            return value
    return ""


def _normalize_record(payload: dict) -> dict[str, str]:
    row = dict(payload)
    return {
        "id": _first_value(row, DEFAULT_FIELD_CANDIDATES["id"]),
        "title": _first_value(row, DEFAULT_FIELD_CANDIDATES["title"]),
        "text": _first_value(row, DEFAULT_FIELD_CANDIDATES["text"]),
        "published_at": _first_value(row, DEFAULT_FIELD_CANDIDATES["published_at"]),
        "source": _first_value(row, DEFAULT_FIELD_CANDIDATES["source"]),
    }


def _parse_filters(raw_filters: list[str]) -> list[tuple[str, str]]:
    filters: list[tuple[str, str]] = []
    for raw_filter in raw_filters:
        if "=" not in raw_filter:
            raise ValueError(f"Invalid filter '{raw_filter}'. Expected FIELD=VALUE.")
        field_name, expected_value = raw_filter.split("=", 1)
        field_name = field_name.strip()
        expected_value = expected_value.strip()
        if not field_name or not expected_value:
            raise ValueError(f"Invalid filter '{raw_filter}'. Expected FIELD=VALUE.")
        filters.append((field_name, expected_value))
    return filters


def _matches_filters(payload: dict, filters: list[tuple[str, str]]) -> bool:
    for field_name, expected_value in filters:
        actual_value = _stringify(_nested_value(payload, field_name))
        if actual_value.lower() != expected_value.lower():
            return False
    return True


def _iter_normalized_records(
    records: Iterable[dict],
    *,
    filters: list[tuple[str, str]] | None = None,
    max_rows: int = 0,
) -> Iterable[dict[str, str]]:
    written = 0
    active_filters = filters or []
    for payload in records:
        row = dict(payload)
        if active_filters and not _matches_filters(row, active_filters):
            continue
        normalized = _normalize_record(row)
        if not normalized["text"]:
            continue
        yield normalized
        written += 1
        if max_rows and written >= max_rows:
            break


def _records_from_json(path: Path) -> Iterable[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
        return
    if isinstance(payload, dict):
        yield payload
        return
    raise ValueError(f"Unsupported JSON structure in {path}; expected object or list of objects.")


def _records_from_tabular_file(path: Path) -> Iterable[dict]:
    import pandas as pd

    suffixes = [part.lower() for part in path.suffixes]
    if suffixes[-2:] == [".jsonl", ".gz"] or path.suffix.lower() == ".jsonl":
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)
        return
    if path.suffix.lower() == ".json":
        yield from _records_from_json(path)
        return
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".tsv":
        frame = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported source file type: {path}")
    for row in frame.to_dict(orient="records"):
        yield dict(row)


def _convert_source_file(
    source: Path,
    incoming_root: Path,
    *,
    label: str = "",
    max_rows: int = 0,
    filters: list[tuple[str, str]] | None = None,
) -> dict[str, object]:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    incoming_root.mkdir(parents=True, exist_ok=True)
    base_name = _slug(label or source.stem)
    target = incoming_root / f"{base_name}.jsonl.gz"
    row_count = _write_jsonl_gz(
        _iter_normalized_records(_records_from_tabular_file(source), filters=filters, max_rows=max_rows),
        target,
    )
    return {
        "source_type": "file_normalize",
        "source": str(source),
        "incoming_path": str(target),
        "records_written": row_count,
    }


def _export_hf_dataset(
    *,
    dataset_name: str,
    config_name: str,
    split_name: str,
    revision: str,
    incoming_root: Path,
    streaming: bool,
    max_rows: int,
    filters: list[tuple[str, str]] | None = None,
) -> dict[str, object]:
    from datasets import load_dataset

    load_kwargs: dict[str, object] = {"path": dataset_name, "split": split_name}
    if config_name:
        load_kwargs["name"] = config_name
    if revision:
        load_kwargs["revision"] = revision
    if streaming:
        load_kwargs["streaming"] = True
    dataset = load_dataset(**load_kwargs)
    target = incoming_root / f"{_slug(dataset_name)}_{_slug(config_name or split_name)}.jsonl.gz"
    row_count = _write_jsonl_gz(
        _iter_normalized_records((dict(row) for row in dataset), filters=filters, max_rows=max_rows),
        target,
    )
    return {
        "source_type": "huggingface_dataset",
        "dataset": dataset_name,
        "config": config_name,
        "split": split_name,
        "revision": revision,
        "streaming": streaming,
        "filters": [f"{field}={value}" for field, value in (filters or [])],
        "incoming_path": str(target),
        "records_written": row_count,
    }


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _clean_previous_outputs() -> None:
    for target in (
        REPO_ROOT / "data" / "raw" / "incoming",
        REPO_ROOT / "data" / "raw" / "ccnews_staged",
        REPO_ROOT / "data" / "processed",
        REPO_ROOT / "data" / "indices",
        REPO_ROOT / "outputs" / "nlp_tools",
    ):
        _remove_path(target)
    for target in (
        REPO_ROOT / "outputs" / "stage_ccnews_summary.json",
        REPO_ROOT / "outputs" / "prepare_dataset_summary.json",
        REPO_ROOT / "outputs" / "build_retrieval_assets_summary.json",
    ):
        _remove_path(target)


def _bundle_members(*, include_nlp: bool) -> list[Path]:
    members = [
        REPO_ROOT / "data" / "processed" / "documents.parquet",
        REPO_ROOT / "data" / "indices" / "doc_metadata.parquet",
        REPO_ROOT / "data" / "indices" / "lexical" / "tfidf_vectorizer.joblib",
        REPO_ROOT / "data" / "indices" / "lexical" / "tfidf_matrix.joblib",
        REPO_ROOT / "data" / "indices" / "lexical" / "tfidf_doc_ids.joblib",
        REPO_ROOT / "data" / "indices" / "dense" / "dense_embeddings.npy",
        REPO_ROOT / "data" / "indices" / "dense" / "dense_doc_ids.joblib",
        REPO_ROOT / "outputs" / "stage_ccnews_summary.json",
        REPO_ROOT / "outputs" / "prepare_dataset_summary.json",
        REPO_ROOT / "outputs" / "build_retrieval_assets_summary.json",
    ]
    if include_nlp:
        members.extend(
            [
                REPO_ROOT / "outputs" / "nlp_tools" / "summary.json",
                REPO_ROOT / "outputs" / "nlp_tools" / "entity_trend.parquet",
                REPO_ROOT / "outputs" / "nlp_tools" / "sentiment_series.parquet",
                REPO_ROOT / "outputs" / "nlp_tools" / "topics_over_time.parquet",
                REPO_ROOT / "outputs" / "nlp_tools" / "burst_events.parquet",
                REPO_ROOT / "outputs" / "nlp_tools" / "keyphrases.parquet",
            ]
        )
    return [path for path in members if path.exists()]


def _write_bundle(bundle_path: Path, members: list[Path], manifest_payload: dict[str, object]) -> Path:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = PREBUILT_ROOT / f"{bundle_path.stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for member in members:
            archive.write(member, member.relative_to(REPO_ROOT).as_posix())
        archive.writestr(
            "outputs/prebuilt_bundle_manifest.json",
            json.dumps(manifest_payload, indent=2),
        )
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a portable preprocessed CorpusAgent2 bundle for transfer from a Slurm/GPU cluster to a VM."
    )
    parser.add_argument("--hf-dataset", default="", help="Hugging Face dataset id, e.g. vblagoje/cc_news.")
    parser.add_argument("--hf-config", default="", help="Optional Hugging Face dataset config name.")
    parser.add_argument("--hf-split", default="train", help="Hugging Face dataset split.")
    parser.add_argument("--hf-revision", default="", help="Optional Hugging Face dataset revision.")
    parser.add_argument("--hf-streaming", action="store_true", help="Stream Hugging Face rows instead of materializing the full split locally first.")
    parser.add_argument("--hf-max-rows", type=int, default=0, help="Optional limit after filtering/normalization. 0 means no limit.")
    parser.add_argument(
        "--hf-filter",
        action="append",
        default=[],
        help="Exact-match filter in FIELD=VALUE form. Supports dotted paths like language=en or meta.outlet=bbc.com.",
    )
    parser.add_argument("--source-file", action="append", default=[], help="Local file to convert/copy into data/raw/incoming.")
    parser.add_argument("--bundle-path", default=str(DEFAULT_BUNDLE_PATH), help="Output zip path.")
    parser.add_argument("--granularity", choices=["year", "month"], default=os.getenv("CORPUSAGENT2_TIME_GRANULARITY", "month").strip().lower() or "month")
    parser.add_argument("--mode", choices=["debug", "full"], default=os.getenv("CORPUSAGENT2_MODE", "full").strip().lower() or "full")
    parser.add_argument("--dense-batch-size", type=int, default=128, help="Dense embedding batch size for cluster preprocessing.")
    parser.add_argument("--dense-chunk-size", type=int, default=2048, help="Dense embedding chunk size for streaming writes.")
    parser.add_argument("--sentiment-device", default="auto", help="Preferred sentiment device for NLP outputs, e.g. auto/cuda/mps/cpu.")
    parser.add_argument("--skip-nlp", action="store_true", help="Skip building outputs/nlp_tools artifacts.")
    parser.add_argument("--clean-existing", action="store_true", help="Delete existing incoming/staged/processed/index/output artifacts first.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    incoming_root = REPO_ROOT / "data" / "raw" / "incoming"
    active_filters = _parse_filters(args.hf_filter)
    if args.clean_existing:
        _clean_previous_outputs()

    source_rows: list[dict[str, object]] = []
    if args.hf_dataset:
        source_rows.append(
            _export_hf_dataset(
                dataset_name=args.hf_dataset,
                config_name=args.hf_config,
                split_name=args.hf_split,
                revision=args.hf_revision,
                incoming_root=incoming_root,
                streaming=args.hf_streaming,
                max_rows=max(args.hf_max_rows, 0),
                filters=active_filters,
            )
        )
    for item in args.source_file:
        source_rows.append(
            _convert_source_file(
                Path(item),
                incoming_root,
                max_rows=max(args.hf_max_rows, 0),
                filters=active_filters,
            )
        )

    if not source_rows and not any(incoming_root.rglob("*.jsonl")) and not any(incoming_root.rglob("*.jsonl.gz")):
        raise RuntimeError("No sources provided and data/raw/incoming is empty.")

    env = os.environ.copy()
    env["CORPUSAGENT2_MODE"] = args.mode
    env["CORPUSAGENT2_TIME_GRANULARITY"] = args.granularity
    env["CORPUSAGENT2_BUILD_LEXICAL_ASSETS"] = "true"
    env["CORPUSAGENT2_BUILD_DENSE_ASSETS"] = "true"
    env["CORPUSAGENT2_STREAM_DENSE_ASSETS"] = "true"
    env["CORPUSAGENT2_DENSE_BATCH_SIZE"] = str(max(args.dense_batch_size, 1))
    env["CORPUSAGENT2_DENSE_CHUNK_SIZE"] = str(max(args.dense_chunk_size, 1))
    env["CORPUSAGENT2_SENTIMENT_DEVICE"] = args.sentiment_device.strip().lower() or "auto"

    python_exe = Path(sys.executable).resolve()
    _run([str(python_exe), str(REPO_ROOT / "scripts" / "00_stage_ccnews_files.py")], env=env)
    _run([str(python_exe), str(REPO_ROOT / "scripts" / "01_prepare_dataset.py")], env=env)
    _run([str(python_exe), str(REPO_ROOT / "scripts" / "02_build_retrieval_assets.py")], env=env)
    if not args.skip_nlp:
        _run([str(python_exe), str(REPO_ROOT / "scripts" / "05_run_nlp_tooling.py")], env=env)

    members = _bundle_members(include_nlp=not args.skip_nlp)
    if not members:
        raise RuntimeError("Nothing was produced for the portable bundle.")

    bundle_path = Path(args.bundle_path).expanduser().resolve()
    manifest_payload = {
        "bundle_path": str(bundle_path),
        "project_root": str(REPO_ROOT),
        "created_with_python": str(python_exe),
        "mode": args.mode,
        "time_granularity": args.granularity,
        "dense_batch_size": int(env["CORPUSAGENT2_DENSE_BATCH_SIZE"]),
        "dense_chunk_size": int(env["CORPUSAGENT2_DENSE_CHUNK_SIZE"]),
        "sentiment_device": env["CORPUSAGENT2_SENTIMENT_DEVICE"],
        "hf_streaming": bool(args.hf_streaming),
        "hf_max_rows": int(max(args.hf_max_rows, 0)),
        "hf_filters": [f"{field}={value}" for field, value in active_filters],
        "include_nlp_outputs": not args.skip_nlp,
        "sources": source_rows,
        "bundle_members": [str(path.relative_to(REPO_ROOT).as_posix()) for path in members],
        "vm_restore_hint": "Extract into the repo root on the VM, then run: python scripts/22_prepare_vm_stack.py --skip-provider-assets",
    }
    manifest_path = _write_bundle(bundle_path, members, manifest_payload)
    print("")
    print(f"Portable bundle ready: {bundle_path}")
    print(f"Manifest: {manifest_path}")
    print("Next on the VM: extract into the repo root, then run scripts/22_prepare_vm_stack.py --skip-provider-assets")


if __name__ == "__main__":
    main()
