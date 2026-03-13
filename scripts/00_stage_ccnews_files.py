from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.io_utils import ensure_absolute, ensure_exists, write_json
from corpusagent2.seed import resolve_run_mode, set_global_seed


def collect_source_files(source_root: Path) -> list[Path]:
    patterns = ("*.jsonl", "*.jsonl.gz")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(source_root.rglob(pattern))
    return sorted(set(files))


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def stage_files(source_files: list[Path], destination_root: Path) -> list[dict]:
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    for source in source_files:
        destination = destination_root / source.name
        shutil.copy2(source, destination)
        manifest_rows.append(
            {
                "source": str(source),
                "destination": str(destination),
                "size_bytes": source.stat().st_size,
                "sha256": file_digest(destination),
            }
        )
    return manifest_rows


if __name__ == "__main__":
    MODE = resolve_run_mode("full")
    SEED = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SOURCE_ROOT = (PROJECT_ROOT / "data" / "raw" / "incoming").resolve()
    DESTINATION_ROOT = (PROJECT_ROOT / "data" / "raw" / "ccnews_staged").resolve()
    OUTPUT_SUMMARY_JSON = (PROJECT_ROOT / "outputs" / "stage_ccnews_summary.json").resolve()

    DEBUG_MAX_FILES = 5

    ensure_absolute(SOURCE_ROOT, "SOURCE_ROOT")
    ensure_absolute(DESTINATION_ROOT, "DESTINATION_ROOT")
    ensure_absolute(OUTPUT_SUMMARY_JSON, "OUTPUT_SUMMARY_JSON")
    ensure_exists(SOURCE_ROOT, "SOURCE_ROOT")

    set_global_seed(SEED)

    all_files = collect_source_files(SOURCE_ROOT)
    if not all_files:
        raise RuntimeError(f"No *.jsonl or *.jsonl.gz files found under: {SOURCE_ROOT}")

    if MODE == "debug":
        selected_files = all_files[:DEBUG_MAX_FILES]
    elif MODE == "full":
        selected_files = all_files
    else:
        raise ValueError(f"Unsupported MODE: {MODE}")

    manifest_rows = stage_files(selected_files, DESTINATION_ROOT)

    summary = {
        "mode": MODE,
        "seed": SEED,
        "source_root": str(SOURCE_ROOT),
        "destination_root": str(DESTINATION_ROOT),
        "selected_files": len(selected_files),
        "total_size_bytes": int(sum(item["size_bytes"] for item in manifest_rows)),
        "files": manifest_rows,
    }
    write_json(OUTPUT_SUMMARY_JSON, summary)

    print(f"Staged {len(selected_files)} files to {DESTINATION_ROOT}")
    print(f"Summary: {OUTPUT_SUMMARY_JSON}")

