from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    default_target = project_root / "data" / "raw" / "incoming" / "cc_news.jsonl.gz"
    target = Path(os.getenv("CORPUSAGENT2_CCNEWS_OUTPUT", str(default_target))).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("vblagoje/cc_news", split="train")
    ds.to_json(str(target), compression="gzip")
    print(f"Downloaded CC-News dataset to {target}")
