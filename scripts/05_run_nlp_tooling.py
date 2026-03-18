from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.analysis_tools import build_nlp_outputs
from corpusagent2.seed import resolve_run_mode
from corpusagent2.temporal import normalize_granularity


if __name__ == "__main__":
    mode = resolve_run_mode("full")
    summary = build_nlp_outputs(
        documents_path=(REPO_ROOT / "data" / "processed" / "documents.parquet").resolve(),
        output_dir=(REPO_ROOT / "outputs" / "nlp_tools").resolve(),
        mode=mode,
        seed=42,
        granularity=normalize_granularity(os.getenv("CORPUSAGENT2_TIME_GRANULARITY", "year")),
        ner_model_id="en_core_web_trf",
        sentiment_model_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
        sentiment_device=os.getenv("CORPUSAGENT2_SENTIMENT_DEVICE", "cpu").strip().lower() or "cpu",
    )
    print(f"Wrote NLP outputs to: {summary['outputs']}")
