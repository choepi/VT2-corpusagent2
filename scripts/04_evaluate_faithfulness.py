from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.faithfulness import NLIVerifier, evaluate_claims_with_nli
from corpusagent2.io_utils import ensure_absolute, ensure_exists, read_jsonl, write_json, write_jsonl
from corpusagent2.seed import resolve_run_mode, runtime_device_report, set_global_seed


def rejection_rate(rows: list[dict], category: str) -> float:
    subset = [row for row in rows if str(row.get("category", "")).upper() == category.upper()]
    if not subset:
        return 0.0
    rejected = 0
    for item in subset:
        label = str(item.get("label", ""))
        if label in {"contradiction", "neutral", "unsupported"}:
            rejected += 1
    return rejected / float(len(subset))


if __name__ == "__main__":
    MODE = resolve_run_mode("full")
    SEED = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    CLAIMS_PATH = (PROJECT_ROOT / "config" / "faithfulness_claims.jsonl").resolve()
    DOC_METADATA_PATH = (PROJECT_ROOT / "data" / "indices" / "doc_metadata.parquet").resolve()

    OUTPUT_DIR = (PROJECT_ROOT / "outputs" / "faithfulness_eval").resolve()
    VERDICTS_PATH = (OUTPUT_DIR / "claim_verdicts.jsonl").resolve()
    SUMMARY_PATH = (OUTPUT_DIR / "summary.json").resolve()

    NLI_MODEL_ID = "FacebookAI/roberta-large-mnli"
    NLI_DEVICE = None  # auto via CORPUSAGENT2_DEVICE or runtime detection

    ensure_absolute(CLAIMS_PATH, "CLAIMS_PATH")
    ensure_absolute(DOC_METADATA_PATH, "DOC_METADATA_PATH")
    ensure_exists(CLAIMS_PATH, "CLAIMS_PATH")
    ensure_exists(DOC_METADATA_PATH, "DOC_METADATA_PATH")

    set_global_seed(SEED)

    claims = read_jsonl(CLAIMS_PATH)
    if MODE == "debug":
        claims = claims[: min(len(claims), 50)]

    if not claims:
        raise RuntimeError("No claims found in claim file")

    metadata_df = pd.read_parquet(DOC_METADATA_PATH)
    doc_text_by_id = {
        str(row.doc_id): f"{str(row.title)} {str(row.text)}".strip()
        for row in metadata_df.itertuples(index=False)
    }

    verifier = NLIVerifier(model_id=NLI_MODEL_ID, device=NLI_DEVICE)
    verdicts, faithfulness_summary = evaluate_claims_with_nli(
        verifier=verifier,
        claims=claims,
        doc_text_by_id=doc_text_by_id,
    )

    verdict_rows = [item.to_dict() for item in verdicts]
    by_claim_id = {row["claim_id"]: row for row in verdict_rows}

    merged_rows: list[dict] = []
    for claim in claims:
        claim_id = str(claim["claim_id"])
        verdict = by_claim_id.get(claim_id, {})
        merged_rows.append(
            {
                **claim,
                **verdict,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(VERDICTS_PATH, merged_rows)

    summary = {
        "mode": MODE,
        "seed": SEED,
        "nli_model_id": NLI_MODEL_ID,
        "nli_device": verifier.device,
        "nli_fallback_reason": verifier.fallback_reason,
        "device_report": runtime_device_report(),
        "faithfulness": faithfulness_summary,
        "negative_rejection_rate_C": rejection_rate(merged_rows, "C"),
        "counterfactual_rejection_rate_D": rejection_rate(merged_rows, "D"),
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Wrote claim verdicts: {VERDICTS_PATH}")
    print(f"Wrote summary: {SUMMARY_PATH}")

