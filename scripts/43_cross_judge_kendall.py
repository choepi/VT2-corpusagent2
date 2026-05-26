"""Cross-judge ablation: Kendall tau between system rankings under two judges.

Run scripts/40_protocol_a.py twice with different JUDGE_MODEL values; this
script reads the two most recent results JSONs and reports the Kendall tau
between the per-system rankings on each headline metric.

Interpretation:
    tau >= 0.85   judges agree on system ranking (UMBRELA threshold)
    tau >= 0.6    judges agree directionally
    tau <  0.6    judge dependence is a concrete threat; do not collapse
                  results across the two judges in headline tables

Run:
    python scripts/43_cross_judge_kendall.py
"""

from __future__ import annotations

import glob
import itertools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "protocol_a" / "results"


def kendall_tau(a: list[float], b: list[float]) -> float:
    """Plain Kendall's tau-b. Returns 0.0 when degenerate."""
    n = len(a)
    if n < 2 or n != len(b):
        return 0.0
    concordant = discordant = ties_a = ties_b = 0
    for i, j in itertools.combinations(range(n), 2):
        da, db = a[i] - a[j], b[i] - b[j]
        if da == 0 and db == 0:
            continue
        if da == 0:
            ties_a += 1; continue
        if db == 0:
            ties_b += 1; continue
        if (da > 0) == (db > 0):
            concordant += 1
        else:
            discordant += 1
    denom_a = (concordant + discordant + ties_a)
    denom_b = (concordant + discordant + ties_b)
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return (concordant - discordant) / ((denom_a * denom_b) ** 0.5)


def main() -> None:
    files = sorted(glob.glob(str(RESULTS_DIR / "*.json")))
    if len(files) < 2:
        print(f"Need at least 2 Protocol A results JSONs in {RESULTS_DIR}; found {len(files)}.")
        print("Run scripts/40_protocol_a.py twice with different JUDGE_MODEL values.")
        sys.exit(1)

    # Use the two most recent runs
    results = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files[-2:]]
    judges = [r.get("judge_model", "?") for r in results]
    print(f"Comparing system rankings between:\n  A: {judges[0]}\n  B: {judges[1]}\n")

    systems = ["lexical", "dense", "hybrid", "hybrid_rerank"]
    metrics = ["ndcg@10", "recall@25", "map"]

    out_rows = []
    for metric in metrics:
        a = [results[0]["per_system"][s][metric]["mean"] for s in systems]
        b = [results[1]["per_system"][s][metric]["mean"] for s in systems]
        tau = kendall_tau(a, b)
        out_rows.append((metric, tau, a, b))
        verdict = "agree" if tau >= 0.85 else ("directional" if tau >= 0.6 else "DIVERGE")
        print(f"  {metric:12s}  tau = {tau:+.4f}   [{verdict}]")
        print(f"    judge A scores: " + ", ".join(f"{s}={v:.3f}" for s, v in zip(systems, a)))
        print(f"    judge B scores: " + ", ".join(f"{s}={v:.3f}" for s, v in zip(systems, b)))
        print()

    out_path = RESULTS_DIR / "cross_judge_kendall.json"
    out_path.write_text(json.dumps({
        "judge_a": judges[0],
        "judge_b": judges[1],
        "by_metric": {m: {"tau": t, "judge_a": a, "judge_b": b} for m, t, a, b in out_rows},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
