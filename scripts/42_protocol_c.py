"""Protocol C: metamorphic robustness testing via query transformations.

For each benchmark question, generate four metamorphic variants and re-run
retrieval. The expected behaviour under each transformation is known a priori,
and we measure stability via output-relation metrics rather than against any
gold standard.

Transformations:
  * paraphrase:       lexical rewrite, semantics preserved
                      expected: high top-K Jaccard overlap with original
  * entity_swap:      replace one named entity with a topical pair
                      expected: low top-K Jaccard; ranked-list shift in
                      the direction of the new entity
  * time_shift:       shift any date window by one unit
                      expected: continuous shift, partial overlap
  * negation:         insert / flip a polarity word ("predicted" -> "denied")
                      expected: top-K reshuffles; rejection decision may flip

Three phases, all disk-cached:

  Phase 1 (mutate):    generate variants -> outputs/protocol_c/variants/{qid}.json
  Phase 2 (retrieve):  run lexical+dense+hybrid+rerank on each variant
                       -> outputs/protocol_c/runs/{qid}_{variant}.json
  Phase 3 (metrics):   stability metrics per transformation
                       -> outputs/protocol_c/results/{timestamp}.{json,csv}

Variant generation uses the same LLM client as Protocols A/B.

Run:
    python scripts/42_protocol_c.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ============================================================================
# CONFIG
# ============================================================================

QUESTIONS_PATH = PROJECT_ROOT / "config" / "smoke_questions_10_rows.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "protocol_c"

MUTATOR_BASE_URL = os.getenv("MUTATOR_BASE_URL", "https://hermes.ai.unturf.com/v1")
MUTATOR_API_KEY = os.getenv("MUTATOR_API_KEY", "")
MUTATOR_MODEL = "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic"
TOP_K = 25

PHASE_MUTATE = True
PHASE_RETRIEVE = True
PHASE_METRICS = True

TRANSFORMATIONS = ["paraphrase", "entity_swap", "time_shift", "negation"]

# ============================================================================
# Prompts
# ============================================================================

MUTATOR_SYSTEM = (
    "You are a query-transformation engine for metamorphic robustness testing. "
    "Given an original research question, you produce four variants under four "
    "named relations. Each variant must preserve the original's grammaticality "
    "and topic while changing exactly the property described:\n"
    "  paraphrase:  lexical rewrite, same meaning, same entities, same time scope.\n"
    "  entity_swap: replace one prominent named entity with a topically paired one\n"
    "               (e.g. NZZ <-> Tages-Anzeiger, Trump <-> Obama, oil <-> gas).\n"
    "               State which entity was swapped.\n"
    "  time_shift:  if the question has a date or year range, shift it by one\n"
    "               step (e.g. 2015-2018 -> 2019-2022). If no time scope is\n"
    "               present, return the original unchanged and set 'applicable'\n"
    "               to false.\n"
    "  negation:    flip a polarity-bearing word ('predicted' -> 'denied',\n"
    "               'supported' -> 'opposed'). If no such word exists, return\n"
    "               the original and set 'applicable' to false.\n"
    "Return JSON exactly of the form "
    '{"variants": {"paraphrase": {"text": "...", "applicable": true, "note": ""}, '
    '"entity_swap": {...}, "time_shift": {...}, "negation": {...}}}. '
    "No prose outside the JSON."
)

MUTATOR_USER = "ORIGINAL QUESTION:\n{question}\n\nOutput the JSON object now."

# ============================================================================
# Helpers
# ============================================================================


def _variants_path(qid: str) -> Path:
    return OUTPUT_DIR / "variants" / f"{qid}.json"


def _run_path(qid: str, variant: str) -> Path:
    return OUTPUT_DIR / "runs" / f"{qid}__{variant}.json"


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _rank_overlap(a: list[str], b: list[str], k: int) -> float:
    return _jaccard(a[:k], b[:k])


# ============================================================================
# Phase 1: mutate
# ============================================================================


def run_mutate(questions: list[dict]) -> None:
    from corpusagent2.llm_provider import LLMProviderConfig, OpenAICompatibleLLMClient

    provider = LLMProviderConfig(
        base_url=MUTATOR_BASE_URL,
        api_key=MUTATOR_API_KEY,
        timeout_s=float(os.getenv("CORPUSAGENT2_LLM_TIMEOUT_S", "60")),
        verify_ssl=True,
    )
    client = OpenAICompatibleLLMClient(provider)

    (OUTPUT_DIR / "variants").mkdir(parents=True, exist_ok=True)

    for q in questions:
        qid = q["question_id"]
        out = _variants_path(qid)
        if out.exists():
            print(f"[mutate] {qid}: cached, skipping")
            continue
        try:
            parsed = client.complete_json(
                [
                    {"role": "system", "content": MUTATOR_SYSTEM},
                    {"role": "user", "content": MUTATOR_USER.format(question=q["raw_question"])},
                ],
                model=MUTATOR_MODEL,
                temperature=0.0,
            )
            variants = parsed.get("variants", {})
        except Exception as exc:
            print(f"[mutate] {qid}: failed -- {type(exc).__name__}: {exc}")
            continue
        out.write_text(json.dumps({
            "question_id": qid,
            "original": q["raw_question"],
            "variants": variants,
            "mutator_model": MUTATOR_MODEL,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        applicable = [k for k, v in variants.items() if v.get("applicable", True)]
        print(f"[mutate] {qid}: {len(applicable)}/{len(TRANSFORMATIONS)} variants applicable")


# ============================================================================
# Phase 2: retrieve under each variant
# ============================================================================


def run_retrieve_variants(questions: list[dict]) -> None:
    from corpusagent2.app_config import AppConfig
    from corpusagent2.agent_runtime import AgentRuntime, AgentRuntimeConfig

    config = AgentRuntimeConfig(project_root=PROJECT_ROOT, outputs_root=PROJECT_ROOT / "outputs" / "agent_runtime")
    runtime = AgentRuntime(config=config, app_config=AppConfig.from_project_root(PROJECT_ROOT))
    backend = runtime.search_backend

    modes = [("lexical", False), ("dense", False), ("hybrid", False), ("hybrid_rerank", True)]
    (OUTPUT_DIR / "runs").mkdir(parents=True, exist_ok=True)

    for q in questions:
        qid = q["question_id"]
        var_path = _variants_path(qid)
        if not var_path.exists():
            continue
        variants_data = json.loads(var_path.read_text(encoding="utf-8"))
        for variant_name in ["original"] + TRANSFORMATIONS:
            out = _run_path(qid, variant_name)
            if out.exists():
                continue
            if variant_name == "original":
                text = q["raw_question"]
                applicable = True
            else:
                v = variants_data["variants"].get(variant_name, {})
                text = str(v.get("text", "")).strip()
                applicable = bool(v.get("applicable", False))
                if not applicable or not text or text == q["raw_question"]:
                    out.write_text(json.dumps({
                        "question_id": qid, "variant": variant_name,
                        "applicable": False, "text": text,
                    }, indent=2, ensure_ascii=False), encoding="utf-8")
                    continue

            per_system = {}
            for system_name, use_rerank in modes:
                mode = "hybrid" if use_rerank else system_name
                t0 = time.monotonic()
                try:
                    rows = backend.search(
                        query=text, top_k=TOP_K, retrieval_mode=mode,
                        use_rerank=use_rerank, rerank_top_k=TOP_K,
                    )
                except Exception as exc:
                    print(f"[retrieve] {qid}/{variant_name}/{system_name}: ERROR {exc}")
                    rows = []
                ids = [str(r.get("doc_id") or r.get("id") or "") for r in rows]
                per_system[system_name] = [d for d in ids if d]
                print(f"[retrieve] {qid}/{variant_name}/{system_name}: {len(ids)} in {time.monotonic()-t0:.2f}s")

            out.write_text(json.dumps({
                "question_id": qid, "variant": variant_name, "applicable": True,
                "text": text, "per_system": per_system,
            }, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================================
# Phase 3: stability metrics
# ============================================================================


def run_metrics(questions: list[dict]) -> None:
    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    systems = ["lexical", "dense", "hybrid", "hybrid_rerank"]

    # Per-transformation aggregate
    agg: dict[str, dict] = {
        t: {s: [] for s in systems} for t in TRANSFORMATIONS
    }
    per_question_rows = []

    for q in questions:
        qid = q["question_id"]
        orig_path = _run_path(qid, "original")
        if not orig_path.exists():
            continue
        orig = json.loads(orig_path.read_text(encoding="utf-8"))
        q_row = {"question_id": qid}
        for variant in TRANSFORMATIONS:
            var_path = _run_path(qid, variant)
            if not var_path.exists():
                continue
            var = json.loads(var_path.read_text(encoding="utf-8"))
            if not var.get("applicable", False):
                q_row[f"{variant}_applicable"] = False
                continue
            for s in systems:
                orig_ids = orig.get("per_system", {}).get(s, [])
                var_ids = var.get("per_system", {}).get(s, [])
                overlap = _rank_overlap(orig_ids, var_ids, 10)
                agg[variant][s].append(overlap)
                q_row[f"{variant}_{s}_top10_jaccard"] = round(overlap, 4)
        per_question_rows.append(q_row)

    # Summarise
    summary = {
        "top_k_used_for_stability": 10,
        "per_transformation": {
            t: {
                s: {
                    "mean_jaccard@10": round(sum(vals) / len(vals), 4) if vals else 0.0,
                    "n": len(vals),
                }
                for s, vals in by_system.items()
            }
            for t, by_system in agg.items()
        },
        "per_question": per_question_rows,
        "interpretation": {
            "paraphrase": "high Jaccard expected (>=0.5 acceptable, >=0.7 strong)",
            "entity_swap": "low Jaccard expected (<=0.3 strong, <=0.5 acceptable)",
            "time_shift": "partial Jaccard expected (0.3-0.7 plausible)",
            "negation": "low Jaccard expected if applicable (<=0.3 strong)",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (OUTPUT_DIR / "results" / f"{stamp}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["transformation,system,mean_jaccard@10,n"]
    for t in TRANSFORMATIONS:
        for s in systems:
            row = summary["per_transformation"][t][s]
            lines.append(f"{t},{s},{row['mean_jaccard@10']},{row['n']}")
    (OUTPUT_DIR / "results" / f"{stamp}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[metrics] wrote results/{stamp}.json")
    print()
    print("=== robustness summary (mean top-10 Jaccard with original) ===")
    print(f"{'transformation':<14}{'lexical':>10}{'dense':>10}{'hybrid':>10}{'hybrid+rr':>12}")
    for t in TRANSFORMATIONS:
        row = summary["per_transformation"][t]
        print(f"{t:<14}"
              f"{row['lexical']['mean_jaccard@10']:>10.4f}"
              f"{row['dense']['mean_jaccard@10']:>10.4f}"
              f"{row['hybrid']['mean_jaccard@10']:>10.4f}"
              f"{row['hybrid_rerank']['mean_jaccard@10']:>12.4f}")
    print()
    print("Interpretation cues:")
    for t, hint in summary["interpretation"].items():
        print(f"  {t}: {hint}")


# ============================================================================
# main
# ============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")
    print()

    if PHASE_MUTATE:
        print("=== Phase 1: query mutation ===")
        run_mutate(questions)
        print()
    if PHASE_RETRIEVE:
        print("=== Phase 2: retrieval under each variant ===")
        run_retrieve_variants(questions)
        print()
    if PHASE_METRICS:
        print("=== Phase 3: stability metrics ===")
        run_metrics(questions)
