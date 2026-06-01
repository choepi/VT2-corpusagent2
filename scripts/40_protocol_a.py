"""Protocol A: LLM-as-judge relevance scoring for retrieval ablation.

Runs four retrieval modes (lexical / dense / hybrid / hybrid+rerank) on the
benchmark questions, pools the top-K per (system, question), has a pinned LLM
judge score each unique (question, document) pair for graded relevance, then
computes nDCG@10 / Recall@K / MAP per system.

Three phases, each idempotent and disk-cached so re-runs cost zero after the
first complete pass:

  Phase 1 (retrieve): writes outputs/protocol_a/pool/{question_id}.json
  Phase 2 (judge):    writes outputs/protocol_a/judge_cache/{hash}.json
  Phase 3 (metrics):  writes outputs/protocol_a/results/{timestamp}.json
                              outputs/protocol_a/results/{timestamp}.csv

Edit the CONFIG block below before running. No argparse by repo convention.

Run:
    python scripts/40_protocol_a.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Path setup --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ============================================================================
# CONFIG -- edit before running
# ============================================================================

# Question source: must be a JSON list of {"question_id", "raw_question"} dicts.
QUESTIONS_PATH = PROJECT_ROOT / "config" / "smoke_questions_10_rows.json"

# Output root.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "protocol_a"

# Retrieval depth pooled per system per question.
TOP_K = 25

# Maximum tokens of document text included in each judge prompt. Truncation
# beyond this point is by simple character count (4 chars per token estimate).
JUDGE_CONTEXT_TOKENS = 1024

# Judge endpoint. Independent of the synthesis-stage LLM provider so judge and
# producer share neither model family nor inference endpoint.
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://hermes.ai.unturf.com/v1")
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", "")  # may be empty for free endpoints

# Judge model. MUST differ from the synthesis-stage LLM (avoid circularity).
# Examples:
#   "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic"  # Unclose, free, cross-family
#   "claude-haiku-4.5"                              # Anthropic, strongest cross-family
#   "gpt-5.4-nano-2026-03-17"                       # OpenAI, soft circularity risk
JUDGE_MODEL = "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic"

# Phase toggles -- skip a phase if its inputs are already on disk.
# PHASE_SANITY_CHECK runs first if enabled: it judges a small hand-curated set
# of (clearly relevant, clearly irrelevant) doc pairs so the operator can
# eyeball judge behaviour before committing budget to the full eval. If
# PHASE_SANITY_CHECK is on and any sanity case fails the expected-label test,
# the script exits before running Phase 1.
PHASE_SANITY_CHECK = True
PHASE_RETRIEVE = True
PHASE_JUDGE = True
PHASE_METRICS = True

# How many docs from the dedup pool to score in this run (None = all). Useful
# for a budget-bounded smoke test before committing to the full eval.
JUDGE_MAX_DOCS = None

# ============================================================================
# Judge prompt -- treated as immutable; changes invalidate cached judgments.
# ============================================================================

JUDGE_SYSTEM_PROMPT = (
    "You are an expert relevance assessor for an information-retrieval "
    "evaluation. Given a research question about a news corpus and a "
    "candidate document, judge how relevant the document is to answering "
    "the question on a 4-point graded scale:\n"
    "  3 = Highly Relevant     (document directly answers a core part)\n"
    "  2 = Relevant            (document provides clearly useful evidence)\n"
    "  1 = Marginally Relevant (document mentions the topic but is tangential)\n"
    "  0 = Not Relevant        (document is off-topic or unrelated)\n"
    "Return JSON exactly of the form {\"label\": <int 0..3>, \"reason\": <str>}."
    " No prose outside the JSON object."
)

JUDGE_USER_TEMPLATE = (
    "QUESTION:\n{question}\n\n"
    "DOCUMENT (truncated to {tokens} tokens):\n{document}\n\n"
    "Output the JSON object now."
)

# ============================================================================
# Helpers
# ============================================================================


def _pool_path(question_id: str) -> Path:
    return OUTPUT_DIR / "pool" / f"{question_id}.json"


def _judge_cache_path(prompt_hash: str) -> Path:
    return OUTPUT_DIR / "judge_cache" / f"{prompt_hash}.json"


def _prompt_hash(question: str, document: str, judge_model: str) -> str:
    h = hashlib.sha256()
    h.update(JUDGE_SYSTEM_PROMPT.encode("utf-8"))
    h.update(b"\x1f")
    h.update(JUDGE_USER_TEMPLATE.encode("utf-8"))
    h.update(b"\x1f")
    h.update(judge_model.encode("utf-8"))
    h.update(b"\x1f")
    h.update(question.encode("utf-8"))
    h.update(b"\x1f")
    h.update(document.encode("utf-8"))
    return h.hexdigest()[:32]


def _truncate_doc(text: str, tokens: int) -> str:
    char_limit = tokens * 4
    return text[:char_limit]


# ============================================================================
# Phase 0: judge sanity check
# ============================================================================

# Four hand-curated probes: two should obviously score 2-3, two should score 0-1.
# If the judge's labels diverge from these expectations, treat as a red flag
# and inspect before spending eval budget.
SANITY_CASES = [
    {
        "id": "sanity_pos_1",
        "question": "What was the Federal Reserve's interest-rate policy in 2022?",
        "document": (
            "The Federal Reserve raised its benchmark interest rate by 75 basis points "
            "in June 2022, the largest single increase since 1994, as policymakers "
            "moved aggressively to contain inflation that had reached a 40-year high. "
            "Chair Jerome Powell signalled further hikes through the rest of the year."
        ),
        "expected_min_label": 2,  # Relevant or Highly Relevant
        "rationale": "directly answers the question with specific 2022 Fed actions",
    },
    {
        "id": "sanity_pos_2",
        "question": "How did Swiss newspapers cover the 2022 invasion of Ukraine?",
        "document": (
            "Neue Zürcher Zeitung devoted its front page on 25 February 2022 to "
            "Russia's invasion of Ukraine, with editorials condemning the action "
            "and live-blog coverage from Kyiv. Tages-Anzeiger and Le Temps "
            "followed similar editorial lines through the early weeks of the war."
        ),
        "expected_min_label": 2,
        "rationale": "directly describes Swiss newspaper coverage of the invasion",
    },
    {
        "id": "sanity_neg_1",
        "question": "What was the Federal Reserve's interest-rate policy in 2022?",
        "document": (
            "Recipe for traditional French onion soup: caramelise four large onions "
            "over low heat for 45 minutes, deglaze with white wine, add beef stock "
            "and simmer. Top with toasted baguette and Gruyère cheese, then broil."
        ),
        "expected_max_label": 1,  # Not Relevant or Marginally Relevant
        "rationale": "soup recipe, unrelated to Fed policy",
    },
    {
        "id": "sanity_neg_2",
        "question": "How did Swiss newspapers cover the 2022 invasion of Ukraine?",
        "document": (
            "FC Basel won the Swiss Super League in 2017 with a record points total. "
            "The club's youth academy produced several internationals during the "
            "2010s, including Granit Xhaka and Mohamed Salah's brief loan spell."
        ),
        "expected_max_label": 1,
        "rationale": "Swiss football history, unrelated to Ukraine war coverage",
    },
]


def run_sanity_check() -> bool:
    """Returns True if every sanity case lands inside its expected range."""
    from corpusagent2.llm_provider import LLMProviderConfig, OpenAICompatibleLLMClient

    provider = LLMProviderConfig(
        base_url=JUDGE_BASE_URL,
        api_key=JUDGE_API_KEY,
        timeout_s=float(os.getenv("CORPUSAGENT2_LLM_TIMEOUT_S", "60")),
        verify_ssl=True,
    )
    client = OpenAICompatibleLLMClient(provider)

    all_passed = True
    rows = []
    for case in SANITY_CASES:
        user = JUDGE_USER_TEMPLATE.format(
            question=case["question"],
            document=case["document"],
            tokens=JUDGE_CONTEXT_TOKENS,
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        try:
            parsed = client.complete_json(messages, model=JUDGE_MODEL, temperature=0.0)
            label = int(parsed.get("label", -1))
            reason = str(parsed.get("reason", ""))
        except Exception as exc:
            print(f"[sanity] {case['id']}: JUDGE CALL FAILED -- {type(exc).__name__}: {exc}")
            return False

        expected = (
            f">= {case['expected_min_label']}" if "expected_min_label" in case
            else f"<= {case['expected_max_label']}"
        )
        if "expected_min_label" in case:
            passed = label >= case["expected_min_label"]
        else:
            passed = label <= case["expected_max_label"]
        if not passed:
            all_passed = False
        rows.append({
            "id": case["id"],
            "expected": expected,
            "label": label,
            "passed": passed,
            "reason": reason,
        })
        marker = "PASS" if passed else "FAIL"
        print(f"[sanity] {case['id']}: label={label} expected {expected}  [{marker}]")
        if reason:
            print(f"          judge said: {reason[:120]}")

    # Persist for auditability
    (OUTPUT_DIR / "sanity").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {
        "judge_model": JUDGE_MODEL,
        "rows": rows,
        "all_passed": all_passed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT_DIR / "sanity" / f"{stamp}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not all_passed:
        print()
        print("[sanity] One or more cases failed expected-label test.")
        print("[sanity] Consider switching judge model or revising the prompt before Phase 1.")
    return all_passed


# ============================================================================
# Phase 1: retrieval
# ============================================================================


def run_retrieval(questions: list[dict]) -> None:
    """Run 4 retrieval modes per question; write pool to disk.

    Imports the runtime lazily so that judging/metrics phases can run without
    a healthy backend.
    """
    from corpusagent2.app_config import AppConfig
    from corpusagent2.agent_runtime import AgentRuntime, AgentRuntimeConfig

    config = AgentRuntimeConfig(project_root=PROJECT_ROOT, outputs_root=PROJECT_ROOT / "outputs" / "agent_runtime")
    runtime = AgentRuntime(config=config, app_config=AppConfig.from_project_root(PROJECT_ROOT))
    backend = runtime.search_backend

    modes = [
        ("lexical", False),
        ("dense", False),
        ("hybrid", False),
        ("hybrid_rerank", True),
    ]

    (OUTPUT_DIR / "pool").mkdir(parents=True, exist_ok=True)

    for q in questions:
        qid = q["question_id"]
        question = q["raw_question"]
        out_path = _pool_path(qid)
        if out_path.exists():
            print(f"[retrieve] {qid}: cached, skipping")
            continue

        per_system: dict[str, list[dict]] = {}
        pool_docs: dict[str, dict] = {}
        for system_name, use_rerank in modes:
            mode = "hybrid" if use_rerank else system_name
            t0 = time.monotonic()
            try:
                rows = backend.search(
                    query=question,
                    top_k=TOP_K,
                    retrieval_mode=mode,
                    use_rerank=use_rerank,
                    rerank_top_k=TOP_K,
                )
            except Exception as exc:
                print(f"[retrieve] {qid} system={system_name}: ERROR {type(exc).__name__}: {exc}")
                rows = []
            dt = time.monotonic() - t0
            ranked_ids = [str(r.get("doc_id") or r.get("id") or "") for r in rows]
            per_system[system_name] = [
                {"doc_id": did, "rank": idx + 1, "score": float(rows[idx].get("score", 0.0))}
                for idx, did in enumerate(ranked_ids) if did
            ]
            for idx, did in enumerate(ranked_ids):
                if not did or did in pool_docs:
                    continue
                row = rows[idx]
                pool_docs[did] = {
                    "doc_id": did,
                    "title": str(row.get("title", "")),
                    "snippet": str(row.get("snippet", "")),
                    "text": str(row.get("text", row.get("body", row.get("content", "")))),
                    "outlet": str(row.get("outlet", "")),
                    "date": str(row.get("date", "")),
                }
            print(f"[retrieve] {qid} system={system_name}: {len(ranked_ids)} docs in {dt:.2f}s")

        out_path.write_text(json.dumps({
            "question_id": qid,
            "question": question,
            "per_system": per_system,
            "pool": list(pool_docs.values()),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[retrieve] {qid}: pool={len(pool_docs)} unique docs -> {out_path}")


# ============================================================================
# Phase 2: judge
# ============================================================================


def run_judging(questions: list[dict]) -> None:
    from corpusagent2.app_config import AppConfig
    from corpusagent2.llm_provider import LLMProviderConfig, OpenAICompatibleLLMClient

    (OUTPUT_DIR / "judge_cache").mkdir(parents=True, exist_ok=True)

    # Judge endpoint independent of the synthesis-stage LLM provider.
    provider = LLMProviderConfig(
        base_url=JUDGE_BASE_URL,
        api_key=JUDGE_API_KEY,
        timeout_s=float(os.getenv("CORPUSAGENT2_LLM_TIMEOUT_S", "60")),
        verify_ssl=True,
    )
    client = OpenAICompatibleLLMClient(provider)

    judged = 0
    cached = 0
    failed = 0
    for q in questions:
        qid = q["question_id"]
        pool_path = _pool_path(qid)
        if not pool_path.exists():
            print(f"[judge] {qid}: no pool on disk, run Phase 1 first")
            continue
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        question = pool["question"]
        docs = pool["pool"]
        if JUDGE_MAX_DOCS is not None:
            docs = docs[:JUDGE_MAX_DOCS]

        for doc in docs:
            text = _truncate_doc(doc.get("text") or doc.get("snippet") or "", JUDGE_CONTEXT_TOKENS)
            if not text.strip():
                continue
            phash = _prompt_hash(question, text, JUDGE_MODEL)
            cache_path = _judge_cache_path(phash)
            if cache_path.exists():
                cached += 1
                continue
            user = JUDGE_USER_TEMPLATE.format(
                question=question, document=text, tokens=JUDGE_CONTEXT_TOKENS
            )
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
            try:
                parsed = client.complete_json(messages, model=JUDGE_MODEL, temperature=0.0)
                label = int(parsed.get("label", 0))
                reason = str(parsed.get("reason", ""))
                if label < 0 or label > 3:
                    raise ValueError(f"label out of range: {label}")
            except Exception as exc:
                failed += 1
                print(f"[judge] {qid} doc={doc['doc_id']}: ERROR {type(exc).__name__}: {exc}")
                continue
            judgment = {
                "question_id": qid,
                "doc_id": doc["doc_id"],
                "label": label,
                "reason": reason,
                "judge_model": JUDGE_MODEL,
                "prompt_hash": phash,
                "judged_at": datetime.now(timezone.utc).isoformat(),
            }
            cache_path.write_text(json.dumps(judgment, ensure_ascii=False, indent=2), encoding="utf-8")
            judged += 1
            if judged % 25 == 0:
                print(f"[judge] progress: {judged} new, {cached} cached, {failed} failed")

    print(f"[judge] DONE: {judged} new, {cached} cached, {failed} failed")


# ============================================================================
# Phase 3: metrics
# ============================================================================


def _load_judgments_for_question(qid: str, question: str, pool_docs: list[dict]) -> dict[str, int]:
    """Returns mapping doc_id -> graded label (0..3), 0 if not judged."""
    labels: dict[str, int] = {}
    for doc in pool_docs:
        text = _truncate_doc(doc.get("text") or doc.get("snippet") or "", JUDGE_CONTEXT_TOKENS)
        if not text.strip():
            continue
        phash = _prompt_hash(question, text, JUDGE_MODEL)
        cache_path = _judge_cache_path(phash)
        if not cache_path.exists():
            continue
        j = json.loads(cache_path.read_text(encoding="utf-8"))
        labels[doc["doc_id"]] = int(j.get("label", 0))
    return labels


def ndcg_at_k(ranked_doc_ids: list[str], labels: dict[str, int], k: int) -> float:
    def gain(label: int) -> float:
        return (2 ** label) - 1
    dcg = 0.0
    for i, did in enumerate(ranked_doc_ids[:k]):
        rel = labels.get(did, 0)
        dcg += gain(rel) / math.log2(i + 2)
    # ideal DCG over all judged docs for this question
    ideal_sorted = sorted(labels.values(), reverse=True)[:k]
    idcg = sum(gain(rel) / math.log2(i + 2) for i, rel in enumerate(ideal_sorted))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_doc_ids: list[str], labels: dict[str, int], k: int, threshold: int = 1) -> float:
    relevant = {did for did, lab in labels.items() if lab >= threshold}
    if not relevant:
        return 0.0
    retrieved = set(ranked_doc_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def average_precision(ranked_doc_ids: list[str], labels: dict[str, int], threshold: int = 1) -> float:
    relevant = {did for did, lab in labels.items() if lab >= threshold}
    if not relevant:
        return 0.0
    hits, sum_p = 0, 0.0
    for i, did in enumerate(ranked_doc_ids):
        if did in relevant:
            hits += 1
            sum_p += hits / (i + 1)
    return sum_p / len(relevant)


def run_metrics(questions: list[dict]) -> None:
    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    systems = ["lexical", "dense", "hybrid", "hybrid_rerank"]
    per_system_agg: dict[str, dict[str, list[float]]] = {
        s: {"ndcg@10": [], "recall@10": [], "recall@25": [], "map": []} for s in systems
    }
    per_question: list[dict] = []

    for q in questions:
        qid = q["question_id"]
        pool_path = _pool_path(qid)
        if not pool_path.exists():
            print(f"[metrics] {qid}: no pool, skipping")
            continue
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        labels = _load_judgments_for_question(qid, pool["question"], pool["pool"])
        if not labels:
            print(f"[metrics] {qid}: no judgments, skipping")
            continue

        q_row = {"question_id": qid, "n_pool": len(pool["pool"]), "n_judged": len(labels)}
        for s in systems:
            ranked = [r["doc_id"] for r in pool["per_system"].get(s, [])]
            ndcg = ndcg_at_k(ranked, labels, 10)
            r10 = recall_at_k(ranked, labels, 10)
            r25 = recall_at_k(ranked, labels, 25)
            mapv = average_precision(ranked, labels)
            per_system_agg[s]["ndcg@10"].append(ndcg)
            per_system_agg[s]["recall@10"].append(r10)
            per_system_agg[s]["recall@25"].append(r25)
            per_system_agg[s]["map"].append(mapv)
            q_row[f"{s}_ndcg@10"] = round(ndcg, 4)
            q_row[f"{s}_recall@25"] = round(r25, 4)
            q_row[f"{s}_map"] = round(mapv, 4)
        per_question.append(q_row)

    summary = {
        "judge_model": JUDGE_MODEL,
        "top_k": TOP_K,
        "judge_context_tokens": JUDGE_CONTEXT_TOKENS,
        "questions_evaluated": len(per_question),
        "per_system": {
            s: {
                metric: {
                    "mean": round(sum(vals) / len(vals), 4) if vals else 0.0,
                    "n": len(vals),
                }
                for metric, vals in per_system_agg[s].items()
            }
            for s in systems
        },
        "per_question": per_question,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = OUTPUT_DIR / "results" / f"{stamp}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = OUTPUT_DIR / "results" / f"{stamp}.csv"
    lines = ["system,metric,mean,n"]
    for s in systems:
        for metric, payload in summary["per_system"][s].items():
            lines.append(f"{s},{metric},{payload['mean']},{payload['n']}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[metrics] wrote {json_path}")
    print(f"[metrics] wrote {csv_path}")
    print()
    print("=== headline summary ===")
    for s in systems:
        print(f"  {s:18s}  nDCG@10={summary['per_system'][s]['ndcg@10']['mean']:.4f}"
              f"  Recall@25={summary['per_system'][s]['recall@25']['mean']:.4f}"
              f"  MAP={summary['per_system'][s]['map']['mean']:.4f}"
              f"  N={summary['per_system'][s]['ndcg@10']['n']}")


# ============================================================================
# main
# ============================================================================


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print()

    if PHASE_SANITY_CHECK:
        print("=== Phase 0: judge sanity check ===")
        if not run_sanity_check():
            print("[main] Sanity check failed -- aborting before Phase 1 to save budget.")
            sys.exit(2)
        print()
    if PHASE_RETRIEVE:
        print("=== Phase 1: retrieval ===")
        run_retrieval(questions)
        print()
    if PHASE_JUDGE:
        print("=== Phase 2: LLM-as-judge ===")
        run_judging(questions)
        print()
    if PHASE_METRICS:
        print("=== Phase 3: metrics ===")
        run_metrics(questions)
