"""Single rerunnable, ORACLE-FREE evaluation suite for CorpusAgent2.

One script runs every quantitative test the thesis reports and writes the
results straight into the LaTeX paper as \\input-able tables plus PNG plots.
Re-run it and the paper refreshes; nothing is hand-copied.

ORACLE-FREE BY DESIGN (supervisor constraint): no test uses a gold document
list, a gold sentence list, or any reference answer. Open-ended questions over
an arbitrarily large corpus have no definable ground-truth set, so every metric
here is computed without one:
  * Protocol A  -> LLM-as-judge graded relevance over a pooled candidate set
                   (nDCG@10, Recall@25, MAP). No gold doc ids.
  * Cross-judge -> the same pool re-scored by a second, different judge model;
                   Kendall's tau between the system rankings (judge robustness).
  * Protocol C  -> metamorphic robustness: paraphrase / entity-swap transforms,
                   top-k Jaccard overlap of retrieval. No gold; the metamorphic
                   relation itself is the oracle.

Everything is disk-cached and phase-toggleable, so a re-run after the first
full pass costs ~0 (retrieval pools and judge labels are reused). To refresh
just the tables/plots after editing rendering, set the RUN_* retrieval/judge
phases to False and keep RENDER on.

Run:
    python scripts/50_run_eval_suite.py
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ============================================================================
# CONFIG -- edit before running (no argparse, by repo convention)
# ============================================================================

# Question file: defaults to the 11-question set; override with the env var to run
# an alternative set (e.g. config/eval_questions_curated.json) without editing here.
QUESTIONS_PATH = (
    Path(os.environ["CORPUSAGENT2_EVAL_QUESTIONS"]).expanduser().resolve()
    if os.environ.get("CORPUSAGENT2_EVAL_QUESTIONS", "").strip()
    else PROJECT_ROOT / "config" / "eval_questions_11.json"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_suite"
LATEX_GEN_DIR = PROJECT_ROOT / "project_paper" / "LATEX" / "generated"

# OpenSearch is reachable from the host only via localhost (the os_news docker
# name resolves inside the container network only). Force it here so the lexical
# BM25 path is real and not silently degraded to TF-IDF.
os.environ.setdefault("CORPUSAGENT2_OPENSEARCH_URL", "https://localhost:9200")
os.environ.setdefault("CORPUSAGENT2_RETRIEVAL_BACKEND", "pgvector")

TOP_K = 25                  # pooled depth per system per question
JUDGE_CONTEXT_TOKENS = 1024 # doc truncation in judge prompt
SEED = 42

# Judge endpoint = the OpenAI-compatible endpoint already configured in .env.
# Judge models differ from the synthesis model (gpt-5.4) to keep judge and
# producer distinct. Retrieval uses no LLM at all, so there is no circularity
# in Protocol A regardless; the second judge exists to measure judge robustness.
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-5.4-nano-2026-03-17")
JUDGE_MODEL_2 = os.getenv("EVAL_JUDGE_MODEL_2", "gpt-5.2")

# Metamorphic transforms to apply (Protocol C). Kept light to bound cost.
C_TRANSFORMS = ["paraphrase", "entity_swap"]
C_SYSTEMS = ["hybrid_rerank", "dense"]  # systems to report robustness for

SYSTEMS = ["lexical", "dense", "hybrid", "hybrid_rerank"]
SYSTEM_LABEL = {
    "lexical": "Lexical (BM25)",
    "dense": "Dense (E5)",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + rerank",
}

# Phase toggles
def _phase(name: str, default: bool = True) -> bool:
    # Per-phase toggle; override with e.g. EVAL_RUN_METAMORPHIC=0 to skip a
    # cached phase (and its model load) on a targeted re-run.
    v = os.getenv(f"EVAL_RUN_{name}")
    return default if v is None else v.strip() not in ("0", "false", "False", "")

RUN_RETRIEVE = _phase("RETRIEVE")
RUN_JUDGE = _phase("JUDGE")
RUN_CROSS_JUDGE = _phase("CROSS_JUDGE")
RUN_METAMORPHIC = _phase("METAMORPHIC")
RUN_RETRIEVABILITY = _phase("RETRIEVABILITY")   # corpus-access bias (Gini), oracle-free, no LLM
RUN_PROTOCOL_B = _phase("PROTOCOL_B")           # claim-to-evidence faithfulness (NLI), oracle-free
RUN_SCALING = _phase("SCALING", default=False)          # RQ1 growing-corpus curve (opt-in)
RUN_SCALING_JUDGE = _phase("SCALING_JUDGE")             # judge scaling pools (within RUN_SCALING)
RUN_SCALING_ANN = _phase("SCALING_ANN", default=False)  # RQ4 ANN benchmark (opt-in)
RENDER = _phase("RENDER")   # write LaTeX tables + plots

# Protocol B: synthesise a grounded answer from the system's OWN retrieved
# evidence, decompose it into atomic claims, and NLI-score each claim against
# that evidence. No gold answer, no gold evidence -- the system's retrieval is
# the only evidence source.
B_SYNTH_MODEL = os.getenv("EVAL_SYNTH_MODEL", "gpt-5.4-2026-03-05")
B_EVIDENCE_DOCS = 8      # top hybrid+rerank docs used as cited evidence per question
B_NLI_MODEL = os.getenv("CORPUSAGENT2_NLI_MODEL", "FacebookAI/roberta-large-mnli")
FAMILIES = ["A", "B", "C", "D", "E", "F"]
FAMILY_LABEL = {"A": "A. Distribution", "B": "B. Comparative", "C": "C. Temporal framing",
                "D": "D. Prediction", "E": "E. Metadata-cond.", "F": "F. Corpus + external"}

# Retrievability (Azzopardi & Vinay): how equitably can the retriever reach the
# whole corpus? Simulated queries are drawn from the corpus; the Gini of the
# per-document retrievability distribution measures access bias. No gold, no LLM.
RETR_N_QUERIES = 120
RETR_TOPK = 20
RETR_SYSTEMS = ["lexical", "dense", "hybrid"]

# Scaling curve (RQ1) + ANN architecture benchmark (RQ4): seeded growing-corpus
# subsets of the 624k embedded universe (docs/scaling_curve_experiment.md).
# Both phases are OPT-IN (first run is overnight-scale): EVAL_RUN_SCALING=1 and/or
# EVAL_RUN_SCALING_ANN=1. Everything is per-condition resumable and reuses the
# same judge cache as Protocol A (pool texts are derived identically on purpose).
SCALING_SIZES = [int(s) for s in os.getenv("EVAL_SCALING_SIZES", "10000,50000,100000,250000").split(",") if s.strip()]
SCALING_REPS = int(os.getenv("EVAL_SCALING_REPS", "5"))
SCALING_INCLUDE_FULL = os.getenv("EVAL_SCALING_FULL", "1").strip() not in ("0", "false", "False")
SCALING_DIR = OUTPUT_DIR / "scaling"
SCALING_OS_INDEX_PREFIX = "ca2-scaling"  # per-subset BM25 indexes: build -> measure -> drop
ANN_N_DOC_QUERIES = 1000        # sampled doc vectors added to the question queries (latency/recall stability)
ANN_PG_LATENCY_QUERIES = 200    # pgvector latency/recall measured on this many queries (server round-trips)
ANN_RECALL_K = (10, 25)

# ============================================================================
# Judge prompt (immutable; editing it invalidates cached judgments)
# ============================================================================

JUDGE_SYSTEM_PROMPT = (
    "You are an expert relevance assessor for an information-retrieval "
    "evaluation. Given a research question about a news corpus and a candidate "
    "document, judge how relevant the document is to answering the question on "
    "a 4-point graded scale:\n"
    "  3 = Highly Relevant     (directly answers a core part)\n"
    "  2 = Relevant            (clearly useful evidence)\n"
    "  1 = Marginally Relevant (mentions the topic but tangential)\n"
    "  0 = Not Relevant        (off-topic or unrelated)\n"
    'Return JSON exactly of the form {"label": <int 0..3>, "reason": <str>}. '
    "No prose outside the JSON object."
)
JUDGE_USER_TEMPLATE = (
    "QUESTION:\n{question}\n\nDOCUMENT (truncated):\n{document}\n\n"
    "Output the JSON object now."
)


# ============================================================================
# small utils
# ============================================================================

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, tokens: int) -> str:
    return text[: tokens * 4]


def _judge_hash(question: str, document: str, model: str) -> str:
    h = hashlib.sha256()
    for part in (JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE, model, question, document):
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


def _load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def _build_runtime():
    from corpusagent2.agent_runtime import AgentRuntime, AgentRuntimeConfig
    cfg = AgentRuntimeConfig(project_root=PROJECT_ROOT, outputs_root=PROJECT_ROOT / "outputs" / "agent_runtime")
    return AgentRuntime(config=cfg)


def _judge_client():
    from corpusagent2.llm_provider import LLMProviderConfig, OpenAICompatibleLLMClient
    return OpenAICompatibleLLMClient(LLMProviderConfig.from_env())


# ============================================================================
# metrics (graded, oracle-free: relevance comes from the judge, not a gold set)
# ============================================================================

def ndcg_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    def gain(l: int) -> float:
        return (2 ** l) - 1
    dcg = sum(gain(labels.get(d, 0)) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = sum(gain(r) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], labels: dict[str, int], k: int, thr: int = 1) -> float:
    rel = {d for d, l in labels.items() if l >= thr}
    if not rel:
        return 0.0
    return len(set(ranked[:k]) & rel) / len(rel)


def average_precision(ranked: list[str], labels: dict[str, int], thr: int = 1) -> float:
    rel = {d for d, l in labels.items() if l >= thr}
    if not rel:
        return 0.0
    hits, s = 0, 0.0
    for i, d in enumerate(ranked):
        if d in rel:
            hits += 1
            s += hits / (i + 1)
    return s / len(rel)


def bootstrap_ci(values: list[float], n: int = 2000) -> tuple[float, float, float]:
    import random
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(SEED)
    means = []
    m = len(values)
    for _ in range(n):
        sample = [values[rng.randrange(m)] for _ in range(m)]
        means.append(sum(sample) / m)
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n)]
    return sum(values) / len(values), lo, hi


def kendall_tau(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    conc = disc = 0
    for i, j in itertools.combinations(range(n), 2):
        s = (a[i] - a[j]) * (b[i] - b[j])
        if s > 0:
            conc += 1
        elif s < 0:
            disc += 1
    denom = conc + disc
    return (conc - disc) / denom if denom else 0.0


# ============================================================================
# Phase 1: retrieval (pools per question) -- oracle-free, no gold
# ============================================================================

def phase_retrieve(questions: list[dict]) -> None:
    backend = _build_runtime().search_backend
    pool_dir = OUTPUT_DIR / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    modes = [("lexical", False), ("dense", False), ("hybrid", False), ("hybrid_rerank", True)]

    for q in questions:
        qid, question = q["question_id"], q["raw_question"]
        out = pool_dir / f"{qid}.json"
        if out.exists():
            print(f"[retrieve] {qid}: cached")
            continue
        per_system: dict[str, list[dict]] = {}
        pool: dict[str, dict] = {}
        for name, rr in modes:
            mode = "hybrid" if name == "hybrid_rerank" else name
            t0 = time.monotonic()
            try:
                rows = backend.search(query=question, top_k=TOP_K, retrieval_mode=mode,
                                      use_rerank=rr, rerank_top_k=TOP_K)
            except Exception as exc:
                print(f"[retrieve] {qid}/{name}: ERROR {type(exc).__name__}: {exc}")
                rows = []
            ids = [str(r.get("doc_id") or "") for r in rows]
            per_system[name] = [{"doc_id": d, "rank": i + 1} for i, d in enumerate(ids) if d]
            for i, d in enumerate(ids):
                if d and d not in pool:
                    r = rows[i]
                    pool[d] = {"doc_id": d, "title": str(r.get("title", "")),
                               "text": str(r.get("text", r.get("snippet", "")))}
            print(f"[retrieve] {qid}/{name}: {len(ids)} hits in {time.monotonic()-t0:.1f}s")
        out.write_text(json.dumps({"question_id": qid, "question": question,
                                   "per_system": per_system, "pool": list(pool.values())},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[retrieve] {qid}: pool={len(pool)} unique -> {out.name}")


# ============================================================================
# Phase 2: judge (cached per model) -- LLM-as-judge graded relevance
# ============================================================================

def _judge_pool(client, pool: dict, model: str, counts: dict) -> None:
    """Judge every unjudged (question, doc) pair of one pool file into the shared cache."""
    cache_dir = OUTPUT_DIR / "judge_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    qid = pool["question_id"]
    for doc in pool["pool"]:
        text = _truncate(doc.get("text") or "", JUDGE_CONTEXT_TOKENS)
        if not text.strip():
            continue
        h = _judge_hash(pool["question"], text, model)
        cpath = cache_dir / f"{h}.json"
        if cpath.exists():
            counts["cached"] += 1
            continue
        user = JUDGE_USER_TEMPLATE.format(question=pool["question"], document=text)
        try:
            parsed = client.complete_json(
                [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                 {"role": "user", "content": user}],
                model=model, temperature=0.0)
            label = int(parsed.get("label", 0))
            if not 0 <= label <= 3:
                raise ValueError(f"label {label}")
        except Exception as exc:
            counts["failed"] += 1
            print(f"[judge:{model}] {qid}/{doc['doc_id'][:8]}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        cpath.write_text(json.dumps({"question_id": qid, "doc_id": doc["doc_id"],
                                     "label": label, "judge_model": model,
                                     "judged_at": _now()}, ensure_ascii=False), encoding="utf-8")
        counts["new"] += 1
        if counts["new"] % 25 == 0:
            print(f"[judge:{model}] {counts['new']} new, {counts['cached']} cached, {counts['failed']} failed")


def phase_judge(questions: list[dict], model: str) -> None:
    client = _judge_client()
    counts = {"new": 0, "cached": 0, "failed": 0}
    for q in questions:
        pool_path = OUTPUT_DIR / "pool" / f"{q['question_id']}.json"
        if not pool_path.exists():
            continue
        _judge_pool(client, json.loads(pool_path.read_text(encoding="utf-8")), model, counts)
    print(f"[judge:{model}] DONE: {counts['new']} new, {counts['cached']} cached, {counts['failed']} failed")


def _labels_for(qid_question_pool: dict, model: str) -> dict[str, int]:
    out: dict[str, int] = {}
    cache_dir = OUTPUT_DIR / "judge_cache"
    for doc in qid_question_pool["pool"]:
        text = _truncate(doc.get("text") or "", JUDGE_CONTEXT_TOKENS)
        if not text.strip():
            continue
        h = _judge_hash(qid_question_pool["question"], text, model)
        cpath = cache_dir / f"{h}.json"
        if cpath.exists():
            out[doc["doc_id"]] = int(json.loads(cpath.read_text(encoding="utf-8")).get("label", 0))
    return out


# ============================================================================
# Phase 3: Protocol A metrics + cross-judge tau
# ============================================================================

def compute_protocol_a(questions: list[dict], model: str) -> dict:
    agg = {s: {"ndcg@10": [], "recall@25": [], "map": []} for s in SYSTEMS}
    per_q = []
    for q in questions:
        pool_path = OUTPUT_DIR / "pool" / f"{q['question_id']}.json"
        if not pool_path.exists():
            continue
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        labels = _labels_for(pool, model)
        if not labels:
            continue
        row = {"question_id": q["question_id"], "family": q.get("family", ""),
               "n_pool": len(pool["pool"]), "n_judged": len(labels)}
        for s in SYSTEMS:
            ranked = [r["doc_id"] for r in pool["per_system"].get(s, [])]
            nd = ndcg_at_k(ranked, labels, 10)
            r25 = recall_at_k(ranked, labels, 25)
            mp = average_precision(ranked, labels)
            agg[s]["ndcg@10"].append(nd)
            agg[s]["recall@25"].append(r25)
            agg[s]["map"].append(mp)
            row[f"{s}_ndcg@10"] = round(nd, 4)
        per_q.append(row)
    summary = {"judge_model": model, "top_k": TOP_K, "questions_evaluated": len(per_q),
               "per_system": {}, "per_question": per_q, "generated_at": _now()}
    for s in SYSTEMS:
        summary["per_system"][s] = {}
        for metric, vals in agg[s].items():
            mean, lo, hi = bootstrap_ci(vals)
            summary["per_system"][s][metric] = {"mean": round(mean, 4), "ci_low": round(lo, 4),
                                                 "ci_high": round(hi, 4), "n": len(vals)}
    return summary


def compute_cross_judge(questions: list[dict], summary1: dict, model2: str) -> dict:
    summary2 = compute_protocol_a(questions, model2)
    taus = {}
    for metric in ["ndcg@10", "recall@25", "map"]:
        a = [summary1["per_system"][s][metric]["mean"] for s in SYSTEMS]
        b = [summary2["per_system"][s][metric]["mean"] for s in SYSTEMS]
        taus[metric] = round(kendall_tau(a, b), 4)
    return {"judge_1": summary1["judge_model"], "judge_2": model2,
            "kendall_tau": taus, "summary_2": summary2, "generated_at": _now()}


# ============================================================================
# Phase 4: Protocol C metamorphic robustness (oracle-free)
# ============================================================================

C_PROMPTS = {
    "paraphrase": "Rewrite this corpus-analysis question as a natural paraphrase that keeps the same meaning and the same entities. Return only the rewritten question.",
    "entity_swap": "Rewrite this corpus-analysis question, replacing its main named entity with a different but topically comparable named entity (e.g. one newspaper for another, one politician for another). Return only the rewritten question.",
}


def jaccard(a: list[str], b: list[str], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def phase_metamorphic(questions: list[dict]) -> dict:
    backend = _build_runtime().search_backend
    client = _judge_client()
    tdir = OUTPUT_DIR / "metamorphic"
    tdir.mkdir(parents=True, exist_ok=True)

    def retrieve(question: str, system: str) -> list[str]:
        mode = "hybrid" if system == "hybrid_rerank" else system
        rr = system == "hybrid_rerank"
        try:
            rows = backend.search(query=question, top_k=TOP_K, retrieval_mode=mode,
                                  use_rerank=rr, rerank_top_k=TOP_K)
            return [str(r.get("doc_id") or "") for r in rows if r.get("doc_id")]
        except Exception as exc:
            print(f"[meta] retrieve ERROR: {exc}")
            return []

    results = {t: {s: [] for s in C_SYSTEMS} for t in C_TRANSFORMS}
    for q in questions:
        qid, question = q["question_id"], q["raw_question"]
        cpath = tdir / f"{qid}.json"
        if cpath.exists():
            data = json.loads(cpath.read_text(encoding="utf-8"))
        else:
            data = {"question_id": qid, "question": question, "transforms": {}, "retrieval": {}}
            # original retrieval
            data["retrieval"]["original"] = {s: retrieve(question, s) for s in C_SYSTEMS}
            for t in C_TRANSFORMS:
                try:
                    rewritten = client.complete(
                        [{"role": "system", "content": C_PROMPTS[t]},
                         {"role": "user", "content": question}],
                        model=JUDGE_MODEL, temperature=0.0).strip().strip('"')
                except Exception as exc:
                    print(f"[meta] {qid}/{t}: transform ERROR {exc}")
                    rewritten = question
                data["transforms"][t] = rewritten
                data["retrieval"][t] = {s: retrieve(rewritten, s) for s in C_SYSTEMS}
            cpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[meta] {qid}: done")
        for t in C_TRANSFORMS:
            for s in C_SYSTEMS:
                orig = data["retrieval"]["original"][s]
                trans = data["retrieval"][t][s]
                results[t][s].append(jaccard(orig, trans, TOP_K))

    summary = {"transforms": C_TRANSFORMS, "systems": C_SYSTEMS, "n": len(questions),
               "jaccard": {}, "generated_at": _now()}
    # Expected: high overlap for paraphrase (invariance), low for entity_swap (sensitivity)
    for t in C_TRANSFORMS:
        summary["jaccard"][t] = {}
        for s in C_SYSTEMS:
            mean, lo, hi = bootstrap_ci(results[t][s])
            summary["jaccard"][t][s] = {"mean": round(mean, 4), "ci_low": round(lo, 4),
                                        "ci_high": round(hi, 4), "n": len(results[t][s])}
    return summary


# ============================================================================
# Phase 4a: Protocol B -- claim-to-evidence faithfulness (oracle-free)
# ============================================================================

B_SYNTH_PROMPT = (
    "You answer open-ended questions about a news corpus using ONLY the provided "
    "documents. Write a concise factual answer of 3-6 sentences, grounded strictly "
    "in the documents. Do not introduce facts that are not supported by them."
)
B_CLAIMS_PROMPT = (
    "Decompose the following answer into a list of atomic, self-contained factual "
    "claims (resolve pronouns; one assertion each). Return JSON exactly of the form "
    '{"claims": ["...", "..."]}.'
)


def phase_protocol_b(questions: list[dict]) -> dict:
    from corpusagent2.faithfulness import NLIVerifier, evaluate_claims_with_nli
    client = _judge_client()
    ans_dir = OUTPUT_DIR / "protocol_b"
    ans_dir.mkdir(parents=True, exist_ok=True)

    # Phase B1: synthesise answers + extract claims (LLM, cached per question).
    for q in questions:
        qid = q["question_id"]
        out = ans_dir / f"{qid}.json"
        if out.exists():
            continue
        pool_path = OUTPUT_DIR / "pool" / f"{qid}.json"
        if not pool_path.exists():
            continue
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        text_by_id = {d["doc_id"]: (d.get("text") or "") for d in pool["pool"]}
        ranked = [r["doc_id"] for r in pool["per_system"].get("hybrid_rerank", [])][:B_EVIDENCE_DOCS]
        cited = [{"doc_id": d, "text": _truncate(text_by_id.get(d, ""), 400)} for d in ranked if text_by_id.get(d)]
        if not cited:
            continue
        context = "\n\n".join(f"[DOC {i+1}] {c['text']}" for i, c in enumerate(cited))
        try:
            answer = client.complete(
                [{"role": "system", "content": B_SYNTH_PROMPT},
                 {"role": "user", "content": f"QUESTION:\n{pool['question']}\n\nDOCUMENTS:\n{context}\n\nAnswer:"}],
                model=B_SYNTH_MODEL, temperature=0.0).strip()
            parsed = client.complete_json(
                [{"role": "system", "content": B_CLAIMS_PROMPT},
                 {"role": "user", "content": answer}],
                model=B_SYNTH_MODEL, temperature=0.0)
            claims = [str(c) for c in parsed.get("claims", []) if str(c).strip()]
        except Exception as exc:
            print(f"[protoB] {qid}: synth/claims ERROR {type(exc).__name__}: {str(exc)[:80]}")
            continue
        out.write_text(json.dumps({"question_id": qid, "family": q.get("family", ""),
                                   "question": pool["question"], "answer": answer,
                                   "cited_doc_ids": [c["doc_id"] for c in cited],
                                   "evidence_text_by_id": {c["doc_id"]: c["text"] for c in cited},
                                   "claims": claims}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[protoB] {qid}: {len(claims)} claims from {len(cited)} cited docs")

    # Phase B2: NLI scoring against the system's own cited evidence.
    verifier = NLIVerifier(model_id=B_NLI_MODEL, device="cpu")
    per_family = {f: {"total": 0, "entailed": 0, "contradicted": 0, "unsupported": 0} for f in FAMILIES}
    overall = {"total": 0, "entailed": 0, "contradicted": 0, "unsupported": 0}
    per_question = []
    for q in questions:
        qid = q["question_id"]
        apath = ans_dir / f"{qid}.json"
        if not apath.exists():
            continue
        data = json.loads(apath.read_text(encoding="utf-8"))
        fam = data.get("family", "")
        claims = [{"claim_id": f"{qid}_c{i}", "claim": c, "evidence_doc_ids": data["cited_doc_ids"]}
                  for i, c in enumerate(data["claims"])]
        if not claims:
            continue
        _, summary = evaluate_claims_with_nli(verifier, claims, data["evidence_text_by_id"])
        n = summary["total_claims"]
        ent = summary["entailed_claims"]
        con = int(round(summary["contradiction_rate"] * n))
        uns = int(round(summary["unsupported_rate"] * n))
        for bucket in (per_family.get(fam), overall):
            if bucket is None:
                continue
            bucket["total"] += n
            bucket["entailed"] += ent
            bucket["contradicted"] += con
            bucket["unsupported"] += uns
        per_question.append({"question_id": qid, "family": fam, "n_claims": n,
                             "faithfulness": round(summary["faithfulness"], 3),
                             "contradiction_rate": round(summary["contradiction_rate"], 3),
                             "unsupported_rate": round(summary["unsupported_rate"], 3)})
        print(f"[protoB] {qid} ({fam}): faith={summary['faithfulness']:.2f} "
              f"contra={summary['contradiction_rate']:.2f} unsup={summary['unsupported_rate']:.2f} (n={n})")

    def rates(b: dict) -> dict:
        t = b["total"]
        return {"n_claims": t,
                "faithfulness": round(b["entailed"] / t, 3) if t else 0.0,
                "unsupported": round(b["unsupported"] / t, 3) if t else 0.0,
                "contradiction": round(b["contradicted"] / t, 3) if t else 0.0}

    return {"nli_model": B_NLI_MODEL, "synth_model": B_SYNTH_MODEL,
            "per_family": {f: rates(per_family[f]) for f in FAMILIES},
            "overall": rates(overall), "per_question": per_question, "generated_at": _now()}


# ============================================================================
# Phase 4b: retrievability bias (Gini) -- oracle-free, no LLM, no gold
# ============================================================================

def gini(values: list[float]) -> float:
    xs = sorted(values)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def phase_retrievability(questions: list[dict]) -> dict:
    import pandas as pd
    cache = OUTPUT_DIR / "retrievability_counts.json"
    meta = pd.read_parquet(PROJECT_ROOT / "data" / "indices" / "doc_metadata.parquet",
                           columns=["doc_id", "title"])
    n_corpus = len(meta)
    rng = random.Random(SEED)
    titles = [str(t) for t in meta["title"].tolist() if isinstance(t, str) and 3 <= len(t.split()) <= 15]
    sim_queries = rng.sample(titles, min(RETR_N_QUERIES, len(titles)))

    if cache.exists():
        counts = {s: collections.Counter(json.loads(cache.read_text())[s]) for s in RETR_SYSTEMS}
        print(f"[retr] using cached counts for {len(sim_queries)} simulated queries")
    else:
        backend = _build_runtime().search_backend
        counts = {s: collections.Counter() for s in RETR_SYSTEMS}
        for i, q in enumerate(sim_queries):
            for s in RETR_SYSTEMS:
                mode = "hybrid" if s == "hybrid_rerank" else s
                try:
                    rows = backend.search(query=q, top_k=RETR_TOPK, retrieval_mode=mode,
                                          use_rerank=False, rerank_top_k=RETR_TOPK)
                    for r in rows:
                        d = str(r.get("doc_id") or "")
                        if d:
                            counts[s][d] += 1
                except Exception as exc:
                    print(f"[retr] q{i}/{s}: {type(exc).__name__}: {str(exc)[:60]}")
            if (i + 1) % 20 == 0:
                print(f"[retr] {i+1}/{len(sim_queries)} simulated queries done")
        cache.write_text(json.dumps({s: dict(counts[s]) for s in RETR_SYSTEMS}), encoding="utf-8")

    summary = {"n_queries": len(sim_queries), "top_k": RETR_TOPK, "n_corpus": n_corpus,
               "systems": {}, "generated_at": _now()}
    for s in RETR_SYSTEMS:
        c = counts[s]
        # retrievability vector over the WHOLE corpus: unreached docs contribute 0.
        vec = list(c.values()) + [0] * (n_corpus - len(c))
        summary["systems"][s] = {
            "gini": round(gini(vec), 4),
            "unique_docs_reached": len(c),
            "corpus_coverage": round(len(c) / n_corpus, 5),
        }
    return summary


# ============================================================================
# Phase S: scaling curve (RQ1) + ANN architecture benchmark (RQ4)
# Design + rationale: docs/scaling_curve_experiment.md. Oracle-free throughout:
# RQ1 reuses the Protocol A judge cache (pool texts derived identically); RQ4's
# ground truth is the exact flat scan over the same vectors (self-referential).
# ============================================================================

_SCALING_STATE: dict = {}


def _scaling_conditions() -> list[dict]:
    conds = [{"name": f"n{size}_s{rep}", "size": size, "rep": rep}
             for size in SCALING_SIZES for rep in range(1, SCALING_REPS + 1)]
    if SCALING_INCLUDE_FULL:
        conds.append({"name": "full", "size": None, "rep": 0})
    return conds


def _scaling_universe():
    """(embeddings float32 in RAM, doc_ids) -- the 624k dense universe, loaded once."""
    if "universe" not in _SCALING_STATE:
        import joblib
        import numpy as np
        dense_dir = PROJECT_ROOT / "data" / "indices" / "dense"
        emb = np.ascontiguousarray(np.load(dense_dir / "dense_embeddings.npy", mmap_mode="r"), dtype=np.float32)
        doc_ids = [str(d) for d in joblib.load(dense_dir / "dense_doc_ids.joblib")]
        if emb.shape[0] != len(doc_ids):
            raise RuntimeError(f"dense matrix rows ({emb.shape[0]}) != doc_ids ({len(doc_ids)})")
        _SCALING_STATE["universe"] = (emb, doc_ids)
    return _SCALING_STATE["universe"]


def _scaling_docs() -> dict[str, dict]:
    """doc_id -> {title,text,published_at,source}; same source of truth as CorpusRuntime.doc_lookup."""
    if "docs" not in _SCALING_STATE:
        import pandas as pd
        meta = pd.read_parquet(PROJECT_ROOT / "data" / "indices" / "doc_metadata.parquet")
        _SCALING_STATE["docs"] = {
            str(row.doc_id): {
                "title": str(getattr(row, "title", "")),
                "text": str(getattr(row, "text", "")),
                "published_at": str(getattr(row, "published_at", "")),
                "source": str(getattr(row, "source", "")),
            }
            for row in meta.itertuples(index=False)
        }
    return _SCALING_STATE["docs"]


def phase_scaling_subsets() -> None:
    import numpy as np
    _, doc_ids = _scaling_universe()
    sub_dir = SCALING_DIR / "subsets"
    sub_dir.mkdir(parents=True, exist_ok=True)
    universe_sha = hashlib.sha256("\n".join(doc_ids).encode("utf-8")).hexdigest()[:16]
    manifest_path = sub_dir / "manifest.json"
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("universe_sha", "")
        if recorded and recorded != universe_sha:
            raise RuntimeError(
                f"scaling subsets were drawn against a different corpus universe "
                f"({recorded} != {universe_sha}); delete {sub_dir} to redraw")
    n = len(doc_ids)
    for cond in _scaling_conditions():
        if cond["size"] is None:
            continue
        path = sub_dir / f"{cond['name']}.npy"
        if path.exists():
            continue
        # Seed strings are deterministic across runs and Python versions.
        rng = random.Random(f"{SEED}:{cond['size']}:{cond['rep']}")
        np.save(path, np.array(sorted(rng.sample(range(n), cond["size"])), dtype=np.int64))
        print(f"[scaling] drew subset {cond['name']} ({cond['size']:,} docs)")
    manifest_path.write_text(json.dumps({
        "master_seed": SEED, "sizes": SCALING_SIZES, "reps": SCALING_REPS,
        "universe_rows": n, "universe_sha": universe_sha, "generated_at": _now()},
        indent=2), encoding="utf-8")


# --- per-subset OpenSearch index (mapping identical to scripts/21_bulk_index_opensearch.py) ---

_OS_SUBSET_MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {"properties": {
        "doc_id": {"type": "keyword"}, "id": {"type": "keyword"},
        "title": {"type": "text"}, "text": {"type": "text"},
        "body": {"type": "text"}, "content": {"type": "text"},
        "published_at": {"type": "date",
                         "format": "strict_date_optional_time||epoch_millis||yyyy-MM-dd HH:mm:ss||yyyy-MM-dd"},
        "source": {"type": "keyword"}, "source_domain": {"type": "keyword"},
    }},
}


def _os_config(index_name: str):
    from corpusagent2.agent_backends import OpenSearchConfig
    base = OpenSearchConfig.from_env()
    return OpenSearchConfig(base_url=base.base_url, index_name=index_name,
                            username=base.username, password=base.password,
                            verify_ssl=base.verify_ssl, timeout_s=base.timeout_s)


def _os_request(method: str, cfg, path: str = "", **kwargs):
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    auth = (cfg.username, cfg.password) if (cfg.username or cfg.password) else None
    url = f"{cfg.base_url.rstrip('/')}/{cfg.index_name}{path}"
    return requests.request(method, url, auth=auth, verify=cfg.verify_ssl,
                            timeout=max(cfg.timeout_s, 300.0), **kwargs)


def _os_build_subset_index(cfg, subset_doc_ids: list[str]) -> None:
    docs = _scaling_docs()
    _os_request("DELETE", cfg)  # clean slate; 404 on first build is fine
    resp = _os_request("PUT", cfg, json=_OS_SUBSET_MAPPING)
    if resp.status_code not in {200, 201}:
        resp.raise_for_status()
    batch: list[str] = []

    def flush() -> None:
        if not batch:
            return
        resp = _os_request("POST", cfg, "/_bulk", data="\n".join(batch) + "\n",
                           headers={"Content-Type": "application/x-ndjson"})
        resp.raise_for_status()
        if resp.json().get("errors"):
            raise RuntimeError(f"OpenSearch bulk indexing into {cfg.index_name} returned item-level errors")
        batch.clear()

    for doc_id in subset_doc_ids:
        row = docs.get(doc_id)
        if row is None:
            continue
        batch.append(json.dumps({"index": {"_id": doc_id}}))
        payload = {"doc_id": doc_id, "id": doc_id, "title": row["title"],
                   "text": row["text"], "body": row["text"], "content": row["text"],
                   "source": row["source"], "source_domain": row["source"]}
        published_at = row["published_at"]
        if published_at and published_at[:4].isdigit():
            payload["published_at"] = published_at
        batch.append(json.dumps(payload, ensure_ascii=True))
        if len(batch) >= 4000:
            flush()
    flush()
    _os_request("POST", cfg, "/_refresh").raise_for_status()


# --- dense scoring: per-doc, so full-corpus scores masked to a subset are EXACT ---

def _scaling_query_embedding(question: str):
    import numpy as np
    cache = _SCALING_STATE.setdefault("qemb", {})
    if question not in cache:
        from corpusagent2.model_config import dense_model_id_from_env
        from corpusagent2.retrieval import _load_sentence_transformer
        model, _dev = _load_sentence_transformer(model_id=dense_model_id_from_env(), device=None)
        cache[question] = model.encode([question], convert_to_numpy=True,
                                       normalize_embeddings=True).astype(np.float32)
    return cache[question]


def _scaling_query_scores(question: str):
    cache = _SCALING_STATE.setdefault("qscores", {})
    if question not in cache:
        emb, _ = _scaling_universe()
        cache[question] = (_scaling_query_embedding(question) @ emb.T).ravel()
    return cache[question]


def _scaling_condition_pools(questions: list[dict], cond: dict, subset_idx, os_cfg) -> None:
    """Run the four Protocol A systems against one condition and write its pools.

    Pool schema and text derivation (raw text[:360]) are identical to
    phase_retrieve on purpose: judge-cache hashes match, so judgments are shared
    with the main Protocol A run and across conditions.
    """
    import numpy as np
    from corpusagent2.agent_backends import OpenSearchBackend
    from corpusagent2.retrieval import RetrievalResult, reciprocal_rank_fusion, rerank_cross_encoder
    from corpusagent2.runtime_context import DEFAULT_RERANK_MODEL_ID

    emb, doc_ids = _scaling_universe()
    docs = _scaling_docs()
    lex_backend = OpenSearchBackend(os_cfg)
    candidate_limit = max(TOP_K * 5, 50)  # mirrors HybridSearchBackend lexical/dense limits
    pool_dir = SCALING_DIR / "pool" / cond["name"]
    pool_dir.mkdir(parents=True, exist_ok=True)

    for q in questions:
        qid, question = q["question_id"], q["raw_question"]
        out = pool_dir / f"{qid}.json"
        if out.exists():
            continue
        t0 = time.monotonic()
        # lexical: per-subset BM25 (collection statistics belong to the subset)
        try:
            raw = lex_backend.search(query=question, top_k=candidate_limit, retrieval_mode="lexical")
        except Exception as exc:
            print(f"[scaling] {cond['name']}/{qid}: lexical ERROR {type(exc).__name__}: {exc}")
            raw = []
        lex = [RetrievalResult(doc_id=str(r["doc_id"]), rank=i + 1, score=float(r.get("score", 0.0)),
                               score_components={"lexical": float(r.get("score", 0.0))})
               for i, r in enumerate(raw)
               if float(r.get("score", 0.0)) > 0 and str(r.get("doc_id", "")) in docs]
        # dense: exact masking of per-doc scores (no re-embedding, no index effects)
        scores = _scaling_query_scores(question)
        sub_scores = scores if subset_idx is None else scores[subset_idx]
        limit = min(candidate_limit, sub_scores.shape[0])
        best = np.argpartition(sub_scores, -limit)[-limit:]
        order = best[np.argsort(sub_scores[best])[::-1]]
        dense = [RetrievalResult(
                     doc_id=doc_ids[int(pos) if subset_idx is None else int(subset_idx[int(pos)])],
                     rank=rank, score=float(sub_scores[int(pos)]),
                     score_components={"dense": float(sub_scores[int(pos)])})
                 for rank, pos in enumerate(order, start=1)]
        fused = reciprocal_rank_fusion({"lexical": lex, "dense": dense}, k=60)[:TOP_K]
        # rerank the fused top-k, exactly like HybridSearchBackend.search
        text_by_id = {r.doc_id: f"{docs[r.doc_id]['title']} {docs[r.doc_id]['text']}".strip()
                      for r in fused if r.doc_id in docs}
        try:
            reranked = rerank_cross_encoder(query=question, candidates=fused,
                                            doc_text_by_id=text_by_id,
                                            model_id=DEFAULT_RERANK_MODEL_ID, top_k=TOP_K)
        except Exception as exc:
            print(f"[scaling] {cond['name']}/{qid}: rerank ERROR {type(exc).__name__}: {exc}")
            reranked = fused
        systems = {"lexical": lex[:TOP_K], "dense": dense[:TOP_K],
                   "hybrid": fused, "hybrid_rerank": reranked}
        per_system = {name: [{"doc_id": r.doc_id, "rank": i + 1} for i, r in enumerate(rs)]
                      for name, rs in systems.items()}
        pool: dict[str, dict] = {}
        for name in SYSTEMS:
            for r in systems[name]:
                if r.doc_id not in pool:
                    row = docs.get(r.doc_id, {})
                    pool[r.doc_id] = {"doc_id": r.doc_id, "title": row.get("title", ""),
                                      "text": str(row.get("text", ""))[:360]}
        out.write_text(json.dumps({"question_id": qid, "question": question,
                                   "condition": cond["name"], "size": cond["size"],
                                   "per_system": per_system, "pool": list(pool.values())},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[scaling] {cond['name']}/{qid}: pool={len(pool)} in {time.monotonic()-t0:.1f}s")


def phase_scaling_retrieve(questions: list[dict]) -> None:
    import numpy as np
    for cond in _scaling_conditions():
        pool_dir = SCALING_DIR / "pool" / cond["name"]
        if all((pool_dir / f"{q['question_id']}.json").exists() for q in questions):
            print(f"[scaling] {cond['name']}: pools cached")
            continue
        if cond["size"] is None:
            # full 624k point: the production BM25 index IS the per-"subset" index
            cfg = _os_config(os.getenv("CORPUSAGENT2_OPENSEARCH_INDEX", "article-corpus-opensearch"))
            _scaling_condition_pools(questions, cond, None, cfg)
            continue
        subset_idx = np.load(SCALING_DIR / "subsets" / f"{cond['name']}.npy")
        _, doc_ids = _scaling_universe()
        cfg = _os_config(f"{SCALING_OS_INDEX_PREFIX}-{cond['name'].replace('_', '-')}")
        t0 = time.monotonic()
        _os_build_subset_index(cfg, [doc_ids[int(i)] for i in subset_idx])
        print(f"[scaling] {cond['name']}: BM25 index built in {time.monotonic()-t0:.0f}s")
        try:
            _scaling_condition_pools(questions, cond, subset_idx, cfg)
        finally:
            # build -> measure -> drop: never keep 20 subset indexes on the 90 GB VM
            _os_request("DELETE", cfg)


def phase_scaling_judge(model: str) -> None:
    client = _judge_client()
    counts = {"new": 0, "cached": 0, "failed": 0}
    for pool_path in sorted((SCALING_DIR / "pool").glob("*/*.json")):
        _judge_pool(client, json.loads(pool_path.read_text(encoding="utf-8")), model, counts)
    print(f"[judge:{model}] scaling DONE: {counts['new']} new, {counts['cached']} cached, {counts['failed']} failed")


def compute_scaling_rq1(questions: list[dict], model: str) -> dict:
    """Per-condition Protocol A metrics, then mean +- sample std across seeds per size."""
    _, doc_ids = _scaling_universe()
    per_condition: dict[str, dict] = {}
    for cond in _scaling_conditions():
        agg = {s: {"ndcg@10": [], "recall@25": [], "map": []} for s in SYSTEMS}
        n_q = 0
        for q in questions:
            pool_path = SCALING_DIR / "pool" / cond["name"] / f"{q['question_id']}.json"
            if not pool_path.exists():
                continue
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
            labels = _labels_for(pool, model)
            if not labels:
                continue
            n_q += 1
            for s in SYSTEMS:
                ranked = [r["doc_id"] for r in pool["per_system"].get(s, [])]
                agg[s]["ndcg@10"].append(ndcg_at_k(ranked, labels, 10))
                agg[s]["recall@25"].append(recall_at_k(ranked, labels, 25))
                agg[s]["map"].append(average_precision(ranked, labels))
        if n_q == 0:
            continue
        per_condition[cond["name"]] = {
            "size": cond["size"] or len(doc_ids), "rep": cond["rep"], "n_questions": n_q,
            "per_system": {s: {m: round(sum(v) / len(v), 4) for m, v in agg[s].items()} for s in SYSTEMS},
        }

    def _mean_std(vals: list[float]) -> dict:
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
        return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4), "n_seeds": len(vals)}

    curve: dict[str, dict] = {}
    for size in SCALING_SIZES:
        reps = [per_condition[f"n{size}_s{rep}"] for rep in range(1, SCALING_REPS + 1)
                if f"n{size}_s{rep}" in per_condition]
        if not reps:
            continue
        curve[str(size)] = {s: {m: _mean_std([r["per_system"][s][m] for r in reps])
                                for m in ["ndcg@10", "recall@25", "map"]} for s in SYSTEMS}
    if "full" in per_condition:
        full = per_condition["full"]
        curve[str(full["size"])] = {s: {m: {"mean": full["per_system"][s][m], "std": 0.0, "n_seeds": 1}
                                        for m in ["ndcg@10", "recall@25", "map"]} for s in SYSTEMS}
    return {"judge_model": model, "top_k": TOP_K, "sizes": SCALING_SIZES,
            "reps": SCALING_REPS, "full_size": len(doc_ids), "curve": curve,
            "per_condition": per_condition, "generated_at": _now()}


# --- RQ4: ANN index architectures over the same subsets (seed 1 per size + full) ---

def _ann_queries():
    """Query matrix: the eval questions' embeddings + seeded sampled doc vectors."""
    import numpy as np
    if "ann_queries" not in _SCALING_STATE:
        emb, _ = _scaling_universe()
        q_embs = [_scaling_query_embedding(q["raw_question"])[0] for q in _load_questions()]
        rng = random.Random(f"{SEED}:ann_queries")
        doc_pick = sorted(rng.sample(range(emb.shape[0]), min(ANN_N_DOC_QUERIES, emb.shape[0])))
        _SCALING_STATE["ann_queries"] = np.vstack([np.asarray(q_embs, dtype=np.float32), emb[doc_pick]])
    return _SCALING_STATE["ann_queries"]


def _ann_ground_truth(sub, queries, k: int):
    """Exact top-k positions per query on the subset matrix (chunked flat scan)."""
    import numpy as np
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for start in range(0, queries.shape[0], 64):
        block = queries[start:start + 64] @ sub.T
        part = np.argpartition(block, -k, axis=1)[:, -k:]
        rows = np.arange(block.shape[0])[:, None]
        order = np.argsort(block[rows, part], axis=1)[:, ::-1]
        out[start:start + block.shape[0]] = part[rows, order]
    return out


def _ann_recall(candidates, truth) -> float:
    hits = sum(len(set(c) & set(t)) for c, t in zip(candidates, truth))
    return hits / float(truth.shape[0] * truth.shape[1])


def _ann_percentiles(lat_s: list[float]) -> tuple[float, float]:
    xs = sorted(lat_s)
    p50 = xs[int(0.50 * (len(xs) - 1))] * 1000.0
    p95 = xs[int(0.95 * (len(xs) - 1))] * 1000.0
    return round(p50, 3), round(p95, 3)


def _ann_flat(sub, queries, k: int, truth) -> dict:
    import numpy as np
    lat: list[float] = []
    cands = np.empty((queries.shape[0], k), dtype=np.int64)
    for i in range(queries.shape[0]):
        t0 = time.perf_counter()
        scores = queries[i] @ sub.T
        part = np.argpartition(scores, -k)[-k:]
        cands[i] = part[np.argsort(scores[part])[::-1]]
        lat.append(time.perf_counter() - t0)
    p50, p95 = _ann_percentiles(lat)
    return {"arch": "flat", "operating_point": "exact", "build_s": 0.0,
            "index_mb": round(sub.nbytes / 1e6, 1), "recall@10": 1.0,
            "lat_p50_ms": p50, "lat_p95_ms": p95, "params": "exact NumPy dot product"}


def _ann_faiss(sub, queries, k: int, truth, size_label: str) -> list[dict]:
    try:
        import faiss
    except ImportError:
        print("[ann] faiss not installed (uv sync --extra ann-bench); skipping FAISS variants")
        return []
    import numpy as np
    rows: list[dict] = []
    n, d = sub.shape
    tmp = SCALING_DIR / "tmp_faiss.index"

    def measure(index, arch: str, ops: list[tuple[str, callable]], params: str, build_s: float) -> None:
        faiss.write_index(index, str(tmp))
        index_mb = round(tmp.stat().st_size / 1e6, 1)
        tmp.unlink()
        for op_label, apply_op in ops:
            apply_op(index)
            lat: list[float] = []
            cands = np.empty((queries.shape[0], k), dtype=np.int64)
            for i in range(queries.shape[0]):
                t0 = time.perf_counter()
                _, ids = index.search(queries[i:i + 1], k)
                cands[i] = ids[0]
                lat.append(time.perf_counter() - t0)
            p50, p95 = _ann_percentiles(lat)
            rows.append({"arch": arch, "operating_point": op_label, "build_s": round(build_s, 1),
                         "index_mb": index_mb, "recall@10": round(_ann_recall(cands, truth), 4),
                         "lat_p50_ms": p50, "lat_p95_ms": p95, "params": params})
            print(f"[ann] {size_label} {arch}/{op_label}: recall@10={rows[-1]['recall@10']:.3f} "
                  f"p50={p50:.2f}ms build={build_s:.0f}s")

    # IVF-PQ: nlist ~ 4*sqrt(N) capped so k-means has >= 39 train points per centroid
    nlist = max(16, min(int(4 * math.sqrt(n)), n // 39))
    quantizer = faiss.IndexFlatIP(d)
    ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, 64, 8, faiss.METRIC_INNER_PRODUCT)
    t0 = time.monotonic()
    ivfpq.train(sub)
    ivfpq.add(sub)
    build_ivf = time.monotonic() - t0

    def set_nprobe(np_val):
        def apply(ix):
            ix.nprobe = np_val
        return apply
    measure(ivfpq, "faiss_ivfpq", [("nprobe=1", set_nprobe(1)), ("nprobe=10", set_nprobe(10))],
            f"nlist={nlist}, m=64, nbits=8", build_ivf)

    hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    t0 = time.monotonic()
    hnsw.add(sub)
    build_hnsw = time.monotonic() - t0

    def set_ef(ix):
        ix.hnsw.efSearch = 40
    measure(hnsw, "faiss_hnsw", [("efSearch=40", set_ef)], "M=32, efConstruction=200", build_hnsw)
    return rows


def _ann_pgvector(sub, queries, k: int, truth, size_label: str, is_full: bool) -> list[dict]:
    from corpusagent2.retrieval import pg_connect_kwargs, pg_dsn_from_env, pg_table_from_env, _vector_literal
    dsn = pg_dsn_from_env(required=False)
    if not dsn:
        print("[ann] no Postgres DSN configured; skipping pgvector variants")
        return []
    from psycopg import connect
    import numpy as np
    rows: list[dict] = []
    n_lat = min(ANN_PG_LATENCY_QUERIES, queries.shape[0])
    lat_queries = queries[:n_lat]
    lat_truth = truth[:n_lat]

    def measure(conn, table: str, arch: str, op_label: str, setup_sql: str,
                build_s, index_name: str, params: str) -> None:
        with conn.cursor() as cur:
            cur.execute(f"SELECT pg_relation_size(%s)", (index_name,))
            index_mb = round((cur.fetchone()[0] or 0) / 1e6, 1)
            if setup_sql:
                cur.execute(setup_sql)
            lat: list[float] = []
            cands = np.empty((n_lat, k), dtype=np.int64)
            for i in range(n_lat):
                vec = _vector_literal(lat_queries[i])
                t0 = time.perf_counter()
                cur.execute(f"SELECT row_idx FROM {table} ORDER BY dense_embedding <=> %s::vector LIMIT %s",
                            (vec, k))
                got = [r[0] for r in cur.fetchall()]
                lat.append(time.perf_counter() - t0)
                cands[i] = (got + [-1] * k)[:k]
        p50, p95 = _ann_percentiles(lat)
        rows.append({"arch": arch, "operating_point": op_label,
                     "build_s": None if build_s is None else round(build_s, 1),
                     "index_mb": index_mb, "recall@10": round(_ann_recall(cands, lat_truth), 4),
                     "lat_p50_ms": p50, "lat_p95_ms": p95, "params": params, "n_queries": n_lat})
        print(f"[ann] {size_label} {arch}/{op_label}: recall@10={rows[-1]['recall@10']:.3f} p50={p50:.2f}ms")

    if is_full:
        # Production table + its existing index: the deployed operating point.
        table = pg_table_from_env()
        with connect(dsn, **pg_connect_kwargs()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT indexname FROM pg_indexes WHERE tablename = %s AND indexdef ILIKE '%%ivfflat%%'",
                    (table,))
                hit = cur.fetchone()
            if not hit:
                print(f"[ann] no ivfflat index found on {table}; skipping full pgvector point")
                return rows
            # Production table keys rows by doc_id; translate to universe row positions.
            _, doc_ids = _scaling_universe()
            pos_by_doc = {d: i for i, d in enumerate(doc_ids)}
            with conn.cursor() as cur:
                for probes in (1, 10):
                    cur.execute(f"SET ivfflat.probes = {probes}")
                    lat: list[float] = []
                    cands = np.full((n_lat, k), -1, dtype=np.int64)
                    for i in range(n_lat):
                        vec = _vector_literal(lat_queries[i])
                        t0 = time.perf_counter()
                        cur.execute(
                            f"SELECT doc_id FROM {table} WHERE dense_embedding IS NOT NULL "
                            f"ORDER BY dense_embedding <=> %s::vector LIMIT %s", (vec, k))
                        got = [pos_by_doc.get(str(r[0]), -1) for r in cur.fetchall()]
                        lat.append(time.perf_counter() - t0)
                        cands[i] = (got + [-1] * k)[:k]
                    p50, p95 = _ann_percentiles(lat)
                    rows.append({"arch": "pgvector_ivfflat", "operating_point": f"probes={probes}",
                                 "build_s": None, "index_mb": None,
                                 "recall@10": round(_ann_recall(cands, lat_truth), 4),
                                 "lat_p50_ms": p50, "lat_p95_ms": p95,
                                 "params": f"production index {hit[0]}", "n_queries": n_lat})
                    print(f"[ann] {size_label} pgvector_ivfflat/probes={probes}: "
                          f"recall@10={rows[-1]['recall@10']:.3f} p50={p50:.2f}ms")
        return rows

    table = "ca2_scaling_ann_bench"
    n = sub.shape[0]
    lists = max(1, n // 1_000)  # mirrors scripts/11 recommended_ivfflat_lists (<= 1M rows)
    with connect(dsn, **pg_connect_kwargs()) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            cur.execute(f"CREATE UNLOGGED TABLE {table} (row_idx int, dense_embedding vector(768))")
            with cur.copy(f"COPY {table} (row_idx, dense_embedding) FROM STDIN") as copy:
                for i in range(n):
                    copy.write_row((i, _vector_literal(sub[i])))
            cur.execute("SET maintenance_work_mem = '512MB'")
            t0 = time.monotonic()
            cur.execute(f"CREATE INDEX {table}_ivf ON {table} "
                        f"USING ivfflat (dense_embedding vector_cosine_ops) WITH (lists = {lists})")
            build_ivf = time.monotonic() - t0
        for probes in (1, 10):
            measure(conn, table, "pgvector_ivfflat", f"probes={probes}",
                    f"SET ivfflat.probes = {probes}", build_ivf, f"{table}_ivf", f"lists={lists}")
        with conn.cursor() as cur:
            cur.execute(f"DROP INDEX {table}_ivf")
            t0 = time.monotonic()
            cur.execute(f"CREATE INDEX {table}_hnsw ON {table} "
                        "USING hnsw (dense_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)")
            build_hnsw = time.monotonic() - t0
        measure(conn, table, "pgvector_hnsw", "ef_search=40", "SET hnsw.ef_search = 40",
                build_hnsw, f"{table}_hnsw", "m=16, ef_construction=64")
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
    return rows


def phase_scaling_ann() -> dict:
    import numpy as np
    emb, _ = _scaling_universe()
    queries = _ann_queries()
    k = ANN_RECALL_K[0]
    out_path = SCALING_DIR / "ann_bench.json"
    results: dict[str, list[dict]] = {}
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8")).get("by_size", {})
    ann_conds = [c for c in _scaling_conditions() if c["rep"] in (0, 1)]
    for cond in ann_conds:
        label = str(cond["size"] or emb.shape[0])
        if label in results:
            print(f"[ann] {label}: cached")
            continue
        if cond["size"] is None:
            sub = emb
        else:
            subset_idx = np.load(SCALING_DIR / "subsets" / f"{cond['name']}.npy")
            sub = np.ascontiguousarray(emb[subset_idx])
        print(f"[ann] === size {label}: computing exact ground truth (flat scan) ===")
        truth = _ann_ground_truth(sub, queries, k)
        rows = [_ann_flat(sub, queries, k, truth)]
        rows += _ann_faiss(sub, queries, k, truth, label)
        rows += _ann_pgvector(sub, queries, k, truth, label, is_full=cond["size"] is None)
        results[label] = rows
        out_path.write_text(json.dumps({"recall_k": k, "n_queries": int(queries.shape[0]),
                                        "by_size": results, "generated_at": _now()},
                                       indent=2), encoding="utf-8")
        if cond["size"] is not None:
            del sub
    return {"recall_k": k, "n_queries": int(queries.shape[0]), "by_size": results,
            "generated_at": _now()}


# ============================================================================
# Phase 5: render LaTeX tables + plots
# ============================================================================

def _w(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[render] wrote {path.relative_to(PROJECT_ROOT)}")


def render_protocol_a(summary: dict) -> None:
    ps = summary["per_system"]
    n = summary["questions_evaluated"]
    rows = []
    for s in SYSTEMS:
        nd = ps[s]["ndcg@10"]; r25 = ps[s]["recall@25"]; mp = ps[s]["map"]
        rows.append(
            f"{SYSTEM_LABEL[s]} & {nd['mean']:.3f} {{\\scriptsize[{nd['ci_low']:.2f},{nd['ci_high']:.2f}]}} "
            f"& {r25['mean']:.3f} {{\\scriptsize[{r25['ci_low']:.2f},{r25['ci_high']:.2f}]}} "
            f"& {mp['mean']:.3f} {{\\scriptsize[{mp['ci_low']:.2f},{mp['ci_high']:.2f}]}} \\\\")
    body = "\n".join(rows)
    tex = (
        "% AUTO-GENERATED by scripts/50_run_eval_suite.py -- do not edit by hand.\n"
        "\\begin{table*}[t]\n\\centering\\small\n"
        f"\\caption{{Protocol A: oracle-free LLM-as-judge retrieval quality over {n} open-ended "
        f"questions (judge: \\texttt{{{summary['judge_model']}}}, pooled top-{summary['top_k']}, "
        "graded relevance $\\{0,1,2,3\\}$). \\textbf{nDCG@10 is the primary metric}: it assumes no "
        "knowledge of the (undefined) total relevant set. $^{\\dagger}$\\,Pooled Recall@25 and MAP are "
        "relative diagnostics only --- they divide by the judge-relevant set \\emph{within the pool}, "
        "not a corpus-complete gold set, so they are not absolute recall/precision. No gold document "
        "set is used anywhere. Brackets are 95\\% bootstrap CIs over questions.}\n"
        "\\label{tab:protocol-a-results}\n"
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "\\textbf{System} & \\textbf{nDCG@10} & \\textbf{Pooled Recall@25}$^{\\dagger}$ & \\textbf{MAP}$^{\\dagger}$ \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table*}}\n"
    )
    _w(LATEX_GEN_DIR / "results_protocol_a.tex", tex)


def render_cross_judge(cj: dict) -> None:
    def verdict(t: float) -> str:
        if t >= 0.85:
            return "judges agree (UMBRELA $\\tau\\geq0.85$)"
        if t >= 0.6:
            return "moderate agreement"
        return "ranking ambiguous"
    rows = "\n".join(
        f"{m} & {cj['kendall_tau'][m]:.3f} & {verdict(cj['kendall_tau'][m])} \\\\"
        for m in ["ndcg@10", "recall@25", "map"])
    tex = (
        "% AUTO-GENERATED by scripts/50_run_eval_suite.py -- do not edit by hand.\n"
        "\\begin{table*}[t]\n\\centering\\small\n"
        f"\\caption{{Cross-judge robustness: Kendall's $\\tau$ between the system rankings produced "
        f"by two different judge models (\\texttt{{{cj['judge_1']}}} vs \\texttt{{{cj['judge_2']}}}).}}\n"
        "\\label{tab:cross-judge}\n\\begin{tabular}{@{}lcl@{}}\n\\toprule\n"
        "\\textbf{Metric} & \\textbf{Kendall $\\tau$} & \\textbf{Verdict} \\\\\n\\midrule\n"
        f"{rows}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table*}}\n"
    )
    _w(LATEX_GEN_DIR / "results_cross_judge.tex", tex)


def render_metamorphic(mc: dict) -> None:
    tname = {"paraphrase": "Paraphrase (expect HIGH)", "entity_swap": "Entity swap (expect LOW)"}
    rows = []
    for t in mc["transforms"]:
        for s in mc["systems"]:
            j = mc["jaccard"][t][s]
            rows.append(f"{tname.get(t, t)} & {SYSTEM_LABEL[s]} & {j['mean']:.3f} "
                        f"{{\\scriptsize[{j['ci_low']:.2f},{j['ci_high']:.2f}]}} \\\\")
    body = "\n".join(rows)
    tex = (
        "% AUTO-GENERATED by scripts/50_run_eval_suite.py -- do not edit by hand.\n"
        "\\begin{table*}[t]\n\\centering\\small\n"
        f"\\caption{{Protocol C: metamorphic robustness over {mc['n']} questions. Top-{TOP_K} Jaccard "
        "overlap of retrieval between the original and transformed query. Paraphrase should preserve "
        "the result set (high overlap); entity swap should change it (low overlap). Oracle-free.}\n"
        "\\label{tab:protocol-c-results}\n\\begin{tabular}{@{}llc@{}}\n\\toprule\n"
        "\\textbf{Transformation} & \\textbf{System} & \\textbf{Top-25 Jaccard} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table*}}\n"
    )
    _w(LATEX_GEN_DIR / "results_protocol_c.tex", tex)


def render_protocol_b(pb: dict) -> None:
    rows = []
    for f in FAMILIES:
        r = pb["per_family"][f]
        if r["n_claims"] == 0:
            rows.append(f"{FAMILY_LABEL[f]} & -- & -- & -- \\\\")
        else:
            rows.append(f"{FAMILY_LABEL[f]} & {r['faithfulness']:.3f} & {r['unsupported']:.3f} & {r['contradiction']:.3f} \\\\")
    o = pb["overall"]
    rows.append("\\midrule")
    rows.append(f"\\textbf{{Overall}} & {o['faithfulness']:.3f} & {o['unsupported']:.3f} & {o['contradiction']:.3f} \\\\")
    body = "\n".join(rows)
    tex = (
        "% AUTO-GENERATED by scripts/50_run_eval_suite.py -- do not edit by hand.\n"
        "\\begin{table*}[t]\n\\centering\\small\n"
        f"\\caption{{Protocol B: oracle-free claim-to-evidence faithfulness over {o['n_claims']} atomic "
        "claims, by question family. Each claim is extracted from a grounded answer synthesised from the "
        "system's own top-retrieved evidence and scored by NLI (\\texttt{roberta-large-mnli}) against that "
        "evidence; faithfulness = entailed/total, unsupported = no evidence sentence found, contradiction = "
        "NLI contradicts. No gold answer or gold evidence is used.}\n"
        "\\label{tab:protocol-b-results}\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "\\textbf{Family} & \\textbf{Faith.} & \\textbf{Unsup.} & \\textbf{Contra.} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table*}}\n"
    )
    _w(LATEX_GEN_DIR / "results_protocol_b.tex", tex)


def render_retrievability(rt: dict) -> None:
    rows = "\n".join(
        f"{SYSTEM_LABEL[s]} & {rt['systems'][s]['gini']:.3f} & "
        f"{rt['systems'][s]['unique_docs_reached']:,} & {100*rt['systems'][s]['corpus_coverage']:.2f}\\% \\\\"
        for s in RETR_SYSTEMS)
    max_slots = rt["n_queries"] * rt["top_k"]
    tex = (
        "% AUTO-GENERATED by scripts/50_run_eval_suite.py -- do not edit by hand.\n"
        "\\begin{table*}[t]\n\\centering\\small\n"
        f"\\caption{{Retrievability access-breadth probe (Azzopardi \\& Vinay) over {rt['n_queries']} "
        f"simulated queries drawn from the corpus, top-{rt['top_k']} (max {max_slots:,} retrieval slots). "
        "``Docs reached'' is the number of distinct documents retrieved at least once: closer to the "
        f"{max_slots:,}-slot ceiling means less repetition across queries, i.e.\\ broader corpus access. "
        "The corpus-wide Gini is reported for completeness but is budget-saturated here (a query count "
        "far below the corpus size cannot help but leave almost all documents unreached); the comparative "
        "docs-reached figure is the interpretable signal. Oracle-free: no gold, no LLM.}\n"
        "\\label{tab:retrievability}\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "\\textbf{System} & \\textbf{Gini} & \\textbf{Distinct docs reached $\\uparrow$} & \\textbf{Coverage} \\\\\n\\midrule\n"
        f"{rows}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table*}}\n"
    )
    _w(LATEX_GEN_DIR / "results_retrievability.tex", tex)


# Categorical colors: validated reference palette (dataviz), fixed entity order --
# a color follows its system/architecture, never its rank.
SYSTEM_COLOR = {"lexical": "#2a78d6", "dense": "#1baf7a", "hybrid": "#eda100", "hybrid_rerank": "#008300"}
ARCH_COLOR = {"flat": "#2a78d6", "faiss_ivfpq": "#1baf7a", "faiss_hnsw": "#eda100",
              "pgvector_ivfflat": "#008300", "pgvector_hnsw": "#4a3aa7"}
ARCH_LABEL = {"flat": "Flat (exact)", "faiss_ivfpq": "FAISS IVF-PQ", "faiss_hnsw": "FAISS HNSW",
              "pgvector_ivfflat": "pgvector IVFFlat", "pgvector_hnsw": "pgvector HNSW"}


def _fmt_size(n: int) -> str:
    return f"{n/1000:.0f}k" if n < 1_000_000 else f"{n/1e6:.1f}M"


def render_scaling_rq1(rq1: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curve = rq1["curve"]
    sizes = sorted(int(s) for s in curve.keys())
    if not sizes:
        print("[render] scaling RQ1: no judged conditions yet; skipped")
        return
    full_size = rq1.get("full_size")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharex=True)
    for ax, metric, title in zip(axes, ["ndcg@10", "recall@25"], ["nDCG@10", "Pooled Recall@25"]):
        for s in SYSTEMS:
            xs, means, stds = [], [], []
            for size in sizes:
                cell = curve[str(size)].get(s)
                if cell is None:
                    continue
                xs.append(size)
                means.append(cell[metric]["mean"])
                stds.append(cell[metric]["std"])
            if not xs:
                continue
            color = SYSTEM_COLOR[s]
            ax.plot(xs, means, color=color, linewidth=1.6, marker="o", markersize=4,
                    label=SYSTEM_LABEL[s])
            lo = [m - sd for m, sd in zip(means, stds)]
            hi = [m + sd for m, sd in zip(means, stds)]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.14, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([_fmt_size(s) for s in sizes], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("corpus size (documents, log scale)", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="major", axis="y", color="#e5e4e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.minorticks_off()
    axes[0].set_ylabel("score", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    note = f"seeded subsets x{rq1['reps']} (+-1 std); {_fmt_size(full_size)} = full corpus, single run" \
        if full_size and full_size in sizes else f"seeded subsets x{rq1['reps']} (+-1 std)"
    fig.suptitle(f"Retrieval quality vs corpus scale (oracle-free, judge={rq1['judge_model']}; {note})",
                 fontsize=8, y=1.02)
    fig.tight_layout()
    out = LATEX_GEN_DIR / "plot_scaling_rq1.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[render] wrote {out.relative_to(PROJECT_ROOT)}")

    rows = []
    for size in sizes:
        cells = " & ".join(
            f"{curve[str(size)][s]['ndcg@10']['mean']:.3f} $\\pm$ {curve[str(size)][s]['ndcg@10']['std']:.3f}"
            if curve[str(size)][s]["ndcg@10"]["n_seeds"] > 1
            else f"{curve[str(size)][s]['ndcg@10']['mean']:.3f}"
            for s in SYSTEMS)
        label = _fmt_size(size) + (" (full)" if size == full_size else "")
        rows.append(f"{label} & {cells} \\\\")
    body = "\n".join(rows)
    heads = " & ".join(f"\\textbf{{{SYSTEM_LABEL[s]}}}" for s in SYSTEMS)
    tex = (
        "% AUTO-GENERATED by scripts/50_run_eval_suite.py -- do not edit by hand.\n"
        "\\begin{table*}[t]\n\\centering\\small\n"
        f"\\caption{{Retrieval quality (nDCG@10) across corpus scale: seeded random subsets of the "
        f"624k corpus ($\\times${rq1['reps']} independent draws per size, mean $\\pm$ 1 std; the full-corpus "
        "point is a single deterministic run). Per-subset BM25 indexes are rebuilt so lexical collection "
        "statistics belong to each scale; dense scores are exact per-document and need no rebuild. "
        "Oracle-free: relevance labels come from the pinned LLM judge over the pooled candidates of each "
        "condition, so values read as ``quality of what the systems surface from a corpus of size $N$,'' "
        "not recall of a fixed relevant set.}\n"
        "\\label{tab:scaling-rq1}\n"
        f"\\begin{{tabular}}{{@{{}}l{'c' * len(SYSTEMS)}@{{}}}}\n\\toprule\n"
        f"\\textbf{{Corpus size}} & {heads} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table*}}\n"
    )
    _w(LATEX_GEN_DIR / "results_scaling_rq1.tex", tex)


def render_scaling_rq4(rq4: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_size = rq4.get("by_size", {})
    sizes = sorted(int(s) for s in by_size.keys())
    if not sizes:
        print("[render] scaling RQ4: no benchmark rows yet; skipped")
        return

    def series(arch: str, field: str, op_filter=None):
        xs, ys = [], []
        for size in sizes:
            match = [r for r in by_size[str(size)]
                     if r["arch"] == arch and (op_filter is None or r["operating_point"] == op_filter)
                     and r.get(field) is not None]
            if match:
                xs.append(size)
                ys.append(match[0][field])
        return xs, ys

    # Preferred operating point per architecture (dashed = its weak default, where measured)
    op_main = {"flat": "exact", "faiss_ivfpq": "nprobe=10", "faiss_hnsw": "efSearch=40",
               "pgvector_ivfflat": "probes=10", "pgvector_hnsw": "ef_search=40"}
    op_weak = {"faiss_ivfpq": "nprobe=1", "pgvector_ivfflat": "probes=1"}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharex=True)
    for arch in ARCH_COLOR:
        color = ARCH_COLOR[arch]
        xs, ys = series(arch, "recall@10", op_main.get(arch))
        if xs:
            axes[0].plot(xs, ys, color=color, linewidth=1.6, marker="o", markersize=4,
                         label=ARCH_LABEL[arch])
        if arch in op_weak:
            xs, ys = series(arch, "recall@10", op_weak[arch])
            if xs:
                axes[0].plot(xs, ys, color=color, linewidth=1.2, linestyle=":", marker="o",
                             markersize=3, alpha=0.75)
        xs, ys = series(arch, "lat_p50_ms", op_main.get(arch))
        if xs:
            axes[1].plot(xs, ys, color=color, linewidth=1.6, marker="o", markersize=4,
                         label=ARCH_LABEL[arch])
    axes[0].set_ylabel("recall@10 vs exact flat scan", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("ANN recall (dotted = probes/nprobe = 1)", fontsize=9)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("per-query latency p50 (ms, log)", fontsize=8)
    axes[1].set_title("Query latency (CPU-only VM)", fontsize=9)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([_fmt_size(s) for s in sizes], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("corpus size (documents, log scale)", fontsize=8)
        ax.grid(True, which="major", axis="y", color="#e5e4e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.minorticks_off()
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    out = LATEX_GEN_DIR / "plot_scaling_rq4.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[render] wrote {out.relative_to(PROJECT_ROOT)}")

    rows = []
    for size in sizes:
        first = True
        for r in by_size[str(size)]:
            size_cell = _fmt_size(size) if first else ""
            first = False
            build = "--" if r.get("build_s") is None else f"{r['build_s']:.0f}"
            mb = "--" if r.get("index_mb") is None else f"{r['index_mb']:.0f}"
            rows.append(f"{size_cell} & {ARCH_LABEL.get(r['arch'], r['arch'])} & {r['operating_point']} "
                        f"& {build} & {mb} & {r['recall@10']:.3f} & {r['lat_p50_ms']:.2f} \\\\")
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    body = "\n".join(rows)
    tex = (
        "% AUTO-GENERATED by scripts/50_run_eval_suite.py -- do not edit by hand.\n"
        "\\begin{table*}[t]\n\\centering\\small\n"
        f"\\caption{{RQ4 index-architecture benchmark on seeded corpus subsets (seed 1 per size) and the "
        f"full corpus, measured on the CPU-only production VM over {rq4['n_queries']} queries "
        "(the evaluation questions' embeddings plus seeded sampled document vectors). Recall@10 is "
        "computed against the exact flat scan over the same vectors (oracle-free, self-referential "
        "ground truth). ``--'' = existing production index reused (build not re-measured). pgvector "
        "rows use the same lists/probes policy as the production deployment.}\n"
        "\\label{tab:scaling-rq4}\n\\begin{tabular}{@{}llccccc@{}}\n\\toprule\n"
        "\\textbf{Size} & \\textbf{Architecture} & \\textbf{Op.\\ point} & \\textbf{Build (s)} & "
        "\\textbf{Index (MB)} & \\textbf{Recall@10} & \\textbf{p50 (ms)} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table*}}\n"
    )
    _w(LATEX_GEN_DIR / "results_scaling_rq4.tex", tex)


def render_plots(summary_a: dict, mc: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps = summary_a["per_system"]
    labels = [SYSTEM_LABEL[s] for s in SYSTEMS]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = range(len(SYSTEMS))
    w = 0.27
    for i, metric in enumerate(["ndcg@10", "recall@25", "map"]):
        means = [ps[s][metric]["mean"] for s in SYSTEMS]
        errs = [[ps[s][metric]["mean"] - ps[s][metric]["ci_low"] for s in SYSTEMS],
                [ps[s][metric]["ci_high"] - ps[s][metric]["mean"] for s in SYSTEMS]]
        ax.bar([xi + (i - 1) * w for xi in x], means, w, yerr=errs, capsize=2,
               label=metric.upper())
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("score")
    ax.set_title(f"Protocol A (oracle-free, N={summary_a['questions_evaluated']}, judge={summary_a['judge_model']})",
                 fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = LATEX_GEN_DIR / "plot_protocol_a.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[render] wrote {out.relative_to(PROJECT_ROOT)}")


# ============================================================================
# main
# ============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_GEN_DIR.mkdir(parents=True, exist_ok=True)
    # Load config/.env (existing env vars keep priority) so phases that talk to
    # OpenSearch/Postgres directly get credentials without building the runtime.
    from corpusagent2.app_config import load_project_configuration
    load_project_configuration(PROJECT_ROOT)
    questions = _load_questions()
    print(f"Loaded {len(questions)} questions | judge={JUDGE_MODEL} | OS={os.environ['CORPUSAGENT2_OPENSEARCH_URL']}\n")

    if RUN_RETRIEVE:
        print("=== Phase 1: retrieval (4 systems, oracle-free pools) ===")
        phase_retrieve(questions)
        print()
    if RUN_JUDGE:
        print(f"=== Phase 2: LLM-as-judge ({JUDGE_MODEL}) ===")
        phase_judge(questions, JUDGE_MODEL)
        print()

    summary_a = compute_protocol_a(questions, JUDGE_MODEL)
    (OUTPUT_DIR / "protocol_a.json").write_text(json.dumps(summary_a, indent=2), encoding="utf-8")
    print("=== Protocol A headline ===")
    for s in SYSTEMS:
        m = summary_a["per_system"][s]
        print(f"  {SYSTEM_LABEL[s]:18s} nDCG@10={m['ndcg@10']['mean']:.3f} "
              f"Recall@25={m['recall@25']['mean']:.3f} MAP={m['map']['mean']:.3f} (N={m['ndcg@10']['n']})")
    print()

    cj = None
    if RUN_CROSS_JUDGE:
        print(f"=== Cross-judge ({JUDGE_MODEL_2}) ===")
        phase_judge(questions, JUDGE_MODEL_2)
        cj = compute_cross_judge(questions, summary_a, JUDGE_MODEL_2)
        (OUTPUT_DIR / "cross_judge.json").write_text(json.dumps(cj, indent=2), encoding="utf-8")
        print("  Kendall tau:", cj["kendall_tau"])
        print()

    mc = None
    if RUN_METAMORPHIC:
        print("=== Protocol C: metamorphic robustness ===")
        mc = phase_metamorphic(questions)
        (OUTPUT_DIR / "metamorphic.json").write_text(json.dumps(mc, indent=2), encoding="utf-8")
        for t in mc["transforms"]:
            for s in mc["systems"]:
                print(f"  {t:11s} {SYSTEM_LABEL[s]:18s} Jaccard@25={mc['jaccard'][t][s]['mean']:.3f}")
        print()

    pb = None
    if RUN_PROTOCOL_B:
        print("=== Protocol B: claim-to-evidence faithfulness (oracle-free) ===")
        pb = phase_protocol_b(questions)
        (OUTPUT_DIR / "protocol_b.json").write_text(json.dumps(pb, indent=2), encoding="utf-8")
        o = pb["overall"]
        print(f"  Overall: faithfulness={o['faithfulness']:.3f} unsupported={o['unsupported']:.3f} "
              f"contradiction={o['contradiction']:.3f} (N={o['n_claims']} claims)")
        print()

    rt = None
    if RUN_RETRIEVABILITY:
        print("=== Retrievability bias (Gini, oracle-free) ===")
        rt = phase_retrievability(questions)
        (OUTPUT_DIR / "retrievability.json").write_text(json.dumps(rt, indent=2), encoding="utf-8")
        for s in RETR_SYSTEMS:
            v = rt["systems"][s]
            print(f"  {SYSTEM_LABEL[s]:18s} Gini={v['gini']:.3f} reached={v['unique_docs_reached']:,} "
                  f"coverage={100*v['corpus_coverage']:.2f}%")
        print()

    scaling_rq1 = None
    if RUN_SCALING:
        print("=== Scaling curve (RQ1): seeded subsets, per-scale Protocol A ===")
        phase_scaling_subsets()
        phase_scaling_retrieve(questions)
        if RUN_SCALING_JUDGE:
            phase_scaling_judge(JUDGE_MODEL)
        scaling_rq1 = compute_scaling_rq1(questions, JUDGE_MODEL)
        (OUTPUT_DIR / "scaling_rq1.json").write_text(json.dumps(scaling_rq1, indent=2), encoding="utf-8")
        for size in sorted(int(s) for s in scaling_rq1["curve"]):
            cells = scaling_rq1["curve"][str(size)]
            print("  " + f"{size:>8,}: " + "  ".join(
                f"{s}={cells[s]['ndcg@10']['mean']:.3f}±{cells[s]['ndcg@10']['std']:.3f}" for s in SYSTEMS))
        print()

    scaling_rq4 = None
    if RUN_SCALING_ANN:
        print("=== ANN architecture benchmark (RQ4) ===")
        phase_scaling_subsets()
        scaling_rq4 = phase_scaling_ann()
        (OUTPUT_DIR / "scaling_rq4.json").write_text(json.dumps(scaling_rq4, indent=2), encoding="utf-8")
        print()

    # RENDER-only re-runs refresh the scaling artifacts from their cached JSON.
    if scaling_rq1 is None and (OUTPUT_DIR / "scaling_rq1.json").exists():
        scaling_rq1 = json.loads((OUTPUT_DIR / "scaling_rq1.json").read_text(encoding="utf-8"))
    if scaling_rq4 is None and (OUTPUT_DIR / "scaling_rq4.json").exists():
        scaling_rq4 = json.loads((OUTPUT_DIR / "scaling_rq4.json").read_text(encoding="utf-8"))

    if RENDER:
        print("=== Render LaTeX + plots ===")
        render_protocol_a(summary_a)
        if cj:
            render_cross_judge(cj)
        if mc:
            render_metamorphic(mc)
        if pb:
            render_protocol_b(pb)
        if rt:
            render_retrievability(rt)
        if scaling_rq1:
            try:
                render_scaling_rq1(scaling_rq1)
            except Exception as exc:
                print(f"[render] scaling RQ1 skipped: {exc}")
        if scaling_rq4:
            try:
                render_scaling_rq4(scaling_rq4)
            except Exception as exc:
                print(f"[render] scaling RQ4 skipped: {exc}")
        try:
            render_plots(summary_a, mc)
        except Exception as exc:
            print(f"[render] plot skipped: {exc}")
    print("\nDONE.")
