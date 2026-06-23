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

def phase_judge(questions: list[dict], model: str) -> None:
    client = _judge_client()
    cache_dir = OUTPUT_DIR / "judge_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    new = cached = failed = 0
    for q in questions:
        qid = q["question_id"]
        pool_path = OUTPUT_DIR / "pool" / f"{qid}.json"
        if not pool_path.exists():
            continue
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        for doc in pool["pool"]:
            text = _truncate(doc.get("text") or "", JUDGE_CONTEXT_TOKENS)
            if not text.strip():
                continue
            h = _judge_hash(pool["question"], text, model)
            cpath = cache_dir / f"{h}.json"
            if cpath.exists():
                cached += 1
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
                failed += 1
                print(f"[judge:{model}] {qid}/{doc['doc_id'][:8]}: {type(exc).__name__}: {str(exc)[:80]}")
                continue
            cpath.write_text(json.dumps({"question_id": qid, "doc_id": doc["doc_id"],
                                         "label": label, "judge_model": model,
                                         "judged_at": _now()}, ensure_ascii=False), encoding="utf-8")
            new += 1
            if new % 25 == 0:
                print(f"[judge:{model}] {new} new, {cached} cached, {failed} failed")
    print(f"[judge:{model}] DONE: {new} new, {cached} cached, {failed} failed")


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
        "\\begin{table}[h]\n\\centering\\small\n"
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
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
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
        "\\begin{table}[h]\n\\centering\\small\n"
        f"\\caption{{Cross-judge robustness: Kendall's $\\tau$ between the system rankings produced "
        f"by two different judge models (\\texttt{{{cj['judge_1']}}} vs \\texttt{{{cj['judge_2']}}}).}}\n"
        "\\label{tab:cross-judge}\n\\begin{tabular}{@{}lcl@{}}\n\\toprule\n"
        "\\textbf{Metric} & \\textbf{Kendall $\\tau$} & \\textbf{Verdict} \\\\\n\\midrule\n"
        f"{rows}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
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
        "\\begin{table}[h]\n\\centering\\small\n"
        f"\\caption{{Protocol C: metamorphic robustness over {mc['n']} questions. Top-{TOP_K} Jaccard "
        "overlap of retrieval between the original and transformed query. Paraphrase should preserve "
        "the result set (high overlap); entity swap should change it (low overlap). Oracle-free.}\n"
        "\\label{tab:protocol-c-results}\n\\begin{tabular}{@{}llc@{}}\n\\toprule\n"
        "\\textbf{Transformation} & \\textbf{System} & \\textbf{Top-25 Jaccard} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
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
        "\\begin{table}[h]\n\\centering\\small\n"
        f"\\caption{{Protocol B: oracle-free claim-to-evidence faithfulness over {o['n_claims']} atomic "
        "claims, by question family. Each claim is extracted from a grounded answer synthesised from the "
        "system's own top-retrieved evidence and scored by NLI (\\texttt{roberta-large-mnli}) against that "
        "evidence; faithfulness = entailed/total, unsupported = no evidence sentence found, contradiction = "
        "NLI contradicts. No gold answer or gold evidence is used.}\n"
        "\\label{tab:protocol-b-results}\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "\\textbf{Family} & \\textbf{Faith.} & \\textbf{Unsup.} & \\textbf{Contra.} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
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
        "\\begin{table}[h]\n\\centering\\small\n"
        f"\\caption{{Retrievability access-breadth probe (Azzopardi \\& Vinay) over {rt['n_queries']} "
        f"simulated queries drawn from the corpus, top-{rt['top_k']} (max {max_slots:,} retrieval slots). "
        "``Docs reached'' is the number of distinct documents retrieved at least once: closer to the "
        f"{max_slots:,}-slot ceiling means less repetition across queries, i.e.\\ broader corpus access. "
        "The corpus-wide Gini is reported for completeness but is budget-saturated here (a query count "
        "far below the corpus size cannot help but leave almost all documents unreached); the comparative "
        "docs-reached figure is the interpretable signal. Oracle-free: no gold, no LLM.}\n"
        "\\label{tab:retrievability}\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "\\textbf{System} & \\textbf{Gini} & \\textbf{Distinct docs reached $\\uparrow$} & \\textbf{Coverage} \\\\\n\\midrule\n"
        f"{rows}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )
    _w(LATEX_GEN_DIR / "results_retrievability.tex", tex)


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
        try:
            render_plots(summary_a, mc)
        except Exception as exc:
            print(f"[render] plot skipped: {exc}")
    print("\nDONE.")
