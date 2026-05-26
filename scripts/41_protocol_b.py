"""Protocol B: claim-to-evidence support labelling (faithfulness).

For each benchmark question, take the agent's grounded synthesis output,
decompose it into atomic claims, retrieve candidate evidence sentences from
the cited documents, and score each claim against its best evidence sentence
with an NLI verifier. Report faithfulness = entailed_claims / total_claims
per question and per question family.

Three phases mirror Protocol A's structure; each phase is independently
toggleable and disk-cached:

  Phase 1 (decompose):  read agent_runtime outputs -> atomic claim spans
                        -> outputs/protocol_b/claims/{question_id}.json
  Phase 2 (verify):     for each (claim, candidate_evidence) -> NLI label
                        -> outputs/protocol_b/nli_cache/{hash}.json
  Phase 3 (metrics):    aggregate per question, per family
                        -> outputs/protocol_b/results/{timestamp}.{json,csv}

Phase 2 uses the same OpenAI-compatible LLM client as Protocol A for claim
extraction; the NLI step uses a local roberta-large-mnli (faithfulness.py)
or, if that is not available on the host, the same LLM as a fallback
entailment classifier (clearly marked in the cache file).

Run:
    python scripts/41_protocol_b.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ============================================================================
# CONFIG
# ============================================================================

QUESTIONS_PATH = PROJECT_ROOT / "config" / "smoke_questions_10_rows.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "protocol_b"

# Reads the most recent agent_runtime output per question. Set this to a
# specific subfolder if you want to evaluate a frozen run set.
AGENT_RUNTIME_ROOT = PROJECT_ROOT / "outputs" / "agent_runtime"

# Claim extraction LLM. Same provider config as Protocol A.
CLAIM_EXTRACTOR_MODEL = "gpt-5.4-nano-2026-03-17"

# NLI: prefer local roberta-large-mnli; fall back to LLM judge if unavailable.
NLI_USE_LOCAL = True
NLI_FALLBACK_LLM_MODEL = "gpt-5.4-nano-2026-03-17"

# Phase toggles
PHASE_DECOMPOSE = True
PHASE_VERIFY = True
PHASE_METRICS = True

# Map question family prefix -> readable label (used in aggregation).
FAMILY_PREFIXES = {
    "A": "A. Distribution",
    "B": "B. Comparative",
    "C": "C. Temporal framing",
    "D": "D. Prediction",
    "E": "E. Metadata-conditional",
    "F": "F. Corpus + external",
}

# Faithfulness threshold: entailment probability above which a claim is
# counted as "entailed". 0.5 is the canonical NLI argmax cut.
ENTAILMENT_THRESHOLD = 0.5

# ============================================================================
# Prompts
# ============================================================================

CLAIM_PROMPT_SYSTEM = (
    "You decompose AI-generated answers into atomic factual claims for "
    "faithfulness evaluation. Each claim must be (1) self-contained "
    "(pronouns resolved, entities named), (2) a single factual assertion, "
    "(3) verifiable against documentary evidence. Drop opinion, hedging, "
    "and meta-commentary. Return JSON exactly of the form "
    '{"claims": [<string>, <string>, ...]}. No prose outside the JSON.'
)

CLAIM_PROMPT_USER = (
    "ANSWER TEXT:\n{answer}\n\n"
    "Decompose into atomic factual claims. Output the JSON object now."
)

NLI_FALLBACK_SYSTEM = (
    "You are an entailment classifier. Given a premise (evidence sentence) "
    "and a hypothesis (claim), decide whether the premise entails, is "
    "neutral with respect to, or contradicts the hypothesis. Return JSON "
    'exactly of the form {"label": "entails"|"neutral"|"contradicts", '
    '"score": <float 0..1, confidence in the chosen label>}. '
    "No prose outside the JSON."
)
NLI_FALLBACK_USER = "PREMISE: {premise}\n\nHYPOTHESIS: {hypothesis}\n\nOutput the JSON object now."

# ============================================================================
# Helpers
# ============================================================================


def _claims_path(qid: str) -> Path:
    return OUTPUT_DIR / "claims" / f"{qid}.json"


def _nli_cache_path(phash: str) -> Path:
    return OUTPUT_DIR / "nli_cache" / f"{phash}.json"


def _nli_hash(premise: str, hypothesis: str, model_tag: str) -> str:
    h = hashlib.sha256()
    h.update(model_tag.encode("utf-8"))
    h.update(b"\x1f")
    h.update(premise.encode("utf-8"))
    h.update(b"\x1f")
    h.update(hypothesis.encode("utf-8"))
    return h.hexdigest()[:32]


def _split_sentences(text: str) -> list[str]:
    # naive splitter that's good enough for evidence retrieval; replace with
    # spaCy if available downstream
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[\.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sents if s.strip()]


def _best_evidence_sentence(claim: str, candidate_text: str) -> str:
    """Highest lexical-overlap sentence from candidate_text."""
    claim_tokens = set(re.findall(r"\w+", claim.lower()))
    if not claim_tokens:
        return ""
    best, best_score = "", 0
    for sent in _split_sentences(candidate_text):
        sent_tokens = set(re.findall(r"\w+", sent.lower()))
        overlap = len(claim_tokens & sent_tokens)
        if overlap > best_score:
            best, best_score = sent, overlap
    return best


def _family_of(qid: str) -> str:
    if not qid:
        return "unknown"
    first = qid[0].upper()
    if qid.lower().startswith("q") and len(qid) >= 2:
        first = qid[1].upper()
    return FAMILY_PREFIXES.get(first, "unknown")


def _read_most_recent_run(qid: str) -> dict | None:
    """Pull the most-recent agent_runtime output for this question_id.

    Manifest structure (per src/corpusagent2/agent_runtime.py): each run dir
    contains run_manifest.json with the synthesised answer + cited evidence.
    Returns dict with keys: synthesis_text, evidence_docs (list of {id, text}).
    """
    if not AGENT_RUNTIME_ROOT.exists():
        return None
    candidates = []
    for run_dir in AGENT_RUNTIME_ROOT.iterdir():
        manifest = run_dir / "run_manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(data.get("question_id", "")) == qid or str(data.get("raw_question_id", "")) == qid:
            candidates.append((run_dir, data))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
    _, manifest_data = candidates[0]
    synthesis = str(manifest_data.get("synthesis_text") or manifest_data.get("answer") or "")
    evidence = []
    for ev in manifest_data.get("evidence", []) or manifest_data.get("evidence_docs", []) or []:
        evidence.append({
            "doc_id": str(ev.get("doc_id") or ev.get("id") or ""),
            "text": str(ev.get("text") or ev.get("body") or ev.get("snippet") or ""),
        })
    return {"synthesis_text": synthesis, "evidence_docs": evidence}


# ============================================================================
# Phase 1: claim decomposition
# ============================================================================


def run_decompose(questions: list[dict]) -> None:
    from corpusagent2.llm_provider import LLMProviderConfig, OpenAICompatibleLLMClient

    provider = LLMProviderConfig(
        base_url=os.getenv("CORPUSAGENT2_OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        timeout_s=float(os.getenv("CORPUSAGENT2_LLM_TIMEOUT_S", "60")),
        verify_ssl=True,
    )
    if not provider.api_key:
        raise SystemExit("OPENAI_API_KEY required for claim extraction.")
    client = OpenAICompatibleLLMClient(provider)

    (OUTPUT_DIR / "claims").mkdir(parents=True, exist_ok=True)

    for q in questions:
        qid = q["question_id"]
        out_path = _claims_path(qid)
        if out_path.exists():
            print(f"[decompose] {qid}: cached, skipping")
            continue

        run = _read_most_recent_run(qid)
        if run is None or not run["synthesis_text"]:
            print(f"[decompose] {qid}: no agent_runtime output found, skipping")
            continue

        try:
            parsed = client.complete_json(
                [
                    {"role": "system", "content": CLAIM_PROMPT_SYSTEM},
                    {"role": "user", "content": CLAIM_PROMPT_USER.format(answer=run["synthesis_text"])},
                ],
                model=CLAIM_EXTRACTOR_MODEL,
                temperature=0.0,
            )
            claims = [str(c).strip() for c in parsed.get("claims", []) if str(c).strip()]
        except Exception as exc:
            print(f"[decompose] {qid}: claim extraction failed -- {type(exc).__name__}: {exc}")
            continue

        out_path.write_text(json.dumps({
            "question_id": qid,
            "question": q.get("raw_question", ""),
            "synthesis_text": run["synthesis_text"],
            "claims": claims,
            "evidence_docs": run["evidence_docs"],
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "extractor_model": CLAIM_EXTRACTOR_MODEL,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[decompose] {qid}: extracted {len(claims)} claims")


# ============================================================================
# Phase 2: NLI verification
# ============================================================================


def _verify_claim_local(claim: str, evidence: str) -> tuple[str, float]:
    """Local roberta-large-mnli; returns (label, entailment_score)."""
    from corpusagent2.faithfulness import NLIVerifier
    verifier = NLIVerifier()
    verdict = verifier.score(premise=evidence, hypothesis=claim)
    return verdict.label, float(verdict.entailment_score)


def _verify_claim_llm(claim: str, evidence: str, client, model: str) -> tuple[str, float]:
    parsed = client.complete_json(
        [
            {"role": "system", "content": NLI_FALLBACK_SYSTEM},
            {"role": "user", "content": NLI_FALLBACK_USER.format(premise=evidence, hypothesis=claim)},
        ],
        model=model, temperature=0.0,
    )
    label = str(parsed.get("label", "neutral"))
    score = float(parsed.get("score", 0.5))
    # Normalise to an entailment-probability-like number for thresholding
    if label == "entails":
        ent = max(score, ENTAILMENT_THRESHOLD)
    elif label == "contradicts":
        ent = min(1.0 - score, ENTAILMENT_THRESHOLD)
    else:
        ent = 0.5 if score < 0.5 else 0.5
    return label, ent


def run_verify(questions: list[dict]) -> None:
    (OUTPUT_DIR / "nli_cache").mkdir(parents=True, exist_ok=True)

    nli_backend = "local"
    client = None
    model_tag = "roberta-large-mnli"
    if not NLI_USE_LOCAL:
        nli_backend = "llm_fallback"
        from corpusagent2.llm_provider import LLMProviderConfig, OpenAICompatibleLLMClient
        client = OpenAICompatibleLLMClient(LLMProviderConfig(
            base_url=os.getenv("CORPUSAGENT2_OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            timeout_s=float(os.getenv("CORPUSAGENT2_LLM_TIMEOUT_S", "60")),
            verify_ssl=True,
        ))
        model_tag = NLI_FALLBACK_LLM_MODEL

    verified = 0
    cached = 0
    failed = 0

    for q in questions:
        qid = q["question_id"]
        claims_path = _claims_path(qid)
        if not claims_path.exists():
            continue
        data = json.loads(claims_path.read_text(encoding="utf-8"))
        evidence_text = "\n".join(d["text"] for d in data["evidence_docs"] if d.get("text"))
        for claim in data["claims"]:
            premise = _best_evidence_sentence(claim, evidence_text)
            if not premise:
                # No supporting evidence sentence found
                phash = _nli_hash("", claim, model_tag)
                _nli_cache_path(phash).write_text(json.dumps({
                    "question_id": qid, "claim": claim, "premise": "",
                    "label": "unsupported", "entailment_score": 0.0,
                    "backend": nli_backend, "model_tag": model_tag,
                }, indent=2, ensure_ascii=False), encoding="utf-8")
                continue
            phash = _nli_hash(premise, claim, model_tag)
            if _nli_cache_path(phash).exists():
                cached += 1
                continue
            try:
                if nli_backend == "local":
                    label, ent = _verify_claim_local(claim, premise)
                else:
                    label, ent = _verify_claim_llm(claim, premise, client, NLI_FALLBACK_LLM_MODEL)
            except Exception as exc:
                failed += 1
                print(f"[verify] {qid}: NLI failed -- {type(exc).__name__}: {exc}")
                continue

            _nli_cache_path(phash).write_text(json.dumps({
                "question_id": qid, "claim": claim, "premise": premise,
                "label": label, "entailment_score": float(ent),
                "backend": nli_backend, "model_tag": model_tag,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            verified += 1

    print(f"[verify] DONE: {verified} new, {cached} cached, {failed} failed (backend={nli_backend})")


# ============================================================================
# Phase 3: metrics
# ============================================================================


def run_metrics(questions: list[dict]) -> None:
    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)

    per_question: list[dict] = []
    for q in questions:
        qid = q["question_id"]
        claims_path = _claims_path(qid)
        if not claims_path.exists():
            continue
        data = json.loads(claims_path.read_text(encoding="utf-8"))
        evidence_text = "\n".join(d["text"] for d in data["evidence_docs"] if d.get("text"))

        entailed = unsupported = contradicted = neutral = 0
        total = 0
        for claim in data["claims"]:
            total += 1
            premise = _best_evidence_sentence(claim, evidence_text)
            if not premise:
                unsupported += 1
                continue
            phash = _nli_hash(premise, claim, "roberta-large-mnli" if NLI_USE_LOCAL else NLI_FALLBACK_LLM_MODEL)
            cache_path = _nli_cache_path(phash)
            if not cache_path.exists():
                continue
            verdict = json.loads(cache_path.read_text(encoding="utf-8"))
            label = verdict.get("label", "neutral")
            score = float(verdict.get("entailment_score", 0.0))
            if label == "entails" or (label == "entailed" and score >= ENTAILMENT_THRESHOLD):
                entailed += 1
            elif label == "contradicts":
                contradicted += 1
            elif label == "unsupported":
                unsupported += 1
            else:
                neutral += 1

        if total == 0:
            continue
        per_question.append({
            "question_id": qid,
            "family": _family_of(qid),
            "n_claims": total,
            "entailed": entailed,
            "neutral": neutral,
            "contradicted": contradicted,
            "unsupported": unsupported,
            "faithfulness": round(entailed / total, 4),
        })

    # Aggregate per family
    family_agg: dict[str, dict] = {}
    for row in per_question:
        fam = row["family"]
        agg = family_agg.setdefault(fam, {"n_questions": 0, "claims": 0, "entailed": 0, "contradicted": 0, "unsupported": 0, "neutral": 0})
        agg["n_questions"] += 1
        agg["claims"] += row["n_claims"]
        agg["entailed"] += row["entailed"]
        agg["contradicted"] += row["contradicted"]
        agg["unsupported"] += row["unsupported"]
        agg["neutral"] += row["neutral"]
    for fam, agg in family_agg.items():
        agg["faithfulness"] = round(agg["entailed"] / agg["claims"], 4) if agg["claims"] else 0.0
        agg["contradiction_rate"] = round(agg["contradicted"] / agg["claims"], 4) if agg["claims"] else 0.0
        agg["unsupported_rate"] = round(agg["unsupported"] / agg["claims"], 4) if agg["claims"] else 0.0

    summary = {
        "nli_backend": "local" if NLI_USE_LOCAL else "llm_fallback",
        "extractor_model": CLAIM_EXTRACTOR_MODEL,
        "entailment_threshold": ENTAILMENT_THRESHOLD,
        "questions_evaluated": len(per_question),
        "per_family": family_agg,
        "per_question": per_question,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (OUTPUT_DIR / "results" / f"{stamp}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_lines = ["family,n_questions,n_claims,faithfulness,contradiction_rate,unsupported_rate"]
    for fam, agg in family_agg.items():
        csv_lines.append(
            f"{fam},{agg['n_questions']},{agg['claims']},{agg['faithfulness']},"
            f"{agg['contradiction_rate']},{agg['unsupported_rate']}"
        )
    (OUTPUT_DIR / "results" / f"{stamp}.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print(f"[metrics] wrote results/{stamp}.json")
    print()
    print("=== faithfulness summary by family ===")
    for fam, agg in family_agg.items():
        print(f"  {fam:30s}  faith={agg['faithfulness']:.4f}  "
              f"contra={agg['contradiction_rate']:.4f}  unsup={agg['unsupported_rate']:.4f}  "
              f"(n_q={agg['n_questions']}, n_c={agg['claims']})")


# ============================================================================
# main
# ============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")
    print()

    if PHASE_DECOMPOSE:
        print("=== Phase 1: claim decomposition ===")
        run_decompose(questions)
        print()
    if PHASE_VERIFY:
        print("=== Phase 2: NLI verification ===")
        run_verify(questions)
        print()
    if PHASE_METRICS:
        print("=== Phase 3: faithfulness aggregation ===")
        run_metrics(questions)
