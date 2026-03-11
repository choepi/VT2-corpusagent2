from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_utils import sentence_split


@dataclass(slots=True)
class ClaimVerdict:
    claim_id: str
    claim: str
    label: str
    entailment: float
    contradiction: float
    neutral: float
    best_evidence_sentence: str

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim": self.claim,
            "label": self.label,
            "entailment": self.entailment,
            "contradiction": self.contradiction,
            "neutral": self.neutral,
            "best_evidence_sentence": self.best_evidence_sentence,
        }


class NLIVerifier:
    def __init__(self, model_id: str, device: int = -1) -> None:
        from transformers import pipeline

        self.model_id = model_id
        self.pipe = pipeline(
            "text-classification",
            model=model_id,
            tokenizer=model_id,
            return_all_scores=True,
            device=device,
        )

    def score(self, premise: str, hypothesis: str) -> dict[str, float]:
        scores = self.pipe({"text": premise, "text_pair": hypothesis}, truncation=True)
        if not scores:
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}

        row: list[dict] = []
        if isinstance(scores, dict):
            row = [scores]
        elif isinstance(scores, list):
            if scores and isinstance(scores[0], dict):
                row = scores
            elif scores and isinstance(scores[0], list):
                row = scores[0]

        if not row:
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}

        mapped: dict[str, float] = {}
        for entry in row:
            if not isinstance(entry, dict):
                continue
            if "label" not in entry or "score" not in entry:
                continue
            label = entry["label"].lower()
            value = float(entry["score"])
            if "entail" in label or label.endswith("2"):
                mapped["entailment"] = value
            elif "contra" in label or label.endswith("0"):
                mapped["contradiction"] = value
            elif "neutral" in label or label.endswith("1"):
                mapped["neutral"] = value

        mapped.setdefault("entailment", 0.0)
        mapped.setdefault("contradiction", 0.0)
        mapped.setdefault("neutral", 0.0)
        return mapped


def lexical_overlap_score(claim: str, sentence: str) -> float:
    claim_tokens = {token.lower() for token in claim.split() if token.strip()}
    sentence_tokens = {token.lower() for token in sentence.split() if token.strip()}
    if not claim_tokens or not sentence_tokens:
        return 0.0
    overlap = claim_tokens.intersection(sentence_tokens)
    return len(overlap) / float(len(claim_tokens))


def pick_best_evidence_sentence(claim: str, evidence_text: str, max_sentences: int = 8) -> str:
    sentences = sentence_split(evidence_text)
    if not sentences:
        return ""

    scored = sorted(
        ((sentence, lexical_overlap_score(claim, sentence)) for sentence in sentences),
        key=lambda pair: pair[1],
        reverse=True,
    )
    shortlist = scored[:max_sentences]
    if not shortlist:
        return ""
    return shortlist[0][0]


def evaluate_claims_with_nli(
    verifier: NLIVerifier,
    claims: list[dict],
    doc_text_by_id: dict[str, str],
) -> tuple[list[ClaimVerdict], dict]:
    verdicts: list[ClaimVerdict] = []

    for item in claims:
        claim_id = str(item["claim_id"])
        claim = str(item["claim"]) 
        evidence_doc_ids = [str(doc_id) for doc_id in item.get("evidence_doc_ids", [])]

        evidence_text = " ".join(doc_text_by_id.get(doc_id, "") for doc_id in evidence_doc_ids)
        best_sentence = pick_best_evidence_sentence(claim, evidence_text)
        if not best_sentence:
            verdicts.append(
                ClaimVerdict(
                    claim_id=claim_id,
                    claim=claim,
                    label="unsupported",
                    entailment=0.0,
                    contradiction=0.0,
                    neutral=1.0,
                    best_evidence_sentence="",
                )
            )
            continue

        nli_scores = verifier.score(premise=best_sentence, hypothesis=claim)
        label = max(nli_scores, key=nli_scores.get)

        verdicts.append(
            ClaimVerdict(
                claim_id=claim_id,
                claim=claim,
                label=label,
                entailment=nli_scores["entailment"],
                contradiction=nli_scores["contradiction"],
                neutral=nli_scores["neutral"],
                best_evidence_sentence=best_sentence,
            )
        )

    entailed = sum(1 for item in verdicts if item.label == "entailment")
    contradicted = sum(1 for item in verdicts if item.label == "contradiction")
    unsupported = sum(1 for item in verdicts if item.label not in {"entailment", "contradiction", "neutral"})

    summary = {
        "total_claims": len(verdicts),
        "entailed_claims": entailed,
        "faithfulness": float(entailed / len(verdicts)) if verdicts else 0.0,
        "contradiction_rate": float(contradicted / len(verdicts)) if verdicts else 0.0,
        "unsupported_rate": float(unsupported / len(verdicts)) if verdicts else 0.0,
        "mean_entailment": float(np.mean([item.entailment for item in verdicts])) if verdicts else 0.0,
    }
    return verdicts, summary
