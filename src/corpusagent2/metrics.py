from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


def dcg_at_k(relevances: Iterable[int], k: int) -> float:
    rel = list(relevances)[:k]
    total = 0.0
    for idx, value in enumerate(rel, start=1):
        total += (2**value - 1) / math.log2(idx + 1)
    return total


def ndcg_at_k(predicted_doc_ids: list[str], relevant_doc_ids: set[str], k: int = 10) -> float:
    gained = [1 if doc_id in relevant_doc_ids else 0 for doc_id in predicted_doc_ids[:k]]
    ideal = sorted(gained, reverse=True)
    denom = dcg_at_k(ideal, k)
    if denom == 0.0:
        return 0.0
    return dcg_at_k(gained, k) / denom


def mrr_at_k(predicted_doc_ids: list[str], relevant_doc_ids: set[str], k: int = 10) -> float:
    for idx, doc_id in enumerate(predicted_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / float(idx)
    return 0.0


def recall_at_k(predicted_doc_ids: list[str], relevant_doc_ids: set[str], k: int = 100) -> float:
    if not relevant_doc_ids:
        return 0.0
    hits = sum(1 for doc_id in predicted_doc_ids[:k] if doc_id in relevant_doc_ids)
    return hits / float(len(relevant_doc_ids))


def evidence_completeness(predicted_doc_ids: list[str], gold_evidence_doc_ids: set[str]) -> float:
    if not gold_evidence_doc_ids:
        return 0.0
    retrieved = set(predicted_doc_ids)
    matched = len(retrieved.intersection(gold_evidence_doc_ids))
    return matched / float(len(gold_evidence_doc_ids))


@dataclass(slots=True)
class PairedTestResult:
    mean_delta: float
    t_statistic: float | None
    p_value: float | None


def paired_t_test(new_scores: list[float], baseline_scores: list[float]) -> PairedTestResult:
    if len(new_scores) != len(baseline_scores):
        raise ValueError("paired_t_test requires equal-length score vectors")
    if not new_scores:
        return PairedTestResult(mean_delta=0.0, t_statistic=None, p_value=None)

    deltas = np.array(new_scores, dtype=np.float64) - np.array(baseline_scores, dtype=np.float64)
    if stats is None:
        return PairedTestResult(mean_delta=float(np.mean(deltas)), t_statistic=None, p_value=None)

    test = stats.ttest_rel(np.array(new_scores), np.array(baseline_scores))
    return PairedTestResult(
        mean_delta=float(np.mean(deltas)),
        t_statistic=float(test.statistic),
        p_value=float(test.pvalue),
    )


@dataclass(slots=True)
class BootstrapCI:
    mean: float
    low: float
    high: float


def bootstrap_confidence_interval(
    scores: list[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    if not scores:
        return BootstrapCI(mean=0.0, low=0.0, high=0.0)

    rng = np.random.default_rng(seed)
    values = np.array(scores, dtype=np.float64)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = rng.choice(values, size=values.shape[0], replace=True)
        means.append(float(np.mean(sample)))

    means_np = np.array(means, dtype=np.float64)
    low = float(np.quantile(means_np, alpha / 2.0))
    high = float(np.quantile(means_np, 1.0 - alpha / 2.0))
    return BootstrapCI(mean=float(np.mean(values)), low=low, high=high)
