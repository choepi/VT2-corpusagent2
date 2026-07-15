# Scaling-curve experiment: seeded growing-corpus evaluation for RQ1 + RQ4

**Date:** 2026-07-14
**Status:** IMPLEMENTED — phases live in `scripts/50_run_eval_suite.py`; full 21-condition run NOT yet executed (smoke-tested only, see §8).
**Extends:** `scripts/50_run_eval_suite.py` (new phases; no new script, no argparse, per suite convention).
**Paper target:** Chapter 4 — new "Retrieval quality across corpus scale" subsection under Protocol A results, plus the pending RQ4 architecture benchmark (`sec:arch-tradeoff`), which this design finally executes with measured anchors.

## 1. Motivation (what the paper currently lacks)

- **RQ1** asks how retrieval quality changes *as corpus size grows* and which signals carry the load *at each scale* — but Protocol A measures exactly one scale (624k). The only at-scale evidence is runtime, not quality.
- **RQ4**'s FAISS-vs-pgvector benchmark is entirely pending; the 13M projection table was deliberately cut because "a projection table without measured anchors would be decoration."
- A two-point (624k vs 13M) comparison is an anecdote. A seeded multi-scale curve with dispersion bands is measured evidence.

## 2. Agreed design decisions (2026-07-14 discussion)

1. **Cap the curve at 624k.** The VM slice has exactly 624,095 embedded docs. Scales above that would require the 13.36M `Geralt-Targaryen/CC-News` instance, which is a *different packaging* (different dedup, year-level dating) — blending it into the same curve is a confound. The 13M evidence stays what it is today (runtime/feasibility, separate sections).
2. **Sizes and seeds:** subset sizes **10k, 50k, 100k, 250k**, each drawn **5×** with seeds derived from one master seed; plus the **full 624k as a single deterministic top point** (no seeds). Rationale: two random 500k draws from 624k share ≥~80% of documents, so seed error bars at 500k would be overlap-deflated fake precision. A 1k size was considered and dropped (2026-07-14): at 1k docs most of the 11 open-ended questions would have zero judge-relevant documents in-sample, making the point floor-noise rather than evidence. The smallest size, 10k, may still show partial floor effects for narrow questions — reported as-is, not hidden.
3. **Independent draws per (size, seed)** (not nested subsets): cleaner variance interpretation; documents recur across draws anyway, which the judge cache exploits.
4. **Per-subset lexical indexes are mandatory.** IDF/avgdl shift with collection size — that *is* part of the scaling phenomenon. Post-hoc filtering of results from the full 624k OpenSearch index would silently keep full-corpus statistics and is scientifically wrong here.
5. **Dense retrieval subsets exactly, no re-embedding.** Dense scoring is per-document (no collection statistics), so restricting the existing 624k×768 E5 matrix to subset rows and scanning exactly is equivalent to a per-subset index, with zero embedding cost. Backend for the quality arm: exact scan (local matrix or pgvector `WHERE doc_id IN` without ANN index); ANN approximation is measured separately in the RQ4 arm so quality and index effects don't mix.
6. **Both arms:** RQ1 quality curve AND RQ4 ANN benchmark, sharing the same subsets. Retrievability/Gini-per-scale explicitly deferred (would add a third results section).

## 3. RQ1 arm — Protocol A across scale

Per condition (4 sizes × 5 seeds + 624k full = 21 conditions), re-run the existing Protocol A phases against the subset:

- **Systems (unchanged):** lexical BM25 (per-subset OpenSearch index), dense E5 (exact subset scan), hybrid RRF, hybrid + cross-encoder rerank. `TOP_K = 25` pooling as today.
- **Questions (unchanged):** the 11 canonical questions from `config/eval_questions_11.json`.
- **Judge (unchanged):** pinned `gpt-5.4-nano`, existing per-(question, doc, model) judgment cache — subsets share documents heavily across sizes/seeds, so most judgments are cache hits after the first conditions. The cross-judge ablation stays at 624k only (budget).
- **Metrics:** nDCG@10 (primary; judge labels are absolute per pair, so comparable across sizes), pooled Recall@25 and MAP (with the standing caveat that the pool is size-relative), reported as **mean ± std over the 5 seeds** per size; the 624k point keeps its current bootstrap CI. Additionally: per-size system *ranking* (answers "which signal carries the load at each scale" directly — e.g. whether rerank's dominance holds at 10k or only emerges at scale).
- **Semantics caveat (must appear in the figure caption):** oracle-free Protocol A measures "quality of what the systems surface from a corpus of size N," not recall of a fixed relevant set — the candidate pool itself changes with N.

**Output:** `generated/plot_scaling_rq1.png` (x = corpus size, log scale; y = nDCG@10; one line per system; shaded ±1 std bands; secondary panel or figure for pooled Recall@25) + `generated/results_scaling_rq1.tex` (per-size × per-system table, mean ± std).

## 4. RQ4 arm — ANN architecture benchmark across scale

On the same subsets (seed 1 of each size is sufficient for index characteristics; document if more seeds are used), plus the full 624k:

- **Architectures:** exact flat scan (baseline = self-referential ground truth, oracle-free), FAISS IVF-PQ, FAISS HNSW, pgvector IVFFlat (production config), pgvector HNSW. These are exactly the architectures RQ4 names.
- **Per (architecture, size), measure:** build wall-time, on-disk/in-memory index size, recall@10 and recall@25 against the flat baseline, and per-query latency (p50/p95) — query set = the 11 question embeddings plus ~1,000 sampled document vectors for a stable latency distribution.
- **Hardware note:** all measurements on the production CPU VM → latencies are internally comparable across the whole curve and honestly labeled CPU-only. No cross-machine latency mixing.
- This turns the cut "projection to 13M" into an extrapolation *from five measured anchors* per architecture — the paper can reinstate the projection with a defensible basis.

**Output:** `generated/plot_scaling_rq4.png` (recall-vs-flat and p50 latency vs corpus size per architecture) + `generated/results_scaling_rq4.tex`.

## 5. Implementation sketch (suite phases)

New env-gated phases in `scripts/50_run_eval_suite.py` following the existing `_phase()` convention:

1. `scaling_subsets` — draw the 20 seeded doc_id subsets from the 624k universe (master seed in the CONFIG block; per-condition seed = f(master, size, replicate)); write manifests to `outputs/eval_suite/scaling/subsets/`.
2. `scaling_retrieve` — per condition: create the per-subset OpenSearch index (reusing the bulk-index path of `scripts/21_*`), run the four systems, write pools per (condition, question); **build → measure → drop** each OpenSearch index sequentially to bound disk on the 90 GB VM (~2.05M docs total across all conditions if kept simultaneously — do not keep them).
3. `scaling_judge` — judge new (question, doc) pairs only (cache does the rest).
4. `scaling_metrics` — aggregate to the RQ1 tables/plot.
5. `scaling_ann_bench` — the RQ4 arm (FAISS via `faiss-cpu`, pgvector via temp tables with timed index builds, dropped after).

Estimated cost: low-thousands of *new* nano-judge calls (rest cached); retrieval + index builds run overnight on the VM. **Scheduling constraint: the lab VM reboots Saturdays ~23:00** — phases are per-condition resumable (skip conditions whose outputs exist), so a reboot loses at most one condition.

## 6. Paper integration (after execution)

- New subsection in Protocol A results: "Retrieval quality across corpus scale (10k–624k, seeded)" with `plot_scaling_rq1` + table; one paragraph interpreting where system separation emerges and any small-size floor effects.
- `sec:arch-tradeoff`: replace the "pending" status text with the measured RQ4 results across scale; optionally reinstate the 13M projection anchored on the measured curve.
- Copy refreshed `generated/` artifacts to the OneDrive paper workspace (`zhaw_onedrive:MSE_school_files/Sem4/project_paper/LATEX/generated/`).

## 7. How to run (implemented 2026-07-14)

Both arms are **opt-in** (a first run is overnight-scale; a plain suite run stays unchanged):

```bash
# RQ1 curve: subsets + per-scale retrieval + judging + metrics + render
EVAL_RUN_SCALING=1 .venv/bin/python scripts/50_run_eval_suite.py

# RQ4 ANN benchmark (can run separately or together with the above)
EVAL_RUN_SCALING_ANN=1 .venv/bin/python scripts/50_run_eval_suite.py
```

Useful knobs: `EVAL_SCALING_SIZES` (csv, default `10000,50000,100000,250000`), `EVAL_SCALING_REPS`
(default 5), `EVAL_SCALING_FULL=0` to skip the full-624k point, `EVAL_RUN_SCALING_JUDGE=0` to defer
judging. FAISS variants need `uv sync --extra ann-bench` (the phase degrades gracefully without it).

Implementation notes (what a reviewer would ask):

- **Judge-cache compatibility is deliberate:** scaling pool files derive their judged text exactly like
  `phase_retrieve` (raw document text truncated to 360 chars), so the per-(prompt, question, doc, model)
  hash matches and judgments are shared with the main Protocol A run and across all conditions.
- **Subsets:** seeded from `random.Random(f"{SEED}:{size}:{rep}")` over the dense universe
  (`data/indices/dense/dense_doc_ids.joblib`); the manifest records a universe SHA and the phase refuses
  to mix subsets drawn against a different corpus. Stored as row-index `.npy` under
  `outputs/eval_suite/scaling/subsets/`.
- **Lexical per scale:** a real OpenSearch index per condition (`ca2-scaling-*`, mapping identical to
  `scripts/21_bulk_index_opensearch.py`), built → measured → dropped so disk usage stays bounded.
  The full-624k point uses the production index (it *is* that condition's index).
- **Dense per scale:** the full-corpus query scores are computed once per question and masked to the
  subset — mathematically identical to a per-subset exact index, zero re-embedding.
- **Hybrid/rerank semantics** mirror `HybridSearchBackend.search`: candidate depth `max(5*top_k, 50)`,
  RRF k=60, cross-encoder rerank of the fused top-25.
- **RQ4 measurements** (`outputs/eval_suite/scaling/ann_bench.json`): per size (seed 1) and per
  architecture — build s, index MB, recall@10 vs the exact flat scan over the same vectors, per-query
  p50/p95 latency; IVF-type architectures at two operating points (probes/nprobe ∈ {1, 10} — probes=1
  is the pgvector production default, and the gap is itself a finding). The full-624k pgvector point
  reuses the production table/index (build time not re-measured, marked "--"). Queries = the 11
  question embeddings + 1,000 seeded document vectors.
- **Outputs:** `outputs/eval_suite/scaling_rq1.json` / `scaling_rq4.json`;
  `generated/plot_scaling_rq1.png`, `results_scaling_rq1.tex`, `plot_scaling_rq4.png`,
  `results_scaling_rq4.tex`. A `RENDER`-only re-run refreshes the LaTeX artifacts from the cached JSON.

## 8. Smoke-test record

Smoke run 2026-07-14 (CPU VM, live OpenSearch + Postgres + OpenAI judge; exit 0): 2 questions
(q01/q02), sizes 3k/8k × 2 seeds, no full point, both arms + render.

- Subset draw, per-condition BM25 index build (3–17 s at these sizes) → retrieve → drop: all 4
  conditions, no leftover `ca2-scaling-*` indexes or pg temp tables afterwards.
- Judge: 338 new / **87 cached / 0 failed** — the cache hits are shared judgments from the main
  Protocol A run, confirming hash-compatible pool texts.
- RQ1 curve behaved sanely (rerank on top at both sizes; dense weak at tiny corpus; large seed
  bands at n=2 seeds as expected). RQ4 produced the full architecture table; already visible in the
  smoke data: pgvector IVFFlat at the production default `probes=1` loses 21–30 % recall vs
  `probes=10`, while HNSW variants hold ≈0.98–0.99 recall — exactly the trade-off RQ4 asks about.
- Both plots and both `.tex` tables rendered and were visually checked.
- Caveat found and handled: a suite run with a non-default question set also re-renders
  `results_protocol_a.tex` from that question set. After the smoke test the real N=11 artifacts were
  restored with a RENDER-only pass (headline reproduced: nDCG@10 0.242/0.207/0.269/0.391). For the
  real scaling run this is a non-issue (it uses the default 11 questions).
- All smoke artifacts (subsets, pools, JSON summaries, generated smoke plots/tables) were deleted;
  the judge cache keeps the 338 new judgments (they are valid (question, doc, model) labels and will
  be reused).

## 9. Explicitly rejected alternatives (for the record)

- **1M point / blended curve** — rejected: requires the different 13M packaging; cross-packaging confound.
- **5 seeds at 500k** — rejected: ≥~80% pairwise draw overlap deflates variance; replaced by 250k seeded + 624k full.
- **Post-hoc filtering of full-index BM25 results** — rejected: keeps full-corpus IDF statistics; per-subset indexes required.
- **Retrievability/Gini per scale** — deferred, not rejected; noted as a candidate extension.
