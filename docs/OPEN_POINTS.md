# OPEN_POINTS

Single source of truth for everything still open on this thesis project: paper writing, bibliography gaps, evaluation runs that need to be executed, implementation work for the scaling pathway, and miscellaneous repo cleanup. Update by editing this file directly; commit it with whatever change closes an item.

## Conventions

| Marker | Meaning |
|---|---|
| `[ ]` | open |
| `[~]` | in progress |
| `[x]` | done — leave in place under "Recently Closed" until at least the next commit cycle |
| `(P0)` | blocks something downstream; do first |
| `(P1)` | important; do this iteration |
| `(P2)` | nice to have; can slip |

When an item closes, move it to "Recently Closed" with the commit hash that closed it, and prune entries older than ~3 weeks.

---

## Paper

### Structure / prose

- [x] **DONE** Align chapters to the new four research questions (Retrieval Scalability, Precomputation vs Query-Time, Evidence Traceability at Scale, Architecture Trade-off) plus the central scaling RQ. Touched: abstract EN+DE (scaling RQ + 4 sub-RQs in first paragraph), Ch1 §1.2 problem statement (scaling framing), Ch1 §1.3 RQs (replaced), Ch1 §1.4 contributions (each item annotated with which RQ it serves), Ch4 chapter intro (explicit protocol→RQ table) + §4.5.1/§4.5.2 RQ tags + new §4.6.2 precomputation ablation for RQ2 + §4.7.4 corrected CorpusAgent-comparison RQ mapping, Ch5 §5.2 RQ revisit (replaced). Ch2 and Ch3 section structure unchanged (topical sections still map cleanly to new RQs as topical containers).
- [ ] **(P2)** Tighten Ch3 §3.1 architecture overview once Figure 3.1 (request-flow) is replaced with real TikZ — paragraph currently leans on the placeholder.
- [ ] **(P2)** Add a one-paragraph "Reading Guide" at the start of Ch3 once §3.1–§3.10 stabilise.
- [ ] **(P2)** Pass over Ch4 §4.4 (Metamorphic robustness) to make the four metamorphic relations concrete with one worked example each.

### Figures / diagrams

- [ ] **(P0)** Replace `[FIGURE PLACEHOLDER]` in Ch3 §3.1 (Figure 3.1, architecture / request-flow) with TikZ or Mermaid render.
- [ ] **(P1)** Replace `[ALGORITHM PLACEHOLDER]` in Ch3 §3.3.1 (Algorithm 3.1, `_compile_plan_dag` pseudocode) — keep current sketch but format as proper `algorithm2e` block in Appendix.
- [ ] **(P1)** Replace `[ALGORITHM PLACEHOLDER]` in Ch3 §3.3.3 (degraded-status propagation) — same treatment.
- [ ] **(P2)** Add one query-flow worked example figure to Ch4 §4.1: show a single question (e.g. Q3 NZZ-vs-Tagi football) going through QuestionSpec → PlanDAG → results.

### Bibliography

- [ ] **(P1)** Resolve DOIs / URLs for the 28 stub entries in `additional_refs.bib` via Zotero. Most-uncertain entries to verify first (author/year already best-effort; venue may be off):
  - `Lu2025RerankerBM25` — author placeholder is `et al.`
  - `Abdallah2025BEIRHybrid` — author placeholder is `et al.`
  - `Stuhlmann2025LongEval` — CLEF 2025 LongEval Lab notebook
  - `Koterwa2025BERTopicLayers` — BERTopic intermediate-layer paper
  - `Li2025HallucinationSurvey` — claudio_points cites arXiv 2510.24476
  - `Boumahdi2024BERTrend` — ACL 2024 FutureD workshop
  - `Bao2025FaithBench` — FaithBench
  - `RAGTruth2024` — Niu et al.
- [ ] **(P2)** Sanity-check the rest of `additional_refs.bib` against Zotero so every entry has DOI + (where applicable) arXiv ID + URL.
- [ ] **(P2)** Decide whether to keep `additional_refs.bib` as a separate file or merge stubs into `VT2.bib` once each is verified.

### Appendix

- [ ] **(P1)** Refresh `AppendixA` §A.1 (generative AI tools) with current model versions used this semester (Claude Code Opus 4.7 entry is added; ChatGPT/Claude entries from last semester may need updating).
- [ ] **(P2)** Move Algorithm 3.1 and 3.2 placeholder pseudocode out of `% comments` and into a proper `algorithm2e` block inside `AppendixA §A.3`.

### Compile + tooling

- [x] **DONE** clean MiKTeX compile, 42 pages, 0 undefined refs / cites (commit `af2aee3`)
- [ ] **(P2)** Commit `project_paper/LATEX/.vscode/settings.json` and the `LATEX.code-workspace` recipe override (currently uncommitted; bypasses latexmk for Perl-free build).

---

## Evaluation (Ch4 TBD tables)

These are the real evaluation runs that populate Tables 4.1, 4.2 and the Protocol-C placeholder. None can be filled in from preliminary data — they need a real run on the eleven worked questions. **All protocols are designed to be label-free** so that scale-up to a 50–100 question bank or to the full corpus does not re-incur annotation cost.

- [ ] **(P0)** Pick the Protocol A judge model. Constraint: must be different from the synthesis-stage LLM (no judge-equals-producer circularity). Pin the snapshot, temperature 0, deterministic decoding. Record the pinned identifier in `config/app_config.toml`.
- [ ] **(P0)** Implement Protocol A: pool dedup → judge call → cache by (question, document, prompt-hash, model-snapshot) → graded nDCG@10 / MAP / Recall@k. Cache keying matters so re-runs are free after the first.
- [ ] **(P0)** Run Protocol A on the eleven worked questions and populate Table 4.1.
- [ ] **(P0)** Run Protocol B (claim-to-evidence support) on the eleven questions with NLI on and NLI off; populate Table 4.2 per question family.
- [ ] **(P0)** Run Protocol C (metamorphic robustness) with the four query transformations; populate the placeholder section.
- [ ] **(P1)** Run the LLM-synthesis vs deterministic-synthesis ablation (Ch4 §4.5.3) on Families A and B where deterministic templates are feasible.
- [ ] **(P1)** Per-stage latency instrumentation on the production VM for the eleven questions (Ch4 §4.6.1).
- [ ] **(P2)** Larger question bank: 50–100 free-form longitudinal questions across the six families. Per-question Protocol A/B/C cost is constant in bank size thanks to the label-free design.
- [ ] **(P2)** Optional judge calibration: $\sim$50 human-labelled (question, document) pairs to put a local Cohen's $\kappa$ error bar on the judge's behaviour for CC-News-style content. Not blocking submission, but a clear reviewer-defensible follow-up.

---

## Implementation — Scaling Pathway

The four-step scaling plan from Ch5 §5.5. Each step must be executable independently and each unlocks a follow-up at full corpus scale.

- [ ] **(P1)** Migrate dense retrieval from pgvector IVFFlat to FAISS IVF-PQ for 13M × 768 dense vectors. Expected size: ~0.8 GB compressed (vs ~40 GB FP32 raw). Out-of-process FAISS service; wrap as `dense_search` backend.
- [ ] **(P1)** Move BERTopic per-year-slice precomputation off-VM to the Slurm cluster; cache models locally on the VM.
- [ ] **(P2)** Shard `article_corpus` PostgreSQL table by publication year. Working-set materialisation should only touch relevant shards.
- [ ] **(P2)** Stand up a GPU-equipped reranking host. Compose overlay (`docker-compose.mcp.gpu.yml`) already exists.

---

## Comparison Plan against CorpusAgent (Ch5 §5.6)

- [ ] **(P1)** Re-deploy the predecessor system (BM25-only + LLM document selection + mocked analytics) inside the production VM against the same working-corpus slice.
- [ ] **(P1)** Run identical eleven-question evaluation set against both systems under Protocols A, B, C.
- [ ] **(P1)** Per-system paired metrics with effect sizes and bootstrap CIs; per-stage attribution of differences via the degraded-status correlation.
- [ ] **(P2)** Re-run the same comparison at full corpus scale once the scaling pathway above delivers it.

---

## Configuration blindspots (from runtime probe)

From the earlier blindspot audit. Not blockers for the paper but should be settled before any real evaluation runs.

- [ ] **(P1)** Reconcile `CORPUSAGENT2_RETRIEVAL_BACKEND=pgvector` vs the CLAUDE.md statement that current operating mode is "Lexical OpenSearch + Postgres fetch + optional rerank". Pick one and align both.
- [ ] **(P2)** Strengthen secret handling: `CORPUSAGENT2_OPENSEARCH_PASSWORD` is in `.env` (which is gitignored) but is printed in clear by `scripts/16_print_effective_config.py`. Either redact in the printer or document the leakage risk.

---

## Repo cleanup

- [ ] **(P2)** Commit `project_paper/LATEX/.vscode/settings.json` + `LATEX.code-workspace` recipe override.
- [ ] **(P2)** `.claude/settings.local.json` is locally modified (harness-recorded permissions). Decide: commit, .gitignore-add, or leave dirty.

---

## Recently closed

| Item | Commit |
|---|---|
| Initial CorpusAgent2 VT2 paper draft (5 chapters + appendix + additional_refs.bib) | `9e1ee55` |
| Graphify incremental update covering paper notes + `.graphifyignore` for runtime artefacts | `4e0f1ae` |
| Clean MiKTeX compile, 42-page PDF, 0 undefined refs/cites | `af2aee3` |
| OpenSearch health probe added to `retrieval_health()` (was probed silently before) | (pre-paper-phase) |
| `response.json` stub + duplicate PDF mirrors + empty `.patch` deleted | (pre-paper-phase) |
| `docs/legacy/README.md` index of the historical/contradictory legacy docs | (pre-paper-phase) |
