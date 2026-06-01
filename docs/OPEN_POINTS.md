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

- [ ] **(P0)** Pick the Protocol A judge model. Constraint: must be different from the synthesis-stage LLM (no judge-equals-producer circularity). Pin the snapshot, temperature 0, deterministic decoding. Record the pinned identifier in `config/app_config.toml`. Candidate `gpt-5.4-nano-2026-03-17` flagged as soft-circularity risk (same family as synthesis `gpt-5.4-2026-03-05`); cross-family options (e.g. `claude-haiku-4.5`) preferred but acceptable to defend nano in a §4.2 subsection.
- [x] **DONE** Implement Protocol A: `scripts/40_protocol_a.py` does pool dedup → judge call → SHA256 cache keyed on (judge_model, prompt-template, question, document) → graded nDCG@10 / MAP / Recall@K. Three phases (retrieve / judge / metrics), each independently toggleable, all disk-cached.
- [ ] **(P0)** Run `scripts/40_protocol_a.py` against the eleven worked questions and populate Table 4.1. Phase 0 (judge sanity check) runs first and aborts if obviously-wrong labels appear.
- [x] **DONE** Implement Protocol B: `scripts/41_protocol_b.py` reads agent_runtime synthesis outputs, extracts atomic claims via LLM, retrieves best evidence sentence by lexical overlap, runs `roberta-large-mnli` (or LLM-fallback) for entailment, aggregates per question and per family.
- [ ] **(P0)** Run `scripts/41_protocol_b.py` and populate Table 4.2.
- [x] **DONE** Implement Protocol C: `scripts/42_protocol_c.py` generates four metamorphic variants per question via LLM, re-runs retrieval on each, computes top-K Jaccard between original and variant per system.
- [ ] **(P0)** Run `scripts/42_protocol_c.py` and populate the Protocol C placeholder section.
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

---

## Scientific rigor audit — what's needed for safe 5.5+

Honest reviewer-perspective audit of the current draft. P0 items are the gaps that risk the paper dropping back toward 4.5; P1 are the items that lift the grade ceiling.

### P0 — fix before submission

- [ ] **(P0)** **Chapter 3 citation density is critically low** (~3.4 cites per 1k words, vs ~12+ expected for a method chapter). Add citations to back specific architecture choices: pgvector vs FAISS justification, why `intfloat/e5-base-v2` (cite paper), why `ms-marco-MiniLM-L-6-v2` rerank, why `roberta-large-mnli` for NLI, why temperature-0 deterministic LLM calls, why RRF over learned fusion. Each non-trivial design decision needs at least one cite.
- [ ] **(P0)** **Replace Figure 3.1 placeholder** with real TikZ architecture diagram (request flow: question → policy → rephrase → planner → PlanDAG → executor → evidence → NLI → synthesis). Without this, Ch3 reads as a system description without a system picture.
- [ ] **(P0)** **Replace Algorithm 3.1 and 3.2 placeholders** with proper `algorithm2e` listings of `_compile_plan_dag` and degraded-status propagation. Comments in AppendixA already sketch them.
- [ ] **(P0)** **Populate at least one Chapter 4 results table with real numbers.** A method paper with all-TBD tables looks like a proposal, not a thesis. Protocol A is the cheapest to actually run (script 40 is already written) — single Table 4.1 with real nDCG/Recall/MAP would change the paper's character substantially.
- [ ] **(P0)** **Cross-judge ablation for methodological defensibility.** Run Protocol A with at least two judge models (e.g. `gpt-5.4-nano` vs `claude-haiku-4.5` vs `unclose/hermes-3`). Report Kendall τ between system rankings under each judge. A reviewer's first concern with LLM-as-judge is judge dependence; pre-empt it. Adds ~1 paragraph to §4.2 plus a small table.
- [ ] **(P0)** **Statistical methodology subsection in Ch4 must address N=11 underpower honestly.** Current text mentions Wilcoxon signed-rank + bootstrap CIs but at N=11 these are weak. Either (a) explicitly state that the eleven-question results are exploratory rather than confirmatory and frame numbers as effect sizes; or (b) commit to scaling to N≥30 before submission. Pick a side.

### P1 — substantially improve

- [ ] **(P1)** **Architecture-decision-rationale table in Ch3** listing each load-bearing design choice with the citation that supports it and at least one alternative considered. Reviewers love seeing decisions justified, not just announced.
- [ ] **(P1)** **Related-work comparison table in Ch2** (current sections are descriptive; a single table comparing CorpusAgent, RAG-from-scratch, ReAct, Toolformer, CorpusAgent2 on the dimensions of retrieval architecture, NLP layer, provenance, evaluation regime) crystallises positioning in one glance.
- [ ] **(P1)** **External validity subsection in §4.7** — currently no acknowledgement that everything is on a single corpus (CC-News English subset). Reviewer will ask: would this generalize to scientific corpora, German news, social media? Address explicitly.
- [ ] **(P1)** **Add a thesis-roadmap figure to Ch1** — RQ × contribution × evaluation-protocol matrix, single page, shows the whole paper at one glance.
- [ ] **(P1)** **Add a scaling-pathway visualisation to Ch5 §5.5** — four-step diagram or table mapping each step to which RQ it unblocks at full scale. Currently bullet text only.
- [ ] **(P1)** **Question taxonomy needs at least one external anchor.** The six families are novel but unvalidated against prior taxonomies. Position against Maron–Kuhns task types, TREC question types, or Bilenko–Anderson information-need categories. Even one paragraph saying "our families partition the space differently because longitudinal corpus questions cut across these" is enough.
- [ ] **(P1)** **Reproducibility statement** at the end of Ch4 or Ch5: explicit model commit IDs, deterministic seeds, hardware specs, full corpus slice spec, environment lockfile reference. Master's-thesis standard practice; missing now.
- [ ] **(P1)** **Power analysis paragraph** in Ch4 §4.7: at the expected effect size between hybrid+rerank and lexical, what N is required for α=0.05, β=0.8? Even back-of-envelope with cited assumptions justifies the future-work follow-up.

### P2 — defensible without, but improves

- [ ] **(P2)** Negative-results paragraph: any preliminary cases where the system failed or produced surprising outputs, especially under metamorphic transformations.
- [ ] **(P2)** Inter-judge calibration appendix (even N=50 pairs) — shifts the methodology from "we lean on UMBRELA precedent" to "we lean on UMBRELA precedent AND verified locally".
- [ ] **(P2)** Per-stage latency real numbers populating §4.6.1 placeholder from the actual VM.
- [ ] **(P2)** Privacy / ethics paragraph (negligible for CC-News public data but reviewers like seeing it explicitly considered).
- [ ] **(P2)** Language coverage discussion: CC-News is multilingual but the working slice likely English-dominant; quantify and note as constraint.
- [ ] **(P2)** Discussion of LLM-judge limitations beyond UMBRELA (e.g. Faggioli 2023 explicit concerns about judge bias on long-tail / underrepresented topics).

---

## Frontend (Bloomberg rebuild)

- [x] **DONE** HTML + CSS skeleton in `web/index.html` and `web/styles.css`: 6-pane CSS-Grid layout (query / plan-DAG / evidence on row 1, synthesis / NLI on row 2, entity trends / sentiment / topics on row 3) + dark top bar + bottom artefact strip. Light theme primary with `theme-dark` body class as drafted alternate. Mono-first typography (JetBrains Mono / IBM Plex Mono), oxblood `#7a1d2e` accent, sharp 0px corners, no glassmorphism. Previous `web/` snapshot saved to `web_old_apple/` for diff/rollback.
- [ ] **(P1)** Migrate `app.js` to the new DOM IDs. Most existing IDs were preserved (`providerBadge`, `modelBadge`, `deviceBadge`, `accessGate*`, `submitButton`, `queryInput`, `clarificationPanel`, etc.) but new tables (`evidenceTableBody`, `nliTableBody`, `entityTableBody`, `sentimentTableBody`, `topicTableBody`) and the plan ASCII pane (`planAsciiTree`) need population code that doesn't currently exist in `app.js`. Without this migration, the new UI renders but stays empty after a query.
- [ ] **(P2)** Add a plan-DAG ASCII renderer: take the manifest's plan structure and render as the indented tree shown in the mockup (root → children with `├──` / `└──` lines).
- [ ] **(P2)** Streaming updates: if `streamCheckbox` is checked, switch the API call to `POST /query/submit` then poll `/runs/{id}/status` with progressive updates to the panes as nodes complete. Already supported by the backend.
- [ ] **(P2)** Theme-dark toggle button in the top bar (CSS already drafted via `body.theme-dark`).
- [ ] **(P2)** Keyboard shortcuts: `cmd/ctrl+enter` to submit, `esc` to abort, `g e` to focus evidence, `g s` to focus synthesis (vim-style mnemonic).

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
