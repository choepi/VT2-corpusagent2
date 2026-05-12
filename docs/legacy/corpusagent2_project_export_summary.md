# CorpusAgent2 — Project Export Summary

**Purpose of this document:** This is a self-contained project handoff/export summary for continuing the CorpusAgent2 VT2/Master project without needing the full prior chat history. It is written for three audiences: a future ChatGPT/Codex session, a supervisor/professor, and the project owner who needs a brutally clear status picture.

**Project:** CorpusAgent2 / VT2  
**Context:** 12 ECTS MSE implementation project, ZHAW  
**Core theme:** Deterministic, measurable, auditable large-corpus analysis for news corpora  
**Current snapshot basis:** Repository/document snapshots around March 2026, especially the handoff state file dated 2026-03-17.

---

## 1. One-sentence project summary

CorpusAgent2 is a deterministic, tool-based successor to the original CorpusAgent idea: instead of relying heavily on LLM planning, LLM filtering, mocked NLP tools, and qualitative case-study evaluation, it implements a staged retrieval + NLP analytics + NLI verification + provenance pipeline for large news corpus analysis, with the goal of making corpus-level answers measurable, reproducible, and scientifically defensible.

---

## 2. Brutal status verdict

The project is **no longer just an idea**. A real deterministic pipeline exists, core scripts are implemented, retrieval and faithfulness evaluation have run, provenance artifacts exist, and the system can process a substantial news corpus slice.

But it is **not yet a finished scientific contribution**. The current system proves engineering feasibility, not superiority. The evaluation set is still tiny, baseline comparison against the original CorpusAgent/corpusagent1 is missing, RQ4 architecture experiments are not yet evidenced, and the current NLP artifacts need regeneration/validation because some summaries may be stale.

**Current maturity:** engineering prototype with partial empirical evidence.  
**Not yet:** defensible final Master's-level result set.

Do not overclaim. The correct claim right now is:

> CorpusAgent2 demonstrates a working deterministic architecture for retrieval, NLP tooling, verification, and provenance on a substantial corpus slice. Early metrics and artifacts exist, but robust scientific validation still requires larger annotated benchmarks, baseline comparisons, ablation studies, and cost/latency experiments.

---

## 3. Origin: what the original CorpusAgent paper did

The original CorpusAgent thesis proposed a hybrid NLP/LLM system for exploratory analysis of large news corpora. Its core idea was good: classical NLP tools extract structured signals, while an LLM performs planning, orchestration, document selection, and final answer synthesis. The architecture was database-centric and aimed to support traceability and reproducibility for temporally structured corpus questions.

The original system had layers roughly like this:

1. **Input / feasibility layer** — user question is checked for answerability and temporal scope.
2. **Multi-stage planning layer** — LLM decides retrieval, NLP analysis, selection, and summarization strategy.
3. **Retrieval layer** — OpenSearch/BM25 retrieves potentially relevant articles.
4. **Analytics layer** — planned NLP tools such as sentiment, topic modeling, NER, and event detection.
5. **Document selection / filtering layer** — LLM selects important documents from batches.
6. **Answer synthesis layer** — LLM writes final temporally organized answer.
7. **Visualization playground** — LLM generates visualization plans and plotting code under a controlled wrapper.

The paper also included a useful evaluation-question structure: longitudinal questions, multi-hop/entity-indirection questions, negative controls, and counterfactual controls. That question design is valuable and should be reused.

### 3.1 Main weakness of the original system

The original version was more of an architecture proof-of-concept than a scientifically validated analysis system. The critical problems were:

- **Mocked NLP analytics:** sentiment, topic modeling, framing/event detection, and related outputs were not real measurements in the prototype.
- **LLM-heavy relevance selection:** instead of a proper reranker or deterministic ranking stack, document selection relied heavily on LLM batch reasoning.
- **Mostly qualitative evaluation:** the system demonstrated plausible case studies, but did not provide a strong benchmark harness with robust metrics.
- **Weak faithfulness guarantees:** answers could be plausible without a strict claim-to-evidence verification layer.
- **Limited temporal resolution:** yearly aggregation is too coarse for many events, especially COVID or market shocks.
- **Scalability gaps:** parallelism, indexing alternatives, and cost/latency tradeoffs were mostly future work.

CorpusAgent2 exists to attack exactly these weaknesses.

---

## 4. CorpusAgent2 research goal

The project goal is to turn the original CorpusAgent idea from a nice demo into a measurable, reproducible, defensible system.

The recommended research claim is:

> Replacing CorpusAgent's LLM-heavy and partially mocked pipeline with a deterministic retrieval + NLP tooling + NLI verification + provenance framework improves auditability and enables measurable evaluation of large-corpus analytical answers. The scientific question is whether this replacement improves evidence completeness and faithfulness compared with the original LLM-heavy baseline.

A stronger final thesis claim is only allowed after the missing experiments are completed:

> A hybrid lexical+dense+fusion+rerank retrieval stack, combined with real NLP tools and claim-level NLI verification, improves evidence recall and answer faithfulness over the original CorpusAgent-style baseline under a controlled benchmark suite.

That stronger version currently remains unproven.

---

## 5. Research questions

The project is organized around four research questions.

### RQ1 — Retrieval quality

**Question:** Does hybrid retrieval improve evidence completeness over lexical-only retrieval or the original LLM-heavy filtering approach?

**Current implementation direction:** TF-IDF lexical retrieval + dense E5 embeddings + rank fusion + reranking.

**Metrics:** Recall@k, nDCG@k, MRR, Evidence Recall@k, Perfect Recall.

**Current status:** partially answered with tiny evaluation only. There are runs for `tfidf`, `dense`, `tfidf_dense_rrf`, and `tfidf_dense_rrf_rerank`, but only 3 retrieval queries exist in the current gold file. This is not enough for a serious statistical claim.

### RQ2 — De-LLM analytics

**Question:** Can LLM-heavy/mocked analysis be replaced by specialized NLP tools while keeping interpretability and improving measurement quality?

**Target tools:** NER/entity trends, sentiment over time, topics over time, burst detection, keyphrases.

**Current status:** code and artifact files exist in the newer repo handoff, but the artifacts must be regenerated and validated because the summary file may be stale and some previous audits observed incomplete NLP output. The system should not claim final RQ2 success until all expected artifacts are cleanly reproduced in both debug and full modes.

### RQ3 — Faithfulness / hallucination reduction

**Question:** Does NLI-based claim verification reduce unsupported or hallucinated final answers?

**Current implementation direction:** `FacebookAI/roberta-large-mnli` / RoBERTa-MNLI-style claim verification.

**Current status:** NLI verification is implemented and has produced claim verdict files and summaries, but the current faithfulness set contains only 4 claims. That is a sanity check, not a research result.

### RQ4 — Retrieval/index architecture tradeoff

**Question:** Which retrieval/index architecture gives the best cost/latency/recall tradeoff for large-scale corpus analysis?

**Candidate systems:** local dense index, FAISS IVF-PQ/HNSW, pgvector, possibly OpenSearch/other ANN alternatives.

**Current status:** pgvector integration exists in code, but there is no strong evidence of completed large-scale architecture benchmarks across FAISS/HNSW/pgvector. RQ4 is currently not answered.

---

## 6. Current repository state

### 6.1 Repository identity

- **Project root:** `D:\OneDrive - ZHAW\MSE_school_files\Sem4\VT2\corpusagent2`
- **Git branch:** `postgres-add-on`
- **Git status at snapshot:** clean working tree
- **Latest commit:** `e0d909e` with message `update`
- **Recent commits:**
  - `e0d909e` — update
  - `baf437e` — postgress feature
  - `78415fb` — cuda available and global full run definition
  - `bc04568` — fixed mcp with test
  - `4805a3a` — gold ids

### 6.2 Python/dependency state

- `pyproject.toml` requires Python `>=3.11`.
- `README.md` is now aligned to Python 3.11.x.
- `psycopg[binary]>=3.2.12` is present for Postgres/pgvector integration.

### 6.3 Core entry points

Main user-facing / scripted entry points:

- `main.py` — interactive retrieval prompt, writes UI retrieval artifacts.
- `scripts/00_stage_ccnews_files.py` — stages raw CC-News `.jsonl/.jsonl.gz` files.
- `scripts/01_prepare_dataset.py` — prepares normalized `documents.parquet`.
- `scripts/02_build_retrieval_assets.py` — builds lexical and dense retrieval assets.
- `scripts/03_evaluate_retrieval.py` — evaluates retrieval systems.
- `scripts/04_evaluate_faithfulness.py` — evaluates claim faithfulness with NLI.
- `scripts/05_run_nlp_tooling.py` — generates NLP tooling artifacts.
- `scripts/06_run_framework.py` — runs the retrieval + verification framework and writes provenance.
- `scripts/07_mcp_server.py` — exposes MCP tools.
- `scripts/08_review_retrieval.py` — retrieval inspection/backtesting and annotation template.
- `scripts/09_init_postgres_schema.py` — initializes Postgres schema.
- `scripts/10_ingest_parquet_to_postgres.py` — ingests parquet into Postgres.
- `scripts/11_build_pgvector_index.py` — builds pgvector index.

Core modules:

- `src/corpusagent2/retrieval.py`
- `src/corpusagent2/faithfulness.py`
- `src/corpusagent2/provenance.py`
- `src/corpusagent2/metrics.py`
- `src/corpusagent2/seed.py`
- `src/corpusagent2/temporal.py`

---

## 7. Implemented architecture

The currently implemented system is a deterministic staged pipeline:

1. **Stage raw corpus files**
   - Input: raw CC-News files.
   - Output: staged raw file(s).

2. **Prepare dataset**
   - Normalizes articles into `documents.parquet`.
   - This becomes the main local processed corpus artifact.

3. **Build retrieval assets**
   - Lexical retrieval assets.
   - Dense embeddings using `intfloat/e5-base-v2`.
   - Metadata mapping for doc IDs.

4. **Evaluate retrieval**
   - Systems include TF-IDF, dense, TF-IDF+dense RRF, and reranked fusion.
   - Uses IR metrics and statistical helpers such as bootstrap and paired tests.

5. **Evaluate faithfulness**
   - Runs NLI claim verification.
   - Produces claim-level verdicts and summary metrics.

6. **Run NLP tooling**
   - Entity trends.
   - Sentiment over time.
   - Topics over time.
   - Burst events.
   - Keyphrases.

7. **Framework run**
   - Runs retrieval + claim verification.
   - Produces reports and provenance logs.

8. **MCP server**
   - Exposes retrieval and verification functionality as tools for agentic integration.

---

## 8. Important recent changes

### 8.1 TF-IDF relabeling fix

Older code/artifacts sometimes called the lexical retrieval baseline `bm25` even though the implementation was TF-IDF. This is scientifically dangerous because BM25 and TF-IDF are not the same baseline.

The newer repo state says active code paths were corrected to `tfidf` naming in:

- `main.py`
- `scripts/03_evaluate_retrieval.py`
- `scripts/06_run_framework.py`
- `scripts/07_mcp_server.py`
- `scripts/08_review_retrieval.py`
- `README.md`

Caution: older artifacts in `outputs/` may still contain legacy `bm25` labels. Any final thesis table must not mix old and new labels.

### 8.2 Temporal granularity enforcement

A new `src/corpusagent2/temporal.py` module handles strict temporal granularity logic.

Supported granularities:

- `year`
- `month`

Behavior:

- `scripts/05_run_nlp_tooling.py` reads `CORPUSAGENT2_TIME_GRANULARITY`, defaulting to `year`.
- MCP exposes strict sentiment-time behavior via `sentiment_over_time(...)`.
- If requested granularity mismatches artifact granularity, the tool raises an error.
- If mixed incompatible bins are detected, the tool raises an error.
- No silent fallback should happen for temporal mismatch.

This is good. Silent fallback would be trash for scientific reproducibility.

---

## 9. Executed artifact evidence

### 9.1 Corpus staging and dataset prep

Current reported artifacts:

- `outputs/stage_ccnews_summary.json`
  - selected files: 1
  - source: `data/raw/incoming/cc_news.jsonl.gz`
  - staged size: 710,321,048 bytes

- `outputs/prepare_dataset_summary.json`
  - documents written: 624,095
  - output: `data/processed/documents.parquet`

### 9.2 Retrieval assets

Current reported artifacts:

- `outputs/build_retrieval_assets_summary.json`
  - documents indexed: 624,095
  - lexical index path: `data/indices/lexical`
  - dense index path: `data/indices/dense`
  - metadata path: `data/indices/doc_metadata.parquet`
  - dense model: `intfloat/e5-base-v2`

Earlier audit also reported dense embeddings shape `(624095, 768)` and about 1.786 GB float32 embeddings.

### 9.3 Retrieval evaluation

Current reported systems:

- `tfidf`
- `dense`
- `tfidf_dense_rrf`
- `tfidf_dense_rrf_rerank`

Current reported issue:

- only 3 queries in the evaluation set.
- dense recall is higher than TF-IDF in the tiny sample.
- rerank improves nDCG/MRR on one query.
- p-values are weak due to tiny `n=3`.

Interpretation: this is a pipeline sanity check, not evidence for a thesis claim.

### 9.4 Faithfulness evaluation

Reported current state:

- total claims: 4
- entailed claims: 0
- faithfulness: 0.0
- contradiction rate: 0.5
- device: CUDA

Interpretation: NLI execution works, but the evaluation is far too small for any faithfulness conclusion.

### 9.5 NLP tooling outputs

Newer handoff says the following files exist:

- `entity_trend.parquet`
- `sentiment_series.parquet`
- `topics_over_time.parquet`
- `burst_events.parquet`
- `keyphrases.parquet`
- `summary.json`

Observed details from current artifacts:

- sentiment rows: 3
- sentiment bins: `2017`, `2018`, `unknown`
- sentiment model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- entity model: `en_core_web_sm`

Important caveat: the current `outputs/nlp_tools/summary.json` is from an earlier run and does not yet include the newer `time_granularity` key, even though the code now supports temporal granularity. Therefore, regenerate NLP outputs before claiming stage 05 is clean.

### 9.6 Framework run artifacts

Latest reported framework run:

- `outputs/framework/run_5486db56bb80/`
  - `run_summary.json`
  - `reports.jsonl`
  - `provenance.jsonl`

Reports now include `score_components` with `tfidf` and `dense` naming.

---

## 10. Gold/evaluation data state

Current gold/evaluation files:

- `config/retrieval_queries.jsonl`
  - row count: 3

- `config/faithfulness_claims.jsonl`
  - row count: 4

- `config/framework_workload.jsonl`
  - includes example workload entries with query + claims.

This is the biggest scientific weakness right now. Three retrieval queries and four claims are not a benchmark; they are smoke tests.

Minimum next target:

- 25–50 retrieval questions with validated relevance/evidence labels.
- 100–200 atomic claims for faithfulness evaluation.
- Explicit category coverage: longitudinal, multi-hop, negative controls, counterfactual controls.
- Annotation protocol with at least partial double annotation and inter-annotator agreement.

---

## 11. Postgres / pgvector state

Implemented in code:

- retrieval backend selector supports `local` and `pgvector`.
- Environment variables:
  - `CORPUSAGENT2_RETRIEVAL_BACKEND`
  - `CORPUSAGENT2_PG_DSN`
  - `CORPUSAGENT2_PG_TABLE`

Scripts:

- `scripts/09_init_postgres_schema.py`
- `scripts/10_ingest_parquet_to_postgres.py`
- `scripts/11_build_pgvector_index.py`

Current status:

- pgvector integration exists.
- No strong artifact evidence yet for large-scale pgvector benchmark results against FAISS/HNSW/local alternatives.

So: pgvector is implemented as a path, but RQ4 is not solved.

---

## 12. Provenance and auditability

This is one of the strongest parts of the project.

Current project evidence indicates:

- `src/corpusagent2/provenance.py` exists.
- Framework runs write provenance JSONL.
- Reports include score components.
- Tool/model/params/evidence linkage exists.

The intended scientific value:

- Every final answer should be decomposable into claims.
- Every claim should point to retrieved documents or evidence sentences.
- Every retrieval/evaluation/verification step should preserve model ID, parameters, and scores.

This is the right direction. But for the final project, provenance must cover all analytics tools, not only retrieval/verification.

Required final provenance coverage:

| Layer | Required provenance |
|---|---|
| Retrieval | query, backend, top_k, scores, score components, doc IDs/chunk IDs |
| Reranking | reranker model, candidate set, rerank scores |
| NLP tools | model/tool ID, version, parameters, input doc IDs, output artifact paths |
| Temporal aggregation | granularity, bins, missing/unknown-bin handling |
| Verification | claim, evidence, NLI label, confidence, model ID |
| Final answer | claim-to-evidence mapping |

---

## 13. What is genuinely proven

The current project proves the following:

1. A deterministic multi-stage prototype exists.
2. It can run on a substantial corpus slice of about 624k documents.
3. Retrieval assets can be built.
4. Dense E5 embeddings can be produced.
5. Retrieval variants can be evaluated.
6. NLI faithfulness evaluation can run on CUDA.
7. Framework runs can produce reports and provenance JSONL.
8. MCP tool integration exists at least in code and self-test direction.
9. The architecture is more inspectable than a pure LLM-black-box pipeline.

---

## 14. What is not proven yet

The current project does **not** prove:

1. That CorpusAgent2 is better than corpusagent1.
2. That hybrid retrieval significantly improves evidence completeness.
3. That NLI verification reduces hallucinations in generated final answers.
4. That the NLP tooling layer is stable and scientifically valid at full scale.
5. That pgvector/FAISS/HNSW choices have known cost/latency/recall tradeoffs.
6. That the system is cheaper or faster than the original baseline.
7. That final user-facing answers are more interpretable in a measurable way.
8. That the full pipeline is one-command reproducible from a clean environment.

This section must remain in the export. Removing it would make the project summary look stronger but less trustworthy.

---

## 15. Scientific risk register

### Risk 1 — Tiny evaluation set

**Problem:** Current retrieval and faithfulness evaluation sets are too small.

**Impact:** No meaningful significance claims.

**Fix:** Build a proper benchmark suite with gold evidence.

### Risk 2 — Stale or inconsistent artifacts

**Problem:** Some older audit files conflict with newer handoff files; NLP outputs may exist but summary metadata is stale.

**Impact:** Supervisor may ask whether results are actually reproducible.

**Fix:** Regenerate all artifacts with current code and store checksums/run logs.

### Risk 3 — Baseline missing

**Problem:** No direct quantitative comparison against original CorpusAgent/corpusagent1.

**Impact:** Cannot claim improvement, only architectural difference.

**Fix:** Run same query/claim set through both systems or define a reproducible baseline approximation.

### Risk 4 — BM25 vs TF-IDF confusion

**Problem:** Earlier artifacts mislabeled TF-IDF as BM25.

**Impact:** Invalid scientific comparison if not cleaned.

**Fix:** Use strict naming in all tables. If BM25 is not actually implemented, do not call it BM25.

### Risk 5 — Temporal granularity

**Problem:** Corpus dates may be only yearly or partially unknown; some bins include `unknown`.

**Impact:** Longitudinal claims can become too coarse or misleading.

**Fix:** Explicitly report temporal coverage, unknown-rate, and bin validity. Do not pretend monthly analysis exists unless it is validated.

### Risk 6 — RQ4 scope creep

**Problem:** FAISS/HNSW/pgvector benchmarking can explode in scope.

**Impact:** Project becomes an infrastructure benchmark instead of a corpus-analysis framework.

**Fix:** Benchmark only two alternatives if time is limited, with clear metrics: recall, latency, memory, build time.

---

## 16. Recommended final evaluation design

### 16.1 Retrieval evaluation

Systems:

1. `tfidf`
2. `dense_e5`
3. `tfidf_dense_rrf`
4. `tfidf_dense_rrf_rerank`
5. optional: `pgvector_dense`
6. optional: `faiss_ivf` / `faiss_hnsw`

Metrics:

- Recall@10 / Recall@50 / Recall@100
- nDCG@10
- MRR@10
- Evidence Recall@k
- Perfect Recall@k
- latency per query
- index build time
- memory footprint

Minimum acceptable evidence:

- 25+ queries for a weak but useful project claim.
- 50+ queries for a much more defensible project claim.

### 16.2 Faithfulness evaluation

Pipeline:

1. Retrieve documents.
2. Generate or provide atomic claims.
3. Match each claim to evidence sentences.
4. Run NLI verification.
5. Score claim as entailed / neutral / contradicted.
6. Compute faithfulness = entailed claims / total claims.

Metrics:

- faithfulness score
- contradiction rate
- unsupported/neutral rate
- negative-control rejection rate
- counterfactual rejection rate

Minimum acceptable evidence:

- 100+ atomic claims.
- Balanced across normal, negative-control, and counterfactual cases.

### 16.3 NLP tooling evaluation

For each NLP artifact, validate at least basic sanity:

| Artifact | Required validation |
|---|---|
| `entity_trend.parquet` | top entities are plausible; entity types are not garbage; time bins valid |
| `sentiment_series.parquet` | model ID present; sentiment distribution sane; unknown bins quantified |
| `topics_over_time.parquet` | top terms readable; topic IDs stable enough; no empty/degenerate topics |
| `burst_events.parquet` | burst periods correspond to frequency changes |
| `keyphrases.parquet` | phrases meaningful; document frequency not zero; no preprocessing artifacts |

---

## 17. Immediate next steps

### Step 1 — Regenerate a clean full run

Run all scripts with current code and ensure every output has matching metadata.

Target output:

- fresh `outputs/stage_ccnews_summary.json`
- fresh `outputs/prepare_dataset_summary.json`
- fresh `outputs/build_retrieval_assets_summary.json`
- fresh `outputs/retrieval_eval/summary.json`
- fresh `outputs/faithfulness_eval/summary.json`
- fresh `outputs/nlp_tools/summary.json` with `time_granularity`
- fresh `outputs/framework/run_*` with provenance
- fresh MCP self-test log

### Step 2 — Expand gold data

The benchmark is the core of the scientific contribution. Without it, the project is just a clean engineering repo.

Create/expand:

- `config/retrieval_queries.jsonl`
- `config/faithfulness_claims.jsonl`
- `config/framework_workload.jsonl`

Required columns/fields:

- `question_id`
- `category` (`A_longitudinal`, `B_multihop`, `C_negative`, `D_counterfactual`)
- `question`
- `time_range`
- `expected_behavior`
- `gold_doc_ids` or `gold_evidence_refs`
- `claims`
- `notes`

### Step 3 — Run ablations

Minimum ablations:

- TF-IDF only
- dense only
- TF-IDF + dense RRF
- TF-IDF + dense RRF + reranker
- with NLI verification
- without NLI verification

### Step 4 — Build thesis figures/tables

Generate from artifacts, not manually:

- retrieval metric comparison table
- faithfulness summary table
- rejection-rate table for negative/counterfactual controls
- runtime/cost table
- pipeline architecture figure
- provenance example figure/table

### Step 5 — Define final claim conservatively

Only after the experiments:

- If metrics improve significantly: claim performance improvement.
- If metrics are weak but provenance is strong: claim auditability/reproducibility improvement.
- If neither is shown: narrow the thesis to a reproducible evaluation framework rather than a better model.

---

## 18. Minimal reproduction commands

From project root on Windows:

```cmd
uv sync
python scripts/00_stage_ccnews_files.py
python scripts/01_prepare_dataset.py
python scripts/02_build_retrieval_assets.py
python scripts/03_evaluate_retrieval.py
python scripts/04_evaluate_faithfulness.py
python scripts/05_run_nlp_tooling.py
python scripts/06_run_framework.py
python scripts/07_mcp_server.py --self-test --self-test-query "inflation" --self-test-top-k 3
```

Temporal mode examples:

```cmd
set CORPUSAGENT2_TIME_GRANULARITY=year
python scripts/05_run_nlp_tooling.py
```

```cmd
set CORPUSAGENT2_TIME_GRANULARITY=month
python scripts/05_run_nlp_tooling.py
```

Retrieval inspection/backtest:

```cmd
python scripts/08_review_retrieval.py inspect
python scripts/08_review_retrieval.py backtest
```

---

## 19. Suggested supervisor-facing milestone framing

### Milestone 1 — Deterministic pipeline closure

**Deliverable:** One-command reproducible run from raw staged corpus to retrieval, NLP artifacts, faithfulness, framework report, and provenance.

**Acceptance criterion:** all expected output files exist, contain current metadata, and pass sanity checks.

### Milestone 2 — Evaluation dataset construction

**Deliverable:** expanded benchmark query/claim set with annotated evidence.

**Acceptance criterion:** at least 25 retrieval questions and 100 claims; categories A/B/C/D represented.

### Milestone 3 — Retrieval ablation experiment

**Deliverable:** comparison of TF-IDF, dense, fusion, and reranked retrieval.

**Acceptance criterion:** table with Recall@k, nDCG@10, MRR, Evidence Recall, confidence intervals.

### Milestone 4 — Faithfulness and robustness experiment

**Deliverable:** NLI-based claim verification and rejection-rate evaluation.

**Acceptance criterion:** faithfulness, contradiction, negative-control rejection, and counterfactual rejection reported.

### Milestone 5 — Architecture/runtime evaluation

**Deliverable:** at least one backend comparison beyond local retrieval path, ideally pgvector vs local dense or FAISS vs pgvector.

**Acceptance criterion:** latency, memory/build-time, and recall tradeoff table.

### Milestone 6 — Final thesis/package export

**Deliverable:** final report, reproducibility appendix, artifact table, threat-to-validity section, and cleaned repo.

**Acceptance criterion:** no unsupported superiority claims; every major claim linked to artifacts and metrics.

---

## 20. What to tell the professor right now

Use this wording, not inflated marketing:

> The current state is a working deterministic prototype that implements the major stages required for the CorpusAgent2 research direction: corpus staging, dataset preparation, lexical+dense retrieval assets, retrieval evaluation, NLI faithfulness evaluation, NLP tooling artifacts, framework runs, provenance, and MCP integration. The project is currently strongest as an auditable architecture and reproducibility framework. The remaining scientific work is to enlarge the benchmark, validate the gold evidence, run proper ablations, compare against the original CorpusAgent/corpusagent1 baseline, and quantify cost/latency/index tradeoffs. I will not claim superiority until those experiments are done.

This is sober, defensible, and much better than pretending the current 3-query evaluation is science.

---

## 21. Notes for a future assistant / Codex session

When continuing this project, do **not** start by adding random features. The next useful work is:

1. Audit current artifact freshness.
2. Regenerate NLP outputs with explicit `time_granularity` metadata.
3. Expand benchmark data.
4. Run ablations.
5. Generate tables/plots from artifacts.
6. Only then improve UI/MCP polish.

Do not implement SPLADE, ColBERT, or three ANN backends before the core evaluation set is fixed. That would be overengineering. The weak point is not another model; it is missing evidence.

---

## 22. Source documents used for this export

Main source files available in the project context:

- `PA25_ciel_Feuchter_Veseli_Large_Doc_Sets_Explorer.pdf`
- `deep-research-report.md`
- `claudio_points.txt`
- `01_project_status_and_next_steps.md`
- `02_scientific_rationale_and_framework_choice.md`
- `03_goal_checklist_and_evidence_audit.md`
- `REPO_STATE_FOR_CHATGPT_PRO.md`
- `CorpusAgent als Masterprojekt ohne externe APIs solides Upgrade statt Spielzeug.pdf`
- `important notes on lates discussion with prof.txt`
- `notes.docx`

---

## 23. Final project export conclusion

CorpusAgent2 is a promising and partially operational deterministic framework for large-scale news corpus analysis. It has moved beyond paper discussion into implemented scripts, generated artifacts, retrieval evaluation, NLI verification, provenance logging, and MCP tooling. The architecture direction makes sense and is much more defensible than a pure LLM demo.

The remaining work is not glamorous, but it is what decides whether the project becomes Master's-level or stays a prototype: benchmark size, gold evidence, ablations, baseline comparison, artifact freshness, runtime measurements, and careful threat-to-validity writing.

The project should proceed with the motto:

> Less new tooling. More evidence. No fake claims.
