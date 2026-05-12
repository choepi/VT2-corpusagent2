# CorpusAgent2: Current Repository State (Handoff)

This document is a self-contained repository state snapshot for in-depth follow-up work.
It is written so another model can continue without needing prior chat context.
Date of snapshot: 2026-03-17 (Europe/Zurich).

## 1) Repository Identity

- Project root: `D:\OneDrive - ZHAW\MSE_school_files\Sem4\VT2\corpusagent2`
- Git branch: `postgres-add-on`
- Git status at snapshot: clean working tree, nothing to commit
- Latest commit on this branch: `e0d909e` with message `update`
- Recent commits:
  - `e0d909e` update
  - `baf437e` postgress feature
  - `78415fb` cuda available and global full run definition
  - `bc04568` fixed mcp with test
  - `4805a3a` gold ids

## 2) Python/Dependency State

- Python requirement in `pyproject.toml`: `>=3.11`
- Key dependency additions present:
  - `psycopg[binary]>=3.2.12` for Postgres/pgvector path
- `README.md` now aligned to Python 3.11.x (not 3.10.x)

## 3) Intended Architecture (as implemented in repo)

Deterministic pipeline for longitudinal news analysis:

1. Stage raw CC-News files
2. Prepare dataset parquet
3. Build retrieval assets:
   - lexical retrieval (TF-IDF)
   - dense embeddings (`intfloat/e5-base-v2`)
4. Retrieval evaluation (TF-IDF, dense, fusion, rerank)
5. Faithfulness evaluation with NLI (`FacebookAI/roberta-large-mnli`)
6. NLP tooling outputs:
   - entity trends
   - sentiment over time
   - topics over time
   - burst events
   - keyphrases
7. Framework run producing:
   - ranked retrieval results
   - claim verdicts
   - provenance logs
8. MCP server for tool-based integration

## 4) Core Code Entry Points

- `main.py` (interactive retrieval prompt, writes UI retrieval artifacts)
- `scripts/00_stage_ccnews_files.py`
- `scripts/01_prepare_dataset.py`
- `scripts/02_build_retrieval_assets.py`
- `scripts/03_evaluate_retrieval.py`
- `scripts/04_evaluate_faithfulness.py`
- `scripts/05_run_nlp_tooling.py`
- `scripts/06_run_framework.py`
- `scripts/07_mcp_server.py`
- `scripts/08_review_retrieval.py` (inspection/backtest + annotation template)
- `scripts/09_init_postgres_schema.py`
- `scripts/10_ingest_parquet_to_postgres.py`
- `scripts/11_build_pgvector_index.py`

Core modules:

- `src/corpusagent2/retrieval.py`
- `src/corpusagent2/faithfulness.py`
- `src/corpusagent2/provenance.py`
- `src/corpusagent2/metrics.py`
- `src/corpusagent2/seed.py`
- `src/corpusagent2/temporal.py` (new temporal granularity logic)

## 5) What Was Recently Changed (Important)

### 5.1 TF-IDF relabeling fix

Previously, multiple outputs/paths referred to lexical retrieval as `bm25` although implementation was TF-IDF.
This was corrected in active code paths to `tfidf` naming.

Files updated for this:

- `main.py`
- `scripts/03_evaluate_retrieval.py`
- `scripts/06_run_framework.py`
- `scripts/07_mcp_server.py`
- `scripts/08_review_retrieval.py`
- `README.md`

Note: older historical artifacts in `outputs/` may still contain legacy `bm25` labels from older runs.

### 5.2 Temporal granularity steering and enforcement

New module:

- `src/corpusagent2/temporal.py`

Behavior now:

- Explicit supported granularities: `year` or `month`
- `scripts/05_run_nlp_tooling.py` uses env var `CORPUSAGENT2_TIME_GRANULARITY` (default `year`)
- MCP now exposes strict sentiment-time behavior via `sentiment_over_time(...)` in `scripts/07_mcp_server.py`
- If requested granularity mismatches available artifact granularity, tool raises error
- If mixed incompatible bins are detected, tool raises error
- No silent LLM/NLI fallback for time-granularity mismatch

## 6) Executed Pipeline Evidence in Outputs

### 6.1 Staging and dataset prep

`outputs/stage_ccnews_summary.json`:

- selected files: 1
- source file: `data/raw/incoming/cc_news.jsonl.gz`
- staged size: 710,321,048 bytes

`outputs/prepare_dataset_summary.json`:

- documents written: 624,095
- output parquet: `data/processed/documents.parquet`

### 6.2 Retrieval asset build

`outputs/build_retrieval_assets_summary.json`:

- documents indexed: 624,095
- lexical index path: `data/indices/lexical`
- dense index path: `data/indices/dense`
- metadata path: `data/indices/doc_metadata.parquet`
- dense model: `intfloat/e5-base-v2`

### 6.3 Retrieval evaluation

`outputs/retrieval_eval/summary.json`:

- query count: 3
- systems:
  - `tfidf`
  - `dense`
  - `tfidf_dense_rrf`
  - `tfidf_dense_rrf_rerank`
- paired tests key: `paired_tests_vs_tfidf`

Current signal from this file:

- very small sample size (3 queries)
- dense recall mean > tfidf recall mean in this tiny set
- rerank improves ndcg/mrr on one query, but statistical evidence is weak (`p_value` not strong with n=3)

### 6.4 Faithfulness evaluation

`outputs/faithfulness_eval/summary.json`:

- total claims: 4
- entailed claims: 0
- faithfulness: 0.0
- contradiction_rate: 0.5
- device used for NLI: `cuda`

### 6.5 NLP tooling outputs

Existing artifact set in `outputs/nlp_tools/`:

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
- entity model id in parquet: `en_core_web_sm`

Important note:

- The current `outputs/nlp_tools/summary.json` is from an earlier run and does not yet include the newly added `time_granularity` key, even though code now supports it.

### 6.6 Framework run artifacts

Latest run folder:

- `outputs/framework/run_5486db56bb80/`
  - `run_summary.json`
  - `reports.jsonl`
  - `provenance.jsonl`

Reports now include `score_components` with `tfidf`/`dense` naming.

## 7) Gold Data State

- Retrieval gold queries file: `config/retrieval_queries.jsonl`
  - row count: 3
- Faithfulness claims file: `config/faithfulness_claims.jsonl`
  - row count: 4
- Framework workload: `config/framework_workload.jsonl`
  - example workload entries with query + claims

This is currently a minimal starter set, not enough for strong scientific claims.

## 8) Postgres/pgvector State

Implemented in code:

- retrieval backend selector supports `local` and `pgvector`
- env vars:
  - `CORPUSAGENT2_RETRIEVAL_BACKEND`
  - `CORPUSAGENT2_PG_DSN`
  - `CORPUSAGENT2_PG_TABLE`

Scripts exist to initialize schema, ingest docs, build vector index:

- `scripts/09_init_postgres_schema.py`
- `scripts/10_ingest_parquet_to_postgres.py`
- `scripts/11_build_pgvector_index.py`

Current status:

- integration is implemented, but there is no strong repo evidence yet of completed large-scale pgvector benchmark results against alternatives in outputs.

## 9) What Is Proven vs Not Proven

### Proven by repo artifacts

- End-to-end deterministic pipeline can run on this machine (local mode)
- TF-IDF + dense + fusion + rerank + NLI path is implemented and executed
- Full NLP artifact set exists (files present)
- Framework provenance outputs exist and are machine-readable

### Not proven yet (scientifically)

- Strong performance claim with statistical confidence (sample too small)
- Quantitative superiority vs `corpusagent1` LLM-layered pipeline
- RQ4 architecture comparison execution evidence (FAISS IVF-PQ vs HNSW vs pgvector)
- High-confidence faithfulness conclusions (only 4 claims currently)
- Robust annotation protocol evidence (multi-annotator agreement, adjudication, IAA)

## 10) Main Risks / Gaps Right Now

1. Evaluation size is too small (`3` retrieval queries, `4` faithfulness claims)
2. Current results are highly unstable to query/gold changes
3. Temporal aggregation policy exists in code, but artifacts need regeneration for explicit granularity metadata consistency
4. No quantitative A/B benchmark artifacts against `corpusagent1`
5. No executed RQ4 benchmark report artifacts across retrieval index architectures

## 11) Minimal Reproduction / Sanity Commands (Windows)

From project root:

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

## 12) Good Prompt Focus for Next Model

If continuing with an in-depth task, focus on:

1. Turning current implementation into defensible scientific evidence:
   - increase gold query/claim set size
   - add strict annotation protocol + IAA
   - run robust significance tests with enough samples
2. Running quantitative A/B against `corpusagent1` outputs
3. Executing RQ4 architecture benchmarks and storing comparable metric artifacts
4. Regenerating NLP artifacts with explicit time granularity and validating rejection behavior for mismatched temporal requests
5. Producing thesis-quality tables/figures directly from current output files

