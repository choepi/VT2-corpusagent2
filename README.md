# CorpusAgent2

CorpusAgent2 is an evidence-first agent runtime for longitudinal news analysis over a large corpus.
The active system surface is the FastAPI backend, the capability-first tool registry, the PlanDAG executor,
the web inspector, and the retrieval stack behind them.

Legacy deterministic framework files still exist as baseline material, but they are not the primary runtime path.

## Runtime Focus

- capability-first tool selection instead of library-specific planning
- LLM planning for clarification, assumptions, PlanDAG generation, revision, and synthesis
- lexical + dense + rerank retrieval with explicit runtime health reporting
- evidence tables, artifacts, and provenance persisted per run
- safe Python sandbox execution for bounded analysis/code paths

## Retrieval Stack

`Lexical -> Dense -> Fusion (RRF) -> Cross-Encoder Rerank -> NLI Verification`

Implemented runtime stack:

- lexical retrieval via TF-IDF (`scripts/02_build_retrieval_assets.py` + `scripts/03_evaluate_retrieval.py`)
- dense retrieval with `intfloat/e5-base-v2`
- dense candidate rerank fallback when full-corpus dense assets are not ready yet
- RRF fusion
- cross-encoder reranking
- NLI claim verification

## Provenance Contract

Every tool call can be persisted with:

- `run_id`
- `tool_name`
- `tool_version`
- `model_id`
- `params_hash`
- `inputs_ref`
- `outputs_ref`
- `evidence` list of `{doc_id, chunk_id, score_components, span_offsets}`

See `src/corpusagent2/agent_runtime.py`, `src/corpusagent2/agent_executor.py`, and `src/corpusagent2/provenance.py`.

## KPI Schemas

See `config/kpi_schema.json`:

- `SentimentSeries(entity, time_bin, mean, std, n_docs, model_id)`
- `EntityTrend(entity, time_bin, count, doc_freq, top_cooccurring, confidence_stats, model_id)`
- `TopicsOverTime(topic_id, time_bin, weight, top_terms, coherence_proxy)`
- `BurstEvents(entity_or_term, burst_level, start, end, intensity)`
- `Keyphrases(phrase, time_bin, score, doc_freq, method)`

## Repository Layout

```text
corpusagent2/
  config/
  data/
    raw/incoming/
    raw/ccnews_staged/
    processed/
    indices/
  log/
  outputs/
  scripts/
  slurm/
  src/corpusagent2/
```

## Local Run (No Docker)

Python requirement: `3.11.x`

Device selection:

- default is automatic: `cuda -> mps -> cpu`
- local override (machine-specific) uses env var `CORPUSAGENT2_DEVICE` with values: `auto`, `cuda`, `mps`, `cpu`
- Windows current shell override example: `set CORPUSAGENT2_DEVICE=cpu`
- hardware check via uv (no project resolution):

```bash
uv run --no-project --python .venv\Scripts\python.exe python -c "import json,sys; sys.path.insert(0,'src'); from corpusagent2.seed import runtime_device_report; print(json.dumps(runtime_device_report(), indent=2))"
```

Retrieval backend selection:

- default is local file-based retrieval assets
- set `CORPUSAGENT2_RETRIEVAL_BACKEND=pgvector` to use Postgres/pgvector for dense retrieval
- required for pgvector mode: `CORPUSAGENT2_PG_DSN`
- optional: `CORPUSAGENT2_PG_TABLE` (default: `ca_documents`)

Temporal KPI aggregation:

- set `CORPUSAGENT2_TIME_GRANULARITY=year` (default) or `month` before `scripts/05_run_nlp_tooling.py`
- sentiment/time-series consumers should request exactly one granularity
- mixed time bins are rejected by MCP `sentiment_over_time` tool (no silent LLM/NLI fallback)

1. Install dependencies:

```bash
uv sync
```

2. Put CC-News source files (`.jsonl` or `.jsonl.gz`) into `data/raw/incoming/`.

3. Run data/retrieval stages:

```bash
python scripts/00_stage_ccnews_files.py
python scripts/01_prepare_dataset.py
python scripts/02_build_retrieval_assets.py
python scripts/03_evaluate_retrieval.py
python scripts/04_evaluate_faithfulness.py
python scripts/05_run_nlp_tooling.py
```

4. Optional interactive prompt:

```bash
python main.py
```

5. Optional Postgres/pgvector setup:

```bash
python scripts/09_init_postgres_schema.py
python scripts/10_ingest_parquet_to_postgres.py
python scripts/11_build_pgvector_index.py
```

Example environment variables (Windows cmd):

```bash
set CORPUSAGENT2_PG_DSN=postgresql://USER:PASSWORD@HOST:5432/DBNAME
set CORPUSAGENT2_PG_TABLE=ca_documents
set CORPUSAGENT2_RETRIEVAL_BACKEND=pgvector
```

## Agent Runtime Quick Start

The agent runtime uses `config/app_config.toml` as the default toggle file and lets `.env` override those values for machine-specific secrets.

Useful overrides include:

- `CORPUSAGENT2_LLM_PROVIDER`
- `CORPUSAGENT2_LLM_BASE_URL`
- `CORPUSAGENT2_LLM_API_KEY`
- `CORPUSAGENT2_FRONTEND_API_BASE_URL`
- `CORPUSAGENT2_PG_DSN`
- `CORPUSAGENT2_OPENSEARCH_URL`

1. Inspect the effective config:

```bash
./.venv/bin/python scripts/16_print_effective_config.py
```

2. Start backend + static frontend together:

```bash
./.venv/bin/python scripts/15_start_local_stack.py
```

3. If you prefer to run them separately:

```bash
./.venv/bin/python scripts/12_run_agent_api.py
./.venv/bin/python scripts/13_write_frontend_config.py
./.venv/bin/python scripts/14_run_static_frontend.py
```

4. Repair or complete dense retrieval when needed:

```bash
./.venv/bin/python scripts/02_build_retrieval_assets.py
./.venv/bin/python scripts/26_backfill_pgvector_embeddings.py
./.venv/bin/python scripts/11_build_pgvector_index.py
```

`/runtime-info` and the frontend now show whether full-corpus dense retrieval is truly ready, whether the runtime is using pgvector, local dense assets, or the dense candidate-rerank fallback, and how many pgvector rows currently have embeddings.

5. Optional temporary public demo path:

- keep the static frontend on GitHub Pages
- expose the FastAPI backend with Cloudflare Tunnel or a VM HTTPS endpoint
- do not expose Postgres or OpenSearch directly
- Quick Tunnels are for testing/development only

6. Optional Ubuntu VM bootstrap:

```bash
bash scripts/bootstrap_ubuntu_vm.sh /home/$USER/corpusagent2
```

7. Deployment notes:

- GitHub Pages deploys only `web/`
- the Pages workflow generates `web/config.js` from `config/app_config.toml`
- keep machine-specific secrets and backend URLs in `.env`
- see `docs/deployment.md` for local, tunnel, and VM deployment paths

## MCP Server

Run the MCP server when you want tool-based integration (stdio transport):

```bash
python scripts/07_mcp_server.py
```

## Slurm Run

Default Slurm resources are encoded directly in each submit script:

- partition: `gpu_top`
- gpu: `1`
- cpus: `16`
- mem: `64G`
- time: `24:00:00`
- account: `cai_nlp`

Submit commands:

```bash
sbatch /home/$USER/corpusagent2/slurm/run_prepare.sbatch
sbatch /home/$USER/corpusagent2/slurm/run_build_assets.sbatch
sbatch /home/$USER/corpusagent2/slurm/run_evaluation.sbatch
sbatch /home/$USER/corpusagent2/slurm/run_framework.sbatch
```

Each Slurm script:

- uses `#!/bin/bash -l`
- stages code to `/scratch/$USER/...` with `rsync -aL` (follows symlinks)
- activates `/home/$USER/corpusagent2/.venv/bin/activate`
- sets HF caches to scratch
- syncs generated outputs back to project home

## Notes on Gold Data

`config/retrieval_queries.jsonl` and `config/faithfulness_claims.jsonl` are starter templates.
For meaningful metrics, populate:

- `relevant_doc_ids`
- `gold_evidence_doc_ids`
- claim categories (`A/B/C/D`) and evidence mapping

Gold files used by scripts:

- retrieval evaluation/backtests: `config/retrieval_queries.jsonl`
- faithfulness evaluation: `config/faithfulness_claims.jsonl`
- framework workload queries + ad-hoc claims: `config/framework_workload.jsonl`

## Reproducibility

- deterministic seeds are set in all scripts (`SEED` constant)
- all stages write machine-readable JSON summaries in `outputs/`
- retrieval + verification outputs are tracked by run id
- runtime manifests persist tool calls, selected documents, evidence tables, artifacts, and final outputs

