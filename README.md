# CorpusAgent2

CorpusAgent2 is a deterministic and reproducible research stack for longitudinal news analysis.
It is designed for two execution modes:

- local prototyping on Windows/macOS/Linux
- Slurm execution on GPU cluster with scratch staging

## Research Focus

- `RQ1`: Does hybrid retrieval (`BM25/TF-IDF + Dense + Fusion + Reranker`) improve evidence completeness vs lexical-only retrieval?
- `RQ2`: Can LLM-heavy analysis be replaced by specialized measurable tools (NER, sentiment, topics, bursts, keyphrases) without losing interpretability?
- `RQ3`: Does NLI-based verification (`roberta-large-mnli`) reduce hallucinations in final answers?
- `RQ4`: Which index architecture yields best cost/recall tradeoff at scale (FAISS IVF-PQ vs HNSW vs pgvector)?

## Retrieval Stack

`Lexical -> Dense -> Fusion (RRF) -> Cross-Encoder Rerank -> NLI Verification`

Implemented baseline:

- lexical retrieval via TF-IDF (`scripts/02_build_retrieval_assets.py` + `scripts/03_evaluate_retrieval.py`)
- dense retrieval with `intfloat/e5-base-v2`
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

See `scripts/06_run_framework.py` and `src/corpusagent2/provenance.py`.

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

1. Install dependencies:

```bash
uv sync
```

2. Put CC-News source files (`.jsonl` or `.jsonl.gz`) into `data/raw/incoming/`.

3. Run pipeline stages:

```bash
python scripts/00_stage_ccnews_files.py
python scripts/01_prepare_dataset.py
python scripts/02_build_retrieval_assets.py
python scripts/03_evaluate_retrieval.py
python scripts/04_evaluate_faithfulness.py
python scripts/05_run_nlp_tooling.py
python scripts/06_run_framework.py
```

4. Optional interactive prompt:

```bash
python main.py
```

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

## Reproducibility

- deterministic seeds are set in all scripts (`SEED` constant)
- all stages write machine-readable JSON summaries in `outputs/`
- retrieval + verification outputs are tracked by run id
