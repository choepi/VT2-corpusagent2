# Paper Corpus Setup

This repo previously used a local `vblagoje/cc_news` train export with 624,095 rows. That corpus was archived under:

```text
data/archive/vblagoje_cc_news_train_624095_2017_2019_20260509/
```

The paper describes a different corpus:

- Hugging Face dataset: `Geralt-Targaryen/CC-News`
- Time span: 2016-2021
- Working slice: 10% sample, roughly 13 million articles
- Metadata granularity in the paper: mainly yearly

If you need the exact paper slice, use the original sampled file from the paper project and pass it with `--source-file`. If that file is not available, the command below builds a reproducible 10% deterministic streaming sample from the same Hugging Face dataset. It is comparable, but not guaranteed byte-identical to the paper slice unless the paper used the same sampling method and seed.

## RTX 3080-Safe Build Parameters

These defaults are conservative for a 10 GB RTX 3080:

- Dense model: `intfloat/e5-base-v2`
- Dense encode batch size: `64`
- Dense memmap chunk size: `4096`
- TF-IDF max features: `250000`
- Sentiment device: `cuda`
- If CUDA OOM occurs: reduce dense batch size to `32`

Expect substantial disk usage. A 13M-row slice needs tens of GB for raw/processed data, about 40 GB just for float32 dense embeddings, plus Postgres/OpenSearch storage. Keep at least 250-400 GB free for a full indexed build.

## Build Retrieval Assets From Hugging Face

PowerShell:

```powershell
cd "D:\Programmieren\GitHub\VT2-corpusagent2"

$env:CORPUSAGENT2_TFIDF_MAX_FEATURES = "250000"
$env:CORPUSAGENT2_RESUME_DENSE_ASSETS = "true"
$env:CORPUSAGENT2_DENSE_MODEL_ID = "D:\Programmieren\GitHub\e5-base-v2"  # optional local clone

.\.venv\Scripts\python.exe scripts\27_build_prebuilt_bundle.py `
  --hf-dataset Geralt-Targaryen/CC-News `
  --hf-split train `
  --hf-streaming `
  --sample-fraction 0.10 `
  --sample-seed 42 `
  --granularity year `
  --mode full `
  --dense-model-id $env:CORPUSAGENT2_DENSE_MODEL_ID `
  --dense-batch-size 64 `
  --dense-chunk-size 4096 `
  --sentiment-device cuda `
  --skip-nlp `
  --bundle-path outputs\prebuilt\paper_ccnews_10pct_2016_2021_retrieval.zip
```

Use `--skip-nlp` for the first build. It creates raw/processed data plus lexical and dense retrieval assets. Full-corpus NLP over roughly 13M articles is better run selectively through the agent tools or as separate long jobs.

If you have the exact paper sample as a local parquet/jsonl file, replace the Hugging Face options with:

```powershell
.\.venv\Scripts\python.exe scripts\27_build_prebuilt_bundle.py `
  --source-file D:\path\to\paper_ccnews_10pct.parquet `
  --granularity year `
  --mode full `
  --dense-batch-size 64 `
  --dense-chunk-size 4096 `
  --sentiment-device cuda `
  --skip-nlp `
  --bundle-path outputs\prebuilt\paper_ccnews_10pct_exact_retrieval.zip
```

## Ingest Into New DB/Index Names

Use separate names so the archived old corpus is not confused with the paper corpus.

```powershell
cd "D:\Programmieren\GitHub\VT2-corpusagent2"

$env:CORPUSAGENT2_PG_TABLE = "article_corpus_paper_ccnews_10pct"
$env:CORPUSAGENT2_OPENSEARCH_INDEX = "article-corpus-paper-ccnews-10pct"
$env:CORPUSAGENT2_PG_INCLUDE_EMBEDDINGS = "true"
$env:CORPUSAGENT2_PG_INGEST_BATCH_SIZE = "500"
$env:CORPUSAGENT2_PG_INGEST_READ_BATCH_SIZE = "2000"
$env:CORPUSAGENT2_PG_INGEST_COMMIT_EVERY_BATCHES = "16"
$env:CORPUSAGENT2_OPENSEARCH_BULK_BATCH_SIZE = "1000"
$env:CORPUSAGENT2_ENABLE_DENSE_RETRIEVAL = "true"
$env:CORPUSAGENT2_PG_MAINTENANCE_WORK_MEM = "2GB"
$env:CORPUSAGENT2_PG_MAX_PARALLEL_MAINTENANCE_WORKERS = "4"

.\.venv\Scripts\python.exe scripts\09_init_postgres_schema.py
.\.venv\Scripts\python.exe scripts\10_ingest_parquet_to_postgres.py
.\.venv\Scripts\python.exe scripts\21_bulk_index_opensearch.py
.\.venv\Scripts\python.exe scripts\11_build_pgvector_index.py
```

If Postgres ingest with embeddings is too slow or memory-heavy, ingest without embeddings first:

```powershell
$env:CORPUSAGENT2_PG_INCLUDE_EMBEDDINGS = "false"
.\.venv\Scripts\python.exe scripts\10_ingest_parquet_to_postgres.py

$env:CORPUSAGENT2_PG_BACKFILL_PREFER_LOCAL_ASSETS = "true"
$env:CORPUSAGENT2_PG_BACKFILL_FETCH_BATCH_SIZE = "256"
$env:CORPUSAGENT2_PG_BACKFILL_ENCODE_BATCH_SIZE = "64"
.\.venv\Scripts\python.exe scripts\26_backfill_pgvector_embeddings.py
```

Then build the pgvector index:

```powershell
$env:CORPUSAGENT2_ENABLE_DENSE_RETRIEVAL = "true"
.\.venv\Scripts\python.exe scripts\11_build_pgvector_index.py
```

## Activate The Paper Corpus

Set these in `.env` or your shell before starting the API:

```dotenv
CORPUSAGENT2_PG_TABLE=article_corpus_paper_ccnews_10pct
CORPUSAGENT2_OPENSEARCH_INDEX=article-corpus-paper-ccnews-10pct
CORPUSAGENT2_DENSE_MODEL_ID=intfloat/e5-base-v2
# For Dockerized API/MCP with a manual model clone next to the repo:
CORPUSAGENT2_DENSE_MODEL_HOST_PATH=../../e5-base-v2
CORPUSAGENT2_DOCKER_DENSE_MODEL_ID=/models/e5-base-v2
CORPUSAGENT2_DEVICE=cuda
```

Then restart the API/backend stack.
