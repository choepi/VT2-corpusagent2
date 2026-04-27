# CorpusAgent2

CorpusAgent2 is a prototype for asking questions over a large news corpus.

The backend does the retrieval and analysis work. The frontend is just a small
inspector UI so I can see the plan, tool calls, evidence, artifacts, and final
answer while a run is happening.

It is not meant to be a polished product yet. It is a working research/prototype
repo, so some parts are cleaner than others.

## What is in here

- FastAPI backend for the agent runtime
- Static frontend in `web/`
- Postgres/pgvector and OpenSearch retrieval support
- local lexical/dense retrieval assets
- planner/executor traces saved per run
- evidence tables, plots, and run artifacts under `outputs/`
- scripts for preparing CC-News style data

## Requirements

- Python 3.11
- `uv`
- Docker, if you want to run Postgres and OpenSearch locally
- an OpenAI-compatible API key, unless you are only testing non-LLM pieces
- corpus data, if you are building the index from scratch

## Quick start on Windows

From PowerShell:

```powershell
uv sync
Copy-Item .env.example .env
notepad .env
```

At minimum, set these in `.env`:

```dotenv
OPENAI_API_KEY=your_key_here
CORPUSAGENT2_FRONTEND_API_BASE_URL=http://127.0.0.1:8001
```

If you want the local database services:

```powershell
docker compose -f deploy\docker-compose.yml up -d postgres opensearch
```

Start the backend and frontend:

```powershell
.\.venv\Scripts\python.exe scripts\15_start_local_stack.py
```

Then open:

```text
http://127.0.0.1:5500
```

The API is on:

```text
http://127.0.0.1:8001
```

## Quick start on Linux/macOS

```bash
uv sync
cp .env.example .env
```

Edit `.env` and set:

```dotenv
OPENAI_API_KEY=your_key_here
CORPUSAGENT2_FRONTEND_API_BASE_URL=http://127.0.0.1:8001
```

Start local services if needed:

```bash
docker compose -f deploy/docker-compose.yml up -d postgres opensearch
```

Run the app:

```bash
.venv/bin/python scripts/15_start_local_stack.py
```

Open `http://127.0.0.1:5500`.

## If the app starts but retrieval is empty

That usually means the UI and backend are running, but the corpus/index is not
ready yet.

For a fresh local corpus build, put `.jsonl` or `.jsonl.gz` files in:

```text
data/raw/incoming/
```

Then run the preparation scripts:

```bash
python scripts/00_stage_ccnews_files.py
python scripts/01_prepare_dataset.py
python scripts/02_build_retrieval_assets.py
python scripts/09_init_postgres_schema.py
python scripts/10_ingest_parquet_to_postgres.py
python scripts/21_bulk_index_opensearch.py
python scripts/26_backfill_pgvector_embeddings.py
python scripts/11_build_pgvector_index.py
```

This can take a while on a real corpus.

For a fresh Ubuntu VM, the easier path is:

```bash
python3 scripts/22_prepare_vm_stack.py --install-system
```

## Useful commands

Print the config the backend is actually using:

```bash
python scripts/16_print_effective_config.py
```

Run the backend only:

```bash
python scripts/12_run_agent_api.py
```

Run the frontend only:

```bash
python scripts/14_run_static_frontend.py
```

Run tests:

```bash
python -m pytest -q
```

Run the MCP server:

```bash
python scripts/07_mcp_server.py
```

## Config notes

Most defaults live in:

```text
config/app_config.toml
```

Machine-specific values go in `.env`. The main ones I usually touch are:

```dotenv
OPENAI_API_KEY=
CORPUSAGENT2_FRONTEND_API_BASE_URL=
CORPUSAGENT2_PG_DSN=
CORPUSAGENT2_RETRIEVAL_BACKEND=
CORPUSAGENT2_OPENSEARCH_URL=
CORPUSAGENT2_DEVICE=
```

The frontend writes `web/config.js` when it starts. If the frontend is calling
the wrong backend URL, check `CORPUSAGENT2_FRONTEND_API_BASE_URL`.

## Where output goes

Runs write files under:

```text
outputs/agent_runtime/
```

The useful files are usually:

- `summary.json`
- `nodes/*.json`
- generated plots/artifacts
- selected evidence rows

When a run looks strange, start with the node JSON files. They show what each
tool received, what it returned, and why it may have produced no data.

## Repo map

```text
config/                default config
data/                  raw and processed corpus data
deploy/                docker compose for Postgres/OpenSearch
docs/                  longer notes
outputs/               generated run output
scripts/               setup, indexing, runtime, and utility scripts
src/corpusagent2/      backend/runtime code
tests/                 pytest suite
web/                   static frontend
```

## Current caveats

- Some analytics are still heuristic in the prototype.
- Large corpus setup needs disk space and patience.
- Full hybrid retrieval expects Postgres/pgvector and OpenSearch to be healthy.
- The frontend is a debugging UI, not a finished app.

