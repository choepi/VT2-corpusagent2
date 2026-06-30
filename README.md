# CorpusAgent2

A prototype for asking questions over a large news corpus. The backend does the
retrieval and analysis; the frontend is a small inspector UI so I can watch the
plan, tool calls, evidence, and final answer as a run happens.

It's a research prototype, not a polished product — some parts are cleaner than
others.

## What's in here

- FastAPI backend for the agent runtime
- Static inspector frontend (`web/`)
- Hybrid retrieval: Postgres/pgvector (dense) + OpenSearch (lexical)
- Local lexical/dense retrieval assets for offline work
- Per-run planner/executor traces, evidence tables, and plots under `outputs/`
- Scripts for preparing CC-News style data

## Requirements

- Python 3.11 and [`uv`](https://github.com/astral-sh/uv)
- Docker, if you want Postgres and OpenSearch locally
- An OpenAI-compatible API key, unless you're only testing the non-LLM pieces
- Corpus data, if you're building the index from scratch

## Setup

The quickest path auto-detects CUDA, writes `.env`, and brings everything up:

```powershell
python scripts/setup.py     # detects CUDA, writes .env, runs uv sync
python scripts/run.py up    # docker compose, picks CPU/GPU profile for you
```

Then open `http://127.0.0.1:8001` — the FastAPI app serves the UI and the API on
the same port. `scripts/run.py` also takes `up-nodb / build / down / stop / logs
/ status / local / api / mcp`.

### Manual setup

If you'd rather do it by hand:

```bash
uv sync
cp .env.example .env        # Copy-Item on Windows PowerShell
```

Set at least these in `.env`:

```dotenv
OPENAI_API_KEY=your_key_here
CORPUSAGENT2_FRONTEND_API_BASE_URL=http://127.0.0.1:8001
```

Start the database services if you need them, then run the stack:

```bash
docker compose -f deploy/docker-compose.yml up -d postgres opensearch
python scripts/15_start_local_stack.py
```

On Windows use `.\.venv\Scripts\python.exe` for the last command. The UI comes up
on `http://127.0.0.1:5500` and the API on `http://127.0.0.1:8001`.

## If the app runs but retrieval comes back empty

Usually the UI and backend are fine but the corpus/index isn't built yet. Drop
`.jsonl` / `.jsonl.gz` files into `data/raw/incoming/` and run the prep scripts
in order:

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

On a real corpus this takes a while. For a fresh Ubuntu VM the shortcut is
`python3 scripts/22_prepare_vm_stack.py --install-system`.

## Useful commands

```bash
python scripts/16_print_effective_config.py   # show the config actually in use
python scripts/14_run_static_frontend.py       # frontend only
python scripts/07_mcp_server.py                # MCP server (dev)
python -m pytest -q                            # tests
```

Dockerized backend stack:

```bash
cd deploy
docker compose -f docker-compose.yml -f docker-compose.mcp.yml up -d --no-recreate postgres opensearch
docker compose -f docker-compose.yml -f docker-compose.mcp.yml up -d --build --no-deps corpusagent2-api corpusagent2-mcp
```

Add `-f docker-compose.mcp.gpu.yml` to the second command on a host where Docker
can see an NVIDIA/CDI GPU.

## Config

Defaults live in `config/app_config.toml`; machine-specific values go in `.env`.
The ones I touch most:

```dotenv
OPENAI_API_KEY=
CORPUSAGENT2_FRONTEND_API_BASE_URL=
CORPUSAGENT2_PG_DSN=
CORPUSAGENT2_RETRIEVAL_BACKEND=
CORPUSAGENT2_OPENSEARCH_URL=
CORPUSAGENT2_DEVICE=
```

The frontend writes `web/config.js` at startup. If it's calling the wrong
backend, check `CORPUSAGENT2_FRONTEND_API_BASE_URL`.

## Output

Runs write to `outputs/agent_runtime/`. The files I actually open:

- `summary.json`
- `nodes/*.json` — what each tool received, returned, and why it may have been empty
- generated plots/artifacts and selected evidence rows

When a run looks off, start with the node JSON.

## Repo map

```text
config/             default config
data/               raw and processed corpus data
deploy/             docker compose for Postgres/OpenSearch
docs/               longer notes
outputs/            generated run output
scripts/            setup, indexing, runtime, and utility scripts
src/corpusagent2/   backend/runtime code
tests/              pytest suite
web/                static frontend
```

## Caveats

- Some analytics are still heuristic.
- Large corpus setup needs disk space and patience.
- Full hybrid retrieval expects healthy Postgres/pgvector and OpenSearch.
- The frontend is a debugging UI, not a finished app.
</content>
