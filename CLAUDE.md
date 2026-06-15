# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CorpusAgent2 is a Master's thesis project (12 ECTS MSE, ZHAW). It is a deterministic, tool-based LLM-orchestrated agent for large-scale news corpus analysis — the scientific successor to CorpusAgent1. The corpus is ~624k CC-News documents. The backend does retrieval, NLP, and planning; the frontend (`web/`) is a debug inspector UI.

This is a research prototype, not a production system.

## Thesis context

**Research questions:**
- **RQ1** — Does hybrid retrieval improve evidence completeness over lexical-only retrieval?
- **RQ2** — Can LLM-heavy/mocked analytics be replaced by real NLP tools while keeping interpretability?
- **RQ3** — Does NLI-based claim verification reduce hallucinated final answers?
- **RQ4** — Which retrieval/index architecture gives the best cost/latency/recall tradeoff?

**System variants for experiments:**
- **V0** — deterministic baseline (`scripts/06_run_framework.py` path, or minimal OpenSearch + direct LLM synthesis)
- **V1** — full agent runtime (`agent_runtime.py` path): PlanDAG, capability registry, clarification policy, evidence tables

**Evaluation protocols (without reference answers):**
- **A** — IR relevance judgements with pooling: nDCG@k, MAP, Recall@k
- **B** — Claim-to-evidence support labeling from the system's own evidence tables
- **C** — Metamorphic robustness testing via query transformations

**Current operating mode (verified 2026-06-14):** Hybrid retrieval is operational end-to-end. pgvector holds 624k×768 E5 embeddings with an IVFFlat index; OpenSearch BM25 is indexed over all 624k docs (reachable from the host only via `localhost:9200`, not the docker name `os_news`); RRF fusion + cross-encoder rerank run live. Local TF-IDF/dense assets are also present. Protocol A has been run (see below).

**Evaluation approach — ORACLE-FREE, NO GOLD SETS (hard rule):** No reference answers exist — the professor confirmed ground truth is not definable for open-ended questions over an arbitrarily large corpus. NEVER add or rely on gold document lists, gold evidence-sentence lists, or gold answers (`relevant_doc_ids` / `gold_evidence_doc_ids` are deprecated; the `scripts/03_evaluate_retrieval.py` gold-ID path must not be extended). All quantitative results come from one rerunnable suite, **`scripts/50_run_eval_suite.py`**, which runs oracle-free protocols (Protocol A = LLM-as-judge graded relevance → nDCG@10/Recall@25/MAP; cross-judge Kendall τ; Protocol C = metamorphic robustness; retrievability/Gini corpus-access bias) and writes tables + plots directly into `project_paper/LATEX/generated/` (the paper `\input`s them, so a re-run refreshes the paper). Questions live in `config/eval_questions_11.json` (no gold fields).

## Commands

**Install dependencies (requires `uv`):**
```powershell
uv sync
```

**Run the full local stack (API + frontend server):**
```powershell
.\.venv\Scripts\python.exe scripts\15_start_local_stack.py
```

**Run only the API:**
```powershell
.\.venv\Scripts\python.exe scripts\12_run_agent_api.py
```

**Run only the frontend static server:**
```powershell
.\.venv\Scripts\python.exe scripts\14_run_static_frontend.py
```

**Run tests:**
```powershell
python -m pytest -q
```

**Run a single test file:**
```powershell
python -m pytest tests/test_retrieval.py -q
```

**Print effective runtime config:**
```powershell
python scripts\16_print_effective_config.py
```

**Start Docker services (Postgres + OpenSearch):**
```powershell
docker compose -f deploy\docker-compose.yml up -d postgres opensearch
```

**Build and run Dockerized backend:**
```powershell
cd deploy
docker compose -f docker-compose.yml -f docker-compose.mcp.yml up -d --build --no-deps corpusagent2-api corpusagent2-mcp
```

**Run NLP tooling pipeline (set granularity first):**
```powershell
$env:CORPUSAGENT2_TIME_GRANULARITY="month"  # or "year"
python scripts\05_run_nlp_tooling.py
```

**Run deterministic framework (V0 path):**
```powershell
python scripts\06_run_framework.py
```

**Run retrieval evaluation:**
```powershell
python scripts\03_evaluate_retrieval.py
```

**Run faithfulness evaluation:**
```powershell
python scripts\04_evaluate_faithfulness.py
```

## Architecture

### Request flow (V1 agent runtime)

1. The frontend (`web/`) sends a `POST /query` to the FastAPI backend (`src/corpusagent2/api.py`).
2. `AgentRuntime` (`agent_runtime.py`) receives the question, runs policy/rejection checks, then calls the planner.
3. `QuestionPlanner` (`planner.py`) classifies the question (sentiment, entities, topics, bursts, keyphrases, verification) and emits a `PlanGraph` — a DAG of `PlanNode` tasks.
4. `AsyncPlanExecutor` (`agent_executor.py`) executes the plan nodes, calling tools from `ToolRegistry` (`tool_registry.py`).
5. Tool implementations are in `agent_capabilities.py` and delegate to NLP providers via `provider_adapters.py`.
6. Retrieval is handled by `agent_backends.py` which selects between:
   - `LocalSearchBackend` — TF-IDF + local dense embeddings (NumPy/sentence-transformers)
   - `HybridSearchBackend` — pgvector (dense) + OpenSearch (lexical) with RRF fusion
7. Results are assembled into an `AgentRunManifest` and written to `outputs/agent_runtime/<run_id>/`.

When debugging a run, start with `outputs/agent_runtime/<run_id>/nodes/*.json` — each shows inputs, outputs, and errors per plan node.

### Capability classification

**Thesis-core** (exercise in experiments, measure, ablate):
`db_search`, `create_working_set`, `fetch_documents`, `build_evidence_table`, `time_series_aggregate`, `ner`, `sentiment`, `topic_model`, `sql_query_search`, `python_runner`, `plot_artifact`

**Convenience heuristic** (use if available, do NOT make thesis claims about):
`entity_link`, `claim_strength_score`, `quote_attribute`, `burst_detect`, `lang_id`, `change_point_detect`

If an experiment result depends on a heuristic tool, flag it explicitly in error analysis.

### Configuration layers

Config loads in priority order (later overrides earlier):
1. `config/app_config.toml` — base defaults
2. `.env` — machine-specific overrides
3. Environment variables — highest priority

Key env vars: `CORPUSAGENT2_RETRIEVAL_BACKEND`, `CORPUSAGENT2_FRONTEND_API_BASE_URL`, `CORPUSAGENT2_PG_DSN`, `CORPUSAGENT2_USE_OPENAI`, `CORPUSAGENT2_DEVICE`, `CORPUSAGENT2_TIME_GRANULARITY`.

The frontend reads its API URL from `web/config.js`, which is generated at startup by `scripts/13_write_frontend_config.py`.

### LLM providers

Two provider paths are supported (switched by `CORPUSAGENT2_USE_OPENAI`):
- **OpenAI-compatible API** (`use_openai = true`): uses `CORPUSAGENT2_OPENAI_*` settings
- **Local/unclose**: uses `CORPUSAGENT2_UNCLOSE_*` settings pointing to an OpenAI-compatible local endpoint

### NLP provider fallback chain

NLP capabilities use a prioritized provider chain defined in `config/app_config.toml` under `[provider_order]`. If a provider fails, the next is tried. The optional `nlp-providers` extras (`uv sync --extra nlp-providers`) install flair, gensim, stanza, textacy, textblob.

### Temporal granularity

`src/corpusagent2/temporal.py` enforces strict temporal bin logic. Supported granularities: `year`, `month`. A granularity mismatch between requested and artifact granularity raises an error — no silent fallback. Set `CORPUSAGENT2_TIME_GRANULARITY` before running NLP tooling.

### MCP server

`mcp_server.py` + `mcp_jobs.py` expose the agent runtime as an MCP tool server (port 8002 by default). Run via `scripts/31_run_mcp_server.py`.

### Corpus data pipeline

Scripts are numbered by intended execution order:
- `00–02`: download, stage, and prepare CC-News data → `data/`
- `03–06`: evaluate retrieval, faithfulness, run NLP tooling, run deterministic framework
- `09–11`: initialize Postgres schema, ingest parquet, build pgvector index
- `21`, `26`: bulk index into OpenSearch, backfill pgvector embeddings

### Docker build notes

- `CORPUSAGENT2_DOCKER_TORCH_PROFILE=cpu` (default) vs `gpu` controls which PyTorch wheel is installed.
- `CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS=true` adds the optional NLP extras.
- GPU compose override: `docker-compose.mcp.gpu.yml` (requires NVIDIA/CDI runtime on the host).

## What NOT to do

- Do not add more NLP tools or capabilities — the bottleneck is evaluation, not capability count.
- Do not claim hybrid/dense retrieval works until pgvector embeddings are validated end-to-end.
- Do not call the lexical retrieval baseline "BM25" — the implementation is TF-IDF. Earlier artifacts may contain the wrong label.
- Do not expand the frontend — it is sufficient for debugging. Zero time on UI until experiments are done.
- Do not run experiments before a config freeze — unreproduble runs are wasted work.
- Do not let heuristic tool results masquerade as architectural findings in experiment analysis.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
