# Outlook: Corpus-General Initialization & Multi-Corpus Switching

**Status:** Deferred (idea captured, not executed). See "Decision" below.
**Branch:** `feature/corpus-general-init` (holds this note only — no feature code).
**Date:** 2026-06-22

## The idea (as requested)

Make CorpusAgent2 genuinely corpus-general from the *user's* point of view, not just
the backend's:

1. A new front-end tab **"Init corpus"** alongside the existing Simple / Advanced modes.
2. A **dropdown to choose between corpora** (a "clean corpus version" picker), so an
   operator can switch the active corpus without editing config and restarting.
3. An **LLM check** that, when a new corpus is pointed at, decides whether it is "the
   same corpus" as the one currently indexed (same dataset/schema/coverage) or a
   different one that needs a fresh ingest/index — so the backend keeps working across
   the switch instead of silently serving a mismatched index.

Motivation: the thesis already claims a *corpus-agnostic* design (Contribution 3,
"corpus-agnostic capability registry"; supervisor feedback that the agent must be
corpus-agnostic). This feature would be the *operator-facing demonstrator* of that
claim — prove generality by actually running a second corpus through the same stack.

## Decision: defer, do not execute now

This is deliberately **not implemented** on this branch. Reasons, tied to the project's
own rules and current stage:

- **Frontend freeze (CLAUDE.md).** "Do not expand the frontend — it is sufficient for
  debugging. Zero time on UI until experiments are done." A new tab is frontend
  expansion at exactly the moment the rules forbid it.
- **Capability freeze (CLAUDE.md).** "Do not add more NLP tools or capabilities — the
  bottleneck is evaluation, not capability count." An LLM corpus-sameness classifier is
  a new capability/subsystem.
- **Config-freeze stage.** The thesis is at the evaluation / paper-finalization stage
  (oracle-free suite `scripts/50_run_eval_suite.py` writes directly into the paper; a
  config freeze precedes reproducible runs). A multi-corpus switcher changes the active
  corpus out from under the eval suite — directly at odds with reproducibility.
- **Low marginal thesis value, high risk.** The corpus-agnostic property is *already* a
  stated contribution and the backend is *already* config-driven (see below). A UI
  switcher demonstrates it but does not add a measured result for any RQ. The time cost
  lands on the thesis critical path; the payoff is a demo, not evidence.

It is therefore recorded here as future work rather than built.

## What already makes the backend corpus-general (no code needed)

The backend is corpus-parametric by configuration today — switching corpora is a
config + re-ingest operation, not a code change:

- **Dataset identity** — `CORPUSAGENT2_CORPUS_NAME` / `CORPUSAGENT2_HF_DATASET`
  (`agent_runtime._runtime_corpus_info`), with a staged-filename fallback. The CPU VM
  reports `vblagoje/cc_news`; the GPU machine reports `Geralt-Targaryen/CC-News`, from
  the *same* code.
- **Storage targets** — `CORPUSAGENT2_PG_DSN`, `CORPUSAGENT2_PG_TABLE`,
  `CORPUSAGENT2_OPENSEARCH_INDEX` select where docs live.
- **Live corpus stats from the active DB** — `AgentRuntime._live_corpus_stats` and
  `CorpusRuntime` report count/date-bounds/schema from the *live* Postgres, not stale
  local `doc_metadata.parquet` (commits 36f2827 / b01f963 / 684c40d). So `corpus_info`,
  date bounds, and `corpus_schema` already follow whatever corpus is indexed.
- **Schema-agnostic columns** — `PostgresWorkingSetStore._document_columns` resolves
  doc_id / text / date / source column names dynamically.
- **Ingest pipeline is corpus-parametric** — scripts `00–02` (download/stage/prepare),
  `09–11` (schema/ingest/pgvector), `21`/`26` (OpenSearch + embedding backfill) take
  the source dataset as input; the 13M scale-up (`00_2`/`00_3`, `_run_ingest_13m.sh`,
  `docker-compose.scale.yml`) already proved a *second* corpus runs through unchanged.

The gap the requested feature would close is purely **operator ergonomics + a safety
check**, not backend capability.

## Concrete design sketch (for whoever picks this up)

Build only after the experiments are frozen and the paper's measured results are in.

### 1. Corpus registry (backend, config-only)
A `config/corpora.toml` listing named corpora, each a frozen bundle of the env knobs
that already exist:
```toml
[corpora.ccnews_624k]
display_name   = "CC-News (vblagoje, 624k)"
hf_dataset     = "vblagoje/cc_news"
pg_table       = "article_corpus"
opensearch_index = "article-corpus-opensearch"

[corpora.ccnews_13m]
display_name   = "CC-News (Geralt-Targaryen, 13M)"
hf_dataset     = "Geralt-Targaryen/CC-News"
pg_table       = "article_corpus_13m"
opensearch_index = "article-corpus-13m"
```
`GET /corpora` lists them with live health (reuse the existing health probe per table/
index); `POST /corpora/activate {name}` swaps the active `pg_table`/`opensearch_index`
on the runtime and busts the corpus-stats / date-bounds / retrieval-health caches.
No new retrieval or NLP capability — it re-points existing ones.

### 2. "Init corpus" tab (frontend)
Read-only dropdown bound to `GET /corpora`, an "activate" button, and a health badge
per corpus (indexed? dense backfilled? date range). This is the only genuinely *new* UI
surface and is small.

### 3. "Same corpus?" check (the LLM part — keep it deterministic-first)
The request frames this as an LLM check. Make the LLM the *last* resort, not the first,
to stay faithful to the project's "NLP not 2015 ML, but also not LLM-where-a-hash-does"
discipline:
- **Identity fast-path (deterministic):** compare staged-source `sha256` + row count +
  schema columns + date bounds against the indexed corpus's recorded manifest. Equal ⇒
  "same corpus", reuse the existing index. This is exact and free.
- **Drift check (deterministic):** if dataset name matches but counts/date-bounds differ,
  flag "same source, changed snapshot ⇒ re-ingest required".
- **LLM adjudication (only on ambiguity):** when names differ but content might overlap
  (e.g. a re-hosted mirror), sample N titles/sources from each and ask the LLM whether
  they are the same underlying collection, returning a structured
  `{same: bool, confidence, reason}` with the sampled evidence attached — i.e. the same
  evidence-table discipline the rest of the system uses, not an opaque yes/no.

### 4. Keeping the backend working across a switch
Activation must be transactional from the operator's view: probe the target index health
*before* swapping; refuse to activate a corpus whose dense backfill / OpenSearch index is
incomplete; keep the previous corpus active on failure. The health probe already returns
exactly the readiness fields needed (`pgvector.ready`, `opensearch.ready`, `dense_rows`).

## Relation to the thesis

If built, this belongs in **Chapter 5 → Future Work** as a fifth direction
("operator-facing multi-corpus activation"), and would let a future evaluation *measure*
corpus-agnosticism by re-running Protocols A/C on a second corpus — turning the current
*design* claim into an *empirical* one. Until then it stays here, in outlook.
