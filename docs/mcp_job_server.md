# MCP Job Server

The MCP server exposes CorpusAgent2 as long-running jobs. It does not expose every
internal helper as a public tool. The MCP tools submit a question, poll job state,
cancel work, and read the completed run manifest/artifact index.

## Start on the VM

Use this when Postgres, OpenSearch, dense rows, and corpus assets already exist:

```bash
python scripts/22_prepare_vm_stack.py --setup-mcp-only
```

That starts the existing Docker services and the MCP container only. It does not
run corpus ingest, dense embedding backfill, pgvector index build, or OpenSearch
bulk indexing.

To start MCP after a normal stack prep:

```bash
python scripts/22_prepare_vm_stack.py --start-mcp
```

Default endpoint:

```text
http://127.0.0.1:8765/mcp
```

## MCP Tools

- `submit_question`: creates a durable job and returns `job_id`, `run_id`, queue
  state, and worker capacity.
- `get_job_status`: returns `queued`, `on_hold`, `running`, final status,
  queue position, capacity, and live AgentRuntime status when available.
- `list_jobs`: lists recent jobs, optionally filtered by `owner` or `status`.
- `cancel_job`: cancels queued work or asks AgentRuntime to abort a running run.
- `get_job_result`: returns job status plus the final run manifest when it exists.
- `get_runtime_info`: returns MCP worker capacity and the underlying runtime
  profile.

## Resources

- `corpusagent2://jobs/{job_id}`
- `corpusagent2://jobs/{job_id}/manifest`
- `corpusagent2://jobs/{job_id}/artifacts`

## Useful Environment Variables

- `CORPUSAGENT2_MCP_MAX_CONCURRENT_JOBS`: worker slots. Default `1`.
- `CORPUSAGENT2_MCP_JOB_STORE`: `postgres` by default when a DSN is available.
- `CORPUSAGENT2_MCP_PG_DSN`: optional Postgres DSN override for Docker or remote DB.
- `CORPUSAGENT2_MCP_STALE_RUNNING_SECONDS`: requeue timeout for jobs whose worker
  heartbeat stopped. Default `900`.
- `CORPUSAGENT2_DOCKER_DOWNLOAD_PROVIDER_ASSETS`: downloads provider assets while
  building the MCP Docker image. Default `true`; set `false` only for a faster
  dev image that may use provider fallbacks.
