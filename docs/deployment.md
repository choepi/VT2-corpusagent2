# Deployment Guide

This repo now supports a simple deployment shape:

1. GitHub Pages serves the static frontend from `web/`
2. an Ubuntu VM runs the FastAPI backend
3. Docker on the VM runs Postgres + OpenSearch
4. Cloudflared can expose the backend for easy public access

## 1. First-time VM setup

Clone the repo on the VM, then run the single bootstrap file:

```bash
python3 scripts/22_prepare_vm_stack.py --install-system
```

What that bootstrap does:

- installs Ubuntu system packages when `--install-system` is passed
- creates `.venv`
- installs Python dependencies with `uv`
- downloads provider assets
- downloads and preprocesses the CC-News dataset when local data is missing
- builds lexical + dense retrieval assets when needed
- starts Docker services from [`deploy/docker-compose.yml`](../deploy/docker-compose.yml)
- initializes Postgres + pgvector
- ingests the corpus into Postgres
- bulk-indexes the corpus into OpenSearch
- writes `web/config.js`

It is idempotent, so rerunning it after updates is safe.

## 2. Start the backend on the VM

```bash
./.venv/bin/python ./scripts/12_run_agent_api.py
```

Recommended for real VM use:

- keep the backend bound to `127.0.0.1`
- keep Postgres and OpenSearch bound to `127.0.0.1`
- expose only the backend through a tunnel or reverse proxy

## 3. Easy public access with Cloudflared

Install `cloudflared` once on the VM, then run:

```bash
./.venv/bin/python ./scripts/23_start_cloudflared_tunnel.py
```

That helper starts a Cloudflare Quick Tunnel for `http://127.0.0.1:8001` and prints the public HTTPS URL.

Quick Tunnel notes:

- easiest setup
- good for demos and remote access
- URL changes each time you restart the tunnel

If you use GitHub Pages as the frontend:

- open the Pages site
- paste the printed HTTPS tunnel URL into the `API Base URL` field
- the UI keeps that override in browser storage

Official Cloudflare references:

- [Cloudflare Tunnel downloads](https://developers.cloudflare.com/tunnel/downloads/)
- [TryCloudflare / Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)

## 4. GitHub Pages frontend

The GitHub Pages workflow lives in [pages.yml](../.github/workflows/pages.yml).

It now redeploys when these change:

- `web/**`
- `config/**`
- `scripts/13_write_frontend_config.py`

So when you push frontend or runtime-config changes to `main`, the static site redeploys automatically.

## 5. Update flow after a new push to `main`

On the VM:

```bash
git pull
python3 scripts/22_prepare_vm_stack.py
```

Then restart the backend process you are using.

If Docker volumes were wiped or you want a full rebuild:

```bash
python3 scripts/22_prepare_vm_stack.py --refresh-postgres --refresh-opensearch
```

If you also want to rebuild the local dataset + retrieval assets:

```bash
python3 scripts/22_prepare_vm_stack.py --refresh-data --refresh-assets --refresh-postgres --refresh-opensearch
```

## 6. Retrieval/backend shape

The default retrieval backend is now `pgvector` + OpenSearch:

- dense retrieval: Postgres/pgvector
- lexical retrieval: OpenSearch
- fusion/rerank: backend runtime

The runtime stays strict about service-backed retrieval:

- `CORPUSAGENT2_REQUIRE_BACKEND_SERVICES=true`
- `CORPUSAGENT2_ALLOW_LOCAL_FALLBACK=false`

That means the VM path matches the intended production architecture much more closely than the older local-fallback mode.
