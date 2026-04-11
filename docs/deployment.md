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
- defaults to the stable `no-dense` VM profile: lexical retrieval via OpenSearch, Postgres document storage, no pgvector embedding dependency
- builds lexical assets automatically and only builds dense assets when you explicitly switch to the `hybrid` profile
- starts Docker services from [`deploy/docker-compose.yml`](../deploy/docker-compose.yml)
- initializes Postgres + pgvector
- ingests the corpus into Postgres
- bulk-indexes the corpus into OpenSearch
- writes `web/config.js`

The default VM retrieval profile is `no-dense`. If you want the full hybrid path, run:

```bash
python3 scripts/22_prepare_vm_stack.py --install-system --retrieval-profile hybrid
```

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

Install `cloudflared` once on the VM, then run one of these:

```bash
./.venv/bin/python ./scripts/23_start_cloudflared_tunnel.py
```

That helper still supports Quick Tunnel for temporary testing. For a stable deployment, configure a named tunnel once and install the persistent services:

```bash
./.venv/bin/python ./scripts/24_configure_vm_services.py
```

Expected `.env` values for the stable named-tunnel path:

- `CORPUSAGENT2_CLOUDFLARED_TUNNEL_ID`
- `CORPUSAGENT2_CLOUDFLARED_TUNNEL_NAME`
- `CORPUSAGENT2_CLOUDFLARED_HOSTNAME`
- optional `CORPUSAGENT2_CLOUDFLARED_CREDENTIALS_FILE`
- optional `CORPUSAGENT2_FRONTEND_API_BASE_URL` if you want to override the inferred `https://<hostname>`

That script writes the Cloudflare config, updates the frontend API base URL in `.env`, regenerates `web/config.js`, and installs reboot-safe systemd services for the API and named tunnel.

Quick Tunnel notes:

- easiest setup
- good for demos and remote access
- URL changes each time you restart the tunnel
- do not use it as the long-term GitHub Pages API origin

If you use GitHub Pages as the frontend, the stable path is a fixed hostname such as `https://api.example.com`. Set it once through `.env` plus `scripts/24_configure_vm_services.py` and stop pasting tunnel URLs into the browser UI.

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

Then restart the backend process you are using, or if you installed services:

```bash
sudo systemctl restart corpusagent2-api.service
sudo systemctl restart corpusagent2-cloudflared.service
```

If Docker volumes were wiped or you want a full rebuild:

```bash
python3 scripts/22_prepare_vm_stack.py --refresh-postgres --refresh-opensearch
```

If you also want to rebuild the local dataset + retrieval assets:

```bash
python3 scripts/22_prepare_vm_stack.py --refresh-data --refresh-assets --refresh-postgres --refresh-opensearch
```

## 6. Retrieval/backend shape

Default VM profile: `no-dense`

- lexical retrieval: OpenSearch
- document storage / working set: Postgres
- rerank / analysis: backend runtime
- no pgvector embeddings required

Optional VM profile: `hybrid`

- dense retrieval: Postgres/pgvector
- lexical retrieval: OpenSearch
- fusion/rerank: backend runtime

The runtime stays strict about service-backed retrieval:

- `CORPUSAGENT2_REQUIRE_BACKEND_SERVICES=true`
- `CORPUSAGENT2_ALLOW_LOCAL_FALLBACK=false`

That means the VM path is now internally consistent in either mode instead of half-requiring dense assets.
