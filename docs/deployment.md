# Deployment Guide

This project is set up for three practical deployment modes:

1. local development on one machine
2. temporary public testing with GitHub Pages + Cloudflare Tunnel
3. proper thesis/demo hosting with GitHub Pages + Ubuntu VM backend

## 1. Config model

There are two configuration layers:

- checked-in defaults: [`config/app_config.toml`](D:/OneDrive%20-%20ZHAW/MSE_school_files/Sem4/VT2/corpusagent2/config/app_config.toml)
- machine-specific overrides: `.env`

Precedence:

1. real process environment
2. `.env`
3. `config/app_config.toml`

That means you can keep safe defaults in the repo and override secrets or URLs locally.

Useful inspection command:

```bash
python scripts/16_print_effective_config.py
```

## 2. Local development

Start backend + static frontend together:

```bash
python scripts/15_start_local_stack.py
```

Endpoints:

- backend: `http://127.0.0.1:8001`
- frontend: `http://127.0.0.1:5500`

## 3. GitHub Pages frontend

The static frontend lives in `web/`.

The GitHub Pages workflow now generates `web/config.js` during deployment by running:

```bash
python scripts/13_write_frontend_config.py
```

So the deployed frontend uses the repo defaults from `config/app_config.toml`.

Important:

- GitHub Pages only hosts the static frontend
- it does not host FastAPI, Postgres, OpenSearch, or the Python sandbox

## 4. Cloudflare Tunnel for temporary public testing

Cloudflare Quick Tunnels are a good temporary option when:

- the frontend is on GitHub Pages
- the backend is still on your machine
- you want to share a demo without opening raw ports

Start the backend locally, then expose it:

```bash
cloudflared tunnel --url http://127.0.0.1:8001
```

Then set the frontend API base URL to the HTTPS tunnel URL.

Use this for testing only. Quick Tunnels are not the final production path.

## 5. Ubuntu VM deployment

For the proper thesis/demo setup:

- GitHub Pages hosts the static frontend
- Ubuntu VM hosts FastAPI
- Postgres and OpenSearch stay private behind the backend
- only HTTPS should be public

Bootstrap the VM after cloning the repo:

```bash
bash scripts/bootstrap_ubuntu_vm.sh /home/$USER/corpusagent2
```

Start backend:

```bash
./.venv/bin/python ./scripts/12_run_agent_api.py
```

Optional local static frontend on the VM:

```bash
./.venv/bin/python ./scripts/14_run_static_frontend.py
```

## 6. Recommended production shape

- OS: Ubuntu Server LTS
- reverse proxy: Nginx or Caddy
- backend: FastAPI on localhost
- DB/search: internal only
- public ports: `443` only

## 7. Current caveat

If a run returns only `python_fallback`, the backend is alive but the real retrieval/fetch path is not healthy yet. Fix that before treating the deployment as demo-ready.
