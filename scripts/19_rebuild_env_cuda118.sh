#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[info] project_root=$PROJECT_ROOT"

rm -rf .venv
rm -f uv.lock

echo "[step] creating venv"
uv venv .venv --python 3.11

echo "[step] syncing dependencies with CUDA 11.8 torch source"
uv sync --extra nlp-providers

echo "[step] exporting fully pinned requirements"
uv export --extra nlp-providers --format requirements-txt -o requirements-cu118.txt

echo "[step] downloading provider assets"
./.venv/bin/python ./scripts/17_download_provider_assets.py

echo "[step] verifying runtime environment"
./.venv/bin/python ./scripts/18_verify_cuda118_env.py

echo "[done] CUDA 11.8 environment rebuilt."
