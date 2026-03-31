#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$HOME/corpusagent2}"

sudo apt-get update
sudo apt-get install -y curl git python3 python3-venv python3-pip ca-certificates

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Expected a git checkout at $REPO_DIR"
  exit 1
fi

cd "$REPO_DIR"
uv venv .venv --python 3.11
uv sync --extra nlp-providers

./.venv/bin/python -m spacy download en_core_web_sm
./.venv/bin/python ./scripts/13_write_frontend_config.py

echo ""
echo "Bootstrap complete."
echo "Backend: ./.venv/bin/python ./scripts/12_run_agent_api.py"
echo "Frontend: ./.venv/bin/python ./scripts/14_run_static_frontend.py"
