#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$HOME/corpusagent2}"
shift $(( $# > 0 ? 1 : 0 ))

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Expected a git checkout at $REPO_DIR"
  exit 1
fi

cd "$REPO_DIR"
python3 ./scripts/22_prepare_vm_stack.py --install-system "$@"

echo ""
echo "Bootstrap complete."
echo "Backend/API:"
echo "  cd deploy"
echo "  docker compose -f docker-compose.yml -f docker-compose.mcp.yml up -d --no-recreate postgres opensearch"
echo "  docker compose -f docker-compose.yml -f docker-compose.mcp.yml up -d --build --no-deps corpusagent2-api corpusagent2-mcp"
echo "Tunnel:  ./.venv/bin/python ./scripts/23_start_cloudflared_tunnel.py"
echo "Services: ./.venv/bin/python ./scripts/24_configure_vm_services.py"
