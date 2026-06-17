#!/usr/bin/env bash
# Autonomous driver for the 2M corpus scale-up: finishes embeddings, builds the
# pgvector index, bumps OpenSearch heap, and BM25-indexes. Resumable at every stage.
# Run from repo root. Each stage logs a banner so progress is visible in the task output.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

export CORPUSAGENT2_PG_BACKFILL_ENCODE_BATCH_SIZE=128
export CORPUSAGENT2_PG_BACKFILL_FETCH_BATCH_SIZE=2048
PY=./.venv/Scripts/python.exe

banner() { echo; echo "==================== $* ===================="; date; }

banner "STAGE 1/4: embedding backfill (E5, GPU)"
"$PY" scripts/26_backfill_pgvector_embeddings.py || { echo "STAGE1_FAILED"; exit 1; }

banner "STAGE 2/4: pgvector IVFFlat index"
"$PY" scripts/11_build_pgvector_index.py || { echo "STAGE2_FAILED"; exit 1; }

banner "STAGE 3/4: recreate OpenSearch with larger heap"
( cd deploy && docker compose -f docker-compose.yml -f docker-compose.scale.yml up -d --no-deps --force-recreate opensearch ) || { echo "STAGE3_FAILED"; exit 1; }
# Wait for the CLUSTER to be ready for indexing (status yellow/green), not just a
# TCP ping — bulk indexing too early returns 503 Service Unavailable.
OS_PW="${CORPUSAGENT2_OPENSEARCH_PASSWORD:-VerySecurePassword123!}"
for i in $(seq 1 60); do
  st=$(docker exec os_news curl -sk -u "admin:${OS_PW}" "https://localhost:9200/_cluster/health" 2>/dev/null | grep -oE '"status":"(green|yellow)"')
  if [ -n "$st" ]; then echo "opensearch cluster ready ($st)"; break; fi
  sleep 6
done

banner "STAGE 4/4: OpenSearch BM25 bulk index (2M, clean recreate)"
CORPUSAGENT2_OPENSEARCH_RECREATE_INDEX=true CORPUSAGENT2_OPENSEARCH_BULK_BATCH_SIZE="${CORPUSAGENT2_OPENSEARCH_BULK_BATCH_SIZE:-5000}" "$PY" scripts/21_bulk_index_opensearch.py || { echo "STAGE4_FAILED"; exit 1; }

banner "SCALE_PIPELINE_DONE"
docker exec corpus_postgres psql -U corpus -d corpus_db -t -c "SELECT count(*) total, count(*) FILTER (WHERE dense_embedding IS NOT NULL) embedded FROM article_corpus;"
