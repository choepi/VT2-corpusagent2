#!/usr/bin/env bash
# Phase B driver: ingest the downloaded shards to reach ~13M, embedding only the
# NEW rows (existing 2M keep their embeddings via the no-embedding upsert path).
# Resumable: the long embed/index stages continue from the last committed batch.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
export CORPUSAGENT2_PG_BACKFILL_ENCODE_BATCH_SIZE=128
export CORPUSAGENT2_PG_BACKFILL_FETCH_BATCH_SIZE=2048
OS_PW="${CORPUSAGENT2_OPENSEARCH_PASSWORD:-VerySecurePassword123!}"
banner() { echo; echo "==================== $* ===================="; date; }
psql() { docker exec corpus_postgres psql -U corpus -d corpus_db "$@"; }

banner "STAGE 1/7: stage incoming -> ccnews_staged"
"$PY" scripts/00_stage_ccnews_files.py || { echo "S1_FAILED"; exit 1; }

banner "STAGE 2/7: prepare -> parquet (~13M, dedup)"
"$PY" scripts/01_prepare_dataset.py || { echo "S2_FAILED"; exit 1; }

banner "STAGE 3/7: drop pgvector index + ensure NULL-fetch helper index"
psql -c "DROP INDEX IF EXISTS idx_article_corpus_embedding_ivfflat;" || { echo "S3_FAILED"; exit 1; }
psql -c "DROP INDEX IF EXISTS idx_article_corpus_embedding_hnsw;"
psql -c "CREATE INDEX IF NOT EXISTS idx_acorpus_null_emb ON article_corpus (doc_id) WHERE dense_embedding IS NULL;"

banner "STAGE 4/7: ingest parquet (upsert; existing 2M keep embeddings)"
"$PY" scripts/10_ingest_parquet_to_postgres.py || { echo "S4_FAILED"; exit 1; }
psql -t -c "SELECT count(*) total, count(*) FILTER (WHERE dense_embedding IS NULL) to_embed FROM article_corpus;"

banner "STAGE 5/7: embed NEW NULL rows (E5 GPU) -- the long pole"
"$PY" scripts/26_backfill_pgvector_embeddings.py || { echo "S5_FAILED"; exit 1; }

banner "STAGE 6/7: rebuild pgvector IVFFlat (~13M)"
"$PY" scripts/11_build_pgvector_index.py || { echo "S6_FAILED"; exit 1; }

banner "STAGE 7/7: OpenSearch recreate (bigger heap) + clean BM25 reindex (~13M)"
( cd deploy && docker compose -f docker-compose.yml -f docker-compose.scale.yml up -d --no-deps --force-recreate opensearch ) || { echo "S7_FAILED"; exit 1; }
for i in $(seq 1 60); do
  st=$(docker exec os_news curl -sk -u "admin:${OS_PW}" "https://localhost:9200/_cluster/health" 2>/dev/null | grep -oE '"status":"(green|yellow)"')
  [ -n "$st" ] && { echo "opensearch ready ($st)"; break; }
  sleep 6
done
CORPUSAGENT2_OPENSEARCH_RECREATE_INDEX=true CORPUSAGENT2_OPENSEARCH_BULK_BATCH_SIZE=2000 "$PY" scripts/21_bulk_index_opensearch.py || { echo "S7_FAILED"; exit 1; }

banner "INGEST_13M_DONE"
psql -t -c "SELECT count(*) total, count(*) FILTER (WHERE dense_embedding IS NOT NULL) embedded FROM article_corpus;"
