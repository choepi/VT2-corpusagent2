#!/bin/bash -l
#SBATCH --job-name=ca2_eval
#SBATCH --partition=gpu_top
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=cai_nlp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/common.sh"

ca2_init_job "${SCRIPT_DIR}" "ca2_eval"
ca2_bootstrap_venv_if_missing

if [ ! -f "${PROJECT_ROOT}/data/indices/doc_metadata.parquet" ]; then
  echo "ERROR: Missing retrieval assets under ${PROJECT_ROOT}/data/indices" >&2
  exit 1
fi

ca2_stage_project
ca2_activate_venv

cd "${RUN_ROOT}"
python scripts/03_evaluate_retrieval.py
python scripts/04_evaluate_faithfulness.py

rsync -a "${RUN_ROOT}/outputs/" "${PROJECT_ROOT}/outputs/"

echo "Evaluation completed successfully"
