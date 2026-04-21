#!/bin/bash -l
#SBATCH --job-name=ca2_framework
#SBATCH --partition=gpu_top
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=cai_nlp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/common.sh"

ca2_init_job "${SCRIPT_DIR}" "ca2_framework"
ca2_bootstrap_venv_if_missing

if [ ! -f "${PROJECT_ROOT}/data/indices/doc_metadata.parquet" ]; then
  echo "ERROR: Missing retrieval assets under ${PROJECT_ROOT}/data/indices" >&2
  exit 1
fi

ca2_stage_project
ca2_activate_venv

cd "${RUN_ROOT}"
python scripts/05_run_nlp_tooling.py
python scripts/06_run_framework.py

rsync -a "${RUN_ROOT}/outputs/" "${PROJECT_ROOT}/outputs/"

echo "Framework run completed successfully"
