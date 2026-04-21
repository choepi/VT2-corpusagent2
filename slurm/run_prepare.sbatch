#!/bin/bash -l
#SBATCH --job-name=ca2_prepare
#SBATCH --partition=gpu_top
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=cai_nlp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/common.sh"

ca2_init_job "${SCRIPT_DIR}" "ca2_prepare"
ca2_bootstrap_venv_if_missing
ca2_stage_project
ca2_activate_venv

cd "${RUN_ROOT}"
python scripts/00_stage_ccnews_files.py
python scripts/01_prepare_dataset.py

rsync -a "${RUN_ROOT}/data/processed/" "${PROJECT_ROOT}/data/processed/"
rsync -a "${RUN_ROOT}/outputs/" "${PROJECT_ROOT}/outputs/"

echo "Prepare stage completed successfully"
