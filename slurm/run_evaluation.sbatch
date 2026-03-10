#!/bin/bash -l
#SBATCH --job-name=ca2_eval
#SBATCH --partition=gpu_top
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=cai_nlp
#SBATCH --output=/home/%u/corpusagent2/log/ca2_eval/%x_%j.out
#SBATCH --error=/home/%u/corpusagent2/log/ca2_eval/%x_%j.err

set -euo pipefail

PROJECT_ROOT="/home/${USER}/corpusagent2"
RUN_ROOT="/scratch/${USER}/corpusagent2/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
OUTPUT_ROOT="${RUN_ROOT}/outputs"
CACHE_ROOT="${RUN_ROOT}/cache"
LOG_DIR="${PROJECT_ROOT}/log/ca2_eval"

if [ ! -f "${PROJECT_ROOT}/data/indices/doc_metadata.parquet" ]; then
  echo "ERROR: Missing retrieval assets under ${PROJECT_ROOT}/data/indices" >&2
  exit 1
fi

if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
  echo "ERROR: Python environment missing: ${PROJECT_ROOT}/.venv" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}" "${OUTPUT_ROOT}" "${CACHE_ROOT}" "${LOG_DIR}"

rsync -aL --delete --exclude '.git' --exclude '.venv' "${PROJECT_ROOT}/" "${RUN_ROOT}/"

source "${PROJECT_ROOT}/.venv/bin/activate"

export HF_HOME="${CACHE_ROOT}/hf"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/hf/transformers"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf/hub"
export MPLCONFIGDIR="${CACHE_ROOT}/mpl"
export PYTHONUNBUFFERED=1

cd "${RUN_ROOT}"
python scripts/03_evaluate_retrieval.py
python scripts/04_evaluate_faithfulness.py

rsync -a "${RUN_ROOT}/outputs/" "${PROJECT_ROOT}/outputs/"

echo "Evaluation completed successfully"
