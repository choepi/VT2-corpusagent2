#!/bin/bash -l
#SBATCH --job-name=ca2_bundle
#SBATCH --partition=gpu_top
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=cai_nlp
#SBATCH --output=/home/%u/corpusagent2/log/ca2_bundle/%x_%j.out
#SBATCH --error=/home/%u/corpusagent2/log/ca2_bundle/%x_%j.err

set -euo pipefail

PROJECT_ROOT="/home/${USER}/corpusagent2"
RUN_ROOT="/scratch/${USER}/corpusagent2/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
CACHE_ROOT="${RUN_ROOT}/cache"
LOG_DIR="${PROJECT_ROOT}/log/ca2_bundle"

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "ERROR: PROJECT_ROOT does not exist: ${PROJECT_ROOT}" >&2
  exit 1
fi

if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
  echo "ERROR: Python environment missing: ${PROJECT_ROOT}/.venv" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}" "${CACHE_ROOT}" "${LOG_DIR}"

rsync -aL --delete --exclude '.git' --exclude '.venv' "${PROJECT_ROOT}/" "${RUN_ROOT}/"

source "${PROJECT_ROOT}/.venv/bin/activate"

export HF_HOME="${CACHE_ROOT}/hf"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/hf/transformers"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf/hub"
export MPLCONFIGDIR="${CACHE_ROOT}/mpl"
export PYTHONUNBUFFERED=1
export CORPUSAGENT2_DENSE_BATCH_SIZE="${CORPUSAGENT2_DENSE_BATCH_SIZE:-128}"
export CORPUSAGENT2_SENTIMENT_DEVICE="${CORPUSAGENT2_SENTIMENT_DEVICE:-auto}"

cd "${RUN_ROOT}"
python scripts/27_build_prebuilt_bundle.py --clean-existing "$@"

rsync -a "${RUN_ROOT}/outputs/prebuilt/" "${PROJECT_ROOT}/outputs/prebuilt/"

echo "Prebuilt transfer bundle completed successfully"
