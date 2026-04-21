#!/bin/bash

ca2_init_job() {
  local script_dir="$1"
  local log_subdir="$2"
  local job_name="${SLURM_JOB_NAME:-ca2_job}"
  local job_id="${SLURM_JOB_ID:-manual}"
  local job_stdout=""
  local job_stderr=""

  export PROJECT_ROOT="$(cd "${script_dir}/.." && pwd)"
  export RUN_ROOT="/scratch/${USER}/corpusagent2/${job_name}_${job_id}"
  export OUTPUT_ROOT="${RUN_ROOT}/outputs"
  export CACHE_ROOT="${RUN_ROOT}/cache"
  export LOG_DIR="${PROJECT_ROOT}/log/${log_subdir}"

  mkdir -p "${RUN_ROOT}" "${OUTPUT_ROOT}" "${CACHE_ROOT}" "${LOG_DIR}"

  job_stdout="${LOG_DIR}/${job_name}_${job_id}.out"
  job_stderr="${LOG_DIR}/${job_name}_${job_id}.err"

  exec > >(tee -a "${job_stdout}") 2> >(tee -a "${job_stderr}" >&2)

  export HF_HOME="${CACHE_ROOT}/hf"
  export TRANSFORMERS_CACHE="${CACHE_ROOT}/hf/transformers"
  export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf/hub"
  export MPLCONFIGDIR="${CACHE_ROOT}/mpl"
  export PYTHONUNBUFFERED=1

  echo "[info] project_root=${PROJECT_ROOT}"
  echo "[info] run_root=${RUN_ROOT}"
  echo "[info] stdout_log=${job_stdout}"
  echo "[info] stderr_log=${job_stderr}"
}


ca2_bootstrap_venv_if_missing() {
  if [ -d "${PROJECT_ROOT}/.venv" ]; then
    return 0
  fi

  echo "[step] .venv missing; bootstrapping Python environment"

  cd "${PROJECT_ROOT}"
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv --python 3.11
    uv sync --extra nlp-providers
  elif command -v python3 >/dev/null 2>&1 && python3 -m uv --help >/dev/null 2>&1; then
    python3 -m uv venv .venv --python 3.11
    python3 -m uv sync --extra nlp-providers
  else
    echo "ERROR: uv is unavailable. Install uv or create ${PROJECT_ROOT}/.venv before submitting." >&2
    exit 1
  fi

  "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/17_download_provider_assets.py"
  "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/18_verify_cuda118_env.py"
}


ca2_stage_project() {
  rsync -aL --delete --exclude '.git' --exclude '.venv' "${PROJECT_ROOT}/" "${RUN_ROOT}/"
}


ca2_activate_venv() {
  source "${PROJECT_ROOT}/.venv/bin/activate"
}
