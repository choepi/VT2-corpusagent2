#!/bin/bash -l
#SBATCH --job-name=vt2_coding_agent
#SBATCH --partition=gpu_top
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --account=cai_nlp
#SBATCH --chdir=/cluster/home/dongtten/VT2/VT2-corpusagent2
#SBATCH --output=/cluster/home/dongtten/VT2/VT2-corpusagent2/slurm/%x_%j.out
#SBATCH --error=/cluster/home/dongtten/VT2/VT2-corpusagent2/slurm/%x_%j.err

set -euo pipefail
set -x

PROJECT_ROOT="/cluster/home/dongtten/VT2/VT2-corpusagent2"
PROMPT_FILE="${PROJECT_ROOT}/slurm/coding_agent_prompt.md"

JOB_ROOT="/scratch/${USER}/vt2_coding_agent_${SLURM_JOB_ID}"
RUN_ROOT="${JOB_ROOT}/repo"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/coding_agent/${SLURM_JOB_ID}"

CACHE_ROOT="/scratch/${USER}/hf_cache"
VENV_ROOT="/scratch/${USER}/venvs/vt2_coding_agent_uv"

# L40S-safe first version. Do not start with 30B/32k. That was too greedy.
MODEL_ID="Qwen/Qwen2.5-Coder-14B-Instruct"
MAX_MODEL_LEN="16384"

VLLM_PORT="$((18000 + SLURM_JOB_ID % 20000))"
VLLM_API_KEY="local-slurm-token"

unset VIRTUAL_ENV || true
unset CONDA_PREFIX || true
unset PYTHONPATH || true

export PATH="/cluster/home/${USER}/.local/bin:/usr/local/bin:/usr/bin:/bin"
export HF_HOME="${CACHE_ROOT}/hf_home"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export HF_HUB_CACHE="${CACHE_ROOT}/hub"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export TOKENIZERS_PARALLELISM=false

export AGENT_RUN_ROOT="${RUN_ROOT}"
export AGENT_OUTPUT_ROOT="${OUTPUT_ROOT}"
export AGENT_PROMPT_FILE="${PROMPT_FILE}"
export AGENT_MODEL_ID="${MODEL_ID}"
export AGENT_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export AGENT_API_KEY="${VLLM_API_KEY}"
export AGENT_MAX_STEPS="24"
export AGENT_CONTEXT_CHAR_BUDGET="45000"

mkdir -p "${JOB_ROOT}" "${RUN_ROOT}" "${OUTPUT_ROOT}" "${CACHE_ROOT}" "$(dirname "${VENV_ROOT}")"

echo "=== job started ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "HOST=$(hostname)"
date

echo "=== gpu ==="
nvidia-smi || true

echo "=== required files ==="
test -d "${PROJECT_ROOT}"
test -f "${PROMPT_FILE}"
test -f "${PROJECT_ROOT}/slurm/coding_agent_runner.py"

echo "=== checking uv ==="
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found on compute node."
  echo "PATH=${PATH}"
  echo "You probably have uv on login in ~/.local/bin, but not visible here, or it is not installed."
  echo "Install uv once in your home bin, or use your existing run_prebuilt_bundle env strategy."
  exit 20
fi
UV_BIN="$(command -v uv)"
echo "UV_BIN=${UV_BIN}"
uv --version

echo "=== staging repo to scratch ==="
rsync -aL --delete \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude ".mypy_cache" \
  --exclude ".ruff_cache" \
  --exclude ".git/lfs" \
  --exclude "data/raw" \
  --exclude "data/processed" \
  --exclude "data/indices" \
  --exclude "outputs" \
  --exclude "log" \
  "${PROJECT_ROOT}/" "${RUN_ROOT}/"

echo "=== creating/reusing uv venv on scratch ==="
if [ ! -x "${VENV_ROOT}/bin/python" ]; then
  uv venv --python 3.11 "${VENV_ROOT}" || uv venv --python 3.12 "${VENV_ROOT}" || uv venv "${VENV_ROOT}"
fi

PYTHON_BIN="${VENV_ROOT}/bin/python"
echo "PYTHON_BIN=${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo "=== installing runtime dependencies inside slurm job ==="
uv pip install --python "${PYTHON_BIN}" --upgrade pip wheel setuptools
uv pip install --python "${PYTHON_BIN}" "openai>=1.0.0" "vllm>=0.8.3"

echo "=== installing repo package inside slurm job ==="
cd "${RUN_ROOT}"
uv pip install --python "${PYTHON_BIN}" -e .

echo "=== versions ==="
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
for pkg in ["openai", "vllm"]:
    print(pkg, md.version(pkg))
PY

echo "=== starting vllm ==="
"${VENV_ROOT}/bin/vllm" serve "${MODEL_ID}" \
  --host 127.0.0.1 \
  --port "${VLLM_PORT}" \
  --api-key "${VLLM_API_KEY}" \
  --dtype auto \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization 0.88 \
  --trust-remote-code \
  > "${OUTPUT_ROOT}/vllm_server.log" 2>&1 &

VLLM_PID="$!"
echo "${VLLM_PID}" > "${OUTPUT_ROOT}/vllm_server.pid"

cleanup() {
  echo "=== cleanup ==="
  kill "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== waiting for vllm ==="
for i in $(seq 1 240); do
  if curl -fsS \
    -H "Authorization: Bearer ${VLLM_API_KEY}" \
    "http://127.0.0.1:${VLLM_PORT}/v1/models" \
    > "${OUTPUT_ROOT}/vllm_models.json" 2>/dev/null; then
    echo "vLLM ready"
    break
  fi

  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM died. Last 200 log lines:"
    tail -200 "${OUTPUT_ROOT}/vllm_server.log" || true
    exit 30
  fi

  sleep 10
done

if [ ! -s "${OUTPUT_ROOT}/vllm_models.json" ]; then
  echo "vLLM did not become ready. Last 200 log lines:"
  tail -200 "${OUTPUT_ROOT}/vllm_server.log" || true
  exit 31
fi

echo "=== running coding agent inside slurm job ==="
"${PYTHON_BIN}" "${PROJECT_ROOT}/slurm/coding_agent_runner.py" \
  > "${OUTPUT_ROOT}/agent_stdout.log" \
  2> "${OUTPUT_ROOT}/agent_stderr.log"

echo "=== saving final git state ==="
cd "${RUN_ROOT}"
git status --short > "${OUTPUT_ROOT}/git_status.txt" || true
git diff -- . > "${OUTPUT_ROOT}/agent.patch" || true
git diff --stat -- . > "${OUTPUT_ROOT}/agent_diff_stat.txt" || true

tar -czf "${OUTPUT_ROOT}/agent_state.tar.gz" -C "${RUN_ROOT}" ".agent_state" || true

echo "=== output root ==="
echo "${OUTPUT_ROOT}"
find "${OUTPUT_ROOT}" -maxdepth 2 -type f -print | sort

date
echo "=== job finished ==="
