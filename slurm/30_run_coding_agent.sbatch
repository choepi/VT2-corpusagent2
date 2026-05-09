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

# Separate envs. Do not mix vLLM and repo deps.
VLLM_VENV_ROOT="/scratch/${USER}/venvs/vt2_vllm_clean_20260509"
AGENT_VENV_ROOT="/scratch/${USER}/venvs/vt2_agent_clean_20260509"

# L40S-safe first. Upgrade only after this works.
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
export UV_TORCH_BACKEND="auto"
export TOKENIZERS_PARALLELISM=false

export AGENT_RUN_ROOT="${RUN_ROOT}"
export AGENT_OUTPUT_ROOT="${OUTPUT_ROOT}"
export AGENT_PROMPT_FILE="${PROMPT_FILE}"
export AGENT_MODEL_ID="${MODEL_ID}"
export AGENT_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export AGENT_API_KEY="${VLLM_API_KEY}"
export AGENT_MAX_STEPS="24"
export AGENT_CONTEXT_CHAR_BUDGET="45000"

mkdir -p "${JOB_ROOT}" "${RUN_ROOT}" "${OUTPUT_ROOT}" "${CACHE_ROOT}" "$(dirname "${VLLM_VENV_ROOT}")" "$(dirname "${AGENT_VENV_ROOT}")"

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
  exit 20
fi
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

echo "=== creating vllm env ==="
if [ ! -x "${VLLM_VENV_ROOT}/bin/python" ]; then
  uv venv --python 3.11 "${VLLM_VENV_ROOT}" || uv venv --python 3.12 "${VLLM_VENV_ROOT}" || uv venv "${VLLM_VENV_ROOT}"
fi
VLLM_PYTHON="${VLLM_VENV_ROOT}/bin/python"

echo "=== installing vllm deps only ==="
uv pip install --python "${VLLM_PYTHON}" --upgrade pip wheel setuptools
uv pip install --python "${VLLM_PYTHON}" --torch-backend=auto "vllm>=0.8.5"

echo "=== vllm env sanity ==="
"${VLLM_PYTHON}" - <<'PY'
import torch
import vllm
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("vllm", vllm.__version__)
print("has_infer_schema", hasattr(torch.library, "infer_schema"))
if not hasattr(torch.library, "infer_schema"):
    raise SystemExit("torch.library.infer_schema missing; vLLM/Torch mismatch")
PY

echo "=== creating agent env ==="
if [ ! -x "${AGENT_VENV_ROOT}/bin/python" ]; then
  uv venv --python 3.11 "${AGENT_VENV_ROOT}" || uv venv --python 3.12 "${AGENT_VENV_ROOT}" || uv venv "${AGENT_VENV_ROOT}"
fi
AGENT_PYTHON="${AGENT_VENV_ROOT}/bin/python"

echo "=== installing agent/repo deps ==="
uv pip install --python "${AGENT_PYTHON}" --upgrade pip wheel setuptools
uv pip install --python "${AGENT_PYTHON}" "openai>=1.0.0"

cd "${RUN_ROOT}"
uv pip install --python "${AGENT_PYTHON}" -e .

echo "=== agent env sanity ==="
"${AGENT_PYTHON}" - <<'PY'
import importlib.metadata as md
print("openai", md.version("openai"))
try:
    print("corpusagent2 installed")
    import corpusagent2
except Exception as exc:
    print("corpusagent2 import warning:", repr(exc))
PY

echo "=== starting vllm ==="
"${VLLM_VENV_ROOT}/bin/vllm" serve "${MODEL_ID}" \
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
export PATH="${AGENT_VENV_ROOT}/bin:${PATH}"

"${AGENT_PYTHON}" "${PROJECT_ROOT}/slurm/coding_agent_runner.py" \
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
