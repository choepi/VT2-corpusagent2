FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    CORPUSAGENT2_PROJECT_ROOT=/app \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker

COPY pyproject.toml uv.lock README.md ./
COPY deploy/requirements.docker-cpu.txt ./deploy/requirements.docker-cpu.txt
COPY deploy/requirements.docker-nlp-providers.txt ./deploy/requirements.docker-nlp-providers.txt
COPY scripts/17_download_provider_assets.py ./scripts/17_download_provider_assets.py

ARG CORPUSAGENT2_DOCKER_TORCH_PROFILE=cpu
ARG CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS=false
RUN set -eu; \
    case "$CORPUSAGENT2_DOCKER_TORCH_PROFILE" in \
      cuda) \
        if [ "$CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS" = "true" ]; then \
          uv sync --frozen --extra nlp-providers --no-install-project; \
        else \
          uv sync --frozen --no-install-project; \
        fi; \
        ;; \
      cpu) \
        uv venv; \
        if [ "$CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS" = "true" ]; then \
          uv pip install \
            --torch-backend cpu \
            --index-strategy unsafe-best-match \
            --index-url https://pypi.org/simple \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            -r deploy/requirements.docker-cpu.txt \
            -r deploy/requirements.docker-nlp-providers.txt; \
        else \
          uv pip install \
            --torch-backend cpu \
            --index-strategy unsafe-best-match \
            --index-url https://pypi.org/simple \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            -r deploy/requirements.docker-cpu.txt; \
        fi; \
        ;; \
      *) \
        echo "Unsupported CORPUSAGENT2_DOCKER_TORCH_PROFILE=$CORPUSAGENT2_DOCKER_TORCH_PROFILE; expected cpu or cuda." >&2; \
        exit 2; \
        ;; \
    esac

ARG CORPUSAGENT2_DOCKER_DOWNLOAD_PROVIDER_ASSETS=false
RUN if [ "$CORPUSAGENT2_DOCKER_DOWNLOAD_PROVIDER_ASSETS" = "true" ]; then \
      .venv/bin/python scripts/17_download_provider_assets.py; \
    fi

# Bake the dense-retrieval model into the image so the runtime container is
# self-contained. The host bind mount in docker-compose.yml still wins at
# runtime when the host path is present — this only matters when the bind
# mount source is empty or missing. Default ON; opt out for slimmer images
# with --build-arg CORPUSAGENT2_DOCKER_BAKE_DENSE_MODEL=false.
ARG CORPUSAGENT2_DOCKER_BAKE_DENSE_MODEL=true
ARG CORPUSAGENT2_DOCKER_DENSE_MODEL_ID=intfloat/e5-base-v2
ARG CORPUSAGENT2_DOCKER_DENSE_MODEL_TARGET=/models/e5-base-v2
COPY deploy/bake_dense_model.py ./deploy/bake_dense_model.py
RUN if [ "$CORPUSAGENT2_DOCKER_BAKE_DENSE_MODEL" = "true" ]; then \
      mkdir -p "$CORPUSAGENT2_DOCKER_DENSE_MODEL_TARGET" && \
      HF_REPO="$CORPUSAGENT2_DOCKER_DENSE_MODEL_ID" \
      HF_TARGET="$CORPUSAGENT2_DOCKER_DENSE_MODEL_TARGET" \
      .venv/bin/python deploy/bake_dense_model.py \
      || (echo "Dense model bake failed for $CORPUSAGENT2_DOCKER_DENSE_MODEL_ID" >&2 && exit 1); \
    fi

COPY config ./config
COPY scripts ./scripts
COPY src ./src
COPY web ./web

RUN set -eu; \
    case "$CORPUSAGENT2_DOCKER_TORCH_PROFILE" in \
      cuda) \
        if [ "$CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS" = "true" ]; then \
          uv sync --frozen --extra nlp-providers --inexact; \
        else \
          uv sync --frozen --inexact; \
        fi; \
        ;; \
      cpu) \
        ;; \
      *) \
        echo "Unsupported CORPUSAGENT2_DOCKER_TORCH_PROFILE=$CORPUSAGENT2_DOCKER_TORCH_PROFILE; expected cpu or cuda." >&2; \
        exit 2; \
        ;; \
    esac

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8001 8765

CMD ["python", "/app/scripts/12_run_agent_api.py"]
