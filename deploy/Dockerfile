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

COPY config ./config
COPY scripts ./scripts
COPY src ./src

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
