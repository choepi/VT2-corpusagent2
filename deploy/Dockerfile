FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    CORPUSAGENT2_PROJECT_ROOT=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker

COPY pyproject.toml uv.lock README.md ./
COPY scripts/17_download_provider_assets.py ./scripts/17_download_provider_assets.py

ARG CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS=false
RUN if [ "$CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS" = "true" ]; then \
      uv sync --frozen --extra nlp-providers --no-install-project; \
    else \
      uv sync --frozen --no-install-project; \
    fi

ARG CORPUSAGENT2_DOCKER_DOWNLOAD_PROVIDER_ASSETS=false
RUN if [ "$CORPUSAGENT2_DOCKER_DOWNLOAD_PROVIDER_ASSETS" = "true" ]; then \
      .venv/bin/python scripts/17_download_provider_assets.py; \
    fi

COPY config ./config
COPY scripts ./scripts
COPY src ./src

RUN if [ "$CORPUSAGENT2_DOCKER_INSTALL_NLP_PROVIDERS" = "true" ]; then \
      uv sync --frozen --extra nlp-providers --inexact; \
    else \
      uv sync --frozen --inexact; \
    fi

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8001 8765

CMD ["python", "/app/scripts/12_run_agent_api.py"]
