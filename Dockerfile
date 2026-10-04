# syntax=docker/dockerfile:1.7
# Multi-stage build: dependencies resolved from uv.lock in a builder image, copied into a slim
# runtime that runs as a non-root user. The model bundles are NOT in the image; they are mounted
# (Compose) or fetched at deploy time (Phase 6) under /models.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SECOPS_MODEL_DIR=/models \
    SECOPS_HOST=0.0.0.0 \
    SECOPS_PORT=8000
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["secops-api"]

# Cloud Run target: the exported bundles are baked into the image because Cloud Run has no
# persistent volume. Build with: docker build --target cloudrun -t secops-api:cloudrun .
# after exporting the champions into deploy/models (see scripts/deploy_cloud_run.sh).
FROM runtime AS cloudrun
COPY --chown=app:app deploy/models /models

# Dashboard target: Streamlit over the API (profile `ui` in Compose). The dashboard group is
# resolved from the same lockfile in the builder, so the runtime stage needs no package manager.
FROM builder AS dashboard-builder
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group dashboard

FROM runtime AS dashboard
COPY --from=dashboard-builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app dashboard /app/dashboard
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
