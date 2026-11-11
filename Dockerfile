# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# Sets runtime/build environment variables:
    # - PYTHONDONTWRITEBYTECODE=1: avoid .pyc files
    # - PYTHONUNBUFFERED=1: immediate logs to stdout/stderr
    # - PATH="/app/.venv/bin:${PATH}": make venv binaries default (python, streamlit, mtg, etc.)
    # - Streamlit host/port defaults

WORKDIR /app
# All following relative paths are under /app.

# No system packages: opencv-python-headless needs no GUI libraries, and numba
# (through rembg) uses its own thread pool without libgomp1. No git either:
# `mtg benchmark run` records an unknown commit without it.


FROM base AS build
# Builds the venv. Only the finished venv leaves this stage, so uv stays out of
# the runtime image.

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# - UV_LINK_MODE=copy: uv copies files from its cache instead of linking them (the cache is a mount, gone after the RUN)
# - UV_PROJECT_ENVIRONMENT=/app/.venv: tell uv to use /app/.venv

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/
# uv at a fixed version, so that builds can be repeated.

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
# Dependencies first, in their own layer, so a change to src doesn't install them again.
# What --mount=type=cache means
#     - This is a BuildKit cache mount available only during that RUN.
#     - It mounts a persistent cache directory at /root/.cache/uv.
#     - uv stores downloaded wheels/sdists there.
#     - On next build, downloads are reused => faster builds.
# Important: this is build-time cache, not runtime volume mounting.
#     - It does not put host files into container image. It just caches package downloads between builds.
# Other flags:
#     - --locked: use exact versions from uv.lock
#     - --no-dev: skip dev deps
#     - --no-install-project: install dependencies only (not local package yet)

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev
# Installs the project (mtg-recognition) into the venv, as an editable install
# that points at /app/src, which the runtime stage copies to the same path.


FROM base AS runtime

RUN groupadd --system --gid 10001 appuser \
    && useradd --system --uid 10001 --gid appuser --create-home --home-dir /home/appuser appuser
# Create unpriviledged (i.e. non-root) user for security

COPY --from=build --chown=appuser:appuser /app/.venv /app/.venv
# The finished venv, owned by appuser as it is copied. A `chown -R` afterwards
# would store the whole venv a second time, in its own layer.

COPY src ./src
# The source, which the editable install in the venv points at

COPY data/reference/catalog.sqlite ./data/reference/catalog.sqlite
# Copy the catalog, which holds the pHashes

COPY models/segmentation/prod ./models/segmentation/prod
# Copy segmentation models

COPY models/ocr/prod ./models/ocr/prod
# Copy OCR models

RUN mkdir -p /app/output && chown appuser:appuser /app/output
# Only the folder the app writes to changes owner: /app/output, where
# `mtg benchmark run` writes its results

USER appuser
# Run as unpriviledged (i.e. non-root) user

EXPOSE 8501
# app listens on 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)"

CMD ["/app/.venv/bin/streamlit", "run", "src/mtg_recognition/app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
# Container startup command:
# - uses venv Streamlit directly
# - runs your app on all interfaces (0.0.0.0) so host port mapping works.
