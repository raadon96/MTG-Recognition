# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# Sets runtime/build environment variables:
    # - PYTHONDONTWRITEBYTECODE=1: avoid .pyc files
    # - PYTHONUNBUFFERED=1: immediate logs to stdout/stderr
    # - UV_LINK_MODE=copy: uv copies files instead of hard-link behavior (safer in Docker layers)
    # - UV_PROJECT_ENVIRONMENT=/app/.venv: tell uv to use /app/.venv
    # - PATH="/app/.venv/bin:${PATH}": make venv binaries default (python, streamlit, etc.)
    # - Streamlit host/port defaults

WORKDIR /app
# All following relative paths are under /app.

RUN groupadd --system --gid 10001 appuser \
    && useradd --system --uid 10001 --gid appuser --create-home --home-dir /home/appuser appuser
# Create unpriviledged (i.e. non-root) user for security

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
# Copies uv and uvx binaries from official uv image into your image (no separate install step needed)


RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Installs system libraries:
#     - git (needed for pip install git+...)
#     - OpenCV/ML runtime libs (libgl1, libglib2.0-0, etc.)
#     Cleans apt cache at end to keep image smaller.


FROM base AS deps
# Create a dependency-only stage.

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

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


FROM base AS runtime

ENV ULTRALYTICS_ASSET_CACHE_DIR=/home/appuser/.cache/ultralytics-assets
# Set env var directory to mount Docker volume

COPY --from=deps /app/.venv /app/.venv
# Copy prebuilt virtualenv from deps stage (fast, reusable).

COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY src ./src
# Copy project metadata + source

COPY data/reference/catalog.sqlite ./data/reference/catalog.sqlite
# Copy the catalog

COPY data/reference/index/phash/hashes.npy ./data/reference/index/phash/hashes.npy
COPY data/reference/index/phash/index_to_uuid.json ./data/reference/index/phash/index_to_uuid.json
# Copy pHash artifacts

COPY models/segmentation/prod ./models/segmentation/prod
# COPY mobileclip_blt.ts ./
# COPY mobileclip2_b.ts ./
# Copy segmentation models and mobileclip files

COPY models/ocr/prod ./models/ocr/prod
# Copy OCR models

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
# Copy startup script (entry point) into the image.

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev
# Again uses build cache mount for uv cache.
# This installs your local project (mtg-recognition) into the venv in runtime stage.

# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv run paddlex --install hpi-cpu
# Install PaddleX CPU high-performance inference plugin in the uv-managed environment.

RUN mkdir -p "${ULTRALYTICS_ASSET_CACHE_DIR}" \
    && mkdir -p /home/appuser/.paddlex/fonts \
    && curl -fsSL https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/simfang.ttf \
       -o /home/appuser/.paddlex/fonts/simfang.ttf \
    && chown -R appuser:appuser /app /home/appuser/.cache /home/appuser/.paddlex
# Create cache directory and give ownership to appuser, so downloads can write there.

USER appuser
# Run as unpriviledged (i.e. non-root) user

EXPOSE 8501
# app listens on 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)"

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
# Set start-up script as ENTRYPOINT, so it runs on every container start before Streamlit.

CMD ["/app/.venv/bin/streamlit", "run", "src/mtg_recognition/app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
# Container startup command:
# - uses venv Streamlit directly
# - runs your app on all interfaces (0.0.0.0) so host port mapping works.
