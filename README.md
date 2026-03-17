# MTG Recognition Docker Compose Guide

Run the MTG Recognition Streamlit app locally with Docker Compose.

## Prerequisites

- Install Docker Engine: <https://docs.docker.com/engine/install/>
- Confirm Docker is running:

```bash
docker --version
```

## Quick Start

### 1) Clone the repository

HTTPS (recommended):

```bash
git clone https://github.com/decksray/MTG-Recognition.git
```

SSH (if your SSH keys are configured):

```bash
git clone git@github.com:decksray/MTG-Recognition.git
```

### 2) Enter the project directory

```bash
cd MTG-Recognition
```

### 3) Build and run with Docker Compose

First run (build image + start app):

```bash
docker compose up --build
```

Next runs (start app without rebuild):

```bash
docker compose up
```

### 5) Open the app

Open in your browser:

```text
http://localhost:8501
```

## Notes

- The first startup may take longer while models and services initialize.
- The container runs as a non-root user and includes a Docker healthcheck.
- MobileCLIP assets are persisted in a named Docker volume (`mtg_ultra_cache`), so large files are downloaded once and reused across restarts.
- Press `Ctrl+C` to stop when running in the foreground.

## Useful Commands

View live logs:

```bash
docker compose logs -f
```

Run in background (detached):

```bash
docker compose up -d
```

Stop and remove containers/network:

```bash
docker compose down
```

Stop and also remove volumes (forces model re-download next run):

```bash
docker compose down -v
```
