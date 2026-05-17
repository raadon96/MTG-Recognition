# Decksray MTG Recognition

## Run with Docker
Run the MTG Recognition Streamlit app locally with Docker Compose.

### Prerequisites
- Install Docker Engine: <https://docs.docker.com/engine/install/>
- Confirm Docker is running:
```bash
docker --version
```

### Quick Start

#### 1. Clone the repository

HTTPS (recommended):
```bash
git clone https://github.com/decksray/MTG-Recognition.git
```

SSH (if your SSH keys are configured):
```bash
git clone git@github.com:decksray/MTG-Recognition.git
```

#### 2. Enter the project directory

```bash
cd MTG-Recognition/
```

#### 3. Build and run with Docker Compose

First run (build image + start app):
```bash
docker compose up --build
```

Next runs (start app without rebuild):
```bash
docker compose up
```

#### 4. Open the app

Open in your browser:
```text
http://localhost:8501
```

### Notes

- The first startup may take longer while models and services initialize.
- The container runs as a non-root user and includes a Docker healthcheck.
- MobileCLIP assets are persisted in a named Docker volume (`mtg_ultra_cache`), so large files are downloaded once and reused across restarts.
- Press `Ctrl+C` to stop when running in the foreground.

### Useful Commands

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
---
---

## Setup for Development
Setup project locally for development.

### Quick Start

#### 1. Install uv
Install uv, see [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

#### 2. cd into repository
```bash
cd MTG-Recognition/
```

#### 3. Install dependencies
```bash
uv sync --dev
```

#### 4. Activate virtual environment
```bash
source .venv/bin/activate
```

#### 5. Download reference data
Build the local card metadata database (MobileCache), download reference images, and build the pHash index in one shot. 

```bash
mtg sync
```
**Note:** This step requires a network connection (~500 MB download). Subsequent runs are incremental — only artifacts that are out of date are rebuilt. 


To force a full rebuild (e.g. for trouble-shooting):
```bash
mtg sync --force
```

#### 6. Run Streamlit app

```bash
uv run streamlit run src/mtg_recognition/app/app.py
```