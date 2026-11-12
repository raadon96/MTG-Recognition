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
git clone https://github.com/raadon96/MTG-Recognition.git
```

SSH (if your SSH keys are configured):
```bash
git clone git@github.com:raadon96/MTG-Recognition.git
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

#### 5. Check the reference data

```bash
mtg reference status
```
The catalog (`data/reference/catalog.sqlite`), which holds each printing's pHash, is committed to git, so recognition works right after cloning. `mtg reference status` shows the MTGJSON release it was built from, how many printings have no pHash, the reference images on disk and the latest MTGJSON release, then whether the catalog is the latest release: up to date, outdated (with the command to run), or unknown when MTGJSON can't be reached.

**Optional — download the reference images:**
```bash
mtg reference download-images
```
Recognition doesn't need the reference images, only the pHashes in the catalog. To look at them, for example to compare a card image with its candidates' reference images while debugging matching, this downloads the images missing from `data/reference/images/` for the printings in the committed catalog (about 116k images and 12 GB on a first run). It changes nothing else: it doesn't download MTGJSON, rebuild the catalog or compute pHashes. Failed downloads are written to `data/reference/download_errors.log`, which is removed when a run has none; `mtg reference update` and `mtg reference build` keep their own log, `data/reference/image_errors.log`, and this command leaves it as it is. Running it again downloads only what is still missing.

**Maintainers only — build the reference data from a new MTGJSON release:**
```bash
mtg reference update
```
This downloads the MTGJSON AllPrintings database (kept in `data/reference/mtgjson/`), rebuilds the catalog, downloads the reference card images it needs that are missing from `data/reference/images/` (about 116k images and 12 GB, a local cache used only to compute the pHashes), then computes the pHashes. A printing already in the previous catalog keeps its pHash when its image URL is unchanged, so only new printings and printings with a new image URL are downloaded and hashed. It does nothing when the catalog already is the latest release.

**Maintainers only — rebuild the catalog's own MTGJSON release:**
```bash
mtg reference build
```
When the code that builds the catalog changes but the MTGJSON release doesn't, this rebuilds the catalog from the release it already is, and never changes which release that is. It uses the AllPrintings on disk when it is that release, without reaching MTGJSON; otherwise it downloads AllPrintings, which works only while that release is still MTGJSON's latest (MTGJSON publishes no older release), and fails otherwise, telling you to run `mtg reference update`. pHashes are reused as for `update`. `--no-cache` reuses nothing: it downloads AllPrintings and every image again and recomputes every pHash, for example after Scryfall replaces a preview image under the same URL.

#### 6. Run Streamlit app

```bash
uv run streamlit run src/mtg_recognition/app/app.py
```