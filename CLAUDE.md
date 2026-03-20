# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MTG card recognition app: takes a photo of a Magic: The Gathering card, segments it, extracts text via OCR, and matches it against a ~1M-card SQLite database. Served as a Streamlit web app via Docker.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager

## Commands

```bash
# Install dependencies
uv sync --dev

# Run app locally
uv run streamlit run src/mtg_recognition/app/app.py

# Docker (recommended)
docker compose up --build   # first run
docker compose up           # subsequent
docker compose logs -f

# Tests
uv run pytest
uv run pytest tests/unit/engine/test_ocr.py
uv run pytest tests/unit/engine/test_ocr.py::TestExtractCollectorNumber::test_simple_collector_number
uv run pytest -k "collector_number" -x -vv

# Sanity check imports
uv run python -c "import mtg_recognition; print('ok')"
uv run python -m compileall src tests
```

No linter/formatter is configured — match the style of surrounding code.

## Architecture

Recognition runs as a two-stage pipeline:

**Stage 1 — Segmentation** (`engine/segmentation/`):
YOLO model detects card boundaries → Canny edge detection → corner finding → perspective transform to 488×680px canonical card size.

**Stage 2 — Matching** (`engine/matching/`, `engine/ocr/`):
1. **PHash matcher**: Perceptual hash (256-bit, 16×16) compared against precomputed numpy index to get top-k candidates fast.
2. **OCR fuzzy matcher**: PaddleOCR (HPI/ONNX mode) extracts text from the card's top ~11% (name) and bottom ~13% (set code, collector number, artist). `rapidfuzz` scores OCR attributes against candidates with weights (set code = 2.0, name = 1.0, etc.) to select the best match.

**App layer** (`app/`):
- `app.py` — Streamlit UI for single and batch uploads.
- `batch_processor.py` — orchestrates per-image pipeline, returns `BatchResult` dataclasses with match + diagnostic timings.

**Data jobs** (`jobs/`):
- `build_database.py` — ingests MTGJson/Scryfall data into `MobileCache.sqlite`.
- `schemas.py` — SQLModel ORM models (`Cards`, `CardIdentifiers`, `Sets`).

**Database** (`database.py`):
Read-only SQLite opened with `mode=ro&nolock=1`. 35 MB, ~1M+ cards.

## Data & Assets

All reference data lives under `data/reference/` (not in git):
- `images/` — ~1M card images indexed by UUID
- `MobileCache.sqlite` — card metadata database
- `index/phash/hashes.npy` + `index_to_uuid.json` — prebuilt pHash index
- YOLO `.pt` models: `engine/segmentation/models/prod/`
- OCR models: `engine/ocr/models/prod/`

## Key Entry Points

| Purpose | Path |
|---------|------|
| Streamlit app | `src/mtg_recognition/app/app.py` |
| End-to-end pipeline | `src/mtg_recognition/engine/pipeline.py` |
| OCR logic + fuzzy ranking | `src/mtg_recognition/engine/ocr/ocr.py` |
| OCR parsing helpers | `src/mtg_recognition/engine/ocr/ocr_utils.py` |
| Segmentation | `src/mtg_recognition/engine/segmentation/segment.py` |
| pHash matcher | `src/mtg_recognition/engine/matching/phash_matcher.py` |
| Database job | `src/mtg_recognition/jobs/build_database.py` |
