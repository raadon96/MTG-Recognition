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
uv run pytest tests/unit/engine/segmentation/
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
Three-path pipeline per image, tried in order:
1. **Bypass** (`yolo.py`): if the image aspect ratio ≈ MTG card ratio (63/88 ± 3 %) and the border is dark → resize only. Handles pre-cropped scans without running any model.
2. **YOLO** (`yolo.py`): `yoloe-11s-seg` batch inference. Each YOLO polygon mask is converted to 4 corners via `corners_from_binary_mask` → perspective warp.
3. **Fallback** (`box.py` then `rembg.py`): classical Otsu CV detector first (fast, good for uniform scanner beds); rembg background removal last resort (slow, works on photos with clear backgrounds).

Warped output size is clamped between `WARPED_MIN_SIZE = (488, 680)` and `WARPED_MAX_SIZE = (976, 1360)` (width × height in pixels) so that close-up phone photos don't produce unbounded images while scans aren't unnecessarily downscaled.

Shared geometry helpers (`seg_utils.py`): `CARD_ASPECT_RATIO`, `WARPED_MIN_SIZE`, `WARPED_MAX_SIZE`, `clamp_size`, `warp_card`, `valid_aspect_ratio`, `corners_from_binary_mask`, and the `CardSegment` / `ImageSegmentationResult` dataclasses. All segmentors import from here to avoid circular dependencies.

**Stage 2 — Matching** (`engine/matching/`, `engine/ocr/`):
1. **PHash matcher**: Perceptual hash (256-bit, 16×16) compared against precomputed numpy index to get top-k candidates fast.
2. **OCR fuzzy matcher**: PaddleOCR extracts text from the card's top ~11% (name ROI) and bottom ~13% (info ROI: set code, collector number, artist). `rapidfuzz` scores OCR attributes against candidates with weights (set code = 2.0, name = 1.0, etc.) to select the best match. The predict call is wrapped in `try/except` — PaddlePaddle's C++ backend throws intermittent `RuntimeError: std::exception` on some inputs; catching it degrades to zero-vector scoring so the phash ranking holds.

`ocr_utils.py` contains all text-extraction helpers: `extract_setcode_candidates`, `extract_collector_number`, `extract_artist_name`, `extract_set_release_year`, `extract_power_toughness`. Artist extraction joins all info-ROI tokens into one string, splits on whitespace, and keeps only word-like tokens (letters/hyphens/apostrophes, no digits or symbols), discarding noise such as set-code artifacts (`TDM·DE`) and copyright boilerplate. `Properties.artist` is `str | None`; all other scalar fields are also `str | None` except `setcode_candidates: list[str]`.

`ocr.py` exposes a `debug=False` flag on `extract_properties` and `find_best_matches`. When `True`, both return an `OCRDebugData` frozen dataclass containing the annotated info-ROI image (PaddleOCR bounding boxes), raw OCR texts/scores, extracted `Properties`, and a per-attribute score matrix for the top candidates.

`pipeline.py` exposes a single `Pipeline.find_best_matches` method returning a 4-tuple `(matches, scores, ocr_debug, diag)`. `diag` is a `PipelineDiagnostics` dataclass (four timing floats: `segmentation_ms`, `phash_ms`, `ocr_ms`, `total_ms`) — always populated, near-zero overhead. `ocr_debug` is `OCRDebugData | None`, populated only when `debug=True`. These two are intentionally separate: timing is performance monitoring, `OCRDebugData` is OCR inspection data. A `_timer(diag, attr)` context manager handles per-stage measurement without boilerplate.

**App layer** (`app/`):
- `app.py` — Streamlit UI for batch uploads. Sidebar toggle `"🔬 Debug Mode"` enables the `OCRDebugData` expanders in the detail view (annotated ROI, raw OCR table, extracted vs. best-match table, score matrix). The detail view reads all data directly from `BatchResult` — it does **not** re-run the pipeline.
- `batch_processor.py` — orchestrates per-image pipeline, returns `BatchResult` dataclasses. `BatchResult` carries the full top-K `matches: list[Row]`, `scores: list[float]`, `ocr_debug: OCRDebugData | None`, and `timings_ms: dict[str, float] | None` so the detail view is self-contained. Owns `K_PHASH_CANDIDATES` and `K_TOP_MATCHES` constants (imported by `app.py`) so both code paths always use the same pipeline parameters. Batch processing always runs with `debug=True` so `OCRDebugData` is available on demand without a second pipeline call.

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
| Segmentation (YOLO primary) | `src/mtg_recognition/engine/segmentation/yolo.py` |
| Segmentation (shared helpers) | `src/mtg_recognition/engine/segmentation/seg_utils.py` |
| pHash matcher | `src/mtg_recognition/engine/matching/phash_matcher.py` |
| Database job | `src/mtg_recognition/jobs/build_database.py` |
