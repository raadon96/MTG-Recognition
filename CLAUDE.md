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
uv run pytest tests/unit/engine/test_ocr.py::TestExtractNumberAndBaseSetSize::test_simple
uv run pytest -k "number_and_base" -x -vv

# Sanity check imports
uv run python -c "import mtg_recognition; print('ok')"
uv run python -m compileall src tests

# Pipeline CLI (recognition on one image or a directory)
uv run python -m mtg_recognition.engine.pipeline -i card.jpg
uv run python -m mtg_recognition.engine.pipeline -i benchmarks/dataset/images/raw -k 3 --k-phash 50
uv run python -m mtg_recognition.engine.pipeline -i data/test/images -o output/results.json
uv run python -m mtg_recognition.engine.pipeline --help

# Segmentation CLI
uv run python -m mtg_recognition.engine.segmentation.cli -i data/test/images -o output/seg --mode auto
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

`ocr_utils.py` contains all text-extraction helpers: `extract_setcode_candidates`, `extract_number_and_base_set_size`, `extract_artist_name`, `extract_set_release_year`, `extract_power_toughness`. Artist extraction joins all info-ROI tokens into one string, splits on whitespace, and keeps only word-like tokens (letters/hyphens/apostrophes, no digits or symbols), discarding noise such as set-code artifacts (`TDM·DE`) and copyright boilerplate. `Properties.artist` is `str | None`; all other scalar fields are also `str | None` except `setcode_candidates: list[str]`.

`extract_number_and_base_set_size` returns `(collector_number, base_set_size)` — both `str | None`. Internally it uses a two-stage private extractor (`_extract_number_field`):
- **Stage 1** — full `cardNumber/baseSetSize` pattern (e.g. `"011/264"`, `"100a/302"`, `"T001/T030"`). Supports optional lowercase/star suffix (`a`, `★`, `†`, `✦`) and symmetric `T`/`C` prefix (token/commander). Four passes: per-token with boundary → 2-token concat → 3-token concat → per-token without boundary (embedded cases). Longest match wins.
- **Stage 2** — standalone `cardNumber` only (old sets like CE/IE that print no set size). Rejects 1-digit values, 2-digit values without a leading zero, and 4-digit values in the copyright year range `[1993, current_year+1]`. Ranks by: leading-zero first → reading-order position → length. Also handles OCR noise where a single uppercase letter is glued to the digits (e.g. `"L0398"` from a basic-land `"L 398"` marker) via a secondary `(?<!\w)[A-Z](\d{1,4})\b` pass.

`extract_power_toughness` matches both numeric P/T (`"3/3"`) and star-based P/T (`"*/*"`, `"2/*"`, `"*/3"`). Uses `(?<!\w)(\*|\d)/(\*|\d)(?!\w)` so the `*` non-word character is handled correctly alongside digits.

`ocr.py` exposes a `debug=False` flag on `extract_properties` and `find_best_matches`. When `True`, both return an `OCRDebugData` frozen dataclass containing the annotated info-ROI image (PaddleOCR bounding boxes), raw OCR texts/scores, extracted `Properties`, and a per-attribute score matrix for the top candidates.

`pipeline.py` exposes a single `Pipeline.find_best_matches` method returning a 4-tuple `(matches, scores, ocr_debug, diag)`. `diag` is a `PipelineDiagnostics` dataclass (four timing floats: `segmentation_ms`, `phash_ms`, `ocr_ms`, `total_ms`) — always populated, near-zero overhead. `ocr_debug` is `OCRDebugData | None`, populated only when `debug=True`. These two are intentionally separate: timing is performance monitoring, `OCRDebugData` is OCR inspection data. A `_timer(diag, attr)` context manager handles per-stage measurement without boilerplate.

`pipeline.py` also exposes a `@click.command()` CLI (`python -m mtg_recognition.engine.pipeline`). Options: `-i` (required, file or dir), `-o` (optional output file path for JSON results), `--k-phash` (pHash candidates, default 20), `-k` (top-k final results, default 1). Results are printed incrementally; when `-o` is given all results are written as a JSON array to that single file after processing completes. If an image filename is a valid UUID, the CLI compares it against the top match and reports `✓ MATCH` / `✗ MISS` / `—`. The CLI always runs with `debug=True` so `raw_texts` and `raw_scores` are included in both terminal and JSON output.

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
| Pipeline (library + CLI) | `src/mtg_recognition/engine/pipeline.py` |
| OCR logic + fuzzy ranking | `src/mtg_recognition/engine/ocr/ocr.py` |
| OCR parsing helpers | `src/mtg_recognition/engine/ocr/ocr_utils.py` |
| Segmentation CLI | `src/mtg_recognition/engine/segmentation/cli.py` |
| Segmentation (YOLO primary) | `src/mtg_recognition/engine/segmentation/yolo.py` |
| Segmentation (shared helpers) | `src/mtg_recognition/engine/segmentation/seg_utils.py` |
| pHash matcher | `src/mtg_recognition/engine/matching/phash_matcher.py` |
| Database job | `src/mtg_recognition/jobs/build_database.py` |
