## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues on `decksray/MTG-Recognition` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Project

MTG card recognition app: takes a photo of a Magic: The Gathering card, segments it, extracts text via OCR, and matches it against a ~1M-card SQLite database. Served as a Streamlit web app via Docker.

## Git

- `dev` is the integration branch; `main` is the release branch and the GitHub default. Branch off `dev` and open PRs against `dev`: always pass `gh pr create --base dev`. Worktrees default to `main`, so base them on `dev` explicitly.
- Feature PRs are squash-merged into `dev`. Releases are `dev` → `main` PRs merged with a merge commit, never squashed, then tagged (e.g. `v0.2.0`).
- Never commit to `main` directly; fixes go through `dev` and a release.
- Never credit Claude or Claude Code as a contributor: no `Co-Authored-By` trailer in commits, no "Generated with Claude Code" line in PR bodies, issues or comments.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager

## Commands

```bash
# Install dependencies (includes DVC)
uv sync --dev

# Pull test/benchmark dataset from Google Drive (required before running tests or pipeline on test images)
uv run dvc pull

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
uv run python -m mtg_recognition.engine.pipeline -i data/test/6_benchmark/images/raw -k 3 --k-phash 50
uv run python -m mtg_recognition.engine.pipeline -i data/test/images -o output/results.json
uv run python -m mtg_recognition.engine.pipeline --help

# Segmentation CLI
uv run python -m mtg_recognition.engine.segmentation.cli -i data/test/images -o output/seg --mode auto

# mtg CLI (registered console script — available after activating the venv)
mtg --help
mtg sync                      # download missing reference images (pinned to committed DB)
mtg sync --update-db          # fetch latest MTGJSON release and rebuild all artifacts
mtg sync --force              # unconditional full rebuild
mtg build-index               # rebuild pHash index from existing reference images
mtg build-index --force --workers 8
mtg recognize -i card.jpg
mtg segment -i card.jpg --mode auto
```

Linting and formatting use [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml` (line length 120, Python 3.12, rules `E`, `F`, `I`, `UP`, `B`, `SIM`):

```bash
uv run ruff check src tests          # lint
uv run ruff check --fix src tests    # lint + autofix
uv run ruff format src tests         # format
```

## Dataset (DVC)

Test and benchmark images are tracked by DVC with a Google Drive remote. They are **not in git**. After cloning, run `uv run dvc pull` to populate `data/test/`.

First-time setup requires a personal GCP OAuth Desktop app credential — see `CONTRIBUTING.md` for the one-time steps. Once configured:

```bash
# One-time secret setup (writes to .dvc/config.local, which is gitignored)
uv run dvc remote modify --local origin gdrive_client_id '<your-client-id>'
uv run dvc remote modify --local origin gdrive_client_secret '<your-client-secret>'

# Pull dataset (~600 MB, triggers browser OAuth on first run)
uv run dvc pull

# After adding new test images: re-track and push
uv run dvc add data/test
uv run dvc push
git add data/test.dvc && git commit -m "data: add ..."
```

The DVC remote is `gdrive://root/decksray/MTG-Recognition/data/test` (named `origin`, set as default).

## Architecture

`Pipeline` (`engine/pipeline.py`) is the engine's single recognition module and runs a two-stage pipeline, one image at a time (no batch inference, see `docs/adr/0002-recognise-one-image-at-a-time.md`):

**Stage 1 — Segmentation** (`engine/segmentation/`):
Three-path pipeline per image, tried in order:
1. **Bypass** (`yolo.py`): if the image aspect ratio ≈ MTG card ratio (63/88 ± 3 %) and the border is dark → resize only. Handles pre-cropped scans without running any model.
2. **YOLO** (`yolo.py`): `yoloe-11s-seg` inference on the single image (`YoloCardSegmentor.segment(img)` is the only entry point). Each YOLO polygon mask is converted to 4 corners via `corners_from_binary_mask` → perspective warp.
3. **Fallback** (`box.py` then `rembg.py`): classical Otsu CV detector first (fast, good for uniform scanner beds); rembg background removal last resort (slow, works on photos with clear backgrounds).

Warped output size is clamped between `WARPED_MIN_SIZE = (488, 680)` and `WARPED_MAX_SIZE = (976, 1360)` (width × height in pixels) so that close-up phone photos don't produce unbounded images while scans aren't unnecessarily downscaled.

Shared geometry helpers (`seg_utils.py`): `CARD_ASPECT_RATIO`, `WARPED_MIN_SIZE`, `WARPED_MAX_SIZE`, `clamp_size`, `warp_card`, `valid_aspect_ratio`, `corners_from_binary_mask`, and the `CardSegment` / `ImageSegmentationResult` dataclasses. All segmentors import from here to avoid circular dependencies.

**Stage 2 — Matching** (`engine/matching/`, `engine/ocr/`):
1. **PHash matcher**: Perceptual hash (256-bit, 16×16) compared against precomputed numpy index to get top-k candidates fast. Returns UUIDs + scores only; it does not touch the database. `Pipeline` resolves the UUIDs to `Printing`s through the card catalog.
2. **OCR fuzzy matcher**: PaddleOCR extracts text from the card's top ~11% (name ROI) and bottom ~13% (info ROI: set code, collector number, artist). `rapidfuzz` scores OCR attributes against candidates with weights (name = 2.0, set code = 1.0, etc.) to select the best match. Each entry in `ATTRIBUTE_MAPPINGS` carries an optional `preprocessor` — `rapidfuzz.utils.default_process` (lowercase + strip) is applied for `name` and `artist`; `None` for all other attributes. The predict call is wrapped in `try/except` — PaddlePaddle's C++ backend throws intermittent `RuntimeError: std::exception` on some inputs; catching it degrades to zero-vector scoring so the phash ranking holds.

`ocr_extraction.py` contains all text-extraction helpers: `filter_flavor_text_zone`, `extract_setcode_candidates`, `extract_number_and_base_set_size`, `extract_artist_name`, `extract_set_release_year`, `extract_power_toughness`. Artist extraction joins all info-ROI tokens into one string, splits on whitespace, and keeps only word-like tokens (letters/hyphens/apostrophes, no digits or symbols), discarding noise such as set-code artifacts (`TDM·DE`) and copyright boilerplate. `Properties.artist` is `str | None`; all other scalar fields are also `str | None` except `setcode_candidates: list[str]`.

`filter_flavor_text_zone` is called in `ocr.py` immediately after PaddleOCR inference, before any property extraction. It discards tokens where `y1 < 0.08 * roi_height AND x2 < 0.90 * roi_width` — flavor text bleeds into the TOP of the info ROI, while P/T sits in the far-RIGHT column and is preserved. The threshold is intentionally conservative (8 % of ROI height) to avoid clipping artist names that appear near the top of the info ROI. A safety guard returns the originals unchanged if all tokens would be discarded.

`extract_setcode_candidates` searches for language-code artifacts (separator chars `·`, `.`, `+`, `-`, `=`, `★`, `*`, `,` — the last two handle OCR variants of `★` and `·`) and returns the validated prefix. When no artifact is found, falls back to tokens matching `[A-Z]{2,5}` that are not in `_SETCODE_BLOCKLIST` (copyright boilerplate like `COAST`, `INC`, `TM`). Returns `[]` rather than noisy tokens when nothing survives — feeding rapidfuzz with irrelevant strings would only add noise.

`extract_number_and_base_set_size` returns `(collector_number, base_set_size)` — both `str | None`. Internally it uses a two-stage private extractor (`_extract_number_field`):
- **Stage 1** — full `cardNumber/baseSetSize` pattern (e.g. `"011/264"`, `"100a/302"`, `"T001/T030"`). Supports optional lowercase/star suffix (`a`, `★`, `†`, `✦`) and symmetric `T`/`C` prefix (token/commander). Four passes: per-token with boundary → 2-token concat → 3-token concat → per-token without boundary (embedded cases). Longest match wins.
- **Stage 2** — standalone `cardNumber` only (old sets that print no set size). Accepts 2-digit values without a leading zero (e.g. `"71"` from Burnished Hart retro-frame). Rejects 1-digit values and 4-digit values in the copyright year range `[1993, current_year+1]`. **Slash guard**: tokens containing `/`, `\`, or `|` are skipped entirely so P/T sub-matches (e.g. `"10"` from `"10/4"`) are never mistaken for a card number. Ranks by: leading-zero first → reading-order position → length. Also handles OCR noise where a single uppercase letter is glued to the digits (e.g. `"L0398"`) via a secondary `(?<!\w)[A-Z](\d{1,4})\b` pass.

`extract_power_toughness` matches both numeric P/T (`"3/3"`) and star-based P/T (`"*/*"`, `"2/*"`, `"*/3"`). Uses `(?<!\w)(\*|\d)/(\*|\d)(?!\w)` so the `*` non-word character is handled correctly alongside digits.

`ocr.py` exposes a `debug=False` flag on `extract_properties` and `find_best_matches`. When `True`, both return an `OCRDebugData` frozen dataclass containing the annotated info-ROI image (PaddleOCR bounding boxes), raw OCR texts/scores, extracted `Properties`, and a per-attribute score matrix for the top candidates.

`Pipeline(k_phash_candidates=20, k_final_results=1, catalog=None, segmentor=None, phash_matcher=None, ocr=None, auto_build=True)` types its stages with the `Protocol`s `CardSegmentor` (`segment`), `CandidateMatcher` (`find_best_matches`, `build_index`, `index_loaded`) and `CandidateReranker` (`find_best_matches`), which `YoloCardSegmentor`, `PHashCardMatcher` and `CardOCR` satisfy structurally; the test fakes in `tests/unit/engine/fakes.py` are type-checked against them (`[tool.ty.src] include`). It builds its defaults inside the constructor and raises `ValueError` on bad k or a missing pHash index. `Pipeline.recognize(image, debug=False) -> RecognitionResult` is the only recognition method: it downscales the decoded BGR image to `settings.MAX_IMAGE_DIMENSION` (clients decode and apply EXIF with `image_utils.load_image`), segments, finds pHash candidates, resolves them through the catalog and reranks with OCR. `RecognitionResult(segments, error)` holds one `SegmentResult(warped_image, ranked: list[RankedPrinting], ocr_debug)` per card found (empty = no card); per-image failures are captured in `error`, never raised. `ocr_debug` is `OCRDebugData | None`, populated only when `debug=True`.

**Printings and the card catalog** (`catalog.py`, next to `database.py`): `Printing` is a frozen dataclass (`uuid, name, set_code, number, base_set_size, artist, set_release_year, power, toughness, image_url`) used everywhere the engine and app handle cards; no `sqlalchemy` `Row`s cross module boundaries. A `CardCatalog` resolves UUIDs to `Printing`s: `SqliteCardCatalog` reads `MobileCache`, `InMemoryCardCatalog` is for tests. `RankedPrinting(printing, score)` (frozen, also in `catalog.py`) pairs a candidate with the score that ranked it; ranked results are always `list[RankedPrinting]`, best first, never parallel printing/score lists. `CardOCR.find_best_matches` takes `list[Printing]` and returns `(list[RankedPrinting], OCRDebugData | None)`; `ATTRIBUTE_MAPPINGS`' `db_attr` is the `Printing` field name.

`pipeline.py` also exposes a `@click.command()` CLI (`python -m mtg_recognition.engine.pipeline`). Options: `-i` (required, file or dir), `-o` (optional output file path), `--k-phash` (pHash candidates, default 20), `-k` (top-k final results, default 1). Results are printed incrementally per segment. When `-o` is given all results are written as a JSON array `[{"image": ..., "segments": [{"segment_idx", "success", "ranked": [...], "debug": {"raw_texts", "raw_scores", "raw_boxes"}}, ...]}, ...]` after processing completes. If an image filename is a valid UUID, each segment result carries a `success` flag comparing `ranked[0].printing.uuid` against the filename stem (`true`/`false`/`null`).

**App layer** (`app/`):
- `app.py` — Streamlit UI for multi-image uploads (`display_upload_detail_view`, session state `upload_results`). Sidebar toggle `"🔬 Debug Mode"` enables the `OCRDebugData` expanders in the detail view (annotated ROI, raw OCR table, extracted vs. best-match table, score matrix). The detail view reads all data directly from `UploadResult` — it does **not** re-run the pipeline. Builds `Pipeline(k_phash_candidates=50, k_final_results=5)`; the "Top N matches" view reads k from the pipeline.
- `upload_processor.py` — `UploadProcessor` loops over uploaded paths, decodes them, calls `Pipeline.recognize` per image and writes temp images; also exports CSV/Excel/JSON. Returns `UploadResult`s (one per card found, or one per image with no card or an error) carrying the full top-K `ranked: list[RankedPrinting]`, `ocr_debug` and temp image paths so the detail view is self-contained; `best` is the top-ranked entry or `None`, and `status` is an `UploadStatus` `StrEnum` (`SUCCESS`, `NO_MATCH`, `ERROR`; exported as `"success"`/`"no_match"`/`"error"`) derived from the result's fields. The CSV/Excel and JSON exporters share one field map, `_BEST_PRINTING_FIELDS`. Always recognizes with `debug=True` so `OCRDebugData` is available on demand without a second pipeline call.

**Data jobs** (`jobs/`):
- `sync.py` — manages the full reference data stack: builds MobileCache, downloads reference images, and builds the pHash index. Exposes `mtg sync` as the CLI entry point. Three modes: default (images-only, DB pinned to git), `--update-db` (fetches remote MTGJSON version, rebuilds if changed), `--force` (unconditional full rebuild). State file, MobileCache, and pHash index are git-tracked; only `data/reference/images/` is not.
- `schemas.py` — SQLModel ORM models (`Cards`, `CardIdentifiers`, `Sets`).

**Database** (`database.py`):
Read-only SQLite opened with `mode=ro&nolock=1`. 35 MB, ~1M+ cards.

**CLI architecture** (`cli.py`):
Thin registration file — no business logic. Each subcommand is defined in its domain module as a `@click.command()` and registered via `cli.add_command()`. To add a new subcommand: define the Click function in the relevant domain module, then add one `cli.add_command()` line to `cli.py`.

## Data & Assets

**DVC-tracked** (restored via `dvc pull`, not in git):
- `data/test/` — test and benchmark images across 8 categories (0–6 + 6_benchmark); ~971 files, ~600 MB
  - `data/test/6_benchmark/images/raw/` — 36 raw benchmark JPGs (UUID filenames)
  - `data/test/6_benchmark/images/ready/` — 36 pre-warped benchmark JPGs
  - `data/test/6_benchmark/ocr/` — 20 OCR ground-truth images
- `data/test.dvc` — the git-tracked DVC pointer (committed; tells `dvc pull` what to fetch)

**Not git-tracked** (generated by `mtg sync`):
- `data/reference/images/` — ~1M card images indexed by UUID

**git-tracked**
- `data/reference/MobileCache.sqlite` — card metadata database (~35 MB, ~1M+ cards)
- `data/reference/index/phash/hashes.npy` + `index_to_uuid.json` — prebuilt pHash index
- YOLO `.pt` and rembg `.onnx` models: `src/mtg_recognition/engine/segmentation/models/prod/`
- OCR models: `src/mtg_recognition/engine/ocr/models/prod/`

## Key Entry Points

| Purpose | Path |
|---------|------|
| Streamlit app | `src/mtg_recognition/app/app.py` |
| Pipeline (library + CLI) | `src/mtg_recognition/engine/pipeline.py` |
| OCR logic + fuzzy ranking | `src/mtg_recognition/engine/ocr/ocr.py` |
| OCR parsing helpers | `src/mtg_recognition/engine/ocr/ocr_extraction.py` |
| Segmentation CLI | `src/mtg_recognition/engine/segmentation/cli.py` |
| Segmentation (YOLO primary) | `src/mtg_recognition/engine/segmentation/yolo.py` |
| Segmentation (shared helpers) | `src/mtg_recognition/engine/segmentation/seg_utils.py` |
| pHash matcher | `src/mtg_recognition/engine/matching/phash_matcher.py` |
| Reference data sync job | `src/mtg_recognition/jobs/sync.py` |


