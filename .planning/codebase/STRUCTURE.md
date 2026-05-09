# Codebase Structure

**Analysis Date:** 2026-05-09

## Directory Layout

```
MTG-Recognition/
├── src/
│   └── mtg_recognition/        # All application source code
│       ├── app/                # Streamlit UI and batch processing
│       ├── core/               # Shared utilities (image loading)
│       ├── engine/             # Recognition engine
│       │   ├── matching/       # pHash index and scoring
│       │   ├── ocr/            # PaddleOCR extraction and parsing
│       │   └── segmentation/   # Card detection and perspective warp
│       ├── jobs/               # Database schema and build scripts
│       ├── database.py         # SQLite engine singleton
│       └── settings.py         # Pydantic-settings configuration
├── tests/
│   ├── unit/                   # Unit tests (fast, no models)
│   │   ├── engine/
│   │   │   ├── segmentation/   # Tests per segmentor
│   │   │   └── test_ocr.py     # OCR extraction unit tests
│   │   └── fixtures/images/    # Small test images
│   └── regression/
│       └── segmentation/       # Image-level regression tests with fixtures
│           └── fixtures/       # Categorised test images + expected masks
├── benchmarks/                 # Benchmark datasets and timing results
│   └── dataset/images/         # raw/ (phone photos) and ready/ (warped)
├── data/
│   ├── reference/              # Runtime reference data (not in git)
│   │   ├── MobileCache.sqlite  # 35 MB card metadata database
│   │   ├── images/             # ~1M card images indexed by UUID
│   │   └── index/phash/        # hashes.npy + index_to_uuid.json
│   └── test/                   # Curated manual test images by category
├── scripts/                    # Utility shell scripts
├── scratch/                    # Exploratory/throwaway scripts (not tests)
├── output/                     # Segmentation CLI output (gitignored)
├── Dockerfile                  # Production container
├── docker-compose.yml          # Compose file for local dev
├── docker-entrypoint.sh        # Container startup script
├── pyproject.toml              # Project metadata and dependencies (uv)
└── uv.lock                     # Locked dependency versions
```

## Directory Purposes

**`src/mtg_recognition/app/`:**
- Purpose: Streamlit web application
- Contains: `app.py` (main page), `batch_processor.py` (batch orchestration)
- Key files: `app.py`, `batch_processor.py`

**`src/mtg_recognition/core/`:**
- Purpose: Domain-agnostic utilities shared by all layers
- Contains: `image_utils.py` — universal image loading from file paths, bytes, PIL, BytesIO, or numpy arrays

**`src/mtg_recognition/engine/pipeline.py`:**
- Purpose: Top-level orchestrator wiring segmentation + pHash + OCR; also the CLI entry point

**`src/mtg_recognition/engine/segmentation/`:**
- Purpose: All card detection and perspective warp logic
- Contains: `yolo.py` (primary, three-path), `box.py` (CV fallback), `rembg.py` (last-resort), `seg_utils.py` (shared geometry + dataclasses), `geometry.py` (low-level math), `cli.py` (segmentation-only CLI)

**`src/mtg_recognition/engine/matching/`:**
- Purpose: pHash-based coarse candidate retrieval
- Contains: `phash_matcher.py` (256-bit Hamming index), `deep_matcher.py` (unused/commented deep feature matcher)

**`src/mtg_recognition/engine/ocr/`:**
- Purpose: PaddleOCR inference and structured text parsing
- Contains: `ocr.py` (CardOCR class, Properties, OCRDebugData, ATTRIBUTE_MAPPINGS), `ocr_extraction.py` (pure parsing functions), `enhance.py` (preprocessing), `models/prod/` (bundled OCR model weights)

**`src/mtg_recognition/jobs/`:**
- Purpose: One-time data build tooling
- Contains: `schemas.py` (SQLModel ORM definitions), `build_database.py` (ingests MTGJson/Scryfall into MobileCache.sqlite)

**`tests/unit/`:**
- Purpose: Fast unit tests that do not load models or reference data
- Contains: Tests for OCR extraction helpers, segmentation geometry, bypass logic, CV fallback
- Key files: `tests/unit/engine/test_ocr.py`, `tests/unit/engine/segmentation/test_seg_utils.py`

**`tests/regression/segmentation/`:**
- Purpose: Image-level regression tests comparing YOLO and rembg output against stored fixture masks
- Contains: `test_yolo_reg.py`, `test_rembg_reg.py`, fixture images and masks organised by category (1_photos_single_card, 2_photos_multiple_cards, etc.)

**`data/reference/`:**
- Purpose: Runtime reference data required by the pipeline; not committed to git
- Generated: Yes (by `build_database.py` and `PHashCardMatcher.build_index()`)
- Committed: No

**`src/mtg_recognition/engine/segmentation/models/prod/`:**
- Purpose: Bundled YOLO model weights and rembg u2net model
- Generated: No (manually downloaded or copied into the image)
- Committed: No (excluded via .gitignore, bundled in Docker image)

**`src/mtg_recognition/engine/ocr/models/prod/`:**
- Purpose: Bundled PaddleOCR detection and recognition model weights
- Generated: No
- Committed: No (bundled in Docker image)

## Key File Locations

**Entry Points:**
- `src/mtg_recognition/app/app.py`: Streamlit web app
- `src/mtg_recognition/engine/pipeline.py`: Pipeline library + `python -m` CLI
- `src/mtg_recognition/engine/segmentation/cli.py`: Segmentation-only CLI

**Configuration:**
- `src/mtg_recognition/settings.py`: All runtime settings (Pydantic-settings, reads `.env`)
- `pyproject.toml`: Package metadata, dependencies, tool config
- `docker-compose.yml`: Container orchestration for local dev

**Core Logic:**
- `src/mtg_recognition/engine/segmentation/yolo.py`: Primary segmentor with three-path logic
- `src/mtg_recognition/engine/segmentation/seg_utils.py`: Shared constants (`CARD_ASPECT_RATIO`, `WARPED_MIN_SIZE`, `WARPED_MAX_SIZE`) and dataclasses
- `src/mtg_recognition/engine/matching/phash_matcher.py`: pHash index build and search
- `src/mtg_recognition/engine/ocr/ocr.py`: PaddleOCR inference + fuzzy scoring; `ATTRIBUTE_MAPPINGS` config
- `src/mtg_recognition/engine/ocr/ocr_extraction.py`: Pure text parsing helpers
- `src/mtg_recognition/database.py`: Shared SQLAlchemy engine

**Testing:**
- `tests/unit/engine/test_ocr.py`: OCR extraction unit tests
- `tests/unit/engine/segmentation/`: Per-module segmentation unit tests
- `tests/regression/segmentation/`: YOLO and rembg regression tests against fixture images

## Naming Conventions

**Files:**
- Snake case for all Python modules: `ocr_extraction.py`, `batch_processor.py`, `seg_utils.py`
- Test files: `test_<module>.py` pattern, e.g., `test_ocr.py`, `test_seg_utils.py`
- Regression fixtures: UUID filenames for card images, numeric prefix for category directories

**Directories:**
- Snake case: `segmentation/`, `matching/`, `batch_processor`
- No barrel `__init__.py` re-exports; imports always reference the specific module

**Classes:**
- PascalCase: `YoloCardSegmentor`, `PHashCardMatcher`, `CardOCR`, `BatchProcessor`, `Pipeline`

**Constants:**
- SCREAMING_SNAKE_CASE module-level: `CARD_ASPECT_RATIO`, `WARPED_MIN_SIZE`, `ATTRIBUTE_MAPPINGS`, `K_PHASH_CANDIDATES`

## Where to Add New Code

**New segmentor:**
- Implementation: `src/mtg_recognition/engine/segmentation/<name>.py`
- Return type: `list[CardSegment]` (use `CardSegment` and `warp_card` from `seg_utils.py`)
- Wire it in: `YoloCardSegmentor._fallback_segment()` in `yolo.py` or as a new path in `segment_batch()`
- Tests: `tests/unit/engine/segmentation/test_<name>.py`

**New OCR extraction helper:**
- Implementation: Add a pure function to `src/mtg_recognition/engine/ocr/ocr_extraction.py`
- Call it in: `CardOCR.extract_properties()` in `ocr.py` and add the field to `Properties` NamedTuple
- Add fuzzy scoring: extend `ATTRIBUTE_MAPPINGS` list in `ocr.py`
- Tests: Add test class/method to `tests/unit/engine/test_ocr.py`

**New matching strategy:**
- Implementation: `src/mtg_recognition/engine/matching/<name>_matcher.py`
- Interface: `find_best_matches(query_img, candidates, k) -> tuple[list[Row], list[float]]`
- Wire it in: `Pipeline._match_one()` in `pipeline.py`

**New app feature (UI):**
- Add to `src/mtg_recognition/app/app.py` or a new file under `src/mtg_recognition/app/`
- Shared pipeline parameters (`K_PHASH_CANDIDATES`, `K_TOP_MATCHES`) live in `batch_processor.py`; import from there

**New configuration setting:**
- Add field to `Settings` class in `src/mtg_recognition/settings.py`
- Access via `from mtg_recognition.settings import settings` anywhere in the codebase

**New utility function (image handling):**
- Add to `src/mtg_recognition/core/image_utils.py` if domain-agnostic
- Add to the relevant engine module if domain-specific

## Special Directories

**`data/reference/`:**
- Purpose: Runtime reference database, pHash index, and 1M+ card images
- Generated: Yes — `build_database.py` and `PHashCardMatcher.build_index()`
- Committed: No

**`benchmarks/`:**
- Purpose: Benchmark datasets and pipeline timing results; not used by tests
- Generated: Partially (timing results generated; fixture images manually curated)
- Committed: Yes (images committed as benchmark fixtures)

**`scratch/`:**
- Purpose: Exploratory scripts; not part of the test suite or production code
- Generated: No
- Committed: Yes (informally)

**`output/`:**
- Purpose: Output from the segmentation CLI (`--output` flag)
- Generated: Yes
- Committed: No (gitignored)

---

*Structure analysis: 2026-05-09*
