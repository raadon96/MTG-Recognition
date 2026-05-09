<!-- refreshed: 2026-05-09 -->
# Architecture

**Analysis Date:** 2026-05-09

## System Overview

```text
┌──────────────────────────────────────────────────────────────────┐
│                   App Layer (Streamlit UI)                        │
│   src/mtg_recognition/app/app.py                                 │
│   src/mtg_recognition/app/batch_processor.py                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │  Pipeline.find_best_matches()
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Pipeline Orchestrator                           │
│   src/mtg_recognition/engine/pipeline.py                         │
│   Also exposed as CLI: python -m mtg_recognition.engine.pipeline │
└──────┬────────────────────────────┬──────────────────────────────┘
       │ Stage 1                    │ Stage 2
       ▼                            ▼
┌──────────────────┐   ┌────────────────────────────────────────┐
│  Segmentation    │   │  Matching                              │
│                  │   │                                        │
│  yolo.py         │   │  phash_matcher.py  ->  ocr/ocr.py     │
│  box.py          │   │  (coarse)             (refinement)     │
│  rembg.py        │   │                                        │
│  seg_utils.py    │   │  ocr/ocr_extraction.py                 │
│  geometry.py     │   │  ocr/enhance.py                        │
└──────────────────┘   └────────────────────────────────────────┘
       │                            │
       ▼                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  Reference Data (not in git)                                     │
│  data/reference/MobileCache.sqlite  (35 MB, ~1M cards)          │
│  data/reference/index/phash/hashes.npy + index_to_uuid.json     │
│  data/reference/images/  (~1M card images by UUID)              │
└──────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| App | Streamlit UI, batch upload, detail view, export | `src/mtg_recognition/app/app.py` |
| BatchProcessor | Per-image pipeline orchestration, mini-batch loading | `src/mtg_recognition/app/batch_processor.py` |
| Pipeline | End-to-end orchestrator: segment -> pHash -> OCR | `src/mtg_recognition/engine/pipeline.py` |
| YoloCardSegmentor | Three-path segmentor (bypass / YOLO / fallback) | `src/mtg_recognition/engine/segmentation/yolo.py` |
| seg_utils | Shared geometry constants and dataclasses | `src/mtg_recognition/engine/segmentation/seg_utils.py` |
| box | CV fallback: Otsu threshold -> contour -> corners | `src/mtg_recognition/engine/segmentation/box.py` |
| rembg | Last-resort segmentor using background removal | `src/mtg_recognition/engine/segmentation/rembg.py` |
| geometry | Low-level perspective transform math | `src/mtg_recognition/engine/segmentation/geometry.py` |
| PHashCardMatcher | 256-bit pHash index, Hamming distance search | `src/mtg_recognition/engine/matching/phash_matcher.py` |
| CardOCR | PaddleOCR extraction + rapidfuzz candidate scoring | `src/mtg_recognition/engine/ocr/ocr.py` |
| ocr_extraction | Pure-function text parsers (set code, number, artist) | `src/mtg_recognition/engine/ocr/ocr_extraction.py` |
| enhance | OCR preprocessing (upscale, contrast) | `src/mtg_recognition/engine/ocr/enhance.py` |
| image_utils | Universal image loading / resizing | `src/mtg_recognition/core/image_utils.py` |
| database | Read-only SQLite engine singleton | `src/mtg_recognition/database.py` |
| settings | Pydantic-settings config, derived paths, env push | `src/mtg_recognition/settings.py` |
| schemas | SQLModel ORM models (MobileCache, Cards, Sets) | `src/mtg_recognition/jobs/schemas.py` |
| build_database | One-time data ingestion from MTGJson/Scryfall | `src/mtg_recognition/jobs/build_database.py` |

## Pattern Overview

**Overall:** Two-stage coarse-to-fine retrieval pipeline behind a Streamlit frontend.

**Key Characteristics:**
- Segmentation is fully decoupled from matching; `Pipeline` wires them together and each subsystem is independently testable
- Frozen dataclasses serve as immutable result containers throughout: `SegmentResult`, `CardSegment`, `ImageSegmentationResult`, `OCRDebugData`, `Properties`
- Debug path is opt-in (`debug=True`) and propagates through all layers without affecting the production code path
- Stateless pure functions for all OCR text parsing (`ocr_extraction.py`); stateful objects only for model handles and the pHash index

## Layers

**App Layer:**
- Purpose: User-facing UI and batch processing orchestration
- Location: `src/mtg_recognition/app/`
- Contains: Streamlit page code, batch loop, export helpers, UI state management
- Depends on: `Pipeline`, `database.py`, `settings.py`, `core/image_utils.py`
- Used by: End users via browser

**Engine Layer:**
- Purpose: All card recognition logic
- Location: `src/mtg_recognition/engine/`
- Contains: `pipeline.py` (orchestrator), `segmentation/`, `matching/`, `ocr/`
- Depends on: `core/`, `database.py`, `settings.py`, `jobs/schemas.py`
- Used by: App layer and CLI entry point

**Core Layer:**
- Purpose: Shared utilities with no domain knowledge
- Location: `src/mtg_recognition/core/`
- Contains: `image_utils.py` — load and resize from any source type (`ImageSource` union)
- Depends on: `settings.py` only
- Used by: All other layers

**Data / Jobs Layer:**
- Purpose: Database schema definitions and one-time build tooling
- Location: `src/mtg_recognition/jobs/`
- Contains: `schemas.py` (SQLModel models), `build_database.py`
- Depends on: External data sources (MTGJson, Scryfall)
- Used by: Matching layer (queries `MobileCache`), data build scripts

## Data Flow

### Primary Request Path (Streamlit batch upload)

1. User uploads files -> `app.py:main()` saves bytes to `tempfile` directories
2. `BatchProcessor.process_batch()` loads images in mini-batches of `BATCH_SIZE=20` (`batch_processor.py:92`)
3. `YoloCardSegmentor.segment_batch()` runs three-path segmentation (`yolo.py:156`):
   - Path 1 Bypass: aspect ratio ~= card ratio + dark border -> `_bypass_warp()` (no model)
   - Path 2 YOLO: `yoloe-11s-seg` batch inference, polygon -> `corners_from_binary_mask()` -> `warp_card()`
   - Path 3 Fallback: `detect_cards_cv()` (Otsu) first, then `RembgCardSegmentor.segment()` last resort
4. `Pipeline._match_one()` called per warped segment (`pipeline.py:102`):
   - Stage 1: `PHashCardMatcher.find_best_matches()` -> top-20 candidates by Hamming distance
   - Stage 2: `CardOCR.find_best_matches()` -> PaddleOCR on name ROI (top 4-11%) + info ROI (bottom 88-100%) -> `Properties` NamedTuple -> rapidfuzz weighted scoring against candidates
5. `BatchResult` dataclasses assembled and stored in `st.session_state.batch_results`

### CLI Path

1. `python -m mtg_recognition.engine.pipeline -i <path>` invokes `main()` click command (`pipeline.py:152`)
2. `Pipeline.find_best_matches()` called with `apply_segmentation=True`
3. Results printed incrementally to terminal; optionally written to JSON via `-o`

**State Management:**
- No global mutable state in engine or core layers
- `PHashCardMatcher` holds the loaded numpy hash array and UUID mapping as instance attributes
- `CardOCR` holds the PaddleOCR model instance
- Streamlit session state (`st.session_state`) holds `batch_results` and UI selection indices across reruns
- `database.py` exposes a module-level `engine` singleton (read-only SQLite connection, opened once at import)

## Key Abstractions

**SegmentResult (frozen dataclass):**
- Purpose: Result bundle for one detected card segment after full pipeline
- Location: `src/mtg_recognition/engine/pipeline.py:20`
- Pattern: `(matches: list[Row], scores: list[float], ocr_debug: OCRDebugData | None)`

**ImageSegmentationResult / CardSegment (dataclasses):**
- Purpose: Carry one or more perspective-corrected card images from the segmentor
- Location: `src/mtg_recognition/engine/segmentation/seg_utils.py:33`
- Pattern: `ImageSegmentationResult.segments: list[CardSegment]`; `CardSegment.warped_image: np.ndarray (BGR)`

**Properties (NamedTuple):**
- Purpose: Structured OCR extraction result passed to the fuzzy scorer
- Location: `src/mtg_recognition/engine/ocr/ocr.py:96`
- Pattern: All fields are `str | None` except `setcode_candidates: list[str]`

**OCRDebugData (frozen dataclass):**
- Purpose: Diagnostics payload for mismatch investigation; only populated when `debug=True`
- Location: `src/mtg_recognition/engine/ocr/ocr.py:109`
- Pattern: Carries annotated ROI image, raw texts/scores/boxes, `Properties`, and `score_matrix: pd.DataFrame | None`

**MobileCache (SQLModel table):**
- Purpose: Denormalized card row queried by both pHash and OCR matchers via UUID
- Location: `src/mtg_recognition/jobs/schemas.py:160`
- Key fields: `uuid`, `name`, `setCode`, `number`, `baseSetSize`, `setReleaseYear`, `artist`, `url`, `power`, `toughness`

**ATTRIBUTE_MAPPINGS (list[dict]):**
- Purpose: Declarative config driving OCR-to-DB fuzzy scoring weights and preprocessors
- Location: `src/mtg_recognition/engine/ocr/ocr.py:23`
- Pattern: Each entry has `db_attr`, `ocr_property`, `weight`, `preprocessor`; consumed by `_compute_scores()`

## Entry Points

**Streamlit App:**
- Location: `src/mtg_recognition/app/app.py`
- Triggers: `uv run streamlit run src/mtg_recognition/app/app.py` or `docker compose up`
- Responsibilities: UI rendering, file upload, batch dispatch, result grid/detail views, CSV/Excel/JSON export

**Pipeline CLI:**
- Location: `src/mtg_recognition/engine/pipeline.py` (`@click.command()` at line 152)
- Triggers: `python -m mtg_recognition.engine.pipeline -i <path>`
- Responsibilities: Single-image or directory batch recognition, incremental terminal output, optional JSON file output

**Segmentation CLI:**
- Location: `src/mtg_recognition/engine/segmentation/cli.py`
- Triggers: `python -m mtg_recognition.engine.segmentation.cli`
- Responsibilities: Segmentation-only inspection and debugging, `--mode auto|yolo|cv|rembg`

## Architectural Constraints

- **Threading:** Single-threaded Python for the main pipeline. `multiprocessing.Pool` used only in `PHashCardMatcher.build_index()` for parallelising hash computation across ~1M reference images. The Streamlit server runs all requests in a single process.
- **Global state:** Two module-level singletons: `database.engine` (`src/mtg_recognition/database.py:11`) and `settings` (`src/mtg_recognition/settings.py:91`). Both are effectively read-only after module initialisation.
- **Deferred import to break circular dependency:** `phash_matcher.py` imports `database.engine` inside `find_best_matches()` at runtime (line 315) rather than at module top level, to avoid a circular import between `engine.matching` and the `database` module.
- **Image format convention:** All internal numpy arrays are BGR (OpenCV convention). Conversion to RGB happens only at PIL / Streamlit display boundaries.
- **Hardcoded model paths:** OCR model directories are hardcoded relative to repo root in `ocr.py:145-148`. YOLO model path comes from `settings.YOLO_SEG_MODEL_PATH`. Both require models to be present under `src/mtg_recognition/engine/`.
- **Output size bounds:** Warped card images are always clamped between `WARPED_MIN_SIZE=(488,680)` and `WARPED_MAX_SIZE=(976,1360)` pixels (`seg_utils.py:17-22`). The pHash index was precomputed at Scryfall "medium" size, so the minimum bound is required for hash comparability.

## Anti-Patterns

### Passing a raw photo with apply_segmentation=False

**What happens:** Calling `Pipeline.find_best_matches(..., apply_segmentation=False)` on an unwarped photo.
**Why it's wrong:** The photo is treated as a pre-warped card and fed directly to pHash, producing meaningless similarity scores.
**Do this instead:** Always use `apply_segmentation=True` for raw photos. Only pass `apply_segmentation=False` with a `segment.warped_image` already produced by `YoloCardSegmentor`. See usage in `batch_processor.py:197`.

### Mutating a BatchResult after creation

**What happens:** Modifying fields on a `BatchResult` instance after `_match_segments` returns it.
**Why it's wrong:** `BatchResult` is a plain `@dataclass` (not frozen), so mutation is possible but violates the assumption in `display_batch_detail_view` that stored results are stable.
**Do this instead:** Construct a new `BatchResult(...)` with updated fields rather than setting attributes on an existing instance.

## Error Handling

**Strategy:** Fail-safe degradation. Errors in individual stages are caught and surfaced as status fields, not propagated as exceptions that abort the batch.

**Patterns:**
- `CardOCR.extract_properties()`: wraps `_paddleocr.predict()` in `try/except RuntimeError`; on failure all text lists stay empty and fuzzy scoring falls back to zero vectors, preserving the pHash ranking (`ocr.py:224`)
- `BatchProcessor.process_batch()`: each image is individually wrapped in `try/except Exception`; failures produce `BatchResult(status="error", error_message=str(e))` and the batch continues with remaining images
- `PHashCardMatcher._load_index()`: catches load errors and falls back to calling `build_index()` when `auto_build=True` (`phash_matcher.py:105`)

## Cross-Cutting Concerns

**Logging:** `rich.print` used throughout the engine layer for console output. No structured logger. `print()` statements in production paths are the primary observability mechanism.
**Validation:** Input images validated by `load_and_resize_image()` which raises `ValueError` on decode failure. OCR output fields are all `str | None`; no schema validation is applied.
**Authentication:** None. Single-user local deployment assumed. No auth layer in the Streamlit app or CLI.

---

*Architecture analysis: 2026-05-09*
