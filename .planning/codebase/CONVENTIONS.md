# Coding Conventions
_Generated: 2026-05-09_

## Summary

The codebase uses Python 3.12+ with full type annotations across all public interfaces. No linter or formatter is configured (explicitly noted in CLAUDE.md: "No linter/formatter is configured — match the style of surrounding code"). Conventions are self-enforced and consistent: snake_case throughout, `dataclass`/`NamedTuple` for data containers, `logging` module in library code, `print()` only in CLI/job scripts, and private helpers prefixed with `_`.

---

## Naming Patterns

**Files:**
- `snake_case.py` for all modules
- Private helpers use a `_` prefix in name: `_extract_standalone_card_number`, `_normalize_slash_variants`
- Test files mirror source path: `src/mtg_recognition/engine/segmentation/box.py` → `tests/unit/engine/segmentation/test_box.py`

**Functions:**
- Public: `snake_case` — `extract_setcode_candidates`, `find_best_matches`, `segment_batch`
- Private: `_snake_case` prefix — `_extract_number_field`, `_make_segmentor`, `_compute_hash`
- Worker functions (used with multiprocessing): `_compute_hash_worker` (static method pattern)

**Classes:**
- `PascalCase` — `YoloCardSegmentor`, `PHashCardMatcher`, `CardOCR`, `BatchResult`
- Abstract concept classes use descriptive noun phrases, not "I" prefix or "Abstract" suffix

**Variables / parameters:**
- `snake_case` — `rec_texts`, `img_path`, `k_phash_candidates`
- Constants: `UPPER_SNAKE_CASE` — `CARD_ASPECT_RATIO`, `WARPED_MIN_SIZE`, `NAME_ROI_TOP`
- Module-level compiled regex: `_UPPER_SNAKE_CASE` — `_ARTIFACT_PATTERN`, `_S1_BOUNDARY_STD`

**Type aliases / NamedTuples:**
- `PascalCase` — `Properties`, `OCRDebugData`, `SegmentResult`
- Settings properties use `UPPER_CASE` to match env var convention: `settings.MAX_IMAGE_DIMENSION`

---

## Type Annotations

Type annotations are present on all public function signatures. Return types are always annotated.

**Patterns observed:**

```python
# Union with None uses `X | None` syntax (Python 3.10+ style, not Optional[X])
def extract_artist_name(texts: list[str]) -> str | None: ...

# Tuples use built-in tuple, not typing.Tuple
def extract_number_and_base_set_size(
    rec_texts: list[str],
) -> tuple[str | None, str | None]: ...

# Multiple return types joined with |
def corners_from_binary_mask(mask: np.ndarray, erode: bool = True) -> np.ndarray | None: ...

# Frozen dataclasses for immutable results
@dataclass(frozen=True)
class SegmentResult:
    matches: list[Row]
    scores: list[float]
    ocr_debug: OCRDebugData | None

# NamedTuple for property bags
class Properties(NamedTuple):
    setcode_candidates: list[str]
    name: str | None
    ...
```

**`from __future__ import annotations`** is used in some modules (`seg_utils.py`, `core/image_utils.py`, `yolo.py`, `rembg.py`, `cli.py`) to allow forward references. Not universally applied.

---

## Data Containers

Use `dataclass` for mutable/structural data, `NamedTuple` for lightweight property tuples:

| Pattern | Used For | Example |
|---------|----------|---------|
| `@dataclass(frozen=True)` | Public result types | `SegmentResult`, `OCRDebugData` |
| `@dataclass` | Internal mutable state or results | `CardSegment`, `ImageSegmentationResult`, `BatchResult` |
| `NamedTuple` | Property bundles returned from extraction | `Properties` |

`@dataclass` on `ImageSegmentationResult` includes a computed `@property`:
```python
@dataclass
class ImageSegmentationResult:
    segments: list[CardSegment] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.segments)
```

---

## Import Organization

**Order (observed):**
1. Standard library: `import datetime`, `import logging`, `import re`, `from pathlib import Path`
2. Third-party: `import numpy as np`, `import cv2`, `from paddleocr import PaddleOCR`
3. Internal (absolute): `from mtg_recognition.engine.ocr.ocr_extraction import ...`
4. Internal (relative): `from .geometry import find_corners_lines_intersect`

Relative imports are used within the same subpackage (e.g., `seg_utils.py` uses `from .geometry import ...`). Absolute imports are used across subpackages.

**No path aliases configured** — all absolute imports use full `mtg_recognition.*` prefix.

---

## Error Handling

**Patterns observed:**

- Explicit `assert` for invariants / pre-conditions in pipeline:
  ```python
  assert k_final_results <= k_phash_candidates, "k_final_results must be <= k_phash_candidates"
  ```

- `raise ValueError` with descriptive message for invalid arguments:
  ```python
  if scores.ndim != 1:
      raise ValueError("scores must be a 1-D array")
  if k is not None and k <= 0:
      raise ValueError("k must be > 0")
  ```

- Defensive returns (return `None` or empty collection on failure, never raise):
  ```python
  # corners_from_binary_mask returns None on failure
  if not contours:
      return None
  # extract_setcode_candidates returns [] rather than raising
  return []
  ```

- Safety guards prevent total failure silently:
  ```python
  # filter_flavor_text_zone: if all tokens would be discarded, return originals unchanged
  if not kept:
      return texts, boxes, scores
  ```

- `try/except` for third-party C++ backend errors (PaddlePaddle):
  Wraps `predict` call and degrades gracefully to zero-vector scoring so pHash ranking holds.

---

## Logging

**Library code** (`engine/`, `core/`): Use `logging` module.
```python
# ocr_extraction.py pattern
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.debug(f"Processing rect_texts: {rect_texts}")
logger.info(f"Extracted copyright year: {result}")
```

**CLI / job scripts** (`jobs/`, `engine/pipeline.py` CLI): Mix of `print()` and `logging.error/warning`. This is inconsistent — `build_database.py` uses both `print()` for progress and `logging.error/warning` for errors.

**App layer** (`app/`): No explicit logging — uses `rich.print` for terminal output in the pipeline CLI.

---

## Docstrings

**Google-style docstrings** on public functions with meaningful behaviour:

```python
def filter_flavor_text_zone(
    texts: list[str],
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    y_thresh_frac: float = 0.35,
    x_right_frac: float = 0.80,
) -> tuple[list[str], list[tuple[int, int, int, int]], list[float]]:
    """Discard tokens that appear in the flavor-text bleedthrough zone.

    Modern cards with flavor text have their last line(s) partially overlap the
    info ROI. ...

    Args:
        texts:         OCR text tokens from the info ROI.
        boxes:         Bounding boxes [[x1, y1, x2, y2], ...] parallel to texts.
        scores:        OCR confidence scores parallel to texts.
        y_thresh_frac: Upper-zone threshold as fraction of ROI height (default 0.35).
        x_right_frac:  Right-column threshold as fraction of ROI width  (default 0.80).

    Returns:
        Filtered (texts, boxes, scores) with flavor-zone tokens removed.
    """
```

Simple private helpers often have no docstring or a single-line comment. Module-level docstrings appear on some files (`ocr.py`, `seg_utils.py`, `test_seg_utils.py`) but not all.

---

## Constants and Configuration

**Module-level constants** for magic numbers:
```python
# seg_utils.py
CARD_ASPECT_RATIO = 63.0 / 88.0  # with comment explaining source
WARPED_MIN_SIZE: tuple[int, int] = (488, 680)
WARPED_MAX_SIZE: tuple[int, int] = (976, 1360)

# ocr.py
NAME_ROI_TOP = 0.04
INFO_ROI_TOP = 0.88
```

**Pydantic `BaseSettings`** for environment-driven configuration: `src/mtg_recognition/settings.py`. A singleton `settings = Settings()` is imported wherever needed. Settings properties are `UPPER_CASE` to mirror env var names.

---

## Comments

Inline comments explain *why* (not what) for non-obvious decisions:
```python
# Pass 2: two-token concat — only when pass 1 found nothing.
# Handles e.g. ["TDM", "·DE"] split across tokens. Skipped when pass 1
# already succeeded to avoid false positives like "U"+"DMR·JP" → "UDMR".
```

Regex patterns always carry a comment explaining the OCR noise case they handle.

Commented-out dead code exists in a few places (`deep_matcher.py`, `pipeline.py`, `schemas.py`) — leftover exploration code not yet cleaned up.

---

## File Size Norms

Most files are 90–410 lines. The largest file (`ocr_extraction.py` at 576 lines) exceeds the 400-line guideline. `app.py` at 539 lines is also large.

| File | Lines | Status |
|------|-------|--------|
| `engine/ocr/ocr_extraction.py` | 576 | Above guideline |
| `app/app.py` | 539 | Above guideline |
| `engine/matching/phash_matcher.py` | 406 | Borderline |
| `engine/ocr/ocr.py` | 405 | Borderline |
| All others | <400 | OK |

---

## Key Observations

- No formatter/linter enforced; consistency depends on developer discipline
- `X | None` union syntax used everywhere (Python 3.10+), never `Optional[X]`
- `frozen=True` dataclasses for result types crossing layer boundaries
- Private helpers consistently use `_` prefix, both at module level and as class methods
- All regex patterns compiled at module level, never inline inside functions
- `safety guard` pattern: always return something sensible rather than crashing on edge inputs
- Logging inconsistency: `logging` module in library code, `print()` in jobs/CLI (acceptable but not uniform)
- Commented-out dead code in `deep_matcher.py`, `pipeline.py`, `schemas.py` — should be removed
