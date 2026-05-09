# Testing Patterns

**Analysis Date:** 2026-05-09

## Test Framework

**Runner:**
- pytest 9.0.2
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`
- `testpaths = ["tests"]`

**Assertion Library:**
- pytest built-in `assert` statements

**Run Commands:**
```bash
uv run pytest                                               # Run all tests
uv run pytest tests/unit/                                  # Unit tests only (no model required)
uv run pytest tests/regression/ -m slow                    # Regression tests (requires models)
uv run pytest -m "not slow"                                # Fast tests only
uv run pytest tests/unit/engine/test_ocr.py -x -vv        # Single file, verbose
uv run pytest -k "number_and_base" -x -vv                  # Filter by keyword
uv run pytest --cov=src --cov-report=term-missing          # With coverage
```

## Test File Organization

**Location:** Tests are separated from source in a top-level `tests/` directory — not co-located.

**Structure:**
```
tests/
├── unit/
│   ├── conftest.py                        # Session-scoped image fixtures
│   ├── fixtures/
│   │   └── images/                        # Synthetic test images (jpg/png)
│   └── engine/
│       ├── test_ocr.py                    # OCR extraction helpers
│       └── segmentation/
│           ├── test_box.py                # CV box detector
│           ├── test_bypass.py             # Scan bypass logic
│           ├── test_seg_utils.py          # Geometry helpers
│           └── test_yolo.py              # YOLO segmentor unit
└── regression/
    └── segmentation/
        ├── test_yolo_reg.py               # YOLO end-to-end regression
        ├── test_rembg_reg.py              # rembg end-to-end regression
        ├── generate_fixtures.py           # Script to regenerate golden fixtures
        └── fixtures/                      # Golden images, masks, corner JSON
            ├── 0_failure_cases/
            ├── 1_photos_single_card/
            ├── 2_photos_multiple_cards/
            ├── 3_scans_single_card_with_bg/
            ├── 4_scans_single_card_no_bg/
            └── 5_scans_multiple_cards/
```

**Naming:**
- Test files: `test_{module_name}.py` (mirrors source module name)
- Regression test files: `test_{model}_reg.py`

## Test Structure

**Suite Organization:**
```python
class TestExtractNumberAndBaseSetSize:
    """Docstring explains what the class covers and return semantics."""

    def test_simple(self):
        assert extract_number_and_base_set_size(["011/264", "WAR-EN"]) == ("11", "264")

    def test_empty_list(self):
        assert extract_number_and_base_set_size([]) == (None, None)

    @pytest.mark.parametrize(
        "rec_texts,expected",
        [
            (["011/264", "WAR-EN", "B0AG"], ("11", "264")),
            (["1/2", "112/280", "M20.EN", "KEE"], ("112", "280")),
        ],
    )
    def test_real_world_examples(self, rec_texts, expected):
        assert extract_number_and_base_set_size(rec_texts) == expected
```

**Patterns:**
- Group related tests in classes named `Test{FunctionOrConcept}`
- Use docstrings on classes to describe scope and invariants
- Comment inline what values represent when not obvious (`# 630/880 ≈ 0.716`)
- Always include an empty-input and a no-match test for every extraction function
- "must not regress" comment marks tests that anchor existing behaviour
- `@pytest.mark.xfail` with `reason=` for known limitations, not removed tests

## Mocking

**Framework:** `unittest.mock.MagicMock` (stdlib)

**Patterns:**
```python
from unittest.mock import MagicMock

def _make_segmentor() -> YoloCardSegmentor:
    # Bypass __init__ to avoid loading the real model
    seg = YoloCardSegmentor.__new__(YoloCardSegmentor)
    seg.target_size = WARPED_MIN_SIZE
    seg.aspect_ratio_tolerance = 0.2
    seg.batch_size = 20
    seg.fallback_enabled = True
    seg._rembg_segmentor = None
    seg._model = MagicMock()
    return seg
```

**What to Mock:**
- Heavy ML model objects (`._model`, `._session`) when testing non-model logic
- Use `__new__` to bypass `__init__` and set attributes directly

**What NOT to Mock:**
- Pure functions with no side effects (OCR extraction helpers, geometry helpers)
- numpy/OpenCV operations — use synthetic inputs instead

## Fixtures and Factories

**conftest.py fixtures (`tests/unit/conftest.py`):**
```python
FIXTURES_DIR = Path("tests/unit/fixtures")

@pytest.fixture(scope="session")
def valid_card_image() -> np.ndarray:
    return load_and_resize_image(FIXTURES_DIR / "images" / "valid_card.jpg")

@pytest.fixture(scope="session")
def no_cards_image() -> np.ndarray:
    return load_and_resize_image(FIXTURES_DIR / "images" / "no_cards.jpg")
```

**Module-scoped fixtures for expensive resources:**
```python
@pytest.fixture(scope="module")
def segmentor() -> RembgCardSegmentor:
    """Model is loaded once per test module."""
    return RembgCardSegmentor()

@pytest.fixture(scope="module")
def single_corners_golden() -> dict:
    path = PHOTOS_SINGLE_CARD_DIR / "corners.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
```

**Synthetic image factories (module-level helpers, not fixtures):**
```python
def _make_rect_mask(rect_h: int = 400, rect_w: int = 280) -> np.ndarray:
    """Binary mask with a single white rectangle on a black canvas."""
    canvas = np.zeros((rect_h + 100, rect_w + 100), dtype=np.uint8)
    cv2.rectangle(canvas, (50, 50), (50 + rect_w, 50 + rect_h), 255, -1)
    return canvas

def _make_card_on_background(bg_color=255, card_w=315, card_h=440, ...) -> np.ndarray:
    """Synthetic BGR image: plain background with a dark card rectangle."""
```

**Location:**
- Shared session fixtures: `tests/unit/conftest.py`
- Module-local helpers: top of the test file as `_make_*` functions
- Fixture images: `tests/unit/fixtures/images/`
- Regression golden data: `tests/regression/segmentation/fixtures/`

## Coverage

**Requirements:** Not enforced (no coverage threshold configured in `pyproject.toml`)

**View Coverage:**
```bash
uv run pytest --cov=src --cov-report=term-missing
```

## Test Types

**Unit Tests (`tests/unit/`):**
- Scope: pure functions and isolated class methods
- No ML model loading; models replaced with `MagicMock` or bypassed via `__new__`
- Synthetic numpy/OpenCV images generated in test helper functions
- Run without any `pytest.mark` decorator → always run by default
- Examples: OCR extraction helpers, geometry (`clamp_size`, `warp_card`, `corners_from_binary_mask`), CV box detector, bypass logic

**Regression Tests (`tests/regression/`):**
- Scope: end-to-end model inference on real fixture images
- Require model weights at `src/mtg_recognition/engine/segmentation/models/prod/`
- All decorated with `@pytest.mark.slow` → excluded with `-m "not slow"`
- Parametrized over directories of fixture images using `_img_paths()` / `_fixture_stems()` helpers
- Assert segment count, output size within `[WARPED_MIN_SIZE, WARPED_MAX_SIZE]`, IoU ≥ 0.90 vs golden masks, corner drift ≤ 5 px
- Known failures documented with `@pytest.mark.xfail(reason=...)`
- Examples: YOLO three-path pipeline, rembg IoU and corner regression

**E2E Tests:** Not present — the regression tests serve as the closest equivalent.

## Custom Markers

Defined in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: requires the segmentation model; deselect with -m 'not slow'",
]
```

Usage:
```python
@pytest.mark.slow
class TestPhotoSingleCard:
    ...
```

Run/skip slow tests:
```bash
uv run pytest -m slow          # Run only slow tests
uv run pytest -m "not slow"    # Skip slow tests (fast CI)
```

## Common Patterns

**Parametrize over real-world OCR examples:**
```python
@pytest.mark.parametrize(
    "rec_texts,expected",
    [
        (["011/264", "WAR-EN", "B0AG"], ("11", "264")),
        (["3/3", "173/361", "CMR-EN", "ACWANE"], ("173", "361")),
    ],
)
def test_real_world_examples(self, rec_texts, expected):
    assert extract_number_and_base_set_size(rec_texts) == expected
```

**Parametrize over image files in a directory:**
```python
def _img_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("*.jpg"))

@pytest.mark.parametrize(
    "img_path", _img_paths(PHOTOS_SINGLE_CARD), ids=lambda p: p.stem
)
def test_one_segment_detected(self, segmentor, img_path):
    ...
```

**Skip when optional data is missing:**
```python
if len(single_paths) < 2:
    pytest.skip("Need at least 2 photos for ordering test")
```

**Error Testing:**
```python
def test_invalid_k_raises_error(self):
    scores = np.array([0.5, 0.9, 0.3])
    with pytest.raises(ValueError, match="k must be > 0"):
        sort_scores_with_indices(scores, k=0)
```

**IoU helper for mask comparison:**
```python
def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 127
    b = mask_b > 127
    intersection = (a & b).sum()
    union = (a | b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0
```

## Adding New Tests

**New OCR extraction function:**
- Add a class `Test{FunctionName}` in `tests/unit/engine/test_ocr.py`
- Include: empty-input, no-match, simple, and parametrized real-world cases

**New segmentation helper:**
- Add a class `Test{FunctionName}` in `tests/unit/engine/segmentation/test_seg_utils.py`
- Use `_make_rect_mask()`, `_make_corners()`, `_rect_corners()` helpers already present

**New segmentor class:**
- Unit tests in `tests/unit/engine/segmentation/test_{name}.py`; bypass `__init__` with `__new__` and mock the model
- Regression tests in `tests/regression/segmentation/test_{name}_reg.py`; mark the class `@pytest.mark.slow`
- Add fixture images under `tests/regression/segmentation/fixtures/`
- Generate golden masks/corners with `tests/regression/segmentation/generate_fixtures.py`

---

*Testing analysis: 2026-05-09*
