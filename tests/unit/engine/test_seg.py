"""
Tests for src/mtg_recognition/engine/segmentation/segment.py

Unit tests   - fast, synthetic data, no model.
Regression   - marked @pytest.mark.slow; require the rembg model.

Run only unit tests:   uv run pytest -m "not slow"
Run only regression:   uv run pytest -m slow
Run everything:        uv run pytest
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from mtg_recognition.engine.segmentation.segment import (
    CardSegment,
    CardSegmentor,
    ImageSegmentationResult,
    _corners_from_binary_mask,
)

# ---------------------------------------------------------------------------
# Paths  (mirrors tests/unit/fixtures/seg/ structure)
# ---------------------------------------------------------------------------

SEG_FIXTURE_DIR = Path("tests/unit/fixtures/seg")
SINGLE_CARD_DIR = SEG_FIXTURE_DIR / "1_imgs_single_card"
MULTI_CARD_DIR = SEG_FIXTURE_DIR / "2_imgs_multiple_cards"
FAILURE_CASES_DIR = SEG_FIXTURE_DIR / "0_failure_cases"

BENCHMARK_RAW_DIR = Path("benchmarks/dataset/images/raw")

IOU_THRESHOLD = 0.90
CORNER_TOLERANCE_PX = 5
MIN_AREA_FRACTION = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rect_mask(rect_h: int = 400, rect_w: int = 280) -> np.ndarray:
    """Binary mask with a single white rectangle on a black canvas."""
    canvas = np.zeros((rect_h + 100, rect_w + 100), dtype=np.uint8)
    cv2.rectangle(canvas, (50, 50), (50 + rect_w, 50 + rect_h), 255, -1)
    return canvas


def _make_corners(*points) -> np.ndarray:
    """Build an OpenCV-format (4, 1, 2) corner array from four (x, y) tuples."""
    return np.array([[p] for p in points], dtype=np.int32)


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 127
    b = mask_b > 127
    intersection = (a & b).sum()
    union = (a | b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def _corners_match(
    detected: list, golden: list, tol: int = CORNER_TOLERANCE_PX
) -> bool:
    """Order-independent comparison: sort both sets by (x, y) before comparing."""
    if len(detected) != len(golden):
        return False
    det = sorted(detected, key=lambda p: (p[0], p[1]))
    gold = sorted(golden, key=lambda p: (p[0], p[1]))
    return all(
        abs(d[0] - g[0]) <= tol and abs(d[1] - g[1]) <= tol for d, g in zip(det, gold)
    )


def _sorted_blobs(mask: np.ndarray) -> list[np.ndarray]:
    """
    Return per-blob binary masks sorted by centroid (top-to-bottom, left-to-right).
    Blobs smaller than MIN_AREA_FRACTION of image area are discarded.
    """
    min_area = MIN_AREA_FRACTION * mask.shape[0] * mask.shape[1]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    blobs = []
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            continue
        blob_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        cx, cy = centroids[label]
        blobs.append((cy, cx, blob_mask))
    blobs.sort(key=lambda b: (b[0], b[1]))
    return [b[2] for b in blobs]


def _fixture_stems(subdir: str) -> list[str]:
    d = SEG_FIXTURE_DIR / subdir / "images"
    if not d.exists():
        return []
    return [p.stem for p in sorted(d.glob("*.jpg"))]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestCornersFromBinaryMask:
    def test_rectangle_returns_four_corners(self):
        mask = _make_rect_mask()
        corners = _corners_from_binary_mask(mask, erode=False)
        assert corners is not None
        assert corners.shape == (4, 1, 2)

    def test_blank_mask_returns_none(self):
        mask = np.zeros((400, 300), dtype=np.uint8)
        assert _corners_from_binary_mask(mask, erode=False) is None

    def test_erode_still_finds_corners(self):
        # Net erosion should shrink the mask but still leave a detectable rectangle
        mask = _make_rect_mask(rect_h=200, rect_w=140)
        corners = _corners_from_binary_mask(mask, erode=True)
        assert corners is not None
        assert corners.shape == (4, 1, 2)


class TestValidAspectRatio:
    """Tests for CardSegmentor._valid_aspect_ratio using a mock session."""

    @pytest.fixture(autouse=True)
    def _seg(self):
        with patch(
            "mtg_recognition.engine.segmentation.segment.new_session",
            return_value=MagicMock(),
        ):
            self.seg = CardSegmentor()

    def test_mtg_card_ratio_is_valid(self):
        # 63 x 88 mm — exact MTG card dimensions
        corners = _make_corners((0, 0), (63, 0), (63, 88), (0, 88))
        assert self.seg._valid_aspect_ratio(corners) is True

    def test_square_is_invalid(self):
        corners = _make_corners((0, 0), (100, 0), (100, 100), (0, 100))
        assert self.seg._valid_aspect_ratio(corners) is False

    def test_extreme_elongation_is_invalid(self):
        corners = _make_corners((0, 0), (10, 0), (10, 200), (0, 200))
        assert self.seg._valid_aspect_ratio(corners) is False

    def test_degenerate_zero_area_is_invalid(self):
        corners = _make_corners((50, 50), (50, 50), (50, 50), (50, 50))
        assert self.seg._valid_aspect_ratio(corners) is False


class TestImageSegmentationResult:
    def test_empty_segments_not_success(self):
        result = ImageSegmentationResult(segments=[])
        assert result.success is False

    def test_nonempty_segments_is_success(self):
        dummy = CardSegment(warped_image=np.zeros((680, 488, 3), dtype=np.uint8))
        result = ImageSegmentationResult(segments=[dummy])
        assert result.success is True


# ---------------------------------------------------------------------------
# Regression tests — single card
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSingleCardIoU:
    """rembg mask quality vs birefnet-general-lite reference masks.

    Guards against rembg library updates silently degrading rembg output.
    Threshold: IoU >= 0.90 per image.
    """

    @pytest.mark.parametrize("stem", _fixture_stems("1_imgs_single_card"))
    def test_iou_above_threshold(self, segmentor, stem):
        from rembg import remove

        img = cv2.imread(str(SINGLE_CARD_DIR / "images" / f"{stem}.jpg"))
        ref_mask = cv2.imread(
            str(SINGLE_CARD_DIR / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE
        )
        assert img is not None and ref_mask is not None

        raw = remove(
            img,
            session=segmentor._session,
            only_mask=True,
            post_process_mask=True,
            alpha_matting=False,
        )
        _, rembg_mask = cv2.threshold(np.array(raw), 127, 255, cv2.THRESH_BINARY)

        iou = _compute_iou(rembg_mask, ref_mask)
        assert iou >= IOU_THRESHOLD, f"{stem}: IoU={iou:.3f} < {IOU_THRESHOLD}"


@pytest.mark.slow
class TestSingleCardCorners:
    """Corner extraction stability using fixed birefnet reference masks.

    Using the reference mask (not the rembg mask) as input makes the test
    fully deterministic — no model involved. Guards against changes to
    _corners_from_binary_mask or the geometry fallbacks.
    Tolerance: ±5 px per corner.
    """

    @pytest.mark.parametrize("stem", _fixture_stems("1_imgs_single_card"))
    def test_corners_within_tolerance(self, single_corners_golden, stem):
        if stem not in single_corners_golden:
            pytest.skip(f"no golden corners for {stem}")

        ref_mask = cv2.imread(
            str(SINGLE_CARD_DIR / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE
        )
        assert ref_mask is not None

        corners = _corners_from_binary_mask(ref_mask, erode=True)
        assert corners is not None, f"{stem}: no corners found in reference mask"

        detected = corners.reshape(4, 2).tolist()
        assert _corners_match(detected, single_corners_golden[stem]), (
            f"{stem}: corners drifted more than {CORNER_TOLERANCE_PX}px from golden"
        )


# ---------------------------------------------------------------------------
# Regression tests — multi-card
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMultiCardIoU:
    """Per-blob rembg mask quality vs birefnet-general-lite reference masks.

    Blobs are sorted by centroid (top-to-bottom, left-to-right) in both masks
    before comparison. The number of detected rembg blobs must match the
    reference blob count.
    Threshold: IoU >= 0.90 per blob.
    """

    @pytest.mark.parametrize("stem", _fixture_stems("2_imgs_multiple_cards"))
    def test_per_blob_iou_above_threshold(self, segmentor, stem):
        from rembg import remove

        img = cv2.imread(str(MULTI_CARD_DIR / "images" / f"{stem}.jpg"))
        ref_mask = cv2.imread(
            str(MULTI_CARD_DIR / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE
        )
        assert img is not None and ref_mask is not None

        raw = remove(
            img,
            session=segmentor._session,
            only_mask=True,
            post_process_mask=True,
            alpha_matting=False,
        )
        _, rembg_mask = cv2.threshold(np.array(raw), 127, 255, cv2.THRESH_BINARY)

        ref_blobs = _sorted_blobs(ref_mask)
        rembg_blobs = _sorted_blobs(rembg_mask)

        assert len(rembg_blobs) == len(ref_blobs), (
            f"{stem}: expected {len(ref_blobs)} blobs, got {len(rembg_blobs)}"
        )
        for i, (rembg_blob, ref_blob) in enumerate(zip(rembg_blobs, ref_blobs)):
            iou = _compute_iou(rembg_blob, ref_blob)
            assert iou >= IOU_THRESHOLD, (
                f"{stem} blob {i}: IoU={iou:.3f} < {IOU_THRESHOLD}"
            )


@pytest.mark.slow
class TestMultiCardCorners:
    """Per-blob corner extraction stability using birefnet reference masks.

    Fully deterministic (no model). Blobs are sorted by centroid before
    matching against the golden values stored in corners.json.
    Tolerance: ±5 px per corner.
    """

    @pytest.mark.parametrize("stem", _fixture_stems("2_imgs_multiple_cards"))
    def test_per_blob_corners_within_tolerance(self, multi_corners_golden, stem):
        if stem not in multi_corners_golden:
            pytest.skip(f"no golden corners for {stem}")

        ref_mask = cv2.imread(
            str(MULTI_CARD_DIR / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE
        )
        assert ref_mask is not None

        ref_blobs = _sorted_blobs(ref_mask)
        golden_blobs = multi_corners_golden[stem]

        assert len(ref_blobs) == len(golden_blobs), (
            f"{stem}: expected {len(golden_blobs)} blobs, got {len(ref_blobs)}"
        )
        for i, (blob_mask, golden_corners) in enumerate(zip(ref_blobs, golden_blobs)):
            corners = _corners_from_binary_mask(blob_mask, erode=True)
            assert corners is not None, (
                f"{stem} blob {i}: no corners found in reference mask"
            )
            detected = corners.reshape(4, 2).tolist()
            assert _corners_match(detected, golden_corners), (
                f"{stem} blob {i}: corners drifted more than {CORNER_TOLERANCE_PX}px from golden"
            )


# ---------------------------------------------------------------------------
# Regression tests — benchmark dataset
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBenchmarkSuccessRate:
    """All 36 benchmark images must segment to exactly one card.

    Guards end-to-end pipeline regressions (mask → corners → warp).
    """

    def test_all_benchmark_images_succeed(self, segmentor):
        image_paths = sorted(BENCHMARK_RAW_DIR.glob("*.jpg"))
        assert image_paths, f"No benchmark images found in {BENCHMARK_RAW_DIR}"

        failures = []
        for path in image_paths:
            img = cv2.imread(str(path))
            result = segmentor.segment(img)
            if not result.success or len(result.segments) != 1:
                failures.append(
                    f"{path.name}: success={result.success}, segments={len(result.segments)}"
                )

        assert not failures, "Benchmark regressions:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Regression tests — multi-card segmentation
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMultiCardSegmentation:
    """Multi-card images with gaps must yield one segment per card."""

    def test_two_cards_detected(self, segmentor):
        img_path = MULTI_CARD_DIR / "images" / "two_cards_with_gap.jpg"
        img = cv2.imread(str(img_path))
        assert img is not None
        result = segmentor.segment(img)
        assert result.success
        assert len(result.segments) == 2, (
            f"{img_path.stem}: expected 2 segments, got {len(result.segments)}"
        )

    def test_four_cards_detected(self, segmentor):
        img_path = MULTI_CARD_DIR / "images" / "four_cards_with_gap.jpg"
        img = cv2.imread(str(img_path))
        assert img is not None
        result = segmentor.segment(img)
        assert result.success
        assert len(result.segments) == 4, (
            f"{img_path.stem}: expected 4 segments, got {len(result.segments)}"
        )


# ---------------------------------------------------------------------------
# Regression tests — failure / edge cases
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEdgeCases:
    def test_card_and_pen_detects_single_card(self, segmentor):
        """Pen blob must be rejected by _valid_aspect_ratio (tolerance=0.2)."""
        img = cv2.imread(str(SINGLE_CARD_DIR / "images" / "card_and_pen.jpg"))
        assert img is not None
        result = segmentor.segment(img)
        assert len(result.segments) == 1, (
            f"Expected exactly 1 segment (card only), got {len(result.segments)}"
        )

    @pytest.mark.xfail(
        reason="Four tightly packed cards produce a single merged blob; "
        "instance separation is not supported for zero-gap layouts"
    )
    def test_four_cards_without_gap(self, segmentor):
        img = cv2.imread(
            str(FAILURE_CASES_DIR / "images" / "four_cards_without_gap.jpg")
        )
        assert img is not None
        result = segmentor.segment(img)
        assert len(result.segments) == 4
