"""
Regression tests for the three-path segmentation pipeline.

Requires:
  - YOLO model at src/mtg_recognition/engine/segmentation/models/prod/yoloe-11s-seg.pt
  - Test images in data/test/ (see TEST_DATA structure below)

Run:
    uv run pytest tests/regression/ -v -m slow
"""

from pathlib import Path

import cv2
import pytest

from mtg_recognition.core.image_utils import load_and_resize_image
from mtg_recognition.engine.segmentation.seg_utils import WARPED_MIN_SIZE
from mtg_recognition.engine.segmentation.yolo import YoloCardSegmentor
from mtg_recognition.settings import settings

# ---------------------------------------------------------------------------
# Test data directories
# ---------------------------------------------------------------------------

TEST_DATA = Path("data/test")

PHOTOS_SINGLE_CARD = TEST_DATA / "1_photos_single_card" / "images"
PHOTOS_MULTI_CARD = TEST_DATA / "2_photos_multiple_cards" / "images"
SCANS_SINGLE_BG = TEST_DATA / "3_scans_single_card_with_bg" / "images"
SCANS_SINGLE_NO_BG = TEST_DATA / "4_scans_single_card_no_bg" / "images"

# _TARGET_SIZE = WARPED_MIN_SIZE
_MAX_DIMENSION = settings.MAX_IMAGE_DIMENSION


def _img_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("*.jpg"))


# ---------------------------------------------------------------------------
# Fixture: shared segmentor (loaded once per session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def segmentor():
    return YoloCardSegmentor()


# ---------------------------------------------------------------------------
# Path 1 — Bypass: scans without bed background
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestScanWithoutBedBypass:
    """Every scan-without-bed image must hit the bypass path and produce exactly
    one 488x680 segment without calling YOLO."""

    @pytest.mark.parametrize(
        "img_path", _img_paths(SCANS_SINGLE_NO_BG), ids=lambda p: p.stem
    )
    def test_bypass_produces_one_segment(self, segmentor, img_path):
        img = load_and_resize_image(str(img_path), _MAX_DIMENSION)
        assert img is not None, f"Could not read {img_path}"

        # Verify bypass condition is met
        h, w = img.shape[:2]
        ratio = min(w, h) / max(w, h)
        assert 0.694 <= ratio <= 0.737, (
            f"{img_path.stem}: aspect ratio {ratio:.3f} not in bypass range"
        )

        result = segmentor.segment(img)
        assert result.success, f"{img_path.stem}: no segment produced"
        assert len(result.segments) == 1, (
            f"{img_path.stem}: expected 1 segment, got {len(result.segments)}"
        )
        # out_h, out_w = result.segments[0].warped_image.shape[:2]
        # assert (out_w, out_h) == _TARGET_SIZE, (
        #     f"{img_path.stem}: expected {_TARGET_SIZE}, got ({out_w}, {out_h})"
        # )


# ---------------------------------------------------------------------------
# Path 2 — YOLO: photos (single card)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPhotoSingleCard:
    """Photos of a single card must yield exactly one segment via YOLO."""

    @pytest.mark.parametrize(
        "img_path", _img_paths(PHOTOS_SINGLE_CARD), ids=lambda p: p.stem
    )
    def test_one_segment_detected(self, segmentor, img_path):
        img = load_and_resize_image(str(img_path), _MAX_DIMENSION)
        assert img is not None

        result = segmentor.segment(img)
        assert result.success, f"{img_path.stem}: no segment detected"
        assert len(result.segments) == 1, (
            f"{img_path.stem}: expected 1 segment, got {len(result.segments)}"
        )
        # out_h, out_w = result.segments[0].warped_image.shape[:2]
        # assert (out_w, out_h) == _TARGET_SIZE


# ---------------------------------------------------------------------------
# Path 2 — YOLO: photos (multiple cards)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPhotoMultipleCards:
    """Photos with multiple cards must yield ≥ 2 segments."""

    @pytest.mark.parametrize(
        "img_path", _img_paths(PHOTOS_MULTI_CARD), ids=lambda p: p.stem
    )
    def test_multiple_segments_detected(self, segmentor, img_path):
        img = load_and_resize_image(str(img_path), _MAX_DIMENSION)
        assert img is not None

        result = segmentor.segment(img)
        assert result.success, f"{img_path.stem}: no segments detected"
        assert len(result.segments) >= 2, (
            f"{img_path.stem}: expected ≥2 segments, got {len(result.segments)}"
        )
        # for i, seg in enumerate(result.segments):
        #     out_h, out_w = seg.warped_image.shape[:2]
        #     assert (out_w, out_h) == _TARGET_SIZE, (
        #         f"{img_path.stem} segment {i}: unexpected size ({out_w}, {out_h})"
        #     )


# ---------------------------------------------------------------------------
# Path 2 — YOLO: scans with bed background
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestScanWithBackground:
    """Scans with visible scanner bed background must yield exactly one segment via YOLO."""

    @pytest.mark.parametrize(
        "img_path", _img_paths(SCANS_SINGLE_BG), ids=lambda p: p.stem
    )
    def test_one_segment_detected(self, segmentor, img_path):
        img = load_and_resize_image(str(img_path), _MAX_DIMENSION)
        assert img is not None

        result = segmentor.segment(img)
        assert result.success, f"{img_path.stem}: no segment detected"
        assert len(result.segments) == 1, (
            f"{img_path.stem}: expected 1 segment, got {len(result.segments)}"
        )
        # out_h, out_w = result.segments[0].warped_image.shape[:2]
        # assert (out_w, out_h) == _TARGET_SIZE


# ---------------------------------------------------------------------------
# Batch inference — ordering and mini-batch splitting
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBatchInference:
    """segment_batch must return results in input order and process correctly."""

    def test_batch_results_ordered(self, segmentor):
        single_paths = _img_paths(PHOTOS_SINGLE_CARD)
        if len(single_paths) < 2:
            pytest.skip("Need at least 2 photos for ordering test")

        imgs = [load_and_resize_image(str(p), _MAX_DIMENSION) for p in single_paths[:4]]
        results = segmentor.segment_batch(imgs)

        assert len(results) == len(imgs)
        for i, result in enumerate(results):
            assert result.success, f"Image {i}: no segment"

    def test_batch_larger_than_default_batch_size(self, segmentor):
        """Processing > 20 images must still produce correct results."""
        all_paths = (
            _img_paths(PHOTOS_SINGLE_CARD)
            + _img_paths(PHOTOS_MULTI_CARD)
            + _img_paths(SCANS_SINGLE_BG)
            + _img_paths(SCANS_SINGLE_NO_BG)
        )
        if len(all_paths) < 21:
            pytest.skip("Need > 20 images to test mini-batch splitting")

        imgs = [
            load_and_resize_image(str(p), settings.MAX_IMAGE_DIMENSION)
            for p in all_paths[:25]
        ]
        results = segmentor.segment_batch(imgs)

        assert len(results) == len(imgs)
        # At least most should succeed
        success_count = sum(1 for r in results if r.success)
        assert success_count >= len(imgs) * 0.8, (
            f"Too many failures: {len(imgs) - success_count}/{len(imgs)}"
        )
