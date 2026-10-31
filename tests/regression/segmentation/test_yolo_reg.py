"""
Regression tests for the three-path segmentation pipeline.

Requires:
  - YOLO model at models/segmentation/prod/yoloe-11s-seg.pt
  - Benchmark images in data/benchmark/ (see the folders below)

Run:
    uv run pytest tests/regression/ -v -m slow
"""

from pathlib import Path

import pytest

from mtg_recognition.engine.image import prepare_image
from mtg_recognition.engine.segmentation.seg_utils import (
    WARPED_MAX_SIZE,
    WARPED_MIN_SIZE,
)
from mtg_recognition.engine.segmentation.yolo import YoloCardSegmentor

# ---------------------------------------------------------------------------
# Benchmark folders
# ---------------------------------------------------------------------------

BENCHMARK = Path("data/benchmark")

PHOTOS_SINGLE_CARD = BENCHMARK / "1_photos_single_card"
PHOTOS_MULTI_CARD = BENCHMARK / "2_photos_multiple_cards"
SCANS_SINGLE_BG = BENCHMARK / "3_scans_single_card_with_bg"
SCANS_SINGLE_NO_BG = BENCHMARK / "4_scans_single_card_no_bg"

# _TARGET_SIZE = WARPED_MIN_SIZE


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
        img = prepare_image(img_path)

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
        out_h, out_w = result.segments[0].warped_image.shape[:2]
        min_w, min_h = WARPED_MIN_SIZE
        max_w, max_h = WARPED_MAX_SIZE
        assert min_w <= out_w <= max_w and min_h <= out_h <= max_h


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
        img = prepare_image(img_path)

        result = segmentor.segment(img)
        assert result.success, f"{img_path.stem}: no segment detected"
        assert len(result.segments) == 1, (
            f"{img_path.stem}: expected 1 segment, got {len(result.segments)}"
        )
        out_h, out_w = result.segments[0].warped_image.shape[:2]
        min_w, min_h = WARPED_MIN_SIZE
        max_w, max_h = WARPED_MAX_SIZE
        assert min_w <= out_w <= max_w and min_h <= out_h <= max_h


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
        img = prepare_image(img_path)

        result = segmentor.segment(img)
        assert result.success, f"{img_path.stem}: no segments detected"
        assert len(result.segments) >= 2, (
            f"{img_path.stem}: expected ≥2 segments, got {len(result.segments)}"
        )
        for i, seg in enumerate(result.segments):
            out_h, out_w = seg.warped_image.shape[:2]
            min_w, min_h = WARPED_MIN_SIZE
            max_w, max_h = WARPED_MAX_SIZE
            assert min_w <= out_w <= max_w and min_h <= out_h <= max_h, (
                f"{img_path.stem} segment {i}: unexpected size ({out_w}, {out_h})"
            )


# ---------------------------------------------------------------------------
# Path 2 — YOLO: scans with bed background
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestScanWithBackground:
    """Scans with visible scanner bed background must yield exactly one segment via
    YOLO.
    """

    @pytest.mark.parametrize(
        "img_path", _img_paths(SCANS_SINGLE_BG), ids=lambda p: p.stem
    )
    def test_one_segment_detected(self, segmentor, img_path):
        img = prepare_image(img_path)

        result = segmentor.segment(img)
        assert result.success, f"{img_path.stem}: no segment detected"
        assert len(result.segments) == 1, (
            f"{img_path.stem}: expected 1 segment, got {len(result.segments)}"
        )
        out_h, out_w = result.segments[0].warped_image.shape[:2]
        min_w, min_h = WARPED_MIN_SIZE
        max_w, max_h = WARPED_MAX_SIZE
        assert min_w <= out_w <= max_w and min_h <= out_h <= max_h, (
            f"{img_path.stem}: unexpected size ({out_w}, {out_h})"
        )
