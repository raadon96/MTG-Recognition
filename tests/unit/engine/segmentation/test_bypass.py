"""Unit tests for the scan-without-bed bypass logic in YoloCardSegmentor."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from mtg_recognition.engine.segmentation.seg_utils import WARPED_TARGET_SIZE
from mtg_recognition.engine.segmentation.yolo import (
    _BYPASS_HI,
    _BYPASS_LO,
    YoloCardSegmentor,
)


def _make_segmentor() -> YoloCardSegmentor:
    seg = YoloCardSegmentor.__new__(YoloCardSegmentor)
    seg.target_size = WARPED_TARGET_SIZE
    seg.aspect_ratio_tolerance = 0.2
    seg.batch_size = 20
    seg.fallback_enabled = True
    seg._rembg_segmentor = None
    seg._model = MagicMock()
    return seg


class TestIsScanWithoutBg:
    """Tests for YoloCardSegmentor._is_scan_without_bg.

    The method returns True when both:
      1. image aspect ratio is within ±3% of the MTG card ratio (63/88 ≈ 0.716)
      2. the outer 5% border strip has a mean grayscale value < 100
    """

    @pytest.fixture
    def seg(self):
        return _make_segmentor()

    def test_exact_card_portrait(self, seg):
        img = np.zeros((880, 630, 3), dtype=np.uint8)  # 630/880 ≈ 0.716
        assert seg._is_scan_without_bg(img) is True

    def test_exact_card_landscape(self, seg):
        img = np.zeros((630, 880, 3), dtype=np.uint8)  # 630/880 ≈ 0.716
        assert seg._is_scan_without_bg(img) is True

    def test_4_by_3_camera(self, seg):
        img = np.zeros((1080, 1440, 3), dtype=np.uint8)  # 1080/1440 = 0.75
        assert seg._is_scan_without_bg(img) is False

    def test_16_by_9_camera(self, seg):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)  # 720/1280 ≈ 0.5625
        assert seg._is_scan_without_bg(img) is False

    def test_square(self, seg):
        img = np.zeros((600, 600, 3), dtype=np.uint8)  # ratio = 1.0
        assert seg._is_scan_without_bg(img) is False

    def test_lower_boundary(self, seg):
        # First integer width that puts ratio strictly above _BYPASS_LO
        w = int(_BYPASS_LO * 1000) + 1
        img = np.zeros((1000, w, 3), dtype=np.uint8)
        assert w / 1000 > _BYPASS_LO
        assert seg._is_scan_without_bg(img) is True

    def test_upper_boundary(self, seg):
        # Largest integer width whose ratio is still within _BYPASS_HI
        w = int(_BYPASS_HI * 1000)
        img = np.zeros((1000, w, 3), dtype=np.uint8)
        assert w / 1000 <= _BYPASS_HI
        assert seg._is_scan_without_bg(img) is True

    def test_just_below_lower_boundary(self, seg):
        # First integer width whose ratio falls below _BYPASS_LO
        w = int(_BYPASS_LO * 1000)
        img = np.zeros((1000, w, 3), dtype=np.uint8)
        assert w / 1000 < _BYPASS_LO
        assert seg._is_scan_without_bg(img) is False

    def test_just_above_upper_boundary(self, seg):
        # First integer width whose ratio exceeds _BYPASS_HI
        w = int(_BYPASS_HI * 1000) + 1
        img = np.zeros((1000, w, 3), dtype=np.uint8)
        assert w / 1000 > _BYPASS_HI
        assert seg._is_scan_without_bg(img) is False

    # -- Border brightness condition ------------------------------------------

    def test_dark_border_passes(self, seg):
        # Card-ratio image with a dark border strip but bright interior → bypass.
        # Isolates the border brightness check from the all-zeros ratio tests above.
        img = np.full((880, 630, 3), 128, dtype=np.uint8)  # bright interior
        bh, bw = int(880 * 0.05), int(630 * 0.05)
        img[:bh, :] = 0    # top border: black
        img[-bh:, :] = 0   # bottom border: black
        img[:, :bw] = 0    # left border: black
        img[:, -bw:] = 0   # right border: black
        assert seg._is_scan_without_bg(img) is True

    def test_bright_border_rejected(self, seg):
        # Card-ratio image with a bright border (scanner bed) → no bypass
        img = np.full((880, 630, 3), 200, dtype=np.uint8)
        assert seg._is_scan_without_bg(img) is False

    def test_bright_border_dark_interior_rejected(self, seg):
        # Simulates a scan-with-background: bright outer strip, dark card inside
        img = np.zeros((880, 630, 3), dtype=np.uint8)
        bh, bw = int(880 * 0.05), int(630 * 0.05)
        img[:bh, :] = 240
        img[-bh:, :] = 240
        img[:, :bw] = 240
        img[:, -bw:] = 240
        assert seg._is_scan_without_bg(img) is False

    def test_border_mean_just_below_threshold(self, seg):
        # border_mean = 99 → should bypass (below threshold of 100)
        img = np.full((880, 630, 3), 99, dtype=np.uint8)
        assert seg._is_scan_without_bg(img) is True

    def test_border_mean_at_threshold(self, seg):
        # border_mean = 100 → should NOT bypass (not strictly below 100)
        img = np.full((880, 630, 3), 100, dtype=np.uint8)
        assert seg._is_scan_without_bg(img) is False


class TestBypassWarp:
    @pytest.fixture
    def seg(self):
        return _make_segmentor()

    def test_portrait_output_size(self, seg):
        img = np.zeros((880, 630, 3), dtype=np.uint8)
        segment = seg._bypass_warp(img)
        h, w = segment.warped_image.shape[:2]
        assert (w, h) == WARPED_TARGET_SIZE

    def test_landscape_rotated_to_portrait(self, seg):
        img = np.zeros((630, 880, 3), dtype=np.uint8)
        segment = seg._bypass_warp(img)
        h, w = segment.warped_image.shape[:2]
        assert (w, h) == WARPED_TARGET_SIZE

    def test_output_is_3_channel(self, seg):
        img = np.zeros((880, 630, 3), dtype=np.uint8)
        segment = seg._bypass_warp(img)
        assert segment.warped_image.ndim == 3
        assert segment.warped_image.shape[2] == 3
