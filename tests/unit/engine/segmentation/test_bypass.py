"""Unit tests for the scan-without-bed bypass logic in YoloCardSegmentor."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest


def _make_segmentor():
    from mtg_recognition.engine.segmentation.yolo import YoloCardSegmentor

    seg = YoloCardSegmentor.__new__(YoloCardSegmentor)
    seg.target_size = (488, 680)
    seg.aspect_ratio_tolerance = 0.2
    seg.batch_size = 20
    seg._rembg_segmentor = None
    seg._model = MagicMock()
    return seg


class TestIsScanWithoutBg:
    @pytest.fixture(autouse=True)
    def seg(self):
        self.seg = _make_segmentor()

    def test_exact_card_portrait(self):
        img = np.zeros((880, 630, 3), dtype=np.uint8)  # 630/880 ≈ 0.716
        assert self.seg._is_scan_without_bg(img) is True

    def test_exact_card_landscape(self):
        img = np.zeros((630, 880, 3), dtype=np.uint8)  # 630/880 ≈ 0.716
        assert self.seg._is_scan_without_bg(img) is True

    def test_4_by_3_camera(self):
        img = np.zeros((1080, 1440, 3), dtype=np.uint8)  # 1080/1440 = 0.75
        assert self.seg._is_scan_without_bg(img) is False

    def test_16_by_9_camera(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)  # 720/1280 ≈ 0.5625
        assert self.seg._is_scan_without_bg(img) is False

    def test_square(self):
        img = np.zeros((600, 600, 3), dtype=np.uint8)  # 1.0
        assert self.seg._is_scan_without_bg(img) is False

    def test_lower_boundary(self):
        # ratio = 695/1000 = 0.695 > _BYPASS_LO (≈ 0.6941)
        img = np.zeros((1000, 695, 3), dtype=np.uint8)
        assert self.seg._is_scan_without_bg(img) is True

    def test_upper_boundary(self):
        # ratio ≈ 0.737 (just at upper boundary)
        img = np.zeros((1000, 737, 3), dtype=np.uint8)
        assert self.seg._is_scan_without_bg(img) is True

    def test_just_below_lower_boundary(self):
        # ratio ≈ 0.69 (just below 0.694)
        img = np.zeros((1000, 690, 3), dtype=np.uint8)
        assert self.seg._is_scan_without_bg(img) is False

    def test_just_above_upper_boundary(self):
        # ratio ≈ 0.74 (just above 0.737)
        img = np.zeros((1000, 740, 3), dtype=np.uint8)
        assert self.seg._is_scan_without_bg(img) is False

    # -- Border brightness condition ------------------------------------------

    def test_dark_border_passes(self):
        # Card-ratio image filled with black (border_mean ≈ 0) → bypass
        img = np.zeros((880, 630, 3), dtype=np.uint8)
        assert self.seg._is_scan_without_bg(img) is True

    def test_bright_border_rejected(self):
        # Card-ratio image with a bright border (scanner bed) → no bypass
        img = np.full((880, 630, 3), 200, dtype=np.uint8)
        assert self.seg._is_scan_without_bg(img) is False

    def test_bright_border_dark_interior_rejected(self):
        # Simulates a scan-with-background: bright outer strip, dark card inside
        img = np.zeros((880, 630, 3), dtype=np.uint8)
        # Paint a 5% border white (≈ 44px top/bottom, 31px left/right)
        bh, bw = int(880 * 0.05), int(630 * 0.05)
        img[:bh, :] = 240
        img[-bh:, :] = 240
        img[:, :bw] = 240
        img[:, -bw:] = 240
        assert self.seg._is_scan_without_bg(img) is False

    def test_border_mean_just_below_threshold(self):
        # border_mean = 99 → should bypass (below threshold of 100)
        img = np.full((880, 630, 3), 99, dtype=np.uint8)
        assert self.seg._is_scan_without_bg(img) is True

    def test_border_mean_at_threshold(self):
        # border_mean = 100 → should NOT bypass (not strictly below 100)
        img = np.full((880, 630, 3), 100, dtype=np.uint8)
        assert self.seg._is_scan_without_bg(img) is False


class TestBypassWarp:
    @pytest.fixture(autouse=True)
    def seg(self):
        self.seg = _make_segmentor()

    def test_portrait_output_size(self):
        img = np.zeros((880, 630, 3), dtype=np.uint8)
        segment = self.seg._bypass_warp(img)
        h, w = segment.warped_image.shape[:2]
        assert (w, h) == (488, 680)

    def test_landscape_rotated_to_portrait(self):
        img = np.zeros((630, 880, 3), dtype=np.uint8)
        segment = self.seg._bypass_warp(img)
        h, w = segment.warped_image.shape[:2]
        assert (w, h) == (488, 680)

    def test_output_is_3_channel(self):
        img = np.zeros((880, 630, 3), dtype=np.uint8)
        segment = self.seg._bypass_warp(img)
        assert segment.warped_image.ndim == 3
        assert segment.warped_image.shape[2] == 3
