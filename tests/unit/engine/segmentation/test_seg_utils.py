"""
Unit tests for seg_utils helpers: corners_from_binary_mask, valid_aspect_ratio,
clamp_size and warp_card.
"""

import cv2
import numpy as np

from mtg_recognition.engine.segmentation.seg_utils import (
    WARPED_MAX_SIZE,
    WARPED_MIN_SIZE,
    clamp_size,
    corners_from_binary_mask,
    valid_aspect_ratio,
    warp_card,
)


def _make_rect_mask(rect_h: int = 400, rect_w: int = 280) -> np.ndarray:
    """Binary mask with a single white rectangle on a black canvas."""
    canvas = np.zeros((rect_h + 100, rect_w + 100), dtype=np.uint8)
    cv2.rectangle(canvas, (50, 50), (50 + rect_w, 50 + rect_h), 255, -1)
    return canvas


def _make_corners(*points) -> np.ndarray:
    """Build an OpenCV-format (4, 1, 2) corner array from four (x, y) tuples."""
    return np.array([[p] for p in points], dtype=np.int32)


class TestCornersFromBinaryMask:
    def test_rectangle_returns_four_corners(self):
        mask = _make_rect_mask()
        corners = corners_from_binary_mask(mask, erode=False)
        assert corners is not None
        assert corners.shape == (4, 1, 2)

    def test_blank_mask_returns_none(self):
        mask = np.zeros((400, 300), dtype=np.uint8)
        assert corners_from_binary_mask(mask, erode=False) is None

    def test_erode_still_finds_corners(self):
        # Net erosion should shrink the mask but still leave a detectable rectangle
        mask = _make_rect_mask(rect_h=200, rect_w=140)
        corners = corners_from_binary_mask(mask, erode=True)
        assert corners is not None
        assert corners.shape == (4, 1, 2)


class TestValidAspectRatio:
    def test_mtg_card_ratio_is_valid(self):
        # 63 x 88 mm — exact MTG card dimensions
        corners = _make_corners((0, 0), (63, 0), (63, 88), (0, 88))
        assert valid_aspect_ratio(corners) is True

    def test_square_is_invalid(self):
        corners = _make_corners((0, 0), (100, 0), (100, 100), (0, 100))
        assert valid_aspect_ratio(corners) is False

    def test_extreme_elongation_is_invalid(self):
        corners = _make_corners((0, 0), (10, 0), (10, 200), (0, 200))
        assert valid_aspect_ratio(corners) is False

    def test_degenerate_zero_area_is_invalid(self):
        corners = _make_corners((50, 50), (50, 50), (50, 50), (50, 50))
        assert valid_aspect_ratio(corners) is False


def _rect_corners(w: int, h: int) -> np.ndarray:
    """Axis-aligned (4, 1, 2) int32 corners for a rectangle from (0,0) to (w, h)."""
    return np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.int32)


def _blank_img(w: int, h: int) -> np.ndarray:
    """Black BGR image with one extra pixel on each edge so all corners are
    in-bounds.
    """
    return np.zeros((h + 1, w + 1, 3), dtype=np.uint8)


class TestClampSize:
    def test_within_range_unchanged(self):
        assert clamp_size((600, 840), WARPED_MIN_SIZE, WARPED_MAX_SIZE) == (600, 840)

    def test_below_min_clamped_to_min(self):
        assert (
            clamp_size((300, 400), WARPED_MIN_SIZE, WARPED_MAX_SIZE) == WARPED_MIN_SIZE
        )

    def test_above_max_clamped_to_max(self):
        assert (
            clamp_size((1100, 1500), WARPED_MIN_SIZE, WARPED_MAX_SIZE)
            == WARPED_MAX_SIZE
        )

    def test_at_min_boundary_unchanged(self):
        assert (
            clamp_size(WARPED_MIN_SIZE, WARPED_MIN_SIZE, WARPED_MAX_SIZE)
            == WARPED_MIN_SIZE
        )

    def test_at_max_boundary_unchanged(self):
        assert (
            clamp_size(WARPED_MAX_SIZE, WARPED_MIN_SIZE, WARPED_MAX_SIZE)
            == WARPED_MAX_SIZE
        )

    def test_axes_clamped_independently(self):
        # Width below min, height above max → each axis clamped independently
        w_out, h_out = clamp_size((300, 1500), WARPED_MIN_SIZE, WARPED_MAX_SIZE)
        assert w_out == WARPED_MIN_SIZE[0]
        assert h_out == WARPED_MAX_SIZE[1]


class TestWarpCard:
    def test_small_corners_clamped_to_min_size(self):
        # Natural size (300×400) is below WARPED_MIN_SIZE → output is WARPED_MIN_SIZE
        corners = _rect_corners(300, 400)
        result = warp_card(_blank_img(300, 400), corners)
        assert result.shape[:2] == (WARPED_MIN_SIZE[1], WARPED_MIN_SIZE[0])

    def test_medium_corners_natural_size_preserved(self):
        # Natural size (600×840) is within [WARPED_MIN_SIZE, WARPED_MAX_SIZE] →
        # unchanged
        corners = _rect_corners(600, 840)
        result = warp_card(_blank_img(600, 840), corners)
        assert result.shape[:2] == (840, 600)

    def test_large_corners_clamped_to_max_size(self):
        # Natural size (980×1370) exceeds WARPED_MAX_SIZE → clamped down
        corners = _rect_corners(980, 1370)
        result = warp_card(_blank_img(980, 1370), corners)
        assert result.shape[:2] == (WARPED_MAX_SIZE[1], WARPED_MAX_SIZE[0])

    def test_landscape_corners_rotated_to_portrait(self):
        # Natural size is landscape (840×600) → warp_card rotates to portrait
        corners = _rect_corners(840, 600)
        result = warp_card(_blank_img(840, 600), corners)
        h, w = result.shape[:2]
        assert h >= w

    def test_explicit_target_size_forces_exact_output(self):
        # Explicit target_size bypasses clamping entirely
        corners = _rect_corners(600, 840)
        result = warp_card(_blank_img(600, 840), corners, target_size=(488, 680))
        assert result.shape[:2] == (680, 488)
