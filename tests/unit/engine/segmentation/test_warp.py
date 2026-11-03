"""Straightening a card image: its size, clamped, and its orientation."""

import numpy as np

from mtg_recognition.engine.segmentation.warp import (
    CARD_IMAGE_MAX_SIZE,
    CARD_IMAGE_MIN_SIZE,
    clamp_size,
    warp_card,
)


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
        assert clamp_size((600, 840), CARD_IMAGE_MIN_SIZE, CARD_IMAGE_MAX_SIZE) == (
            600,
            840,
        )

    def test_below_min_clamped_to_min(self):
        assert (
            clamp_size((300, 400), CARD_IMAGE_MIN_SIZE, CARD_IMAGE_MAX_SIZE)
            == CARD_IMAGE_MIN_SIZE
        )

    def test_above_max_clamped_to_max(self):
        assert (
            clamp_size((1100, 1500), CARD_IMAGE_MIN_SIZE, CARD_IMAGE_MAX_SIZE)
            == CARD_IMAGE_MAX_SIZE
        )

    def test_at_min_boundary_unchanged(self):
        assert (
            clamp_size(CARD_IMAGE_MIN_SIZE, CARD_IMAGE_MIN_SIZE, CARD_IMAGE_MAX_SIZE)
            == CARD_IMAGE_MIN_SIZE
        )

    def test_at_max_boundary_unchanged(self):
        assert (
            clamp_size(CARD_IMAGE_MAX_SIZE, CARD_IMAGE_MIN_SIZE, CARD_IMAGE_MAX_SIZE)
            == CARD_IMAGE_MAX_SIZE
        )

    def test_axes_clamped_independently(self):
        # Width below min, height above max → each axis clamped independently
        w_out, h_out = clamp_size((300, 1500), CARD_IMAGE_MIN_SIZE, CARD_IMAGE_MAX_SIZE)
        assert w_out == CARD_IMAGE_MIN_SIZE[0]
        assert h_out == CARD_IMAGE_MAX_SIZE[1]


class TestWarpCard:
    def test_small_corners_clamped_to_min_size(self):
        # Natural size (300×400) is below CARD_IMAGE_MIN_SIZE → output is that size
        corners = _rect_corners(300, 400)
        result = warp_card(_blank_img(300, 400), corners)
        assert result.shape[:2] == (CARD_IMAGE_MIN_SIZE[1], CARD_IMAGE_MIN_SIZE[0])

    def test_medium_corners_natural_size_preserved(self):
        # Natural size (600×840) is within [CARD_IMAGE_MIN_SIZE, CARD_IMAGE_MAX_SIZE] →
        # unchanged
        corners = _rect_corners(600, 840)
        result = warp_card(_blank_img(600, 840), corners)
        assert result.shape[:2] == (840, 600)

    def test_large_corners_clamped_to_max_size(self):
        # Natural size (980×1370) exceeds CARD_IMAGE_MAX_SIZE → clamped down
        corners = _rect_corners(980, 1370)
        result = warp_card(_blank_img(980, 1370), corners)
        assert result.shape[:2] == (CARD_IMAGE_MAX_SIZE[1], CARD_IMAGE_MAX_SIZE[0])

    def test_landscape_corners_rotated_to_portrait(self):
        # Natural size is landscape (840×600) → warp_card rotates to portrait
        corners = _rect_corners(840, 600)
        result = warp_card(_blank_img(840, 600), corners)
        h, w = result.shape[:2]
        assert h >= w
