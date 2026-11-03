"""Straightening a card image: its size, clamped, and its orientation."""

import numpy as np
import pytest

from mtg_recognition.engine.segmentation.warp import (
    CARD_IMAGE_MAX_SIZE,
    CARD_IMAGE_MIN_SIZE,
    upright_card_image,
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


def _blank(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestUprightCardImage:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [
            ((600, 840), (600, 840)),  # within the limits
            (CARD_IMAGE_MIN_SIZE, CARD_IMAGE_MIN_SIZE),
            (CARD_IMAGE_MAX_SIZE, CARD_IMAGE_MAX_SIZE),
            ((300, 400), CARD_IMAGE_MIN_SIZE),
            ((1100, 1500), CARD_IMAGE_MAX_SIZE),
            # each axis is clamped on its own
            ((300, 1500), (CARD_IMAGE_MIN_SIZE[0], CARD_IMAGE_MAX_SIZE[1])),
        ],
    )
    def test_the_size_is_clamped_between_the_limits(self, size, expected):
        h, w = upright_card_image(_blank(*size)).shape[:2]
        assert (w, h) == expected

    def test_a_landscape_card_image_is_turned_upright(self):
        h, w = upright_card_image(_blank(880, 630)).shape[:2]
        assert (w, h) == (630, 880)


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
