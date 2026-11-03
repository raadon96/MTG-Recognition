"""Finding a card's four corners from a mask, and checking they are card-shaped."""

import cv2
import numpy as np

from mtg_recognition.engine.segmentation.corners import (
    corners_from_binary_mask,
    valid_aspect_ratio,
)
from mtg_recognition.engine.segmentation.segment import rembg_blob_corners


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
        corners = corners_from_binary_mask(mask)
        assert corners is not None
        assert corners.shape == (4, 1, 2)

    def test_blank_mask_returns_none(self):
        mask = np.zeros((400, 300), dtype=np.uint8)
        assert corners_from_binary_mask(mask) is None


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


def test_a_rembg_blob_still_has_four_corners_after_shrinking():
    corners = rembg_blob_corners(_make_rect_mask(rect_h=200, rect_w=140))
    assert corners is not None
    assert corners.shape == (4, 1, 2)
