"""Unit tests for the classical-CV box card detector."""

import cv2
import numpy as np
import pytest

from mtg_recognition.engine.segmentation.box import detect_cards_cv
from mtg_recognition.engine.segmentation.seg_utils import WARPED_MIN_SIZE


def _make_card_on_background(
    bg_color: int = 255,
    card_w: int = 315,
    card_h: int = 440,
    canvas_w: int = 700,
    canvas_h: int = 900,
) -> np.ndarray:
    """Create a synthetic BGR image: plain background with a dark card rectangle."""
    img = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
    x0 = (canvas_w - card_w) // 2
    y0 = (canvas_h - card_h) // 2
    card_color = 0 if bg_color > 127 else 200
    cv2.rectangle(img, (x0, y0), (x0 + card_w, y0 + card_h), card_color, thickness=-1)
    return img


class TestDetectCardsCv:
    def test_card_on_white_background(self):
        img = _make_card_on_background(bg_color=255)
        segments = detect_cards_cv(img)
        assert len(segments) == 1

    def test_card_on_dark_background(self):
        img = _make_card_on_background(bg_color=30)
        segments = detect_cards_cv(img)
        assert len(segments) == 1

    def test_small_card_output_clamped_to_min_size(self):
        # Synthetic card (315×440) is smaller than WARPED_MIN_SIZE on both axes,
        # so the natural warp is clamped up to WARPED_MIN_SIZE.
        img = _make_card_on_background()
        segments = detect_cards_cv(img)
        assert len(segments) == 1
        h, w = segments[0].warped_image.shape[:2]
        assert (w, h) == WARPED_MIN_SIZE

    def test_blank_image_returns_empty(self):
        img = np.full((900, 700, 3), 200, dtype=np.uint8)
        segments = detect_cards_cv(img)
        assert segments == []

    def test_square_contour_rejected_by_aspect_ratio(self):
        # A perfect square does not match the MTG card ratio
        img = np.full((700, 700, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (500, 500), 0, thickness=-1)
        segments = detect_cards_cv(img)
        assert segments == []

    def test_tiny_contour_filtered_by_area(self):
        img = np.full((900, 700, 3), 255, dtype=np.uint8)
        # A tiny rectangle that is well below min_area_fraction
        cv2.rectangle(img, (340, 440), (350, 454), 0, thickness=-1)
        segments = detect_cards_cv(img)
        assert segments == []

    def test_warp_img_used_for_output(self):
        """When warp_img is provided, the warped result comes from warp_img, not img."""
        detect_img = _make_card_on_background(bg_color=255)
        # warp_img is filled with a distinct colour (pure blue)
        warp_img = np.zeros_like(detect_img)
        warp_img[:, :, 0] = 200  # blue channel

        segments = detect_cards_cv(detect_img, warp_img=warp_img)
        assert len(segments) == 1
        # The result should be predominantly blue (from warp_img)
        warped = segments[0].warped_image
        assert int(warped[:, :, 0].mean()) > int(warped[:, :, 2].mean())
