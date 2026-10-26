"""The bypass for scans without background: a query image that is already one
card image, recognised by its card aspect ratio and dark card border.
"""

import numpy as np
import pytest

from mtg_recognition.engine.segmentation.seg_utils import CARD_ASPECT_RATIO
from mtg_recognition.engine.segmentation.yolo import bypass_warp, is_scan_without_bg


def _with_border(h: int, w: int, border: int, interior: int) -> np.ndarray:
    """An image whose outer 5% strip has grey value `border`, inside `interior`."""
    img = np.full((h, w, 3), border, dtype=np.uint8)
    bh, bw = int(h * 0.05), int(w * 0.05)
    img[bh:-bh, bw:-bw] = interior
    return img


@pytest.mark.parametrize(
    ("h", "w"),
    [
        (880, 630),  # portrait card
        (630, 880),  # landscape card
        (1000, int(CARD_ASPECT_RATIO * 0.97 * 1000) + 1),  # 3% narrower
        (1000, int(CARD_ASPECT_RATIO * 1.03 * 1000)),  # 3% wider
    ],
)
def test_a_dark_image_with_the_card_ratio_is_a_scan_without_bg(h, w):
    assert is_scan_without_bg(np.zeros((h, w, 3), dtype=np.uint8)) is True


@pytest.mark.parametrize(
    ("h", "w"),
    [
        (1080, 1440),  # 4:3 camera
        (720, 1280),  # 16:9 camera
        (600, 600),  # square
        (1000, int(CARD_ASPECT_RATIO * 0.97 * 1000)),  # over 3% narrower
        (1000, int(CARD_ASPECT_RATIO * 1.03 * 1000) + 1),  # over 3% wider
    ],
)
def test_an_image_without_the_card_ratio_is_not_a_scan_without_bg(h, w):
    assert is_scan_without_bg(np.zeros((h, w, 3), dtype=np.uint8)) is False


def test_a_dark_border_around_a_bright_card_is_a_scan_without_bg():
    assert is_scan_without_bg(_with_border(880, 630, border=0, interior=128)) is True


def test_a_bright_border_around_a_dark_card_is_a_scanner_bed():
    assert is_scan_without_bg(_with_border(880, 630, border=240, interior=0)) is False


@pytest.mark.parametrize(("grey", "expected"), [(99, True), (100, False)])
def test_the_border_must_be_darker_than_100(grey, expected):
    img = np.full((880, 630, 3), grey, dtype=np.uint8)
    assert is_scan_without_bg(img) is expected


def test_a_scan_within_the_size_limits_keeps_its_size():
    segment = bypass_warp(np.zeros((880, 630, 3), dtype=np.uint8))
    assert segment.warped_image.shape[:2] == (880, 630)


def test_a_landscape_scan_is_turned_upright():
    segment = bypass_warp(np.zeros((630, 880, 3), dtype=np.uint8))
    assert segment.warped_image.shape[:2] == (880, 630)
