"""The corners of a card found from YOLO's outline of it: on the card's edges, not
on the steps of the coarse grid the outline is computed on.
"""

import cv2
import numpy as np
import pytest

from mtg_recognition.engine.segmentation.yolo import card_corners_from_outline

# YOLO computes masks on a grid of 160 cells along the query image's long side,
# as _YOLO_MASK_CELLS in yolo.py
_YOLO_MASK_CELLS = 160


def _query_image_with_card(corners: np.ndarray) -> np.ndarray:
    """A light table with a card on it: black border around a mid-grey frame."""
    query_image = np.full((1536, 2048, 3), 190, dtype=np.uint8)
    cv2.fillPoly(query_image, [_subpixel(corners)], (25, 25, 25), shift=4)
    centre = corners.mean(axis=0)
    inner = centre + (corners - centre) * 0.92
    cv2.fillPoly(query_image, [_subpixel(inner)], (110, 130, 150), shift=4)
    return query_image


def _yolo_outline(query_image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """A stepped outline of the card, as YOLO gives it: the card's mask computed
    at low resolution, scaled up to the query image, and traced.
    """
    h, w = query_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [_subpixel(corners)], (255,), shift=4)
    cell = max(h, w) / _YOLO_MASK_CELLS
    low = cv2.resize(
        mask, (round(w / cell), round(h / cell)), interpolation=cv2.INTER_AREA
    )
    low = (low > 127).astype(np.uint8) * 255
    stepped = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
    stepped = (stepped > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(stepped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)


def _subpixel(points: np.ndarray) -> np.ndarray:
    return np.round(points * 16).astype(np.int32)


def _card_corners(angle_deg: float) -> np.ndarray:
    """Corners of a 630 × 880 px card centred at (1000, 760), turned by angle."""
    half = np.array([[-315, -440], [315, -440], [315, 440], [-315, 440]], float)
    a = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return half @ rotation.T + [1000.3, 760.7]


@pytest.mark.parametrize("angle_deg", [0.0, 3.0, 7.0, -11.0, 25.0])
def test_the_corners_found_from_a_stepped_outline_are_the_card_corners(angle_deg):
    card = _card_corners(angle_deg)
    query_image = _query_image_with_card(card)
    outline = _yolo_outline(query_image, card)

    corners = card_corners_from_outline(query_image, outline)

    assert corners is not None
    found = corners.reshape(4, 2).astype(float)
    distances = np.linalg.norm(card[:, None, :] - found[None, :, :], axis=2)
    # each drawn corner has a found corner within 2 px
    assert distances.min(axis=1).max() <= 2.0


def test_an_outline_that_is_not_card_shaped_has_no_corners():
    square = np.array([[500, 500], [1100, 500], [1100, 1100], [500, 1100]], float)
    query_image = _query_image_with_card(square)
    outline = _yolo_outline(query_image, square)

    assert card_corners_from_outline(query_image, outline) is None
