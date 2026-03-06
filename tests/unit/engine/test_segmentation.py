from pathlib import Path

import cv2
from numpy.testing import assert_array_equal, assert_raises

from tests.unit.conftest import (
    high_contrast_card_image,
    no_cards_image,
    segmentor,
    valid_card_image,
)


def test_segmentation_valid_card_image(segmentor, valid_card_image):
    result = segmentor.segment(
        valid_card_image, return_annotated=True, return_warped=True
    )
    assert result.success
    assert result.warped_image is not None
    assert result.annotated_image is not None
    with assert_raises(AssertionError):
        assert_array_equal(valid_card_image, result.annotated_image)


def test_warped_image_correct_shape(segmentor, valid_card_image):
    result = segmentor.segment(valid_card_image, return_warped=True)
    assert result.success
    assert result.warped_image is not None
    assert result.warped_image.shape == (680, 488, 3)


def test_segmentation_successful_with_high_contrast_image(
    segmentor, high_contrast_card_image
):
    result = segmentor.segment(high_contrast_card_image, return_warped=True)
    assert result.success
    assert result.warped_image is not None


def test_segmentation_image_without_cards(segmentor, no_cards_image):
    result = segmentor.segment(
        no_cards_image, return_annotated=True, return_warped=True
    )
    assert not result.success
    assert result.warped_image is None
    assert result.corners is None
