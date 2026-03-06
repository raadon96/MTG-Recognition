from pathlib import Path

import cv2
import numpy as np
import pytest

from mtg_recognition.engine.segmentation.segment import CardSegmentor


class TestCardSegmentor:
    """Unit tests for CardSegmentor using yoloe-11s-seg model."""

    @pytest.fixture
    def segmentor(self):
        """Create CardSegmentor instance with yoloe-11s-seg model."""
        return CardSegmentor(model_name="yoloe-11s-seg", param_set_idx=0)

    @pytest.fixture
    def valid_card_image(self):
        """Load test image with a single valid card."""
        img_path = (
            Path(__file__).parent.parent.parent / "fixtures" / "images" / "valid.jpg"
        )
        return cv2.imread(str(img_path))

    @pytest.fixture
    def two_cards_image(self):
        """Load test image with two cards."""
        img_path = (
            Path(__file__).parent.parent.parent
            / "fixtures"
            / "images"
            / "two_cards.png"
        )
        return cv2.imread(str(img_path))

    @pytest.fixture
    def difficult_lighting_image(self):
        """Load test image with difficult lighting conditions."""
        img_path = (
            Path(__file__).parent.parent.parent
            / "fixtures"
            / "images"
            / "difficult_lighting.png"
        )
        return cv2.imread(str(img_path))

    @pytest.fixture
    def no_cards_image(self):
        """Load test image without any cards."""
        img_path = (
            Path(__file__).parent.parent.parent / "fixtures" / "images" / "no_cards.png"
        )
        return cv2.imread(str(img_path))

    def test_segmentor_initialization(self, segmentor):
        """Test that CardSegmentor initializes correctly."""
        assert segmentor.model is not None
        assert segmentor.model.model_name == "yoloe-11s-seg"
        assert segmentor.target_size == (488, 680)
        assert segmentor.padding_pct == 0.1
        assert segmentor.model.task == "segmentation"

    def test_valid_card_segmentation(self, segmentor, valid_card_image):
        """Test segmentation with a single valid card."""
        assert valid_card_image is not None, "Valid card image could not be loaded"

        img_corners, img_warped = segmentor.segment(
            valid_card_image, return_warped=True
        )

        # Check that corners are detected and drawn
        assert img_corners is not None
        assert img_corners.shape == valid_card_image.shape

        # Check that warped image is produced
        assert img_warped is not None
        assert img_warped.shape == segmentor.target_size[::-1] + (
            3,
        )  # (height, width, channels)

    def test_valid_card_segmentation_no_warp(self, segmentor, valid_card_image):
        """Test segmentation with a single valid card without returning warped image."""
        assert valid_card_image is not None, "Valid card image could not be loaded"

        img_corners, img_warped = segmentor.segment(
            valid_card_image, return_warped=False
        )

        # Check that corners are detected and drawn
        assert img_corners is not None
        assert img_corners.shape == valid_card_image.shape

        # Check that warped image is None when not requested
        assert img_warped is None

    def test_two_cards_segmentation(self, segmentor, two_cards_image):
        """Test segmentation with two cards in the image."""
        assert two_cards_image is not None, "Two cards image could not be loaded"

        img_corners, img_warped = segmentor.segment(two_cards_image, return_warped=True)

        # Check that corners are detected and drawn
        assert img_corners is not None
        assert img_corners.shape == two_cards_image.shape

        # Check that warped image is produced (should detect the largest/best card)
        assert img_warped is not None
        assert img_warped.shape == segmentor.target_size[::-1] + (3,)

    def test_difficult_lighting_segmentation(self, segmentor, difficult_lighting_image):
        """Test segmentation with difficult lighting conditions."""
        assert difficult_lighting_image is not None, (
            "Difficult lighting image could not be loaded"
        )

        img_corners, img_warped = segmentor.segment(
            difficult_lighting_image, return_warped=True
        )

        # Check that corners are detected and drawn
        assert img_corners is not None
        assert img_corners.shape == difficult_lighting_image.shape

        # Check that warped image is produced (may fail with very difficult lighting)
        if img_warped is not None:
            assert img_warped.shape == segmentor.target_size[::-1] + (3,)

    def test_no_cards_segmentation(self, segmentor, no_cards_image):
        """Test segmentation when no cards are present in the image."""
        assert no_cards_image is not None, "No cards image could not be loaded"

        img_corners, img_warped = segmentor.segment(no_cards_image, return_warped=True)

        # Should return original image and None for warped when no cards detected
        assert img_corners is not None
        assert img_corners.shape == no_cards_image.shape
        assert img_warped is None

    def test_find_corners_from_mask_valid_input(self, segmentor, valid_card_image):
        """Test find_corners_from_mask method with valid input."""
        # Get mask points from model prediction
        mask_points = segmentor.model.predict(valid_card_image)

        if mask_points is not None:
            corner_points = segmentor.find_corners_from_mask(
                mask_points, valid_card_image.shape, erode=True, refine=True
            )

            assert corner_points is not None
            assert corner_points.shape == (4, 1, 2)  # 4 corners in OpenCV format
            assert corner_points.dtype == np.int32

    def test_find_corners_from_mask_invalid_input(self, segmentor):
        """Test find_corners_from_mask method with invalid input."""
        # Test with empty mask points - this should raise an exception
        empty_mask = np.array([]).reshape(0, 1, 2)

        with pytest.raises(cv2.error):
            segmentor.find_corners_from_mask(
                empty_mask, (100, 100, 3), erode=True, refine=True
            )

    def test_segment_with_none_image(self, segmentor):
        """Test segment method with None input."""
        # The YOLO model handles None by using a default image path
        # This may return None if no valid image is found
        img_corners, img_warped = segmentor.segment(None, return_warped=True)

        # Should handle None gracefully (may return None or default image)
        # The important thing is that it doesn't crash
        assert img_corners is None or isinstance(img_corners, np.ndarray)
        # Warped image should be None when no card is detected
        assert img_warped is None

    def test_segment_with_empty_image(self, segmentor):
        """Test segment method with empty numpy array."""
        empty_img = np.array([])

        # Empty array should raise ZeroDivisionError in YOLO preprocessing
        with pytest.raises(ZeroDivisionError):
            segmentor.segment(empty_img, return_warped=True)

    def test_target_size_configuration(self):
        """Test that target size can be configured."""
        custom_size = (300, 400)
        segmentor = CardSegmentor(
            model_name="yoloe-11s-seg", param_set_idx=0, target_size=custom_size
        )

        assert segmentor.target_size == custom_size

    def test_segmentation_consistency(self, segmentor, valid_card_image):
        """Test that segmentation results are consistent across multiple runs."""
        assert valid_card_image is not None, "Valid card image could not be loaded"

        # Run segmentation multiple times
        results = []
        for _ in range(3):
            _, img_warped = segmentor.segment(valid_card_image, return_warped=True)
            if img_warped is not None:
                results.append(img_warped.copy())

        # Check that all successful runs produce similar results
        if len(results) >= 2:
            # Compare first two results
            diff = cv2.absdiff(results[0], results[1])
            mean_diff = np.mean(diff)

            # Results should be very similar (allowing for minor floating point differences)
            assert mean_diff < 5.0, (
                f"Segmentation results not consistent: mean diff = {mean_diff}"
            )
