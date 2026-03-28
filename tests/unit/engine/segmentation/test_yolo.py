"""
Unit tests for YoloCardSegmentor.

All YOLO model I/O is mocked so tests run without model files.
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from mtg_recognition.engine.segmentation.seg_utils import (
    CARD_ASPECT_RATIO,
    WARPED_MAX_SIZE,
    WARPED_MIN_SIZE,
)
from mtg_recognition.engine.segmentation.yolo import (
    _RATIO_TOLERANCE,
    YoloCardSegmentor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_poly_result(h: int, w: int, card_h: int = 440, card_w: int = 315):
    """Build a fake Ultralytics Results object with one card-shaped polygon mask."""
    x0 = (w - card_w) // 2
    y0 = (h - card_h) // 2
    polygon = np.array(
        [[x0, y0], [x0 + card_w, y0], [x0 + card_w, y0 + card_h], [x0, y0 + card_h]],
        dtype=np.float32,
    )
    result = MagicMock()
    result.masks = MagicMock()
    result.masks.xy = [polygon]
    return result


def _make_empty_result():
    result = MagicMock()
    result.masks = None
    return result


def _make_segmentor():
    seg = YoloCardSegmentor.__new__(
        YoloCardSegmentor
    )  # allocates the object without calling __init__
    seg.aspect_ratio_tolerance = _RATIO_TOLERANCE
    seg.batch_size = 20
    seg.fallback_enabled = True
    seg._rembg_segmentor = None
    seg._model = MagicMock()  # fake YOLO model
    return seg


def _non_bypass_img():
    """Return a synthetic image whose aspect ratio is NOT in bypass range."""
    height = 900
    width = int(height * CARD_ASPECT_RATIO * (1 + 2 * _RATIO_TOLERANCE))
    return np.zeros((height, width, 3), dtype=np.uint8)


def _bypass_img():
    """Return a synthetic image whose aspect ratio IS in bypass range."""
    height = 880
    width = int(height * CARD_ASPECT_RATIO)
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Bypass path
# ---------------------------------------------------------------------------


class TestBypassPath:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.seg = _make_segmentor()

    def test_bypass_image_never_reaches_yolo(self):
        imgs = [_bypass_img()]
        self.seg._model.predict.return_value = []
        self.seg.segment_batch(imgs)
        # self.seg._model.predict.return_value = []
        self.seg._model.predict.assert_not_called()

    def test_bypass_returns_one_segment(self):
        imgs = [_bypass_img()]
        results = self.seg.segment_batch(imgs)
        assert len(results) == 1
        assert results[0].success
        assert len(results[0].segments) == 1

    def test_bypass_segment_output_size(self):
        # The bypass image (630×880) is already within [WARPED_MIN_SIZE, WARPED_MAX_SIZE],
        # so its natural size is preserved — it is NOT forced to WARPED_MIN_SIZE.
        imgs = [_bypass_img()]
        results = self.seg.segment_batch(imgs)
        h, w = results[0].segments[0].warped_image.shape[:2]
        from mtg_recognition.engine.segmentation.seg_utils import WARPED_MAX_SIZE

        min_w, min_h = WARPED_MIN_SIZE
        max_w, max_h = WARPED_MAX_SIZE
        assert min_w <= w <= max_w and min_h <= h <= max_h
        assert h >= w  # output is portrait


# ---------------------------------------------------------------------------
# YOLO path
# ---------------------------------------------------------------------------


class TestYoloPath:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.seg = _make_segmentor()

    def test_yolo_called_for_non_bypass_image(self):
        imgs = [_non_bypass_img()]
        self.seg._model.predict.return_value = [_make_empty_result()]
        with patch.object(self.seg, "_fallback_segment", return_value=[]):
            self.seg.segment_batch(imgs)
        self.seg._model.predict.assert_called_once()

    def test_yolo_returns_one_segment_per_detection(self):
        img = _non_bypass_img()
        h, w = img.shape[:2]
        self.seg._model.predict.return_value = [_make_poly_result(h, w)]
        results = self.seg.segment_batch([img])
        assert results[0].success
        assert len(results[0].segments) == 1

    def test_segment_output_size(self):
        img = _non_bypass_img()
        h, w = img.shape[:2]
        self.seg._model.predict.return_value = [_make_poly_result(h, w)]
        results = self.seg.segment_batch([img])
        out_h, out_w = results[0].segments[0].warped_image.shape[:2]
        min_w, min_h = WARPED_MIN_SIZE
        max_w, max_h = WARPED_MAX_SIZE
        assert min_w <= out_w <= max_w and min_h <= out_h <= max_h

    def test_single_image_convenience_method(self):
        img = _non_bypass_img()
        h, w = img.shape[:2]
        self.seg._model.predict.return_value = [_make_poly_result(h, w)]
        result = self.seg.segment(img)
        assert result.success


# ---------------------------------------------------------------------------
# Batch splitting
# ---------------------------------------------------------------------------


class TestMiniBatching:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.seg = _make_segmentor()
        self.seg.batch_size = 20

    def test_25_images_produce_two_predict_calls(self):
        imgs = [_non_bypass_img() for _ in range(25)]
        self.seg._model.predict.return_value = [_make_empty_result()]

        # return value will be recycled; patch to return correct count per call
        def predict_side_effect(batch, **kwargs):
            return [_make_empty_result() for _ in batch]

        self.seg._model.predict.side_effect = predict_side_effect

        with patch.object(self.seg, "_fallback_segment", return_value=[]):
            self.seg.segment_batch(imgs)

        assert self.seg._model.predict.call_count == 2
        first_batch_size = len(self.seg._model.predict.call_args_list[0][0][0])
        second_batch_size = len(self.seg._model.predict.call_args_list[1][0][0])
        assert first_batch_size == 20
        assert second_batch_size == 5

    def test_20_images_produce_one_predict_call(self):
        imgs = [_non_bypass_img() for _ in range(20)]

        def predict_side_effect(batch, **kwargs):
            return [_make_empty_result() for _ in batch]

        self.seg._model.predict.side_effect = predict_side_effect

        with patch.object(self.seg, "_fallback_segment", return_value=[]):
            self.seg.segment_batch(imgs)

        assert self.seg._model.predict.call_count == 1

    def test_result_order_preserved(self):
        """Results must align with input order despite bypass/YOLO splitting."""
        bypass = _bypass_img()
        photo = _non_bypass_img()

        # Arrange: bypass, photo, bypass (indices 0, 1, 2)
        imgs = [bypass, photo, bypass]

        h, w = photo.shape[:2]
        self.seg._model.predict.return_value = [_make_poly_result(h, w)]

        results = self.seg.segment_batch(imgs)
        assert len(results) == 3
        # All should succeed (bypass and YOLO both produce segments)
        for r in results:
            assert r.success


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------


class TestFallbackPath:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.seg = _make_segmentor()

    def test_fallback_called_when_yolo_returns_no_masks(self):
        img = _non_bypass_img()
        self.seg._model.predict.return_value = [_make_empty_result()]
        with patch.object(self.seg, "_fallback_segment", return_value=[]) as mock_fb:
            self.seg.segment_batch([img])
            mock_fb.assert_called_once()
            assert mock_fb.call_args[0][0] is img  # first arg is original image

    def test_fallback_not_called_when_yolo_succeeds(self):
        img = _non_bypass_img()
        h, w = img.shape[:2]
        self.seg._model.predict.return_value = [_make_poly_result(h, w)]
        with patch.object(self.seg, "_fallback_segment", return_value=[]) as mock_fb:
            self.seg.segment_batch([img])
            mock_fb.assert_not_called()

    def test_fallback_skipped_when_disabled(self):
        self.seg.fallback_enabled = False
        img = _non_bypass_img()
        self.seg._model.predict.return_value = [_make_empty_result()]
        with patch.object(self.seg, "_fallback_segment", return_value=[]) as mock_fb:
            results = self.seg.segment_batch([img])
            mock_fb.assert_not_called()
        assert not results[0].success

    def test_rembg_lazy_loaded_only_on_fallback(self):
        assert self.seg._rembg_segmentor is None
        img = _non_bypass_img()
        self.seg._model.predict.return_value = [_make_empty_result()]

        with (
            patch(
                "mtg_recognition.engine.segmentation.yolo.detect_cards_cv",
                return_value=[],
            ),
            patch(
                "mtg_recognition.engine.segmentation.yolo.RembgCardSegmentor"
            ) as mock_cs,
        ):
            mock_instance = MagicMock()
            mock_instance.segment.return_value = MagicMock(segments=[])
            mock_cs.return_value = mock_instance
            self.seg.segment_batch([img])
            mock_cs.assert_called_once()

    def test_cv_fallback_tried_before_rembg(self):
        img = _non_bypass_img()
        self.seg._model.predict.return_value = [_make_empty_result()]

        call_order = []

        with (
            patch(
                "mtg_recognition.engine.segmentation.yolo.detect_cards_cv"
            ) as mock_cv,
            patch(
                "mtg_recognition.engine.segmentation.yolo.RembgCardSegmentor"
            ) as mock_cs,
        ):
            mock_cv.side_effect = lambda *a, **kw: call_order.append("cv") or []
            mock_instance = MagicMock()
            mock_instance.segment.side_effect = lambda *a, **kw: (
                call_order.append("rembg") or MagicMock(segments=[])
            )
            mock_cs.return_value = mock_instance

            self.seg.segment_batch([img])

        assert call_order[0] == "cv"
        assert call_order[1] == "rembg"


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.seg = _make_segmentor()

    def test_empty_input_list_returns_empty(self):
        results = self.seg.segment_batch([])
        assert results == []
