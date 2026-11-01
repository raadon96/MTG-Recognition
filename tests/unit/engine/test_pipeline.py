"""Pipeline's k validation, and recognition over fake models and the committed
reference data. The `mtg recognize` command is tested in tests/unit/cli.

Recognition with the real models is covered end to end in tests/regression/engine.

The recognition tests replace the YOLO and PaddleOCR classes in the pipeline module,
which CODING_STANDARDS.md forbids (no patching a module's internals): loading the
real models takes seconds, and the behaviour under test is how `recognize` handles a
reader error, which a plain-data function can't show. Whether to allow this is
decided in #127.
"""

import pytest

from mtg_recognition.engine import pipeline as pipeline_module
from mtg_recognition.engine.pipeline import Pipeline
from mtg_recognition.engine.segmentation.seg_utils import (
    CardSegment,
    ImageSegmentationResult,
)
from tests.unit.engine.builders import card_image, printing, reading_of


@pytest.mark.parametrize(("k_phash", "k_final"), [(5, 6), (5, 0), (0, 0)])
def test_invalid_k_parameters_raise(k_phash, k_final):
    with pytest.raises(ValueError):
        Pipeline(k_phash_candidates=k_phash, k_final_results=k_final)


# --- Pipeline.recognize over fake models ---


class OneCardSegmentor:
    """Finds one card: the whole image."""

    def segment(self, img):
        return ImageSegmentationResult(segments=[CardSegment(warped_image=img)])


class FailingReader:
    def read(self, card_image, *, debug=False):
        raise RuntimeError("paddle exploded")


class P0Reader:
    """Reads every card as the printing "p0"."""

    def read(self, card_image, *, debug=False):
        return reading_of(printing("p0"))


def _pipeline(monkeypatch, reader) -> Pipeline:
    monkeypatch.setattr(pipeline_module, "YoloCardSegmentor", OneCardSegmentor)
    monkeypatch.setattr(pipeline_module, "PaddleCardReader", reader)
    return Pipeline(k_final_results=3)


def test_an_ocr_failure_is_recorded_and_scores_every_candidate_0(monkeypatch):
    pipeline = _pipeline(monkeypatch, FailingReader)

    result = pipeline.recognize(card_image())

    assert result.error is None
    [segment] = result.segments
    assert segment.ocr_error == "RuntimeError: paddle exploded"
    assert segment.reading is None
    assert len(segment.ranked) == 3
    assert all(r.score == 0.0 for r in segment.ranked)


def test_a_successful_reading_is_no_ocr_fallback(monkeypatch):
    pipeline = _pipeline(monkeypatch, P0Reader)

    [segment] = pipeline.recognize(card_image()).segments

    assert segment.ocr_error is None
    assert segment.reading == reading_of(printing("p0"))
