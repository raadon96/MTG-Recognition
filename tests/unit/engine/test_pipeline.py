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
    SegmentationResult,
    SegmentedCard,
)
from tests.unit.engine.builders import card_image, printing, reading_of


@pytest.mark.parametrize(("k_phash", "k_final"), [(5, 6), (5, 0), (0, 0)])
def test_invalid_k_parameters_raise(k_phash, k_final):
    with pytest.raises(ValueError):
        Pipeline(k_phash_candidates=k_phash, k_final_results=k_final)


# --- Pipeline.recognize over fake models ---


class WholeImageSegmentor:
    """Finds one card: the whole query image."""

    def segment(self, query_image):
        return SegmentationResult(cards=[SegmentedCard(card_image=query_image)])


class FailingReader:
    def read(self, card_image, *, debug=False):
        raise RuntimeError("paddle exploded")


class P0Reader:
    """Reads every card as the printing "p0"."""

    def read(self, card_image, *, debug=False):
        return reading_of(printing("p0"))


def _pipeline(monkeypatch, reader) -> Pipeline:
    monkeypatch.setattr(pipeline_module, "YoloSegmentor", WholeImageSegmentor)
    monkeypatch.setattr(pipeline_module, "PaddleCardReader", reader)
    return Pipeline(k_final_results=3)


def test_an_ocr_failure_is_recorded_and_scores_every_candidate_0(monkeypatch):
    pipeline = _pipeline(monkeypatch, FailingReader)

    result = pipeline.recognize(card_image())

    assert result.error is None
    [card] = result.cards
    assert card.ocr_error == "RuntimeError: paddle exploded"
    assert card.reading is None
    assert len(card.ranked) == 3
    assert all(r.score == 0.0 for r in card.ranked)


def test_a_successful_reading_is_no_ocr_fallback(monkeypatch):
    pipeline = _pipeline(monkeypatch, P0Reader)

    [card] = pipeline.recognize(card_image()).cards

    assert card.ocr_error is None
    assert card.reading == reading_of(printing("p0"))
