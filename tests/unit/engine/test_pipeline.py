"""Pipeline's k validation, and recognition over fake models and the committed
reference data. The `mtg recognize` command is tested in tests/unit/cli.

Recognition with the real models is covered end to end in tests/regression/engine.

The recognition tests replace `segment`, `load_pp_ocr` and `read_card` in the
pipeline module, which CODING_STANDARDS.md forbids (no patching a module's
internals): loading the real models takes seconds, and the behaviour under test is
how `recognize` handles an OCR error, which a plain-data function can't show.
Whether to allow this is decided in #127.
"""

import pytest

from mtg_recognition.engine import pipeline as pipeline_module
from mtg_recognition.engine.pipeline import Pipeline
from tests.unit.engine.builders import card_image, printing, reading_of


@pytest.mark.parametrize(("k_phash", "k_final"), [(5, 6), (5, 0), (0, 0)])
def test_invalid_k_parameters_raise(k_phash, k_final):
    with pytest.raises(ValueError):
        Pipeline(k_phash_candidates=k_phash, k_final_results=k_final)


# --- Pipeline.recognize over fake models ---


def segment_whole_image(query_image):
    """Finds one card: the whole query image."""
    return [query_image]


def read_card_failing(ocr, card_image, *, debug=False):
    raise RuntimeError("OCR exploded")


def read_card_as_p0(ocr, card_image, *, debug=False):
    """Reads every card as the printing "p0"."""
    return reading_of(printing("p0"))


def _pipeline(monkeypatch, read_card) -> Pipeline:
    monkeypatch.setattr(pipeline_module, "segment", segment_whole_image)
    monkeypatch.setattr(pipeline_module, "load_pp_ocr", lambda: None)
    monkeypatch.setattr(pipeline_module, "read_card", read_card)
    return Pipeline(k_final_results=3)


def test_an_ocr_failure_is_recorded_and_scores_every_candidate_0(monkeypatch):
    pipeline = _pipeline(monkeypatch, read_card_failing)

    result = pipeline.recognize(card_image())

    assert result.error is None
    [card] = result.cards
    assert card.ocr_error == "RuntimeError: OCR exploded"
    assert card.reading is None
    assert len(card.ranked) == 3
    assert all(r.score == 0.0 for r in card.ranked)


def test_a_successful_reading_is_no_ocr_fallback(monkeypatch):
    pipeline = _pipeline(monkeypatch, read_card_as_p0)

    [card] = pipeline.recognize(card_image()).cards

    assert card.ocr_error is None
    assert card.reading == reading_of(printing("p0"))
