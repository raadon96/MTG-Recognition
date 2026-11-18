"""The Streamlit app run end to end with a fake pipeline, so a crash in the app
script, such as a runtime dependency gone missing (#214), fails a test.
"""

from dataclasses import replace
from pathlib import Path

from streamlit.testing.v1 import AppTest

import mtg_recognition.app
from mtg_recognition.engine.pipeline import CardResult, RecognitionResult
from mtg_recognition.engine.ranking import FieldScore, RankedPrinting
from mtg_recognition.settings import settings
from tests.unit.engine.builders import card_image, card_png, printing

# found without importing it, since its top level calls Streamlit
APP_SCRIPT = Path(mtg_recognition.app.__file__).parent / "app.py"


def _recognised_card() -> RecognitionResult:
    ranked = [
        RankedPrinting(
            printing("p1"),
            90.0,
            breakdown=[
                FieldScore(field="name", value="Card p1", score=100.0, text="Card p1"),
                # a detail the printing doesn't show
                FieldScore(field="power/toughness", value=None, score=None, text=None),
            ],
        ),
        RankedPrinting(
            printing("p2"),
            40.0,
            breakdown=[
                FieldScore(field="name", value="Card p2", score=40.0, text="Card p1"),
                FieldScore(field="power/toughness", value="1/1", score=0.0, text=None),
            ],
        ),
    ]
    card = CardResult(card_image=card_image(), ranked=ranked, reading=None)
    return RecognitionResult(cards=[card], error=None)


class FakePipeline:
    """Returns the same recognition result for every query image, without models
    or the reference data; a test sets the result.
    """

    recognition = _recognised_card()

    def __init__(self, **_options):
        pass

    def recognize(self, data: bytes, debug: bool = False) -> RecognitionResult:
        return self.recognition


def _recognise(monkeypatch, recognition: RecognitionResult) -> AppTest:
    """The app after recognising two query images, each to recognition."""
    # The app script takes no pipeline to pass a fake through, so replace Pipeline
    # in its module; the script imports it each time it runs, so it gets the fake.
    monkeypatch.setattr("mtg_recognition.engine.pipeline.Pipeline", FakePipeline)
    monkeypatch.setattr(FakePipeline, "recognition", recognition)
    at = AppTest.from_file(APP_SCRIPT).run()
    assert not at.exception
    at.sidebar.file_uploader[0].set_value(
        [("a.png", card_png(), "image/png"), ("b.png", card_png(), "image/png")]
    ).run()
    assert not at.exception
    # start recognition: the first sidebar button
    at.sidebar.button[0].click().run()
    assert not at.exception
    return at


def test_a_recognised_card_s_detail_view_shows_table_2(monkeypatch):
    at = _recognise(monkeypatch, _recognised_card())
    # Table 2 only shows in debug mode
    at.sidebar.toggle[0].set_value(True).run()

    assert [m.value for m in at.markdown].count("**Card p1**") == 2

    # open the first card's detail view
    at.button(key="detail_btn_0").click().run()

    assert not at.exception
    assert any("Score matrix" in e.label for e in at.expander)


def test_debug_mode_starts_off_unless_set(monkeypatch):
    monkeypatch.setattr("mtg_recognition.engine.pipeline.Pipeline", FakePipeline)
    monkeypatch.setattr(settings, "DEBUG_MODE", False)

    at = AppTest.from_file(APP_SCRIPT).run()

    assert at.sidebar.toggle[0].value is False


def test_a_card_whose_ocr_failed_says_so_on_its_tile_and_detail_view(monkeypatch):
    [card] = _recognised_card().cards
    card = replace(card, ocr_error="RuntimeError: OCR exploded")
    at = _recognise(monkeypatch, RecognitionResult(cards=[card], error=None))

    assert any("Text not read" in c.value for c in at.caption)

    at.button(key="detail_btn_0").click().run()

    assert not at.exception
    assert any("Text not read" in c.value for c in at.caption)


def test_a_failed_upload_result_shows_its_error_in_one_line(monkeypatch):
    failure = RecognitionResult(cards=[], error="RuntimeError: model exploded")
    at = _recognise(monkeypatch, failure)

    assert any(c.value == "Error: RuntimeError: model exploded" for c in at.caption)

    at.button(key="detail_btn_0").click().run()

    assert not at.exception
    assert [e.value for e in at.error] == ["Error: RuntimeError: model exploded"]


def test_the_sidebar_names_the_licence_and_links_the_source(monkeypatch):
    monkeypatch.setattr("mtg_recognition.engine.pipeline.Pipeline", FakePipeline)

    at = AppTest.from_file(APP_SCRIPT).run()

    footer = " ".join(c.value for c in at.sidebar.caption)
    assert "AGPL-3.0" in footer
    assert "https://github.com/raadon96/MTG-Recognition" in footer
    assert "Fan Content Policy" in footer
