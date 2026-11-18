"""The Streamlit app run end to end with a fake pipeline, so a crash in the app
script, such as a runtime dependency gone missing (#214), fails a test.
"""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mtg_recognition.app
from mtg_recognition.engine.pipeline import CardResult, RecognitionResult
from mtg_recognition.engine.ranking import FieldScore, RankedPrinting
from mtg_recognition.settings import settings
from tests.unit.engine.builders import card_image, card_png, printing

# found without importing it, since its top level calls Streamlit
APP_SCRIPT = Path(mtg_recognition.app.__file__).parent / "app.py"


def _recognition_of_one_card() -> RecognitionResult:
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

    recognition = _recognition_of_one_card()

    def __init__(self, **_options):
        pass

    def recognize(self, data: bytes, debug: bool = False) -> RecognitionResult:
        return self.recognition


def _recognise(monkeypatch, recognition: RecognitionResult) -> AppTest:
    """The app after recognising two query images, each as recognition."""
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
    at = _recognise(monkeypatch, _recognition_of_one_card())
    # Table 2 only shows in debug mode
    at.sidebar.toggle[0].set_value(True).run()

    assert [m.value for m in at.markdown].count("**Card p1**") == 2

    # open the first card's detail view
    at.button(key="detail_btn_0").click().run()

    assert not at.exception
    assert any("Score matrix" in e.label for e in at.expander)


@pytest.mark.parametrize("debug_mode", [False, True])
def test_debug_mode_starts_as_the_setting_says(monkeypatch, debug_mode):
    monkeypatch.setattr("mtg_recognition.engine.pipeline.Pipeline", FakePipeline)
    monkeypatch.setattr(settings, "DEBUG_MODE", debug_mode)

    at = AppTest.from_file(APP_SCRIPT).run()

    assert at.sidebar.toggle[0].value is debug_mode


def test_a_card_whose_ocr_failed_says_so_on_its_tile_and_detail_view(monkeypatch):
    [card] = _recognition_of_one_card().cards
    card = replace(card, ocr_error="RuntimeError: OCR exploded")
    at = _recognise(monkeypatch, RecognitionResult(cards=[card], error=None))

    assert any("Text not read" in c.value for c in at.caption)

    at.button(key="detail_btn_0").click().run()

    assert not at.exception
    assert any("Text not read" in c.value for c in at.caption)


def test_a_failed_upload_result_shows_its_error_in_one_line(monkeypatch):
    error = "RuntimeError: model exploded\nat node 7"
    failure = RecognitionResult(cards=[], error=error)
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


def test_results_from_before_a_code_change_are_dropped(monkeypatch):
    # When app code changes, Streamlit imports uploads.py again before the next
    # run, so the results in session state are instances of the old classes.
    # A second copy of the module stands for the old one.
    spec = importlib.util.spec_from_file_location(
        "old_uploads", APP_SCRIPT.parent / "uploads.py"
    )
    assert spec is not None and spec.loader is not None
    old_uploads = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_uploads)
    monkeypatch.setattr("mtg_recognition.engine.pipeline.Pipeline", FakePipeline)
    at = AppTest.from_file(APP_SCRIPT).run()
    at.session_state["upload_results"] = [old_uploads.NotRecognized("a.png", None)]

    at.run()

    assert not at.exception
    assert at.session_state["upload_results"] is None
    assert any("Recognize cards" in i.value for i in at.info)
