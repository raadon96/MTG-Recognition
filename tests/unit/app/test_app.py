"""The Streamlit app run end to end with a fake pipeline, so a crash in the app
script, such as a runtime dependency gone missing (#214), fails a test.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import mtg_recognition.app
from mtg_recognition.engine.pipeline import CardResult, RecognitionResult
from mtg_recognition.engine.ranking import FieldScore, RankedPrinting
from tests.unit.engine.builders import card_image, card_png, printing

# found without importing it, since its top level calls Streamlit
APP_SCRIPT = Path(mtg_recognition.app.__file__).parent / "app.py"


class FakePipeline:
    """Recognises every query image as the same card, without models or the
    reference data.
    """

    def __init__(self, **_options):
        pass

    def recognize(self, data: bytes, debug: bool = False) -> RecognitionResult:
        ranked = [
            RankedPrinting(
                printing("p1"),
                90.0,
                breakdown=[
                    FieldScore(
                        field="name", value="Card p1", score=100.0, text="Card p1"
                    ),
                    # a detail the printing doesn't show
                    FieldScore(
                        field="power/toughness", value=None, score=None, text=None
                    ),
                ],
            ),
            RankedPrinting(
                printing("p2"),
                40.0,
                breakdown=[
                    FieldScore(
                        field="name", value="Card p2", score=40.0, text="Card p1"
                    ),
                    FieldScore(
                        field="power/toughness", value="1/1", score=0.0, text=None
                    ),
                ],
            ),
        ]
        card = CardResult(card_image=card_image(), ranked=ranked, reading=None)
        return RecognitionResult(cards=[card], error=None)


def test_a_recognised_card_s_detail_view_shows_table_2(monkeypatch):
    # The app script takes no pipeline to pass a fake through, so replace Pipeline
    # in its module; the script imports it each time it runs, so it gets the fake.
    monkeypatch.setattr("mtg_recognition.engine.pipeline.Pipeline", FakePipeline)
    at = AppTest.from_file(APP_SCRIPT).run()
    assert not at.exception
    # Table 2 only shows in debug mode, so set it on whatever its default
    at.sidebar.toggle[0].set_value(True)

    # submit two query images, then start recognition: the first sidebar button
    at.sidebar.file_uploader[0].set_value(
        [("a.png", card_png(), "image/png"), ("b.png", card_png(), "image/png")]
    ).run()
    assert not at.exception
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert [m.value for m in at.markdown].count("**Card p1**") == 2

    # open the first card's detail view
    at.button(key="detail_btn_0").click().run()

    assert not at.exception
    assert any("Score Matrix" in e.label for e in at.expander)
