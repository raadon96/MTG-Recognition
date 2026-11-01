"""End-to-end recognition with the real stages: on synthetic images, and on real
UUID-named images (requires `dvc pull` and the reference data).
"""

from pathlib import Path

import numpy as np
import pytest

from mtg_recognition.engine.pipeline import Pipeline
from tests.unit.engine.builders import card_image, card_png

TEXTURED_BG_DIR = Path("data/benchmark/6_photos_single_card_textured_bg")
UUIDS = [
    "01013cab-be73-532f-b6bf-0f24d0c3792f",
    "0a2f85cd-f66d-5330-a9d5-314554f871a1",
    "0c941678-45c6-5aab-a863-136c1e0b7a7f",
]


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return Pipeline(k_final_results=3)


@pytest.mark.slow
def test_a_query_image_without_a_card_gives_no_cards_and_no_error(pipeline):
    result = pipeline.recognize(np.full((1000, 1000, 3), 128, dtype=np.uint8))

    assert result.cards == []
    assert result.error is None


@pytest.mark.slow
def test_a_card_sized_image_gives_one_card_ranking_k_final_results(pipeline):
    result = pipeline.recognize(card_image())

    assert result.error is None
    [card] = result.cards
    assert len(card.ranked) == 3


@pytest.mark.slow
def test_the_reading_holds_debug_artefacts_only_in_debug_mode(pipeline):
    [plain] = pipeline.recognize(card_image(), debug=False).cards
    [debugged] = pipeline.recognize(card_image(), debug=True).cards

    assert plain.reading is not None
    assert plain.reading.debug is None
    assert debugged.reading is not None
    assert debugged.reading.debug is not None


@pytest.mark.slow
def test_unreadable_bytes_become_the_error(pipeline):
    result = pipeline.recognize(b"not an image")

    assert result.cards == []
    assert result.error is not None
    assert result.error.startswith("ValueError: ")


@pytest.mark.slow
def test_encoded_image_bytes_are_recognized(pipeline):
    result = pipeline.recognize(card_png())

    assert result.error is None
    assert len(result.cards) == 1


@pytest.mark.slow
@pytest.mark.parametrize("uuid", UUIDS)
def test_recognizes_the_printing_named_by_the_file(pipeline, uuid):
    img_path = TEXTURED_BG_DIR / f"{uuid}.jpg"
    if not img_path.exists():
        pytest.skip(f"{img_path} missing; run `uv run dvc pull`")

    result = pipeline.recognize(img_path)

    assert result.error is None
    [card] = result.cards
    assert card.ranked[0].printing.uuid == uuid
