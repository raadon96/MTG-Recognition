"""Upload results built from hand-made recognition results."""

import io
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from mtg_recognition.app.uploads import (
    Failed,
    NotRecognized,
    Recognized,
    export_bytes,
    upload_results,
)
from mtg_recognition.engine.ocr.reading import Reading, ReadingDebug
from mtg_recognition.engine.pipeline import CardResult, RecognitionResult
from mtg_recognition.engine.ranking import RankedPrinting
from tests.unit.engine.builders import card_image, printing

NAME = "card.jpg"
DATA = b"uploaded bytes"
ERROR = RecognitionResult(cards=[], error="RuntimeError: model exploded")
NO_CARD = RecognitionResult(cards=[], error=None)


def _card(uuid: str = "p0", value: int = 3) -> CardResult:
    return CardResult(
        card_image=card_image(value),
        ranked=[RankedPrinting(printing(uuid), 100.0)],
        reading=None,
    )


def _recognition(*cards: CardResult) -> RecognitionResult:
    return RecognitionResult(cards=list(cards), error=None)


def test_a_card_with_candidates_is_recognition():
    [result] = upload_results(NAME, DATA, _recognition(_card()))

    assert isinstance(result, Recognized)


def test_a_query_image_without_a_card_is_not_recognition():
    [result] = upload_results(NAME, DATA, NO_CARD)

    assert isinstance(result, NotRecognized)


def test_a_card_without_candidates_is_not_recognized_and_holds_no_card_image():
    card = CardResult(card_image=card_image(), ranked=[], reading=None)

    [result] = upload_results(NAME, DATA, _recognition(card))

    assert result == NotRecognized(NAME, DATA)


def test_a_recognition_failure_holds_its_error_message():
    [result] = upload_results(NAME, DATA, ERROR)

    assert result == Failed(NAME, DATA, "RuntimeError: model exploded")


def test_one_result_per_card_in_the_order_found():
    [first, second] = upload_results(NAME, DATA, _recognition(_card("p1"), _card("p2")))

    assert isinstance(first, Recognized) and isinstance(second, Recognized)
    assert first.ranked[0].printing.uuid == "p1"
    assert second.ranked[0].printing.uuid == "p2"


def test_the_uploaded_filename_is_the_name_given():
    [result] = upload_results("card.png", DATA, _recognition(_card()))

    assert result.uploaded_filename == "card.png"


def test_a_recognized_result_holds_the_card_image_as_jpeg():
    [result] = upload_results(NAME, DATA, _recognition(_card(value=3)))

    assert isinstance(result, Recognized)
    decoded = cv2.imdecode(
        np.frombuffer(result.card_image_jpeg, np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None
    assert decoded.shape == card_image().shape
    assert abs(int(decoded[0, 0, 0]) - 3) <= 2


@pytest.mark.parametrize("recognition", [_recognition(_card()), NO_CARD, ERROR])
def test_every_kind_of_result_holds_the_query_image(recognition):
    [result] = upload_results(NAME, DATA, recognition)

    assert result.query_image == DATA


def test_a_recognized_result_holds_the_readings_debug_artefacts():
    debug = ReadingDebug(
        annotated_image=None, texts=["M10·EN"], scores=[0.9], boxes=[(0, 0, 9, 9)]
    )
    card = CardResult(
        card_image=card_image(),
        ranked=[RankedPrinting(printing("p0"), 100.0)],
        reading=Reading(name="Card p0", debug=debug),
    )

    [result] = upload_results(NAME, DATA, _recognition(card))

    assert isinstance(result, Recognized)
    assert result.reading_debug is debug


def test_csv_export_has_one_row_per_result_with_the_best_printing():
    results = upload_results(NAME, DATA, _recognition(_card())) + upload_results(
        NAME, DATA, ERROR
    )

    csv = export_bytes(results, "CSV")

    df = pd.read_csv(io.BytesIO(csv), dtype=str, keep_default_na=False)
    assert list(df.columns) == [
        "Uploaded Filename",
        "Status",
        "Card Name",
        "UUID",
        "Score",
        "Set Code",
        "Collector Number",
        "Base Set Size",
        "Artist",
        "Scryfall URL",
    ]
    assert df.to_dict("records") == [
        {
            "Uploaded Filename": "card.jpg",
            "Status": "success",
            "Card Name": "Card p0",
            "UUID": "p0",
            "Score": "100.0",
            "Set Code": "TST",
            "Collector Number": "1",
            "Base Set Size": "100",
            "Artist": "Artist",
            "Scryfall URL": "",
        },
        {
            "Uploaded Filename": "card.jpg",
            "Status": "error",
            "Card Name": "",
            "UUID": "",
            "Score": "",
            "Set Code": "",
            "Collector Number": "",
            "Base Set Size": "",
            "Artist": "",
            "Scryfall URL": "",
        },
    ]


def test_excel_export_has_one_row_per_result_with_the_best_printing():
    results = upload_results(NAME, DATA, _recognition(_card())) + upload_results(
        NAME, DATA, ERROR
    )

    excel = export_bytes(results, "Excel")

    df = pd.read_excel(io.BytesIO(excel), sheet_name="Results")
    assert list(df.columns) == [
        "Uploaded Filename",
        "Status",
        "Card Name",
        "UUID",
        "Score",
        "Set Code",
        "Collector Number",
        "Base Set Size",
        "Artist",
        "Scryfall URL",
    ]
    assert df["Status"].tolist() == ["success", "error"]
    assert df["Card Name"][0] == "Card p0"
    assert df["Score"][0] == 100.0
    assert df["Card Name"].isna()[1]


def test_an_unknown_export_format_fails_with_a_message_for_the_download_button():
    results = upload_results(NAME, DATA, _recognition(_card()))

    with pytest.raises(RuntimeError, match="Export failed"):
        export_bytes(results, "PDF")


def test_json_export_has_one_item_per_result_with_the_best_printing():
    results = (
        upload_results(NAME, DATA, _recognition(_card()))
        + upload_results(NAME, DATA, NO_CARD)
        + upload_results(NAME, DATA, ERROR)
    )

    items = json.loads(export_bytes(results, "JSON"))

    no_printing = {
        "card_name": None,
        "uuid": None,
        "score": None,
        "set_code": None,
        "collector_number": None,
        "base_set_size": None,
        "artist": None,
        "scryfall_url": None,
    }
    assert items == [
        {
            "uploaded_filename": "card.jpg",
            "status": "success",
            "card_name": "Card p0",
            "uuid": "p0",
            "score": 100.0,
            "set_code": "TST",
            "collector_number": "1",
            "base_set_size": "100",
            "artist": "Artist",
            "scryfall_url": None,
            "error_message": None,
        },
        {
            "uploaded_filename": "card.jpg",
            "status": "not_recognized",
            **no_printing,
            "error_message": None,
        },
        {
            "uploaded_filename": "card.jpg",
            "status": "error",
            **no_printing,
            "error_message": "RuntimeError: model exploded",
        },
    ]
    assert list(items[0]) == [
        "uploaded_filename",
        "status",
        "card_name",
        "uuid",
        "score",
        "set_code",
        "collector_number",
        "base_set_size",
        "artist",
        "scryfall_url",
        "error_message",
    ]
