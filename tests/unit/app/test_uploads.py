"""Upload results built from hand-made recognition results."""

import io
import json
from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import ExifTags, Image

from mtg_recognition.app.uploads import (
    Failed,
    NotRecognized,
    Recognized,
    export_bytes,
    scryfall_card_url,
    upload_results,
)
from mtg_recognition.engine.pipeline import CardResult, RecognitionResult
from mtg_recognition.engine.ranking import RankedPrinting
from tests.unit.engine.builders import card_image, printing, reading

NAME = "card.jpg"
# A phone photo's size, so the display copy is smaller than what was uploaded.
DATA = cv2.imencode(".jpg", np.full((3000, 2000, 3), 128, np.uint8))[1].tobytes()
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

    assert isinstance(result, NotRecognized)


def test_a_recognition_failure_holds_its_error_message():
    [result] = upload_results(NAME, DATA, ERROR)

    assert isinstance(result, Failed)
    assert result.error_message == "RuntimeError: model exploded"


def test_one_result_per_card_in_the_order_found():
    [first, second] = upload_results(NAME, DATA, _recognition(_card("p1"), _card("p2")))

    assert isinstance(first, Recognized) and isinstance(second, Recognized)
    assert first.ranked[0].printing.uuid == "p1"
    assert second.ranked[0].printing.uuid == "p2"


@pytest.mark.parametrize("recognition", [_recognition(_card()), NO_CARD, ERROR])
def test_the_uploaded_filename_is_the_name_given(recognition):
    [result] = upload_results("card.png", DATA, recognition)

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
def test_every_kind_of_result_holds_the_query_image_as_a_jpeg_at_most_800_pixels(
    recognition,
):
    [result] = upload_results(NAME, DATA, recognition)

    decoded = cv2.imdecode(
        np.frombuffer(result.query_image_jpeg, np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded.shape == (800, 533, 3)


@pytest.mark.parametrize("recognition", [_recognition(_card()), NO_CARD, ERROR])
def test_a_query_image_that_cannot_be_decoded_for_display_leaves_none(recognition):
    [result] = upload_results(NAME, b"not an image", recognition)

    assert result.query_image_jpeg is None


def test_the_query_image_is_turned_upright_by_its_exif_orientation():
    # a landscape photo whose EXIF says to rotate it 90° to show it upright
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = 6
    buffer = io.BytesIO()
    Image.new("RGB", (1200, 900)).save(buffer, "JPEG", exif=exif)

    [result] = upload_results(NAME, buffer.getvalue(), NO_CARD)

    decoded = cv2.imdecode(
        np.frombuffer(result.query_image_jpeg, np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded.shape == (800, 600, 3)


def test_a_recognized_result_holds_the_reading():
    card_reading = reading("Card p0", ["M10·EN"])
    card = CardResult(
        card_image=card_image(),
        ranked=[RankedPrinting(printing("p0"), 100.0)],
        reading=card_reading,
    )

    [result] = upload_results(NAME, DATA, _recognition(card))

    assert isinstance(result, Recognized)
    assert result.reading is card_reading


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
        "Image URL",
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
            "Image URL": "",
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
            "Image URL": "",
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
        "Image URL",
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
        "image_url": None,
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
            "image_url": None,
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
        "image_url",
        "error_message",
    ]


def test_a_card_whose_ocr_failed_holds_the_ocr_error():
    card = CardResult(
        card_image=card_image(),
        ranked=[RankedPrinting(printing("p0"), 0.0)],
        reading=None,
        ocr_error="RuntimeError: OCR exploded",
    )

    [result] = upload_results(NAME, DATA, _recognition(card))

    assert isinstance(result, Recognized)
    assert result.ocr_error == "RuntimeError: OCR exploded"


def test_the_scryfall_card_page_is_named_by_the_image_url_s_file_name():
    image_url = (
        "https://cards.scryfall.io/normal/front/f/6/"
        "f6555d1f-d4cf-41f7-99d3-88fd53e75457.jpg?1562550034"
    )

    assert scryfall_card_url(image_url) == (
        "https://scryfall.com/card/f6555d1f-d4cf-41f7-99d3-88fd53e75457"
    )


def test_export_holds_the_scryfall_card_page_and_the_image_url():
    image_url = "https://cards.scryfall.io/normal/front/a/b/abc-123.jpg"
    card = CardResult(
        card_image=card_image(),
        ranked=[RankedPrinting(replace(printing("p0"), image_url=image_url), 100.0)],
        reading=None,
    )
    results = upload_results(NAME, DATA, _recognition(card))

    [item] = json.loads(export_bytes(results, "JSON"))

    assert item["scryfall_url"] == "https://scryfall.com/card/abc-123"
    assert item["image_url"] == image_url
