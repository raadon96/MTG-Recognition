"""Upload results built from hand-made recognition results."""

import io
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from mtg_recognition.app.uploads import (
    Upload,
    UploadStatus,
    export_bytes,
    upload_results,
)
from mtg_recognition.engine.ocr.reading import Reading, ReadingDebug
from mtg_recognition.engine.pipeline import CardResult, RecognitionResult
from mtg_recognition.engine.ranking import RankedPrinting
from tests.unit.engine.builders import card_image, printing

ERROR = RecognitionResult(cards=[], error="RuntimeError: model exploded")
NO_CARD = RecognitionResult(cards=[], error=None)


def _card(uuid: str = "p0", value: int = 3) -> CardResult:
    return CardResult(
        card_image=card_image(value),
        ranked=[RankedPrinting(printing(uuid), 100.0)],
        reading=None,
    )


def _recognized(*cards: CardResult) -> RecognitionResult:
    return RecognitionResult(cards=list(cards), error=None)


@pytest.fixture
def upload() -> Upload:
    return Upload(name="card.jpg", data=b"uploaded bytes")


def test_a_recognized_card_is_a_success(upload):
    [result] = upload_results(upload, _recognized(_card()))

    assert result.status == UploadStatus.SUCCESS
    assert result.error_message is None


def test_a_query_image_without_a_card_is_not_recognized(upload):
    [result] = upload_results(upload, NO_CARD)

    assert result.status == UploadStatus.NOT_RECOGNIZED


def test_a_card_without_candidates_is_not_recognized(upload):
    card = CardResult(card_image=card_image(), ranked=[], reading=None)

    [result] = upload_results(upload, _recognized(card))

    assert result.status == UploadStatus.NOT_RECOGNIZED


def test_a_recognition_failure_is_an_error(upload):
    [result] = upload_results(upload, ERROR)

    assert result.status == UploadStatus.ERROR
    assert result.error_message == "RuntimeError: model exploded"


def test_one_result_per_card_in_the_order_found(upload):
    results = upload_results(upload, _recognized(_card("p1"), _card("p2")))

    assert [r.status for r in results] == [UploadStatus.SUCCESS, UploadStatus.SUCCESS]
    assert [r.ranked[0].printing.uuid for r in results] == ["p1", "p2"]


def test_the_uploaded_filename_is_the_name_of_the_upload():
    png = Upload(name="card.png", data=b"png bytes")

    [result] = upload_results(png, _recognized(_card()))

    assert result.uploaded_filename == "card.png"


def test_a_result_holds_the_query_image_and_the_card_image_as_jpeg(upload):
    [result] = upload_results(upload, _recognized(_card(value=3)))

    assert result.query_image == upload.data
    assert result.card_image_jpeg is not None
    decoded = cv2.imdecode(
        np.frombuffer(result.card_image_jpeg, np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None
    assert decoded.shape == card_image().shape
    assert abs(int(decoded[0, 0, 0]) - 3) <= 2


@pytest.mark.parametrize("recognition", [NO_CARD, ERROR])
def test_a_result_without_a_card_still_holds_the_query_image(upload, recognition):
    [result] = upload_results(upload, recognition)

    assert result.query_image == upload.data
    assert result.card_image_jpeg is None


def test_a_result_holds_the_readings_debug_artefacts(upload):
    debug = ReadingDebug(
        annotated_image=None, texts=["M10·EN"], scores=[0.9], boxes=[(0, 0, 9, 9)]
    )
    card = CardResult(
        card_image=card_image(),
        ranked=[RankedPrinting(printing("p0"), 100.0)],
        reading=Reading(name="Card p0", debug=debug),
    )

    [result] = upload_results(upload, _recognized(card))

    assert result.reading_debug is debug


def test_best_is_the_top_ranked_entry(upload):
    card = CardResult(
        card_image=card_image(),
        ranked=[
            RankedPrinting(printing("p0"), 100.0),
            RankedPrinting(printing("p1"), 40.0),
        ],
        reading=None,
    )

    [result] = upload_results(upload, _recognized(card))

    assert result.best is not None
    assert (result.best.printing, result.best.score) == (printing("p0"), 100.0)


def test_best_is_none_with_an_empty_ranking(upload):
    [result] = upload_results(upload, NO_CARD)

    assert result.best is None


def test_csv_export_has_one_row_per_result_with_the_best_printing(upload):
    results = upload_results(upload, _recognized(_card())) + upload_results(
        upload, ERROR
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


def test_excel_export_has_one_row_per_result_with_the_best_printing(upload):
    results = upload_results(upload, _recognized(_card())) + upload_results(
        upload, ERROR
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


def test_an_unknown_export_format_fails_with_a_message_for_the_download_button(upload):
    results = upload_results(upload, _recognized(_card()))

    with pytest.raises(RuntimeError, match="Export failed"):
        export_bytes(results, "PDF")


def test_json_export_has_one_item_per_result_with_the_best_printing(upload):
    results = (
        upload_results(upload, _recognized(_card()))
        + upload_results(upload, NO_CARD)
        + upload_results(upload, ERROR)
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
