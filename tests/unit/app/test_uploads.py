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
    results_json,
    results_table,
    upload_results,
)
from mtg_recognition.engine.ocr.reading import Reading, ReadingDebug
from mtg_recognition.engine.pipeline import RecognitionResult, SegmentResult
from mtg_recognition.engine.ranking import RankedPrinting
from tests.unit.engine.builders import card_image, printing

ERROR = RecognitionResult(segments=[], error="RuntimeError: model exploded")
NO_CARD = RecognitionResult(segments=[], error=None)


def _segment(uuid: str = "p0", value: int = 3) -> SegmentResult:
    return SegmentResult(
        warped_image=card_image(value),
        ranked=[RankedPrinting(printing(uuid), 100.0)],
        reading=None,
    )


def _recognized(*segments: SegmentResult) -> RecognitionResult:
    return RecognitionResult(segments=list(segments), error=None)


@pytest.fixture
def upload() -> Upload:
    return Upload(name="card.jpg", data=b"uploaded bytes")


def test_a_recognized_card_is_a_success(upload):
    [result] = upload_results(upload, _recognized(_segment()))

    assert result.status == UploadStatus.SUCCESS
    assert result.error_message is None


def test_an_image_without_a_card_is_no_match(upload):
    [result] = upload_results(upload, NO_CARD)

    assert result.status == UploadStatus.NO_MATCH


def test_a_card_without_candidates_is_no_match(upload):
    segment = SegmentResult(warped_image=card_image(), ranked=[], reading=None)

    [result] = upload_results(upload, _recognized(segment))

    assert result.status == UploadStatus.NO_MATCH


def test_a_recognition_failure_is_an_error(upload):
    [result] = upload_results(upload, ERROR)

    assert result.status == UploadStatus.ERROR
    assert result.error_message == "RuntimeError: model exploded"


def test_one_result_per_card_found_in_segment_order(upload):
    results = upload_results(upload, _recognized(_segment("p1"), _segment("p2")))

    assert [r.status for r in results] == [UploadStatus.SUCCESS, UploadStatus.SUCCESS]
    assert [r.ranked[0].printing.uuid for r in results] == ["p1", "p2"]


def test_the_uploaded_filename_is_the_name_of_the_upload():
    png = Upload(name="card.png", data=b"png bytes")

    [result] = upload_results(png, _recognized(_segment()))

    assert result.uploaded_filename == "card.png"


def test_a_result_carries_the_original_upload_and_the_warped_card_as_jpeg(upload):
    [result] = upload_results(upload, _recognized(_segment(value=3)))

    assert result.original_image == upload.data
    assert result.warped_image_jpeg is not None
    warped = cv2.imdecode(
        np.frombuffer(result.warped_image_jpeg, np.uint8), cv2.IMREAD_COLOR
    )
    assert warped.shape == card_image().shape
    assert abs(int(warped[0, 0, 0]) - 3) <= 2


@pytest.mark.parametrize("recognition", [NO_CARD, ERROR])
def test_a_result_without_a_card_still_carries_the_original_upload(upload, recognition):
    [result] = upload_results(upload, recognition)

    assert result.original_image == upload.data
    assert result.warped_image_jpeg is None


def test_a_result_carries_the_readings_debug_artefacts(upload):
    debug = ReadingDebug(
        annotated_image=None, texts=["M10·EN"], scores=[0.9], boxes=[(0, 0, 9, 9)]
    )
    segment = SegmentResult(
        warped_image=card_image(),
        ranked=[RankedPrinting(printing("p0"), 100.0)],
        reading=Reading(name="Card p0", debug=debug),
    )

    [result] = upload_results(upload, _recognized(segment))

    assert result.reading_debug is debug


def test_best_is_the_top_ranked_entry(upload):
    segment = SegmentResult(
        warped_image=card_image(),
        ranked=[
            RankedPrinting(printing("p0"), 100.0),
            RankedPrinting(printing("p1"), 40.0),
        ],
        reading=None,
    )

    [result] = upload_results(upload, _recognized(segment))

    assert result.best is not None
    assert (result.best.printing, result.best.score) == (printing("p0"), 100.0)


def test_best_is_none_without_a_match(upload):
    [result] = upload_results(upload, NO_CARD)

    assert result.best is None


def test_csv_export_has_one_row_per_result_with_the_best_printing(upload):
    results = upload_results(upload, _recognized(_segment())) + upload_results(
        upload, ERROR
    )

    csv = results_table(results).to_csv(index=False)

    df = pd.read_csv(io.StringIO(csv), dtype=str, keep_default_na=False)
    assert list(df.columns) == [
        "Uploaded Filename",
        "Status",
        "Card Name",
        "UUID",
        "Score",
        "Set Code",
        "Card Number",
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
            "Card Number": "1",
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
            "Card Number": "",
            "Base Set Size": "",
            "Artist": "",
            "Scryfall URL": "",
        },
    ]


def test_json_export_has_one_item_per_result_with_the_best_printing(upload):
    results = (
        upload_results(upload, _recognized(_segment()))
        + upload_results(upload, NO_CARD)
        + upload_results(upload, ERROR)
    )

    items = json.loads(results_json(results))

    no_printing = {
        "card_name": None,
        "uuid": None,
        "score": None,
        "set_code": None,
        "card_number": None,
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
            "card_number": "1",
            "base_set_size": "100",
            "artist": "Artist",
            "scryfall_url": None,
            "error_message": None,
        },
        {
            "uploaded_filename": "card.jpg",
            "status": "no_match",
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
        "card_number",
        "base_set_size",
        "artist",
        "scryfall_url",
        "error_message",
    ]
