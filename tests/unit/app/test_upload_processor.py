"""UploadProcessor over a Pipeline built from fake stages."""

import cv2
import pytest

from mtg_recognition.app.upload_processor import UploadProcessor, UploadStatus
from mtg_recognition.catalog import RankedPrinting
from tests.unit.engine.fakes import FakeSegmentor, card_image, make_pipeline, printing


class BrokenSegmentor(FakeSegmentor):
    def segment(self, img):
        raise RuntimeError("model exploded")


@pytest.fixture
def image_path(tmp_path) -> str:
    path = tmp_path / "card.jpg"
    cv2.imwrite(str(path), card_image())
    return str(path)


def _processor(segmentor: FakeSegmentor, k: int = 2) -> UploadProcessor:
    return UploadProcessor(
        make_pipeline(segmentor, k_phash_candidates=k, k_final_results=k)
    )


def test_a_recognized_card_is_a_success(image_path):
    [result] = _processor(FakeSegmentor([card_image(3)])).process([image_path])

    assert result.status == UploadStatus.SUCCESS
    assert result.error_message is None


def test_an_image_without_a_card_is_no_match(image_path):
    [result] = _processor(FakeSegmentor([])).process([image_path])

    assert result.status == UploadStatus.NO_MATCH


def test_a_recognition_failure_is_an_error(image_path):
    [result] = _processor(BrokenSegmentor([])).process([image_path])

    assert result.status == UploadStatus.ERROR
    assert result.error_message == "RuntimeError: model exploded"


def test_an_unreadable_file_is_an_error(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image")

    [result] = _processor(FakeSegmentor([])).process([str(path)])

    assert result.status == UploadStatus.ERROR


def test_one_result_per_card_found(image_path):
    results = _processor(FakeSegmentor([card_image(1), card_image(2)]), k=1).process(
        [image_path]
    )

    assert [r.status for r in results] == [UploadStatus.SUCCESS, UploadStatus.SUCCESS]
    assert [r.ranked[0].printing.uuid for r in results] == ["p1", "p2"]


def test_csv_export_has_one_row_per_result_with_the_best_printing(image_path, tmp_path):
    import pandas as pd

    processor = _processor(FakeSegmentor([card_image(3)]))
    results = processor.process([image_path]) + _processor(BrokenSegmentor([])).process(
        [image_path]
    )
    csv_path = tmp_path / "results.csv"

    processor.export_to_csv(results, str(csv_path))

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
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
            "Score": "90.0",
            "Set Code": "TST",
            "Card Number": "1",
            "Base Set Size": "100",
            "Artist": "",
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


def test_json_export_has_one_item_per_result_with_the_best_printing(
    image_path, tmp_path
):
    import json

    processor = _processor(FakeSegmentor([card_image(3)]))
    results = (
        processor.process([image_path])
        + _processor(FakeSegmentor([])).process([image_path])
        + _processor(BrokenSegmentor([])).process([image_path])
    )
    json_path = tmp_path / "results.json"

    processor.export_to_json(results, str(json_path))

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
    assert json.loads(json_path.read_text()) == [
        {
            "uploaded_filename": "card.jpg",
            "status": "success",
            "card_name": "Card p0",
            "uuid": "p0",
            "score": 90.0,
            "set_code": "TST",
            "card_number": "1",
            "base_set_size": "100",
            "artist": None,
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
    assert list(json.loads(json_path.read_text())[0]) == [
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


def test_best_is_the_top_ranked_entry(image_path):
    [result] = _processor(FakeSegmentor([card_image(3)])).process([image_path])

    assert result.best == RankedPrinting(printing("p0"), 90.0)


def test_best_is_none_without_a_match(image_path):
    [result] = _processor(FakeSegmentor([])).process([image_path])

    assert result.best is None
