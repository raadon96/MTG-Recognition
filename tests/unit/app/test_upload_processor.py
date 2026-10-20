"""UploadProcessor over a Pipeline built from fake stages."""

import tempfile

import cv2
import numpy as np
import pytest

from mtg_recognition.app.upload_processor import Upload, UploadProcessor, UploadStatus
from mtg_recognition.engine.ocr.reading import Reading, ReadingDebug
from tests.unit.engine.fakes import (
    FakeReader,
    FakeSegmentor,
    card_image,
    card_png,
    make_pipeline,
    printing,
)


class BrokenSegmentor(FakeSegmentor):
    def segment(self, img):
        raise RuntimeError("model exploded")


def _upload(name: str = "card.jpg", value: int = 0) -> Upload:
    return Upload(name=name, data=card_png(value))


@pytest.fixture
def upload() -> Upload:
    return _upload()


def _processor(
    segmentor: FakeSegmentor, k: int = 2, reader: FakeReader | None = None
) -> UploadProcessor:
    return UploadProcessor(
        make_pipeline(segmentor, k_phash_candidates=k, k_final_results=k, reader=reader)
    )


def test_a_recognized_card_is_a_success(upload):
    [result] = _processor(FakeSegmentor([card_image(3)])).process([upload])

    assert result.status == UploadStatus.SUCCESS
    assert result.error_message is None


def test_an_image_without_a_card_is_no_match(upload):
    [result] = _processor(FakeSegmentor([])).process([upload])

    assert result.status == UploadStatus.NO_MATCH


def test_a_recognition_failure_is_an_error(upload):
    [result] = _processor(BrokenSegmentor([])).process([upload])

    assert result.status == UploadStatus.ERROR
    assert result.error_message == "RuntimeError: model exploded"


def test_an_unreadable_upload_is_an_error():
    broken = Upload(name="broken.jpg", data=b"not an image")

    [result] = _processor(FakeSegmentor([])).process([broken])

    assert result.status == UploadStatus.ERROR
    assert result.error_message is not None
    assert result.uploaded_filename == "broken.jpg"


def test_one_result_per_card_found(upload):
    results = _processor(FakeSegmentor([card_image(1), card_image(2)]), k=1).process(
        [upload]
    )

    assert [r.status for r in results] == [UploadStatus.SUCCESS, UploadStatus.SUCCESS]
    assert [r.ranked[0].printing.uuid for r in results] == ["p1", "p2"]


def test_csv_export_has_one_row_per_result_with_the_best_printing(upload, tmp_path):
    import pandas as pd

    processor = _processor(FakeSegmentor([card_image(3)]))
    results = processor.process([upload]) + _processor(BrokenSegmentor([])).process(
        [upload]
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


def test_json_export_has_one_item_per_result_with_the_best_printing(upload, tmp_path):
    import json

    processor = _processor(FakeSegmentor([card_image(3)]))
    results = (
        processor.process([upload])
        + _processor(FakeSegmentor([])).process([upload])
        + _processor(BrokenSegmentor([])).process([upload])
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


def test_best_is_the_top_ranked_entry(upload):
    [result] = _processor(FakeSegmentor([card_image(3)])).process([upload])

    assert result.best is not None
    assert (result.best.printing, result.best.score) == (printing("p0"), 100.0)


def test_best_is_none_without_a_match(upload):
    [result] = _processor(FakeSegmentor([])).process([upload])

    assert result.best is None


def test_a_result_carries_the_readers_debug_artefacts(upload):
    debug = ReadingDebug(
        annotated_image=None, texts=["M10·EN"], scores=[0.9], boxes=[(0, 0, 9, 9)]
    )
    reader = FakeReader(Reading(name="Card p0", debug=debug))

    [result] = _processor(FakeSegmentor([card_image()]), reader=reader).process(
        [upload]
    )

    assert result.reading_debug is debug
    assert reader.debugs == [True]


def test_the_uploaded_filename_is_the_name_of_the_upload():
    results = _processor(FakeSegmentor([card_image(3)])).process(
        [_upload("card.png", value=1), _upload("card.jpg", value=2)]
    )

    assert [r.uploaded_filename for r in results] == ["card.png", "card.jpg"]


def test_uploads_with_the_same_stem_are_recognized_from_their_own_pixels():
    segmentor = FakeSegmentor([card_image(3)])

    UploadProcessor(make_pipeline(segmentor)).process(
        [_upload("card.png", value=1), _upload("card.jpg", value=2)]
    )

    assert [int(img[0, 0, 0]) for img in segmentor.inputs] == [1, 2]


def test_a_result_carries_the_original_upload_and_the_warped_card_as_jpeg(upload):
    [result] = _processor(FakeSegmentor([card_image(3)])).process([upload])

    assert result.original_image == upload.data
    assert result.warped_image_jpeg is not None
    warped = cv2.imdecode(
        np.frombuffer(result.warped_image_jpeg, np.uint8), cv2.IMREAD_COLOR
    )
    assert warped.shape == card_image().shape
    assert abs(int(warped[0, 0, 0]) - 3) <= 2


@pytest.mark.parametrize("segmentor", [FakeSegmentor([]), BrokenSegmentor([])])
def test_a_result_without_a_card_still_carries_the_original_upload(upload, segmentor):
    [result] = _processor(segmentor).process([upload])

    assert result.original_image == upload.data
    assert result.warped_image_jpeg is None


def test_no_temp_files_are_written(upload, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    _processor(FakeSegmentor([card_image(3)])).process([upload])

    assert list(tmp_path.iterdir()) == []
