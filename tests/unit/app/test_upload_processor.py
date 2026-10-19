"""UploadProcessor over a Pipeline built from fake stages."""

import cv2
import pytest

from mtg_recognition.app.upload_processor import UploadProcessor, UploadStatus
from tests.unit.engine.fakes import FakeSegmentor, card_image, make_pipeline


class BrokenSegmentor(FakeSegmentor):
    def segment(self, img):
        raise RuntimeError("model exploded")


@pytest.fixture
def image_path(tmp_path) -> str:
    path = tmp_path / "card.jpg"
    cv2.imwrite(str(path), card_image())
    return str(path)


def _processor(segmentor: FakeSegmentor, k: int = 2) -> UploadProcessor:
    return UploadProcessor(make_pipeline(segmentor, k_phash_candidates=k, k_final_results=k))


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
    results = _processor(FakeSegmentor([card_image(1), card_image(2)]), k=1).process([image_path])

    assert [r.status for r in results] == [UploadStatus.SUCCESS, UploadStatus.SUCCESS]
    assert [r.matches[0].uuid for r in results] == ["p1", "p2"]
