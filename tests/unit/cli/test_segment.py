"""`mtg segment` over a fake segmentor."""

import cv2
import numpy as np
from click.testing import CliRunner

from mtg_recognition.cli import cli
from mtg_recognition.cli import segment as segment_cli
from mtg_recognition.engine.segmentation.seg_utils import (
    SegmentationResult,
    SegmentedCard,
)
from tests.unit.engine.builders import card_image


class FakeSegmentor:
    """Finds the given card images in every query image."""

    def __init__(self, card_images: list[np.ndarray]):
        self.card_images = card_images
        self.inputs: list[np.ndarray] = []

    def segment(self, query_image: np.ndarray) -> SegmentationResult:
        self.inputs.append(query_image)
        return SegmentationResult(
            cards=[SegmentedCard(card_image=c) for c in self.card_images]
        )


def _run(monkeypatch, segmentor, input_path, output_dir):
    monkeypatch.setattr(segment_cli, "_build_segmentor", lambda mode: segmentor)
    return CliRunner().invoke(
        cli, ["segment", "-i", str(input_path), "-o", str(output_dir)]
    )


def test_an_unreadable_file_is_skipped_with_a_message(monkeypatch, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "broken.jpg").write_bytes(b"not an image")
    cv2.imwrite(str(images / "card.png"), card_image())

    result = _run(monkeypatch, FakeSegmentor([card_image(3)]), images, tmp_path / "out")

    assert result.exit_code == 0, result.output
    assert "unreadable query image, skipped: broken.jpg" in result.output
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["card.jpg"]


def test_images_are_prepared_like_for_recognition(monkeypatch, tmp_path):
    path = tmp_path / "large.png"
    cv2.imwrite(str(path), np.zeros((4000, 3000, 3), dtype=np.uint8))
    segmentor = FakeSegmentor([])

    result = _run(monkeypatch, segmentor, path, tmp_path / "out")

    assert result.exit_code == 0, result.output
    assert segmentor.inputs[0].shape == (2048, 1536, 3)
