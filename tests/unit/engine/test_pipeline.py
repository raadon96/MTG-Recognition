"""Pipeline's k validation, recognition over fake models and the committed
reference data, and the `mtg recognize` CLI over a stubbed Pipeline.

Recognition with the real models is covered end to end in tests/regression/engine.
"""

import json
from dataclasses import replace

import cv2
import pytest
from click.testing import CliRunner

from mtg_recognition.engine import pipeline as pipeline_module
from mtg_recognition.engine.ocr.reading import ReadingDebug
from mtg_recognition.engine.pipeline import Pipeline, RecognitionResult, SegmentResult
from mtg_recognition.engine.ranking import RankedPrinting
from mtg_recognition.engine.segmentation.seg_utils import (
    CardSegment,
    ImageSegmentationResult,
)
from tests.unit.engine.builders import card_image, printing, reading_of


@pytest.mark.parametrize(("k_phash", "k_final"), [(5, 6), (5, 0), (0, 0)])
def test_invalid_k_parameters_raise(k_phash, k_final):
    with pytest.raises(ValueError):
        Pipeline(k_phash_candidates=k_phash, k_final_results=k_final)


# --- Pipeline.recognize over fake models ---


class OneCardSegmentor:
    """Finds one card: the whole image."""

    def segment(self, img):
        return ImageSegmentationResult(segments=[CardSegment(warped_image=img)])


class FailingReader:
    def read(self, card_image, *, debug=False):
        raise RuntimeError("paddle exploded")


class P0Reader:
    """Reads every card as the printing "p0"."""

    def read(self, card_image, *, debug=False):
        return reading_of(printing("p0"))


def _pipeline(monkeypatch, reader) -> Pipeline:
    monkeypatch.setattr(pipeline_module, "YoloCardSegmentor", OneCardSegmentor)
    monkeypatch.setattr(pipeline_module, "PaddleCardReader", reader)
    return Pipeline(k_final_results=3)


def test_an_ocr_failure_is_recorded_and_scores_every_candidate_0(monkeypatch):
    pipeline = _pipeline(monkeypatch, FailingReader)

    result = pipeline.recognize(card_image())

    assert result.error is None
    [segment] = result.segments
    assert segment.ocr_error == "RuntimeError: paddle exploded"
    assert segment.reading is None
    assert len(segment.ranked) == 3
    assert all(r.score == 0.0 for r in segment.ranked)


def test_a_successful_reading_is_no_ocr_fallback(monkeypatch):
    pipeline = _pipeline(monkeypatch, P0Reader)

    [segment] = pipeline.recognize(card_image()).segments

    assert segment.ocr_error is None
    assert segment.reading == reading_of(printing("p0"))


# --- Pipeline CLI ---

DEBUG = ReadingDebug(
    annotated_image=None,
    texts=["Card p0"],
    scores=[0.99],
    boxes=[(0, 0, 10, 10)],
)
ONE_CARD = RecognitionResult(
    segments=[
        SegmentResult(
            warped_image=card_image(3),
            ranked=[
                RankedPrinting(printing("p0"), 100.0),
                RankedPrinting(printing("p3"), 40.0),
            ],
            reading=replace(reading_of(printing("p0")), debug=DEBUG),
        )
    ],
    error=None,
)
ERROR = RecognitionResult(segments=[], error="RuntimeError: model exploded")


def _run_cli(monkeypatch, tmp_path, result, filename, *args, built=None):
    class StubPipeline:
        def __init__(self, k_phash_candidates, k_final_results):
            if built is not None:
                built.append((k_phash_candidates, k_final_results))

        def recognize(self, image, debug=False):
            return result

    image_path = tmp_path / filename
    cv2.imwrite(str(image_path), card_image())
    monkeypatch.setattr(pipeline_module, "Pipeline", StubPipeline)
    return CliRunner().invoke(pipeline_module.main, ["-i", str(image_path), *args])


def test_cli_writes_ranked_printings_to_the_output_json(monkeypatch, tmp_path):
    output = tmp_path / "out.json"
    built = []

    result = _run_cli(
        monkeypatch,
        tmp_path,
        ONE_CARD,
        "p0.jpg",
        "--k-phash",
        "7",
        "-k",
        "2",
        "-o",
        str(output),
        built=built,
    )

    assert result.exit_code == 0, result.output
    assert built == [(7, 2)]
    [image] = json.loads(output.read_text())
    [segment] = image["segments"]
    assert segment["segment_idx"] == 0
    assert segment["success"] is None  # "p0" is not a UUID
    first, second = segment["ranked"]
    assert first == {
        "rank": 1,
        "name": "Card p0",
        "uuid": "p0",
        "setCode": "TST",
        "number": "1",
        "score": 100.0,
    }
    assert (second["rank"], second["uuid"]) == (2, "p3")
    assert second["score"] < 100.0
    assert segment["debug"] == {
        "raw_texts": ["Card p0"],
        "raw_scores": [0.99],
        "raw_boxes": [[0, 0, 10, 10]],
    }
    assert segment["ocr_error"] is None


def test_cli_writes_the_ocr_fallback_to_the_output_json(monkeypatch, tmp_path):
    output = tmp_path / "out.json"
    fallback = RecognitionResult(
        segments=[
            SegmentResult(
                warped_image=card_image(3),
                ranked=[RankedPrinting(printing("p0"), 0.0)],
                reading=None,
                ocr_error="RuntimeError: paddle exploded",
            )
        ],
        error=None,
    )

    result = _run_cli(monkeypatch, tmp_path, fallback, "p0.jpg", "-o", str(output))

    assert result.exit_code == 0, result.output
    [image] = json.loads(output.read_text())
    [segment] = image["segments"]
    assert segment["ocr_error"] == "RuntimeError: paddle exploded"


def test_cli_reports_an_error_without_a_segment_count(monkeypatch, tmp_path):
    result = _run_cli(monkeypatch, tmp_path, ERROR, "card.jpg")

    assert result.exit_code == 0
    assert "ERROR: RuntimeError: model exploded" in result.output
    assert "segment(s) detected" not in result.output
