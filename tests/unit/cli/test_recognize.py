"""`mtg recognize` over a stubbed Pipeline."""

import json
from dataclasses import replace

import cv2
from click.testing import CliRunner

from mtg_recognition.cli import cli
from mtg_recognition.cli import recognize as recognize_cli
from mtg_recognition.engine.ocr.reading import ReadingDebug
from mtg_recognition.engine.pipeline import CardResult, RecognitionResult
from mtg_recognition.engine.ranking import RankedPrinting
from tests.unit.engine.builders import card_image, printing, reading_of

DEBUG = ReadingDebug(
    annotated_image=None,
    texts=["Card p0"],
    scores=[0.99],
    boxes=[(0, 0, 10, 10)],
)
ONE_CARD = RecognitionResult(
    cards=[
        CardResult(
            card_image=card_image(3),
            ranked=[
                RankedPrinting(printing("p0"), 100.0),
                RankedPrinting(printing("p3"), 40.0),
            ],
            reading=replace(reading_of(printing("p0")), debug=DEBUG),
        )
    ],
    error=None,
)
ERROR = RecognitionResult(cards=[], error="RuntimeError: model exploded")


def _run_cli(monkeypatch, tmp_path, result, filename, *args, built=None):
    class StubPipeline:
        def __init__(self, k_phash_candidates, k_final_results):
            if built is not None:
                built.append((k_phash_candidates, k_final_results))

        def recognize(self, query_image, debug=False):
            return result

    image_path = tmp_path / filename
    cv2.imwrite(str(image_path), card_image())
    monkeypatch.setattr(recognize_cli, "Pipeline", StubPipeline)
    return CliRunner().invoke(cli, ["recognize", "-i", str(image_path), *args])


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
    [query_image] = json.loads(output.read_text())
    [card] = query_image["cards"]
    assert card["card_idx"] == 0
    assert card["success"] is None  # "p0" is not a UUID
    first, second = card["ranked"]
    assert first == {
        "rank": 1,
        "name": "Card p0",
        "uuid": "p0",
        "set_code": "TST",
        "collector_number": "1",
        "score": 100.0,
    }
    assert (second["rank"], second["uuid"]) == (2, "p3")
    assert second["score"] < 100.0
    assert card["debug"] == {
        "raw_texts": ["Card p0"],
        "raw_scores": [0.99],
        "raw_boxes": [[0, 0, 10, 10]],
    }
    assert card["ocr_error"] is None


def test_cli_writes_the_ocr_fallback_to_the_output_json(monkeypatch, tmp_path):
    output = tmp_path / "out.json"
    fallback = RecognitionResult(
        cards=[
            CardResult(
                card_image=card_image(3),
                ranked=[RankedPrinting(printing("p0"), 0.0)],
                reading=None,
                ocr_error="RuntimeError: paddle exploded",
            )
        ],
        error=None,
    )

    result = _run_cli(monkeypatch, tmp_path, fallback, "p0.jpg", "-o", str(output))

    assert result.exit_code == 0, result.output
    [query_image] = json.loads(output.read_text())
    [card] = query_image["cards"]
    assert card["ocr_error"] == "RuntimeError: paddle exploded"


def test_cli_reports_an_error_without_a_card_count(monkeypatch, tmp_path):
    result = _run_cli(monkeypatch, tmp_path, ERROR, "card.jpg")

    assert result.exit_code == 0
    assert "ERROR: RuntimeError: model exploded" in result.output
    assert "card(s) detected" not in result.output
