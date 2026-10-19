"""Pipeline.recognize with fake segmentor, pHash matcher and OCR plus the in-memory
catalog.
"""

import numpy as np
import pytest

from mtg_recognition.catalog import RankedPrinting
from tests.unit.engine.fakes import (
    FakeOCR,
    FakePHashMatcher,
    FakeSegmentor,
    card_image,
    make_pipeline,
    printing,
)


def test_no_card_found_gives_no_segments_and_no_error():
    result = make_pipeline(FakeSegmentor([])).recognize(card_image())

    assert result.segments == []
    assert result.error is None


def test_one_card_is_matched_by_phash_then_reranked_by_ocr():
    warped = card_image(3)

    result = make_pipeline(
        FakeSegmentor([warped]), k_phash_candidates=2, k_final_results=2
    ).recognize(card_image())

    assert result.error is None
    [segment] = result.segments
    assert segment.warped_image is warped
    # pHash ranks p3 then p0; the fake OCR reverses that order.
    assert segment.ranked == [
        RankedPrinting(printing("p0"), 90.0),
        RankedPrinting(printing("p3"), 89.0),
    ]


def test_several_cards_give_one_segment_each_in_segmentation_order():
    segmentor = FakeSegmentor([card_image(1), card_image(2), card_image(3)])

    result = make_pipeline(
        segmentor, k_phash_candidates=1, k_final_results=1
    ).recognize(card_image())

    assert [[r.printing for r in s.ranked] for s in result.segments] == [
        [printing("p1")],
        [printing("p2")],
        [printing("p3")],
    ]


def test_k_parameters_are_passed_to_phash_and_bound_the_results():
    phash, ocr = FakePHashMatcher(), FakeOCR()
    pipeline = make_pipeline(
        FakeSegmentor([card_image(5)]),
        k_phash_candidates=7,
        k_final_results=3,
        phash_matcher=phash,
        ocr=ocr,
    )

    [segment] = pipeline.recognize(card_image()).segments

    assert phash.ks == [7]
    assert ocr.ks == [3]
    assert len(segment.ranked) == 3


def test_results_are_truncated_to_k_even_if_ocr_returns_more():
    class AllCandidatesOCR(FakeOCR):
        def find_best_matches(
            self, query_img, candidates, k=10, thresh=0.0, debug=False
        ):
            return [RankedPrinting(c, 0.0) for c in candidates], None

    pipeline = make_pipeline(
        FakeSegmentor([card_image()]),
        k_phash_candidates=5,
        k_final_results=2,
        ocr=AllCandidatesOCR(),
    )

    [segment] = pipeline.recognize(card_image()).segments

    assert len(segment.ranked) == 2


def test_candidates_missing_from_the_catalog_are_dropped_with_their_scores():
    phash = FakePHashMatcher(index=["p4", "not-in-catalog", "p5"])
    pipeline = make_pipeline(
        FakeSegmentor([card_image(4)]),
        k_phash_candidates=3,
        k_final_results=2,
        phash_matcher=phash,
    )

    [segment] = pipeline.recognize(card_image()).segments

    assert [r.printing for r in segment.ranked] == [printing("p5"), printing("p4")]


def test_the_lone_candidate_left_in_the_catalog_is_scored_by_ocr():
    # pHash ranks "not-in-catalog" first (1.0) and p5 second (0.99);
    # only p5 survives, and OCR scores it.
    phash = FakePHashMatcher(index=["not-in-catalog", "p5"])
    pipeline = make_pipeline(
        FakeSegmentor([card_image(7)]),
        k_phash_candidates=2,
        k_final_results=1,
        phash_matcher=phash,
    )

    [segment] = pipeline.recognize(card_image()).segments

    assert segment.ranked == [RankedPrinting(printing("p5"), 90.0)]


def test_no_candidate_in_the_catalog_gives_an_empty_ranking_without_ocr():
    phash, ocr = FakePHashMatcher(index=["not-in-catalog"]), FakeOCR()
    pipeline = make_pipeline(
        FakeSegmentor([card_image()]), phash_matcher=phash, ocr=ocr
    )

    [segment] = pipeline.recognize(card_image()).segments

    assert segment.ranked == []
    assert ocr.ks == []


def test_a_single_phash_candidate_is_scored_by_ocr():
    ocr = FakeOCR()
    pipeline = make_pipeline(
        FakeSegmentor([card_image(6)]), k_phash_candidates=1, k_final_results=1, ocr=ocr
    )

    [segment] = pipeline.recognize(card_image()).segments

    assert segment.ranked == [RankedPrinting(printing("p6"), 90.0)]
    assert ocr.ks == [1]


def test_an_exception_while_processing_becomes_the_error():
    class BrokenSegmentor(FakeSegmentor):
        def segment(self, img):
            raise RuntimeError("model exploded")

    result = make_pipeline(BrokenSegmentor([])).recognize(card_image())

    assert result.segments == []
    assert result.error == "RuntimeError: model exploded"


def test_large_images_are_downscaled_before_segmentation():
    segmentor = FakeSegmentor([])

    make_pipeline(segmentor).recognize(np.zeros((4000, 3000, 3), dtype=np.uint8))

    assert segmentor.inputs[0].shape == (2048, 1536, 3)


def test_small_images_reach_the_segmentor_unchanged():
    segmentor = FakeSegmentor([])
    image = card_image()

    make_pipeline(segmentor).recognize(image)

    assert segmentor.inputs[0] is image


@pytest.mark.parametrize(("k_phash", "k_final"), [(5, 6), (5, 0), (0, 0)])
def test_invalid_k_parameters_raise(k_phash, k_final):
    with pytest.raises(ValueError):
        make_pipeline(
            FakeSegmentor([]), k_phash_candidates=k_phash, k_final_results=k_final
        )


def test_missing_phash_index_raises():
    phash = FakePHashMatcher()
    phash.index_loaded = False

    with pytest.raises(ValueError, match="pHash index"):
        make_pipeline(FakeSegmentor([]), phash_matcher=phash)


# --- Pipeline CLI ---


def _run_cli(monkeypatch, tmp_path, segmentor, filename, *args):
    import cv2
    from click.testing import CliRunner

    from mtg_recognition.engine import pipeline as pipeline_module

    image_path = tmp_path / filename
    cv2.imwrite(str(image_path), card_image())
    monkeypatch.setattr(
        pipeline_module,
        "Pipeline",
        lambda k_phash_candidates, k_final_results: make_pipeline(
            segmentor,
            k_phash_candidates=k_phash_candidates,
            k_final_results=k_final_results,
        ),
    )
    return CliRunner().invoke(pipeline_module.main, ["-i", str(image_path), *args])


def test_cli_writes_ranked_printings_to_the_output_json(monkeypatch, tmp_path):
    import json

    output = tmp_path / "out.json"

    result = _run_cli(
        monkeypatch,
        tmp_path,
        FakeSegmentor([card_image(3)]),
        "p0.jpg",
        "--k-phash",
        "2",
        "-k",
        "2",
        "-o",
        str(output),
    )

    assert result.exit_code == 0, result.output
    [image] = json.loads(output.read_text())
    [segment] = image["segments"]
    assert segment["segment_idx"] == 0
    assert segment["success"] is None  # "p0" is not a UUID
    assert segment["ranked"] == [
        {
            "rank": 1,
            "name": "Card p0",
            "uuid": "p0",
            "setCode": "TST",
            "number": "1",
            "score": 90.0,
        },
        {
            "rank": 2,
            "name": "Card p3",
            "uuid": "p3",
            "setCode": "TST",
            "number": "1",
            "score": 89.0,
        },
    ]
    assert set(segment["debug"]) == {"raw_texts", "raw_scores", "raw_boxes"}


def test_cli_reports_an_error_without_a_segment_count(monkeypatch, tmp_path):
    class BrokenSegmentor(FakeSegmentor):
        def segment(self, img):
            raise RuntimeError("model exploded")

    result = _run_cli(monkeypatch, tmp_path, BrokenSegmentor([]), "card.jpg")

    assert result.exit_code == 0
    assert "ERROR: RuntimeError: model exploded" in result.output
    assert "segment(s) detected" not in result.output
