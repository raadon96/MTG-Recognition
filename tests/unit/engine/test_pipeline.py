"""Pipeline.recognize with fake segmentor, pHash matcher and reader plus the in-memory
catalog; scoring is real.
"""

import numpy as np
import pytest

from mtg_recognition.catalog import InMemoryCardCatalog, Printing
from mtg_recognition.engine.ocr.reading import Reading
from tests.unit.engine.fakes import (
    FakePHashMatcher,
    FakeReader,
    FakeSegmentor,
    card_image,
    card_png,
    make_pipeline,
    printing,
    reading_of,
)


def test_no_card_found_gives_no_segments_and_no_error():
    result = make_pipeline(FakeSegmentor([])).recognize(card_image())

    assert result.segments == []
    assert result.error is None


def test_one_card_is_matched_by_phash_then_ranked_by_its_reading():
    warped = card_image(3)

    result = make_pipeline(
        FakeSegmentor([warped]), k_phash_candidates=2, k_final_results=2
    ).recognize(card_image())

    assert result.error is None
    [segment] = result.segments
    assert segment.warped_image is warped
    # pHash ranks p3 then p0; the card reads as p0.
    assert [r.printing for r in segment.ranked] == [printing("p0"), printing("p3")]
    assert segment.ranked[0].score == 100.0
    assert segment.ranked[1].score < 100.0


def test_a_reading_of_the_name_ranks_that_card_first():
    shock = Printing(
        uuid="shock", name="Shock", set_code="M19", number="156", base_set_size="280"
    )
    bolt = Printing(
        uuid="bolt",
        name="Lightning Bolt",
        set_code="M10",
        number="146",
        base_set_size="249",
    )
    pipeline = make_pipeline(
        FakeSegmentor([card_image()]),
        k_phash_candidates=2,
        k_final_results=2,
        phash_matcher=FakePHashMatcher(index=["shock", "bolt"]),
        reader=FakeReader(Reading(name="LIGHTNING BOLT")),
        catalog=InMemoryCardCatalog([shock, bolt]),
    )

    [segment] = pipeline.recognize(card_image()).segments

    assert [r.printing for r in segment.ranked] == [bolt, shock]


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
    phash = FakePHashMatcher()
    pipeline = make_pipeline(
        FakeSegmentor([card_image(5)]),
        k_phash_candidates=7,
        k_final_results=3,
        phash_matcher=phash,
    )

    [segment] = pipeline.recognize(card_image()).segments

    assert phash.ks == [7]
    assert len(segment.ranked) == 3


def test_candidates_missing_from_the_catalog_are_dropped():
    phash = FakePHashMatcher(index=["p4", "not-in-catalog", "p5"])
    pipeline = make_pipeline(
        FakeSegmentor([card_image(4)]),
        k_phash_candidates=3,
        k_final_results=2,
        phash_matcher=phash,
        reader=FakeReader(reading_of(printing("p5"))),
    )

    [segment] = pipeline.recognize(card_image()).segments

    assert [r.printing for r in segment.ranked] == [printing("p5"), printing("p4")]


def test_the_lone_candidate_left_in_the_catalog_is_scored():
    # pHash ranks "not-in-catalog" first and p5 second; only p5 survives.
    phash = FakePHashMatcher(index=["not-in-catalog", "p5"])
    pipeline = make_pipeline(
        FakeSegmentor([card_image(7)]),
        k_phash_candidates=2,
        k_final_results=1,
        phash_matcher=phash,
        reader=FakeReader(reading_of(printing("p5"))),
    )

    [segment] = pipeline.recognize(card_image()).segments

    assert [(r.printing, r.score) for r in segment.ranked] == [(printing("p5"), 100.0)]


def test_no_candidate_in_the_catalog_gives_an_empty_ranking_without_reading():
    phash, reader = FakePHashMatcher(index=["not-in-catalog"]), FakeReader()
    pipeline = make_pipeline(
        FakeSegmentor([card_image()]), phash_matcher=phash, reader=reader
    )

    [segment] = pipeline.recognize(card_image()).segments

    assert segment.ranked == []
    assert segment.reading is None
    assert reader.debugs == []


def test_a_single_phash_candidate_is_scored():
    pipeline = make_pipeline(
        FakeSegmentor([card_image(6)]), k_phash_candidates=1, k_final_results=1
    )

    [segment] = pipeline.recognize(card_image()).segments

    assert [r.printing for r in segment.ranked] == [printing("p6")]


@pytest.mark.parametrize("debug", [False, True])
def test_the_debug_flag_reaches_the_reader_and_the_reading_is_kept(debug):
    reading = Reading(name="Card p0")
    reader = FakeReader(reading)
    pipeline = make_pipeline(FakeSegmentor([card_image()]), reader=reader)

    [segment] = pipeline.recognize(card_image(), debug=debug).segments

    assert reader.debugs == [debug]
    assert segment.reading is reading


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


def test_an_image_file_path_is_recognized(tmp_path):
    path = tmp_path / "card.png"
    path.write_bytes(card_png(9))
    segmentor = FakeSegmentor([])

    result = make_pipeline(segmentor).recognize(path)

    assert result.error is None
    assert (segmentor.inputs[0] == card_image(9)).all()


def test_encoded_image_bytes_are_recognized():
    segmentor = FakeSegmentor([])

    result = make_pipeline(segmentor).recognize(card_png(9))

    assert result.error is None
    assert (segmentor.inputs[0] == card_image(9)).all()


def test_an_unreadable_source_becomes_the_error():
    segmentor = FakeSegmentor([])

    result = make_pipeline(segmentor).recognize(b"not an image")

    assert result.segments == []
    assert result.error is not None
    assert result.error.startswith("ValueError: ")
    assert segmentor.inputs == []


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
    assert set(segment["debug"]) == {"raw_texts", "raw_scores", "raw_boxes"}


def test_cli_reports_an_error_without_a_segment_count(monkeypatch, tmp_path):
    class BrokenSegmentor(FakeSegmentor):
        def segment(self, img):
            raise RuntimeError("model exploded")

    result = _run_cli(monkeypatch, tmp_path, BrokenSegmentor([]), "card.jpg")

    assert result.exit_code == 0
    assert "ERROR: RuntimeError: model exploded" in result.output
    assert "segment(s) detected" not in result.output
