"""The benchmark's validation, scoring and comparison, on plain data."""

import pytest

from mtg_recognition.benchmark.results import (
    Flip,
    ImageOutcome,
    ImageScore,
    ManifestProblems,
    compare,
    read_results,
    score_image,
    summarise,
    validate_manifest,
    write_results,
)
from mtg_recognition.reference.catalog import Printing

BOLT = "11111111-1111-1111-1111-111111111111"
BOLT_REPRINT = "22222222-2222-2222-2222-222222222222"
GIANT = "33333333-3333-3333-3333-333333333333"


def printing(uuid: str, name: str = "Lightning Bolt") -> Printing:
    return Printing(
        uuid=uuid, name=name, set_code="M10", collector_number="1", base_set_size="1"
    )


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "photos").mkdir()
    (tmp_path / "photos" / "bolt.jpg").write_bytes(b"")
    return tmp_path


@pytest.mark.unit
class TestValidateManifest:
    def test_a_manifest_matching_the_images_and_catalog_has_no_problems(
        self, dataset
    ) -> None:
        problems = validate_manifest(
            dataset, {"photos/bolt.jpg": [BOLT]}, {BOLT: printing(BOLT)}
        )

        assert problems == ManifestProblems()

    def test_an_image_without_a_label_is_reported(self, dataset) -> None:
        problems = validate_manifest(dataset, {}, {})

        assert problems.unlabelled_images == ["photos/bolt.jpg"]

    def test_a_label_for_a_missing_file_is_reported(self, dataset) -> None:
        manifest = {"photos/bolt.jpg": [BOLT], "photos/gone.jpg": [BOLT]}

        problems = validate_manifest(dataset, manifest, {BOLT: printing(BOLT)})

        assert problems.missing_images == ["photos/gone.jpg"]

    def test_a_uuid_not_in_the_catalog_is_reported(self, dataset) -> None:
        problems = validate_manifest(dataset, {"photos/bolt.jpg": [BOLT]}, {})

        assert problems.unknown_uuids == [BOLT]

    def test_the_manifest_and_hidden_files_need_no_label(self, dataset) -> None:
        (dataset / "manifest.json").write_text("{}")
        (dataset / "photos" / ".DS_Store").write_bytes(b"")

        problems = validate_manifest(dataset, {"photos/bolt.jpg": [BOLT]}, {})

        assert problems.unlabelled_images == []


@pytest.mark.unit
class TestScoreImage:
    def test_the_labelled_printings_in_another_order_are_exact(self) -> None:
        label = [printing(BOLT), printing(GIANT, "Realm-Cloaked Giant // Cast Off")]
        recognised = [label[1], label[0]]

        assert score_image(label, recognised) == ImageScore(
            exact_printing=True, card_count=True, same_card_name=True
        )

    def test_a_missing_card_fails_the_count_and_the_printing(self) -> None:
        label = [printing(BOLT), printing(GIANT, "Realm-Cloaked Giant // Cast Off")]

        assert score_image(label, [label[0]]) == ImageScore(
            exact_printing=False, card_count=False, same_card_name=False
        )

    def test_an_extra_card_fails_the_count_and_the_printing(self) -> None:
        score = score_image([printing(BOLT)], [printing(BOLT), printing(BOLT)])

        assert score == ImageScore(
            exact_printing=False, card_count=False, same_card_name=False
        )

    def test_another_printing_of_the_card_has_the_same_card_name(self) -> None:
        score = score_image([printing(BOLT)], [printing(BOLT_REPRINT)])

        assert score == ImageScore(
            exact_printing=False, card_count=True, same_card_name=True
        )

    def test_a_card_without_candidates_is_counted_but_not_recognised(self) -> None:
        score = score_image([printing(BOLT)], [None])

        assert score == ImageScore(
            exact_printing=False, card_count=True, same_card_name=False
        )

    def test_a_query_image_without_cards_matches_an_empty_label(self) -> None:
        assert score_image([], []) == ImageScore(
            exact_printing=True, card_count=True, same_card_name=True
        )


def outcome(exact: bool, count: bool = True, same_name: bool = True) -> ImageOutcome:
    return ImageOutcome(
        score=ImageScore(exact, count, same_name),
        recognised=[BOLT],
        runtime_s=1.0,
        ocr_fallbacks=0,
    )


def results(**images: ImageOutcome):
    return summarise("abc123", "5.3.0", images)


@pytest.mark.unit
class TestCompare:
    def test_identical_outcomes_do_not_differ(self) -> None:
        comparison = compare(results(a=outcome(True)), results(a=outcome(True)))

        assert comparison.flips == []
        assert comparison.changed_totals == {}
        assert not comparison.differs

    def test_an_image_that_starts_passing_is_a_difference(self) -> None:
        comparison = compare(results(a=outcome(True)), results(a=outcome(False)))

        assert comparison.flips == [Flip("a", "exact_printing", passes=True)]
        assert comparison.changed_totals == {"exact_printing": (0, 1)}
        assert comparison.differs

    def test_an_image_that_stops_passing_is_a_difference(self) -> None:
        comparison = compare(
            results(a=outcome(True, count=False)), results(a=outcome(True))
        )

        assert comparison.flips == [Flip("a", "card_count", passes=False)]
        assert comparison.changed_totals == {"card_count": (1, 0)}
        assert comparison.differs

    def test_a_flip_in_same_card_name_is_reported_but_no_difference(self) -> None:
        comparison = compare(
            results(a=outcome(False, same_name=False)), results(a=outcome(False))
        )

        assert comparison.flips == [Flip("a", "same_card_name", passes=False)]
        assert not comparison.differs

    def test_a_new_query_image_is_a_difference(self) -> None:
        comparison = compare(
            results(a=outcome(True), b=outcome(True)), results(a=outcome(True))
        )

        assert comparison.added_images == ["b"]
        assert comparison.differs

    def test_a_removed_query_image_is_a_difference(self) -> None:
        comparison = compare(
            results(a=outcome(True)), results(a=outcome(True), b=outcome(True))
        )

        assert comparison.removed_images == ["b"]
        assert comparison.differs


@pytest.mark.unit
def test_written_results_read_back_unchanged(tmp_path) -> None:
    written = results(a=outcome(True), b=outcome(False, count=False))
    written.images["b"].recognised = [None]
    written.images["b"].error = "ValueError: unreadable"
    path = tmp_path / "out" / "latest.json"

    write_results(written, path)

    assert read_results(path) == written


@pytest.mark.unit
def test_totals_are_worked_out_from_the_entries_not_the_stored_totals(
    tmp_path,
) -> None:
    written = results(a=outcome(True), b=outcome(False, count=False))
    written.totals.exact_printing = 2
    written.totals.card_count = 0
    path = tmp_path / "latest.json"
    write_results(written, path)

    totals = read_results(path).totals

    assert (totals.exact_printing, totals.card_count) == (1, 1)
