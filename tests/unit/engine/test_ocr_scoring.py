"""Scoring a card's reading against candidate printings."""

import pytest

from mtg_recognition.engine.ocr.reading import Reading
from mtg_recognition.engine.ocr.scoring import rank
from mtg_recognition.engine.ranking import FieldScore
from mtg_recognition.reference.catalog import Printing

BOLT_M10 = Printing(
    uuid="bolt-m10",
    name="Lightning Bolt",
    set_code="M10",
    number="146",
    base_set_size="249",
    artist="Christopher Moeller",
    set_release_year="2009",
)
BOLT_M11 = Printing(
    uuid="bolt-m11",
    name="Lightning Bolt",
    set_code="M11",
    number="149",
    base_set_size="249",
    artist="Christopher Moeller",
    set_release_year="2010",
)
SHOCK = Printing(
    uuid="shock-m19",
    name="Shock",
    set_code="M19",
    number="156",
    base_set_size="280",
    artist="Jason Rainville",
    set_release_year="2018",
)
GRIZZLY = Printing(
    uuid="grizzly-m10",
    name="Grizzly Bears",
    set_code="M10",
    number="186",
    base_set_size="249",
    artist="D. J. Cleland-Hura",
    set_release_year="2009",
    power="2",
    toughness="2",
)


def test_name_alone_ranks_matching_card_above_other_cards():
    ranked = rank(Reading(name="LIGHTNING BOLT"), [SHOCK, BOLT_M10], k=2)

    assert ranked[0].printing == BOLT_M10


def test_set_code_and_number_pick_the_right_printing_among_reprints():
    reading = Reading(
        name="Lightning Bolt",
        set_code_candidates=["M11"],
        number="149",
        base_set_size="249",
    )

    ranked = rank(reading, [BOLT_M10, SHOCK, BOLT_M11], k=3)

    assert ranked[0].printing == BOLT_M11


def test_power_and_toughness_count_towards_the_score():
    reading = Reading(set_code_candidates=["M10"], power="2", toughness="2")

    ranked = rank(reading, [BOLT_M10, GRIZZLY], k=2)

    assert ranked[0].printing == GRIZZLY


def test_returns_the_k_best_printings_with_descending_scores():
    reading = Reading(name="Lightning Bolt", set_code_candidates=["M10"], number="146")

    ranked = rank(reading, [SHOCK, BOLT_M11, GRIZZLY, BOLT_M10], k=2)

    assert [r.printing for r in ranked] == [BOLT_M10, BOLT_M11]
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert 0 < scores[0] <= 100


def _number_score(read: str, printed: str) -> float:
    printing = Printing(
        uuid="p", name="Card", set_code="TST", number=printed, base_set_size="100"
    )
    [ranked] = rank(Reading(number=read), [printing], k=1)
    return next(s.score for s in ranked.breakdown if s.field == "number")


@pytest.mark.parametrize(
    ("read", "printed"),
    [("0146", "146"), ("146", "0146"), ("0", "0"), ("0000", "0"), ("T001", "T001")],
)
def test_collector_numbers_match_regardless_of_leading_zeros(read, printed):
    assert _number_score(read, printed) == 100.0


def test_leading_zeros_are_not_stripped_to_nothing():
    assert _number_score("0", "") == 0.0


def test_tied_printings_keep_their_candidate_order():
    # Both reprints agree equally with a reading of the name alone.
    reading = Reading(name="Lightning Bolt")

    assert [r.printing for r in rank(reading, [BOLT_M11, BOLT_M10], k=2)] == [
        BOLT_M11,
        BOLT_M10,
    ]
    assert [r.printing for r in rank(reading, [BOLT_M10, BOLT_M11], k=2)] == [
        BOLT_M10,
        BOLT_M11,
    ]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_returns_exactly_k_printings(k):
    candidates = [SHOCK, BOLT_M11, GRIZZLY, BOLT_M10]

    assert len(rank(Reading(), candidates, k=k)) == k


def test_returns_every_candidate_when_k_exceeds_them():
    assert len(rank(Reading(), [SHOCK, BOLT_M10], k=5)) == 2


def test_breakdown_holds_read_value_printing_value_and_score_of_every_field():
    reading = Reading(
        name="LIGHTNING BOLT",
        set_code_candidates=["M1O", "M10"],
        number="0146",
        base_set_size="249",
        artist="Christopher Moeller",
        set_release_year="2009",
    )

    [ranked] = rank(reading, [BOLT_M10], k=1)

    assert ranked.breakdown == [
        FieldScore("name", "LIGHTNING BOLT", "Lightning Bolt", 100.0),
        FieldScore("set_code", ["M1O", "M10"], "M10", 100.0),
        FieldScore("number", "0146", "146", 100.0),
        FieldScore("base_set_size", "249", "249", 100.0),
        FieldScore("artist", "Christopher Moeller", "Christopher Moeller", 100.0),
        FieldScore("set_release_year", "2009", "2009", 100.0),
        FieldScore("power", None, None, 0.0),
        FieldScore("toughness", None, None, 0.0),
    ]
    # Weighted average: name x2, artist x0.5, others x1; power/toughness score 0.
    assert ranked.score == pytest.approx((200 + 400 + 50) / 8.5)
