"""Scoring a card's reading against candidate printings: each detail a candidate
shows on the card is looked for in the reading's raw text.
"""

from dataclasses import replace

import pytest

from mtg_recognition.engine.ocr.scoring import rank
from mtg_recognition.engine.ranking import FieldScore
from mtg_recognition.reference.catalog import Printing
from tests.unit.engine.builders import reading

BOLT_M10 = Printing(
    uuid="bolt-m10",
    name="Lightning Bolt",
    set_code="M10",
    collector_number="146",
    base_set_size="249",
    artist="Christopher Moeller",
    set_release_year="2009",
    printed_set_code=None,
    printed_collector_number="146",
    printed_base_set_size="249",
    printed_year="2009",
    is_plain=True,
)
BOLT_M11 = replace(
    BOLT_M10,
    uuid="bolt-m11",
    set_code="M11",
    collector_number="149",
    set_release_year="2010",
    printed_collector_number="149",
    printed_year="2010",
)
SHOCK = Printing(
    uuid="shock-m19",
    name="Shock",
    set_code="M19",
    collector_number="156",
    base_set_size="280",
    artist="Jason Rainville",
    set_release_year="2018",
    printed_collector_number="156",
    printed_base_set_size="280",
    printed_year="2018",
    is_plain=True,
)
GRIZZLY = Printing(
    uuid="grizzly-m10",
    name="Grizzly Bears",
    set_code="M10",
    collector_number="186",
    base_set_size="249",
    artist="D. J. Cleland-Hura",
    set_release_year="2009",
    power="2",
    toughness="2",
    printed_collector_number="186",
    printed_base_set_size="249",
    printed_year="2009",
    is_plain=True,
)
# 2015 frame: shows its set code; numbered above the base set size, so shows no
# base set size
SHIVAN = Printing(
    uuid="shivan-thb",
    name="Shivan Dragon",
    set_code="THB",
    collector_number="301",
    base_set_size="254",
    artist="Donato Giancola",
    set_release_year="2020",
    power="5",
    toughness="5",
    printed_set_code="THB",
    printed_collector_number="301",
    printed_year="2020",
    is_plain=True,
)
# Alpha: shows only its name and artist
ALPHA_BOLT = Printing(
    uuid="bolt-lea",
    name="Lightning Bolt",
    set_code="LEA",
    collector_number="161",
    base_set_size="295",
    artist="Christopher Rush",
    set_release_year="1993",
    is_plain=True,
)


# --- what is scored ---


def test_a_detail_the_printing_does_not_show_is_left_out_of_its_score():
    [ranked] = rank(reading("Lightning Bolt", ["Christopher Rush"]), [ALPHA_BOLT], k=1)

    # name and artist are found; set code, number, size, year and P/T aren't shown
    assert ranked.score == 100.0
    not_shown = [s.field for s in ranked.breakdown if s.score is None]
    assert not_shown == [
        "set_code",
        "collector_number",
        "base_set_size",
        "year",
        "power_toughness",
    ]


def test_the_score_is_the_weighted_average_of_the_details_shown():
    # name 100 (x2), number 100, size 0, year 0, artist 100 (x0.5)
    [ranked] = rank(
        reading("Lightning Bolt", ["146", "Christopher Moeller"]), [BOLT_M10], k=1
    )

    assert ranked.score == pytest.approx((200 + 100 + 50) / 5.5)


def test_the_field_score_holds_the_value_looked_for_its_score_and_the_text():
    info = ["146/249", "Illus. Christopher Moeller", "™ & © 2009 Wizards"]

    [ranked] = rank(reading("LIGHTNING BOLT", info), [BOLT_M10], k=1)

    assert ranked.breakdown == [
        FieldScore("name", "Lightning Bolt", 100.0, "LIGHTNING BOLT"),
        FieldScore("set_code", None, None, None),
        FieldScore("collector_number", "146", 100.0, "146/249"),
        FieldScore("base_set_size", "249", 100.0, "146/249"),
        FieldScore("artist", "Christopher Moeller", 100.0, "Christopher Moeller"),
        FieldScore("year", "2009", 100.0, "™ & © 2009 Wizards"),
        FieldScore("power_toughness", None, None, None),
    ]


def test_a_detail_not_found_scores_0_with_no_text():
    scores = field_scores(BOLT_M10, info=["Wizards of the Coast"])

    assert scores["collector_number"] == FieldScore(
        "collector_number", "146", 0.0, None
    )


def field_scores(
    printing: Printing, name: str = "", info: list[str] | None = None
) -> dict[str, FieldScore]:
    """The printing's field scores against a reading of `name` and `info`, by
    field.
    """
    [ranked] = rank(reading(name, info), [printing], k=1)
    return {s.field: s for s in ranked.breakdown}


# --- name ---


@pytest.mark.parametrize("read", ["Wear", "Tear"])
def test_the_name_is_found_as_either_face(read):
    wear_tear = replace(BOLT_M10, name="Wear // Tear")

    assert field_scores(wear_tear, name=read)["name"].score == 100.0


def test_the_name_is_found_as_the_flavor_name():
    flavored = replace(BOLT_M10, name="Mothra's Great Cocoon", flavor_name="Mothra")

    scores = field_scores(flavored, name="Mothra")

    assert (scores["name"].value, scores["name"].score) == ("Mothra", 100.0)


# --- set code ---


@pytest.mark.parametrize("token", ["THB", "THB·EN", "THBDE", "THB EN", "thb-en"])
def test_the_set_code_is_found_alone_or_followed_by_a_language(token):
    assert field_scores(SHIVAN, info=[token])["set_code"].score == 100.0


@pytest.mark.parametrize("token", ["THBX", "THBENG", "XTHB", "TH"])
def test_the_set_code_is_not_found_inside_another_word(token):
    assert field_scores(SHIVAN, info=[token])["set_code"].score == 0.0


# --- collector number and year ---


@pytest.mark.parametrize(
    ("token", "number"),
    [
        ("146/249", "146"),
        ("0146", "146"),
        ("C 0146", "146"),
        ("O146", "146"),  # O read as 0
        ("14o", "140"),
    ],
)
def test_the_collector_number_is_found_as_a_whole_run_of_digits(token, number):
    printing = replace(BOLT_M10, printed_collector_number=number)

    assert field_scores(printing, info=[token])["collector_number"].score == 100.0


def test_a_collector_number_is_not_found_inside_a_longer_run():
    fourteen = replace(BOLT_M10, printed_collector_number="14")

    assert field_scores(fourteen, info=["149/249"])["collector_number"].score == 0.0


@pytest.mark.parametrize(
    ("token", "year"),
    [
        ("© 2009 Wizards", "2009"),
        ("©1993-2001 Wizards", "2001"),
        ("©199O Wizards", "1990"),
    ],
)
def test_the_year_is_found_as_a_whole_run_of_digits(token, year):
    printing = replace(BOLT_M10, printed_year=year)

    assert field_scores(printing, info=[token])["year"].score == 100.0


def test_a_year_is_not_found_inside_a_longer_run():
    assert field_scores(BOLT_M10, info=["20090"])["year"].score == 0.0


# --- base set size ---


@pytest.mark.parametrize(
    "tokens",
    [
        ["146/249"],
        ["146", "/249"],
        ["146 / 249"],
        ["1461249"],  # slash read as 1
        ["146249"],  # slash lost
        ["0146/0249"],
    ],
)
def test_the_base_set_size_is_found_right_after_the_number(tokens):
    assert field_scores(BOLT_M10, info=tokens)["base_set_size"].score == 100.0


def test_the_base_set_size_is_not_found_away_from_the_number():
    tokens = ["146", "2009", "249"]

    assert field_scores(BOLT_M10, info=tokens)["base_set_size"].score == 0.0


# --- artist ---


def test_the_artist_is_found_inside_the_collector_information():
    scores = field_scores(BOLT_M10, info=["Illus. Christopher Moeller 146/249"])

    assert scores["artist"].score == 100.0


# --- power/toughness ---


@pytest.mark.parametrize("token", ["2/2", "2 / 2", "2|2", "2\\2", "2I2", "2l2", "212"])
def test_power_toughness_is_found_with_the_slash_read_as_another_mark(token):
    assert field_scores(GRIZZLY, info=[token])["power_toughness"].score == 100.0


@pytest.mark.parametrize("token", ["12/2", "2/22", "X2/2"])
def test_power_toughness_is_not_found_inside_other_numbers(token):
    assert field_scores(GRIZZLY, info=[token])["power_toughness"].score == 0.0


def test_power_toughness_is_shown_only_by_a_printing_with_a_power():
    scores = field_scores(BOLT_M10, info=["2/2"])

    assert scores["power_toughness"].score is None


def test_power_toughness_counts_towards_the_score():
    card_reading = reading("", ["186/249", "2/2"])

    ranked = rank(card_reading, [BOLT_M10, GRIZZLY], k=2)

    assert ranked[0].printing == GRIZZLY


# --- ranking ---


def test_the_collector_number_picks_the_printing_among_reprints():
    card_reading = reading("Lightning Bolt", ["149/249"])

    ranked = rank(card_reading, [BOLT_M10, SHOCK, BOLT_M11], k=3)

    assert [r.printing for r in ranked] == [BOLT_M11, BOLT_M10, SHOCK]


def test_tied_printings_put_the_plain_printing_first():
    on_the_list = replace(BOLT_M10, uuid="bolt-plst", is_plain=False)

    ranked = rank(reading("Lightning Bolt", ["146/249"]), [on_the_list, BOLT_M10], k=2)

    assert [r.printing for r in ranked] == [BOLT_M10, on_the_list]
    assert ranked[0].score == ranked[1].score


def test_tied_plain_printings_keep_their_candidate_order():
    card_reading = reading("Lightning Bolt")
    other = replace(BOLT_M10, uuid="bolt-other")

    assert [r.printing for r in rank(card_reading, [other, BOLT_M10], k=2)] == [
        other,
        BOLT_M10,
    ]
    assert [r.printing for r in rank(card_reading, [BOLT_M10, other], k=2)] == [
        BOLT_M10,
        other,
    ]


def test_the_ranking_records_whether_the_tie_break_decided_a_place():
    on_the_list = replace(BOLT_M10, uuid="bolt-plst", is_plain=False)

    ranked = rank(
        reading("Lightning Bolt", ["146/249"]), [SHOCK, on_the_list, BOLT_M10], k=3
    )

    assert [(r.printing, r.decided_by_tie_break) for r in ranked] == [
        (BOLT_M10, True),
        (on_the_list, True),
        (SHOCK, False),
    ]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_returns_exactly_k_printings(k):
    candidates = [SHOCK, BOLT_M11, GRIZZLY, BOLT_M10]

    assert len(rank(reading(), candidates, k=k)) == k


def test_returns_every_candidate_when_k_exceeds_them():
    assert len(rank(reading(), [SHOCK, BOLT_M10], k=5)) == 2
