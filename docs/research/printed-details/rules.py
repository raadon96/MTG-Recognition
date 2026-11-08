"""Throwaway: what each printing prints for the details scoring uses, derived from
AllPrintings fields only.

`printed_details(card, sets, sources)` takes one `cards` row, every `sets` row
(both as dicts) and the result of `printed_sources`, and returns the printed set
code, collector number, base set size and copyright year. `None` means the card
prints nothing for that detail, or that what it prints can't be derived (scoring
should leave the detail out either way). `rule` names the rule that decided each
detail, for counting.
"""

import re

# Set release dates from AllPrintings where what is printed changes; the
# boundaries are checked on reference images in docs/research/printed-details.md.
LEGENDS = "1994-06-01"  # first set with a copyright year on the card
EXODUS = "1998-06-15"  # first set with collector numbers
MAGIC_2015 = "2014-07-18"  # first set with the 2015 frame and a printed set code
# Throne of Eldraine (2019-10-04) is the first set whose cards numbered above the
# base set size print the number alone; Core Set 2020 (2019-07-12) still prints
# 312/280
EXTRAS_NUMBER_ALONE_FROM = "2019-10-01"
# March of the Machine (2023-04-21) is the first set with the rarity-first number
# and no base set size; Phyrexia: All Will Be One (2023-02-10) still prints 114/271
NO_BASE_SET_SIZE_FROM = "2023-04-01"

WORLD_CHAMPIONSHIP_SETS = {
    "PTC", "WC97", "WC98", "WC99", "WC00", "WC01", "WC02", "WC03", "WC04"
}  # fmt: skip


def printed_sources(
    cards: list[dict], sets: dict[str, dict]
) -> dict[str, tuple[str, str, str]]:
    """`uuid -> (set code, number, rule)` for cards that print another printing's
    set code, number and size:

    - The List: the set code and number in its number (`HOP-128`).
    - A set with a parent set (promos, oversized cards) whose card has the same
      name and number digits as a card of the parent set (PKHM `214p`, KHM `214`).
    - World Championship decks: the latest expansion or core set printing,
      released before the deck, with the same name, artist and number digits.

    Every other card prints its own set code and number.
    """
    by_set_and_name: dict[tuple[str, str], set[str | None]] = {}
    by_name: dict[str, list[dict]] = {}
    for c in cards:
        by_set_and_name.setdefault((c["setCode"], c["name"]), set()).add(
            digits(c["number"])
        )
        if sets[c["setCode"]]["type"] in ("expansion", "core"):
            by_name.setdefault(c["name"], []).append(c)

    sources = {}
    for c in cards:
        own_set = sets[c["setCode"]]
        parent = own_set["parentCode"]
        number = digits(c["number"])
        if c["setCode"] == "PLST":
            code, original_number = c["number"].split("-", 1)
            sources[c["uuid"]] = (code, original_number, "The List: original set")
        elif parent and number in by_set_and_name.get((parent, c["name"]), ()):
            sources[c["uuid"]] = (parent, c["number"], "parent set's card")
        elif c["setCode"] in WORLD_CHAMPIONSHIP_SETS:
            released = own_set["releaseDate"]
            matches = [
                o
                for o in by_name.get(c["name"], [])
                if number
                and o["artist"] == c["artist"]
                and digits(o["number"]) == number
                and sets[o["setCode"]]["releaseDate"] <= released
            ]
            if matches:
                o = max(matches, key=lambda o: sets[o["setCode"]]["releaseDate"])
                sources[c["uuid"]] = (
                    o["setCode"], o["number"], "World Championship: original set"
                )  # fmt: skip
    return sources


def digits(number: str) -> str | None:
    """The collector number as printed: its first run of digits, without leading
    zeros (`214p` -> `214`, `jt191sb` -> `191`, `0254` -> `254`)."""
    m = re.search(r"\d+", number)
    return str(int(m.group())) if m else None


def printed_details(
    card: dict, sets: dict[str, dict], sources: dict[str, tuple[str, str, str]]
) -> dict:
    """The details `card` prints, and the rule that decided each."""
    own_set = sets[card["setCode"]]
    frame = card["frameVersion"]

    # whose details the card prints, and the date that decides the format
    if card["uuid"] in sources:
        code, number, source_rule = sources[card["uuid"]]
        source = sets[code]
        date = source["releaseDate"]
    else:
        code, number, source_rule = card["setCode"], card["number"], "own set"
        source = own_set
        date = card["originalReleaseDate"] or own_set["releaseDate"]
    own_promo = source_rule == "own set" and (
        own_set["type"] == "promo"
        or (
            own_set["keyruneCode"] == "PMEI"
            and own_set["type"] in ("funny", "memorabilia")
        )
    )
    playtest = "playtest" in (card["promoTypes"] or "")
    # an older frame than the date's own frame: a retro reprint
    retro = frame in ("1993", "1997", "2003") and date >= MAGIC_2015
    printed_number = digits(number)

    # set code: only on the 2015 frame
    if frame != "2015":
        printed_code, code_rule = None, f"not printed: {frame} frame"
    elif playtest:
        printed_code, code_rule = None, "not printed: playtest card"
    elif own_promo:
        # these print PRM, FNM, H19 and the like, which no field holds
        printed_code, code_rule = None, "not derivable: promo set's own numbering"
    elif source_rule == "own set" and card["setCode"] == f"O{own_set['parentCode']}":
        # oversized cards (OPCA, OARC) print their parent set's code
        printed_code, code_rule = own_set["parentCode"], "oversized: parent set's code"
    else:
        printed_code, code_rule = code.upper(), source_rule

    # collector number: from Exodus on
    if date < EXODUS and not retro:
        printed_number, number_rule = None, "not printed: before Exodus"
    elif playtest:
        printed_number, number_rule = None, "not printed: playtest card"
    elif own_promo and frame != "2015":
        # old-frame promos print another set's number, an event mark, or nothing
        printed_number, number_rule = None, "not derivable: old-frame promo"
    elif printed_number is None:
        printed_number, number_rule = None, "not printed: no digits"
    else:
        number_rule = "promo set's own number" if own_promo else source_rule

    # base set size: after the number, until 2023, not on retro reprints nor on
    # extras numbered above the base set
    size = source["baseSetSize"]
    if printed_number is None:
        printed_size, size_rule = None, "not printed: no number"
    elif date >= NO_BASE_SET_SIZE_FROM:
        printed_size, size_rule = None, "not printed: from April 2023"
    elif retro:
        printed_size, size_rule = None, "not printed: retro reprint"
    elif int(printed_number) > size and date >= EXTRAS_NUMBER_ALONE_FROM:
        printed_size, size_rule = None, "not printed: above the base set size"
    elif own_promo:
        # event promos print a size of their own (001/003, 004/005) that
        # baseSetSize often doesn't hold
        printed_size, size_rule = None, "not derivable: promo set's own number"
    else:
        printed_size, size_rule = str(size), source_rule

    # copyright year: the year of the date above
    if date < LEGENDS and not retro:
        printed_year, year_rule = None, "not printed: before Legends"
    elif source_rule == "own set" and card["originalReleaseDate"]:
        printed_year, year_rule = date[:4], "originalReleaseDate"
    else:
        printed_year, year_rule = date[:4], source_rule

    return {
        "set_code": printed_code,
        "number": printed_number,
        "base_set_size": printed_size,
        "year": printed_year,
        "rule": {
            "set_code": code_rule,
            "number": number_rule,
            "base_set_size": size_rule,
            "year": year_rule,
        },
    }
