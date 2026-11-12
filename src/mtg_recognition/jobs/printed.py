"""What each printing shows on the card: its printed set code, collector number,
base set size and year, derived from AllPrintings fields only (#201, #206).

Many printings show other values than the catalog's own: The List shows its
original set's set code and collector number, a promo shows its parent set's
details, a retro reprint shows no base set size. `None` means the card doesn't
show the detail. The 35 sets whose cards show values the rules can't derive keep
the rules' values, with no hand-kept table (#206).
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

# Set release dates where what a card shows changes, checked on reference images
# in the research of #201.
LEGENDS = "1994-06-01"  # first set with a copyright year on the card
EXODUS = "1998-06-15"  # first set with collector numbers
MAGIC_2015 = "2014-07-18"  # first set in the 2015 frame, which shows the set code
# Throne of Eldraine (2019-10-04) is the first set whose cards numbered above the
# base set size show the collector number alone; Core Set 2020 (2019-07-12) still
# shows 312/280
NUMBER_ALONE_ABOVE_BASE_SET_SIZE_FROM = "2019-10-01"
# March of the Machine (2023-04-21) is the first set showing no base set size;
# Phyrexia: All Will Be One (2023-02-10) still shows 114/271
NO_BASE_SET_SIZE_FROM = "2023-04-01"

THE_LIST = "PLST"
WORLD_CHAMPIONSHIP_SETS = frozenset(
    {"PTC", "WC97", "WC98", "WC99", "WC00", "WC01", "WC02", "WC03", "WC04"}
)
OLD_FRAMES = frozenset({"1993", "1997", "2003"})


@dataclass(frozen=True, slots=True)
class SetRow:
    """The fields of an AllPrintings `sets` row the rules use."""

    code: str
    type: str | None
    parent_code: str | None
    keyrune_code: str | None
    base_set_size: int
    release_date: str


@dataclass(frozen=True, slots=True)
class PrintingRow:
    """The fields of an AllPrintings `cards` row the rules use."""

    uuid: str
    set_code: str
    collector_number: str
    name: str
    artist: str | None
    frame_version: str | None
    original_release_date: str | None
    promo_types: str | None


@dataclass(frozen=True, slots=True)
class Source:
    """The printing whose set code and collector number a printing shows instead
    of its own.
    """

    set_code: str
    collector_number: str


@dataclass(frozen=True, slots=True)
class PrintedDetails:
    """What a printing shows on the card; `None` when it doesn't show the detail."""

    set_code: str | None
    collector_number: str | None
    base_set_size: str | None
    year: str | None


def printed_sources(
    printings: Iterable[PrintingRow], sets: dict[str, SetRow]
) -> dict[str, Source]:
    """The printings that show another printing's set code and collector number,
    by UUID:

    - The List: the set code and collector number in its own (`HOP-128`).
    - A printing of a set with a parent set (promos, oversized cards), when the
      parent has a printing with the same name and number digits (`PKHM 214p`
      shows `KHM 214`).
    - A World Championship deck printing: the latest expansion or core set
      printing released before the deck, with the same name, artist and number
      digits.
    """
    printings = list(printings)
    # number digits of each card name in each set, and the expansion and core
    # set printings of each card name
    digits_by_set_and_name: dict[tuple[str, str], set[str | None]] = {}
    originals_by_name: dict[str, list[PrintingRow]] = {}
    for printing in printings:
        digits_by_set_and_name.setdefault(
            (printing.set_code, printing.name), set()
        ).add(digits(printing.collector_number))
        if sets[printing.set_code].type in ("expansion", "core"):
            originals_by_name.setdefault(printing.name, []).append(printing)

    sources: dict[str, Source] = {}
    for printing in printings:
        own_set = sets[printing.set_code]
        number_digits = digits(printing.collector_number)
        parent_digits = digits_by_set_and_name.get(
            (own_set.parent_code or "", printing.name), set()
        )
        if printing.set_code == THE_LIST:
            code, original_number = printing.collector_number.split("-", 1)
            sources[printing.uuid] = Source(code, original_number)
        elif own_set.parent_code and number_digits in parent_digits:
            sources[printing.uuid] = Source(
                own_set.parent_code, printing.collector_number
            )
        elif printing.set_code in WORLD_CHAMPIONSHIP_SETS and number_digits:
            originals = [
                original
                for original in originals_by_name.get(printing.name, [])
                if original.artist == printing.artist
                and digits(original.collector_number) == number_digits
                and sets[original.set_code].release_date <= own_set.release_date
            ]
            if originals:
                original = max(originals, key=lambda o: sets[o.set_code].release_date)
                sources[printing.uuid] = Source(
                    original.set_code, original.collector_number
                )
    return sources


def printed_details(
    printing: PrintingRow, sets: dict[str, SetRow], source: Source | None
) -> PrintedDetails:
    """What `printing` shows on the card, given whose set code and collector
    number it shows (`source`, from `printed_sources`; `None` for its own).
    """
    # a World Championship printing whose original isn't found: its details
    # can't be derived
    if source is None and printing.set_code in WORLD_CHAMPIONSHIP_SETS:
        return PrintedDetails(None, None, None, None)

    # whose details the printing shows, and the date that decides their format
    own_set = sets[printing.set_code]
    own_date = printing.original_release_date or own_set.release_date
    if source is None:
        code, collector_number = printing.set_code, printing.collector_number
        details_set, date = own_set, own_date
    else:
        code, collector_number = source.set_code, source.collector_number
        details_set = sets[code]
        date = details_set.release_date
    frame = printing.frame_version
    # promo sets with their own numbering show PRM, FNM, H19 and the like, and
    # base set sizes of their own (001/003), which no field holds
    event_promo = source is None and (
        own_set.type == "promo"
        or (own_set.keyrune_code == "PMEI" and own_set.type in ("funny", "memorabilia"))
    )
    playtest = "playtest" in (printing.promo_types or "")
    # drawn in an older frame than its date's own: a retro reprint
    retro = frame in OLD_FRAMES and date >= MAGIC_2015

    # set code: only on the 2015 frame
    if frame != "2015" or playtest or event_promo:
        printed_set_code = None
    elif source is None and printing.set_code == f"O{own_set.parent_code}":
        # oversized cards (OPCA, OARC) show their parent set's code
        printed_set_code = own_set.parent_code
    else:
        printed_set_code = code.upper()

    # collector number: from Exodus on; old-frame promos show another set's
    # number, an event mark or nothing
    if (date < EXODUS and not retro) or playtest or (event_promo and frame != "2015"):
        printed_collector_number = None
    else:
        printed_collector_number = digits(collector_number)

    # base set size: after the collector number, until April 2023; not on retro
    # reprints, nor on extras numbered above it
    base_set_size = details_set.base_set_size
    if (
        printed_collector_number is None
        or date >= NO_BASE_SET_SIZE_FROM
        or retro
        or (
            int(printed_collector_number) > base_set_size
            and date >= NUMBER_ALONE_ABOVE_BASE_SET_SIZE_FROM
        )
        or event_promo
    ):
        printed_base_set_size = None
    else:
        printed_base_set_size = str(base_set_size)

    # year: The List shows its own (#216); the end year of a range (1993-2001)
    # is the date's year
    on_the_list = printing.set_code == THE_LIST
    year_date = own_date if source is None or on_the_list else date
    printed_year = None if year_date < LEGENDS and not retro else year_date[:4]

    return PrintedDetails(
        printed_set_code, printed_collector_number, printed_base_set_size, printed_year
    )


def is_plain(printing: PrintingRow, source: Source | None) -> bool:
    """Whether `printing` is the plain one among those showing the same details:
    it shows its own details (not The List, not a promo showing its parent's,
    not a World Championship deck), and its collector number has no suffix
    (`254★`, `71z`).
    """
    return (
        source is None
        and printing.set_code not in WORLD_CHAMPIONSHIP_SETS
        and re.fullmatch(r"[0-9]+", printing.collector_number) is not None
    )


def digits(collector_number: str) -> str | None:
    """The collector number as shown: its first run of digits, without leading
    zeros (`214p` -> `214`, `jt191sb` -> `191`, `0254` -> `254`).
    """
    match = re.search(r"[0-9]+", collector_number)
    return str(int(match.group())) if match else None
