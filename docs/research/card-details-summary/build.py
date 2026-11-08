# ruff: noqa: E501  (throwaway: HTML in long f-strings)
"""Throwaway: the summary of what each card shows and where (issue #203).

For every paper printing in AllPrintings, decides with the rules in
`printed-details/rules.py` which details the card shows, and sorts it into
groups: frame and collector style (table A), and the layouts that move details
(table B). The collector style is which details the collector information
prints, and in what form; the rules' dates and conditions decide it. Each detail of each group gets a label:

- always shown: on at least 95% of the group's paper rows;
- never shown: on at most 5%;
- sometimes shown: in between, with the main reason it is missing.

Then re-sorts the 540 readings of #193 (`card-details/positions.json`) into the
same groups, tags each token with the detail it holds (using the rules' values,
not the catalog's), and measures whether each detail sits inside today's crops.
Writes `summary.json` and `../card-details-summary.html`, and prints the
Markdown tables.

uv run python docs/research/card-details-summary/build.py \
    <AllPrintings.sqlite> <positions.json> [<extra positions.json>]
"""

import html
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "printed-details"))

from check import found  # noqa: E402  # ty: ignore[unresolved-import]
from rules import (  # noqa: E402  # ty: ignore[unresolved-import]
    EXODUS,
    EXTRAS_NUMBER_ALONE_FROM,
    LEGENDS,
    MAGIC_2015,
    NO_BASE_SET_SIZE_FROM,
    WORLD_CHAMPIONSHIP_SETS,
    digits,
    printed_details,
    printed_sources,
)

# (top, bottom, left, right), as in engine/ocr/reading.py
NAME_ROI = (0.04, 0.11, 0.06, 0.9)
COLLECTOR_INFORMATION_ROI = (0.88, 1.0, 0.05, 0.95)

# detail -> (label, box colour); no red, which marks today's crops
DETAILS = {
    "name": ("name", "#4363d8"),
    "second name": ("second name", "#42d4f4"),
    "set code": ("set code", "#f58231"),
    "number": ("collector number", "#911eb4"),
    "base set size": ("base set size", "#3cb44b"),
    "artist": ("artist", "#9a6324"),
    "year": ("year", "#f032e6"),
    "power/toughness": ("power/toughness", "#ffe119"),
}
RULE_DETAILS = {
    "set code": "set_code",
    "number": "number",
    "base set size": "base_set_size",
    "year": "year",
}
COLUMNS_A = [d for d in DETAILS if d != "second name"]
COLUMNS_B = list(DETAILS)

TWO_NAME_LAYOUTS = {"adventure", "prepare", "aftermath", "flip", "split"}
SAME_PLACES_AS_NORMAL = {
    "transform", "modal_dfc", "saga", "class", "case", "mutate", "prototype", "meld"
}  # fmt: skip
# #193's groups -> table B rows
OLD_LAYOUT_GROUPS = {
    "layout adventure": "adventure",
    "layout prepare": "prepare",
    "layout aftermath": "aftermath",
    "layout flip": "flip",
    "layout leveler": "leveler",
    "layout planar": "planar",
    "layout transform, battle": "battle",
    "layout split, 2003 frame or older": "split, before the 2015 frame",
    "layout split, 2015 frame": "split, 2015 frame",
    **{
        f"layout {lay}": "same places as a normal card" for lay in SAME_PLACES_AS_NORMAL
    },
}
# table A: (frame, collector style) -> the condition besides the dates, if any
ROWS_A = {
    ("1993", "artist only"): "",
    ("1993", "artist and year"): "",
    ("1997", "artist and year"): "",
    ("1997", "number and size"): "",
    ("1997", "number alone"): "retro reprints",
    ("2003", "number and size"): "",
    ("2015", "number and size"): "",
    ("2015", "number alone"): "numbered above the base set size",
    ("2015", "rarity first"): "",
    ("future", "number and size"): "",
    ("future", "number alone"): "",
}
# the form each collector style prints, for the legend
STYLES = {
    "artist only": "Illus. Artist",
    "artist and year": "Illus. Artist © 1995",
    "number and size": "123/280",
    "number alone": "123",
    "rarity first": "R 0123",
}
ROWS_B = [
    "same places as a normal card",
    "adventure",
    "prepare",
    "aftermath",
    "flip",
    "split, 2015 frame",
    "split, before the 2015 frame",
    "battle",
    "planar",
    "leveler",
]
LAYOUT_NOTES = {
    "same places as a normal card": (
        "transform, modal_dfc, saga, class, case, mutate, prototype, meld"
    ),
}

ALWAYS, NEVER = 95.0, 5.0


def main(allprintings: str, positions_paths: list[str]) -> None:
    ap = sqlite3.connect(f"file:{allprintings}?mode=ro", uri=True)
    ap.row_factory = sqlite3.Row
    sets = {r["code"]: dict(r) for r in ap.execute("select * from sets")}
    cards = {
        r["uuid"]: dict(r)
        for r in ap.execute(
            """select c.uuid, c.setCode, c.number, c.frameVersion,
                      c.originalReleaseDate, c.layout, c.promoTypes, c.name,
                      c.artist, c.power, c.toughness, c.type, i.scryfallId,
                      c.availability like '%paper%' as paper
               from cards c join cardIdentifiers i on i.uuid = c.uuid"""
        )
    }
    sources = printed_sources(list(cards.values()), sets)
    derived = {u: printed_details(c, sets, sources) for u, c in cards.items()}
    paper = [c for c in cards.values() if c["paper"]]

    # what each group shows: count paper rows per group, detail and rule
    groups = {
        row_name(frame, style): new_group(
            "A", row_name(frame, style), frame, style, only
        )
        for (frame, style), only in ROWS_A.items()
    } | {name: new_group("B", name) for name in ROWS_B}
    for c in paper:
        d = derived[c["uuid"]]
        row_b = layout_row(c)
        row_a, date = frame_row(c, sets, sources)
        for name in (row_a, row_b):
            if name is None:
                continue
            g = groups[name]
            g["rows"] += 1
            g["dates"].append(date)
            for detail in DETAILS:
                g["shown"][detail][shown_on_card(detail, c, d, row_b)] += 1

    # where it sits: #193's readings, re-sorted into the same groups
    samples = []
    for path in positions_paths:
        for old_group, data in json.loads(Path(path).read_text()).items():
            samples += [(old_group, s) for s in data["samples"]]
    for old_group, s in samples:
        c = cards[s["uuid"]]
        d = derived[s["uuid"]]
        tag_tokens(s, d)
        if old_group in OLD_LAYOUT_GROUPS:
            name = OLD_LAYOUT_GROUPS[old_group]
        elif old_group.startswith("layout"):
            continue
        else:
            name = frame_row(c, sets, sources)[0]
        g = groups[name]
        g["samples"].append(s)
        for detail in DETAILS:
            if shown_on_card(detail, c, d, layout_row(c)) != "shown":
                continue
            g["crop"][detail]["printed"] += 1
            boxes = [t for t in s["tokens"] if detail in t["tags"]]
            if not boxes:
                continue
            g["crop"][detail]["read"] += 1
            if all(t["orientation"] != "up" for t in boxes):
                g["crop"][detail]["sideways"] += 1
            else:
                upright = [t["box"] for t in boxes if t["orientation"] == "up"]
                best = max(overlap(b, crop_for(detail)) for b in upright)
                if best >= 0.9:
                    g["crop"][detail]["inside"] += 1
                elif best >= 0.5:
                    g["crop"][detail]["clipped"] += 1

    total = len(paper)
    summary = {name: summarise(g, total) for name, g in groups.items()}
    (HERE / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False)
    )
    (HERE.parent / "card-details-summary.html").write_text(page(summary, total))
    print(markdown_tables(summary))
    missing = [n for n, g in summary.items() if not g["example"]]
    print("groups without a sample:", missing, file=sys.stderr)


def new_group(
    table: str, name: str, frame: str = "", style: str = "", only: str = ""
) -> dict:
    return {
        "table": table,
        "name": name,
        "frame": frame,
        "style": style,
        "only": only,
        "rows": 0,
        "shown": {d: Counter() for d in DETAILS},
        "crop": {d: Counter() for d in DETAILS},
        "samples": [],
        "dates": [],
    }


def frame_row(card: dict, sets: dict, sources: dict) -> tuple[str, str]:
    """The table A row (frame and collector style), and the date the card's
    details come from: the source set's release date for The List, promos and
    World Championship cards, else `originalReleaseDate` or the set's release
    date. The rules in `printed_details` decide by the same date."""
    if card["uuid"] in sources:
        code = sources[card["uuid"]][0]
        date = sets[code]["releaseDate"]
        size = sets[code]["baseSetSize"]
        number = digits(sources[card["uuid"]][1])
    else:
        date = card["originalReleaseDate"] or sets[card["setCode"]]["releaseDate"]
        size = sets[card["setCode"]]["baseSetSize"]
        number = digits(card["number"])
    frame = card["frameVersion"]
    retro = frame in ("1993", "1997", "2003") and date >= MAGIC_2015
    above_size = number is not None and int(number) > size

    if date < LEGENDS and not retro:
        style = "artist only"
    elif date < EXODUS and not retro:
        style = "artist and year"
    elif frame == "2003":
        # 6 paper retro reprints, too few for a row of their own
        style = "number and size"
    elif retro:
        style = "number alone"
    elif date >= NO_BASE_SET_SIZE_FROM:
        style = "rarity first" if frame == "2015" else "number alone"
    elif above_size and date >= EXTRAS_NUMBER_ALONE_FROM:
        style = "number alone"
    else:
        style = "number and size"
    # no 1993-frame card is from Exodus on, and no future-frame card is older
    return row_name(frame, style), date


def row_name(frame: str, style: str) -> str:
    return f"{frame} frame, {style}"


def layout_row(card: dict) -> str | None:
    """The table B row, or None for normal cards and rare layouts."""
    layout = card["layout"]
    if layout == "transform" and "Battle" in (card["type"] or ""):
        return "battle"
    if layout in SAME_PLACES_AS_NORMAL:
        return "same places as a normal card"
    if layout == "split":
        if card["frameVersion"] == "2015":
            return "split, 2015 frame"
        return "split, before the 2015 frame"
    if layout in {"adventure", "prepare", "aftermath", "flip", "leveler", "planar"}:
        return layout
    return None


def shown_on_card(detail: str, card: dict, d: dict, row_b: str | None) -> str:
    """'shown', 'not shown: <reason>' or 'not derivable: <reason>'."""
    match detail:
        case "name":
            return "shown"
        case "second name":
            if card["layout"] in TWO_NAME_LAYOUTS:
                return "shown"
            return "not shown: one name"
        case "artist":
            return "shown" if card["artist"] else "not shown: no artist"
        case "power/toughness":
            if card["power"] is not None or card["toughness"] is not None:
                return "shown"
            return "not shown: not a creature or vehicle"
    rule = d["rule"][RULE_DETAILS[detail]]
    if d[RULE_DETAILS[detail]] is not None:
        return "shown"
    if rule.startswith("not derivable"):
        return rule
    return "not shown: " + rule.removeprefix("not printed: ")


def tag_tokens(sample: dict, d: dict) -> None:
    """Tag each token with the details it holds. Name, second name, artist and
    power/toughness keep #193's tags; set code, number, base set size and year
    are matched against the rules' values, since #193 matched them against the
    catalog's."""
    renamed = {
        "name": "name",
        "name (other face)": "second name",
        "artist": "artist",
        "power/toughness": "power/toughness",
    }
    for t in sample["tokens"]:
        tags = {renamed[x] for x in t["details"] if x in renamed}
        for detail, key in RULE_DETAILS.items():
            value = d[key]
            if value is not None and found(key, value, d["number"], [t["text"]]):
                tags.add(detail)
        t["tags"] = sorted(tags)


def crop_for(detail: str) -> tuple:
    return NAME_ROI if detail in ("name", "second name") else COLLECTOR_INFORMATION_ROI


def overlap(box: list, crop: tuple) -> float:
    """The share of the box that lies inside the crop."""
    left, top, right, bottom = box
    c_top, c_bottom, c_left, c_right = crop
    w = max(0.0, min(right, c_right) - max(left, c_left))
    h = max(0.0, min(bottom, c_bottom) - max(top, c_top))
    area = (right - left) * (bottom - top)
    return w * h / area if area > 0 else 0.0


def summarise(g: dict, total: int) -> dict:
    details = {}
    for detail in DETAILS:
        counts = g["shown"][detail]
        rows = sum(counts.values()) or 1
        shown = 100 * counts["shown"] / rows
        not_derivable = sum(
            v for k, v in counts.items() if k.startswith("not derivable")
        )
        missing = Counter({k: v for k, v in counts.items() if k != "shown"})
        if shown >= ALWAYS:
            label = "always"
        elif shown <= NEVER:
            label = "never"
        else:
            label = "sometimes"
        crop = g["crop"][detail]
        details[detail] = {
            "label": label,
            "shown_percent": round(shown, 1),
            "not_derivable_percent": round(100 * not_derivable / rows, 1),
            "reasons": [
                (k, round(100 * v / rows, 1)) for k, v in missing.most_common(3)
            ],
            "crop": crop_label(crop),
            "printed_samples": crop["printed"],
            "read": crop["read"],
            "inside": crop["inside"],
            "sideways": crop["sideways"],
            "clipped": crop["clipped"],
        }
    return {
        "table": g["table"],
        "name": g["name"],
        "frame": g["frame"],
        "style": g["style"],
        "only": g["only"],
        # the dates the row's details come from, 1st to 99th percentile, so a
        # few promos with odd dates don't stretch the range; no "to" while the
        # style is still printed
        "from": month(percentile(g["dates"], 0.01)),
        "to": ""
        if percentile(g["dates"], 0.99) >= "2026"
        else month(percentile(g["dates"], 0.99)),
        "rows": g["rows"],
        "share_percent": round(100 * g["rows"] / total, 2),
        "samples": len(g["samples"]),
        "details": details,
        "example": example(g),
    }


def percentile(dates: list[str], q: float) -> str:
    ordered = sorted(dates)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def month(date: str) -> str:
    """'1998-06-15' -> 'June 1998'."""
    names = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]  # fmt: skip
    return f"{names[int(date[5:7]) - 1]} {date[:4]}"


def crop_label(crop: Counter) -> str:
    """'inside', 'clipped', 'outside', 'sideways', 'partly' or 'not measured'.
    Inside: at least 90% of the box inside the crop; clipped: 50% to 90%."""
    read = crop["read"]
    if read == 0:
        return "not measured"
    if crop["sideways"] * 2 > read:
        return "sideways"
    if crop["inside"] >= 0.9 * read:
        return "inside"
    if crop["inside"] + crop["clipped"] >= 0.9 * read:
        return "clipped"
    if crop["inside"] <= 0.1 * read:
        return "outside"
    return "partly"


def example(g: dict) -> dict | None:
    """The sample with the most details read, as the row's annotated card."""
    if not g["samples"]:
        return None
    # prefer a card that prints its own set's details over List and promo cards
    s = max(
        g["samples"],
        key=lambda s: (
            s["setCode"] not in {"PLST", *WORLD_CHAMPIONSHIP_SETS}
            and not s.get("parentCode"),
            len({x for t in s["tokens"] for x in t["tags"]}),
            s["uuid"],
        ),
    )
    return {
        "name": s["name"],
        "printing": f"{s['setCode']} {s['number']}",
        "scryfallId": s["scryfallId"],
        "boxes": [
            {"detail": tag, "box": t["box"], "text": t["text"]}
            for t in s["tokens"]
            for tag in t["tags"]
        ],
    }


# ---- HTML

LABEL_CLASS = {"always": "always", "never": "never", "sometimes": "sometimes"}
CROP_MARK = {
    "inside": ("✓", "inside today's crops"),
    "outside": ("✗", "outside today's crops"),
    "sideways": (
        "↻",
        "printed sideways or upside down: read only after rotating the card",
    ),
    "clipped": ("✂", "partly cut off by today's crop"),
    "partly": ("◐", "inside today's crops on some cards only"),
    "not measured": ("", "not read on any sample"),
}

STYLE = """
body { font-family: system-ui, sans-serif; margin: 1.5rem; color: #222; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.15rem; margin-top: 2rem; }
table { border-collapse: collapse; font-size: 0.85rem; }
th, td { border: 1px solid #ccc; padding: 4px 6px; text-align: center; vertical-align: middle; }
th.group { text-align: left; max-width: 15rem; font-weight: normal; }
th.group b { display: block; }
td.always { background: #c8efc8; } td.never { background: #e4e4e4; color: #777; }
td.sometimes { background: #fff2a8; }
td small { display: block; color: #555; max-width: 9rem; margin: 0 auto; }
.mark { font-size: 1.1rem; font-weight: bold; }
.card { position: relative; width: 140px; display: inline-block; cursor: zoom-in; }
.card img { width: 100%; display: block; border-radius: 5px; }
.card.big { width: 420px; cursor: zoom-out; }
.box { position: absolute; box-sizing: border-box; border: 2px solid; }
.crop { position: absolute; box-sizing: border-box; border: 3px dashed #ff0000; }
.swatch { display: inline-block; width: 12px; height: 12px; border: 3px solid; vertical-align: middle; }
.legend td, .legend th { text-align: left; }
ul.traps li { margin-bottom: 0.4rem; }
"""


def page(summary: dict, total: int) -> str:
    a = [g for g in summary.values() if g["table"] == "A"]
    b = [g for g in summary.values() if g["table"] == "B"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>What each card shows, and where</title><style>{STYLE}</style></head><body>
<h1>What each card shows, and where</h1>
<p>Summary for issue #203, from #193 (where details sit) and #201 (what each printing
prints). {total:,} paper printings, MTGJSON <code>5.3.0+20261103</code>. Details:
<code>card-details-summary.md</code>. Click a card to enlarge it.</p>
{legend()}
<h2>A. By frame and collector style</h2>
<p>Every paper printing is in exactly one row. The <b>frame</b> decides where the details
sit; the <b>collector style</b> decides which details the collector information prints.
"From" and "to" are the release dates the row's details come from (the original set's,
for The List and promos).</p>
{table(a, COLUMNS_A)}
<h2>B. Layouts that move details</h2>
<p>These cards are also counted in table A. Rarer layouts (reversible, scheme,
vanguard, host, augment: 0.3%) are left out.</p>
{table(b, COLUMNS_B)}
<h2>Traps</h2>
{TRAPS}
<script>
document.querySelectorAll('.card').forEach(c =>
  c.addEventListener('click', () => c.classList.toggle('big')));
</script>
</body></html>
"""


def legend() -> str:
    swatches = " ".join(
        f'<span class="swatch" style="border-color:{colour}"></span> {label}&nbsp;&nbsp;'
        for label, colour in DETAILS.values()
    )
    return f"""<table class="legend">
<tr><th>Cell colour: is the detail on the card?</th><td>
<span class="swatch" style="background:#c8efc8;border-color:#c8efc8"></span> always shown
(95% of the row's printings or more) &nbsp;
<span class="swatch" style="background:#fff2a8;border-color:#fff2a8"></span> sometimes shown,
with the main reason it's missing &nbsp;
<span class="swatch" style="background:#e4e4e4;border-color:#e4e4e4"></span> never shown
(5% or less)</td></tr>
<tr><th>Mark: where is it, for today's extraction?</th><td>
<span class="mark">✓</span> inside today's crops &nbsp;
<span class="mark">✗</span> outside &nbsp;
<span class="mark">✂</span> partly cut off by the crop &nbsp;
<span class="mark">↻</span> printed sideways or upside down &nbsp;
<span class="mark">◐</span> inside on some cards only &nbsp; no mark: not read on any sample</td></tr>
<tr><th>Collector style: which details are printed</th><td>
{" &nbsp; ".join(f"<b>{name}</b> <code>{html.escape(form)}</code>" for name, form in STYLES.items())}
</td></tr>
<tr><th>Boxes on the cards</th><td>
<span class="swatch" style="border:3px dashed #ff0000"></span> today's two crops &nbsp;
{swatches}</td></tr>
<tr><th>Hover a cell</th><td>the exact share, the reasons, and the read rate on the
sampled reference images (for information only: it doesn't decide the colour).</td></tr>
</table>"""


def table(groups: list[dict], columns: list[str]) -> str:
    head = "".join(
        f'<th><span class="swatch" style="border-color:{DETAILS[d][1]}"></span><br>'
        f"{DETAILS[d][0]}</th>"
        for d in columns
    )
    by_style = groups[0]["table"] == "A"
    rows = []
    for g in groups:
        if by_style:
            only = f"<small>only {html.escape(g['only'])}</small>" if g["only"] else ""
            group = (
                f'<th class="group"><b>{g["frame"]}</b></th>'
                f'<th class="group"><b>{g["style"]}</b>{only}</th>'
                f"<td>{g['from']}</td><td>{g['to'] or 'today'}</td>"
            )
        else:
            note = LAYOUT_NOTES.get(g["name"], "")
            note = f"<br><small>{note}</small>" if note else ""
            group = f'<th class="group"><b>{html.escape(g["name"])}</b>{note}</th>'
        cells = "".join(cell(g["details"][d]) for d in columns)
        rows.append(
            f"<tr>{group}"
            f"<td>{card(g['example'])}</td>"
            f"<td>{g['share_percent']:.1f}%<br><small>{g['rows']:,} rows</small></td>"
            f"{cells}</tr>"
        )
    first = (
        "<th>frame</th><th>collector style</th><th>from</th><th>to</th>"
        if by_style
        else "<th>layout</th>"
    )
    return (
        f"<table><tr>{first}<th>example</th><th>share</th>{head}</tr>"
        + "\n".join(rows)
        + "</table>"
    )


def cell(d: dict) -> str:
    mark, crop_text = CROP_MARK[d["crop"]]
    reasons = "; ".join(f"{reason_text(r)}: {p}%" for r, p in d["reasons"])
    hover = (
        f"shown on {d['shown_percent']}% of the row's paper printings"
        + (f"; missing: {reasons}" if reasons else "")
        + f". Read on {d['read']} of {d['printed_samples']} samples where it is shown"
        + f"; inside today's crops on {d['inside']}"
        + (f", sideways on {d['sideways']}" if d["sideways"] else "")
        + (f", partly cut off on {d['clipped']}" if d["clipped"] else "")
        + f". {crop_text}."
    )
    small = ""
    if d["label"] == "sometimes" and d["reasons"]:
        small = f"<small>{html.escape(reason_text(d['reasons'][0][0]))}</small>"
    show_mark = mark if d["label"] != "never" else ""
    return (
        f'<td class="{LABEL_CLASS[d["label"]]}" title="{html.escape(hover)}">'
        f'<span class="mark">{show_mark}</span>{small}</td>'
    )


REASON_TEXT = {
    "not shown: not a creature or vehicle": "creatures and vehicles only",
    "not shown: playtest card": "not on playtest cards",
    "not shown: no number": "not when no number is printed",
    "not shown: from April 2023": "not from April 2023",
    "not shown: before Exodus": "not before Exodus",
    "not shown: before Legends": "not before Legends",
    "not shown: retro reprint": "not on retro reprints",
    "not shown: above the base set size": "not when numbered above the size",
    "not shown: no artist": "not on some Un-cards",
    "not shown: one name": "one name only",
    "not derivable: old-frame promo": "old-frame promos: value unknown",
    "not derivable: promo set's own numbering": "event promos: value unknown",
    "not derivable: promo set's own number": "event promos: value unknown",
}


def reason_text(reason: str) -> str:
    """The rule's reason in words; frame reasons mean the 2015 frame only."""
    if reason.endswith(" frame") and reason.startswith("not shown: "):
        return "2015 frame only"
    return REASON_TEXT.get(reason, reason)


def card(e: dict | None) -> str:
    if e is None:
        return "no sample"
    url = (
        f"https://cards.scryfall.io/normal/front/{e['scryfallId'][0]}/"
        f"{e['scryfallId'][1]}/{e['scryfallId']}.jpg"
    )
    crops = "".join(
        f'<div class="crop" style="{position((left, top, right, bottom))}"></div>'
        for top, bottom, left, right in (NAME_ROI, COLLECTOR_INFORMATION_ROI)
    )
    boxes = "".join(
        f'<div class="box" style="{position(b["box"])}border-color:{DETAILS[b["detail"]][1]}"'
        f' title="{html.escape(b["detail"])}: {html.escape(b["text"])}"></div>'
        for b in e["boxes"]
    )
    title = html.escape(f"{e['name']} ({e['printing']})")
    return (
        f'<div class="card" title="{title}"><img loading="lazy" src="{url}" alt="{title}">'
        f"{crops}{boxes}</div><br><small>{title}</small>"
    )


def position(box) -> str:
    left, top, right, bottom = box
    return (
        f"left:{100 * left:.2f}%;top:{100 * top:.2f}%;"
        f"width:{100 * (right - left):.2f}%;height:{100 * (bottom - top):.2f}%;"
    )


TRAPS = """<ul class="traps">
<li><b>The catalog isn't what the card shows.</b> A List card is <code>PLST HOP-128</code>,
base set size 5,075 in the catalog, but prints <code>HOP 128/169</code>. A prerelease promo
<code>PKHM 214p</code> prints <code>KHM 214/285</code>.</li>
<li><b>The year on the card isn't the set's year.</b> Secret Lair prints 2023 to 2026, the
catalog says 2019; Time Spiral Remastered prints 2020, the catalog says 2021.</li>
<li><b>The frame doesn't tell the collector style.</b> Retro reprints (The List,
Remasters, 30A, Secret Lair) use an old frame with modern details: a 1997-frame card from
2023 prints a collector number, but no base set size.</li>
<li><b>Two-name cards have one name in the catalog.</b> Both rows of
<code>Bonecrusher Giant // Stomp</code> carry the full name, the same number and the same
image.</li>
<li><b>The big name may not be the card's name.</b> On flavor-name cards (2.9%) the large
name is the flavor name; the real name is printed small below it, outside the name
crop.</li>
<li><b>Other text sits in the collector information crop.</b> The last line of flavor text,
the MDFC hint bar, licensing lines (MARVEL), and the grey back-face power/toughness on
transform cards.</li>
<li><b>Many printings show exactly the same details.</b> An expansion card and its
prerelease, promo-pack and List versions differ only by a stamp: 29% of paper printings
can only be told apart by the picture (issue #202).</li>
</ul>"""


# ---- Markdown


def markdown_tables(summary: dict) -> str:
    out = []
    for table_name, columns in (("A", COLUMNS_A), ("B", COLUMNS_B)):
        groups = [g for g in summary.values() if g["table"] == table_name]
        first = (
            ["Frame", "Collector style", "From", "To"]
            if table_name == "A"
            else ["Layout"]
        )
        details = [DETAILS[d][0] for d in columns]
        out.append(f"\n### Table {table_name}\n")
        out.append("| " + " | ".join([*first, "Share", *details]) + " |")
        out.append("|" + " --- |" * (len(first) + 1 + len(columns)))
        for g in groups:
            if table_name == "A":
                style = g["style"] + (f" (only {g['only']})" if g["only"] else "")
                group = [g["frame"], style, g["from"], g["to"] or "today"]
            else:
                group = [g["name"]]
            cells = [md_cell(g["details"][d]) for d in columns]
            share = f"{g['share_percent']:.1f}% ({g['rows']:,})"
            out.append("| " + " | ".join([*group, share, *cells]) + " |")
    return "\n".join(out)


def md_cell(d: dict) -> str:
    text = d["label"]
    if d["label"] != "always" or d["shown_percent"] < 100:
        text += f" ({d['shown_percent']:.0f}%)"
    if d["label"] == "sometimes" and d["reasons"]:
        text += f", {reason_text(d['reasons'][0][0])}"
    if d["label"] != "never":
        text += f"; {d['crop']}"
        if d["printed_samples"]:
            text += f", read {d['read']}/{d['printed_samples']}"
    return text


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
