# ruff: noqa: E501 -- throwaway research script; its SQL and HTML strings stay on one line
"""Throwaway: summarise ocr_positions.py's output per group, and write the HTML page.

uv run python docs/research/card-details/summarise.py \
    <AllPrintings.sqlite> <positions.json> <summary.json> <out.html>
"""

import html
import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

from ocr_positions import GROUPS

# (left, top, right, bottom) fractions, from reading.py's (top, bottom, left, right)
NAME_ROI = (0.06, 0.04, 0.9, 0.11)
COLLECTOR_INFORMATION_ROI = (0.05, 0.88, 0.95, 1.0)

DETAILS = {
    "name": "#e6194b",
    "name (other face)": "#f58231",
    "set code": "#3cb44b",
    "collector number": "#4363d8",
    "base set size": "#911eb4",
    "artist": "#42d4f4",
    "copyright year": "#f032e6",
    "power/toughness": "#bfef45",
}
DOUBLE_FACED_LAYOUTS = {"transform", "modal_dfc", "reversible_card"}
RARITY_FIRST = re.compile(r"^[CURMSLTP]\s?\d{3,4}$")
YEAR = re.compile(r"(?<!\d)(19[89]\d|20[0-3]\d)(?!\d)")


def main(
    allprintings: str, positions_path: str, summary_path: str, html_path: str
) -> None:
    positions = json.loads(Path(positions_path).read_text())
    con = sqlite3.connect(f"file:{allprintings}?mode=ro", uri=True)
    total = con.execute("select count(*) from cards").fetchone()[0]

    summary = {}
    for group, data in positions.items():
        samples = data["samples"]
        # share of the catalog: every row, both faces, online-only included
        condition = GROUPS[group].replace(" and c.side = 'a'", "")
        rows = con.execute(
            f"select count(*) from cards c join sets s on c.setCode = s.code where {condition}"
        ).fetchone()[0]
        summary[group] = {
            "catalog_rows": rows,
            "catalog_share": 100 * rows / total,
            "paper_with_image": data["population"],
            "read": len(samples),
            "details": {d: detail_stats(samples, d) for d in DETAILS},
            "traps": trap_stats(samples),
        }
    Path(summary_path).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    Path(html_path).write_text(page(positions, summary))


def detail_stats(samples: list[dict], detail: str) -> dict:
    """How often the detail was read, where, and whether it falls in today's crop."""
    roi = NAME_ROI if detail.startswith("name") else COLLECTOR_INFORMATION_ROI
    found, found_upright, inside, boxes = 0, 0, 0, []
    for s in samples:
        tokens = [t for t in s["tokens"] if detail in t["details"]]
        if not tokens:
            continue
        found += 1
        upright = [t for t in tokens if t["orientation"] == "up"]
        if upright:
            found_upright += 1
            # the occurrence nearest today's crop: the same name in rules text
            # doesn't count against the name crop
            best = max(upright, key=lambda t: inside_share(t["box"], roi))
            boxes.append(best["box"])
            inside += inside_share(best["box"], roi) >= 0.9
        else:
            boxes.append(tokens[0]["box"])
    median = (
        [round(statistics.median(b[i] for b in boxes), 3) for i in range(4)]
        if boxes
        else None
    )
    return {
        "found": found,
        "found_upright": found_upright,
        "inside_crop": inside,
        "median_box": median,
    }


def inside_share(box: list[float], roi: tuple) -> float:
    """Share of the box's area inside roi."""
    x1, y1, x2, y2 = box
    ix = max(0.0, min(x2, roi[2]) - max(x1, roi[0]))
    iy = max(0.0, min(y2, roi[3]) - max(y1, roi[1]))
    area = (x2 - x1) * (y2 - y1)
    return ix * iy / area if area > 0 else 0.0


def trap_stats(samples: list[dict]) -> dict:
    year_read, year_differs, other_size, rarity_first, other_text = 0, [], 0, 0, []
    for s in samples:
        upright = [t for t in s["tokens"] if t["orientation"] == "up"]
        years = [
            int(y)
            for t in upright
            if "copyright" in t["details"]
            for y in YEAR.findall(t["text"])
        ]
        if years:
            year_read += 1
            if str(max(years)) != s["releaseDate"][:4]:
                year_differs.append(
                    f"{s['setCode']} {s['releaseDate'][:4]} printed {max(years)}"
                )
        other_size += any("slash and other size" in t["details"] for t in upright)
        rarity_first += any(RARITY_FIRST.match(t["text"].strip()) for t in upright)
        # text in the collector information crop that is none of the details
        for t in upright:
            if (
                not t["details"]
                and len(t["text"].strip()) >= 3
                and inside_share(t["box"], COLLECTOR_INFORMATION_ROI) >= 0.9
            ):
                other_text.append(t["text"])
    return {
        "copyright_year_read": year_read,
        "copyright_year_differs_from_set_year": year_differs,
        "printed_size_differs_from_base_set_size": other_size,
        "rarity_before_number": rarity_first,
        "samples_with_other_text_in_collector_crop": sum(
            any(
                not t["details"]
                and len(t["text"].strip()) >= 3
                and inside_share(t["box"], COLLECTOR_INFORMATION_ROI) >= 0.9
                for t in s["tokens"]
                if t["orientation"] == "up"
            )
            for s in samples
        ),
        "other_text_examples": other_text[:8],
    }


def image_url(sample: dict) -> str:
    sid = sample["scryfallId"]
    face = (
        "back"
        if sample["layout"] in DOUBLE_FACED_LAYOUTS and sample["side"] == "b"
        else "front"
    )
    return f"https://cards.scryfall.io/normal/{face}/{sid[0]}/{sid[1]}/{sid}.jpg"


def representative(samples: list[dict]) -> dict:
    """The sample on which the most details were read."""
    # gold-bordered and silver-bordered cards aren't typical of their frame
    return max(
        samples,
        key=lambda s: (
            s["borderColor"] in ("black", "white", "borderless"),
            len({d for t in s["tokens"] for d in t["details"] if d in DETAILS}),
        ),
    )


def box_div(box: list[float], colour: str, label: str, dashed: bool = False) -> str:
    x1, y1, x2, y2 = box
    style = (
        f"left:{x1 * 100:.2f}%;top:{y1 * 100:.2f}%;"
        f"width:{(x2 - x1) * 100:.2f}%;height:{(y2 - y1) * 100:.2f}%;"
        f"border:2px {'dashed' if dashed else 'solid'} {colour};"
    )
    return f'<div class="box" style="{style}" title="{html.escape(label)}"></div>'


def page(positions: dict, summary: dict) -> str:
    cards = []
    for group, data in positions.items():
        s = representative(data["samples"])
        overlays = [
            box_div(list(NAME_ROI), "#000", "today's name crop", dashed=True),
            box_div(
                list(COLLECTOR_INFORMATION_ROI),
                "#000",
                "today's collector information crop",
                dashed=True,
            ),
        ]
        for t in s["tokens"]:
            for d in t["details"]:
                if d in DETAILS:
                    rotated = (
                        ""
                        if t["orientation"] == "up"
                        else f" (read rotated {t['orientation']})"
                    )
                    overlays.append(
                        box_div(t["box"], DETAILS[d], f"{d}: {t['text']}{rotated}")
                    )
                    break
        cards.append(
            f"""<figure><div class="card"><img loading="lazy" src="{image_url(s)}" alt="">{"".join(overlays)}</div>
<figcaption><b>{html.escape(group)}</b><br>{html.escape(s["name"])} ({s["setCode"]} {s["number"]}, {s["releaseDate"][:4]})</figcaption></figure>"""
        )

    head = "".join(f"<th>{html.escape(d)}</th>" for d in DETAILS)
    rows = []
    for group, g in summary.items():
        cells = []
        for d in DETAILS:
            st = g["details"][d]
            n = g["read"]
            if st["found"] == 0:
                cells.append('<td class="none">not read</td>')
                continue
            rotated = st["found"] - st["found_upright"]
            share_inside = st["inside_crop"] / st["found"]
            cls = "in" if share_inside >= 0.9 else "part" if share_inside > 0 else "out"
            note = f"<br><small>{rotated} only sideways</small>" if rotated else ""
            cells.append(
                f'<td class="{cls}">read {st["found"]}/{n}<br>in crop {st["inside_crop"]}/{st["found"]}{note}</td>'
            )
        rows.append(
            f"<tr><th>{html.escape(group)}</th><td>{g['catalog_share']:.2f}%<br><small>{g['catalog_rows']} rows</small></td>{''.join(cells)}</tr>"
        )
    legend = "".join(
        f'<span class="key" style="border-color:{c}">{html.escape(d)}</span>'
        for d, c in DETAILS.items()
    )
    return TEMPLATE.format(
        legend=legend,
        cards="\n".join(cards),
        head=head,
        rows="\n".join(rows),
        traps=(Path(__file__).parent / "traps.html").read_text(),
    )


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Where each detail sits, per frame and layout (#193)</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2em; max-width: 1400px; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 18px; }}
figure {{ margin: 0; width: 244px; font-size: 12px; }}
.card {{ position: relative; width: 244px; height: 340px; background: #ddd; }}
.card img {{ width: 100%; height: 100%; display: block; }}
.box {{ position: absolute; box-sizing: border-box; }}
.key {{ border: 3px solid; padding: 2px 6px; margin: 2px; display: inline-block; font-size: 13px; }}
.key.crop {{ border-style: dashed; border-color: #000; }}
table {{ border-collapse: collapse; font-size: 12px; margin-top: 1em; }}
td, th {{ border: 1px solid #aaa; padding: 3px 5px; vertical-align: top; }}
td.in {{ background: #d9f2d9; }} td.part {{ background: #fff2c2; }}
td.out {{ background: #f9d3d3; }} td.none {{ background: #eee; color: #888; }}
</style></head><body>
<h1>Where each detail sits, per frame and layout</h1>
<p>Research for issue #193; notes in <code>docs/research/card-details.md</code>.
Boxes are the production PP-OCRv5 model's text lines on 20 sampled reference images per group
(read on the whole card at 2x, sideways layouts also read rotated), matched against each printing's own details.
Dashed black boxes are today's two crops (<code>NAME_ROI</code>, <code>COLLECTOR_INFORMATION_ROI</code>).
One text line can hold several details (e.g. "OGW*EN LIUS LASAHIDO"); each box takes the colour of its first detail. Hover a box for its text.</p>
<p>{legend} <span class="key crop">today's crops</span></p>
<div class="grid">
{cards}
</div>
<h2>Which details were read, and whether they fall inside today's crops</h2>
<p>"read a/n": samples on which OCR found the detail anywhere on the card (a detail not read is either not printed or misread: see the notes).
"in crop b/a": of those, how many had it at least 90% inside today's crop (the name crop for names, the collector information crop for the rest).
Green: always inside; yellow: sometimes; red: never. Share of catalog counts every row, both faces and online-only printings included.</p>
<table><tr><th>group</th><th>share of catalog</th>{head}</tr>
{rows}
</table>
<h2>Traps</h2>
{traps}
</body></html>
"""

if __name__ == "__main__":
    main(*sys.argv[1:5])
