"""Throwaway: where each detail sits on sampled reference images, per frame and
layout, read with the production PaddleOCR model on the whole card.

For each sampled printing, every OCR token is matched against the printing's own
details (name, set code, collector number, base set size, artist, copyright, P/T)
and its box is recorded as fractions of the card's width and height. Cards with
sideways or upside-down halves are also read rotated, and the boxes are mapped
back onto the upright card.

uv run python docs/research/card-details/ocr_positions.py \
    <AllPrintings.sqlite> <images dir> <out.json> [samples per group]
"""

import json
import random
import re
import sqlite3
import sys
from pathlib import Path

import cv2
from rapidfuzz import fuzz

from mtg_recognition.engine.ocr.reading import load_paddleocr

# group name -> SQL condition on cards c joined with sets s (paper printings only)
GROUPS = {
    "frame 1993": "c.frameVersion = '1993' and c.layout = 'normal'",
    "frame 1997, before Exodus": (
        "c.frameVersion = '1997' and c.layout = 'normal'"
        " and s.releaseDate < '1998-06-15'"
    ),
    "frame 1997, 1998-2003": (
        "c.frameVersion = '1997' and c.layout = 'normal'"
        " and s.releaseDate between '1998-06-15' and '2003-07-27'"
    ),
    "frame 1997, retro reprint (2014+)": (
        "c.frameVersion = '1997' and c.layout = 'normal'"
        " and s.releaseDate >= '2014-07-18' and c.setCode != 'PLST'"
    ),
    "frame 2003": (
        "c.frameVersion = '2003' and c.layout = 'normal'"
        " and s.releaseDate between '2003-07-28' and '2014-07-17'"
    ),
    "frame 2015, 2014-2022": (
        "c.frameVersion = '2015' and c.layout = 'normal'"
        " and s.releaseDate between '2014-07-18' and '2022-12-31'"
        " and c.borderColor != 'borderless'"
    ),
    "frame 2015, 2023+": (
        "c.frameVersion = '2015' and c.layout = 'normal'"
        " and s.releaseDate >= '2023-01-01' and c.borderColor != 'borderless'"
    ),
    "frame 2015, borderless": (
        "c.frameVersion = '2015' and c.layout = 'normal'"
        " and c.borderColor = 'borderless'"
    ),
    "frame future": "c.frameVersion = 'future' and c.layout = 'normal'",
    "The List (PLST)": "c.setCode = 'PLST'",
    "layout transform": "c.layout = 'transform' and c.type not like '%Battle%'",
    "layout transform, battle": "c.layout = 'transform' and c.type like '%Battle%'",
    "layout modal_dfc": "c.layout = 'modal_dfc'",
    "layout adventure": "c.layout = 'adventure' and c.side = 'a'",
    "layout split, 2003 frame or older": (
        "c.layout = 'split' and c.side = 'a' and c.frameVersion != '2015'"
    ),
    "layout split, 2015 frame": (
        "c.layout = 'split' and c.side = 'a' and c.frameVersion = '2015'"
    ),
    "layout aftermath": "c.layout = 'aftermath' and c.side = 'a'",
    "layout flip": "c.layout = 'flip' and c.side = 'a'",
    "layout prepare": "c.layout = 'prepare' and c.side = 'a'",
    "layout saga": "c.layout = 'saga'",
    "layout class": "c.layout = 'class'",
    "layout case": "c.layout = 'case'",
    "layout mutate": "c.layout = 'mutate'",
    "layout leveler": "c.layout = 'leveler'",
    "layout prototype": "c.layout = 'prototype'",
    "layout meld": "c.layout = 'meld'",
    "layout planar": "c.layout = 'planar'",
}

# rotations read besides upright, for layouts with sideways or upside-down text
ROTATIONS = {
    "layout split, 2003 frame or older": ["cw", "ccw"],
    "layout split, 2015 frame": ["cw", "ccw"],
    "layout aftermath": ["cw", "ccw"],
    "layout transform, battle": ["cw", "ccw"],
    "layout planar": ["cw", "ccw"],
    "layout flip": ["180"],
}
CV_ROTATE = {
    "cw": cv2.ROTATE_90_CLOCKWISE,
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}

PT_RE = re.compile(r"^\s*([0-9Xx*+½∞?-]{1,3})\s*/\s*([0-9Xx*+½∞?-]{1,3})\s*$")
YEAR_RE = re.compile(r"(?<!\d)(19[89]\d|20[0-3]\d)(?!\d)")
RARITY_LETTERS = "CURMSTLP"


def main(allprintings: str, images_dir: str, out_path: str, per_group: int) -> None:
    con = sqlite3.connect(f"file:{allprintings}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    images = Path(images_dir)
    ocr = load_paddleocr()
    rng = random.Random(193)
    results = {}
    for group, condition in GROUPS.items():
        rows = con.execute(
            f"""select c.uuid, c.name, c.faceName, c.side, c.number, c.setCode,
                       c.artist, c.power, c.toughness, c.layout, c.frameVersion,
                       c.borderColor, c.frameEffects, s.baseSetSize, s.releaseDate,
                       s.parentCode, s.keyruneCode, i.scryfallId
                from cards c join sets s on c.setCode = s.code
                join cardIdentifiers i on i.uuid = c.uuid
                where c.availability like '%paper%' and {condition}"""
        ).fetchall()
        rows = [r for r in rows if (images / f"{r['uuid']}.jpg").exists()]
        sample = rng.sample(rows, min(per_group, len(rows)))
        print(f"{group}: {len(rows)} paper printings with an image, {len(sample)} read")
        results[group] = {
            "population": len(rows),
            "samples": [
                read_sample(ocr, images / f"{r['uuid']}.jpg", dict(r), group)
                for r in sample
            ],
        }
    Path(out_path).write_text(json.dumps(results, indent=1, ensure_ascii=False))


def read_sample(ocr, image_path: Path, row: dict, group: str) -> dict:
    image = cv2.imread(str(image_path))
    # 2x: the collector information's text is ~7 px tall on a 680 px reference image
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    orientations = ["up", *ROTATIONS.get(group, [])]
    batch = [
        image if o == "up" else cv2.rotate(image, CV_ROTATE[o]) for o in orientations
    ]
    raw = ocr.predict(batch, text_det_thresh=0.05)

    tokens = []
    for orientation, img, result in zip(orientations, batch, raw, strict=True):
        h, w = img.shape[:2]
        for text, box, score in zip(
            result["rec_texts"],
            result["rec_boxes"].tolist(),
            result["rec_scores"],
            strict=True,
        ):
            frac = (box[0] / w, box[1] / h, box[2] / w, box[3] / h)
            details = classify(text, row)
            # rotated reads are only kept when they found a detail
            if orientation != "up" and not details:
                continue
            tokens.append(
                {
                    "text": text,
                    "score": round(score, 3),
                    "orientation": orientation,
                    "box": [round(v, 4) for v in upright(frac, orientation)],
                    "details": details,
                }
            )
    return {**row, "tokens": tokens}


def upright(box: tuple, orientation: str) -> tuple:
    """A box read on a rotated image, as fractions of the upright card."""
    a1, b1, a2, b2 = box
    match orientation:
        case "up":
            return box
        case "cw":
            return (b1, 1 - a2, b2, 1 - a1)
        case "ccw":
            return (1 - b2, a1, 1 - b1, a2)
        case "180":
            return (1 - a2, 1 - b2, 1 - a1, 1 - b1)
    raise ValueError(orientation)


def classify(text: str, row: dict) -> list[str]:
    """Which of the printing's details the token holds."""
    details = []
    t = text.strip()
    upper = re.sub(r"[^A-Z0-9]+", " ", t.upper())

    # name: each face's name, on its own token
    faces = row["name"].split(" // ")
    for i, face in enumerate(faces):
        if len(t) >= 3 and fuzz.ratio(t.lower(), face.lower()) >= 75:
            details.append("name" if i == 0 else "name (other face)")

    # power/toughness: the whole token is P/T and equals the printing's
    # (older frames' slash is often read as I, l or 1)
    pt = PT_RE.match(t)
    if row["power"] is not None and t.replace(" ", "") in {
        f"{row['power']}{sep}{row['toughness']}" for sep in "/Il|1"
    }:
        details.append("power/toughness")
        pt = True

    # collector number, with leading zeros, and base set size after a slash
    digits = re.sub(r"\D", "", row["number"].split("-")[-1])
    if digits and not pt:
        m = re.search(rf"(?<![\d/])0*{digits}(?!\d)", t)
        if m:
            after = t[m.end() :].lstrip()
            before = t[: m.start()].rstrip()
            specific = (
                after.startswith("/")
                or len(digits) >= 3
                or m.group(0).startswith("0")
                or (after[:1] and after[0] in RARITY_LETTERS)
                or (before[-1:] and before[-1] in RARITY_LETTERS)
            )
            if specific:
                details.append("collector number")
                if after.startswith("/"):
                    size = re.match(r"/\s*(\d+)", after)
                    details.append(
                        "base set size"
                        if size and str(row["baseSetSize"]) == size.group(1)
                        else "slash and other size"
                    )

    # set code: the printing's set, its parent (promos), or the List's original set
    codes = {row["setCode"], row["parentCode"], row["keyruneCode"]}
    if "-" in row["number"]:
        codes.add(row["number"].split("-")[0])
    codes.discard(None)
    if any(
        re.search(rf"\b{re.escape(c.upper())}\b", upper) for c in codes if len(c) >= 2
    ):
        details.append("set code")

    # artist, often after "Illus." or the language
    if (
        row["artist"]
        and len(t) >= 4
        and fuzz.partial_ratio(row["artist"].lower(), t.lower()) >= 85
        and not any(d.startswith("name") for d in details)
    ):
        details.append("artist")
    if re.search(r"\billus", t, re.IGNORECASE):
        details.append("illus. label")

    # copyright line
    lower = t.lower()
    if (
        "©" in t
        or "wizards" in lower
        or "coast" in lower
        or re.search(r"\btm\b", lower)
    ):
        details.append("copyright")
        if YEAR_RE.search(t):
            details.append("copyright year")
    return details


if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]) if len(sys.argv) > 4 else 20,
    )
