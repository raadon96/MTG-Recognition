"""Throwaway (#197): candidate-driven extraction. Instead of guessing each field from
the OCR tokens, ask whether each detail a candidate printing shows appears in them.

`candidate_field_scores(tokens, printing, positions)` takes the tokens of the name
crop and the collector information crop (texts and boxes, as PaddleOCR returns
them) and one printing's details, and returns a field score (0-100) per field, or
None for a detail the printing doesn't show. With `positions`, the collector
details are looked for on the bottom lines only, and power/toughness in the
rightmost token only.
"""

import re

from rapidfuzz import fuzz, utils

# OCR reads 0 as O or o next to digits ("199o", "6o/350")
ZERO = re.compile(r"(?<=\d)[Oo]|[Oo](?=\d)")
DIGITS = re.compile(r"\d+")
WORDS = re.compile(r"[A-Z0-9]+")
# the slash between power and toughness is often read as 1, I, l, | or )
PT_SLASH = r"\s*[/|\\Il1)]\s*"
# details sit on the collector information's last two lines: number and set code
# on the 2015 frame, artist and copyright line on older frames
INFO_LINES = 2


def candidate_field_scores(tokens: dict, printing: dict, positions: bool) -> dict:
    p = printing
    shown = p["printed"]
    name_text = utils.default_process(" ".join(tokens["name"]["texts"]))
    info = tokens["collector_information"]
    texts = [_fix_zeros(t) for t in info["texts"]]
    boxes = info["boxes"]

    # power/toughness: the rightmost token above the bottom line (the copyright
    # line reaches further right); the other details then skip it
    pt_tokens = texts
    if positions:
        lines_idx = _lines(boxes)
        if p["power"] is not None and len(lines_idx) > 1:
            above = [i for line in lines_idx[:-1] for i in line]
            pt = max(above, key=lambda i: boxes[i][2])
            pt_tokens = [texts[pt]]
            lines_idx = [[i for i in line if i != pt] for line in lines_idx]
        lines = [[texts[i] for i in line] for line in lines_idx if line][-INFO_LINES:]
    else:
        lines = [texts]

    # number runs, line by line in reading order: "0254/281" -> 254, 281
    runs = [
        [r.lstrip("0") or "0" for t in line for r in DIGITS.findall(t)]
        for line in lines
    ]
    words = [w for line in lines for t in line for w in WORDS.findall(t.upper())]
    text = utils.default_process(" ".join(t for line in lines for t in line))

    scores: dict[str, float | None] = {}
    # the name: the closest of the card's names (faces, flavor name)
    scores["name"] = max(
        fuzz.ratio(utils.default_process(n), name_text) for n in p["names"]
    )
    # the set code, alone or glued to the language ("THBDE", "WAREN")
    code = shown["set_code"]
    scores["set_code"] = (
        None
        if code is None
        else 100.0
        * any(
            w == code or (w.startswith(code) and len(w) == len(code) + 2) for w in words
        )
    )
    number = shown["number"]
    scores["number"] = (
        None if number is None else 100.0 * any(number in line for line in runs)
    )
    # the base set size right after the number, or glued to it when the slash is
    # lost or read as 1 ("751350", "7511350")
    size = shown["base_set_size"]
    scores["base_set_size"] = (
        None
        if size is None
        else 100.0
        * any(
            (a, b) == (number, size) or re.fullmatch(rf"{number}1?0*{size}", a)
            for line in runs
            for a, b in zip(line, [*line[1:], ""], strict=False)
        )
    )
    year = shown["year"]
    scores["year"] = (
        None if year is None else 100.0 * any(year in line for line in runs)
    )
    artist = p["artist"]
    scores["artist"] = (
        None
        if artist is None
        else fuzz.partial_ratio(utils.default_process(artist), text)
    )
    if p["power"] is None:
        scores["power"] = scores["toughness"] = None
    else:
        pt = re.compile(
            rf"(?<![\dX*]){re.escape(p['power'])}{PT_SLASH}{re.escape(p['toughness'])}(?!\d)"
        )
        hit = 100.0 * any(pt.search(t) for t in pt_tokens)
        scores["power"] = scores["toughness"] = hit
    return scores


def _fix_zeros(text: str) -> str:
    while (fixed := ZERO.sub("0", text)) != text:
        text = fixed
    return text


def _lines(boxes: list) -> list[list[int]]:
    """Token indices grouped into lines, top to bottom, each left to right: a token
    starts a new line when its vertical centre is below the current line's bottom."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    lines: list[list[int]] = []
    bottom = -1.0
    for i in order:
        centre = (boxes[i][1] + boxes[i][3]) / 2
        if not lines or centre > bottom:
            lines.append([])
            bottom = boxes[i][3]
        lines[-1].append(i)
    return [sorted(line, key=lambda i: boxes[i][0]) for line in lines]
