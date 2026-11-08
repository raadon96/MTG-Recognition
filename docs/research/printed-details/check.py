"""Throwaway: check the rules in rules.py against what OCR read.

Two sets of readings:
- positions.json from #193 (540 whole cards in 27 groups); the collector
  information tokens are those whose box starts at 0.85 of the card's height or
  lower, plus tokens read on a rotated card.
- survey.json (survey.py): the bottom 15% of up to 4 normal-layout cards per set
  and frame.

For each reading and each detail (set code, collector number, base set size,
copyright year), two values are compared with the tokens: the value the rules
derive, and the value the catalog holds today (the one scoring compares against).
"token" means the tokens contain the value (normalised, see the matchers);
"extracted" means the production extractor (engine/ocr/ocr_extraction.py) returns
it. For details the rules say aren't printed, "found anyway" counts readings where
the tokens hold such a detail all the same.

uv run python docs/research/printed-details/check.py \
    <AllPrintings.sqlite> <catalog.sqlite> <positions.json> <survey.json> <out.json>
"""

import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from rules import printed_details, printed_sources

from mtg_recognition.engine.ocr import ocr_extraction

DETAILS = ["set_code", "number", "base_set_size", "year"]
PT_TOKEN = re.compile(r"^\s*[\dX*+]{1,2}\s*[/Il|]\s*[\dX*+]{1,2}\s*$")
YEAR = re.compile(r"(?<!\d)(19[89]\d|20[0-3]\d)(?!\d)")
RARITY = "CURMSTLP"


def main(
    allprintings: str, catalog: str, positions: str, survey: str, out_path: str
) -> None:
    ap = sqlite3.connect(f"file:{allprintings}?mode=ro", uri=True)
    ap.row_factory = sqlite3.Row
    sets = {r["code"]: dict(r) for r in ap.execute("select * from sets")}
    cards = {
        r["uuid"]: dict(r)
        for r in ap.execute(
            """select uuid, setCode, number, frameVersion, originalReleaseDate,
                      layout, promoTypes, name, artist
               from cards"""
        )
    }
    sources = printed_sources(list(cards.values()), sets)
    cat = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True)
    catalog_values = {
        uuid: {"set_code": code, "number": number, "base_set_size": size, "year": year}
        for uuid, code, number, size, year in cat.execute(
            "select uuid, setCode, number, baseSetSize, setReleaseYear from catalog"
        )
    }

    # readings: (source, group, uuid, collector information tokens)
    readings = []
    for group, data in json.loads(Path(positions).read_text()).items():
        for s in data["samples"]:
            texts = [
                t["text"]
                for t in s["tokens"]
                if t["orientation"] != "up" or t["box"][1] >= 0.85
            ]
            readings.append(("positions", group, s["uuid"], texts))
    for s in json.loads(Path(survey).read_text()):
        readings.append(("survey", None, s["uuid"], s["texts"]))

    results = []
    for source, group, uuid, texts in readings:
        card = cards[uuid]
        derived = printed_details(card, sets, sources)
        extracted = extract(texts)
        row = {
            "source": source,
            "group": group,
            "uuid": uuid,
            "name": card["name"],
            "printing": f"{card['setCode']} {card['number']}",
            "frame": card["frameVersion"],
            "texts": texts,
            "extracted": extracted,
        }
        for detail in DETAILS:
            value = derived[detail]
            catalog_value = catalog_values[uuid][detail]
            row[detail] = {
                "rule": derived["rule"][detail],
                "derived": value,
                "catalog": catalog_value,
                "token_derived": value is not None
                and found(detail, value, derived["number"], texts),
                "token_catalog": catalog_value is not None
                and found(detail, str(catalog_value), card["number"], texts),
                "extracted_derived": value is not None
                and extracted_equals(detail, value, extracted),
                "extracted_catalog": catalog_value is not None
                and extracted_equals(detail, str(catalog_value), extracted),
                "found_anyway": value is None and found_anyway(detail, card, texts),
            }
        results.append(row)

    Path(out_path).write_text(json.dumps(results, indent=0, ensure_ascii=False))
    report(results)


def normalise(text: str) -> str:
    # OCR reads 0 as O or o next to digits ("199o", "6o/350", "002/oo8")
    while (fixed := re.sub(r"(?<=\d)[Oo]|[Oo](?=\d)", "0", text)) != text:
        text = fixed
    return text


def found(detail: str, value: str, number: str | None, texts: list[str]) -> bool:
    """Whether the tokens hold `value` for `detail`; a base set size must follow
    `number`."""
    tokens = [normalise(t) for t in texts]
    match detail:
        case "set_code":
            code = normalise(value.upper())
            return any(
                re.search(
                    rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])|"
                    rf"(?<![A-Z0-9]){re.escape(code)}(?=EN|JP)",
                    t.upper(),
                )
                for t in tokens
            )
        case "number":
            return any(number_in(value, t) for t in tokens)
        case "base_set_size":
            digits = re.search(r"\d+", (number or "").split("-")[-1])
            if not digits:
                return False
            digits = str(int(digits.group()))
            # the slash is often read as 1, I, l or ), or dropped ("751350")
            pattern = rf"(?<!\d)0*{digits}[a-z★]?\s*[/|Il1\\)]?\s*0*{value}(?!\d)"
            pairs = zip(tokens, tokens[1:], strict=False)
            return any(re.search(pattern, t) for t in tokens) or any(
                re.search(pattern, a + b) for a, b in pairs
            )
        case "year":
            return any(value in YEAR.findall(t) for t in tokens)
    raise ValueError(detail)


def number_in(digits: str, token: str) -> bool:
    """Whether `token` holds the collector number `digits`, not as part of P/T or
    some other number."""
    if not digits.isdigit() or PT_TOKEN.match(token):
        return False
    for m in re.finditer(rf"(?<!\d)0*{int(digits)}(?!\d)", token):
        before, after = token[: m.start()], token[m.end() :]
        if (
            after.lstrip().startswith("/")
            or len(m.group()) >= 3
            or (before.rstrip()[-1:] and before.rstrip()[-1] in RARITY)
            or (after.lstrip()[:1] and after.lstrip()[0] in RARITY)
            or token.strip() == m.group()
            # older frames glue the number to the copyright line
            or re.search(r"(coast|inc|llc)\W*$", before, re.IGNORECASE)
        ):
            return True
    # number and base set size read as one run of digits at the end of the
    # copyright line, the slash lost or read as 1 ("751350" for 75/350)
    run = re.search(r"(?:coast|inc|llc)\W*(\d{3,8})\W*$", token, re.IGNORECASE)
    return bool(run) and run.group(1).lstrip("0").startswith(str(int(digits)))


def found_anyway(detail: str, card: dict, texts: list[str]) -> bool:
    """For a detail the rules say isn't printed: whether the tokens hold it."""
    tokens = [normalise(t) for t in texts]
    digits = re.search(r"\d+", card["number"].split("-")[-1])
    match detail:
        case "set_code":
            # a set code with its language, the 2015 frame's form
            return any(re.search(r"[A-Z0-9]{2,5}\s*[·★*]\s*(EN|JP)", t) for t in tokens)
        case "number":
            return bool(digits) and any(number_in(digits.group(), t) for t in tokens)
        case "base_set_size":
            return bool(digits) and any(
                re.search(rf"(?<!\d)0*{int(digits.group())}\s*/\s*\d{{2,4}}", t)
                for t in tokens
            )
        case "year":
            return any(YEAR.search(t) for t in tokens)
    raise ValueError(detail)


def extract(texts: list[str]) -> dict:
    number, size = ocr_extraction.extract_collector_number_and_base_set_size(texts)
    return {
        "set_code": ocr_extraction.extract_possible_set_codes(texts),
        "number": number,
        "base_set_size": size,
        "year": ocr_extraction.extract_copyright_year(texts),
    }


def extracted_equals(detail: str, value: str, extracted: dict) -> bool:
    """Whether the production extractor read exactly `value` (a field score of
    100)."""
    read = extracted[detail]
    if read is None:
        return False
    if detail == "set_code":
        return value in read
    if detail == "number":
        return re.sub(r"^0+(?=\d)", "", read) == re.sub(r"^0+(?=\d)", "", value)
    return read == value


def report(results: list[dict]) -> None:
    # per #193 group: readings holding the derived value (of those the rules say
    # are printed) and the catalog's value (of all readings)
    rows = [r for r in results if r["source"] == "positions"]
    groups = list(dict.fromkeys(r["group"] for r in rows))
    print("group | n | " + " | ".join(f"{d} rule, catalog" for d in DETAILS))
    for group in groups:
        items = [r for r in rows if r["group"] == group]
        cells = []
        for d in DETAILS:
            printed = [r for r in items if r[d]["derived"] is not None]
            cells.append(
                f"{sum(r[d]['token_derived'] for r in printed)}/{len(printed)}, "
                f"{sum(r[d]['token_catalog'] for r in items)}/{len(items)}"
            )
        print(f"{group} | {len(items)} | " + " | ".join(cells))

    for source in ("positions", "survey"):
        rows = [r for r in results if r["source"] == source]
        print(f"\n### {source}: {len(rows)} readings")
        # totals: where the rules derive a value, how often the tokens and the
        # extractor hold it, against the catalog's value; where they don't, how
        # often the catalog's value is in the tokens all the same
        for detail in DETAILS:
            printed = [r[detail] for r in rows if r[detail]["derived"] is not None]
            differs = [p for p in printed if str(p["catalog"]) != p["derived"]]
            left_out = [r[detail] for r in rows if r[detail]["derived"] is None]
            print(
                f"{detail}: derived {len(printed)}:"
                f" token {count(printed, 'token_derived')}"
                f" vs catalog {count(printed, 'token_catalog')}, extracted"
                f" {count(printed, 'extracted_derived')} vs"
                f" {count(printed, 'extracted_catalog')}; of these differing from the"
                f" catalog {len(differs)}: token {count(differs, 'token_derived')} vs"
                f" {count(differs, 'token_catalog')}; left out {len(left_out)}: catalog"
                f" value in tokens {count(left_out, 'token_catalog')}"
            )
        for detail in DETAILS:
            print(
                f"\n{detail}: rule | n | token derived | token catalog | "
                "extracted derived | extracted catalog | found anyway"
            )
            by_rule = defaultdict(list)
            for r in rows:
                by_rule[r[detail]["rule"]].append(r[detail])
            for rule, items in sorted(by_rule.items(), key=lambda kv: -len(kv[1])):
                print(
                    f"  {rule} | {len(items)} | {count(items, 'token_derived')} | "
                    f"{count(items, 'token_catalog')} | "
                    f"{count(items, 'extracted_derived')} | "
                    f"{count(items, 'extracted_catalog')} | "
                    f"{count(items, 'found_anyway')}"
                )


def count(items: list[dict], key: str) -> int:
    return sum(i[key] for i in items)


if __name__ == "__main__":
    main(*sys.argv[1:6])
