"""Throwaway prototype for the ticket "Prototype extraction: current,
candidate-driven, and with positions" (#197) of the map #189. Never merged.

It replays the OCR tokens saved by scripts/ocr_compare.py (#196) through three ways
of extracting a reading, so no OCR runs:

1. current: today's reading (engine/ocr/ocr_extraction.py), saved with the tokens.
2. candidate: for each candidate printing, whether each detail the card shows
   appears in the raw text (extraction_candidate.py).
3. positions: as 2, with relative positions (bottom lines, rightmost token).

Each is scored two ways: "all", a weighted average over every field with a fixed
denominator (today's `rank`; a field the card doesn't show scores 0), and "shown",
a weighted average over the fields the candidate shows only.

    uv run python scripts/extraction.py prepare
    uv run python scripts/extraction.py report RUN [RUN ...] [--subset all|tune|confirm]

`prepare` writes output/extraction/cases.json: every printing the report needs
(printed values from the rules of #201 and the catalog's values), the other
printings of each sampled card's name, and the pHash candidates of each benchmark
card image. `report` prints the tables and writes output/extraction/report-*.json.
"""

import argparse
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

from extraction_candidate import candidate_field_scores
from rapidfuzz import fuzz, utils

SAMPLE_PATH = Path("scripts/ocr_sample.json")
READINGS_ROOT = Path("output/ocr_compare")
OUT = Path("output/extraction")
BENCHMARK_CARDS = Path("output/measure/g-11s-dyn-640-0.05")
ALLPRINTINGS = "data/reference/mtgjson/AllPrintings.sqlite"
CATALOG = "data/reference/catalog.sqlite"
SEED = 189
FIELDS = (
    "name", "set_code", "number", "base_set_size", "artist", "year", "power",
    "toughness",
)  # fmt: skip
# today's weights (engine/ocr/scoring.py)
WEIGHTS = {f: 1.0 for f in FIELDS} | {"name": 2.0, "artist": 0.5}
VARIANTS = (
    ("current, catalog values, all", "current", "catalog", "all"),
    ("current, printed values, all", "current", "printed", "all"),
    ("current, printed values, shown", "current", "printed", "shown"),
    ("candidate, all", "candidate", "printed", "all"),
    ("candidate, shown", "candidate", "printed", "shown"),
    ("positions, all", "positions", "printed", "all"),
    ("positions, shown", "positions", "printed", "shown"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    report_parser = commands.add_parser("report")
    report_parser.add_argument("runs", nargs="+")
    report_parser.add_argument(
        "--subset", choices=["all", "tune", "confirm"], default="all"
    )
    # #198: only these variants, and drop tokens below a recognition score, as
    # text_rec_score_thresh would (the runs keep every token: its default is 0)
    report_parser.add_argument("--variant", action="append")
    report_parser.add_argument("--min-rec-score", type=float, default=0.0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        report(args.runs, args.subset, args.variant, args.min_rec_score)


def prepare() -> None:
    import cv2
    from printed_rules import printed_details, printed_sources

    from mtg_recognition.engine.matching.phash import load_phashes, shortlist
    from mtg_recognition.reference.layout import ReferenceData

    sample = json.loads(SAMPLE_PATH.read_text())

    # pHash candidates of each benchmark card image, as the pipeline shortlists them
    phashes = load_phashes(ReferenceData.default())
    phash = {}
    for e in sample:
        if e["source"] == "benchmark":
            uuids, _ = shortlist(phashes, cv2.imread(e["path"]), k=20)
            phash[e["path"]] = uuids

    # benchmark query images: each card image's sampled path, or, for a card the
    # sample leaves out (its correct printing isn't a candidate), the printing the
    # #195 run ranked first, which no extraction can change
    manifest = json.loads(Path("data/benchmark/manifest.json").read_text())
    details = json.loads((BENCHMARK_CARDS / "images.json").read_text())
    sampled = {e["path"] for e in sample}
    images = {}
    for image, label in manifest.items():
        card_dir = BENCHMARK_CARDS / "cards" / Path(image).with_suffix("")
        cards = []
        for i, card in enumerate(details[image]["cards"]):
            path = str(card_dir / f"{i}.png")
            cards.append(
                {"path": path}
                if path in sampled
                else {"fixed": card.get("first") if card["candidates"] else None}
            )
        images[image] = {"label": label, "cards": cards}

    # the other printings of each sampled card's name, from the catalog
    db = sqlite3.connect(f"file:{CATALOG}?mode=ro", uri=True)
    name_of = dict(db.execute("select uuid, name from catalog"))
    by_name: dict[str, list[str]] = {}
    for uuid, name in sorted(name_of.items()):
        by_name.setdefault(name, []).append(uuid)
    same_name = {e["path"]: by_name[name_of[e["uuid"]]] for e in sample}

    # every printing needed: printed values by the rules of #201, catalog values
    needed = {e["uuid"] for e in sample}
    needed |= {u for us in phash.values() for u in us}
    needed |= {u for us in same_name.values() for u in us}
    needed |= {u for i in images.values() for u in i["label"]}
    needed |= {
        c["fixed"] for i in images.values() for c in i["cards"] if c.get("fixed")
    }
    catalog_rows = {
        r[0]: r
        for r in db.execute(
            """select uuid, name, setCode, number, baseSetSize, artist,
                      setReleaseYear, power, toughness from catalog"""
        )
        if r[0] in needed
    }
    ap = sqlite3.connect(f"file:{ALLPRINTINGS}?mode=ro", uri=True)
    ap.row_factory = sqlite3.Row
    sets = {r["code"]: dict(r) for r in ap.execute("select * from sets")}
    cards = [
        dict(r)
        for r in ap.execute(
            """select uuid, setCode, number, frameVersion, originalReleaseDate,
                      layout, promoTypes, name, artist, flavorName, faceFlavorName,
                      side from cards"""
        )
    ]
    sources = printed_sources(cards, sets)
    printings = {}
    for card in cards:
        if card["uuid"] not in needed:
            continue
        row = catalog_rows[card["uuid"]]
        shown = printed_details(card, sets, sources)
        # the large name on the card: the flavor name when there is one
        flavor = card["faceFlavorName"] or card["flavorName"]
        printings[card["uuid"]] = {
            "name": row[1],
            "names": row[1].split(" // ") + ([flavor] if flavor else []),
            "layout": card["layout"],
            # the plain printing among those showing the same details: it prints
            # its own set's details, not another printing's (The List, promos
            # reprinting a parent set's card), and its number has no suffix
            # (not 254★, 71z, 240a)
            "own": card["uuid"] not in sources and card["number"].isdigit(),
            "artist": row[5],
            "power": row[7],
            "toughness": row[8],
            "printed": {
                k: shown[k] for k in ("set_code", "number", "base_set_size", "year")
            },
            "catalog": {
                "set_code": row[2],
                "number": row[3],
                "base_set_size": row[4],
                "year": row[6],
            },  # fmt: skip
        }
    missing = needed - printings.keys()
    print(f"{len(printings)} printings, {len(missing)} not in AllPrintings")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases.json").write_text(
        json.dumps({
            "printings": printings, "same_name": same_name, "phash": phash,
            "images": images,
        }) + "\n"
    )  # fmt: skip


def report(
    runs: list[str], subset: str, variants: list[str] | None, min_rec_score: float
) -> None:
    cases = json.loads((OUT / "cases.json").read_text())
    printings = cases["printings"]
    tables = {}
    for run in runs:
        readings = json.loads((READINGS_ROOT / run / "readings.json").read_text())
        if min_rec_score > 0:
            for r in readings:
                for t in r["tokens"].values():
                    keep = [i for i, s in enumerate(t["scores"]) if s >= min_rec_score]
                    for k in ("texts", "scores", "boxes"):
                        t[k] = [t[k][i] for i in keep]
        by_path = {r["path"]: r for r in readings}
        picked = [
            r for r in readings if subset == "all" or r["half"] in (subset, "both")
        ]
        table = {}
        for label, method, values, average in VARIANTS:
            if variants and label not in variants:
                continue

            def score(r, uuid, method=method, values=values, average=average):
                # printings showing the same details tie on printed values: prefer
                # the one printing its own details, the original over a reprint
                p = printings[uuid]
                tie_break = p["own"] if values == "printed" else False
                return (total(field_scores(method, values, r, p), average), tie_break)

            table[label] = {
                "benchmark": benchmark(cases, by_path, picked, printings, score),
                "same_name": same_name(cases, picked, printings, score),
                "fields": correct_fields(method, values, picked, printings),
            }
        tables[run] = table
        print_run(run, table)
    suffix = f"-rec{min_rec_score}" if min_rec_score > 0 else ""
    (OUT / f"report-{subset}{suffix}.json").write_text(
        json.dumps(tables, indent=1) + "\n"
    )


def field_scores(method: str, values: str, r: dict, p: dict) -> dict:
    """Field score (0-100) per field, None when the candidate doesn't show it."""
    if method != "current":
        return candidate_field_scores(r["tokens"], p, positions=method == "positions")
    reading = r["reading"]
    v = p[values]
    printed = {
        "name": p["name"], "set_code": v["set_code"], "number": v["number"],
        "base_set_size": v["base_set_size"], "artist": p["artist"],
        "year": v["year"], "power": p["power"], "toughness": p["toughness"],
    }  # fmt: skip
    read = {
        "name": reading["name"], "set_code": reading["possible_set_codes"],
        "number": reading["collector_number"],
        "base_set_size": reading["base_set_size"], "artist": reading["artist"],
        "year": reading["copyright_year"], "power": reading["power"],
        "toughness": reading["toughness"],
    }  # fmt: skip
    # today's _field_score; with catalog values every field counts, as today
    scores = {}
    for f in FIELDS:
        if printed[f] is None:
            scores[f] = 0.0 if values == "catalog" else None
            continue
        queries = read[f] if isinstance(read[f], list) else [read[f]] if read[f] else []
        target = printed[f]
        if f in ("name", "artist"):
            target, queries = (
                utils.default_process(target),
                [utils.default_process(q) for q in queries],
            )
        elif f == "number":
            target, queries = (
                target.lstrip("0") or "0",
                [q.lstrip("0") or "0" for q in queries],
            )
        scores[f] = max((fuzz.ratio(q, target) for q in queries), default=0.0)
    return scores


def total(scores: dict, average: str) -> float:
    if average == "all":
        return sum(WEIGHTS[f] * (s or 0.0) for f, s in scores.items()) / sum(
            WEIGHTS.values()
        )
    shown = [f for f, s in scores.items() if s is not None]
    if not shown:
        return 0.0
    return sum(WEIGHTS[f] * scores[f] for f in shown) / sum(WEIGHTS[f] for f in shown)


def benchmark(cases, by_path, picked, printings, score) -> dict:
    """Exact printing, card count and same card name per query image, with pHash
    candidates; ties keep pHash order, as `rank` does. Only query images whose
    card images are all in `picked` (or fixed) count.
    """
    picked_paths = {r["path"] for r in picked}
    result = {"exact": 0, "count": 0, "name": 0, "images": 0, "exact_images": []}
    for image, entry in sorted(cases["images"].items()):
        paths = [c["path"] for c in entry["cards"] if "path" in c]
        # a query image with no sampled card image has no half: count it in "all"
        if (paths and not set(paths) <= picked_paths) or (
            not paths and len(picked) < len(by_path)
        ):
            continue
        recognised = []
        for card in entry["cards"]:
            if "path" not in card:
                recognised.append(card["fixed"])
                continue
            r = by_path[card["path"]]
            candidates = cases["phash"][card["path"]]
            recognised.append(max(candidates, key=lambda u: score(r, u)))
        label = entry["label"]
        exact = Counter(label) == Counter(recognised)
        result["images"] += 1
        result["exact"] += exact
        result["count"] += len(recognised) == len(label)
        result["name"] += Counter(printings[u]["name"] for u in label) == Counter(
            printings[u]["name"] if u else None for u in recognised
        )
        if exact:
            result["exact_images"].append(image)
    return result


def same_name(cases, picked, printings, score) -> dict:
    """Per group: how often the correct printing scores strictly above every other
    printing of its name (shuffled with a fixed seed), leaving out names with one
    printing; and how often the card shows exactly the same details as another
    printing of its name, so OCR alone can't separate them.
    """
    counts: dict[str, list[int]] = {}
    for r in picked:
        others = [u for u in cases["same_name"][r["path"]] if u != r["uuid"]]
        if not others or r["uuid"] not in printings:
            continue
        others = [u for u in others if u in printings]
        random.Random(SEED).shuffle(others)
        correct = score(r, r["uuid"])
        first = all(correct > score(r, u) for u in others)
        same = any(
            shown_details(printings[u]) == shown_details(printings[r["uuid"]])
            for u in others
        )
        group = "benchmark" if r["source"] == "benchmark" else r["category"]
        for key in (group, "all"):
            c = counts.setdefault(key, [0, 0, 0])
            c[0] += first
            c[1] += 1
            c[2] += same
    return counts


def shown_details(p: dict) -> tuple:
    return (p["artist"], p["power"], p["toughness"], *p["printed"].values())


def correct_fields(method, values, picked, printings) -> dict:
    """Mean field score of the correct printing per field, over the cards that
    show the field (printed values)."""
    sums: dict[str, list[float]] = {}
    for r in picked:
        p = printings.get(r["uuid"])
        if p is None:
            continue
        scores = field_scores(method, values, r, p)
        shown = field_scores("candidate", "printed", r, p)
        for f in FIELDS:
            if shown[f] is None:
                continue
            s = sums.setdefault(f, [0.0, 0])
            s[0] += scores[f] or 0.0
            s[1] += 1
    return {f: round(s / n, 1) for f, (s, n) in sums.items()}


def print_run(run: str, table: dict) -> None:
    print(f"\n## {run}\n")
    print(
        "| variant | benchmark exact / count / name | same name first | "
        + " | ".join(FIELDS)
        + " |"
    )
    print("|---" * (3 + len(FIELDS)) + "|")
    for label, t in table.items():
        b, s, f = t["benchmark"], t["same_name"]["all"], t["fields"]
        print(
            f"| {label} | {b['exact']} / {b['count']} / {b['name']} of {b['images']} "
            f"| {s[0]}/{s[1]} | "
            + " | ".join(str(f.get(x, "-")) for x in FIELDS)
            + " |"
        )
    groups = sorted({g for t in table.values() for g in t["same_name"]} - {"all"})
    identical = next(iter(table.values()))["same_name"]
    counts = ", ".join(f"{g} {identical[g][2]}" for g in groups)
    print(f"\nSame name first, per group (identical details: {counts})\n")
    print("| variant | " + " | ".join(groups) + " |")
    print("|---" * (1 + len(groups)) + "|")
    for label, t in table.items():
        print(
            f"| {label} | "
            + " | ".join(
                f"{t['same_name'][g][0]}/{t['same_name'][g][1]}" for g in groups
            )
            + " |"
        )


if __name__ == "__main__":
    main()
