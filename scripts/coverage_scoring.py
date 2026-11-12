"""PROTOTYPE, never merged: scoring with coverage of the reading, and the weights
set again (#234). Throwaway branch `research/coverage`.

Builds on #197's harness (`research/extraction`) but uses dev's engine as it is:
`read_card` (PP-OCRv5_mobile det + PP-OCRv6 tiny rec, 960 / max, 64 px; #218,
#219) and `scoring.rank` (#217). OCR runs once, in `prepare`; `report` replays the
saved readings through every variant.

    uv run python scripts/coverage_scoring.py prepare
    uv run python scripts/coverage_scoring.py report

`prepare` writes output/coverage/readings.json:
- the 1,291 card images of #196's sample (scripts/ocr_sample.json), each with its
  reading, its 20 pHash candidates and every printing of its name;
- the benchmark, end to end: each query image segmented as the pipeline does,
  each card image with its pHash candidates and its reading.
"""

import argparse
import json
import random
import re
import sqlite3
import time
from collections import Counter
from itertools import pairwise, product
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mtg_recognition.benchmark.results import DATASET_ROOT, MANIFEST_NAME
from mtg_recognition.engine.ocr import scoring
from mtg_recognition.engine.ocr.pp_ocr import TextLines
from mtg_recognition.engine.ocr.reading import Reading
from mtg_recognition.reference.catalog import Printing, SqliteCardCatalog
from mtg_recognition.reference.layout import ReferenceData

SAMPLE_PATH = Path("scripts/ocr_sample.json")
OUT = Path("output/coverage")
SEED = 189  # #197's shuffle of same-name printings
FIELDS = (
    "name", "set_code", "collector_number", "base_set_size", "artist", "year",
    "power_toughness",
)  # fmt: skip
TODAY = tuple(scoring.WEIGHTS[f] for f in FIELDS)
GRID = (0.0, 0.5, 1.0, 2.0)
COVERAGE_GRID = (0.5, 1.0, 2.0)
# coverage forms: count each digit run once, or weight it by its length; with
# power/toughness digits explained for creatures, or not
FORMS = ("runs", "runs+pt", "length", "length+pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("report")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        report()


def prepare() -> None:
    from mtg_recognition.engine.image import prepare_image
    from mtg_recognition.engine.matching.phash import shortlist
    from mtg_recognition.engine.ocr.reading import read_card
    from mtg_recognition.engine.pipeline import Pipeline
    from mtg_recognition.engine.segmentation import segment

    pipeline = Pipeline()
    db = sqlite3.connect(
        f"file:{ReferenceData.default().catalog_path}?mode=ro", uri=True
    )
    by_name: dict[str, list[str]] = {}
    for uuid, name in sorted(db.execute("select uuid, name from catalog")):
        by_name.setdefault(name, []).append(uuid)
    name_of = {u: n for n, us in by_name.items() for u in us}

    def texts(reading) -> dict:
        return {
            "name": reading.name.texts,
            "collector_information": reading.collector_information.texts,
        }

    # the sample: reading, pHash candidates, every printing of the name
    started = time.perf_counter()
    sample = []
    for entry in json.loads(SAMPLE_PATH.read_text()):
        card_image = cv2.imread(entry["path"])
        uuids, _ = shortlist(pipeline.phashes, card_image, k=20)
        sample.append(
            entry
            | texts(read_card(pipeline.ocr, card_image))
            | {"phash": list(uuids), "same_name": by_name[name_of[entry["uuid"]]]}
        )
    print(f"sample: {len(sample)} card images, {time.perf_counter() - started:.0f}s")

    # the benchmark, end to end, as Pipeline.recognize does it
    started = time.perf_counter()
    manifest = json.loads((DATASET_ROOT / MANIFEST_NAME).read_text())
    benchmark = {}
    for image, label in sorted(manifest.items()):
        cards = []
        for card_image in segment(prepare_image(DATASET_ROOT / image)):
            uuids, _ = shortlist(pipeline.phashes, card_image, k=20)
            printings = pipeline.catalog.lookup(uuids)
            candidates = [u for u in uuids if u in printings]
            card = {"candidates": candidates}
            if candidates:
                card |= texts(read_card(pipeline.ocr, card_image))
            cards.append(card)
        benchmark[image] = {"label": label, "cards": cards}
    elapsed = time.perf_counter() - started
    print(f"benchmark: {len(benchmark)} query images, {elapsed:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "readings.json").write_text(
        json.dumps({"sample": sample, "benchmark": benchmark}) + "\n"
    )


def report() -> None:
    data = json.loads((OUT / "readings.json").read_text())
    sample, benchmark = data["sample"], data["benchmark"]
    needed = {u for e in sample for u in e["phash"] + e["same_name"]}
    needed |= {
        u for i in benchmark.values() for c in i["cards"] for u in c["candidates"]
    }
    needed |= {u for i in benchmark.values() for u in i["label"]}
    printings = SqliteCardCatalog(ReferenceData.default().catalog_path).lookup(
        sorted(needed)
    )

    # field scores and coverage of every competitor, once: they don't depend on
    # the weights
    started = time.perf_counter()
    cases = []
    for e in sample:
        same = [u for u in e["same_name"] if u != e["uuid"]]
        random.Random(SEED).shuffle(same)
        competitors = list(dict.fromkeys([e["uuid"], *e["phash"], *same]))
        case = score_matrix(e, [printings[u] for u in competitors])
        case["half"] = e["half"]
        case["group"] = "benchmark" if e["source"] == "benchmark" else e["category"]
        # rank's tie-break keeps candidate order: pHash order, then the others
        order = {
            u: i for i, u in enumerate(dict.fromkeys([*e["phash"], *same, e["uuid"]]))
        }
        case["order"] = np.array([order[u] for u in competitors])
        case["same_name"] = np.array([u in same for u in competitors])
        cases.append(case)
    bench_cards = []
    for entry in benchmark.values():
        for card in entry["cards"]:
            if card["candidates"]:
                c = score_matrix(card, [printings[u] for u in card["candidates"]])
                c["order"] = np.arange(len(card["candidates"]))
                c["uuids"] = card["candidates"]
                bench_cards.append(c)
                card["case"] = c
    print(f"field scores: {time.perf_counter() - started:.0f}s\n")

    tune = [c for c in cases if c["half"] in ("tune", "both")]
    confirm = [c for c in cases if c["half"] in ("confirm", "both")]
    with_runs = sum(not np.isnan(c["cov"][0, 0]) for c in cases)
    print(f"{with_runs} of {len(cases)} sample card images hold a run of digits\n")

    # variant 1: coverage with today's weights; pick the form and weight on tune
    print("## Variant 1: coverage with today's weights (tune half)\n")
    print("| form | " + " | ".join(f"×{c}" for c in COVERAGE_GRID) + " |")
    print("|---" * (1 + len(COVERAGE_GRID)) + "|")
    best1 = None
    for form_index, form in enumerate(FORMS):
        row = []
        for c in COVERAGE_GRID:
            w = np.array([[*TODAY, c]])
            first = measure(tune, w, form_index)["first"][0]
            row.append(f"{first} / {measure(tune, w, form_index)['same_name'][0]}")
            key = (first, -abs(c - 1), -form_index)
            if best1 is None or key > best1[0]:
                best1 = (key, form_index, c)
        print(f"| {form} | " + " | ".join(row) + " |")
    today = measure(tune, np.array([[*TODAY, 0]]), 0)
    print(f"\ntoday: {today['first'][0]} / {today['same_name'][0]} "
          "(first among candidates / same name first)\n")  # fmt: skip
    assert best1 is not None
    _, form1, c1 = best1

    # variant 2: weights alone; variant 3: weights and coverage together
    grid = np.array([[*w, 0.0] for w in product(GRID, repeat=len(FIELDS))])
    best2 = grid_search("Variant 2: weights, no coverage", tune, grid, 0)
    grid3 = np.array(
        [[*w, c] for w in product(GRID, repeat=len(FIELDS)) for c in COVERAGE_GRID]
    )
    best3 = grid_search(
        f"Variant 3: weights and coverage ({FORMS[form1]})", tune, grid3, form1
    )
    # what tuning on same name first alone would pick: the name's weight only
    # separates different cards, which that measure never sees
    grid_search(
        "Tuned on same name first only (for comparison)", tune, grid, 0, "same_name"
    )

    variants = {
        "today": ([*TODAY, 0.0], 0),
        f"1: coverage {FORMS[form1]} ×{c1}, today's weights": ([*TODAY, c1], form1),
        "1b: coverage runs ×1, today's weights (the issue's form)": ([*TODAY, 1.0], 0),
        "2: weights " + weights_text(best2): (list(best2), 0),
        "3: weights and coverage " + weights_text(best3): (list(best3), form1),
    }
    print("## Results\n")
    print(
        "| variant | benchmark exact / count / name | changed images | first among "
        "candidates (strict) tune / confirm | same name first all / tune / confirm |"
    )
    print("|---|---|---|---|---|")
    today_images = None
    results: dict[str, Any] = {}
    groups = sorted({c["group"] for c in cases})
    for label, (w, form) in variants.items():
        w = np.array([w])
        b = benchmark_result(benchmark, printings, w, form)
        if today_images is None:
            today_images = b["images"]
        changed = sorted(
            f"{i} ({'gained' if b['images'][i] else 'lost'})"
            for i in b["images"] if b["images"][i] != today_images[i]
        )  # fmt: skip
        t, k = measure(tune, w, form), measure(confirm, w, form)
        a = measure(cases, w, form)
        print(
            f"| {label} | {b['exact']} / {b['count']} / {b['name']} "
            f"of {len(benchmark)} | {len(changed)} "
            f"| {t['first'][0]} / {k['first'][0]} of {len(tune)} / {len(confirm)} "
            f"| {a['same_name'][0]} / {t['same_name'][0]} / {k['same_name'][0]} "
            f"of {a['same_name_n']} / {t['same_name_n']} / {k['same_name_n']} |"
        )
        per_group = {
            g: measure([c for c in cases if c["group"] == g], w, form) for g in groups
        }
        results[label] = {
            "weights": w[0].tolist(), "form": FORMS[form], "benchmark": b,
            "changed": changed,
            "per_group": {
                g: (int(m["same_name"][0]), m["same_name_n"])
                for g, m in per_group.items()
            },
            "same_name_cards": same_name_cards(cases, w, form),
        }  # fmt: skip
    print("\nSame name first per group\n")
    print("| variant | " + " | ".join(groups) + " |")
    print("|---" * (1 + len(groups)) + "|")
    for label, r in results.items():
        cells = [f"{r['per_group'][g][0]}/{r['per_group'][g][1]}" for g in groups]
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\nSame name first, cards gained / lost against today (tune, confirm)\n")
    base = results["today"]["same_name_cards"]
    for label, r in results.items():
        cells = []
        for half in ("tune", "confirm"):
            now = r["same_name_cards"]
            idx = [
                i
                for i, c in enumerate(cases)
                if c["half"] in (half, "both") and c["same_name"].any()
            ]
            cells.append(
                f"+{sum(1 for i in idx if now[i] and not base[i])} / "
                f"-{sum(1 for i in idx if base[i] and not now[i])}"
            )
        print(f"- {label}: " + ", ".join(cells))
    print("\nChanged benchmark images")
    for label, r in results.items():
        if r["changed"]:
            print(f"- {label}: " + ", ".join(r["changed"]))
    (OUT / "report.json").write_text(json.dumps(results, indent=1) + "\n")


def score_matrix(entry: dict, competitors: list[Printing]) -> dict:
    """Per competitor: the 7 field scores from `rank` (NaN when not shown), and
    coverage in each form (NaN when the text holds no run of digits)."""
    # scoring reads only the texts
    reading = Reading(
        name=TextLines(entry["name"], [], []),
        collector_information=TextLines(entry["collector_information"], [], []),
    )
    ranked = {
        r.printing.uuid: r
        for r in scoring.rank(reading, competitors, k=len(competitors))
    }
    runs = [run for run, _ in scoring._prepare(reading).runs]
    F = np.full((len(competitors), 8), np.nan)
    cov = np.full((len(competitors), len(FORMS)), np.nan)
    for i, p in enumerate(competitors):
        for j, s in enumerate(ranked[p.uuid].breakdown):
            assert s.field == FIELDS[j]
            if s.score is not None:
                F[i, j] = s.score
        if runs:
            cov[i] = [coverage(p, runs, form) for form in FORMS]
    return {"F": F, "cov": cov, "plain": np.array([p.is_plain for p in competitors])}


def coverage(p: Printing, runs: list[str], form: str) -> float:
    """The share (0-100) of the collector information's digit runs that equal one
    of the printing's shown values: number, base set size, year."""
    number, size, year = (
        p.printed_collector_number, p.printed_base_set_size, p.printed_year
    )  # fmt: skip
    # rank only looks for the size after a number
    values = {v for v in (number, year, size if number else None) if v}
    if "pt" in form and p.power is not None and p.toughness is not None:
        values |= {p.power, p.toughness}
    glued = (
        re.compile(rf"{re.escape(number)}1?0*{re.escape(size)}")
        if number and size
        else None
    )
    explained = total = 0
    for run, next_run in pairwise([*runs, ""]):
        weight = len(run) if form.startswith("length") else 1
        total += weight
        if (
            run in values
            or (glued and glued.fullmatch(run))
            # as rank's year: an open `1993-` is any shown year's range (#217)
            or (
                run == scoring.COPYRIGHT_START
                and year
                and not scoring._YEAR.fullmatch(next_run)
            )
        ):
            explained += weight
    return 100.0 * explained / total


def scores(case: dict, W: np.ndarray, form: int) -> np.ndarray:
    """Scores (competitors × weight vectors): the weighted average over the
    details each competitor shows, as rank, coverage as the 8th."""
    F = case["F"].copy()
    F[:, 7] = case["cov"][:, form]
    shown = ~np.isnan(F)
    num = np.nan_to_num(F) @ W.T
    den = shown @ W.T
    # with the name at weight 0 a printing may have nothing left to score
    return np.round(np.where(den > 0, num / np.where(den > 0, den, 1), 0.0), 9)


def picked(case: dict, S: np.ndarray) -> np.ndarray:
    """Index of the competitor rank puts first, per weight vector: the highest
    score, then the plain printing, then candidate order."""
    order = np.argsort(case["order"])
    S, plain = S[order], case["plain"][order]
    tied = S.max(axis=0) == S
    plain_tied = tied & plain[:, None]
    first = np.where(plain_tied.any(0), plain_tied.argmax(0), tied.argmax(0))
    return order[first]


def measure(cases: list[dict], W: np.ndarray, form: int) -> dict:
    """Per weight vector: how often the correct printing (index 0) scores strictly
    above every other competitor (pHash candidates and same-name printings), and
    above every other printing of its name (#197's same name first; names with
    one printing left out). Strict, because a reference image is its own first
    pHash candidate, so rank's tie-break would credit every tie to it."""
    first = np.zeros(len(W), dtype=int)
    same_name = np.zeros(len(W), dtype=int)
    n = 0
    for chunk in range(0, len(W), 4096):
        w = W[chunk : chunk + 4096]
        for case in cases:
            S = scores(case, w, form)
            first[chunk : chunk + 4096] += S[0] > S[1:].max(0)
            if case["same_name"].any():
                if chunk == 0:
                    n += 1
                same_name[chunk : chunk + 4096] += S[0] > S[case["same_name"]].max(0)
    return {"first": first, "same_name": same_name, "same_name_n": n}


def same_name_cards(cases: list[dict], W, form) -> list[bool]:
    """Same name first per card, for one weight vector."""
    result = []
    for case in cases:
        S = scores(case, W, form)
        result.append(
            bool(case["same_name"].any() and S[0, 0] > S[case["same_name"], 0].max())
        )
    return result


def grid_search(title: str, tune, grid, form, objective="first") -> np.ndarray:
    """The best weights on the tune half; among equals, the closest to today's."""
    started = time.perf_counter()
    m = measure(tune, grid, form)
    best = m[objective].max()
    ties = np.flatnonzero(m[objective] == best)
    distance = np.abs(grid[ties, :7] - np.array(TODAY)).sum(1)
    chosen = grid[ties[np.argsort(distance, kind="stable")]]
    print(f"## {title}\n")
    print(f"{len(grid)} weight vectors, {time.perf_counter() - started:.0f}s; "
          f"best {objective} on tune: {best} (today: "
          f"{measure(tune, np.array([[*TODAY, 0]]), 0)[objective][0]}), "
          f"{len(ties)} vectors reach it; closest to today's:\n")  # fmt: skip
    for w in chosen[:5]:
        print(f"- {weights_text(w)}")
    print()
    return chosen[0]


def weights_text(w) -> str:
    names = (*FIELDS, "coverage")
    return ", ".join(
        f"{n} {v:g}" for n, v in zip(names, w, strict=True) if n != "coverage" or v
    )


def benchmark_result(benchmark: dict, printings: dict, W, form) -> dict:
    """End to end: the printing rank puts first on each card image, scored
    against each query image's label as `mtg benchmark run` does."""
    result = {"exact": 0, "count": 0, "name": 0, "images": {}}
    for image, entry in benchmark.items():
        recognised = []
        for card in entry["cards"]:
            if not card["candidates"]:
                recognised.append(None)
                continue
            c = card["case"]
            recognised.append(c["uuids"][picked(c, scores(c, W, form))[0]])
        label = entry["label"]
        exact = Counter(label) == Counter(recognised)
        result["exact"] += exact
        exact = bool(exact)
        result["count"] += len(recognised) == len(label)
        result["name"] += Counter(printings[u].name for u in label) == Counter(
            printings[u].name if u else None for u in recognised
        )
        result["images"][image] = exact
    return result


if __name__ == "__main__":
    main()
