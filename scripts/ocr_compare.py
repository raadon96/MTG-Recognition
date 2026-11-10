"""Throwaway script for the ticket "Compare OCR models on raw text: PP-OCRv5
against PP-OCRv6 tiers" (#196) of the map #189. Never merged.

It reads the same card images with each OCR model and scores the raw text against
what each card prints (the printed-detail rules of #201), independently of
today's extraction code. It also ranks the correct printing against the other
printings of the same name with today's extraction and scoring.

    uv run python scripts/ocr_compare.py sample
    uv run python scripts/ocr_compare.py run NAME --det NAME=DIR --rec NAME=DIR
        [--threads 4] [--repeats 1] [--subset all|tune|confirm] [--hpi]
        [--no-fp16] [--det-thresh 0.05] [--det-box-thresh B] [--det-unclip U]
    uv run python scripts/ocr_compare.py report NAME [NAME ...] [--subset ...]

`sample` writes scripts/ocr_sample.json: the card images to read, with their
printing, category and half, and the card images of the uncommon-layout photos.
`run` writes output/ocr_compare/NAME/: readings.json and summary.json.
"""

import argparse
import json
import os
import random
import re
import resource
import sqlite3
import time
from pathlib import Path

SAMPLE_PATH = Path("scripts/ocr_sample.json")
OUTPUT_ROOT = Path("output/ocr_compare")
BENCHMARK_CARDS = Path("output/measure/g-11s-dyn-640-0.05")
UNCOMMON_PHOTOS = Path("data/scratch/uncommon_layout")
UNCOMMON_CARDS = Path("output/ocr_compare/uncommon_cards")
REFERENCE_IMAGES = Path("data/reference/images")
ALLPRINTINGS = "data/reference/mtgjson/AllPrintings.sqlite"
CATALOG = "data/reference/catalog.sqlite"
SEED = 189
PER_CATEGORY = 100
FRAMES = ("1993", "1997", "2003", "2015", "future")  # on normal-layout printings
# the layouts with at least 400 catalog rows
LAYOUTS = ("transform", "adventure", "modal_dfc", "saga", "split")
FIELDS = ("name", "set_code", "number", "base_set_size", "year", "artist")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sample")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("name")
    run_parser.add_argument("--det", required=True, help="NAME=DIR")
    run_parser.add_argument("--rec", required=True, help="NAME=DIR")
    run_parser.add_argument("--threads", type=int, default=4)
    run_parser.add_argument("--repeats", type=int, default=1)
    run_parser.add_argument(
        "--subset", choices=["all", "tune", "confirm"], default="all"
    )
    run_parser.add_argument("--hpi", action="store_true")
    # paddleocr 3.5+: run the ONNX export in DIR through onnxruntime
    run_parser.add_argument("--onnxruntime", action="store_true")
    run_parser.add_argument("--no-fp16", action="store_true")
    run_parser.add_argument("--det-thresh", type=float, default=0.05)
    run_parser.add_argument("--det-box-thresh", type=float)
    run_parser.add_argument("--det-unclip", type=float)
    # #198: detection input size, and the collector information crop's upscaling
    run_parser.add_argument("--det-limit-side-len", type=int)
    run_parser.add_argument("--det-limit-type", choices=["min", "max"])
    run_parser.add_argument("--target-height", type=int)
    run_parser.add_argument("--ratio-text-to-crop", type=float)
    run_parser.add_argument("--max-upscale", type=float)
    # both crops' smallest side, upscaled before OCR (32 today)
    run_parser.add_argument("--min-dim", type=int)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("names", nargs="+")
    report_parser.add_argument(
        "--subset", choices=["all", "tune", "confirm"], default="all"
    )
    args = parser.parse_args()

    if args.command == "sample":
        write_sample()
    elif args.command == "run":
        # thread pools read these when the libraries are imported
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(args.threads)
        os.environ["PADDLE_PDX_CPU_NUM_THREADS"] = str(args.threads)  # HPI's count
        run(args)
    else:
        report(args.names, args.subset)


def write_sample() -> None:
    """The card images to read: the benchmark's saved card images, 100 reference
    images per frame and per common layout, and the uncommon-layout photos, each
    with its printing, its category and its half (tune, confirm or both).
    """
    import cv2

    from mtg_recognition.engine.image import prepare_image
    from mtg_recognition.engine.segmentation import segment

    entries = []

    # benchmark: each saved card image, with the labelled printing the #195 run
    # matched it to; single-card images take their label directly
    manifest = json.loads(Path("data/benchmark/manifest.json").read_text())
    split = json.loads(Path("scripts/measure_split.json").read_text())
    tune, confirm = set(split["tune"]), set(split["confirm"])
    details = json.loads((BENCHMARK_CARDS / "images.json").read_text())
    for image, label in sorted(manifest.items()):
        half = (
            "both"
            if image in tune and image in confirm
            else ("tune" if image in tune else "confirm")
        )
        card_dir = BENCHMARK_CARDS / "cards" / Path(image).with_suffix("")
        for i, card in enumerate(details[image]["cards"]):
            uuid = label[0] if len(label) == 1 else card.get("correct")
            if uuid is None:
                continue  # a card of a multi-card image matched to no label
            entries.append({
                "source": "benchmark", "category": image.split("/")[0],
                "path": str(card_dir / f"{i}.png"), "uuid": uuid, "half": half,
            })  # fmt: skip

    # reference images: per category, one printing per set in a shuffled set
    # order, round after round, until 100
    db = sqlite3.connect(f"file:{ALLPRINTINGS}?mode=ro", uri=True)
    db.execute(f"attach 'file:{CATALOG}?mode=ro' as c")
    rows = db.execute(
        """select cards.uuid, cards.setCode, cards.layout, cards.frameVersion
           from c.catalog join cards on cards.uuid = c.catalog.uuid
           order by cards.uuid"""
    ).fetchall()
    have_image = {p.stem for p in REFERENCE_IMAGES.glob("*.jpg")}
    rng = random.Random(SEED)
    categories = [(f"frame_{f}", "normal", f) for f in FRAMES]
    categories += [(f"layout_{lay}", lay, None) for lay in LAYOUTS]
    for category, layout, frame in categories:
        by_set: dict[str, list[str]] = {}
        for uuid, set_code, row_layout, row_frame in rows:
            if (
                row_layout == layout
                and (frame is None or row_frame == frame)
                and uuid in have_image
            ):
                by_set.setdefault(set_code, []).append(uuid)
        sets = sorted(by_set)
        rng.shuffle(sets)
        for uuids in by_set.values():
            rng.shuffle(uuids)
        picked = []
        while len(picked) < PER_CATEGORY and any(by_set.values()):
            for s in sets:
                if by_set[s] and len(picked) < PER_CATEGORY:
                    picked.append(by_set[s].pop())
        for i, uuid in enumerate(picked):
            entries.append({
                "source": "reference", "category": category,
                "path": str(REFERENCE_IMAGES / f"{uuid}.jpg"), "uuid": uuid,
                "half": "tune" if i % 2 == 0 else "confirm",
            })  # fmt: skip

    # uncommon-layout photos: segmented once with today's segmentation
    UNCOMMON_CARDS.mkdir(parents=True, exist_ok=True)
    for i, photo in enumerate(sorted(UNCOMMON_PHOTOS.glob("*.jpg"))):
        cards = segment(prepare_image(photo))
        if len(cards) != 1:
            print(f"{photo.name}: {len(cards)} cards found, skipped")
            continue
        path = UNCOMMON_CARDS / f"{photo.stem}.png"
        cv2.imwrite(str(path), cards[0])
        entries.append({
            "source": "uncommon", "category": "uncommon_layout", "path": str(path),
            "uuid": photo.stem, "half": "tune" if i % 2 == 0 else "confirm",
        })  # fmt: skip

    SAMPLE_PATH.write_text(json.dumps(entries, indent=1) + "\n")
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["category"]] = counts.get(e["category"], 0) + 1
    print(json.dumps(counts, indent=1), len(entries))


def run(args: argparse.Namespace) -> None:
    # imported here: the thread variables must be set first
    import cv2
    import numpy as np
    from paddleocr import PaddleOCR

    from mtg_recognition.engine.ocr import reading as reading_module

    cv2.setNumThreads(args.threads)
    # read_card reads these module constants on every call
    upscale = {
        k: v
        for k, v in (
            ("TARGET_TEXT_HEIGHT", args.target_height),
            ("RATIO_TEXT_TO_CROP", args.ratio_text_to_crop),
            ("MAX_UPSCALE", args.max_upscale),
            ("_MIN_DIM", args.min_dim),
        )
        if v is not None
    }
    for k, v in upscale.items():
        setattr(reading_module, k, v)
    entries = _subset(json.loads(SAMPLE_PATH.read_text()), args.subset)
    out = OUTPUT_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)

    # load the model, as load_paddleocr with the model and settings as arguments
    det_name, det_dir = args.det.split("=")
    rec_name, rec_dir = args.rec.split("=")
    start = time.perf_counter()
    paddleocr = PaddleOCR(
        use_doc_unwarping=False,
        doc_unwarping_model_name=None,
        use_textline_orientation=False,
        text_detection_model_name=det_name,
        text_detection_model_dir=det_dir,
        text_recognition_model_name=rec_name,
        text_recognition_model_dir=rec_dir,
        device="cpu",
        lang="en",
        # with an engine, PaddleOCR ignores enable_hpi, precision and cpu_threads
        **(
            {
                "engine": "onnxruntime",
                "engine_config": {"intra_op_num_threads": args.threads},
            }
            if args.onnxruntime
            else {
                "enable_hpi": args.hpi,
                "cpu_threads": args.threads,
                **({} if args.no_fp16 else {"precision": "fp16"}),
            }
        ),
    )
    load_s = time.perf_counter() - start
    peak_after_load_mb = _peak_rss_mb()

    # time detection and recognition inside PaddleOCR's pipeline; instance
    # attributes shadow the pipeline's models, so its own code runs unchanged
    stage_s = {"detection": 0.0, "recognition": 0.0}
    inner = paddleocr.paddlex_pipeline._pipeline
    for stage, attr in (
        ("detection", "text_det_model"),
        ("recognition", "text_rec_model"),
    ):
        model = getattr(inner, attr)

        def timed(*a, _model=model, _stage=stage, **kw):
            start = time.perf_counter()
            results = list(_model(*a, **kw))
            stage_s[_stage] += time.perf_counter() - start
            return results

        setattr(inner, attr, timed)

    # read_card passes text_det_thresh itself; set it and the other detection
    # settings on every predict call
    predict = paddleocr.predict
    raw: dict[str, list] = {}
    det_kwargs = {
        k: v
        for k, v in (
            ("text_det_box_thresh", args.det_box_thresh),
            ("text_det_unclip_ratio", args.det_unclip),
            ("text_det_limit_side_len", args.det_limit_side_len),
            ("text_det_limit_type", args.det_limit_type),
        )
        if v is not None
    }

    def predict_with_settings(*a, **kw):
        kw["text_det_thresh"] = args.det_thresh
        raw["last"] = predict(*a, **kw, **det_kwargs)
        return raw["last"]

    paddleocr.predict = predict_with_settings

    # warm up: the first inference initialises lazily
    start = time.perf_counter()
    reading_module.read_card(paddleocr, cv2.imread(entries[0]["path"]))
    first_inference_s = time.perf_counter() - start

    # read every card image; tokens and readings from the first repeat
    readings = []
    times = []
    for idx, entry in enumerate(entries):
        card_image = cv2.imread(entry["path"])
        card_times = []
        for repeat in range(args.repeats):
            stage_s["detection"] = stage_s["recognition"] = 0.0
            start = time.perf_counter()
            reading = reading_module.read_card(paddleocr, card_image)
            total = time.perf_counter() - start
            card_times.append((total, stage_s["detection"], stage_s["recognition"]))
            if repeat == 0:
                readings.append({
                    **entry,
                    "tokens": {
                        crop: {
                            "texts": list(r["rec_texts"]),
                            "scores": [round(float(s), 4) for s in r["rec_scores"]],
                            "boxes": r["rec_boxes"].tolist(),
                        }
                        for crop, r in zip(
                            ("name", "collector_information"), raw["last"], strict=True
                        )
                    },
                    "reading": {
                        k: v for k, v in reading.__dict__.items() if k != "debug"
                    },
                })  # fmt: skip
        times.append(np.median(np.array(card_times), axis=0))
        if idx % 100 == 0:
            print(f"[{idx}/{len(entries)}] {times[-1][0]:.3f}s", flush=True)

    times = np.array(times)
    timing = {
        stage: {
            "median_s": round(float(np.median(times[:, i])), 4),
            "p95_s": round(float(np.percentile(times[:, i], 95)), 4),
            "sum_s": round(float(times[:, i].sum()), 2),
        }
        for i, stage in enumerate(("total", "detection", "recognition"))
    }
    for r, t in zip(readings, times, strict=True):
        r["time_s"] = [round(float(x), 4) for x in t]
    summary = {
        "configuration": {
            "detection": det_name, "recognition": rec_name, "hpi": args.hpi,
            "onnxruntime": args.onnxruntime,
            "fp16": not args.no_fp16, "threads": args.threads,
            "repeats": args.repeats, "subset": args.subset,
            "det_thresh": args.det_thresh, **det_kwargs, **upscale,
        },
        "cards": len(entries),
        "load_s": round(load_s, 2),
        "first_inference_s": round(first_inference_s, 2),
        "timing": timing,
        "peak_rss_mb": {"after_load": peak_after_load_mb, "end": _peak_rss_mb()},
        "model_mb": round(
            sum(_dir_mb(d) for d in (det_dir, rec_dir)), 1
        ),
    }  # fmt: skip
    (out / "readings.json").write_text(json.dumps(readings) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def report(names: list[str], subset: str) -> None:
    """For each run: raw-text hits per field and category against what each card
    prints, and how often today's extraction and scoring rank the correct printing
    above the other printings of its name.
    """
    from printed_match import found
    from printed_rules import printed_details, printed_sources
    from rapidfuzz import fuzz, utils

    from mtg_recognition.engine.ocr import scoring
    from mtg_recognition.engine.ocr.reading import Reading
    from mtg_recognition.reference.catalog import SqliteCardCatalog

    runs = {
        n: _subset(json.loads((OUTPUT_ROOT / n / "readings.json").read_text()), subset)
        for n in names
    }
    uuids = {r["uuid"] for rs in runs.values() for r in rs}

    # what each printing prints, by the rules of #201
    ap = sqlite3.connect(f"file:{ALLPRINTINGS}?mode=ro", uri=True)
    ap.row_factory = sqlite3.Row
    sets = {r["code"]: dict(r) for r in ap.execute("select * from sets")}
    cards = [
        dict(r)
        for r in ap.execute(
            """select uuid, setCode, number, frameVersion, originalReleaseDate,
                      layout, promoTypes, name, artist from cards"""
        )
    ]
    sources = printed_sources(cards, sets)
    by_uuid = {c["uuid"]: c for c in cards}
    printed = {}
    for uuid in uuids:
        card = by_uuid[uuid]
        details = printed_details(card, sets, sources)
        printed[uuid] = {
            "name": card["name"], "artist": card["artist"],
            "set_code": details["set_code"], "number": details["number"],
            "base_set_size": details["base_set_size"], "year": details["year"],
        }  # fmt: skip

    # the other printings of each name, for the same-name ranking
    catalog = SqliteCardCatalog(Path(CATALOG))
    db = sqlite3.connect(f"file:{CATALOG}?mode=ro", uri=True)
    names_of = dict(
        db.execute(
            "select uuid, name from catalog where uuid in "
            f"({','.join('?' * len(uuids))})",
            sorted(uuids),
        ).fetchall()
    )
    same_name: dict[str, list[str]] = {}
    for uuid, name in db.execute(
        f"""select uuid, name from catalog where name in
            ({",".join("?" * len(set(names_of.values())))}) order by uuid""",
        sorted(set(names_of.values())),
    ):
        same_name.setdefault(name, []).append(uuid)
    printings = catalog.lookup(sorted({u for us in same_name.values() for u in us}))

    table = {}
    for run_name, readings in runs.items():
        per_category: dict[str, dict[str, list[int]]] = {}
        for r in readings:
            p = printed[r["uuid"]]
            names_text = " ".join(r["tokens"]["name"]["texts"])
            info = r["tokens"]["collector_information"]["texts"]
            hits = {
                # either face of a multi-face card
                "name": any(
                    _alnum(face) and _alnum(face) in _alnum(names_text)
                    for face in p["name"].split(" // ")
                ),
                "artist": p["artist"] is not None
                and fuzz.partial_ratio(
                    utils.default_process(p["artist"]),
                    utils.default_process(" ".join(info)),
                )
                >= 90,
            }
            for f in ("set_code", "number", "base_set_size", "year"):
                if p[f] is not None:
                    hits[f] = found(f, p[f], p["number"], info)

            # same-name ranking with today's extraction and scoring, shuffled with
            # a fixed seed so pHash order can't break ties in its favour
            others = same_name[names_of[r["uuid"]]]
            if len(others) > 1:
                candidates = [printings[u] for u in others]
                random.Random(SEED).shuffle(candidates)
                ranked = scoring.rank(
                    Reading(**r["reading"]), candidates, k=len(candidates)
                )
                correct = next(x for x in ranked if x.printing.uuid == r["uuid"])
                best_other = max(
                    x.score for x in ranked if x.printing.uuid != r["uuid"]
                )
                hits["same_name_first"] = correct.score > best_other

            group = "benchmark" if r["source"] == "benchmark" else r["category"]
            for key in (group, "all"):
                counts = per_category.setdefault(key, {})
                for f, hit in hits.items():
                    c = counts.setdefault(f, [0, 0])
                    c[0] += int(hit)
                    c[1] += 1
        table[run_name] = per_category

    columns = (*FIELDS, "same_name_first")
    groups = sorted({g for t in table.values() for g in t}, key=_group_order)
    for group in groups:
        print(f"\n### {group}\n")
        print("| run | " + " | ".join(columns) + " |")
        print("|---" * (len(columns) + 1) + "|")
        for run_name, per_category in table.items():
            counts = per_category.get(group, {})
            cells = [
                f"{counts[c][0]}/{counts[c][1]}" if c in counts else "-"
                for c in columns
            ]
            print(f"| {run_name} | " + " | ".join(cells) + " |")
    (OUTPUT_ROOT / f"report-{subset}.json").write_text(json.dumps(table, indent=1))


def _subset(entries: list[dict], subset: str) -> list[dict]:
    if subset == "all":
        return entries
    return [e for e in entries if e["half"] in (subset, "both")]


def _group_order(group: str) -> tuple[int, str]:
    return ({"all": 0, "benchmark": 1}.get(group, 2), group)


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _dir_mb(directory: str) -> float:
    return sum(p.stat().st_size for p in Path(directory).glob("inference.*")) / 1e6


def _peak_rss_mb() -> int:
    # ru_maxrss is in kilobytes on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


if __name__ == "__main__":
    main()
