"""Throwaway measuring script for the map "Wayfinder: evaluate upgrading PaddleOCR
and YOLOE, and how readings are extracted" (#189), from the ticket "Build the
throwaway script that measures candidates" (#194). Never merged.

It runs the same steps as `Pipeline.recognize`, timing each one, and records
accuracy as `mtg benchmark run` does, plus raw-text accuracy, the margin of the
correct printing, the card images and the OCR tokens.

    uv run python scripts/measure.py split
    uv run python scripts/measure.py run NAME [--threads 4] [--repeats 3]
        [--subset all|tune|confirm] [--cards-from output/measure/OTHER]
        [--yolo models/.../model.pt] [--ocr-det NAME=DIR] [--ocr-rec NAME=DIR]
        [--prompts "a|b"] [--conf C] [--iou I] [--imgsz N] [--head one2one|one2many]
        [--retina] [--no-enhance] [--onnx-runner direct|ultralytics] [--segment-only]

`split` writes scripts/measure_split.json, the fixed tuning and confirming halves.
`run` writes output/measure/NAME/: summary.json, images.json, ocr.json and cards/.
`--cards-from` reads the card images a previous run saved instead of segmenting.
"""

import argparse
import importlib
import json
import os
import random
import re
import resource
import time
from pathlib import Path

DATASET_ROOT = Path("data/benchmark")
SPLIT_PATH = Path("scripts/measure_split.json")
OUTPUT_ROOT = Path("output/measure")
# in both halves: too few to split (4 multi-card images, 1 without a card)
IN_BOTH = ("2_photos_multiple_cards", "5_scans_multiple_cards", "7_photos_no_cards")
SPLIT_SEED = 189
STAGES = (
    "prepare",
    "segmentation",
    "yolo",  # inside segmentation: YOLO's predict alone
    "shortlist",
    "ocr_detection",
    "ocr_recognition",
    "ocr_other",  # cropping, resizing, PaddleOCR's own steps, extraction
    "scoring",
    "total",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("split")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("name")
    run_parser.add_argument("--threads", type=int, default=0, help="0: defaults")
    run_parser.add_argument("--repeats", type=int, default=3)
    run_parser.add_argument(
        "--subset", choices=["all", "tune", "confirm"], default="all"
    )
    run_parser.add_argument("--cards-from", type=Path)
    run_parser.add_argument("--yolo", type=Path)
    run_parser.add_argument("--ocr-det", help="NAME=DIR")
    run_parser.add_argument("--ocr-rec", help="NAME=DIR")
    # segmentation candidates (#195)
    run_parser.add_argument("--prompts", help="'|'-separated; .pt only")
    run_parser.add_argument("--conf", type=float)
    run_parser.add_argument("--iou", type=float)
    run_parser.add_argument("--imgsz", type=int)
    run_parser.add_argument("--head", choices=["one2one", "one2many"])
    run_parser.add_argument("--retina", action="store_true")
    run_parser.add_argument("--no-enhance", action="store_true")
    run_parser.add_argument(
        "--onnx-runner", choices=["direct", "ultralytics"], default="direct"
    )
    run_parser.add_argument(
        "--segment-only", action="store_true", help="no OCR: card count only"
    )
    # OCR detection settings (#196); read_card sets text_det_thresh to 0.05
    run_parser.add_argument("--det-thresh", type=float)
    run_parser.add_argument("--det-box-thresh", type=float)
    run_parser.add_argument("--det-unclip", type=float)
    args = parser.parse_args()

    if args.command == "split":
        write_split()
    else:
        # thread pools read these when the libraries are imported, so set them
        # before run() imports numpy, torch and paddle
        if args.threads:
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ[var] = str(args.threads)
        run(args)


def write_split() -> None:
    """A fixed random half of each benchmark category for tuning, the rest for
    confirming; the categories in IN_BOTH go in both.
    """
    manifest = json.loads((DATASET_ROOT / "manifest.json").read_text())
    by_category: dict[str, list[str]] = {}
    for image in sorted(manifest):
        by_category.setdefault(image.split("/")[0], []).append(image)

    rng = random.Random(SPLIT_SEED)
    tune, confirm = [], []
    for category, images in sorted(by_category.items()):
        if category in IN_BOTH:
            tune += images
            confirm += images
            continue
        shuffled = images[:]
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        tune += sorted(shuffled[:half])
        confirm += sorted(shuffled[half:])
    SPLIT_PATH.write_text(
        json.dumps({"seed": SPLIT_SEED, "tune": tune, "confirm": confirm}, indent=2)
        + "\n"
    )
    print(f"{SPLIT_PATH}: {len(tune)} to tune, {len(confirm)} to confirm")


def run(args: argparse.Namespace) -> None:
    # imported here: the thread variables must be set first
    import cv2
    import numpy as np
    import torch
    from paddleocr import PaddleOCR

    from mtg_recognition.benchmark.results import (
        BASELINE_PATH,
        ImageOutcome,
        compare,
        read_results,
        score_image,
        summarise,
    )
    from mtg_recognition.engine.image import prepare_image
    from mtg_recognition.engine.matching.phash import load_phashes, shortlist
    from mtg_recognition.engine.ocr import scoring
    from mtg_recognition.engine.ocr.reading import Reading, read_card
    from mtg_recognition.engine.segmentation import segment
    from mtg_recognition.reference.catalog import SqliteCardCatalog
    from mtg_recognition.reference.layout import ReferenceData
    from mtg_recognition.settings import settings

    if args.threads:
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)
    out = OUTPUT_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)

    # the query images and their labels
    manifest: dict[str, list[str]] = json.loads(
        (DATASET_ROOT / "manifest.json").read_text()
    )
    if args.subset == "all":
        images = sorted(manifest)
    else:
        images = json.loads(SPLIT_PATH.read_text())[args.subset]
    reference = ReferenceData.default()
    catalog = SqliteCardCatalog(reference.catalog_path)
    labelled = catalog.lookup(sorted({u for i in images for u in manifest[i]}))

    # load the models, timing each
    load_s = {}
    start = time.perf_counter()
    phashes = load_phashes(reference)
    load_s["phashes"] = time.perf_counter() - start

    start = time.perf_counter()
    # segment() calls the module's _yolo_model(); replace it to try another model
    segment_module = importlib.import_module(
        "mtg_recognition.engine.segmentation.segment"
    )
    yolo_path = args.yolo or settings.YOLO_SEG_MODEL_PATH
    prompts = args.prompts.split("|") if args.prompts else segment_module._YOLO_CLASSES
    if yolo_path.suffix == ".onnx" and args.onnx_runner == "direct":
        import yoloe_onnx

        yolo = yoloe_onnx.OnnxYoloe(str(yolo_path), args.threads)
    else:
        from ultralytics import YOLO

        yolo = YOLO(str(yolo_path), task="segment")
        if yolo_path.suffix == ".pt":
            yolo.set_classes(prompts)
    segment_module._yolo_model = lambda: yolo
    load_s["yolo"] = time.perf_counter() - start

    # predict settings: segment() reads the module's dict at each call
    seg_kwargs = segment_module._YOLO_PREDICT_KWARGS
    for key, value in (("conf", args.conf), ("iou", args.iou), ("imgsz", args.imgsz)):
        if value is not None:
            seg_kwargs[key] = value
    if args.head:
        seg_kwargs["end2end"] = args.head == "one2one"
    if args.retina:
        seg_kwargs["retina_masks"] = True
    if args.no_enhance:
        segment_module.enhance_contrast = lambda image: image

    # which way segment() found the cards: count the calls into each path
    paths_taken: list[str] = []
    for function, path_name in (
        ("takes_scan_bypass", "scan_bypass"),
        ("box_card_images", "box_fallback"),
        ("rembg_card_images", "rembg"),
    ):
        original = getattr(segment_module, function)

        def counted(*a, _original=original, _path=path_name, **kw):
            result = _original(*a, **kw)
            if result:
                paths_taken.append(_path)
            return result

        setattr(segment_module, function, counted)
    predict_yolo = yolo.predict

    yolo_s = [0.0]

    def counted_predict(*a, **kw):
        start = time.perf_counter()
        results = predict_yolo(*a, **kw)
        yolo_s[0] += time.perf_counter() - start
        masks = results[0].masks
        paths_taken.append(f"yolo:{len(masks.xy) if masks is not None else 0}")
        return results

    yolo.predict = counted_predict

    # as load_paddleocr, with the models and thread count as arguments
    det_name, det_dir = (
        (args.ocr_det or "").split("=")
        if args.ocr_det
        else (
            "PP-OCRv5_mobile_det",
            str(settings.OCR_DETECTION_MODEL_DIR),
        )
    )
    rec_name, rec_dir = (
        (args.ocr_rec or "").split("=")
        if args.ocr_rec
        else (
            "PP-OCRv5_mobile_rec",
            str(settings.OCR_RECOGNITION_MODEL_DIR),
        )
    )
    start = time.perf_counter()
    paddleocr = (
        None
        if args.segment_only
        else PaddleOCR(
            use_doc_unwarping=False,
            doc_unwarping_model_name=None,
            use_textline_orientation=False,
            text_detection_model_name=det_name,
            text_detection_model_dir=det_dir,
            text_recognition_model_name=rec_name,
            text_recognition_model_dir=rec_dir,
            device="cpu",
            enable_hpi=False,
            precision="fp16",
            lang="en",
            **({"cpu_threads": args.threads} if args.threads else {}),
        )
    )
    load_s["ocr"] = time.perf_counter() - start
    peak_after_load_mb = _peak_rss_mb()

    # time OCR detection and recognition inside PaddleOCR's pipeline, and keep
    # the raw result of each predict call; instance attributes shadow the
    # pipeline's models, so its own code runs unchanged
    ocr_s = {"ocr_detection": 0.0, "ocr_recognition": 0.0}
    raw: dict[str, list] = {}
    inner = paddleocr.paddlex_pipeline._pipeline if paddleocr else None
    for stage, attr in (
        ()
        if args.segment_only
        else (
            ("ocr_detection", "text_det_model"),
            ("ocr_recognition", "text_rec_model"),
        )
    ):
        model = getattr(inner, attr)

        def timed(*a, _model=model, _stage=stage, **kw):
            start = time.perf_counter()
            results = list(_model(*a, **kw))
            ocr_s[_stage] += time.perf_counter() - start
            return results

        setattr(inner, attr, timed)
    if paddleocr is not None:
        predict = paddleocr.predict

        det_settings = {
            k: v
            for k, v in (
                ("text_det_thresh", args.det_thresh),
                ("text_det_box_thresh", args.det_box_thresh),
                ("text_det_unclip_ratio", args.det_unclip),
            )
            if v is not None
        }

        def keep_raw(*a, **kw):
            raw["last"] = predict(*a, **{**kw, **det_settings})
            return raw["last"]

        paddleocr.predict = keep_raw

    def card_images_of(image: str, query_image) -> list:
        if args.cards_from is None:
            return segment(query_image)
        card_dir = args.cards_from / "cards" / Path(image).with_suffix("")
        paths = sorted(card_dir.glob("*.png"), key=lambda p: int(p.stem))
        return [cv2.imread(str(p)) for p in paths]

    # warm up on the first image: the first inference initialises lazily
    start = time.perf_counter()
    first_query_image = prepare_image(DATASET_ROOT / images[0])
    for card_image in card_images_of(images[0], first_query_image):
        if paddleocr is not None:
            read_card(paddleocr, card_image)
    first_inference_s = time.perf_counter() - start

    # recognise every query image, timing each stage; accuracy, card images and
    # tokens come from the first repeat
    times: dict[str, list[dict[str, float]]] = {image: [] for image in images}
    outcomes: dict[str, ImageOutcome] = {}
    details: dict[str, dict] = {}
    tokens: dict[str, list] = {}
    repeat_totals_s = []
    for repeat in range(args.repeats):
        repeat_start = time.perf_counter()
        for idx, image in enumerate(images, 1):
            stage_s = dict.fromkeys(STAGES, 0.0)
            ocr_s["ocr_detection"] = ocr_s["ocr_recognition"] = 0.0
            image_start = time.perf_counter()
            # (card image, pHash candidates, ranking, raw OCR result) per card
            read_cards = []
            best = []
            fallbacks = 0
            error = None
            paths_taken.clear()
            yolo_s[0] = 0.0
            try:
                start = time.perf_counter()
                query_image = prepare_image(DATASET_ROOT / image)
                stage_s["prepare"] = time.perf_counter() - start

                start = time.perf_counter()
                card_images = card_images_of(image, query_image)
                stage_s["segmentation"] = time.perf_counter() - start

                for card_image in [] if args.segment_only else card_images:
                    # shortlist pHash candidates, resolved through the catalog
                    start = time.perf_counter()
                    uuids, _ = shortlist(phashes, card_image, k=20)
                    printings = catalog.lookup(uuids)
                    candidates = [printings[u] for u in uuids if u in printings]
                    stage_s["shortlist"] += time.perf_counter() - start
                    if not candidates:
                        best.append(None)
                        read_cards.append((card_image, uuids, [], None))
                        continue

                    # read the card; an OCR failure falls back to pHash order
                    start = time.perf_counter()
                    raw.clear()
                    try:
                        reading = read_card(paddleocr, card_image)
                    except RuntimeError:
                        reading = Reading()
                        fallbacks += 1
                    stage_s["ocr_other"] += time.perf_counter() - start

                    # rank every candidate, to see the runner-up
                    start = time.perf_counter()
                    ranked = scoring.rank(reading, candidates, k=len(candidates))
                    stage_s["scoring"] += time.perf_counter() - start
                    best.append(ranked[0].printing)
                    read_cards.append((card_image, uuids, ranked, raw.get("last")))
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                best = []
            stage_s["total"] = time.perf_counter() - image_start
            stage_s["yolo"] = yolo_s[0]
            stage_s["ocr_detection"] = ocr_s["ocr_detection"]
            stage_s["ocr_recognition"] = ocr_s["ocr_recognition"]
            stage_s["ocr_other"] -= ocr_s["ocr_detection"] + ocr_s["ocr_recognition"]
            times[image].append(stage_s)

            label = [labelled[u] for u in manifest[image]]
            if args.segment_only:
                # card count only: no printing is recognised
                best = [None] * len(card_images) if error is None else []
            outcome = ImageOutcome(
                score=score_image(label, best),
                recognised=[p.uuid if p is not None else None for p in best],
                runtime_s=round(stage_s["total"], 3),
                ocr_fallbacks=fallbacks,
                error=error,
            )
            if repeat == 0:
                # outside the timed steps: saving the card images takes time
                cards, card_tokens = [], []
                for card_idx, (card_image, uuids, ranked, raw_result) in enumerate(
                    read_cards
                ):
                    card_tokens.append(_tokens(raw_result))
                    cards.append(
                        _card_details(ranked, uuids, set(manifest[image]))
                        if ranked
                        else {"candidates": 0}
                    )
                    if args.cards_from is None:
                        card_dir = out / "cards" / Path(image).with_suffix("")
                        card_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(card_dir / f"{card_idx}.png"), card_image)
                if args.segment_only and error is None:
                    card_dir = out / "cards" / Path(image).with_suffix("")
                    card_dir.mkdir(parents=True, exist_ok=True)
                    for card_idx, card_image in enumerate(card_images):
                        cv2.imwrite(str(card_dir / f"{card_idx}.png"), card_image)
                outcomes[image] = outcome
                tokens[image] = card_tokens
                details[image] = {
                    "segmentation": list(paths_taken),
                    "cards_found": len(card_images) if error is None else None,
                    "cards_labelled": len(manifest[image]),
                    "cards": cards,
                    "raw_text": _raw_text_hits(label, card_tokens),
                }
            elif outcome.recognised != outcomes[image].recognised:
                details[image].setdefault("unstable_repeats", []).append(repeat)
            print(
                f"[{repeat + 1}/{args.repeats}] [{idx}/{len(images)}] "
                f"{stage_s['total']:.2f}s {image}",
                flush=True,
            )
        repeat_totals_s.append(time.perf_counter() - repeat_start)

    # accuracy, as `mtg benchmark run` measures it, against the baseline
    results = summarise("measure", None, None, outcomes)
    baseline = read_results(BASELINE_PATH)
    baseline.images = {i: o for i, o in baseline.images.items() if i in outcomes}
    comparison = compare(results, baseline)

    # time per query image: the median of the repeats, then the median and the
    # 95th percentile over the query images
    per_image = {
        stage: np.array([np.median([t[stage] for t in times[i]]) for i in images])
        for stage in STAGES
    }
    timing = {
        stage: {
            "median_s": round(float(np.median(v)), 4),
            "p95_s": round(float(np.percentile(v, 95)), 4),
            "sum_s": round(float(v.sum()), 2),
        }
        for stage, v in per_image.items()
    }

    # raw-text accuracy and margins
    raw_text = {
        field: sum(d["raw_text"][field] for d in details.values())
        for field in ("labelled", "name", "set_code", "collector_number")
    }
    all_cards = [c for d in details.values() for c in d["cards"] if "margin" in c]
    margins = [c["margin"] for c in all_cards if c["margin"] is not None]
    top_margins = [m for m in margins if m > 0]
    margin_summary = {
        "cards": len(all_cards),
        "correct_among_candidates": len(margins),
        "correct_first": len(top_margins),
        "median_margin_when_first": round(float(np.median(top_margins)), 2)
        if top_margins
        else None,
        "first_by_under_5": sum(m < 5 for m in top_margins),
        "first_by_under_10": sum(m < 10 for m in top_margins),
    }

    summary = {
        "configuration": {
            "yolo": str(yolo_path),
            "prompts": prompts if yolo_path.suffix == ".pt" else "exported",
            "onnx_runner": args.onnx_runner if yolo_path.suffix == ".onnx" else None,
            "predict": {k: v for k, v in seg_kwargs.items() if k != "verbose"},
            "enhance_contrast": not args.no_enhance,
            "segment_only": args.segment_only,
            "ocr_detection": det_name,
            "ocr_recognition": rec_name,
            "ocr_detection_settings": {
                "thresh": args.det_thresh,
                "box_thresh": args.det_box_thresh,
                "unclip_ratio": args.det_unclip,
            },
            "threads": args.threads or "default",
            "repeats": args.repeats,
            "subset": args.subset,
            "cards_from": str(args.cards_from) if args.cards_from else None,
        },
        "totals": {
            "images": results.totals.images,
            "exact_printing": results.totals.exact_printing,
            "card_count": results.totals.card_count,
            "same_card_name": results.totals.same_card_name,
            "ocr_fallbacks": results.totals.ocr_fallbacks,
        },
        "flips_from_baseline": [
            {"image": f.image, "metric": f.metric, "passes": f.passes}
            for f in comparison.flips
        ],
        "unstable_images": [i for i, d in details.items() if "unstable_repeats" in d],
        "segmentation_paths": _path_counts(details),
        "card_count_misses": [
            {"image": i, "found": d["cards_found"], "labelled": d["cards_labelled"]}
            for i, d in details.items()
            if d["cards_found"] != d["cards_labelled"]
        ],
        "multi_card_images": {
            i: {
                "found": d["cards_found"],
                "labelled": d["cards_labelled"],
                "exact_printing": outcomes[i].score.exact_printing,
            }
            for i, d in details.items()
            if d["cards_labelled"] > 1
        },
        "raw_text": raw_text,
        "margin": margin_summary,
        "load_s": {k: round(v, 2) for k, v in load_s.items()},
        "first_inference_s": round(first_inference_s, 2),
        "repeat_totals_s": [round(t, 1) for t in repeat_totals_s],
        "timing": timing,
        "peak_rss_mb": {"after_load": peak_after_load_mb, "end": _peak_rss_mb()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "images.json").write_text(json.dumps(details, indent=2) + "\n")
    (out / "ocr.json").write_text(json.dumps(tokens) + "\n")
    print(json.dumps(summary, indent=2))


def _path_counts(details: dict) -> dict:
    """How many query images each segmentation path found the cards in."""
    counts: dict[str, int] = {}
    for d in details.values():
        steps = d["segmentation"]
        found = [s for s in steps if not s.startswith("yolo")] or [
            "yolo" if steps and steps[-1] != "yolo:0" else "none"
        ]
        counts[found[-1]] = counts.get(found[-1], 0) + 1
    return counts


def _tokens(raw_results) -> dict:
    """The OCR tokens, confidences and boxes of the name crop and the collector
    information crop, before read_card filters out flavor text.
    """
    if raw_results is None:
        return {}
    return {
        crop: {
            "texts": list(result["rec_texts"]),
            "scores": [round(float(s), 4) for s in result["rec_scores"]],
            "boxes": result["rec_boxes"].tolist(),
        }
        for crop, result in zip(
            ("name", "collector_information"), raw_results, strict=True
        )
    }


def _card_details(ranked, uuids: list[str], label: set[str]) -> dict:
    """Where the correct printing ranks, and its margin: its score minus the best
    score of a printing not in the label; negative when another one ranks first,
    None when the correct printing isn't among the candidates.
    """
    correct = next((r for r in ranked if r.printing.uuid in label), None)
    other = next((r.score for r in ranked if r.printing.uuid not in label), 0.0)
    return {
        "candidates": len(ranked),
        "first": ranked[0].printing.uuid,
        "correct": correct.printing.uuid if correct else None,
        "phash_rank": uuids.index(correct.printing.uuid) if correct else None,
        "score_rank": ranked.index(correct) if correct else None,
        "margin": round(correct.score - other, 2) if correct else None,
        "fields": {f.field: f.read for f in ranked[0].breakdown},
    }


def _raw_text_hits(label, card_tokens: list[dict]) -> dict:
    """Of the image's labelled printings, how many have their name, set code and
    collector number in the raw OCR text of any card in the image. Not every
    printing prints its set code or collector number, so 100% isn't reachable.
    """
    names = " ".join(t for c in card_tokens for t in c.get("name", {}).get("texts", []))
    info = " ".join(
        t
        for c in card_tokens
        for t in c.get("collector_information", {}).get("texts", [])
    )
    names_alnum = _alnum(names)
    hits = {"labelled": len(label), "name": 0, "set_code": 0, "collector_number": 0}
    for printing in label:
        # either face of a double-faced card
        if any(_alnum(face) in names_alnum for face in printing.name.split(" // ")):
            hits["name"] += 1
        set_code = re.escape(printing.set_code)
        if re.search(rf"(?<![a-z0-9]){set_code}(?![a-z0-9])", info, re.I):
            hits["set_code"] += 1
        number = re.sub(r"^0+(?=\d)", "", printing.collector_number)
        if re.search(rf"(?<![0-9])0*{re.escape(number)}(?![0-9a-z])", info, re.I):
            hits["collector_number"] += 1
    return hits


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _peak_rss_mb() -> int:
    # ru_maxrss is in kilobytes on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


if __name__ == "__main__":
    main()
