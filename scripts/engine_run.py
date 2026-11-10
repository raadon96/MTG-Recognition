"""Throwaway script for the ticket "Test PP-OCR in onnxruntime without
paddlepaddle" (#199) of the map #189. Never merged.

Reads the card images of scripts/ocr_sample.json with RapidOCR or with our own
prototype (scripts/own_ocr.py) instead of PaddleOCR, through today's read_card,
and writes the same files as `ocr_compare.py run`, so `ocr_compare.py report` and
`readings_diff.py` read them. Runs in a venv without paddleocr or paddlex.

    python scripts/engine_run.py NAME --engine rapidocr|own --det PATH --rec PATH
        [--keys YML] [--paddle-settings] [--det-thresh 0.05]
        [--det-box-thresh 0.6] [--det-unclip 1.5] [--threads 4] [--repeats 1]
        [--limit N]

For rapidocr, --det and --rec are ONNX files; for own, model directories holding
inference.onnx and inference.yml. --keys takes the model's inference.yml and
writes its character list to a text file, for ONNX files without RapidOCR's
metadata. --paddle-settings sets what PaddleX does differently from RapidOCR:
ImageNet normalisation for detection, a 64 px minimum side, no resizing or
padding before detection, no dilation, no score filter.
"""

import argparse
import json
import os
import resource
import sys
import time
import types
from pathlib import Path

SAMPLE_PATH = Path("scripts/ocr_sample.json")
OUTPUT_ROOT = Path("output/ocr_compare")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--engine", choices=["rapidocr", "own"], required=True)
    parser.add_argument("--det", required=True)
    parser.add_argument("--rec", required=True)
    parser.add_argument("--keys")
    parser.add_argument("--paddle-settings", action="store_true")
    parser.add_argument("--det-thresh", type=float, default=0.05)
    parser.add_argument("--det-box-thresh", type=float, default=0.6)
    parser.add_argument("--det-unclip", type=float, default=1.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, help="read only the first N")
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(args.threads)

    import cv2
    import numpy as np
    import yaml

    # read_card's module imports PaddleOCR only for a type; this venv has none
    sys.modules["paddleocr"] = types.SimpleNamespace(PaddleOCR=None)
    from mtg_recognition.engine.ocr import reading as reading_module

    cv2.setNumThreads(args.threads)
    entries = json.loads(SAMPLE_PATH.read_text())[: args.limit]
    out = OUTPUT_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)

    # read_card calls predict on two crops and reads PaddleOCR's result keys
    stage_s = {"detection": 0.0, "recognition": 0.0}
    start = time.perf_counter()
    if args.engine == "own":
        import own_ocr

        own = own_ocr.load(Path(args.det), Path(args.rec), args.threads)

        def read_one(image):
            det_start = time.perf_counter()
            quads = own_ocr.detect(
                own.det, image, args.det_thresh, args.det_box_thresh, args.det_unclip
            )
            rec_start = time.perf_counter()
            lines = own_ocr.recognize(
                own, [own_ocr._crop_quad(image, q) for q in quads]
            )
            stage_s["detection"] += rec_start - det_start
            stage_s["recognition"] += time.perf_counter() - rec_start
            boxes = (
                np.concatenate([quads.min(axis=1), quads.max(axis=1)], axis=1)
                if len(quads)
                else np.zeros((0, 4), dtype=np.int16)
            )
            return [t for t, _ in lines], [s for _, s in lines], boxes

    else:
        from rapidocr import RapidOCR

        # the character list, from the model's inference.yml
        rec_params = {}
        if args.keys:
            characters = yaml.safe_load(Path(args.keys).read_text())["PostProcess"][
                "character_dict"
            ]
            keys_path = out / "keys.txt"
            keys_path.write_text("\n".join(characters) + "\n", encoding="utf-8")
            rec_params["Rec.rec_keys_path"] = str(keys_path)
        params = {
            "Global.use_cls": False,
            "Global.log_level": "warning",
            "Det.model_path": args.det,
            "Det.thresh": args.det_thresh,
            "Det.box_thresh": args.det_box_thresh,
            "Det.unclip_ratio": args.det_unclip,
            "Rec.model_path": args.rec,
            "EngineConfig.onnxruntime.intra_op_num_threads": args.threads,
            **rec_params,
        }
        if args.paddle_settings:
            params |= {
                "Global.text_score": 0.0,
                "Global.use_preprocess_img": False,
                "Global.use_vertical_padding": False,
                "Det.limit_side_len": 64,
                "Det.limit_type": "min",
                "Det.mean": [0.485, 0.456, 0.406],
                "Det.std": [0.229, 0.224, 0.225],
                "Det.use_dilation": False,
            }
        engine = RapidOCR(params=params)

        def read_one(image):
            result = engine(image)
            if result.boxes is None:
                return [], [], np.zeros((0, 4), dtype=np.int16)
            stage_s["detection"] += result.elapse_list[0]
            stage_s["recognition"] += result.elapse_list[2]
            # PaddleOCR's rec_boxes: the polygon's bounding rectangle
            polygons = np.asarray(result.boxes)
            boxes = np.concatenate([polygons.min(axis=1), polygons.max(axis=1)], axis=1)
            return list(result.txts), [float(s) for s in result.scores], boxes

    load_s = time.perf_counter() - start
    peak_after_load_mb = _peak_rss_mb()

    def predict(images, text_det_thresh):
        results = []
        for image in images:
            texts, scores, boxes = read_one(image)
            results.append({
                "rec_texts": texts, "rec_scores": scores,
                "rec_boxes": np.asarray(boxes).astype(np.int16),
            })  # fmt: skip
        raw["last"] = results
        return results

    raw: dict[str, list] = {}
    ocr = types.SimpleNamespace(predict=predict)

    reading_module.read_card(ocr, cv2.imread(entries[0]["path"]))

    readings = []
    times = []
    for idx, entry in enumerate(entries):
        card_image = cv2.imread(entry["path"])
        card_times = []
        for repeat in range(args.repeats):
            stage_s["detection"] = stage_s["recognition"] = 0.0
            start = time.perf_counter()
            reading = reading_module.read_card(ocr, card_image)
            total = time.perf_counter() - start
            card_times.append((total, stage_s["detection"], stage_s["recognition"]))
            if repeat == 0:
                readings.append({
                    **entry,
                    "tokens": {
                        crop: {
                            "texts": r["rec_texts"],
                            "scores": [round(s, 4) for s in r["rec_scores"]],
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
        if idx % 200 == 0:
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
            "engine": args.engine, "detection": args.det, "recognition": args.rec,
            "paddle_settings": args.paddle_settings, "threads": args.threads,
            "repeats": args.repeats, "det_thresh": args.det_thresh,
            "det_box_thresh": args.det_box_thresh, "det_unclip": args.det_unclip,
        },
        "cards": len(entries),
        "load_s": round(load_s, 2),
        "timing": timing,
        "peak_rss_mb": {"after_load": peak_after_load_mb, "end": _peak_rss_mb()},
    }  # fmt: skip
    (out / "readings.json").write_text(json.dumps(readings) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def _peak_rss_mb() -> int:
    # ru_maxrss is in kilobytes on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


if __name__ == "__main__":
    main()
