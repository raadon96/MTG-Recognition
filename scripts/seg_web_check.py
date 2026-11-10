"""Throwaway script for the ticket "Test segmentation in ONNX Runtime Web, and
the Docker image without torch, clip and ultralytics" (#204) of the map #189.
Never merged.

Does the dynamic YOLOE-11s export give the same output in onnxruntime-web
(WASM) as in onnxruntime here? For each benchmark query image that reaches
YOLO, builds the input as segment() does (enhance_contrast, then
yoloe_onnx's letterbox), runs it here and in Node, and compares the raw
outputs, the outlines yoloe_onnx decodes from each, and the card images
segment() would warp from them.

    ORT_WEB_DIR=/tmp/ortweb uv run python scripts/seg_web_check.py \
        output/measure/onnx/yoloe-11s-seg-dynamic.onnx [--limit N]
"""

import argparse
import importlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yoloe_onnx

from mtg_recognition.engine.segmentation.corners import card_corners_from_outline
from mtg_recognition.engine.segmentation.warp import warp_card

BENCHMARK = Path("data/benchmark")
WORK = Path("output/seg_web_check")
# the package re-exports segment() under the module's name
segment_module = importlib.import_module("mtg_recognition.engine.segmentation.segment")
# the settings recommended by "Compare segmentation candidates" (#195)
SETTINGS = {"conf": 0.05, "iou": 0.0, "imgsz": 640, "static": False, "max_det": 50}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        args.model, options, providers=["CPUExecutionProvider"]
    )
    node = subprocess.Popen(
        ["node", "scripts/seg_web_check.mjs", str(Path(args.model).resolve())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    images = sorted(p for p in BENCHMARK.rglob("*.jpg"))[: args.limit]
    rows = []
    for image_path in images:
        query_image = cv2.imread(str(image_path))
        if segment_module.takes_scan_bypass(query_image):
            rows.append({"image": str(image_path), "bypass": True})
            continue
        enhanced = segment_module.enhance_contrast(query_image)

        # native run, keeping the input tensor and the outputs
        native = RecordedSession(session)
        native_outlines = yoloe_onnx.yoloe_outlines(native, enhanced, **SETTINGS)

        # the same input through onnxruntime-web
        (WORK / "in.bin").write_bytes(native.tensor.tobytes())
        job = {
            "input": str((WORK / "in.bin").resolve()),
            "shape": list(native.tensor.shape),
            "output0": str((WORK / "out0.bin").resolve()),
            "output1": str((WORK / "out1.bin").resolve()),
        }
        node.stdin.write(json.dumps(job) + "\n")
        node.stdin.flush()
        web_ms = json.loads(node.stdout.readline())["ms"]
        web_outputs = [
            np.fromfile(WORK / f"out{i}.bin", np.float32).reshape(o.shape)
            for i, o in enumerate(native.outputs)
        ]
        web_outlines = yoloe_onnx.yoloe_outlines(
            ReplayedSession(session, web_outputs), enhanced, **SETTINGS
        )

        # compare raw outputs, outlines and the card images warped from them
        output0_diff, protos_diff = (
            float(np.abs(w - n).max())
            for w, n in zip(web_outputs, native.outputs, strict=True)
        )
        same_outlines = len(web_outlines) == len(native_outlines) and all(
            w.shape == n.shape and np.array_equal(w, n)
            for w, n in zip(web_outlines, native_outlines, strict=False)
        )
        native_cards = card_images(query_image, native_outlines)
        web_cards = card_images(query_image, web_outlines)
        same_cards = len(web_cards) == len(native_cards) and all(
            w.shape == n.shape and np.array_equal(w, n)
            for w, n in zip(web_cards, native_cards, strict=False)
        )
        largest_outline_diff = max(
            (
                float(np.abs(w - n).max())
                for w, n in zip(web_outlines, native_outlines, strict=False)
                if w.shape == n.shape
            ),
            default=0.0,
        )
        rows.append(
            {
                "image": str(image_path),
                "bypass": False,
                "input_shape": list(native.tensor.shape),
                "output0_diff": output0_diff,
                "protos_diff": protos_diff,
                "native_outlines": len(native_outlines),
                "web_outlines": len(web_outlines),
                "same_outlines": same_outlines,
                "largest_outline_diff": largest_outline_diff,
                "native_cards": len(native_cards),
                "web_cards": len(web_cards),
                "same_cards": same_cards,
                "web_ms": web_ms,
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    node.stdin.close()
    node.wait()
    (WORK / "results.json").write_text(json.dumps(rows, indent=1))

    # summary
    ran = [r for r in rows if not r["bypass"]]
    print(f"\n{len(rows)} images, {len(ran)} reach YOLO, {len(rows) - len(ran)} bypass")
    print(f"largest output0 diff {max(r['output0_diff'] for r in ran):.2e}")
    print(f"largest protos diff {max(r['protos_diff'] for r in ran):.2e}")
    same_count = sum(r["native_outlines"] == r["web_outlines"] for r in ran)
    print(f"same outline count {same_count}")
    print(f"identical outlines {sum(r['same_outlines'] for r in ran)}")
    outline_diff = max(r["largest_outline_diff"] for r in ran)
    print(f"largest outline point diff {outline_diff:.3f} px")
    print(f"same card count {sum(r['native_cards'] == r['web_cards'] for r in ran)}")
    print(f"identical card images {sum(r['same_cards'] for r in ran)}")
    print(f"web median {np.median([r['web_ms'] for r in ran]):.0f} ms (1 thread)")


def card_images(query_image: np.ndarray, outlines: list) -> list[np.ndarray]:
    """As segment() after predict."""
    cards = []
    for outline in outlines:
        corners = card_corners_from_outline(query_image, outline)
        if corners is not None:
            cards.append(warp_card(query_image, corners))
    return cards


class RecordedSession:
    """Runs the real session and keeps its input and outputs."""

    def __init__(self, session: ort.InferenceSession) -> None:
        self.session = session

    def get_inputs(self) -> list:
        return self.session.get_inputs()

    def run(self, names, feed: dict) -> list:
        self.tensor = next(iter(feed.values()))
        self.outputs = self.session.run(names, feed)
        return self.outputs


class ReplayedSession:
    """Returns outputs computed elsewhere, so yoloe_outlines decodes them."""

    def __init__(self, session: ort.InferenceSession, outputs: list) -> None:
        self.session = session
        self.outputs = outputs

    def get_inputs(self) -> list:
        return self.session.get_inputs()

    def run(self, names, feed: dict) -> list:
        return self.outputs


if __name__ == "__main__":
    main()
