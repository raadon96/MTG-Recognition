"""Throwaway script for the ticket "Test PP-OCR in onnxruntime without
paddlepaddle" (#199) of the map #189. Never merged.

Do the official ONNX exports run in ONNX Runtime Web with the same output? For
each model, builds an input from a real card image, runs it through onnxruntime
here and through onnxruntime-web (WASM) in Node, and prints the largest output
difference and whether the decoded text or the text map agrees.

    ORT_WEB_DIR=/tmp/ortweb python scripts/ort_web_check.py CARD_IMAGE
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODELS = Path("models/ocr/test/onnx")
WORK = Path("output/ort_web_check")
DET_MODELS = ("PP-OCRv5_mobile_det", "PP-OCRv6_small_det", "PP-OCRv6_tiny_det")
REC_MODELS = ("PP-OCRv5_mobile_rec", "PP-OCRv6_small_rec", "PP-OCRv6_tiny_rec")


def main() -> None:
    card = cv2.imread(sys.argv[1])
    WORK.mkdir(parents=True, exist_ok=True)
    h, w = card.shape[:2]

    # detection input: the collector information crop, ImageNet-normalised, sides
    # rounded to multiples of 32 as PaddleX does
    crop = card[int(h * 0.88) :, int(w * 0.05) : int(w * 0.95)]
    crop = cv2.resize(
        crop, (round(crop.shape[1] * 2 / 32) * 32, round(crop.shape[0] * 2 / 32) * 32)
    )
    det_input = (crop.astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [
        0.229,
        0.224,
        0.225,
    ]
    det_input = det_input.transpose(2, 0, 1)[None].astype(np.float32)

    # recognition input: the name line, 48 px high, normalised to [-1, 1]
    name = card[int(h * 0.04) : int(h * 0.11), int(w * 0.06) : int(w * 0.9)]
    name = cv2.resize(name, (round(name.shape[1] * 48 / name.shape[0]), 48))
    rec_input = (name.astype(np.float32) / 255 - 0.5) / 0.5
    rec_input = rec_input.transpose(2, 0, 1)[None].astype(np.float32)

    cases = []
    native = {}
    for model, x in [(m, det_input) for m in DET_MODELS] + [
        (m, rec_input) for m in REC_MODELS
    ]:
        path = MODELS / model / "inference.onnx"
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        native[model] = session.run(None, {session.get_inputs()[0].name: x})[0]
        (WORK / f"{model}.in.bin").write_bytes(x.tobytes())
        cases.append(
            {"name": model, "model": str(path.resolve()), "shape": list(x.shape)}
        )
    (WORK / "cases.json").write_text(json.dumps(cases))

    subprocess.run(["node", "scripts/ort_web_check.mjs", str(WORK)], check=True)

    for c in cases:
        model = c["name"]
        expected = native[model]
        web = np.frombuffer((WORK / f"{model}.web.bin").read_bytes(), np.float32)
        web = web.reshape(expected.shape)
        diff = float(np.abs(web - expected).max())
        if model.endswith("_det"):
            # the text map binarised at today's threshold
            agree = float(((web > 0.05) == (expected > 0.05)).mean())
            print(f"{model}: max diff {diff:.2e}, text map pixels agreeing {agree:.6f}")
        else:
            same = bool((web.argmax(-1) == expected.argmax(-1)).all())
            print(
                f"{model}: max diff {diff:.2e}, same characters at every step: {same}"
            )


if __name__ == "__main__":
    main()
