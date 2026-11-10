"""Throwaway script for the ticket "Test PP-OCR in onnxruntime without
paddlepaddle" (#199) of the map #189. Never merged.

Compares the tokens of two `ocr_compare.py run` outputs on the same card images:
how many card images give the same texts, the largest score and box difference,
and how many give the same reading.

    uv run python scripts/readings_diff.py RUN_A RUN_B
"""

import json
import sys
from pathlib import Path

OUTPUT_ROOT = Path("output/ocr_compare")


def main() -> None:
    run_a, run_b = sys.argv[1:3]
    a = json.loads((OUTPUT_ROOT / run_a / "readings.json").read_text())
    b = json.loads((OUTPUT_ROOT / run_b / "readings.json").read_text())
    by_path = {r["path"]: r for r in b}

    same_texts = same_reading = 0
    max_score_diff = max_box_diff = 0.0
    differing = []
    for ra in a:
        rb = by_path[ra["path"]]
        texts_equal = True
        for crop in ("name", "collector_information"):
            ta, tb = ra["tokens"][crop], rb["tokens"][crop]
            if ta["texts"] != tb["texts"]:
                texts_equal = False
                continue
            # scores and boxes only compare when the tokens line up
            for sa, sb in zip(ta["scores"], tb["scores"], strict=True):
                max_score_diff = max(max_score_diff, abs(sa - sb))
            for ba, bb in zip(ta["boxes"], tb["boxes"], strict=True):
                for xa, xb in zip(ba, bb, strict=True):
                    max_box_diff = max(max_box_diff, abs(xa - xb))
        same_texts += texts_equal
        same_reading += ra["reading"] == rb["reading"]
        if not texts_equal:
            differing.append(ra["path"])

    print(f"{run_a} against {run_b}, {len(a)} card images")
    print(f"same texts: {same_texts}, same reading: {same_reading}")
    print(f"largest score difference: {max_score_diff:.4f}")
    print(f"largest box difference: {max_box_diff:.0f} px")
    for path in differing[:10]:
        ra, rb = next(r for r in a if r["path"] == path), by_path[path]
        print(path)
        for crop in ("name", "collector_information"):
            if ra["tokens"][crop]["texts"] != rb["tokens"][crop]["texts"]:
                print(f"  {crop}: {ra['tokens'][crop]['texts']}")
                print(f"  {' ' * len(crop)}  {rb['tokens'][crop]['texts']}")


if __name__ == "__main__":
    main()
