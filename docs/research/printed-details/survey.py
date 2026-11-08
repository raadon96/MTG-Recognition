"""Throwaway: read the collector information of a few reference images per set and
frame, to see what each set prints, beyond the 540 samples of #193.

For each paper set and frame, up to 4 normal-layout printings with a reference
image are sampled (seed 201). The bottom 15% of the card, the area the collector
information sits in on upright layouts (docs/research/card-details.md), is upscaled
2x and read by the production model, as ocr_positions.py does for whole cards.

uv run python docs/research/printed-details/survey.py \
    <AllPrintings.sqlite> <images dir> <out.json>
"""

import json
import random
import sqlite3
import sys
from pathlib import Path

import cv2

from mtg_recognition.engine.ocr.reading import load_paddleocr

PER_SET_AND_FRAME = 4


def main(allprintings: str, images_dir: str, out_path: str) -> None:
    con = sqlite3.connect(f"file:{allprintings}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    images = Path(images_dir)
    rows = con.execute(
        """select c.uuid, c.setCode, c.frameVersion
           from cards c
           where c.availability like '%paper%' and c.layout = 'normal'
           order by c.uuid"""
    ).fetchall()
    by_set_and_frame: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        if (images / f"{r['uuid']}.jpg").exists():
            key = (r["setCode"], r["frameVersion"])
            by_set_and_frame.setdefault(key, []).append(r["uuid"])

    rng = random.Random(201)
    sample = [
        uuid
        for key in sorted(by_set_and_frame)
        for uuid in rng.sample(
            by_set_and_frame[key], min(PER_SET_AND_FRAME, len(by_set_and_frame[key]))
        )
    ]
    print(f"{len(by_set_and_frame)} sets and frames, {len(sample)} images")

    ocr = load_paddleocr()
    results = []
    for i, uuid in enumerate(sample):
        image = cv2.imread(str(images / f"{uuid}.jpg"))
        h = image.shape[0]
        crop = image[int(h * 0.85) :, :]
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        result = ocr.predict(crop, text_det_thresh=0.05)[0]
        results.append({"uuid": uuid, "texts": list(result["rec_texts"])})
        if i % 200 == 0:
            print(i, flush=True)
    Path(out_path).write_text(json.dumps(results, indent=0, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
