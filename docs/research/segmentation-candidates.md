# Segmentation candidates: YOLOE-11 and YOLOE-26, as .pt and ONNX

Measurements for the ticket "Compare segmentation candidates: YOLOE-11 and YOLOE-26, as .pt and ONNX" (#195), on the map #189. Run 2026-11-09 at `2616dd1` (`research/measure`, same engine code as `aa54a1a`), ultralytics 8.4.21, torch 2.10.0 CPU, onnxruntime 1.24.3, AMD Ryzen AI 7 350 (16 threads), 4 threads for every run. Nothing outside this branch changed: production models, `pyproject.toml`, the catalog and the baseline are as on `dev`.

Scripts on this branch: `scripts/measure.py` (extended with segmentation options) and `scripts/yoloe_onnx.py` (runs a YOLOE ONNX export in onnxruntime without ultralytics). Outputs are under `output/measure/` (gitignored), named as in the tables.

## Answer

- **Keep YOLOE-11s, and lower `conf` from 0.1 to 0.05.** 254 / 257 / 257 (exact printing / card count / same card name) against today's 253 / 256 / 256, with no image worse. It fixes the 9-card scan with border gaps, the image that today misses one card. Both halves agree: tune 129/130/130, confirm 129/132/132. The card count is clean for every `conf` from 0.02 to 0.08 on the tune half, so 0.05 sits in a stable range.
- **Run it as a dynamic ONNX export, through onnxruntime directly.** Its results match `.pt` exactly (identical card images, identical benchmark results), it isn't faster at inference on this CPU, but it loads in 0.08 s instead of 1.16 s and uses about 420 MB less memory after loading, because neither torch weights nor the 600 MB text encoder are loaded. The browser port rule allows it: it doesn't change the model. A static (square) export changes results and is ruled out.
- **YOLOE-26n is not adopted automatically.** At its best settings (one-to-many head with NMS, `conf=0.05`) it is 1.09x faster in total and 59% smaller than 11s `.pt`, but it loses the card name on 3 images (252 / 257 / 254). Losing card names goes to the maintainer under the adoption rule; the affected images are listed below. Compared with 11s at `conf=0.05`, it is 2 exact printings and 3 card names worse.
- **YOLOE-26s is rejected**: worse than 11s on every measure, no faster, and larger.
- **YOLOE-26's one-to-one head (no NMS) is unusable on our images** at every `conf` and `imgsz`: it returns the art box as a separate object, so low `conf` overcounts and high `conf` misses cards.
- **No candidate reaches a 1.2x total speedup.** Segmentation is about 30% of the time per query image, and YOLO itself only about 20%.

## Results

All 257 query images, 3 repeats, 4 threads, full pipeline (k=20, PP-OCRv5 mobile). Changed images: how many images pass more or fewer of the three measures than in `data/benchmark-baseline.json`. Times: median and 95th percentile per query image, over the medians of the 3 repeats. "YOLO" is `predict` alone; "segmentation" adds `enhance_contrast`, corner fitting and warping. Peak RSS is measured after loading the models; the peak at the end of the run is dominated by OCR (1.8–2.7 GB, varies between runs).

| Candidate | Exact / count / name | Changed images | Per image: median / p95 | Segmentation median | YOLO median | Run, 3 repeats | Speedup (summed medians) | Load | Peak RSS after load | Size |
|---|---|---|---|---|---|---|---|---|---|---|
| 11s `.pt`, today's settings (`conf` 0.1, 640) | 253 / 256 / 256 | (baseline) | 0.429 / 0.526 s | 0.132 s | ≈0.08 s | 109.0, 111.7, 110.2 s | 1.00x | 1.18 s | 1210 MB | 27.8 MB + 600 MB text encoder |
| 11s ONNX, today's settings | 253 / 256 / 256 | none | 0.428 / 0.513 s | 0.130 s | 0.087 s | 106.5, 109.9, 109.4 s | 1.00x | 0.09 s | 793 MB | 40.8 MB |
| 11s `.pt`, `conf` 0.05 | 254 / 257 / 257 | 1 better | 0.429 / 0.535 s | 0.130 s | 0.082 s | 117.8, 109.2, 108.4 s | 0.99x | 1.16 s | 1208 MB | 27.8 MB + 600 MB |
| **11s ONNX, `conf` 0.05** | **254 / 257 / 257** | **1 better** | 0.433 / 0.497 s | 0.130 s | 0.087 s | 105.9, 108.5, 115.4 s | 1.00x | 0.08 s | 793 MB | 40.8 MB |
| 11s ONNX, `conf` 0.05, `imgsz` 480 | 253 / 257 / 256 | 2 better, 1 worse | 0.401 / 0.483 s | 0.095 s | 0.049 s | 95.2, 103.1, 102.7 s | 1.07x | 0.08 s | 793 MB | 40.8 MB |
| 26n `.pt`, one-to-many, `conf` 0.05 | 252 / 257 / 254 | 1 better, 3 worse | 0.382 / 0.475 s | 0.085 s | 0.039 s | 102.9, 97.9, 99.2 s | 1.10x | 1.02 s | 920 MB | 11.5 MB + 254 MB text encoder |
| 26n ONNX, one-to-many, `conf` 0.05 | 252 / 257 / 254 | 1 better, 3 worse | 0.391 / 0.480 s | 0.080 s | 0.034 s | 103.5, 100.0, 99.5 s | 1.09x | 0.06 s | 766 MB | 11.5 MB |

One repeat each (accuracy is the same across repeats: no run had an unstable image):

| Candidate | Exact / count / name | Changed images | Per image median | Paths: YOLO / box fallback / rembg |
|---|---|---|---|---|
| 26s `.pt`, today's settings (one-to-one head on 8.4.21) | 244 / 252 / 246 | 1 better, 11 worse | 0.443 s | 156 / 83 / 8 |
| 26n `.pt`, today's settings | 233 / 238 / 235 | 21 worse | 0.385 s | 209 / 30 / 8 |
| 26s ONNX, one-to-many, `conf` 0.03 (its best) | 245 / 257 / 247 | 2 better, 10 worse | 0.441 s | 229 / 17 / 1 |
| 26n `.pt`, one-to-many, `conf` 0.05, `retina_masks` | 253 / 257 / 256 | 1 better, 2 worse | 0.377 s | 244 / 2 / 1 |
| 11s `.pt`, `conf` 0.05, `imgsz` 480, `retina_masks` | 254 / 257 / 257 | 1 better | 0.391 s | 246 / 0 / 1 |
| 11s ONNX, `conf` 0.05, no `enhance_contrast` | 251 / 257 / 256 | 1 better, 2 worse | 0.378 s | 247 / 0 / 0 |
| 26n ONNX, one-to-many, `conf` 0.05, no `enhance_contrast` | 253 / 257 / 255 | 1 better, 2 worse | 0.341 s | 244 / 3 / 0 |

Every run also takes the scan bypass on 9 images and finds no card in the no-card image.

### Images that change

Against the baseline, at the settings in the first table:

- 11s, `conf` 0.05 (`.pt` and ONNX): `5_scans_multiple_cards/with_inner_gaps_with_border_gaps.jpg` now finds all 9 cards and passes all three measures.
- 11s, `imgsz` 480: the same scan passes card count and card name (not exact printing); `1_photos_single_card/b845b1d7…` gains exact printing; `1_photos_single_card/118a45de…` loses exact printing and card name.
- 26n, one-to-many, `conf` 0.05: the 9-card scan with border gaps passes all three; it loses exact printing and card name on `1_photos_single_card/118a45de…` and `1_photos_single_card/27da5926…`, and card name on `5_scans_multiple_cards/with_inner_gaps_without_border_gaps.jpg`. Three card names lost, two of them single-card photos: for the maintainer.

### The 4 multi-card images

Cards found / 9 or 4 or 2, then E, C, N for exact printing, card count and same card name (upper case: passes).

| Candidate | four cards with gap | two cards with gap | 9-card scan, border gaps | 9-card scan, no border gaps |
|---|---|---|---|---|
| 11s, today's settings (`.pt` or ONNX) | 4 ECN | 2 ECN | 8 e c n | 9 e C N |
| 11s, `conf` 0.05 | 4 ECN | 2 ECN | 9 ECN | 9 e C N |
| 11s, `conf` 0.05, `imgsz` 480 | 4 ECN | 2 ECN | 9 e C N | 9 e C N |
| 26n one-to-many, `conf` 0.05 | 4 ECN | 2 ECN | 9 ECN | 9 e C n |
| 26s one-to-many, `conf` 0.03 | 4 ECN | 2 ECN | 9 e C N | 9 e C n |
| 26s, today's settings | 4 ECN | 2 ECN | 6 e c n | 6 e c n |
| 26n, today's settings | 4 ECN | 2 ECN | 2 e c n | 4 e c n |

The 9-card scan without border gaps fails exact printing in every run, as in the baseline (a tie lost on pHash order and a card second by 5.9).

## Sweeps

Done segmentation-only (no OCR) on the tune half (130 images), 1 repeat, judging card count, how many images YOLO found the cards in without a fallback, and the 4 multi-card images; then the best settings went through the full pipeline on all images, and the tune and confirm halves were compared. Outputs: `output/measure/{a,b,c,d}-*`.

**Prompts** (`conf` 0.1, 640). Tried today's `book cover|art work`; `book cover`, `card`, `game card`, `trading card`, `playing card`, `magic card` alone; `trading card|art work`; `book cover|art work|trading card`. Today's pair is the best or tied for best on every model and head; adding `trading card` changes nothing. Single prompts mostly fail: with `card`, `game card` or `playing card`, YOLO finds the cards in almost none of the photos, and the box fallback and rembg do the work. So one prompt isn't enough, and the prompts tuned on YOLOE-11 also suit MobileCLIP2.

**Head** (YOLOE-26). The one-to-many head with NMS (`iou` 0.0) beats the one-to-one head everywhere. The one-to-one head has no workable `conf`, with `imgsz` 640, tune half, card count misses / images found by YOLO:

| `conf` | 0.01 | 0.03 | 0.05 | 0.1 | 0.15 | 0.25 |
|---|---|---|---|---|---|---|
| 26s one-to-one | 42 / 123 | 22 / 114 | 12 / 107 | 3 / 74 | 3 / 49 | 3 / 22 |
| 26n one-to-one | 44 / 125 | 24 / 123 | 20 / 120 | 10 / 107 | 8 / 94 | 4 / 61 |
| 26s one-to-many | 1 / 121 | 0 / 116 | 1 / 114 | 1 / 107 | 3 / 89 | 3 / 48 |
| 26n one-to-many | 1 / 125 | 1 / 125 | 0 / 125 | 2 / 122 | 2 / 112 | 3 / 83 |
| 11s | 0 / 125 | 0 / 125 | 0 / 125 | 1 / 125 | 1 / 122 | 2 / 109 |

At low `conf` the one-to-one head finds up to 17 objects on a 9-card scan: it returns the art box as a second object, which `iou=0.0` removes on the NMS heads.

**`conf`**, finer, at 640: 11s has 0 misses from 0.01 to 0.08 (0.1, today's value, misses the card in the border-gaps scan); 26n one-to-many only at 0.05 and 0.06 (below, it finds 5 cards in the 4-card photo; above, it misses cards in the 9-card scans); 26s one-to-many only at 0.03 and 0.04, and leaves 9 or 10 images to the box fallback. So only 11s has a stable range.

**`imgsz`** (480, 640, 800, 960): 11s is clean at 480 for `conf` 0.03 to 0.15 and YOLO takes 45 ms instead of 75 ms, but in the full pipeline it loses one image (see above). Larger sizes don't help any model: more images go to the box fallback and YOLO is 1.5–4x slower. YOLOE-26 at 480 misses 2 cards for every `conf`.

**`iou`** (0.0, 0.3, 0.5, 0.7, NMS heads only): anything above 0.0 overcounts, on every model (26s: 22 to 32 misses), because the card and its art box both survive. Keep 0.0.

**`retina_masks`**: on 11s at 480 it recovers the image that 480 loses (254 / 257 / 257), but costs the time 480 saved; on 26n it trades one card-name loss for an exact-printing loss. Not worth it at 640.

**`enhance_contrast`**: it costs 30–40 ms per query image (segmentation 0.130 → 0.094 s on 11s), more than half of YOLO's own time, but turning it off loses 3 exact printings on 11s (251 / 257 / 256), and 2 on 26n. Keep it; a cheaper contrast step could be worth a later look.

## Fallbacks

At the recommended settings the box fallback and rembg never run (YOLO finds the cards in 247 images; 9 take the scan bypass; 1 has no card). Today's settings run rembg once, on the border-gaps scan, which also dominates its 95th percentile. YOLOE-26 at today's settings sends 30 (26n) to 83 (26s) images to the box fallback and 8 to rembg.

## ONNX

- **Exports** (ultralytics 8.4.21, `uv run --with onnx --with onnxslim`, `YOLO_AUTOINSTALL=false`): `set_classes(["book cover", "art work"])`, then `export(format="onnx", agnostic_nms=True, imgsz=640, opset=20, max_det=50)`, static and dynamic. For the one-to-many head of YOLOE-26, set `end2end = False` on the model and its head modules before exporting; the output is then `[1, 38, 8400]` like YOLOE-11. Files in `output/measure/onnx/`.
- **Static exports change results**: the square 640×640 input pads differently from `.pt`. 11s static finds 5 cards in the 4-card photo; 26s and 26n static lose or gain cards in the 9-card scans. Dynamic exports pad like `.pt`.
- **Dynamic exports give the same results as `.pt`**: card images were byte-identical on every image compared (149 card images each for 11s at 480 and 26n one-to-many, on the tune half), and the full runs gave the same benchmark results.
- **Speed**: no gain in inference at 4 threads on this CPU (11s: 0.087 s ONNX against 0.082 s `.pt`, YOLO median; 26n: 0.034 against 0.039 s). The gain is in loading and memory: 0.08 s instead of 1.16 s, and about 420 MB less after loading.
- **Size**: the 11s ONNX file is 40.8 MB, larger than its 27.8 MB `.pt` (the `.pt` stores half-precision weights). 26n ONNX is 11.5 MB.
- **Odd result, not explained**: 26s `.pt` with the one-to-one head takes 0.26–0.29 s for YOLO alone, 3x its one-to-many head and its own ONNX export (0.083 s). Not profiled, since that head is ruled out anyway.

**Model load** (separate process each time, median of 3, 4 threads): 11s with the text encoder 0.97 s, with precomputed embeddings 0.58 s, ONNX 0.11 s. 26s: 0.86 / 0.59 / 0.15 s. 26n: 0.86 / 0.60 / 0.08 s.

## Removing `torch` and `clip`

What it would involve, from running it here:

- **Our own pre- and post-processing reproduces ultralytics exactly.** `scripts/yoloe_onnx.py` (about 130 lines with NMS) copies 8.4.21's letterbox, normalisation, top-k filter or NMS, mask decoding, contours and scaling back. On the same exports, its card images are byte-identical to ultralytics' own ONNX path: 277 of 277 for 11s, 297 of 297 for 26s. It still imports `merge_multi_segment` from ultralytics for a mask with several contours; production would copy that function (about 30 lines).
- With onnxruntime used directly, the thread count can be set (`intra_op_num_threads`), which ultralytics 8.4.21's ONNX path doesn't allow.
- **Runtime would need only onnxruntime, numpy and OpenCV**, all already dependencies (rembg already runs through onnxruntime). `torch`, `torchvision`, `ultralytics` and `clip` would move to a dev-only export step. In this venv they take about 730 MB (`torch` 714 MB). `ultralytics` is the only code in `src/` that imports any of them.
- The text encoders (`mobileclip_blt.ts` 600 MB, `mobileclip2_b.ts` 254 MB) would no longer be loaded at runtime; the prompts are fused into the exported weights.
- Not measured: the Docker image size, and whether the export runs in ONNX Runtime Web with the same output and at which opset (the exports here use opset 20).

## Adoption and browser port rules

| Candidate | Rule | Outcome |
|---|---|---|
| 11s, `conf` 0.05 | Matches or beats the baseline; no loss | Adopt |
| 11s as dynamic ONNX, onnxruntime directly | Same model, faster way of running it: allowed by the browser port rule | Adopt; no inference speedup, but faster loading and less memory |
| 11s, `imgsz` 480 | 1.07x, below 1.2x; one image lost | Not adopted |
| 26n, one-to-many, `conf` 0.05 | 59% smaller and no slower, but 3 card names lost (2 single-card photos and a 9-card scan) | Maintainer decides; recommended against, since 11s at `conf` 0.05 is better on every accuracy measure |
| 26s, any settings | Worse accuracy, no faster, larger | Rejected |
| YOLOE-26 one-to-one head | No workable `conf` | Rejected |

YOLOE-26n's ONNX export works and stays within the measured results above when run as ONNX, so it meets the browser port rule if the maintainer takes it.

## Not measured, left open

- Whether results change with the latest ultralytics (8.4.207). Production would no longer depend on it at runtime if ONNX is adopted; an export made with a newer version should be checked against these results.
- The Docker image size without `torch`, and ONNX Runtime Web.
- Why `enhance_contrast` costs as much as it does.
- Timing noise: the same configuration varied by up to 9% between repeats (105.9 to 115.4 s), so differences under about 5% in total time are not meaningful.
