# OCR settings sweep

Findings for the ticket "Sweep OCR settings on the winning combination" (#198) of the map #189. Throwaway branch `research/ocr-settings`, never merged, built on `research/extraction`. Scripts: `scripts/ocr_settings_sweep.sh` (the tune-half runs), `scripts/ocr_settings_confirm.sh` (the confirm half and the thread count), `scripts/ocr_settings_table.sh` (one row per run), with new flags on `scripts/ocr_compare.py run` (`--det-limit-side-len`, `--det-limit-type`, `--target-height`, `--ratio-text-to-crop`, `--max-upscale`, `--min-dim`) and on `scripts/extraction.py report` (`--variant`, `--min-rec-score`).

## Setup

- **Both models stay in play**, since the maintainer hasn't yet chosen between them: PP-OCRv6 small (detection and recognition) and the fast option (PP-OCRv5_mobile detection with PP-OCRv6 tiny recognition).
- **Starting points**: v6 small with its own detection settings (`text_det_thresh` 0.2, `text_det_box_thresh` 0.45, `text_det_unclip_ratio` 1.4); the fast option with today's (0.05, 0.6, 1.5). Both with the pipeline's `text_det_limit_side_len` 64 / `text_det_limit_type` min, and today's upscaling (`TARGET_TEXT_HEIGHT` 48, `RATIO_TEXT_TO_CROP` 0.15, `MAX_UPSCALE` 4).
- **Runtime**: PaddleOCR 3.7.0 with `engine="onnxruntime"` and the official ONNX exports, which gives the same text as Paddle (#196, #199). `fp16` off (it does nothing on CPU).
- **Scoring**: candidate-driven extraction, "candidate, shown" (#197), replayed with `scripts/extraction.py report` on the saved tokens. Main measure: same name first; also the benchmark (exact printing / card count / card name, with pHash candidates) and the field score of the correct printing.
- **Halves**: the sample of #196 (1,291 card images), tuned on the tune half (656 card images, 561 same-name cases), confirmed on the confirm half (658, 563). Tune-half runs ran three at a time, so their times aren't compared; confirm-half runs ran one at a time, 3 repeats, 4 threads.
- **Not swept**: oneDNN (it applies to the Paddle runtime only, and the map is heading to onnxruntime); the Paddle against onnxruntime comparison, measured in #196 (onnxruntime up to 1.2x faster, same text); HPI, skipped by the maintainer.

## Round 1: one setting at a time, tune half

Same name first, of 561. Each row changes one setting from the model's starting point.

| Setting | v6 small | Fast option |
|---|---|---|
| starting point | 347 | 338 |
| `text_det_thresh` 0.05 / 0.1 / 0.2 / 0.3 | 348 / 348 / *347* / 348 | *338* / 338 / 337 / 337 (0.05 is the start) |
| `text_det_box_thresh` 0.3 / 0.4 / 0.45 / 0.5 / 0.6 | 348 / – / *347* / – / 346 | – / 338 / – / 338 / *338* |
| `text_det_unclip_ratio` 1.2 / 1.4 / 1.5 / 1.6 / 1.8 / 2.0 | 349 / *347* / – / 347 / – / 348 | – / 335 / *338* / – / 334 / 335 |
| collector information text height 32 / 40 / 48 / 64 px | 348 / 349 / *347* / 348 | 335 / 332 / *338* / 336 |
| detection input 736 / min | 344, power/toughness 96 → 81, detection 8x slower | 333, power/toughness 94 → 11 |
| detection input 960 / max | **351**, faster detection | 336, name 85 → 75 |
| `text_rec_score_thresh` 0.3 / 0.5 / 0.7 (filtering the saved tokens) | 347 / 347 / 346 | 338 / 338 / 336 |

The benchmark is 129 / 130 / 130 in every row except three (128 exact): v6 small at 736 / min, and the fast option at text height 32 and 40 and unclip 1.4.

**Thresholds, box threshold, unclip ratio, recognition score**: flat. Every value is within 3 of the starting point, which is the noise between the two halves (see round 3). Keep each model's starting values.

**Text height**: flat for v6 small, 48 best for the fast option. Keep 48 px.

**Detection input size** is the only setting with an effect, and the reason is in PaddleX's resize (`paddlex/inference/models/text_detection/processors.py`, `resize_image_type0`):

- The default, `64 / min`, upscales a crop only when its smallest side is under 64 px. That applies to the name crop (about 47 px high on a 680 px reference image) and never to the collector information crop, which reaches about 320 × 1,700 px after upscaling. So detection runs on the full 1,700 px crop.
- `960 / max` shrinks the collector information crop to at most 960 px wide for detection only. Recognition still reads text boxes cut from the full-size crop. It also stops upscaling the name crop, which the v5 detection model needs (the fast option's name score falls from 85 to 75).
- `736 / min` upscales every crop to at least 736 px on its short side: slow and worse.

## Round 2: a smaller detection input, the name crop kept at 64 px

`--min-dim 64` upscales a crop to at least 64 px on its short side before OCR (today's `_MIN_DIM` is 32), so the name crop gets what `64 / min` gave it.

| Detection input (max side) | v6 small | Fast option |
|---|---|---|
| default (`64 / min`) | 347 | 338 |
| 640 | 346 | 331 |
| 800 | 352 | 336 |
| 960 | 350 | 336 |
| 1280 | 349 | 337 |
| 960, text height 40 | 352 | – |
| 960, text height 64 | 348 | 334 |

The benchmark is 129 / 130 / 130 in each, except the fast option at 1280 (128 exact).

- **v6 small**: 800 to 960 is a stable range, 3 to 5 above the starting point, with the set code, number, base set size, year and power/toughness scores equal or higher. Chosen: **960** (it keeps the field scores of round 1's 960 / max and the name score of the starting point).
- **Fast option**: no size beats the starting point on the tune half; 960 is 2 below it, the same score as the name-losing 960 / max of round 1 but with the names back (85.1). Carried to the confirm half for its speed.

## Round 3: confirm half, timed

One run at a time, 3 repeats, 4 threads, onnxruntime. OCR time per card image (both crops).

| Run | Benchmark | Same name first (of 563) | OCR median / p95 | Detection median | Recognition median | Load | Peak RSS |
|---|---|---|---|---|---|---|---|
| v6 small, starting point | 130 / 132 / 132 | 365 | 0.171 / 0.236 s | 0.084 s | 0.084 s | 0.45 s | 1,319 MB |
| **v6 small, 960 / max, name ≥ 64 px** | 130 / 132 / 132 | **371** | **0.116 / 0.179 s** | 0.028 s | 0.085 s | 0.42 s | 1,126 MB |
| fast option, starting point | 130 / 132 / 132 | 352 | 0.086 / 0.103 s | 0.065 s | 0.019 s | 0.12 s | 899 MB |
| **fast option, 960 / max, name ≥ 64 px** | 130 / 132 / 132 | **356** | **0.040 / 0.055 s** | 0.020 s | 0.017 s | 0.11 s | 822 MB |

Peak RSS is the whole process, so only differences mean something.

**Both halves together** (tune run of round 2 plus the confirm run), all 1,291 card images and all 257 benchmark images:

| Run | Benchmark | Same name first (of 1,112) | name | set code | number | size | artist | year | P/T |
|---|---|---|---|---|---|---|---|---|---|
| v6 small, starting point | 255 / 257 / 257 | 702 | 84.2 | 86.4 | 86.8 | 83.5 | 93.0 | 82.3 | 96.5 |
| **v6 small, chosen** | 255 / 257 / 257, no image changed | **711** | 84.1 | 87.7 | 87.6 | 84.5 | 93.2 | 83.0 | 97.4 |
| fast option, starting point | 255 / 257 / 257 | 681 | 85.8 | 88.2 | 83.8 | 79.3 | 91.0 | 78.5 | 94.7 |
| **fast option, chosen** | 255 / 257 / 257, no image changed | **683** | 85.6 | 88.4 | 83.8 | 78.6 | 91.0 | 78.8 | 94.6 |

Same name first per group:

| Run | benchmark | 1993 | 1997 | 2003 | 2015 | future | transform | adventure | modal_dfc | saga | split | uncommon |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v6 small, starting point | 196 | 41 | 55 | 60 | 72 | 70 | 52 | 37 | 57 | 48 | 0 | 14 |
| v6 small, chosen | 196 | 43 | 56 | 62 | 72 | 71 | 52 | 39 | 57 | 48 | 1 | 14 |
| fast option, starting point | 193 | 41 | 51 | 54 | 72 | 70 | 51 | 31 | 57 | 47 | 0 | 14 |
| fast option, chosen | 191 | 40 | 52 | 57 | 71 | 71 | 51 | 31 | 57 | 48 | 0 | 14 |
| out of | 200 | 90 | 91 | 87 | 93 | 88 | 88 | 100 | 91 | 70 | 100 | 14 |

v6 small: no group worse. Fast option: three groups down by 1 or 2 (benchmark, 1993, 2015), three up; level overall.

## Thread count, confirm half

One repeat each, onnxruntime's `intra_op_num_threads` and the OMP, MKL and OpenBLAS variables set to the same value. Readings are identical at every count.

| Threads | v6 small, chosen: OCR median / p95 | Fast option, starting point: OCR median / p95 |
|---|---|---|
| 1 | 0.267 / 0.410 s | 0.157 / 0.187 s |
| 2 | 0.156 / 0.242 s | 0.103 / 0.125 s |
| 4 | 0.116 / 0.179 s (3 repeats) | 0.086 / 0.103 s (3 repeats) |
| 8 | 0.108 / 0.166 s | 0.086 / 0.106 s |
| 16 (the machine's count, onnxruntime's default) | 0.245 / 0.349 s | 0.183 / 0.226 s |

4 to 8 is the stable range; 16, every hardware thread, is about 2x slower than 4 (oversubscription, as #194 found for the whole pipeline). Recommended: **4**, the same result as 8 for half the CPU.

## Chosen values

| Setting | Today | v6 small | Fast option | Gain |
|---|---|---|---|---|
| `text_det_thresh` | 0.05 | 0.2 (the model's own) | 0.05 | none from sweeping (flat) |
| `text_det_box_thresh` | 0.6 (pipeline default) | 0.45 (the model's own) | 0.6 | none (flat) |
| `text_det_unclip_ratio` | 1.5 (pipeline default) | 1.4 (the model's own) | 1.5 | none (flat) |
| `text_det_limit_side_len` / `text_det_limit_type` | 64 / min (pipeline default) | **960 / max** | **960 / max** | v6 small: same name first 702 → 711, detection 3x faster; fast option: 681 → 683, detection 3.2x faster |
| smallest crop side before OCR (`_MIN_DIM`) | 32 | **64** | **64** | keeps the name crop at the 64 px the default gave detection; required with 960 / max |
| `TARGET_TEXT_HEIGHT` / `RATIO_TEXT_TO_CROP` / `MAX_UPSCALE` | 48 / 0.15 / 4 | unchanged | unchanged | none (flat or worse) |
| `text_rec_score_thresh` | 0.0 (pipeline default) | 0.0 | 0.0 | none (0.7 is worse) |
| threads | library defaults | **4** | **4** | about 2x against 16 |

### Whole pipeline (estimated)

From #194, the pipeline takes a median 0.429 s per query image at 4 threads, of which OCR is 48%, with today's v5_mobile on Paddle (0.136 s per card image on the full sample, #196). Scaling the OCR share by the measured ratios (chosen against starting point on the confirm half, applied to #196's full-sample onnxruntime times: v6 small 0.161 → 0.109 s, fast option 0.078 → 0.037 s):

| Option | OCR per card image | Whole pipeline against today |
|---|---|---|
| v6 small, chosen | 0.109 s | about 1.10x faster (was about the same speed as today on onnxruntime, #196) |
| fast option, chosen | 0.037 s | about 1.54x faster (was 1.37x, #196) |

Not measured end to end: the settings aren't in the engine yet.

## Verified and assumed

**Verified**:
- Every number in the tables, on the saved tokens, with the commands in the scripts. 37 tune-half runs, 12 confirm-half runs.
- The tune-half baseline on onnxruntime gives the same scores as the Paddle run of #196.
- Detection input size explained from PaddleX 3.7.2's resize code (`resize_image_type0`): `max` shrinks the long side to the limit only when it is over it, `min` grows the short side only when it is under.

**Assumed or not measured**:
- The whole-pipeline speedups are estimates from the OCR share, not runs of `mtg benchmark run`.
- Settings were swept one at a time, then the one with an effect was combined with `_MIN_DIM` and the text height; other combinations weren't tried, on the grounds that every other setting is flat.
- The thread results come from this machine (16 hardware threads); the best count may differ on the Docker host. The setting should be configurable.
- Our own OCR code (#199) must reproduce `960 / max` and the 64 px minimum; it uses the same resize, so it should give the same readings, but that wasn't rerun.
- Same name first moves by up to about 4 between halves for the same settings, so differences under that are noise.
