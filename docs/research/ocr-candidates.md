# OCR candidates: PP-OCRv5_mobile against the PP-OCRv6 tiers

Task for issue #196 (map #189): which OCR model and runtime should we use, judged without depending on today's extraction code? Measured 2026-11-09 on the research branch `research/ocr-candidates` (never merged), on the AMD Ryzen AI 7 350 (16 threads), at 4 threads throughout.

## Setup

- **Library upgrade alone** (v5_mobile, Paddle): paddleocr 3.4.0 and 3.7.0 give identical text, scores and boxes on all 1,291 card images. Speed isn't settled: 3.4.0 ran once (0.163 s per card) against three runs on 3.7.0 (0.136 s), and the 3.7.0 install also moved numpy from 2.4.2 to 2.3.5.
- **Libraries**: the experiment venv has paddleocr 3.7.0 and paddlex 3.7.2 (installed with `uv pip`; `pyproject.toml` unchanged), with paddlepaddle 3.2.0 and onnxruntime 1.24.3. paddlepaddle 3.2.0 loads and runs the PP-OCRv6 static graphs (verified).
- **Models**: `PP-OCRv5_mobile_{det,rec}` from `models/ocr/prod`; `PP-OCRv6_{tiny,small}_{det,rec}` from Hugging Face `PaddlePaddle/PP-OCRv6_*`, and the official ONNX exports `PaddlePaddle/*_onnx`, all in the gitignored `models/ocr/test`.
- **Card images** (`scripts/ocr_sample.json`, built by `scripts/ocr_compare.py sample`), 1,291 in all:
  - **benchmark**: the 275 card images saved by #195's recommended segmentation (`output/measure/g-11s-dyn-640-0.05`), each with its labelled printing; one card of the 9-card scan without border gaps has no matching label and is left out.
  - **reference**: 100 reference images per frame (`1993`, `1997`, `2003`, `2015`, `future`, normal layout only) and per layout with at least 400 catalog rows (`transform`, `adventure`, `modal_dfc`, `saga`, `split`), one printing per set in a shuffled set order (seed 189), round after round.
  - **uncommon_layout**: the 16 photos in `data/scratch/uncommon_layout`, segmented once with today's segmentation.
  - Halves: benchmark cards follow `scripts/measure_split.json`; the others alternate, 50 to tune and 50 to confirm per category (8/8 for the photos).
- **Reading**: `read_card` unchanged (today's crops, upscaling and `text_det_thresh=0.05`), except where a run sets other detection settings. Timing wraps the detection and recognition models inside PaddleOCR's pipeline; each card image is read 3 times and the median kept, then the median and 95th percentile over card images are reported.

## Two accuracy measures

1. **Raw text** (`ocr_compare.py report`), independent of extraction. For each card image: does the raw text of the name crop hold the name (either face; letters and digits only), and does the raw text of the collector information crop hold the set code, collector number, base set size, year (matched as in `printed-details/check.py`, #201) and artist (rapidfuzz `partial_ratio` ≥ 90)? Values are **what the card prints**, by the printed-detail rules of #201; a detail the card doesn't print is left out of the count, so the denominators differ per field.
2. **Same-name ranking**, with today's extraction and scoring: today's reading ranked against every printing of the same name (shuffled with seed 189), leaving out names with one printing. Counts how often the correct printing scores strictly first.
3. **End to end**, with today's extraction: `scripts/measure.py run --cards-from output/measure/g-11s-dyn-640-0.05`, 3 repeats, against `o-v5` (v5_mobile on the same card images) as the reference. These card images give 254/257/257 with v5_mobile (#195), not the baseline's 253/256/256.

## Results

### Every card image (1,291)

| Candidate | name | set code | number | size | year | artist | same name first | OCR per card, median / p95 | det / rec median | size |
|---|---|---|---|---|---|---|---|---|---|---|
| v5_mobile (today) | 1014 | 639 | 887 | 435 | 880 | 1031 | 612 | 0.136 / 0.200 s | 0.055 / 0.075 s | 21.7 MB |
| v6 small, pipeline settings | 1033 | 622 | 932 | 472 | 1020 | 1097 | 628 | 0.174 / 0.237 s | 0.083 / 0.087 s | 31.5 MB |
| **v6 small, own detection settings** | **1046** | **653** | **932** | **474** | **1035** | **1116** | **643** | 0.196 / 0.272 s | 0.084 / 0.107 s | 31.5 MB |
| v6 tiny, pipeline settings | 994 | 608 | 878 | 420 | 944 | 1059 | 569 | 0.060 / 0.083 s | 0.036 / 0.021 s | 6.5 MB |
| v6 tiny, own detection settings | 995 | 643 | 904 | 430 | 937 | 1053 | 577 | 0.075 / 0.107 s ¹ | 0.037 / 0.034 s | 6.5 MB |
| **v5_mobile det + v6 tiny rec** | 996 | 644 | 901 | 450 | 956 | 1061 | 601 | 0.078 / 0.103 s | 0.048 / 0.024 s | 9.5 MB |
| v5_mobile det + v6 small rec | 1037 | 653 | 924 | 464 | 1010 | 1108 | 629 | 0.183 / 0.259 s ¹ | 0.055 / 0.123 s | 26.4 MB |
| v6 tiny det + v6 small rec | 1033 | 618 | 897 | 436 | 1002 | 1103 | 614 | 0.163 / 0.243 s ¹ | 0.045 / 0.113 s | 23.4 MB |
| v6 tiny det + v5_mobile rec | 1016 | 618 | 866 | 405 | 842 | 1024 | 589 | 0.142 / 0.225 s ¹ | 0.044 / 0.092 s | 18.8 MB |
| out of | 1291 | 730 | 1078 | 569 | 1262 | 1291 | 1112 | | | |

¹ 1 repeat; the others 3. "Own detection settings" are those in the model's `inference.yml`: `thresh 0.2`, `box_thresh 0.45` (small) or `0.4` (tiny), `unclip_ratio 1.4`. "Pipeline settings" are today's: `thresh 0.05`, `box_thresh 0.6`, `unclip_ratio 1.5`.

The tune and confirm halves agree. Same name first, tune / confirm: v5_mobile 302/561 and 319/563; v6 small with own settings 317 and 336; v5 det + v6 tiny rec 298 and 309; v6 tiny with own settings 280 and 302.

### Benchmark card images (275)

| Candidate | name | set code | number | size | year | artist | same name first |
|---|---|---|---|---|---|---|---|
| v5_mobile | 222 | 235 | 253 | 175 | 185 | 182 | 193 |
| v6 small, pipeline settings | 231 | 209 | 261 | 184 | 227 | 209 | 191 |
| v6 small, own settings | 232 | 241 | 264 | 187 | 236 | 215 | 196 |
| v6 tiny, pipeline settings | 218 | 201 | 220 | 141 | 210 | 182 | 162 |
| v6 tiny, own settings | 218 | 234 | 246 | 158 | 212 | 177 | 173 |
| v5 det + v6 tiny rec | 216 | 244 | 256 | 177 | 220 | 182 | 186 |
| out of | 275 | 253 | 274 | 197 | 275 | 275 | 200 |

### Per category, the leading candidates

Name / year / artist hits of 100 (frames 1993 to 2003 print no set code; 1993 prints no number):

| Category | v5_mobile | v6 small, own settings | v5 det + v6 tiny rec |
|---|---|---|---|
| frame 1993 | 59 / 36 of 71 / 85 | 58 / 53 / 92 | **39** / 44 / 83 |
| frame 1997 | 68 / 66 / 91 | 84 / 83 / 96 | 76 / 74 / 93 |
| frame 2003 | 93 / 62 / 90 | 92 / 83 / 92 | 92 / 66 / 90 |
| frame 2015 | 90 / 76 / 88 | 93 / 84 / 93 | 91 / 83 / 94 |
| frame future | 99 / 92 / 93 | 99 / 97 / 94 | 98 / 95 / 94 |
| adventure | 94 / 75 / 81 | 95 / 83 / 92 | 94 / 79 / 94 |
| modal_dfc | 90 / 67 / 90 | 90 / 78 / 92 | 90 / 73 / 89 |
| saga | 94 / 79 / 83 | 96 / 87 / 94 | 96 / 80 / 89 |
| split | 5 / 50 / 48 | 5 / 52 / 51 | 5 / 49 / 49 |
| transform | 94 / 77 / 84 | 95 / 83 / 89 | 94 / 79 / 89 |
| uncommon photos (of 16) | 6 / 15 / 16 | 7 / 16 / 16 | 5 / 14 / 15 |

Collector number hits, old frames: 1997 35 / 46 / 40 of 56, 2003 38 / 50 / 38 of 61, future 83 / 91 / 90 of 92. Split cards: no model reads the names (5/100), since they are sideways and outside the name crop; same name first is 0/100 for every model, because the name field scores near zero.

### End to end, with today's extraction (257 query images, 3 repeats)

| Candidate | exact / count / name | changed images, against v5_mobile on these cards | run, per repeat | per query image, median / p95 | OCR det + rec, summed |
|---|---|---|---|---|---|
| v5_mobile | 254 / 257 / 257 | (reference) | 82.0–83.8 s | 0.325 / 0.412 s | 54.6 s |
| **v6 small, own settings** | **255 / 257 / 257** | `1_photos_single_card/b845b1d7…` now exact; none worse | 91.8–92.5 s | 0.348 / 0.467 s | 64.9 s |
| v6 small, pipeline settings | 254 / 257 / 257 | b845b1d7 now exact; `6_photos…/89ba71fa…` loses exact | 86.1–87.4 s | 0.335 / 0.422 s | 58.2 s |
| **v5 det + v6 tiny rec** | 253 / 257 / 256 | `1_photos_single_card/b86a6181…` loses exact and card name | 51.1–51.6 s | 0.207 / 0.245 s | 25.1 s |
| v5 det + v6 small rec | 253 / 257 / 256 | b86a6181 loses exact and card name | 85.6–88.0 s | 0.338 / 0.425 s | 59.3 s |
| v6 tiny, own settings | 243 / 257 / 256 | 12 images lose exact, 1 loses card name | 48.0–49.5 s | 0.197 / 0.239 s | 22.3 s |
| v6 tiny, pipeline settings | 238 / 257 / 256 | 17 images lose exact, 1 loses card name | 48.3–50.2 s | 0.203 / 0.238 s | 21.5 s |

Here segmentation reads saved PNGs, so these runs are not whole-pipeline times. **Estimated whole-pipeline speed**, from #194's 4-thread run (109–112 s per repeat) minus the OCR time each candidate saves: v5 det + v6 tiny rec about 80 s, **1.37x**; v6 small with own settings about 120 s, **0.91x**.

**b86a6181 is an extraction miss, not a reading miss.** It is a German *Skyshroud Elf* (Tempest). v6 tiny rec reads the year as `01997` where v5_mobile read `C1997`, and `extract_copyright_year` doesn't accept the leading `0`, so the year field is lost and another printing wins by 1.44. The raw text holds `1997` in both.

### Runtime: Paddle against onnxruntime, through PaddleOCR

`PaddleOCR(engine="onnxruntime", engine_config={"intra_op_num_threads": 4})` with the official ONNX exports, no new package (paddleocr 3.7.0 already has the engine), 3 repeats:

| Candidate | runtime | OCR per card, median / p95 | det / rec median | load | peak RSS after load / end | same texts as Paddle |
|---|---|---|---|---|---|---|
| v5_mobile | Paddle | 0.136 / 0.200 s | 0.055 / 0.075 s | 0.82 s | 697 / 1526 MB | |
| v5_mobile | onnxruntime | 0.130 / 0.192 s | 0.059 / 0.068 s | 0.41 s | 498 / 1269 MB | 1291 / 1291 |
| v6 small, own settings | Paddle | 0.196 / 0.272 s | 0.084 / 0.107 s | 0.76 s | 707 / 1544 MB | |
| v6 small, own settings | onnxruntime | 0.161 / 0.224 s | 0.077 / 0.082 s | 0.40 s | 509 / 1350 MB | 1288 / 1291 |
| v5 det + v6 tiny rec | Paddle | 0.078 / 0.103 s | 0.048 / 0.024 s | 0.67 s | 680 / 1131 MB | |
| v5 det + v6 tiny rec | onnxruntime | 0.078 / 0.096 s | 0.056 / 0.019 s | 0.11 s | 464 / 898 MB | 1291 / 1291 |

Raw-text and same-name counts are identical between the two runtimes, except one more year with v6 small on onnxruntime. Peak RSS is the whole process (catalog, sample bookkeeping), so only the differences mean something.

### `precision="fp16"`

Removing it changes nothing: identical tokens, scores and boxes on all 1,291 card images (v5_mobile, Paddle).

## Answers

- **Does v6 read better?** v6 small does, on every field and every category, by the most on old frames (1997 names 68 → 84, years 66 → 83; 2003 numbers 38 → 50) and on years and artists everywhere. It needs its own detection settings: with today's pipeline settings v6 small detection misses the set code and artist line on many 2015-frame photos (benchmark set codes 235 → 209), and with its own it reads 241.
- **Do the claimed gains hold?** Accuracy, yes for small; speed, no. On this CPU and these crops v6 small is about 1.4x slower than v5_mobile on Paddle (0.196 against 0.136 s per card), not equal as Baidu's table says, and about 1.24x slower on onnxruntime. Tiny is 2.2x faster.
- **Is tiny accurate enough to replace small?** No, not as a pair: tiny detection misses text on photos (benchmark numbers 253 → 220–246), and end to end it loses 11 to 16 exact printings net. Tiny *recognition* after v5_mobile detection is close to v5_mobile on raw text (better on set code, number, size, year; worse on 1993-frame names, 59 → 39) at 1.37x on the whole pipeline.
- **The gap between raw text and extraction**: v5 det + v6 tiny rec reads more set codes, numbers and years than v5_mobile, yet ranks the correct printing first less often (601 against 612). The one benchmark image it loses is the `01997` year, an extraction miss. So part of its loss is extraction tuned on v5 output.
- **`enable_hpi=True`**: **skipped**, by the maintainer's decision. It needs `ultra-infer-python`, published only on Baidu's server. The onnxruntime engine above is HPI's second choice on CPU after OpenVINO, gives most of the gain, and unlike OpenVINO the browser port could use it (ONNX Runtime Web).
- **`precision="fp16"`**: verified no effect on CPU. It can be removed.
- **48 px upscale, detection settings**: left to "Sweep OCR settings on the winning combination" (#198). Only v6's own detection settings against today's were compared here.

## Recommendation

**The extraction prototype should use PP-OCRv6 small with its own detection settings** (`thresh 0.2`, `box_thresh 0.45`, `unclip_ratio 1.4`). It reads best on every field and category, so the prototype starts from the best raw text. End to end it gains one exact printing with no image worse.

Its cost: about 0.91x on the whole pipeline on Paddle, and about the same speed as today on onnxruntime (estimated from the per-card times: its detection and recognition would take about 53 s per run, against 55 s for v5_mobile on Paddle), and 45% larger (31.5 against 21.7 MB). The agreed rules accept trades for speed or size, not for accuracy, so **adopting it is the maintainer's call**.

**The fast option, v5_mobile detection with v6 tiny recognition, is also for the maintainer.** It is 1.37x faster on the whole pipeline (estimated), 56% smaller (9.5 MB), and loses one card name. That loss is an extraction miss, which a new extraction may fix. Its tokens are saved too, so the prototype can score both at almost no cost before the final decision.

Whichever model is chosen, run it through onnxruntime. The gain comes from changing the runtime, not from upgrading paddleocr; the upgrade only makes it possible, since the `engine` option exists from paddleocr 3.5. It gives the same text, loads in half the time or less, uses about 200 MB less memory, is as fast or faster, and is the runtime the browser port needs.

## Assumptions and things not measured

- Whole-pipeline speedups are estimates: #194's whole-pipeline time minus the OCR time measured with `--cards-from`. The candidates weren't run with live segmentation.
- Timing noise between repeats is up to about 5%, so differences under that aren't meaningful. Runs marked ¹ have one repeat.
- Raw-text name hits can't reach 100%: non-English cards print a translated name, and split cards print their names sideways.
- Mixed pairs (v5 det with a v6 rec) use v5_mobile's detection with today's settings. Whether v6 tiny rec reads better at another upscale height is for #198.
- ONNX Runtime Web and RapidOCR weren't tested; HPI (OpenVINO) was skipped.

## How to rerun

```bash
# card image sample (already committed as scripts/ocr_sample.json)
uv run python scripts/ocr_compare.py sample
# one candidate; add --onnxruntime with a directory holding inference.onnx
uv run python scripts/ocr_compare.py run v6small-own --repeats 3 \
    --det PP-OCRv6_small_det=models/ocr/test/PP-OCRv6_small_det \
    --rec PP-OCRv6_small_rec=models/ocr/test/PP-OCRv6_small_rec \
    --det-thresh 0.2 --det-box-thresh 0.45 --det-unclip 1.4
uv run python scripts/ocr_compare.py report v5-mobile v6small-own [--subset tune]
# end to end with today's extraction
uv run python scripts/measure.py run o-v6small-own --threads 4 --repeats 3 \
    --cards-from output/measure/g-11s-dyn-640-0.05 \
    --ocr-det PP-OCRv6_small_det=models/ocr/test/PP-OCRv6_small_det \
    --ocr-rec PP-OCRv6_small_rec=models/ocr/test/PP-OCRv6_small_rec \
    --det-thresh 0.2 --det-box-thresh 0.45 --det-unclip 1.4
```

Outputs (gitignored, on the measuring machine): `output/ocr_compare/<run>/readings.json` holds each card image's tokens, boxes, reading and times; `output/measure/o-*` holds the end-to-end runs.
