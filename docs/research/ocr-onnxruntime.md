# PP-OCR in onnxruntime without paddlepaddle

Task for issue #199 (map #189): can the PP-OCR weights run in `onnxruntime` without `paddlepaddle` or `paddlex`, at the same accuracy? Measured 2026-11-10 on the research branch `research/ocr-onnxruntime` (never merged, based on `research/ocr-candidates`), on the AMD Ryzen AI 7 350 (16 threads), at 4 threads throughout, on the 1,291 card images of #196 (`scripts/ocr_sample.json`).

## Three ways to run the ONNX exports

All three use the official ONNX exports (`PaddlePaddle/*_onnx` on Hugging Face, in `models/ocr/test/onnx`) and today's `read_card` (crops, upscaling, `text_det_thresh=0.05`).

1. **PaddleOCR's onnxruntime engine**: `PaddleOCR(engine="onnxruntime", engine_config={"intra_op_num_threads": 4})`, paddleocr 3.7.0 and paddlex 3.7.2, **paddlepaddle uninstalled**. Run in two venvs: the project's with paddlepaddle removed, and a minimal one (`paddleocr==3.7.0`, `onnxruntime`, the package itself) with `opencv-contrib-python` 4.10.0.84 as the only OpenCV package.
2. **RapidOCR 3.10.0** (`scripts/engine_run.py --engine rapidocr`), in its own venv: rapidocr, onnxruntime and the package, with no paddleocr or paddlex. It brings `opencv-python-headless` 5.0.0.93 and numpy 2.5.4. Character lists come from each model's `inference.yml`, since the official exports carry none. Run with its defaults (thresholds set to today's) and with `--paddle-settings`, which sets what PaddleX does differently: ImageNet normalisation for detection, a 64 px minimum side, no resizing or padding before detection, no dilation, no score filter.
3. **Our own code** (`scripts/own_ocr.py`, `--engine own`): a prototype that reproduces PaddleX's OCR pipeline with today's settings, on onnxruntime, numpy, OpenCV, pyclipper and PyYAML. Run in a venv with only those (`opencv-python-headless` 4.10.0.84, numpy 2.3.5).

`scripts/readings_diff.py` compares two runs token by token; `ocr_compare.py report` scores raw text as in #196.

## Results

### Same output as Paddle?

| Runtime | v5_mobile: same texts / same reading | v6 small, own settings | v5 det + v6 tiny rec |
|---|---|---|---|
| Paddle (reference) | 1291 / 1291 | 1291 / 1291 | 1291 / 1291 |
| PaddleOCR engine, no paddlepaddle | **1291 / 1291** | 1288 / 1289 ¹ | **1291 / 1291** |
| Own code | **1291 / 1291** | 1288 / 1289 ¹ | **1291 / 1291** |
| RapidOCR, `--paddle-settings` | 459 / 784 | 517 / 780 ² | |
| RapidOCR, its defaults | 3 / 76 | | |

¹ The same 3 card images as #196's onnxruntime run (a space in `140/269 C`, `2029` against `2020`, `JaimeJones`); against PaddleOCR's onnxruntime engine, both give 1291 / 1291. ² Against PaddleOCR's onnxruntime engine.

Boxes are identical (at most 1 px apart), and scores differ by at most 0.0003 (v5_mobile). **PaddleOCR's engine and our own code give the same tokens, boxes and readings as each other on every card image, for all three candidates.**

**RapidOCR doesn't, even with PaddleX's settings**: it sorts boxes on one line differently, drops empty tokens, and recognises some lines differently. Part of that is OpenCV: our own code run under OpenCV 5.0.0.93 (RapidOCR's venv) differs from Paddle on 13 of 80 card images, and under `opencv-python-headless` 4.10.0.84 on none. So the OpenCV version changes readings whatever the runtime, and the headless and contrib packages of the same version read the same.

### Accuracy (raw text, against what each card prints, as in #196)

Every card image (1,291):

| Run | name | set code | number | size | year | artist | same name first |
|---|---|---|---|---|---|---|---|
| v5_mobile, Paddle | 1014 | 639 | 887 | 435 | 880 | 1031 | 612 |
| v5_mobile, own code | 1014 | 639 | 887 | 435 | 880 | 1031 | 612 |
| v5_mobile, RapidOCR `--paddle-settings` | 1015 | 640 | 889 | 435 | 889 | 1039 | 615 |
| v5_mobile, RapidOCR defaults | 942 | 643 | 899 | 425 | 870 | 984 | 581 |
| v6 small own settings, PaddleOCR onnxruntime | 1046 | 653 | 932 | 474 | 1036 | 1116 | 643 |
| v6 small own settings, own code | 1046 | 653 | 932 | 474 | 1036 | 1116 | 643 |
| v6 small own settings, RapidOCR `--paddle-settings` | 1041 | 654 | 932 | 475 | 1038 | 1116 | 645 |

RapidOCR with PaddleX's settings reads about as well as Paddle (a few more hits on most fields, 5 fewer v6 small names), but with different tokens, so end-to-end results with today's extraction would move. With its defaults it reads worse (72 fewer names, 47 fewer artists).

### Speed, memory, load

| Run | OCR per card, median / p95 | det / rec median | load | peak RSS after load / end | repeats |
|---|---|---|---|---|---|
| v5_mobile, Paddle (#196) | 0.136 / 0.200 s | 0.055 / 0.075 s | 0.82 s | 697 / 1526 MB | 3 |
| v5_mobile, PaddleOCR onnxruntime, paddlepaddle installed (#196) | 0.130 / 0.192 s | 0.059 / 0.068 s | 0.41 s | 498 / 1269 MB | 3 |
| v5_mobile, PaddleOCR onnxruntime, minimal venv | 0.135 / 0.209 s | 0.061 / 0.070 s | 0.36 s | 242 / 1010 MB | 3 |
| v5_mobile, own code | 0.135 / 0.204 s | 0.064 / 0.069 s | 0.29 s | 147 / 827 MB | 3 |
| v5_mobile, RapidOCR `--paddle-settings` | 0.153 / 0.230 s | 0.076 / 0.074 s | 0.39 s | 96 / 433 MB | 3 |
| v5_mobile, RapidOCR defaults | 0.892 / 1.027 s | 0.793 / 0.088 s | 0.40 s | 96 / 872 MB | 1 |
| v6 small, PaddleOCR onnxruntime (#196) | 0.161 / 0.224 s | 0.077 / 0.082 s | 0.40 s | 509 / 1350 MB | 3 |
| v6 small, own code | 0.167 / 0.237 s | 0.081 / 0.084 s | 0.28 s | 159 / 755 MB | 1 |
| v5 det + v6 tiny rec, Paddle (#196) | 0.078 / 0.103 s | 0.048 / 0.024 s | 0.67 s | 680 / 1131 MB | 3 |
| v5 det + v6 tiny rec, own code | 0.081 / 0.099 s | 0.062 / 0.019 s | 0.11 s | 113 / 465 MB | 1 |

Peak RSS is the whole process (catalog, sample bookkeeping, readings kept for writing), so only the "after load" column compares runtimes cleanly. All onnxruntime routes run at Paddle's speed (within the ~5% noise of #196); the gain is in load time and memory: **our own code takes 550 MB less than Paddle after loading, and about 100 MB less than PaddleOCR's engine**. RapidOCR's defaults scale every crop up to a 736 px short side before detection, which makes them 6.6x slower.

### Does it run without paddlepaddle, and with which OpenCV?

- **PaddleOCR's engine**: yes. paddleocr 3.7 builds and runs the pipeline with paddlepaddle uninstalled, and in a venv that never had it. paddlex still needs a package named `opencv-contrib-python` (#192), and it ran with that as the only OpenCV package. It still installs paddlex and its dependencies (modelscope, huggingface-hub, pandas, pycryptodome, pypdfium2, …).
- **RapidOCR**: needs `opencv-python-headless` and pins nothing, so it took 5.0.0.93; it also brings omegaconf.
- **Own code**: any OpenCV package; `opencv-python-headless` 4.10.0.84 reads the same as today's contrib package.

### Dependencies and image size

Production venvs (`uv sync --no-dev`, everything else unchanged, so torch and ultralytics stay):

| OCR route | venv | packages |
|---|---|---|
| today (paddlepaddle, paddleocr 3.4) | 2,960 MB | 151 |
| PaddleOCR onnxruntime engine, no paddlepaddle | 2,285 MB (−675) | 148 |
| RapidOCR instead of paddleocr | 2,337 MB (−623) | 137 |
| own code (onnxruntime already installed; pyclipper, 5 MB, added) | 2,215 MB (−745) | 131 |

paddlepaddle alone is 676 MB. Removing paddleocr and paddlex as well drops 17 more packages (about 70 MB). RapidOCR's venv is larger than the PaddleOCR one because it adds a third OpenCV package (`opencv-python-headless` 5.0.0.93, about 80 MB of libraries), which writes into the same `cv2` directory as today's two: the install-order problem of #110, with a third version. With one OpenCV package (#192) it would be about own code's size plus 10 MB. Own code needs only `pyclipper` (already installed through paddlex today, so it would become a direct dependency) and PyYAML for the character list (or a text file).

A venv for OCR alone: own code 280 MB (onnxruntime, numpy, OpenCV headless, pyclipper, PyYAML), PaddleOCR's engine 535 MB, RapidOCR 358 MB.

Docker image, built from today's Dockerfile with only the dependencies changed (`docker images`: disk usage / compressed content size):

| Image | disk | content |
|---|---|---|
| today | 9.25 GB | 2.16 GB |
| no paddlepaddle, paddleocr 3.7 | 7.47 GB (−1.78) | 1.79 GB (−0.37) |

The disk saving is about 2.6 times the venv's, because the Dockerfile holds the venv in more than one layer (the `deps` stage copied into `runtime`, then `uv sync` again). An image with our own code wasn't built; by the same ratio it would be about 0.2 GB smaller again on disk. Removing torch and ultralytics, and the image's duplicated venv layers, are outside this ticket.

### ONNX Runtime Web

`scripts/ort_web_check.py` runs each of the six official exports (v5_mobile, v6 small and v6 tiny, detection and recognition) through onnxruntime-web 1.31.1 on its WASM backend in Node 22, on inputs built from a real card image, and compares with Python onnxruntime 1.24.3. **All six load and run**, with a largest output difference of 1.5e-4 (detection) and 6.9e-5 (recognition): every text-map pixel binarises the same at 0.05, and recognition picks the same character at every step. One input per model; operator support is what this establishes, not speed (WASM measurements are out of scope).

### RapidOCR's ONNX files

RapidOCR's v6 small files (`RapidAI/RapidOCR` v3.10.0 on ModelScope) are separate exports, not copies: detection is opset 11 where the official one is 14 (464 nodes against 242, with Identity nodes), and recognition carries the character list in its metadata. They hold the same weights: on the same input, detection output is identical and recognition differs by at most 7e-7.

## What production would need

- **PaddleOCR's onnxruntime engine**: a one-line change to `load_paddleocr`, plus the ONNX files in `models/ocr/prod` and `paddleocr>=3.5` (3.7 tested). Keeps paddleocr, paddlex and their 17 extra packages, and paddlex's check for a package named `opencv-contrib-python`, so OpenCV stays tied to paddlex (#192). Nothing of our own to maintain, but upgrades of paddleocr can change behaviour (3.4 → 3.7 changed numpy, #196).
- **RapidOCR**: not a drop-in. Its readings differ from Paddle's on most card images, so extraction tuned on Paddle tokens would need rechecking, and it needs PaddleX's settings to match its speed and accuracy. It also brings its own OpenCV package and version.
- **Own code**: `scripts/own_ocr.py`, **177 lines of code** (242 with comments and docstrings), highest function complexity 14 (`detect`, `uvx radon cc`). It reproduces PaddleX's detection resize and normalisation, DB post-processing (contours, mean score, unclip with pyclipper), box sorting, rotated crops, batched recognition (6 lines per batch, sorted by width and padded to the widest, which changes the output, so it is kept) and CTC decoding. Same output as Paddle on every card image tested, for v5 and v6 models without changes. What it costs to maintain: it is PP-OCR's DB/CTC pre- and post-processing, which has been stable since PP-OCRv2 and is shared by every model we compared; a different model family would need new code. It is plain functions over numpy arrays, so tests can call `detect` and `recognize` directly. It is also what the browser port has to write in JavaScript anyway, so the Python version doubles as its reference.

## Answers

- **Can the PP-OCR weights run in onnxruntime without paddlepaddle or paddlex, at the same accuracy?** Yes. Through PaddleOCR's engine without paddlepaddle, and through our own code without paddlepaddle, paddlex or paddleocr, the tokens, boxes and readings are the same as Paddle's on all 1,291 card images (v5_mobile, v5 det + v6 tiny rec), and the same as PaddleOCR's onnxruntime engine for v6 small.
- **Do the outputs still provide what extraction needs, boxes included?** Yes: texts, scores and boxes in PaddleOCR's format (`rec_texts`, `rec_scores`, `rec_boxes`), and `read_card` runs unchanged on them.
- **Time and memory**: the same speed as Paddle; loads in 0.1–0.3 s against 0.7–0.8 s; about 550 MB less memory after loading.
- **Dependencies**: −675 MB without paddlepaddle; −745 MB and 20 packages without paddleocr and paddlex as well.
- **ONNX Runtime Web**: all six exports run with the same output.
- **OpenCV**: with paddlex gone, any OpenCV package works for OCR, and `opencv-python-headless` 4.10.0.84 reads the same as today's package. Keep the version fixed: OpenCV 5 changes readings.

## Recommendation

**Run PP-OCR through onnxruntime with our own code** (about 180 lines in `engine/ocr`), and drop paddlepaddle, paddleocr and paddlex. It gives the same readings as today, at the same speed, with less memory and a faster load. It removes paddlex, one of the two packages behind the OpenCV conflict, and it is the code the browser port needs.

PaddleOCR's onnxruntime engine is a cheaper first step if the final decision wants one: a one-line change with the same readings that drops paddlepaddle (−675 MB), but keeps paddlex and its OpenCV check. RapidOCR is not recommended: its readings differ from today's, and it brings its own OpenCV version.

Whichever model the final decision picks (v6 small, or v5 det + v6 tiny rec), these results hold for it: all three candidates gave the same output through our own code.

## Assumptions and things not measured

- Timing noise between repeats is up to about 5% (#196). Runs with 1 repeat are marked.
- End-to-end runs (`measure.py`) weren't repeated: identical readings on the same card images give identical end-to-end results, so #196's end-to-end numbers hold for PaddleOCR's engine and own code. RapidOCR's end-to-end results weren't measured.
- The own-code prototype hasn't been tested on images outside the 1,291 card images: very wide lines (over 3,200 px after scaling) and tiny crops follow PaddleX's code but weren't exercised.
- ONNX Runtime Web was checked in Node, not in a browser, on one input per model.
- Docker sizes compare images built from the same Dockerfile; the apt packages for OpenCV's GUI libraries are left to the OpenCV decision.

## How to rerun

```bash
# PaddleOCR's engine without paddlepaddle (a venv with paddleocr==3.7.0, no paddlepaddle)
python scripts/ocr_compare.py run min-ort-v5 --onnxruntime --repeats 3 \
    --det PP-OCRv5_mobile_det=models/ocr/test/onnx/PP-OCRv5_mobile_det \
    --rec PP-OCRv5_mobile_rec=models/ocr/test/onnx/PP-OCRv5_mobile_rec
# own code (a venv with onnxruntime, numpy, opencv-python-headless==4.10.0.84, pyclipper, pyyaml, pillow, and the package with --no-deps)
PYTHONPATH=scripts python scripts/engine_run.py own-v5 --engine own --repeats 3 \
    --det models/ocr/test/onnx/PP-OCRv5_mobile_det --rec models/ocr/test/onnx/PP-OCRv5_mobile_rec
# RapidOCR (a venv with rapidocr==3.10.0 and onnxruntime)
PYTHONPATH=scripts python scripts/engine_run.py rapid-v5-paddle --engine rapidocr --paddle-settings \
    --det models/ocr/test/onnx/PP-OCRv5_mobile_det/inference.onnx \
    --rec models/ocr/test/onnx/PP-OCRv5_mobile_rec/inference.onnx \
    --keys models/ocr/test/onnx/PP-OCRv5_mobile_rec/inference.yml
# compare two runs, and score raw text
python scripts/readings_diff.py v5-mobile own-v5
python scripts/ocr_compare.py report v5-mobile own-v5 rapid-v5-paddle
# ONNX Runtime Web (npm install onnxruntime-web in ORT_WEB_DIR)
ORT_WEB_DIR=/tmp/ortweb python scripts/ort_web_check.py CARD_IMAGE
```

Outputs (gitignored, on the measuring machine): `output/ocr_compare/<run>/` for each run above.
