# A Docker image with headless OpenCV

Task for issue #205 (map #189). Run 2026-11-10 on the research branch `research/headless-opencv` (never merged, based on `research/segmentation-onnx`), on the AMD Ryzen AI 7 350, Docker 29.8.2 (containerd image store). Sizes are `docker images` disk / compressed, the same measure as #204.

## Answer

- **Yes.** With `opencv-python-headless` 4.10.0.84 as the only OpenCV package, no system packages installed at all, and OCR on our own onnxruntime code (no paddlepaddle, paddleocr or paddlex), the Streamlit app starts, becomes healthy and recognises an uploaded query image, and `mtg benchmark run` inside the image gives exact printing 254/257, card count 257/257, same card name 257/257, OCR fallbacks 0: every image's result is identical to a run of the same code outside Docker.
- **Size: 3.85 GB on disk, 0.98 GB compressed**, against 9.26 / 2.17 GB today (#204). Removing the system packages accounts for 0.43 GB of that against the same image with them.
- **Every system package in the Dockerfile can go**: `curl`, `git`, `libgl1`, `libglib2.0-0`, `libgomp1`, `libsm6`, `libxext6` and `libxrender1`.
- **Also measured**: copying the venv with `COPY --chown` instead of `chown -R /app` takes the image to **2.04 / 0.52 GB**, with the same benchmark results.

## What changed on this branch

- `pyproject.toml`: `opencv-contrib-python` → `opencv-python-headless==4.10.0.84`; paddlepaddle, paddleocr and the Paddle index removed; `pyclipper` and `pyyaml` added. In the runtime set (`uv export --no-dev`) the only OpenCV package is then `opencv-python-headless`, so `exclude-dependencies` isn't needed for the image. The dev group still brings `opencv-python` 4.13 through `rapidocr` (as #192 found), so a dev venv still has two OpenCV packages until rapidocr leaves the dev dependencies or `opencv-python` is excluded.
- `engine/ocr/onnx_ocr.py`: #199's `scripts/own_ocr.py`, unchanged. `engine/ocr/reading.py` loads it instead of PaddleOCR and calls `read_text` on each crop with `thresh` 0.05 (today's), `box_thresh` 0.6 and `unclip_ratio` 1.5 (PaddleX's OCR pipeline defaults), 4 threads. The debug view's annotated crop is `None`, since PaddleX drew it.
- Models, in the worktree only (not committed): `models/ocr/prod/PP-OCRv5_mobile_{det,rec}` replaced by the official ONNX exports (`inference.onnx` + `inference.yml`, 21 MB together, no `.pdiparams`), and `models/segmentation/prod` holding #195's dynamic `yoloe-11s-seg.onnx` and `silueta.onnx`, with the `.pt` files moved out.
- Three Dockerfiles in `scripts/`, all from today's `Dockerfile`, all without the PaddleX font download (nothing reads the font once paddlex is gone, so #204's Baidu download problem goes too):
  - `Dockerfile.no-paddle`: today's system packages kept.
  - `Dockerfile.headless`: no `apt-get` step at all.
  - `Dockerfile.headless-chown`: as `headless`, with `COPY --from=deps --chown=appuser:appuser /app/.venv /app/.venv`, and `chown` on `/app` itself (not recursive) and on the cache directory.

## Sizes

| Image | Disk | Compressed | Runs the benchmark |
|---|---|---|---|
| today (`dev`, #204) | 9.26 GB | 2.17 GB | yes |
| without torch, torchvision, clip, ultralytics (#204) | 6.45 GB | 1.56 GB | yes |
| also without paddle packages, OCR code unchanged (#204, size only) | 4.45 GB | 1.14 GB | no |
| **`no-paddle`**: own OCR, headless OpenCV, ONNX models, system packages kept | 4.28 GB | 1.09 GB | not run |
| **`headless`**: also without system packages | **3.85 GB** | **0.98 GB** | yes, identical |
| **`headless-chown`**: also `COPY --chown` | **2.04 GB** | **0.52 GB** | yes, identical |

The venv is 1,137 MB and 113 packages in all three new images. The `apt-get` layer is 323 MB (`docker history`), 79 Debian packages, mostly two groups:

- **GL** (`libgl1` → Mesa): `libllvm19` 124 MB, `mesa-libgallium` 41 MB, `libz3-4` 27 MB, X11 libraries; about 200 MB.
- **git** (with perl): `git` 48 MB, `libperl5.40` 29 MB, `perl-modules-5.40` 20 MB, `git-man` 2 MB; about 100 MB.

On the OpenCV package itself (installed size, 4.10.0.84, no dependencies): `opencv-contrib-python` 181 MB, `opencv-python` 168 MB, `opencv-python-headless` 136 MB. The 4.45 → 4.28 GB step against #204 mixes this 45 MB with the font and the `.pdiparams` files leaving, so it isn't the OpenCV saving alone.

Not measured: why `headless-chown` saves 1.81 GB when the duplicated `chown` layer is 1.34 GB in `docker history`. With the containerd image store, `docker images` disk usage isn't the sum of the history's layer sizes, so the per-layer numbers explain direction, not exact totals.

## Why each system package can go

Checked with `ldd` on every `.so` in the venv of the `headless` image, then by running the app and the benchmark.

- `libgl1`, `libsm6`, `libxext6`, `libxrender1`, `libglib2.0-0`: needed by `opencv-python` and `opencv-contrib-python` for their GUI backend. The headless wheel has no GUI backend (`GUI: NONE` in `cv2.getBuildInformation()`) and links none of them. Nothing else in the venv needs them.
- `libgomp1`: the only library left that wants it is numba's optional OpenMP thread pool (`numba/np/ufunc/omppool…so`; numba comes through rembg → pymatting). numba falls back to its own thread pool when that can't load. `libtbb.so.12`, for numba's TBB pool, is missing in today's image too. onnxruntime and OpenCV's wheels carry their own threading libraries. The no-card image, which runs rembg, passes in the container.
- `curl`: only downloaded the PaddleX font.
- `git`: only the benchmark uses it, for `git rev-parse HEAD` and `git status`. The image has no `.git`, so the benchmark fails there with or without git installed (#204). For these runs a two-line stand-in script was mounted at `/usr/local/bin/git` (`output/git-shim`, prints the commit). The implementation issue should let `_commit` work without git, for example from an environment variable set at build time, rather than install git.

The other `ldd` "not found" entries are hash-named libraries that wheels bundle and load through their own paths (`pillow.libs`, `opencv_python_headless.libs`, `numpy.libs`); the same list appears with all system packages installed, and the code runs.

## The app

The `headless` and `headless-chown` containers report healthy; a script (`/tmp/apptest/upload.py`, Playwright driving the system Chrome, not committed) uploads `1_photos_single_card/0731017a….jpg`, presses Start processing, and the page shows Spined Karok, STX #143, score 94.9, 1.3 s after the click, with no exceptions in the page or the container log.

## Verified and assumed

- Verified: the image sizes and layer sizes; the missing libraries; the app run; the benchmark in the `headless` and `headless-chown` images (per-image results identical to each other and to the run outside Docker); the OCR swap leaves the benchmark at #195/#204's segmentation result (254/257/257: the only change from the baseline is the 9-card scan with border gaps, which the ONNX segmentation fixes).
- Not measured: speed (Docker runs took 141 s and the same code outside Docker 157 s with default segmentation threads, but neither is a measurement under the map's rules); the no-card image takes 32–33 s **outside Docker too**, so #204's slow no-card image isn't Docker's VM; a dev venv with only one OpenCV package.
- Side finding, for the dependency map: `rembg[cli]` brings gradio (155 MB in the venv) and its dependencies, which nothing here uses.

## How to rerun

```bash
# put the ONNX models in models/ (see "What changed"), then
docker build -f scripts/Dockerfile.headless -t mtg-recognition:headless .
printf '#!/bin/sh\ncase "$1" in rev-parse) echo <commit> ;; esac\n' > output/git-shim; chmod 755 output/git-shim
docker run --rm --entrypoint /app/.venv/bin/mtg \
    -v "$PWD/output/git-shim:/usr/local/bin/git:ro" \
    -v "$PWD/data/benchmark:/app/data/benchmark:ro" \
    -v "$PWD/data/benchmark-baseline.json:/app/data/benchmark-baseline.json:ro" \
    -v "$PWD/output/docker-bench:/app/output" mtg-recognition:headless benchmark run
```

Outputs (gitignored): `output/local-bench.log`, `output/docker-bench.log`, `output/docker-bench-chown.log`. Images: `mtg-recognition:no-paddle`, `mtg-recognition:headless`, `mtg-recognition:headless-chown`.
