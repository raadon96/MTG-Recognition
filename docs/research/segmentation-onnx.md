# Segmentation in ONNX Runtime Web, and the Docker image without torch, clip and ultralytics

Task for issue #204 (map #189). Run 2026-11-10 on the research branch `research/segmentation-onnx` (never merged, based on `research/ocr-onnxruntime`), on the AMD Ryzen AI 7 350, onnxruntime 1.24.3, onnxruntime-web 1.31.1 (WASM backend, Node 22.22.1, 1 thread), Docker Desktop. The model throughout is #195's dynamic YOLOE-11s export (`output/measure/onnx/yoloe-11s-seg-dynamic.onnx`: ultralytics 8.4.21, `set_classes(["book cover", "art work"])`, `export(format="onnx", dynamic=True, agnostic_nms=True, imgsz=640, opset=20)`), run with `conf` 0.05, `iou` 0.0, `max_det` 50.

## Answer

- **ONNX Runtime Web: yes, with the same results.** On all 248 benchmark query images that reach YOLO (the other 9 take the scan bypass), the export runs in onnxruntime-web and decodes to identical outlines and byte-identical card images, with the same card count on every image. Largest raw difference: 1.5e-2 on `output0` (box coordinates in input pixels, so a hundredth of a pixel), 2.3e-5 on the mask prototypes.
- **Docker image: 9.26 GB → 6.45 GB on disk (−2.81 GB), 2.17 → 1.56 GB compressed (−0.61 GB).** The image runs the benchmark with segmentation on onnxruntime and none of torch, torchvision, clip or ultralytics installed, and gets #195's result (exact printing 254/257, with the same 3 failing images).
- **With paddlepaddle, paddleocr and paddlex also gone** (the dependencies of #199's own OCR code; size only, since OCR code wasn't changed in this image): 4.45 GB on disk, 1.14 GB compressed, about half of today's.

## ONNX Runtime Web

`scripts/seg_web_check.py` reads each benchmark query image, skips it if `takes_scan_bypass` holds, and otherwise builds YOLO's input as `segment()` does: `enhance_contrast`, then the letterbox in `yoloe_onnx.yoloe_outlines`. It runs that input in Python onnxruntime (4 threads) and in onnxruntime-web (`scripts/seg_web_check.mjs`, one Node process holding the session). Then it decodes both outputs with the same post-processing (filter, NMS, masks, contours) and warps card images as `segment()` does.

| Measure | Result |
|---|---|
| query images | 257: 248 reach YOLO, 9 bypass |
| input shapes | 640×512 (166), 640×384 (37), 384×640 (37), 640×480 (8): the dynamic axes work in WASM |
| largest `output0` difference | 1.5e-2 (median image about 3e-3); boxes are in input pixels |
| largest prototype difference | 2.3e-5 |
| same number of outlines | 248 / 248 |
| identical outlines | 248 / 248 (largest point difference 0 px) |
| same card count / byte-identical card images | 248 / 248 |
| WASM time | median 756 ms per query image, 1 thread (not a speed measurement; WASM measurements are out of scope) |

Opset 20 needs no lower export: every operator in the graph is supported by the WASM backend. Pre- and post-processing in the browser would be a JavaScript port of `yoloe_onnx.py` (about 160 lines with `merge_multi_segment`), as #199 found for OCR.

## Docker image

All three images are built from the same Dockerfile, with one change applied to all three: the PaddleX font (`simfang.ttf`) is downloaded on the host and copied in, because the `curl` from `paddle-model-ecology.bj.bcebos.com` broke off mid-download on every try (exit 56). The Paddle package index was also returning 502, so `uv lock --offline` was used (from uv's cache).

| Image | Disk | Compressed | `/app/.venv` | Packages |
|---|---|---|---|---|
| today (`dev`, `aa54a1a`) | 9.26 GB | 2.17 GB | 2,976 MB | 152 |
| without torch, torchvision, clip, ultralytics | **6.45 GB (−2.81)** | **1.56 GB (−0.61)** | 1,940 MB (−1,036) | 136 |
| also without paddlepaddle, paddleocr, paddlex; with pyclipper, PyYAML (size only) | 4.45 GB (−4.81) | 1.14 GB (−1.03) | 1,182 MB (−1,794) | 113 |

What changed for the second image (commit on this branch):
- `pyproject.toml`: removed `torch`, `torchvision`, `clip`, `ultralytics`, and `ftfy` and `regex` (only CLIP uses them), with the PyTorch index and the CLIP git source. The lock also drops matplotlib, polars, ultralytics-thop and their dependencies.
- `engine/segmentation/yoloe_onnx.py`: `scripts/yoloe_onnx.py` from #195, with ultralytics' `merge_multi_segment` copied in (AGPL-3.0, as the model). `segment._yolo_model()` returns it; `conf` is 0.05; `YOLO_SEG_MODEL_PATH` points to `yoloe-11s-seg.onnx`.
- `models/segmentation/prod/`: the dynamic export instead of `yoloe-11s-seg.pt` and `yolov8s-worldv2.pt` (the export was copied in for the build, not committed: 40.8 MB).

**Why the image saves 2.7 times the venv's saving**: the venv is in the image twice. `COPY --from=deps /app/.venv` makes a 2.03 GB layer, and the later `RUN … chown -R appuser:appuser /app …` rewrites every file under `/app` into a new 2.2 GB layer (`docker history`). #199 put the duplication down to the second `uv sync`; it is the `chown`. Copying with `COPY --chown=appuser:appuser` (or changing ownership only of the directories that need writing) would remove about one venv's size again, in every image. Not measured.

**The image works**: `mtg benchmark run` inside the second image (benchmark images and baseline mounted read-only) recognises all 257 images. Exact printing passes on 254, failing on `1_photos_single_card/b845b1d7…`, `4_scans_single_card_no_bg/0a0ebcb7…` and `5_scans_multiple_cards/with_inner_gaps_without_border_gaps.jpg`: three of the baseline's four failures, and the 9-card scan with border gaps passes, as in #195. The command then stops with an error before writing its results file, because it reads the commit with `git rev-parse HEAD` and the image has no `.git`. So card count and same card name weren't recorded for this run (#195's 257/257 came from the same model, settings and post-processing).

## Verified and assumed

- Verified: the WASM results on every benchmark query image that reaches YOLO; the image sizes, venv sizes and package counts; that torch, torchvision, ultralytics and clip can't be imported in the second image; the exact-printing results in the container.
- Not measured: a real browser (Node's WASM backend is the same runtime, as in #199); WebGPU; speed in Docker (Docker Desktop's VM, default threads: median 0.7 s per query image, not comparable with #194's numbers); the third image running OCR.
- The no-card image took 32.5 s in the container: YOLO finds nothing, so the box fallback and then rembg run. Not explained. It isn't a download: `settings.py` sets `U2NET_HOME` to the models directory, and `models/segmentation/prod/silueta.onnx` matches the hash rembg expects, so rembg loads the local file. The time is more likely rembg loading and running in Docker Desktop's VM; not measured.

## How to rerun

```bash
# ONNX Runtime Web (npm install onnxruntime-web@1.31.1 in ORT_WEB_DIR)
ORT_WEB_DIR=/tmp/ortweb PYTHONPATH=scripts uv run python scripts/seg_web_check.py \
    output/measure/onnx/yoloe-11s-seg-dynamic.onnx
# Docker: copy the export to models/segmentation/prod/yoloe-11s-seg.onnx, move the .pt files out,
# put simfang.ttf in the build context, then build with the font-copying Dockerfile
docker build -f scripts/Dockerfile.font-copied -t mtg-recognition:seg-no-torch .
docker run --rm --entrypoint /app/.venv/bin/mtg \
    -v "$PWD/data/benchmark:/app/data/benchmark:ro" \
    -v "$PWD/data/benchmark-baseline.json:/app/data/benchmark-baseline.json:ro" \
    -v "$PWD/output/docker-bench:/app/output" mtg-recognition:seg-no-torch benchmark run
```

Outputs (gitignored): `output/seg_web_check/results.json`, `output/docker-bench.log`. Images: `mtg-recognition:seg-today`, `mtg-recognition:seg-no-torch`, `mtg-recognition:seg-no-torch-no-paddle`.
