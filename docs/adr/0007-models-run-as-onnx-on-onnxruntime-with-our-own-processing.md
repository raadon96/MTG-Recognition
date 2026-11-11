# Models run as ONNX on onnxruntime, with our own pre- and post-processing

The engine runs its models as ONNX files committed under `models/*/prod`, on onnxruntime, with pre- and post-processing written in this repo on NumPy and OpenCV. The libraries the models come from are needed only to write those files, not to recognise a query image. Their runtimes are large (torch alone is about 700 MB), they load slowly, and they can't run in the browser, where the in-browser port (ADR 0003, ADR 0004) runs the same ONNX files on ONNX Runtime Web. Our own processing is a short, readable copy of the steps the port must also implement.

## Segmentation

YOLOE-11s runs as `models/segmentation/prod/yoloe-11s-seg.onnx`, written by `scripts/export_segmentation.py`:

```bash
uv run --group export python scripts/export_segmentation.py
```

The script sets the prompts (`book cover`, `art work`), which fuses them into the weights, and writes a dynamic export (opset 20, `agnostic_nms`, `imgsz` 640). The text encoder (MobileCLIP, 600 MB) is needed only to set the prompts, so it no longer loads at runtime. A static (square) export pads query images differently from ultralytics and changes results, so the export is dynamic (#195). The script runs again when the prompts, the model or the ultralytics version change, and its output is committed.

The `export` dependency group holds ultralytics, torch, torchvision, clip, onnx and onnxslim, with the PyTorch CPU index and the clip git source. `uv sync --dev` doesn't install it, and the Docker image doesn't have it.

`ultralytics` is fixed to 8.4.21, the version our processing was copied from and the benchmark was measured with. A new version can change the exported model: 8.4.142 changed YOLOE-26's default head. Moving to a new version means running the export again and checking the benchmark.

`engine/segmentation/yolo.py` copies ultralytics 8.4.21's steps: letterbox to the next multiple of 32, `conf` 0.05, class-agnostic NMS at IoU 0.0, at most 50 detections, masks from the prototypes, contours (joined with ultralytics' `merge_multi_segment` when a mask has several) and scaling back to the query image. On all 257 benchmark query images, its card images are byte-identical to ultralytics' (#195, #204, #214), and the export gives the same outputs in ONNX Runtime Web (#204).

## Licences

- Ultralytics' code and weights are AGPL-3.0. That covers the YOLOE weights, the export made from them, and `merge_multi_segment`, copied into `yolo.py`.
- The MobileCLIP text encoders are licensed for research use only. The export holds the prompts' embeddings, not the encoder.
- Shipping the weights to browsers distributes them, so the app must then be AGPL-3.0, or run under an Ultralytics Enterprise licence.

## Consequences

- torch, torchvision, clip and ultralytics are not runtime dependencies. The Docker image no longer keeps a cache volume for the MobileCLIP files.
- torch used to set OpenMP's thread count as a side effect. Settings now sets `OMP_NUM_THREADS` from `THREAD_COUNT`; without it the benchmark ran 1.3x slower.
- A change to processing is checked against the benchmark, as any segmentation change (ADR 0004).
