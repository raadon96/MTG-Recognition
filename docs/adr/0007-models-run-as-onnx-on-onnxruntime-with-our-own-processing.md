# Models run as ONNX on onnxruntime, with our own pre- and post-processing

The engine runs its models as ONNX files committed under `models/*/prod`, on onnxruntime, with pre- and post-processing written in this repo on NumPy and OpenCV. The libraries the models come from are needed only to write those files, not to recognise a query image. Their runtimes are large (torch alone is about 700 MB), they load slowly, and they can't run in the browser, where the in-browser port (ADR 0003, ADR 0004) runs the same ONNX files on ONNX Runtime Web. Our own processing is a short, readable copy of the steps the port must also implement.

## Segmentation

YOLOE-11s runs as `models/segmentation/prod/yoloe-11s-seg.onnx`, written by `scripts/export_segmentation.py`; its docstring gives the command. The script sets the prompts (`book cover`, `art work`), which fuses them into the weights, and writes a dynamic export (opset 20, `agnostic_nms`, `imgsz` 640). The text encoder (MobileCLIP, 600 MB) is needed only to set the prompts, so it no longer loads at runtime. A static (square) export pads query images differently from ultralytics and changes results, so the export is dynamic (#195). The script runs again when the prompts, the model or the ultralytics version change, and its output is committed.

The `export` dependency group holds ultralytics, torch, torchvision, clip, onnx and onnxslim, with the PyTorch CPU index and the clip git source. `uv sync --dev` doesn't install it, and the Docker image doesn't have it.

`ultralytics` is fixed to 8.4.21, the version our processing was copied from and the benchmark was measured with. A new version can change the exported model: 8.4.142 changed YOLOE-26's default head. Moving to a new version means running the export again and checking the benchmark.

`engine/segmentation/yolo.py` copies ultralytics 8.4.21's steps: letterbox to the next multiple of 32, `conf` 0.05, class-agnostic NMS at IoU 0.0, at most 50 detections, masks from the prototypes, contours (joined with ultralytics' `merge_multi_segment` when a mask has several) and scaling back to the query image. On all 257 benchmark query images, its card images are byte-identical to ultralytics' (#195, #204, #214), and the export gives the same outputs in ONNX Runtime Web (#204).

## OCR

OCR runs PP-OCRv5_mobile detection and PP-OCRv6 tiny recognition, as their official ONNX exports (`PaddlePaddle/PP-OCRv5_mobile_det_onnx` at revision `e6f4fa8` and `PaddlePaddle/PP-OCRv6_tiny_rec_onnx` at `2612ab3`, on Hugging Face), committed as `models/ocr/prod/*/inference.onnx` with the `inference.yml` beside each. Recognition's `inference.yml` holds the character dictionary, which belongs to that model. PP-OCRv6 tiny recognition replaced PP-OCRv5_mobile's (`ed152b8`): it is 4.5 MB against 16.5 MB, and moved the benchmark from 254 to 255 exact printings with no image worse (#218). PP-OCRv6 small is the more accurate choice, to revisit if telling printings apart becomes the limit (#207). paddlepaddle, paddleocr and paddlex are not dependencies, not even for development: the exports are downloaded as published, so nothing of Paddle's is needed to write them.

`engine/ocr/pp_ocr.py` copies the steps of PaddleX's OCR pipeline with the settings PaddleOCR ran with: detection resize (short side at least 64 px, both sides rounded to multiples of 32) with ImageNet normalisation; DB post-processing at `thresh` 0.05, `box_thresh` 0.6 and `unclip_ratio` 1.5; boxes sorted into reading order; rotated crops; recognition in batches of 6, each padded to its widest line, since the padding changes the output; CTC decoding. There is no document unwarping, no textline orientation, no `fp16` and no high-performance inference. On all 1,291 card images of #196 its tokens, boxes and readings are the same as PaddleOCR's (#199), and on all 276 card images of the benchmark query images the tokens and readings are the same, boxes the same to the pixel, and scores within 0.0003 (#215).

OpenCV stays at 4.10: under OpenCV 5 the same code reads 13 of 80 card images differently (#199).

## Licences

- Ultralytics' code and weights are AGPL-3.0. That covers the YOLOE weights, the export made from them, and `merge_multi_segment`, copied into `yolo.py`.
- The MobileCLIP text encoders are licensed for research use only. The export holds the prompts' embeddings, not the encoder.
- PP-OCRv5_mobile's and PP-OCRv6 tiny's weights and exports, and PaddleX's code our OCR processing copies, are Apache-2.0.
- Shipping the weights to browsers distributes them, so the app must then be AGPL-3.0, or run under an Ultralytics Enterprise licence.

## Considered Options

- **Keep ultralytics at runtime**: the same results, but torch, torchvision, clip and ultralytics stay in the image (1.4 GB of the 4.8 GB Docker image, #214), the model takes 1.2 s instead of 0.1 s to load and about 420 MB more memory (#195), and the browser port would still need its own processing.
- **ultralytics' own ONNX path**: it still needs ultralytics and torch installed, and doesn't let the thread count be set (#195).
- **A static (square) export**: changes results (#195).
- **YOLOE-26n**: a quarter of the size, but 3 benchmark images worse (#195); not adopted.
- **PaddleOCR on paddlepaddle**, as before: the same readings, but paddlepaddle alone is 676 MB, the models take 0.8 s instead of 0.3 s to load and about 550 MB more memory, and paddlex fixes which OpenCV package can be installed (#199).
- **PaddleOCR's onnxruntime engine**: the same readings without paddlepaddle, but it keeps paddleocr, paddlex and 17 more packages, and paddlex's OpenCV constraint (#199).
- **RapidOCR**: reads differently from PaddleOCR on most card images, even with PaddleX's settings, and brings its own OpenCV version (#199).

## Consequences

- torch, torchvision, clip and ultralytics are not runtime dependencies. The Docker image no longer keeps a cache volume for the MobileCLIP files.
- torch used to set OpenMP's thread count as a side effect; without it the benchmark ran 1.3x slower, so Settings set `OMP_NUM_THREADS` from `THREAD_COUNT`. Only numba, through rembg, still loads OpenMP, and it ignores `OMP_NUM_THREADS`, so Settings sets `NUMBA_NUM_THREADS` instead.
- A change to processing is checked against the benchmark, as any segmentation or OCR change (ADR 0004).
- paddlepaddle, paddleocr and paddlex are not dependencies, so nothing is installed from Paddle's package index, and the Docker image no longer downloads PaddleX's font. The app's debug view draws the OCR boxes on the collector information crop itself, in place of PaddleOCR's annotated image.
