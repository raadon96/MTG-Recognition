# Recognise one image at a time: no YOLO batch inference

The engine segments and matches one image per call; there is no batch path. Batching was added to speed up multi-image uploads, but only the YOLO predict call was ever batched (bypass, fallbacks, pHash and OCR run per image), and on CPU it is slower: 3.1 s batched vs 2.3 s one-by-one for 20 phone photos, with segmentation only ~20 % of the total recognition time. Worse, it changes results: ultralytics letterboxes a batch of mixed-size images to a square instead of a minimal rectangle (`BasePredictor.pre_transform`, `same_shapes`), so 10 of those 20 photos segmented differently in a batch than alone. The app (batched) and the CLI and benchmark (one at a time) were therefore not measuring the same thing.

## Consequences

- `Pipeline.recognize(image)` is the only recognition entry point; clients loop over their images. Re-adding batching must keep that interface and first show, on the accuracy benchmark, a speed-up with identical top-1 results.
- The measurement was taken on 2026-10-18 on CPU with ultralytics 8.4.21 and `data/test/1_photos_single_card`. A GPU or a different runtime (e.g. the WASM port) may change the speed trade-off, but not the letterbox difference.
