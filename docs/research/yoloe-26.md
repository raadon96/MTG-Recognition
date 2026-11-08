# YOLOE-26: interface, NMS, prompts and ONNX export

Research for issue #191. Question: what does moving segmentation from `yoloe-11s-seg` to YOLOE-26 (`yoloe-26s-seg`, `yoloe-26n-seg`) involve?

Researched 2026-11-08. Package sources were read from the project venv (**ultralytics 8.4.21**, the installed version) and from the PyPI wheel of **ultralytics 8.4.207** (the latest release, 2026-11-07). Paths below are relative to the `ultralytics/` package directory, at the version named beside them. Wheels of 8.4.50 to 8.4.150 were also downloaded, to find the release where predict and export behaviour changed (section 2). The Ultralytics docs, the Ultralytics licence page, Hugging Face, the Apple MobileCLIP repository and the arXiv papers were read on the same date. The small checks that were run (loading the models, `set_classes`, one `predict` on one image, ONNX exports into a temporary directory) are listed in section 8. No benchmark and no timing was run: those belong to #195.

## Summary

- **`segment` works unchanged with both YOLOE-26 checkpoints** under ultralytics 8.4.21. `YOLO(path, task="segment")` returns a `YOLOE` model, `set_classes(["book cover", "art work"])` works, the `predict` arguments are accepted, and `result.masks.xy` is filled. Switching needs only a new `YOLO_SEG_MODEL_PATH` in `settings.py`, the weights in `models/segmentation/prod`, and the MobileCLIP2 text encoder file `mobileclip2_b.ts` (254 MB) instead of `mobileclip_blt.ts` (600 MB) wherever the model loads.
- **YOLOE-26 has two heads.** The one-to-one head needs no NMS: it takes the top `max_det` scores and stops, so `iou` does nothing. The one-to-many head needs NMS, like YOLOE-11. **ultralytics 8.4.21 uses the one-to-one head by default; ultralytics 8.4.142 and later use the one-to-many head with NMS by default**, and the one-to-one head only with `nms=False`. Our requirement `ultralytics>=8.4.1` means an upgrade silently changes which head runs. Both heads are in the local checkpoints.
- **`agnostic_nms` still matters on the one-to-one head** with two prompts: without it, one anchor can return both `book cover` and `art work`, which would give the same card twice. Keep `agnostic_nms=True`, and pass it at export time too.
- **Settings worth sweeping:** the head (one-to-one or one-to-many), the prompts, `conf` (the one-to-one head scores lower than the one-to-many head), `imgsz`, and for the one-to-many head `iou`. Section 3 lists values.
- **ONNX export bakes the prompts into the weights**: call `set_classes` with the production prompts, then `export(format="onnx", agnostic_nms=True, ...)`. Exporting needs torch, ultralytics and the text encoder; running the export needs none of them. The local `.onnx` files hold one prompt each (`book`, `game card`, `card`) and must be re-exported. A static export takes a square 640×640 input; `.pt` and dynamic ONNX take the smallest rectangle (for example 640×512 for a 3:4 photo). The padding differs, so results can differ.
- **No pretrained YOLO26 model for cards was found** on Hugging Face. The card models that exist are YOLO11 or YOLOv8 (OBB, keypoints, or segmentation of card elements), so a standard YOLO26 model stays out of scope.
- **Licences:** Ultralytics code and weights are AGPL-3.0, and Ultralytics says this covers trained models; shipping the ONNX weights in a browser port is distribution, which means either releasing the whole app's source under AGPL-3.0 or buying an Enterprise licence. This is the same for YOLOE-11 and YOLOE-26. Separately, the MobileCLIP text encoders are under Apple's research-only model licence. This is for the maintainer to decide; it is not legal advice.
- **ONNX without `ultralytics`** needs our own letterbox resize, a confidence filter (one-to-one head) or NMS (YOLOE-11 or one-to-many head), mask decoding from the 32 prototypes, contour extraction, and scaling back to the query image: about the work in section 7. At runtime nothing about prompts is needed.

## 1. Does `segment` work with YOLOE-26 unchanged?

### Verified

**Loading.** `YOLO(path, task="segment")` switches to the `YOLOE` class when the file stem contains `yoloe` and the suffix is `.pt` (8.4.21 `models/yolo/model.py` lines 65-73). In 8.4.207 the switch is by the loaded model's class, not the file name (`models/yolo/model.py` lines 72-95), so it keeps working whatever the file is called. Run on 8.4.21 (section 8): all three checkpoints load as `YOLOE` wrapping `YOLOESegModel`; the head is `YOLOESegment` for YOLOE-11 and `YOLOESegment26` for YOLOE-26.

**Architecture facts** (8.4.21 `cfg/models/26/yoloe-26-seg.yaml` lines 9-11): `end2end: True`, `reg_max: 1` (no DFL), `text_model: mobileclip2:b`. `yoloe-11-seg.yaml` has no `text_model` key, so YOLOE-11 uses the default `mobileclip:blt` (`nn/tasks.py` line 1017).

**`set_classes`.** `YOLOE.set_classes(classes, embeddings=None)` (8.4.21 `models/yolo/model.py` lines 306-323) computes text embeddings with the model's text encoder when the names differ from the stored ones, then stores them as `model.pe`. All three checkpoints ship with the placeholder names `'0'` to `'79'`, so the text encoder always runs at load. `build_text_model` (`nn/text_model.py` lines 338-360) maps `mobileclip:blt` to `mobileclip_blt.ts` and `mobileclip2:b` to `mobileclip2_b.ts`, loaded with `torch.jit.load(attempt_download_asset(...))` (line 292). `attempt_download_asset` looks in the current directory, then in the Ultralytics weights directory, then downloads (`utils/downloads.py` lines 450-468). The docs give the YOLOE-26 encoder as about 254 MB (https://docs.ultralytics.com/models/yoloe/, "Installation and Requirements"); the local files are 253,794,476 B (`mobileclip2_b.ts`) and 599,764,649 B (`mobileclip_blt.ts`). The Dockerfile copies neither (lines 86-87 are commented out), so the container downloads the encoder on first load today, and would do the same with YOLOE-26.

**Precomputed embeddings.** `set_classes(classes, embeddings)` skips the text encoder when embeddings are passed (8.4.21 `models/yolo/model.py` lines 316-319). 8.4.207 adds `save_prompt_embeddings(file)` and `load_prompt_embeddings(file)`, which store the embeddings and names in an `.npz` file tied to the model (8.4.207 `models/yolo/model.py` lines 379-430). Either way the text encoder (and the `clip` tokenizer) would not be needed at runtime.

**`predict` arguments.** Each of our arguments, on the YOLOE-26 path:

| Argument | Today | With YOLOE-26 (8.4.21) | Source |
| --- | --- | --- | --- |
| `conf=0.1` | filters scores before NMS | filters the top-k rows | `utils/nms.py` lines 66-67 |
| `iou=0.0` | NMS removes any box overlapping a kept box | ignored on the one-to-one head | `utils/nms.py` lines 66-70 (returns before NMS) |
| `agnostic_nms=True` | NMS across classes | one class per anchor in the head's top-k | `nn/modules/head.py` lines 224-247; set in `engine/predictor.py` lines 390-394 |
| `overlap_mask=False` | training only | training only | `cfg/default.yaml` line 43; only read in `utils/loss.py` and `data/dataset.py` |
| `max_det=50` | cap after NMS | k of the head's top-k, then cap | `engine/predictor.py` line 394, `utils/nms.py` line 67 |
| `verbose=False` | | | |

**`result.masks.xy`.** Built the same way for both models: `SegmentationPredictor.construct_result` (`models/yolo/segment/predict.py` lines 84-110) calls `ops.process_mask` and wraps the masks in `Results`; `Masks.xy` runs `ops.masks2segments` then `ops.scale_coords` (`engine/results.py` lines 1084-1106). The check in section 8 returned a filled `masks.xy` from YOLOE-26n on the example image.

**Head settings are fixed at the first `predict` in 8.4.21.** `Model.predict` creates the predictor and calls `setup_model` only on the first call; later calls only update `predictor.args` (8.4.21 `engine/model.py` lines 527-531). `setup_model` is where `max_det` and `agnostic_nms` are written into the one-to-one head (`engine/predictor.py` lines 390-394). So with 8.4.21, changing `max_det` or `agnostic_nms` between calls on the same loaded YOLOE-26 model has no effect on the head. 8.4.207 writes them before every call (`engine/predictor.py` lines 374-376). Production passes constant values, so this only matters for sweeps.

### Code changes required

1. `settings.py` line 45: `YOLO_SEG_MODEL_PATH` points at `yoloe-26s-seg.pt` or `yoloe-26n-seg.pt`.
2. `models/segmentation/prod`: the new weights (30,759,531 B for 26s, 11,463,019 B for 26n, against 27,803,986 B for 11s).
3. `mobileclip2_b.ts` available where the model loads, or embeddings precomputed and passed to `set_classes` (removes the text encoder from runtime).
4. If ultralytics is upgraded to 8.4.142 or later: decide the head explicitly (`nms=False` for one-to-one), see section 2.
5. `_YOLO_PREDICT_KWARGS`: `iou` has no effect on the one-to-one head and can go if that head is chosen. The other values are for the sweep (section 3).

### Assumption

The `_YOLO_CLASSES` prompts were tuned for YOLOE-11 and its MobileCLIP text encoder. YOLOE-26 uses MobileCLIP2, so the same words give different embeddings and may not be the best prompts. In the single check in section 8, YOLOE-26s with the one-to-one head found no card on one example image where YOLOE-11s and YOLOE-26n each found one. One image says nothing about accuracy, but it is a reason to sweep prompts and `conf` rather than reuse them.

## 2. Does YOLOE-26 need NMS?

### Verified

**Two heads.** `Detect.__init__` builds the one-to-many branches (`cv2`, `cv3`) and, when `end2end` is true, copies them as one-to-one branches (`one2one_cv2`, `one2one_cv3`; 8.4.21 `nn/modules/head.py` lines 111-113). `YOLOESegment26` does the same for the contrastive and mask-coefficient branches (`one2one_cv4`, `one2one_cv5`). In `forward` (lines 146-160), the one-to-one branch is decoded and passed to `postprocess`, which keeps the top `max_det` anchors by score (`get_topk_index`, lines 224-247) and returns rows of `[x1, y1, x2, y2, score, class, 32 mask coefficients]`. That is the ONNX shape `[1, 300, 38]`: 300 is the default `max_det`, 38 is 4 + 1 + 1 + 32. No IoU step exists in this path. The YOLOE-11 ONNX output `[1, 37, 8400]` is 4 box values + 1 class score (one prompt) + 32 coefficients for each of 8,400 anchors, which then need NMS.

**How the predictor treats it.** `non_max_suppression` returns early for end-to-end output: it keeps rows with score above `conf`, cuts to `max_det`, and returns (8.4.21 `utils/nms.py` lines 66-70). `iou` is never read on this path.

**`agnostic_nms` changes the head's top-k.** With `agnostic_nms=True`, each anchor contributes only its best class (`head.py` lines 238-242). With `False`, the top-k runs over every (anchor, class) pair (lines 243-247), so one anchor can appear once per prompt. With our two prompts that could return the same card twice, once as `book cover` and once as `art work`. The YOLOE docs say the same: "agnostic mode only keeps the single best class per anchor instead of letting one anchor emit several class labels" (https://docs.ultralytics.com/models/yoloe/, "Deployment Notes"). 8.4.207 makes `agnostic_nms=True` the default for `YOLOE.predict` (`models/yolo/model.py` line 566); 8.4.21 does not, but we pass it.

**Both heads are in the checkpoints.** Loading `yoloe-26s-seg.pt` and `yoloe-26n-seg.pt` on 8.4.21 shows `head.cv2` and `head.cv3` (one-to-many) present, and `predict(..., end2end=False)` runs and returns masks (section 8). `end2end` is a predict argument in 8.4.21 (`cfg/default.yaml` line 59; `engine/predictor.py` lines 390-392).

**The default head changed in ultralytics 8.4.142** (PyPI, 2026-09-05). Found by comparing `cfg/default.yaml` and `engine/predictor.py` across the 8.4.100 to 8.4.150 wheels:

- 8.4.141 and earlier: `end2end` argument, default unset, so the model's own flag applies: YOLOE-26 runs the one-to-one head. `nms: False` only means "do not embed NMS in an export".
- 8.4.142 and later: `end2end` is gone (mapped onto `nms` with a deprecation warning, 8.4.207 `cfg/__init__.py` lines 596-613). `nms` is `None` by default, documented as "None: external NMS (default); True: embed NMS on export; False: NMS-free head if available" (8.4.207 `cfg/default.yaml` line 61). The predictor passes `end2end=self.args.nms is False` to `AutoBackend` (8.4.207 `engine/predictor.py` line 529), and the PyTorch backend sets `model.end2end` to that value before fusing (8.4.207 `nn/backends/pytorch.py` lines 56-58). So by default YOLOE-26 `.pt` runs the one-to-many head plus NMS; `nms=False` selects the one-to-one head. Export follows the same rule (8.4.207 `engine/exporter.py` lines 766-769): a default export has the one-to-many head and YOLOE-11-style output.
- The YOLO26 docs match: one-to-many "Generates traditional YOLO outputs requiring NMS post-processing" and is the default; "Use `nms=False` for NMS-free inference" (https://docs.ultralytics.com/models/yolo26/). Their CPU speed claim ("up to 43% faster CPU ONNX inference for YOLO26n compared with YOLO11n on an Intel Xeon CPU @ 2.00 GHz") is measured with `nms=False`.
- `pyproject.toml` requires `ultralytics>=8.4.1`, so re-locking would cross this change.

**What the one-to-one head costs, per Ultralytics.** YOLO26 COCO detection: 40.9 mAP50-95 with NMS against 40.1 end-to-end for YOLO26n, 48.6 against 47.8 for YOLO26s (https://docs.ultralytics.com/models/yolo26/). The YOLOE docs say "with nms=False the YOLOE-26 one-to-one head scores lower still, with its best threshold near 0.01" (said of visual prompts in dense scenes; "Deployment Notes"). Paper: "Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models", https://arxiv.org/abs/2606.03748.

### Answer

With the one-to-one head, YOLOE-26 needs no NMS and `iou` does nothing, as the ticket suggests. `agnostic_nms` still matters (see above). With the one-to-many head, which is the default from 8.4.142, it needs NMS exactly like YOLOE-11 and `iou` applies again.

### Assumption

Our `iou=0.0` (drop any box that overlaps a kept box at all) was a fix for YOLOE-11 returning both a card and its art box. The one-to-one head is trained to return one prediction per object, so it may not need that suppression, but whether it also returns the art box as a second, separate object on our images is for the experiment.

## 3. Settings worth sweeping

### Verified facts the sweep depends on

- `imgsz` is the longest side after resizing: `LetterBox` scales by `min(new_h / h, new_w / w)` (8.4.21 `data/augment.py` line 1602). A 4080×3072 photo becomes 640×482 at `imgsz=640`. Sizes are rounded up to a multiple of the stride, 32 (`engine/predictor.py` line 255, `check_imgsz`).
- For a static ONNX export the predictor replaces `imgsz` with the export's own (`engine/predictor.py` lines 407-408), so `imgsz` can only be swept on `.pt` or a dynamic export.
- `conf` defaults to 0.25 for `predict` (`engine/model.py` line 523); we pass 0.1.
- `retina_masks=True` builds masks at the query image's resolution instead of the model input's (`models/yolo/segment/predict.py` lines 101-106, `ops.process_mask_native`). We use the default `False`, so outlines come from a mask the size of the letterboxed input, upsampled.
- `max_det=50` is far above the most cards in a query image; it only bounds the work.

### Suggested sweep, per candidate (assumption: values chosen from the above)

| Setting | Today | Values to try | Why |
| --- | --- | --- | --- |
| Head (YOLOE-26 only) | one-to-one (8.4.21 default) | one-to-one; one-to-many with NMS | Default changes with the ultralytics version; the one-to-one head is NMS-free but scores lower |
| Prompts | `book cover`, `art work` | each alone; `card`, `game card`, `trading card`, `playing card`, `magic card`; today's pair plus one of these | Tuned for YOLOE-11's text encoder; YOLOE-26 uses MobileCLIP2. The docs advise everyday category names describing appearance |
| `conf` | 0.1 | 0.01, 0.03, 0.05, 0.1, 0.15, 0.25 | The one-to-one head scores lower; the docs mention thresholds near 0.01 for it |
| `imgsz` | 640 | 480, 640, 800, 960 | Cards take much of a photo, so smaller may be enough; larger may help multi-card images. Static ONNX fixes it at export |
| `iou` (one-to-many and YOLOE-11 only) | 0.0 | 0.0, 0.3, 0.5, 0.7 | Today's value removes any overlap; 0.7 is the Ultralytics default |
| `agnostic_nms` | True | True only, with two or more prompts | `False` can return the same card once per prompt |
| `retina_masks` | False | False, True | Outline precision at the card's corners; costs time on large photos |
| `enhance_contrast` before YOLO | on | on, off | Its CLAHE and gamma values were tuned with YOLOE-11 |

Following the map's sweep rule, tune on half the images and confirm on the other half, and prefer values in a stable range.

## 4. ONNX export with text prompts

### Verified

**Prompts are fused into the weights at export.** For each `Detect` head, the exporter sets `dynamic`, `export`, `max_det` (clamped to the anchor count), `agnostic_nms`, and, for YOLOE models, calls `m.fuse(model.pe)` (8.4.21 `engine/exporter.py` lines 517-531; 8.4.207 lines 990-998). `YOLOEDetect.fuse` folds the text embeddings into the last convolution of each classification branch and replaces the text-prompt module with an identity (8.4.21 `nn/modules/head.py` lines 1033-1088). The ONNX model then has one input, `images`, and the classes set before export. The YOLOE paper describes this re-parameterisation (RepRTA) as giving text prompts no inference cost ("YOLOE: Real-Time Seeing Anything", https://arxiv.org/abs/2503.07465). The YOLOE docs: set the classes before exporting; "exported models keep the prompted classes and cannot accept new prompts" ("Export Usage").

So the export must be:

```python
model = YOLO("yoloe-26s-seg.pt", task="segment")
model.set_classes(["book cover", "art work"])  # or set_classes(names, embeddings)
model.export(format="onnx", agnostic_nms=True, imgsz=640, dynamic=False)
```

`agnostic_nms` and `max_det` are read at export and fixed in the graph; `agnostic_nms` defaults to `False` (8.4.21 `cfg/default.yaml` line 67), so leaving it out gives the duplicate-per-prompt behaviour described in section 2. With 8.4.142 or later, add `nms=False` to export the one-to-one head; otherwise the export has the one-to-many head.

**The local exports.** Read with `onnx.load` (section 8):

| File | Input | Outputs | Names | end2end | Exported with |
| --- | --- | --- | --- | --- | --- |
| `yoloe-11s-seg.onnx` | `[1, 3, 640, 640]` | `[1, 37, 8400]`, `[1, 32, 160, 160]` | `('book',)` | False | 8.4.21, opset 22, `dynamic=False` |
| `yoloe-26s-seg.onnx` | `[1, 3, 640, 640]` | `[1, 300, 38]`, `[1, 32, 160, 160]` | `{0: 'game card'}` | True | 8.4.21, opset 22, `dynamic=False` |
| `yoloe-26n-seg.onnx` | `[1, 3, 640, 640]` | `[1, 300, 38]`, `[1, 32, 160, 160]` | `('card',)` | True | 8.4.21, opset 22, `dynamic=False` |

Each holds one prompt, not the production pair, so all three must be re-exported for #195. Operators: YOLOE-11 uses only `Add, Concat, Conv, ConvTranspose, Div, MatMul, MaxPool, Mul, Reshape, Resize, Sigmoid, Slice, Softmax, Split, Sub, Transpose`; YOLOE-26 adds `Cast, Flatten, Gather, GatherElements, Mod, ReduceMax, TopK, Tile, Unsqueeze` for the top-k step. All are standard ONNX operators.

**Re-exporting works.** Exporting `yoloe-26s-seg.pt` with the production prompts on 8.4.21 gave `names {0: 'book cover', 1: 'art work'}`, output `[1, 300, 38]` (the shape does not grow with the number of prompts, because the class is one column), and it ran in onnxruntime 1.24.3 (section 8). The export logged two warnings: torch's TorchScript exporter supports opset 20 at most while Ultralytics asked for 22 (`best_onnx_opset`, 8.4.21 `engine/exporter.py` lines 186-211, picks the torch maximum minus one), and an `aten::index` warning that negative indices give wrong results. Passing `opset=20` or lower avoids the first.

**What `dynamic` changes.** With `dynamic=True` the input becomes `[batch, 3, height, width]` and the mask prototypes `[batch, 32, height/4, width/4]` (8.4.21 `engine/exporter.py` lines 702-711). Checked: the dynamic export ran on 640×640, 640×480 and 1024×768 inputs, giving prototypes of 160×160, 160×120 and 256×192, and `output0` stayed `(1, 300, 38)`. The exporter labels the last axis of `output0` "anchors", a name meant for the `[1, 37, 8400]` layout; on YOLOE-26 that axis is the fixed 38. The dynamic file was 40.5 MB against 39.9 MB static.

**What `imgsz` changes.** For a static export, `imgsz` fixes the input size (default: the checkpoint's `imgsz`, 640; 8.4.21 `engine/model.py` lines 704-710). For a dynamic export, it only sets the size used to trace the graph; any multiple of 32 runs.

**How `.pt` and ONNX resize differently.** The predictor letterboxes every image (8.4.21 `engine/predictor.py` lines 184-199; `data/augment.py` lines 1512-1640): it scales the longest side to `imgsz` with `cv2.INTER_LINEAR`, then pads with grey (114) centred. `predict` sets `rect=True` (`engine/model.py` line 523), and `auto` (pad only up to the next multiple of 32) applies when the model is `.pt` or a dynamic export. So:

| Model | Input for a 4080×3072 portrait photo at `imgsz=640` |
| --- | --- |
| `.pt` | 640×512 (482 px of image, 15 px of padding each side) |
| dynamic ONNX, through ultralytics | 640×512, as `.pt` |
| static ONNX | 640×640 (79 px of padding each side) |

The scale is the same in all three; only the padding differs. Checked on the example image: masks came back 640×512 from `.pt` and 640×640 from the static export. A convolutional network's outputs near the image edge depend on the padding, so a static export can give slightly different boxes, scores and masks from `.pt`, and a card near the `conf` threshold can appear in one and not the other.

Other differences between `.pt` and ONNX:

- With 8.4.21, loading an ONNX model calls `check_requirements(("onnx", "onnxruntime"))` (`nn/autobackend.py` line 262), which installs `onnx` into the environment if it is missing unless `YOLO_AUTOINSTALL=false` is set. With the variable set it only warns, and inference works.
- 8.4.21 creates the onnxruntime session with no session options (`nn/autobackend.py` line 280), so the thread count cannot be set through ultralytics; it is onnxruntime's default. 8.4.207 sets torch to at most 2 threads when the model is ONNX or OpenVINO on CPU (`engine/predictor.py` lines 533-534).
- Mask cropping changed between versions: 8.4.21 crops the low-resolution mask to the box and then upsamples it (`utils/ops.py` lines 480-504); 8.4.207 upsamples first and crops at image resolution, "cropping first smears the bilinear edge outside the bbox (#24272)" (`utils/ops.py` lines 622-650). Outlines can differ by a pixel or two between versions, for `.pt` and ONNX alike.

### Assumption

The ONNX export will run in ONNX Runtime Web with the WASM backend: all its operators are standard and the WASM backend runs onnxruntime's CPU kernels. Not tested; opset 22 support in the ONNX Runtime Web version the port will use should be checked, or the export made with a lower opset.

## 5. A pretrained standard YOLO26 segmentation model for cards

### Verified

Hugging Face model search on 2026-11-08 (API `huggingface.co/api/models?search=...`, terms `yolo26 card`, `yolo26-seg card`, `yolo26 seg`, `mtg yolo`, `mtg card`, `card segmentation`, `card detector`, `tcg`, `pokemon card`, `playing card yolo26`, `magic the gathering segmentation`) found no YOLO26 model that segments trading cards. The card-related Ultralytics models found:

| Model | Architecture | What it does | Licence |
| --- | --- | --- | --- |
| `matteot11/collector-mtg-pkm-detector-yolo11n-obb` | YOLO11n OBB | Oriented boxes around MTG and Pokemon cards; trained on synthetic scenes | AGPL-3.0 |
| `duclvQ/tcg-card-detector` | YOLOv8n pose | Four card corners as keypoints | AGPL-3.0 |
| `JakeTurner616/mtg-card-segmentation-yolo11` | YOLO11 seg | Elements inside a card scan (title, art, type line, ...), not the card | MIT tag |
| `chayuto/thai-id-ocr-yolo26n-card-*` | YOLO26n detect | Thai ID cards | |

### Answer

None is a suitable pretrained YOLO26 segmentation model for cards, so a standard YOLO26 model stays out of scope as the map says. The YOLO11 OBB and YOLOv8 keypoint models are different approaches (oriented boxes or corners rather than masks) and are noted only for completeness.

## 6. Licences for a browser port

### Verified

- The ultralytics package and the YOLOE reference repository are AGPL-3.0 (`ultralytics-8.4.21.dist-info/METADATA` `License: AGPL-3.0`; GitHub licence of `ultralytics/ultralytics` and `THU-MIG/yoloe`). The local ONNX files carry `license: AGPL-3.0 License (https://ultralytics.com/license)` in their metadata.
- Ultralytics' licence page (https://www.ultralytics.com/license): to use YOLO you must either "Open-source your entire project under AGPL-3.0" or obtain an Enterprise License; compliance "means publicly releasing the complete corresponding source code for the entire derivative work", "including the larger application, modifications, scripts, configuration files, and, where applicable, model weights", and this applies even when you "Deploy through a SaaS platform, API, or other private system". "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default." The Enterprise License lets organisations "embed Ultralytics YOLO code and models in commercial products". The page says nothing specific about exported formats (ONNX) or browser distribution.
- This repository is public on GitHub and declares no licence (`gh repo view`: `licenseInfo: null`; no `LICENSE` file).
- The text encoders are Apple's: `apple/MobileCLIP2-B` and `apple/MobileCLIP-B-LT` on Hugging Face are tagged `apple-amlr`, and the `apple/ml-mobileclip` README lists "ML models: Apple ML Research Model TOU". Its `LICENSE_MODELS` grants use, copying, modification and distribution "exclusively for Research Purposes", which "does not include any commercial exploitation, product development or use in any commercial product or service", and the same limit applies to "Model Derivatives".

### What it means for a browser port (assumption: a reading for the maintainer, not legal advice)

- Shipping the ONNX weights to the user's browser distributes them. Under Ultralytics' reading, that requires either releasing the whole port's source (and the server's, which Ultralytics already counts through the SaaS clause) under AGPL-3.0, or an Enterprise licence. Writing our own pre- and post-processing instead of using `ultralytics` does not change this, because Ultralytics counts the weights themselves.
- This is the same for YOLOE-11 and YOLOE-26; moving to YOLOE-26 adds no new obligation from Ultralytics.
- The MobileCLIP text encoder runs only when prompts are turned into embeddings. With embeddings precomputed (section 1), it is used once at export, not shipped. Whether embeddings fused into the exported weights count as a "Model Derivative" under Apple's research-only terms is an open question. It applies to today's YOLOE-11 setup too, which runs `mobileclip_blt.ts` at every model load.
- If the project is to stay closed or commercial, both points need a decision before the port: an Ultralytics Enterprise licence, and an answer on the Apple terms (or a non-YOLOE model).

## 7. Running ONNX without `ultralytics`

### Verified: what the ultralytics path does, step by step (8.4.21)

1. **Resize** (`data/augment.py` lines 1571-1640, `LetterBox.__call__`): gain `r = min(H/h, W/w)`; resize to `(round(w·r), round(h·r))` with `cv2.INTER_LINEAR`; pad to `H×W` (static) or to the next multiple of 32 (dynamic, `auto=True`), half on each side (`round(d - 0.1)` before, `round(d + 0.1)` after), value 114.
2. **Tensor** (`engine/predictor.py` lines 151-173): BGR to RGB, HWC to CHW, float32, divide by 255. No mean or standard deviation.
3. **Run**: one input `images`; outputs `output0` and `output1` (prototypes `[1, 32, H/4, W/4]`).
4. **Detections**:
   - YOLOE-26 one-to-one: `output0[0]` rows are `[x1, y1, x2, y2, score, class, c0..c31]`, boxes already corners (`decode_bboxes` uses `xywh=False` when end-to-end, `nn/modules/head.py` lines 199-206) and scores already sigmoid. Keep rows with `score > conf` (`utils/nms.py` line 67).
   - YOLOE-11 or YOLOE-26 one-to-many: `output0[0]` is `[4 + nc + 32, anchors]`, boxes as centre and size, then class scores, then coefficients. Needs the score filter, conversion to corners and NMS (`utils/nms.py` lines 72-170); `cv2.dnn.NMSBoxes` can do the NMS.
5. **Masks** (`utils/ops.py` lines 480-504, `process_mask`): `coefficients @ prototypes.reshape(32, -1)` reshaped to the prototype size; zero everything outside the box scaled to prototype size (`crop_mask`, line 453); bilinear upsample to the input size; threshold at 0 (the logit, so no sigmoid needed).
6. **Outlines** (`utils/ops.py` lines 615-643, `masks2segments`): `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`; with several contours, the default `strategy="all"` joins them into one polygon.
7. **Back to the query image** (`utils/ops.py` lines 562-594, `scale_coords`): subtract the padding, divide by the gain, clip to the image. These are the points `card_corners_from_outline` receives.

**Prompts.** Nothing at runtime: the classes are fused into the weights at export (section 4). Export needs torch, ultralytics, the `clip` tokenizer and `mobileclip2_b.ts` (or precomputed embeddings, which avoid the tokenizer and encoder). Runtime needs only `onnxruntime`, `numpy` and `cv2`, which the project already has.

### Assumptions

- Our own code for steps 1-7 with the one-to-one head is about 60-80 lines of numpy and OpenCV; YOLOE-11 or the one-to-many head adds NMS. This is the code the map's "Removing `torch` and `clip`" item would add.
- Following the steps above should reproduce ultralytics' outlines to within a pixel; whether it does, on our images, is for the experiment. It also decides which ultralytics version the code should copy (mask cropping changed, section 4).
- With onnxruntime used directly, the thread count can be set (`SessionOptions.intra_op_num_threads`), which the map's 4-thread measuring rule needs and the ultralytics ONNX path does not allow.

## 8. What was run

All in throwaway scripts under `/tmp`, with outputs only under `/tmp`. Nothing in the repository, `models/`, `pyproject.toml` or the venv changed (`git status` clean, no `onnx` package added to the venv).

1. Loaded `yoloe-11s-seg.pt` (prod), `yoloe-26s-seg.pt` and `yoloe-26n-seg.pt` (test) with `YOLO(path, task="segment")` on ultralytics 8.4.21, called `set_classes(["book cover", "art work"])`, and ran one `predict` with `_YOLO_PREDICT_KWARGS` on `data/examples/0b382265-73df-5d46-b2b1-b802d97075f4.jpg` (4080×3072, without `enhance_contrast`). All three loaded and returned `Results`; masks were 640×512. Detections: 11s one (score 0.263), 26s none, 26n one (0.385).
2. Same image, YOLOE-26 with `end2end=True` and `end2end=False` (`conf=0.1, iou=0.7, agnostic_nms=True`): both checkpoints keep the one-to-many head. 26s: none with one-to-one, one (0.192) with one-to-many. 26n: one (0.385) and one (0.427).
3. Read the three local ONNX files with `onnx` (`uv run --no-project --with onnx`): table in section 4.
4. Exported `yoloe-26s-seg.pt` with the production prompts to ONNX, static and dynamic (`imgsz=640`, other arguments default, so `agnostic_nms=False`), into a temporary directory, through `uv run --with onnx --with onnxslim` and `YOLO_AUTOINSTALL=false`. Ran both in onnxruntime 1.24.3 on zero inputs of several sizes, and through `YOLO(onnx).predict` on the image above (`conf=0.1`): static found one card (640×640 masks), dynamic none.

Checks 1, 2 and 4 are one image each and say nothing about accuracy. They show that the head and the resize can each change whether a weak detection passes `conf`.

## Questions only an experiment can answer

For #195 (compare segmentation candidates):

1. With today's settings, how do `yoloe-26s-seg` and `yoloe-26n-seg` compare with `yoloe-11s-seg` on exact printing, card count and card name, and on the 4 multi-card images individually?
2. One-to-one head or one-to-many head with NMS: which is more accurate on our images, and is the one-to-one head measurably faster on CPU, as the YOLO26 docs claim?
3. Which prompts work best with MobileCLIP2? Is a single prompt enough, or do two prompts help multi-card images?
4. Which `conf` works with each head, given that the one-to-one head scores lower? Is there a stable range?
5. Does `imgsz` other than 640 help or hurt, especially on multi-card images and small cards?
6. Does the one-to-one head return the art box as a second object, which `iou=0.0` used to remove on YOLOE-11?
7. Is `enhance_contrast` still useful with YOLOE-26?
8. Does `retina_masks=True` give corners that improve exact printing, and at what cost?
9. Static versus dynamic ONNX: how often does the square padding of a static export change the result compared with `.pt`?
10. Is ONNX faster than `.pt` per query image, with the same thread count? (Run onnxruntime directly to set 4 threads; the ultralytics ONNX path in 8.4.21 cannot.)
11. How long does model load take with `mobileclip2_b.ts` against `mobileclip_blt.ts`, and with precomputed embeddings and no text encoder?
12. Is YOLOE-26n accurate enough for its smaller size to count under the adoption rule (the `.pt` is 59% smaller than 11s, the ONNX 73% smaller)?
13. How often does segmentation fall back to the box fallback or rembg with each candidate?
14. Do results change between ultralytics 8.4.21 and the latest release, given the default head and the mask cropping both changed? Setting the head explicitly removes the first difference.

For the map's "Removing `torch` and `clip`" item (no ticket yet; #195 asks to record what it would involve):

15. Does our own resize, decode and contour code reproduce ultralytics' outlines and benchmark results exactly?
16. Does the export run in ONNX Runtime Web (WASM) with the same output as onnxruntime on CPU, and at which opset?

For the final decision (map, "Not yet specified"):

17. The licence questions in section 6 (Ultralytics AGPL-3.0 or Enterprise for the browser port; Apple's research-only terms for MobileCLIP) are for the maintainer, not an experiment.
