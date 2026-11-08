# PP-OCRv6: API changes, tiers, claimed gains and inference options

Research for issue #190. Question: what does moving from PP-OCRv5 to PP-OCRv6 involve, and what do its claims rest on?

Researched 2026-11-08. Each fact below names its source. Package sources were read from wheels downloaded from PyPI (paddleocr 3.6.0 and 3.7.0, paddlex 3.6.1, 3.7.0, 3.7.1 and 3.7.2, rapidocr 3.8.4, 3.9.0 and 3.10.0) and from the installed project venv (paddleocr 3.4.0, paddlex 3.4.2). PaddleOCR documentation was read from the GitHub repository at commit `1af0448a200eae430b9addb40e7118d67f9840ab` (the latest commit touching `docs/version3.x/algorithm/PP-OCRv6` on that date); paths below marked "PaddleOCR repo" are relative to that commit. No model was run: everything about accuracy and speed on our card images is left to the experiments listed at the end.

## Summary

- PP-OCRv6 exists. It shipped in **paddleocr 3.7.0** (PyPI, 2026-06-11) with **paddlex 3.7.x**. paddleocr 3.6.0 and paddlex 3.6.1 contain no PP-OCRv6 code.
- Three tiers: **tiny, small, medium**. There is no `mobile` or `server` name in v6. Our `PP-OCRv5_mobile_*` corresponds to **small** by positioning and speed; **tiny** is the smaller, faster option.
- The API used by `load_paddleocr` and `read_card` is unchanged. Switching needs new model names, new weights in `models/ocr/prod`, new model directory paths in `settings.py`, and a dependency bump. The `predict` arguments and the result keys we read are the same.
- The +4.6 / +5.1 figures are percentage points, **PP-OCRv6_medium against PP-OCRv5_server**, on Baidu's unpublished in-house benchmark. The 5.2x CPU speedup is **PP-OCRv6_medium against PP-OCRv5_server on an Intel Xeon 8350C with the OpenVINO backend**, where PP-OCRv5_server is unusually slow. None of the three figures compares against `PP-OCRv5_mobile`, which is what we run.
- `precision="fp16"` has no effect on CPU, in 3.4.0 and in 3.7.0.
- `enable_hpi=True` needs paddlepaddle, `ultra-infer-python` and, for automatic ONNX conversion, `paddle2onnx`. On CPU it selects OpenVINO first for every PP-OCRv6 tier. Only paddlex **3.7.2** knows the PP-OCRv6 models in its HPI table; with 3.7.0 or 3.7.1 HPI cannot choose a backend for them.
- New in paddleocr 3.5+: `engine="onnxruntime"` runs ONNX models through onnxruntime inside PaddleOCR, and paddleocr 3.7.0 imports without paddlepaddle installed. Official ONNX exports of every PP-OCRv6 tier are on Hugging Face. This is directly relevant to #199.
- RapidOCR supports PP-OCRv6 from **3.9.0**; in **3.10.0** PP-OCRv6 small is its default detection and recognition model.

## 1. Is the PP-OCRv6 API a drop-in replacement?

### Verified

**Versions.**

| Package | Version needed | Source |
| --- | --- | --- |
| paddleocr | `>=3.7.0` (first release with PP-OCRv6) | paddleocr 3.7.0 `METADATA` lines 113-121 (release note "2026.06.11: Release of PaddleOCR 3.7.0 - PP-OCRv6 highlights"); `grep PP-OCRv6` finds nothing in paddleocr 3.6.0 or paddlex 3.6.1 |
| paddlex | `>=3.7.0,<3.8.0` (required by paddleocr 3.7.0); `3.7.2` if `enable_hpi=True` is used (see section 4) | paddleocr 3.7.0 `METADATA` line 27: `paddlex[ocr-core]<3.8.0,>=3.7.0` |
| paddlepaddle | `>=3.0.0` per the compatibility table; our 3.2.0 is inside that range | PaddleOCR repo `docs/version3.x/paddleocr_and_paddlex.en.md` lines 27-39 (row `3.7.x`) |

Latest releases on PyPI on 2026-11-08: paddleocr 3.7.0, paddlex 3.7.2, paddlepaddle 3.3.1, rapidocr 3.10.0. paddlex 3.7.2 keeps `numpy<2.4,>=1.24` and `opencv-contrib-python==4.10.0.84` (paddlex 3.7.2 `METADATA` lines 35, 102), the same OpenCV version as today.

**Constructor.** `PaddleOCR.__init__` in 3.7.0 (`paddleocr/_pipelines/ocr.py` lines 66-95) accepts the same keyword arguments we pass: `use_doc_unwarping`, `doc_unwarping_model_name`, `use_textline_orientation`, `text_detection_model_name/_dir`, `text_recognition_model_name/_dir`, `lang`. `device`, `enable_hpi` and `precision` are still accepted by `parse_common_args` (`paddleocr/_common_args.py` lines 38-74). Two arguments are new: `engine` and `engine_config` (lines 29-35, 41-42).

The diff of `_pipelines/ocr.py` from 3.4.0 to 3.7.0 only adds `"PP-OCRv6"` to `_SUPPORTED_OCR_VERSIONS` (line 58), changes the default models to `PP-OCRv6_medium_det`/`PP-OCRv6_medium_rec` when no model, `lang` or `ocr_version` is given (line 331), and routes `lang="en"` to PP-OCRv6 when no model names are given (lines 337-360). We pass model names, so `lang="en"` is ignored with a warning, as it is in 3.4.0 (line 125).

**`predict` and the result.** The `predict` signature of the paddlex OCR pipeline still takes `text_det_thresh` (paddlex 3.7.2 `inference/pipelines/ocr/pipeline.py` lines 292-305). `inference/pipelines/ocr/result.py` is byte-identical between paddlex 3.4.2 and 3.7.2, so `rec_texts`, `rec_boxes`, `rec_scores` and `.img["ocr_res_img"]` are unchanged. The only behaviour change in `pipeline.py` drops empty crops (zero width or height) before recognition (lines 415-428 in 3.7.2).

**Recognition input height.** Every PP-OCRv6 recognition model still takes 48 px high input: `RecResizeImg image_shape: [3, 48, 320]` in the `inference.yml` of `PP-OCRv6_tiny_rec` and `PP-OCRv6_small_rec` on Hugging Face, and the ONNX exports declare input `x: [N, 3, 48, W]` (read with `onnx.load`). Training used multi-scale heights 32/48/64 (PaddleOCR repo `configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml`, `MultiScaleSampler scales`). The comment and `TARGET_TEXT_HEIGHT = 48` in `reading.py` stay correct.

**Character dictionary.** All characters we expect in collector information (`•`, `★`, `©`, `™`, `&`, `/`, accented Latin letters) are in both the tiny (6,904 entries) and small (18,708 entries) dictionaries. `PP-OCRv5_mobile_rec` in `models/ocr/prod` (18,383 entries) has all of them except `™`. Checked by parsing `PostProcess.character_dict` in each `inference.yml`.

### Code changes required

1. `load_paddleocr`: `text_detection_model_name` and `text_recognition_model_name` set to the chosen tier, for example `"PP-OCRv6_small_det"` / `"PP-OCRv6_small_rec"`. The name must equal `Global.model_name` in the directory's `inference.yml`, or paddlex raises "Model name mismatch" (paddlex 3.7.2 `inference/models/__init__.py` line 290).
2. `settings.py`: `OCR_DETECTION_MODEL_DIR` and `OCR_RECOGNITION_MODEL_DIR` (lines 48-53) point at the new directories.
3. `models/ocr/prod`: new weights, from `https://huggingface.co/PaddlePaddle/PP-OCRv6_<tier>_<det|rec>` (`inference.json`, `inference.pdiparams`, `inference.yml`).
4. `pyproject.toml`: `paddleocr>=3.7.0`.
5. Optional: remove `precision="fp16"` (no effect on CPU, section 5) and decide on `enable_hpi` / `engine` (sections 4 and 7).

## 2. Tiers, names and sizes

### Verified

| Model | Params (whole det+rec pair) | Detection (MB) | Recognition (MB) | Pair (MB) |
| --- | --- | --- | --- | --- |
| PP-OCRv6_tiny | 1.5M | 1.9 | 4.4 | 6.3 |
| PP-OCRv6_small | 7.7M | 9.6 | 20.4 | 30.0 |
| PP-OCRv6_medium | 34.5M | 59.4 | 73.3 | 132.7 |
| PP-OCRv5_mobile (ours) | | 4.7 | 16 | 20.7 |
| PP-OCRv5_server | | 84.3 | 81 | 165.3 |

Sources: parameter counts from paddleocr 3.7.0 `METADATA` line 120 ("tiny (1.5M) / small (7.7M) / medium (34.5M)"); sizes from PaddleOCR repo `docs/version3.x/module_usage/text_detection.en.md` lines 27-70 and `text_recognition.en.md` lines 25-62. The `inference.pdiparams` sizes on Hugging Face agree: tiny det 1,718,055 B, tiny rec 4,420,410 B, small det 9,817,888 B, small rec 21,074,618 B; our `PP-OCRv5_mobile_det` and `_rec` files are 4,692,937 B and 16,458,665 B. The "about 6 MB" (tiny) and "about 30 MB" (small) figures in the issue are correct for a detection plus recognition pair.

Tier positioning: tiny for "edge/IoT", small for "mobile/desktop", medium for "server" (PaddleOCR repo `docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md` line 3; detection table descriptions in `text_detection.en.md` lines 33, 42, 51). The tech doc says "PP-OCRv6_small matches PP-OCRv5_mobile in latency on most platforms with higher accuracy" (`PP-OCRv6.en.md` line 122). Tiny recognition is distilled from medium, has no neck, and does not cover Japanese (lines 3, 63).

Names in each package:

- PaddleOCR/PaddleX: `PP-OCRv6_{tiny,small,medium}_{det,rec}`; ONNX exports `PP-OCRv6_*_onnx` (paddlex 3.7.2 `inference/utils/official_models.py` lines 515-560, 570-571).
- RapidOCR: `ocr_version="PP-OCRv6"`, `model_type` in `tiny/small/medium`, model keys `multi_PP-OCRv6_{det,rec}_{tiny,small,medium}` (rapidocr 3.10.0 `default_models.yaml` lines 111-147, `utils/typings.py` lines 49-50, 63-68).

### Assumption

Small is the closest replacement for `PP-OCRv5_mobile`: same intended use and same CPU latency in Baidu's table. Tiny is a candidate because it is smaller and faster than what we run today. Medium is roughly 6x the size of our pair and, on CPU with PaddlePaddle, about 2.6x slower than v5_mobile in Baidu's table (2.05 s against 0.80 s), so it is unlikely to fit the browser port, but its accuracy sets an upper reference for the comparison.

## 3. Where the claims come from

### Verified

All figures come from the PP-OCRv6 technical doc (PaddleOCR repo `docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md`, published at https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.html). No paper or public dataset is cited.

**+4.6% detection, +5.1% recognition.** Line 3: "On our in-house multi-scenario benchmark, PP-OCRv6_medium achieves +5.1% recognition accuracy and +4.6% detection Hmean over PP-OCRv5_server". The tables give the arithmetic, in percentage points:

- Detection Hmean average over 16 categories (lines 76-87): medium 86.2 against v5_server 81.6, a difference of 4.6.
- Recognition weighted-average accuracy over 15 categories (lines 91-102): medium 83.2 against v5_server 78.1, a difference of 5.1.

The benchmark is "in-house" and not published. The module docs add that PP-OCRv6 metrics come from "an internal multi-scenario evaluation set, while PP-OCRv5/v4 metrics are based on a general evaluation set ... not directly comparable" (`text_detection.en.md` line 92, `text_recognition.en.md` line 108). That warning applies to the module tables. In the tech-doc tables, v5 models were evaluated on the same in-house set (for example PP-OCRv5_server detection is 81.6 there and 83.8 in the module table).

The comparison that matters to us, PP-OCRv5_mobile against tiny and small, on the same tables:

| Column | v5_mobile | v6_tiny | v6_small | v6_medium |
| --- | --- | --- | --- | --- |
| Detection, average (16 categories) | 75.2 | 80.6 | 84.1 | 86.2 |
| Detection, Print-EN | 91.0 | 92.3 | 93.6 | 93.7 |
| Detection, HW-EN | 77.7 | 85.9 | 87.1 | 84.0 |
| Recognition, weighted average (15 categories) | 73.7 | 73.5 | 81.3 | 83.2 |
| Recognition, Print-EN | 86.0 | 88.4 | 93.3 | 94.1 |
| Recognition, Card | 81.7 | 80.5 | 86.9 | 88.1 |

What "Card" contains is not stated (it may be ID cards or business cards rather than playing cards).

**5.2x CPU speedup.** Section 3.3 (lines 104-123): end-to-end seconds per image, including image I/O, pre- and post-processing, on 200 images of "general + document scenes". Line 121: "5.2× on Intel Xeon OpenVINO (1.40s vs 7.30s)", i.e. PP-OCRv6_medium against PP-OCRv5_server, Intel Xeon 8350C, OpenVINO backend. In the same table PP-OCRv5_server takes 2.04 s on that CPU with PaddlePaddle, so OpenVINO is 3.6x slower than PaddlePaddle for v5_server; with PaddlePaddle on both sides medium and v5_server are equal (2.05 s against 2.04 s). The 5.2x comes from a slow baseline, not from a 5.2x faster model.

Intel Xeon 8350C rows (seconds per image) against what we run:

| Backend | v5_mobile | v6_tiny | v6_small | v6_medium |
| --- | --- | --- | --- | --- |
| PaddlePaddle | 0.80 | 0.32 (2.5x) | 0.79 (1.0x) | 2.05 |
| OpenVINO | 0.78 | 0.20 (3.9x) | 0.59 (1.3x) | 1.40 |
| ONNX Runtime | 0.61 | 0.22 (2.8x) | 0.61 (1.0x) | 3.31 |

The tech doc does not state thread count, batch size, image sizes or which detection resize settings were used.

The release note in paddleocr 3.7.0 `METADATA` line 97 repeats the three figures without saying which models they compare ("PP-OCRv6 achieves +4.6% detection and +5.1% recognition accuracy over PP-OCRv5 ... 5.2× CPU inference speedup end-to-end"); line 116 does say "Medium tier ... over PP-OCRv5_server".

### Assumptions on applicability to card text

- Our inputs are two small crops per card image (the name and the collector information), not document pages. Detection time scales with input size, so the per-image seconds above (on document and general scenes) do not predict our timing. The ratios between models are a better guide than the absolute values, but even those may change with small inputs, where fixed per-call overhead weighs more.
- The Print-EN columns are the closest match to card text and show smaller gains than the averages: tiny improves Print-EN recognition by 2.4 points and small by 7.3 points over v5_mobile. Collector information is small, low-contrast printed text with set codes and numbers, which neither column describes exactly.
- Tiny's recognition average (73.5) is equal to v5_mobile's (73.7); on "Card" it is lower (80.5 against 81.7). Tiny may not read better than what we have; it may only be faster.

## 4. `enable_hpi=True`

### Verified

**Packages.**

- paddlepaddle: "The high-performance inference functionality depends on the PaddlePaddle framework" (PaddleOCR repo `docs/version3.x/inference_deployment/local_inference/high_performance_inference.en.md` line 19).
- `ultra-infer-python` (the module is `ultra_infer`). Installed with `paddleocr install_hpi_deps cpu` (same doc, line 24) or `paddlex --install hpi-cpu`, which runs pip with `--find-links` on paddlex's bundled `hpip_links.html` (paddlex 3.7.2 `paddlex_cli.py` lines 255-322). That file lists `ultra_infer_python-1.2.0` wheels for cp38-cp312, `linux_x86_64` only, hosted on `paddle-model-ecology.bj.bcebos.com`, not PyPI. The doc says CPU support is "Linux systems, x86-64 architecture processors, and Python 3.8-3.12" (line 29). The project is on Python 3.12 (`.python-version`), so it is supported today but would block a move to 3.13.
- `paddle2onnx==2.0.2rc3`, only needed for automatic Paddle-to-ONNX conversion (paddlex 3.7.2 `METADATA` line 202; installed with `paddlex --install paddle2onnx`).

**Backend chosen on CPU.** `suggest_inference_backend_and_config` (paddlex 3.7.2 `inference/models/runners/hpi/backend.py` lines 44-223) builds the list of available backends from the model files present and what `ultra_infer` was built with (lines 55-80), then takes the first entry of the model's list in `hpi_model_info_collection.json` that is available (lines 125-176). For paddlepaddle 3.1 and later it uses the 3.1.1 entries (lines 88-94), and needs a release build (no `.dev` suffix, line 96). Lists for `cpu_x64` / `paddle311` in paddlex 3.7.2:

| Model | Order |
| --- | --- |
| PP-OCRv6_tiny_det, _rec; PP-OCRv6_small_det, _rec | openvino, onnxruntime, paddle_mkldnn, paddle |
| PP-OCRv6_medium_det, _rec | openvino, paddle_mkldnn, onnxruntime, paddle |
| PP-OCRv5_mobile_det | paddle_mkldnn, openvino, onnxruntime, paddle |
| PP-OCRv5_mobile_rec | openvino, onnxruntime, paddle_mkldnn, paddle |

So on CPU, HPI selects **OpenVINO** for every PP-OCRv6 tier, if an ONNX model is in the directory or the paddle2onnx plugin is installed (backend.py lines 55-66).

**paddlex 3.7.0 and 3.7.1 lack the PP-OCRv6 entries.** Their `hpi_model_info_collection.json` contains no `PP-OCRv6` key; the entries were added in 3.7.2 (diffed). Without them `suggest_inference_backend_and_config` returns "'PP-OCRv6_small_det' is not a known model" (backend.py lines 130-131) and `HPIRunner` raises `RuntimeError` (runner.py lines 112-118). HPI with PP-OCRv6 needs paddlex 3.7.2.

**Local model directories work, but conversion writes into them.** The HPI engine uses the `model_dir` it is given and the model name passed in (paddlex 3.7.2 `inference/models/engines/hpi.py` lines 72-95). If the directory has no `inference.onnx` and `auto_paddle2onnx` is true (the default, `runners/hpi/config.py` line 36), the runner calls `paddlex --paddle2onnx --paddle_model_dir <model_dir> --onnx_model_dir <model_dir>` in a subprocess (runner.py lines 195-224). With our layout that writes `inference.onnx` into `models/ocr/prod/<model>/`. Putting the official ONNX export (`PaddlePaddle/PP-OCRv6_<tier>_<det|rec>_onnx`, which also ships `inference.yml`) in the directory avoids the conversion and the paddle2onnx dependency.

**Thread count.** With `enable_hpi=True`, PaddleOCR passes `engine_config={"paddle_static": {...}}` (paddleocr 3.7.0 `_common_args.py` lines 113-124). The engine resolves to `hpi` (paddlex 3.7.2 `inference/models/__init__.py` lines 228-239), the bucket has no `hpi` entry, so paddlex logs a warning and uses an empty config (lines 176-215). `cpu_threads` is therefore not passed to OpenVINO or ONNX Runtime; they use `PADDLE_PDX_CPU_NUM_THREADS`, default 10 (`runners/hpi/config.py` lines 39-48).

**`engine` overrides `enable_hpi`.** "When `engine` is explicitly set, `enable_hpi` no longer takes effect" (PaddleOCR repo `docs/version3.x/inference_deployment/local_inference/inference_engine.en.md`, section 4.4; code: paddlex 3.7.2 `inference/models/__init__.py` lines 410-415).

## 5. Is `precision="fp16"` ignored on CPU?

### Verified: yes, in both 3.4.0 and 3.7.0

- paddleocr 3.4.0 `_common_args.py`: `precision` is renamed to `pptrt_precision` (line 55) and read only inside `if device_type == "gpu": if common_args["use_pptrt"]:` (lines 71-79). The CPU branch (lines 82-88) sets only `mkldnn_cache_capacity` or `run_mode`, and `cpu_threads`.
- paddleocr 3.7.0 `_common_args.py`: the same rename (line 72); `pptrt_precision` is read only in the GPU-and-TensorRT branch of `_build_paddle_static_engine_config` (lines 79-87). The CPU branch is lines 90-95.
- `pptrt_precision` is not read anywhere else in either paddleocr version (grep).
- Under HPI, fp16 is only applied for the TensorRT backend (paddlex 3.7.2 `runners/hpi/backend.py` lines 215-218; `runner.py` lines 261-262), which is GPU only.

So `precision="fp16"` in `load_paddleocr` does nothing on CPU. It is even ignored on GPU unless `use_tensorrt=True`. The CPU run mode is MKL-DNN (oneDNN) by default (`enable_mkldnn` defaults to `True` in paddleocr 3.7.0 `_constants.py` line 18; no PP-OCR model is in paddlex's `MKLDNN_BLOCKLIST`, `runners/paddle_static/config/blocklists.py` line 46).

## 6. RapidOCR and PP-OCRv6

### Verified

- rapidocr 3.9.0 (PyPI, 2026-06-23) is the first release with PP-OCRv6 in `default_models.yaml`; 3.8.4 has none.
- rapidocr 3.10.0 (2026-10-08) registers tiny, small and medium detection and recognition for the `onnxruntime`, `openvino` and `mnn` engines, downloaded from ModelScope `RapidAI/RapidOCR` at tag v3.10.0 with SHA256 checksums (`default_models.yaml` lines 111-147, 426-454, 573-601, 747-765).
- In 3.10.0 the default `Det` and `Rec` are `ocr_version: "PP-OCRv6"`, `model_type: "small"`, `engine_type: "onnxruntime"` (`config.yaml` lines 130-134, 169-173). Code using RapidOCR defaults gets a different model after upgrading.
- RapidOCR applies its own detection settings: `limit_side_len: 736`, `limit_type: min`, `thresh: 0.3`, `box_thresh: 0.5`, `unclip_ratio: 1.6`, `use_dilation: true`, and normalisation `mean = std = [0.5, 0.5, 0.5]` (`config.yaml` lines 141-151). PaddleX normalises PP-OCRv6 detection input with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` (`inference.yml` of `PP-OCRv6_tiny_det`, `NormalizeImage`). The same difference exists for PP-OCRv5 in RapidOCR. Recognition input shape is `[3, 48, 320]` in both (`config.yaml` line 181).
- rapidocr 3.10.0 also bundles a PyTorch implementation of the PP-OCRv6 recognition backbone (`inference_engine/pytorch/networks/backbones/rec_lcnetv4.py`), not needed for onnxruntime.

### Unknown

Whether RapidOCR's ONNX files are the same as the official `PaddlePaddle/PP-OCRv6_*_onnx` exports (the checksums were not compared), and whether its normalisation difference changes detection output on card images.

## 7. Notes for #199 (PP-OCR in onnxruntime without paddlepaddle)

### Verified

- paddleocr 3.5 added `engine` / `engine_config`; `engine="onnxruntime"` runs a model's `inference.onnx` through `onnxruntime.InferenceSession` (PaddleOCR repo `inference_engine.en.md` sections 1-4; paddlex 3.7.2 `inference/models/engines/onnxruntime.py`, `inference/models/runners/onnxruntime_runner.py`). It reads an existing ONNX file and does not convert (`onnxruntime_runner.py` lines 133-137: "No valid ONNX model found"). `engine_config` takes `intra_op_num_threads`, `inter_op_num_threads`, `graph_optimization_level`, `execution_mode`, `enable_mem_pattern`, `enable_cpu_mem_arena` and `session_options` (lines 32-48).
- For a pipeline, the engine can be set for all modules with `PaddleOCR(engine="onnxruntime", engine_config=...)`, or per module through an exported PaddleX config (`inference_engine.en.md` sections 4.3 and 5.6). When `engine_config` is given, `precision`, `enable_mkldnn`, `cpu_threads` and the other compatibility arguments are ignored (section 4.4).
- `import paddleocr` and `from paddleocr import PaddleOCR` succeed in a fresh Python 3.12 venv with only `paddleocr==3.7.0` and `onnxruntime==1.24.3` installed, with no `paddle` module present and none loaded afterwards. paddleocr's own dependencies do not include paddlepaddle (paddleocr 3.7.0 `METADATA` lines 27-31).
- Official ONNX exports exist on Hugging Face for every PP-OCRv6 tier and for PP-OCRv5_mobile: `PaddlePaddle/PP-OCRv6_{tiny,small,medium}_{det,rec}_onnx`, `PaddlePaddle/PP-OCRv5_mobile_{det,rec}_onnx`; each repo holds `inference.onnx` and `inference.yml`. Sizes: tiny det 1,780,590 B, tiny rec 4,462,639 B, small det 9,880,512 B, small rec 21,159,378 B.
- Those ONNX files (tiny and small, read with `onnx.load`): detection uses opset 14, recognition opset 11, all weights float32, and only standard ONNX operators (`Conv`, `ConvTranspose`, `Resize`, `MatMul`, `Softmax`, `Erf`, `HardSigmoid`, `BatchNormalization`, `ReduceMean`, and similar). Input `x` is `[N, 3, H, W]` with dynamic H and W for detection, and `[N, 3, 48, W]` for recognition. That answers the ADR 0003 requirement that a new model have a working ONNX export, as far as reading the file goes; running it in ONNX Runtime Web is still untested.

### Assumption

A whole `PaddleOCR(..., engine="onnxruntime")` pipeline may construct and run without paddlepaddle, given local directories holding the ONNX exports. Only the import was tested, not construction or inference.

## 8. OCR settings worth sweeping (#198)

### Verified facts the sweep depends on

- The PaddleOCR pipeline sets detection post-processing from the PaddleX `OCR.yaml`, not from the model's own `inference.yml`: `limit_side_len: 64`, `limit_type: min`, `thresh: 0.3`, `box_thresh: 0.6`, `unclip_ratio: 1.5` (paddlex 3.7.2 `configs/pipelines/OCR.yaml` lines 29-34; read in `inference/pipelines/ocr/pipeline.py` lines 103-110 and passed to the model at lines 126-135). The PP-OCRv6 `inference.yml` files ship different values: `thresh: 0.2`, `box_thresh: 0.4` (tiny) or `0.45` (small, medium), `unclip_ratio: 1.4`, `max_candidates: 3000`. PP-OCRv5_mobile_det ships `thresh: 0.3`, `box_thresh: 0.6`, `unclip_ratio: 1.5`. So through the pipeline, PP-OCRv6 detection runs with the PP-OCRv5 settings unless we pass others.
- All of these can be passed to `PaddleOCR(...)` or per call to `predict(...)`: `text_det_limit_side_len`, `text_det_limit_type`, `text_det_thresh`, `text_det_box_thresh`, `text_det_unclip_ratio`, `text_rec_score_thresh` (paddleocr 3.7.0 `_pipelines/ocr.py` lines 82-90; paddlex pipeline `predict` lines 292-305).
- `cpu_threads` defaults to 10 (paddleocr 3.7.0 `_constants.py` line 20); the benchmark machine has 16 threads. Under HPI the thread count comes from `PADDLE_PDX_CPU_NUM_THREADS` (default 10); under `engine="onnxruntime"` from `engine_config["intra_op_num_threads"]`.
- `enable_mkldnn` defaults to `True`; `mkldnn_cache_capacity` defaults to 10 (`_constants.py` lines 18-19).

### Suggested sweep, per candidate (assumption: chosen from the above)

| Setting | Today | Values to try | Why |
| --- | --- | --- | --- |
| `text_det_thresh` | 0.05 | 0.05, 0.1, 0.2, 0.3 | We set 0.05 for v5_mobile; PP-OCRv6 ships 0.2 |
| `text_det_box_thresh` | 0.6 (pipeline default) | 0.4, 0.45, 0.5, 0.6 | PP-OCRv6 ships 0.4/0.45 |
| `text_det_unclip_ratio` | 1.5 (pipeline default) | 1.4, 1.5, 1.8, 2.0 | PP-OCRv6 ships 1.4; affects whether set code and collector number split |
| `text_det_limit_side_len` / `text_det_limit_type` | 64 / min | 64/min, 736/min, 960/max | Our crops are small; the upscale size may matter more for detection than for recognition |
| Upscale of the collector information crop (`TARGET_TEXT_HEIGHT`, `MAX_UPSCALE`) | 48 px, up to 4x | 32, 48, 64 px | PP-OCRv6 recognition was trained at heights 32/48/64 |
| `cpu_threads` (or `intra_op_num_threads`) | 10 | 4, 8, 10, 16 | Speed only; 16 threads available |
| `enable_mkldnn` | True | True, False | Speed only, Paddle backend |
| Backend | Paddle (oneDNN) | Paddle, `engine="onnxruntime"`, HPI with OpenVINO | Speed, and whether output text is identical across backends |
| `text_rec_score_thresh` | 0.0 | 0.0, 0.3, 0.5 | Whether low-confidence tokens hurt scoring |

## Questions only an experiment can answer

For #196 (compare OCR models on raw text):

1. Do PP-OCRv6 tiny and small read the name and the collector information on our labelled card images better than PP-OCRv5_mobile? Compare set code, collector number, artist and name per image, not only overall scores.
2. Is tiny's reading as good as v5_mobile's on card text, given its equal average and lower "Card" score in Baidu's table?
3. How long do detection and recognition take per card image for each tier on the Ryzen AI 7 350, on our small crops?
4. Does paddlepaddle 3.2.0 load and run the PP-OCRv6 static graphs (`inference.json`), or is a newer paddlepaddle needed? The compatibility table says `>=3.0.0`, but this was not run.
5. With PP-OCRv6, are the PP-OCRv5-tuned pipeline detection settings worse than the model's own (`thresh 0.2`, `box_thresh 0.4/0.45`, `unclip_ratio 1.4`)?
6. Does the 48 px upscale of the collector information crop still give the best reading with PP-OCRv6?

For #198 (sweep OCR settings on the winning combination):

7. Every row of the sweep table above, scored by the benchmark.
8. Does the speedup Baidu reports for OpenVINO (HPI) on Intel Xeon hold on an AMD Ryzen CPU?
9. Does `cpu_threads=16` beat 10 on this machine, or does it compete with the rest of the engine?

For #199 (PP-OCR in onnxruntime without paddlepaddle):

10. Does `PaddleOCR(engine="onnxruntime", ...)` construct and run with local ONNX directories and no paddlepaddle installed?
11. Do the official ONNX exports give the same text, scores and boxes as the Paddle models for the same card images?
12. Are RapidOCR's PP-OCRv6 ONNX files the same as the official exports, and does RapidOCR's `[0.5, 0.5, 0.5]` detection normalisation change its output on card images compared with PaddleX's ImageNet normalisation?
13. Do the PP-OCRv6 ONNX exports run in ONNX Runtime Web (WASM) with the same output as onnxruntime on CPU (for the browser port, ADR 0003)?
14. Is onnxruntime on CPU faster or slower than Paddle with oneDNN for each tier on this machine?
