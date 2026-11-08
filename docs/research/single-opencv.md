# One OpenCV package in the environment

Research for issue #192 (map #189, background in #110). Question: how can the environment end up with exactly one package that provides `cv2`, for each combination of libraries we might choose?

Researched 2026-11-08 with uv 0.13.1 on Linux x86_64. Package requirements were read from PyPI's JSON API (`https://pypi.org/pypi/<package>/<version>/json`, field `requires_dist`). Package source was read from environments built with `uv sync --dev --locked` in a throwaway worktree. Each uv mechanism was tried by editing `pyproject.toml`, running `uv lock`, deleting `.venv` and running `uv sync --dev --locked`, then listing the OpenCV distributions (`uv pip list | grep -i opencv`) and importing `cv2`. None of these `pyproject.toml` or `uv.lock` changes is committed; the snippets below are what was tried.

## Summary

- **Who asks for which package.** `ultralytics` asks for `opencv-python` in every 8.4.x release. `paddlex[ocr-core]` (pulled in by every `paddleocr` 3.x) asks for exactly `opencv-contrib-python==4.10.0.84`, in 3.4.x and in 3.7.x. `rapidocr` asks for `opencv-python` up to 3.9.2 and for `opencv-python-headless` from **3.10.0**.
- **paddlex needs the distribution named `opencv-contrib-python`.** It decides whether OCR can run by looking up that distribution name in the installed package metadata, not by importing `cv2`. With `opencv-python` or `opencv-contrib-python-headless` as the only OpenCV package, `PaddleOCR(...)` fails with `DependencyError`. So while paddlex is installed, the package that remains has to be `opencv-contrib-python`, and at the version paddlex sets (4.10.0.84).
- **Mechanism: `exclude-dependencies` in `[tool.uv]`.** It removes the other OpenCV packages from the lockfile, records the exclusion in `uv.lock`, holds under `uv sync --dev --locked` and under `uv run`, and `--locked` fails if the lockfile was not regenerated. The other options are worse: an override with a never-matching marker leaves an unused `opencv-python` entry in the lockfile; `--no-install-package` is a command-line flag that the next `uv run` undoes; constraints cannot remove a package.
- **Existing environments break once.** Uninstalling `opencv-python` from an environment where both packages wrote into `cv2/` deletes shared files such as `cv2/__init__.py`, and `import cv2` then fails. Afterwards `uv sync --locked` reports nothing to do. One `uv sync --dev --locked --reinstall-package opencv-contrib-python` repairs it. Fresh environments (CI, Docker) are not affected.
- **Smoke check with today's versions plus `exclude-dependencies = ["opencv-python"]`:** only `opencv-contrib-python` 4.10.0.84 is installed; `pytest -m "not slow"` 313 passed, `pytest -m slow` 67 passed and 1 xfailed, `ty check` passes, `lint-imports` passes, and `mtg benchmark run` reports "Same as the baseline".
- **No library we use calls a contrib-only module**, and none of the GUI functions (`imshow`, `namedWindow`, `waitKey`) runs on the code paths we use. Contrib is only needed for paddlex's distribution-name check.
- **`rapidocr` can be removed from the dev dependencies.** Nothing in the repository imports or mentions it except `pyproject.toml` line 47.

## 1. What each library declares

### Verified (PyPI `requires_dist`)

| Package | Versions | OpenCV requirement |
| --- | --- | --- |
| ultralytics | 8.4.0 to 8.4.88 (8.4.21 is locked today) | `opencv-python>=4.6.0` |
| ultralytics | 8.4.89 to 8.4.115 | `opencv-python!=4.13.0.90,>=4.6.0` |
| ultralytics | 8.4.116 to 8.4.208 (latest, 2026-11-08; includes 8.4.142) | `opencv-python!=4.13.0.90,>=4.7.0` |
| paddleocr | 3.4.0 (locked), 3.4.1, 3.7.0 | none directly; requires `paddlex[ocr-core]` (`<3.5.0,>=3.4.0` in 3.4.0, `<3.8.0,>=3.7.0` in 3.7.0) |
| paddlex | 3.4.0 to 3.4.3 (3.4.2 locked), 3.7.0 to 3.7.2 | `opencv-contrib-python==4.10.0.84` in each of the extras `cv`, `multimodal`, `ie`, `trans`, `ocr-core`, `ocr`, `video`, `base`, `serving`; nothing without an extra |
| rapidocr | 3.7.0 (locked), 3.9.0 to 3.9.2 | `opencv_python>=4.5.1.48` |
| rapidocr | 3.10.0 (latest) | `opencv-python-headless>=4.5.1.48` |

Other relevant requirements:

- `paddlex[ocr-core]` 3.7.2 also brings `imagesize`, `pyclipper`, `pypdfium2>=4`, `python-bidi`, `shapely`. paddlex 3.7.x requires `numpy<2.4,>=1.24` (3.4.x: `numpy>=1.24`). Resolving paddleocr 3.7.0 moved numpy from 2.4.2 to 2.3.5.
- No other package in `uv.lock` depends on an OpenCV package: the lockfile lists `opencv-*` dependencies only under `mtg-recognition`, `paddlex`, `rapidocr` and `ultralytics`. `cv2-enumerate-cameras` (dev) declares no OpenCV requirement (its `METADATA` has only a macOS `pyobjc` requirement).
- Latest OpenCV releases on PyPI: `opencv-python` and `opencv-contrib-python` **5.0.0.93** (2026-07-02), `opencv-contrib-python` 4.14.0.94 (2026-07-29). Today's `opencv-python` 4.13.0.92 is from 2026-02-05. ultralytics sets no upper bound, so when `opencv-python` is resolved freely uv now picks 5.0.0.93 (seen in sections 2 and 5).
- opencv-python's own PyPI description (4.10.0.84): "There are four different packages ... and you should SELECT ONLY ONE OF THEM. Do not install multiple different packages in the same environment. There is no plugin architecture: all the packages use the same namespace (`cv2`)."

## 2. uv mechanisms

### Verified

Sources: uv settings reference https://docs.astral.sh/uv/reference/settings/ (`override-dependencies`, `constraint-dependencies`, `exclude-dependencies`, all read only from the workspace root `pyproject.toml`), resolution concepts https://docs.astral.sh/uv/concepts/resolution/ ("Dependency exclusions remove packages from the dependency graph"; constraints "only take effect if a requested package is already pulled in"; "If the same dependency is both overridden and excluded in a matching scope, the exclusion takes precedence"), and `uv sync --help`.

**`exclude-dependencies` (recommended).**

```toml
[tool.uv]
package = true
exclude-dependencies = ["opencv-python"]
```

- `uv lock` prints `Removed opencv-python v4.13.0.92`. The `uv.lock` diff adds `[manifest] excludes = ["opencv-python"]`, deletes the `opencv-python` package entry, and drops `{ name = "opencv-python" }` from the dependencies of `rapidocr` and `ultralytics`. Nothing else changes.
- Fresh `uv sync --dev --locked`: one OpenCV distribution, `opencv-contrib-python 4.10.0.84`; `import cv2` gives `4.10.0` from `.venv/lib/python3.12/site-packages/cv2/__init__.py`. Same with `--no-dev`.
- With the exclusion in `pyproject.toml` but the old `uv.lock`, `uv sync --dev --locked` fails: "The lockfile at `uv.lock` needs to be updated, but `--locked` was provided." So CI (`uv sync --locked --dev` in `.github/workflows/checks.yml` and `engine.yml`) and the Dockerfile (`uv sync --locked --no-dev`) cannot quietly install a different set.
- A plain name excludes the package wherever it is requested. A scoped form (`{ package = { name = "ultralytics" }, dependencies = ["opencv-python"] }`) exists but is not needed here.

**Override with a never-matching marker.**

```toml
override-dependencies = ["opencv-python; sys_platform == 'never'"]
```

Also leaves only `opencv-contrib-python` installed, but `uv.lock` keeps an `opencv-python` package entry (resolved to **5.0.0.93**, a version nobody uses) and every dependent's entry becomes `{ name = "opencv-python", marker = "sys_platform == 'never'" }`. It works, but the lockfile then names a package that is never installed. Before `exclude-dependencies` existed this was the usual workaround.

**`--no-install-package opencv-python`.** A flag of `uv sync` ("Do not install the given package(s)"); there is no matching `pyproject.toml` setting. `uv sync --dev --locked --no-install-package opencv-python` installs only contrib, but the next `uv run python -c "import cv2"` re-syncs, installs `opencv-python 4.13.0.92`, and `cv2` then reports `4.13.0`. Every `uv sync`, `uv run`, CI step and Docker step would need the flag. Not usable.

**`constraint-dependencies`.** Only narrows the versions of packages that are already requested; it cannot remove one. It is still useful next to an exclusion to set the version of the package that remains, for example `constraint-dependencies = ["opencv-python-headless<5"]`.

**Existing environments.** In an environment built from today's lock (both packages installed, `cv2` from contrib), `uv sync --dev --locked` with the new lock printed `- opencv-python==4.13.0.92`, after which `cv2/__init__.py` and the compiled module were gone and `import cv2; cv2.__version__` raised `AttributeError: module 'cv2' has no attribute '__version__'`. A second `uv sync --dev --locked` printed "Checked 261 packages" and repaired nothing. `uv sync --dev --locked --reinstall-package opencv-contrib-python` restored `cv2` 4.10.0. The same happened when moving to the section 5 configuration. CI uses `setup-uv` with `enable-cache: true`, which caches uv's download cache, not `.venv`, and the Docker build starts from an empty `.venv`, so only developer machines need the repair command.

## 3. Which package should remain

### Verified

**paddlex checks the distribution name.** `paddlex/utils/deps.py` (3.4.2 and 3.7.2): `is_dep_available(dep)` calls `importlib.metadata.version(dep)` for every dependency except paddlepaddle, paddle-custom-device, ultra-infer, fastdeploy and onnxruntime (which it checks with `find_spec`). The OCR pipeline is decorated `@pipeline_requires_extra("ocr", alt="ocr-core")` (`paddlex/inference/pipelines/ocr/pipeline.py` line 485), and `is_extra_available` requires every dependency of the extra to be available by distribution name. 43 paddlex files gate classes or functions on `"opencv-contrib-python"`, including `common/reader/image_reader.py`, `models/text_detection/processors.py` and `models/text_recognition/processors.py`.

Tried with paddleocr 3.7.0 / paddlex 3.7.2:

| Only OpenCV package installed | `is_extra_available("ocr-core")` | `load_paddleocr()` |
| --- | --- | --- |
| `opencv-contrib-python` 4.10.0.84 | True | loads |
| `opencv-contrib-python-headless` 4.10.0.84 | False | `RuntimeError: A dependency error occurred during pipeline creation`, caused by `DependencyError: OCR requires additional dependencies ...` |
| `opencv-python` 5.0.0.93 | False | same `DependencyError` |

**No contrib-only module is used.** `grep -rnE` for `cv2.` followed by `ximgproc`, `xfeatures2d`, `aruco`, `text`, `face`, `bgsegm`, `dnn_superres`, `saliency`, `optflow`, `legacy`, `wechat_qrcode`, `img_hash`, `xphoto`, `freetype`, `ccm`, `mcc` (and the names `createGuidedFilter`, `thinning`, `niBlackThreshold`, `SURF_create`, `TrackerCSRT`, `TrackerKCF`) finds nothing in paddlex 3.4.2 or 3.7.2, paddleocr 3.4.0 or 3.7.0, ultralytics 8.4.21 or 8.4.208, rapidocr 3.7.0 or 3.10.0, rembg, imagehash, or our `src` and `tests`.

**GUI calls.** ultralytics calls `cv2.imshow`, `namedWindow`, `waitKey` and `destroyAllWindows` only when `show=True` (`engine/predictor.py` lines 510-511 and the `show` method from line 642), in `solutions/`, and in `utils/checks.py` `check_imshow`. `_YOLO_PREDICT_KWARGS` in `engine/segmentation/segment.py` does not set `show`. paddlex, paddleocr and rapidocr call none of them. Our `src` calls none.

**ultralytics and the `opencv-python` name.** The only runtime check for the name is `check_requirements("opencv-python>=4.5.4")` in `nn/autobackend.py` line 256, on the `dnn=True` (ONNX through OpenCV DNN) path, which we do not use. Had it run with `opencv-python` excluded, ultralytics would try to install `opencv-python`.

**Headless works for segmentation.** With `opencv-python-headless` 4.14.0.94 as the only OpenCV package and ultralytics 8.4.208, `mtg segment -m yolo` on `data/benchmark/1_photos_single_card` wrote 201 card images.

**Dockerfile.** The base stage installs `libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1`, commented as "OpenCV/ML runtime libs". opencv-python's PyPI description says the headless packages are "for server (headless) environments (such as Docker ...), no GUI library dependencies" and avoid "a heavy dependency chain to X11 libraries".

### Choice per paddlex

- **paddlex installed:** keep `opencv-contrib-python==4.10.0.84`. Nothing else passes paddlex's check, and paddlex sets the exact version, so the version cannot move until paddlex changes its requirement.
- **paddlex not installed:** keep `opencv-python-headless`. No library needs contrib or a GUI, rapidocr 3.10 already asks for headless, and it is the smallest package.

## 4. Removing `rapidocr` from the dev dependencies

### Verified

`grep -rIni rapidocr` over the repository (excluding `.venv`, `.git` and `uv.lock`) finds only `pyproject.toml` line 47. Nothing in `src`, `tests`, `docs`, `.github/workflows`, `.pre-commit-config.yaml`, `Dockerfile` or `CONTRIBUTING.md` imports, runs or mentions it. It is one of the two packages that request `opencv-python` today; removing it leaves ultralytics, so an exclusion is still needed while ultralytics is installed.

## 5. Each combination of libraries

### Verified (resolved, synced fresh with `--locked`, `cv2` imported)

| Runtime libraries | Remaining package | Exclude | Result |
| --- | --- | --- | --- |
| A. Today: ultralytics 8.4.21, paddleocr 3.4.0 / paddlex 3.4.2, paddlepaddle, rapidocr 3.7.0 (dev) | `opencv-contrib-python==4.10.0.84` (our requirement, already in `pyproject.toml`) | `["opencv-python"]` | one package, `cv2` 4.10.0; all tests, ty and benchmark pass (section 6) |
| B. ultralytics `>=8.4.142` (8.4.208), paddleocr `>=3.7.0` (3.7.0 / paddlex 3.7.2), rapidocr `>=3.10.0` (dev) | `opencv-contrib-python==4.10.0.84` | `["opencv-python", "opencv-python-headless"]` | one package, `cv2` 4.10.0; `load_paddleocr()` loads; `import rapidocr, ultralytics` works |
| C. No paddlepaddle / paddleocr; OCR through rapidocr 3.10 in onnxruntime; ultralytics 8.4.208 at runtime | `opencv-python-headless` (add `"opencv-python-headless>=4.13,<5"` as our requirement; resolved 4.14.0.94) | `["opencv-python"]` | one package, `cv2` 4.14.0; YOLO segmentation and `from rapidocr import RapidOCR` work |
| D. No paddle, ultralytics moved to the dev group (export tool only) | `opencv-python-headless` | `["opencv-python"]` | `--no-dev`: only headless. Without the exclusion the dev environment had `opencv-python` **5.0.0.93** and `opencv-python-headless` 4.14.0.94, with `cv2` 5.0.0; with it, only headless 4.14.0.94 |
| E. paddle kept, ultralytics removed or dev-only | `opencv-contrib-python==4.10.0.84` | `["opencv-python"]` while ultralytics or rapidocr is in any group; none otherwise | not run; inferred from A, B and section 1: paddlex alone asks only for contrib |

Snippet for B:

```toml
dependencies = [
    # ...
    "ultralytics>=8.4.142",
    "opencv-contrib-python==4.10.0.84",
    "paddleocr>=3.7.0",
    # ...
]

[tool.uv]
package = true
exclude-dependencies = ["opencv-python", "opencv-python-headless"]
```

Snippet for C and D:

```toml
dependencies = [
    # ...
    "rapidocr>=3.10.0",
    "opencv-python-headless>=4.13,<5",
    # ...
]

[tool.uv]
package = true
exclude-dependencies = ["opencv-python"]
```

Notes:

- `exclude-dependencies` applies to every dependency group, so the dev environment is covered too (D).
- In B, `opencv-python-headless` is excluded only because rapidocr 3.10 asks for it; if rapidocr is removed (section 4), `["opencv-python"]` is enough.
- In C and D, `ty check` reported 4 errors: one `unresolved-import` for `paddleocr` (expected, the OCR code was not changed) and three in `engine/segmentation/segment.py` lines 99-100, where ultralytics 8.4.208 types `predict()` as returning `Iterator[Results | Tensor]`. These come from the ultralytics upgrade, not from OpenCV.

## 6. Smoke check (combination A)

### Verified

Today's `pyproject.toml` plus `exclude-dependencies = ["opencv-python"]`, fresh `.venv`, `uv sync --dev --locked`:

- `uv pip list | grep -i opencv`: `opencv-contrib-python 4.10.0.84` only.
- `uv run --locked pytest -m "not slow" -q`: 313 passed, 68 deselected.
- `uv run --locked pytest -m slow -q -x`: 67 passed, 1 xfailed. The DVC images came from the main checkout's DVC cache (`dvc checkout` with the cache directory set locally); `dvc pull` was not run because the Google Drive remote needs authentication.
- `uv run --locked ty check`: "All checks passed!"
- `uv run --locked lint-imports`: 3 contracts kept.
- `uv run --locked mtg benchmark run`: exact printing 253/257, "Totals unchanged", "Same as the baseline", exit 0.

Fresh environments built from today's lock already import `cv2` 4.10.0 from contrib (#110), so A changes no runtime behaviour on a fresh build; it removes the dependence on install order.

## Assumptions

- With paddlex installed, any upgrade path keeps `opencv-contrib-python==4.10.0.84`, because paddlex 3.4.x and 3.7.x set it exactly. A later paddlex may change this.
- The headless packages would let the Dockerfile drop `libgl1`, `libsm6`, `libxext6` and `libxrender1`; `libgomp1` is still used by onnxruntime and paddlepaddle. Not checked in a container.
- ultralytics 8.4.x works on OpenCV 4.10 (combinations A and B), because it accepts `>=4.6.0` / `>=4.7.0` and the segmentation tests and benchmark pass on 8.4.21. Not checked for 8.4.142 and later beyond import.
- Capping OpenCV below 5 in C and D is a precaution: no library we use was tested with OpenCV 5.0.

## Questions only a later experiment can answer

1. With combination B (ultralytics 8.4.142+ and paddleocr 3.7 on contrib 4.10.0.84), do `pytest -m slow` and `mtg benchmark run` still match the baseline? This also needs the ty errors from the ultralytics upgrade fixed.
2. In C or D, does moving from OpenCV 4.10 (contrib) to 4.14 (headless) change segmentation corners, pHashes or OCR crops enough to change benchmark results? `cv2.resize`, `findContours` and colour conversion feed the pHash and the OCR crop.
3. Does a Docker image built with `opencv-python-headless` start and serve Streamlit without `libgl1`, `libsm6`, `libxext6` and `libxrender1`, and how much smaller is it?
4. If paddleocr 3.7 is kept only for `engine="onnxruntime"` (without paddlepaddle, see #199), does it run with `opencv-contrib-python` as the only OpenCV package? paddleocr still requires `paddlex[ocr-core]`, so the `opencv-contrib-python` name stays required; whether the onnxruntime path gives the same readings is untested.
5. Does a later paddlex release relax `opencv-contrib-python==4.10.0.84` or switch to a headless name? If it does, the remaining package and its version can be revisited.
