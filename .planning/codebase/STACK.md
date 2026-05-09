# Technology Stack
_Generated: 2026-05-09_

## Summary

MTG-Recognition is a Python 3.12 application that combines computer vision, deep learning, and fuzzy text matching to identify Magic: The Gathering cards from photos. It is served as a Streamlit web app via Docker. The entire inference stack is CPU-only — PyTorch and PaddlePaddle are pinned to CPU-specific wheels from custom package indices. Package management uses `uv` with a committed lockfile.

## Languages

**Primary:**
- Python 3.12 — all application and ML code (`src/mtg_recognition/`)

**Secondary:**
- Shell — Docker entrypoint (`docker-entrypoint.sh`)

## Runtime

**Environment:**
- CPython 3.12 (`requires-python = ">=3.12"`)
- Docker base image: `python:3.12-slim`

**Package Manager:**
- `uv` (latest, pulled from `ghcr.io/astral-sh/uv` during Docker build)
- Lockfile: `uv.lock` — present and committed, used with `--locked` flag
- Build backend: `uv_build>=0.10.7,<0.11.0`

## Frameworks

**Web / UI:**
- Streamlit `>=1.54.0` — batch-upload UI, debug expanders, sidebar controls (`src/mtg_recognition/app/app.py`)

**ML — Object Detection / Segmentation:**
- Ultralytics `>=8.4.1` — YOLO wrapper; uses `yoloe-11s-seg` instance-segmentation model (`src/mtg_recognition/engine/segmentation/yolo.py`)
- PyTorch `>=2.10.0` — CPU-only wheel from `https://download.pytorch.org/whl/cpu`; inference backend for YOLO and MobileNetV3
- torchvision `>=0.25.0` — CPU-only; `MobileNetV3-Small` pretrained model and image transforms (`src/mtg_recognition/engine/matching/deep_matcher.py`)

**ML — Background Removal:**
- rembg `>=2.0.73` (with `[cli,cpu]` extras) — last-resort segmentation fallback using U2Net model (`src/mtg_recognition/engine/segmentation/rembg.py`)

**ML — OCR:**
- PaddlePaddle `3.2.0` — CPU-only wheel from `https://www.paddlepaddle.org.cn/packages/stable/cpu/`; backend for PaddleOCR
- PaddleOCR `>=3.3.3` — text extraction from card name and info ROI (`src/mtg_recognition/engine/ocr/ocr.py`)
- ONNX Runtime `>=1.23.2` + paddle2onnx `>=2.1.0` — transitive PaddleOCR dependencies; not called directly in application code
- onnxslim `>=0.1.88` — model optimization utility in PaddleOCR toolchain

**ML — Deep Embedding (inactive):**
- OpenAI CLIP `1.0` — installed from `git+https://github.com/openai/CLIP`; referenced by root-level TorchScript files `mobileclip2_b.ts` / `mobileclip_blt.ts`, not wired into the active pipeline

**Computer Vision:**
- OpenCV `opencv-contrib-python==4.10.0.84` — image loading, CLAHE preprocessing, Otsu thresholding, perspective warp, classical card detection (`src/mtg_recognition/engine/segmentation/box.py`, `yolo.py`, `src/mtg_recognition/engine/ocr/enhance.py`)
- Pillow `>=12.0.0` — PIL Image I/O, format conversions

**Perceptual Hashing:**
- imagehash `>=4.3.2` — 256-bit pHash (16×16) for coarse card matching (`src/mtg_recognition/engine/matching/phash_matcher.py`)

**Fuzzy Matching:**
- rapidfuzz `>=3.14.3` — weighted fuzzy scoring of OCR-extracted attributes against DB candidates (`src/mtg_recognition/engine/ocr/ocr.py`)

**Database / ORM:**
- SQLModel `>=0.0.31` — ORM over SQLite; schemas in `src/mtg_recognition/jobs/schemas.py`, shared engine in `src/mtg_recognition/database.py`
- SQLAlchemy (transitive via SQLModel) — `Row` type used as the match result container throughout the pipeline

**Data Handling:**
- NumPy (transitive) — pHash index stored as `.npy`; Hamming distance via bitwise XOR
- numpydantic `>=1.7.0` — typed NDArray annotations for hash/embedding arrays (`src/mtg_recognition/engine/matching/phash_matcher.py`)
- pandas `>=2.3.3` — score matrix construction in OCR debug view
- ijson `>=3.4.0.post0` — streaming JSON parser for database build jobs

**Settings / Config:**
- pydantic-settings `>=2.12.0` — `Settings` class with `.env` file support (`src/mtg_recognition/settings.py`)
- python-dotenv `>=1.2.1` — env file loading

**CLI:**
- click `>=8.3.1` — pipeline CLI (`src/mtg_recognition/engine/pipeline.py`)

**HTTP:**
- aiohttp `>=3.13.2` — async image downloads in `src/mtg_recognition/jobs/build_database.py`
- requests `>=2.32.5` — sync Scryfall API fetch in `src/mtg_recognition/app/app.py`

**Utilities:**
- tqdm `>=4.67.1` — progress bars for index build and batch jobs
- ftfy `>=6.3.1` — Unicode text fixing in OCR post-processing
- regex `>=2026.2.28` — advanced regex patterns in `src/mtg_recognition/engine/ocr/ocr_extraction.py`
- rich `>=14.3.2` — colored terminal output in CLI
- icecream `>=2.1.10` — debug print utility
- openpyxl `>=3.1.5` — Excel export (dev/reporting, declared in main deps)

**Testing:**
- pytest `>=9.0.2` — test runner; config in `pyproject.toml` (`testpaths = ["tests"]`)
- Custom marker: `slow` — marks tests requiring the segmentation model; deselect with `-m 'not slow'`

**Dev Extras:**
- ipykernel `>=7.1.0` — Jupyter notebook support
- sqlacodegen `>=3.2.0` — SQLAlchemy model generation from existing DB
- rapidocr `>=3.7.0` — alternative OCR library for dev comparison
- streamlit-webrtc `>=0.64.5` — webcam capture (experimental, not in prod app)
- pyarrow `>=22.0.0`, ipyplot `>=1.1.2` — notebook data exploration
- cv2-enumerate-cameras `>=1.3.3` — camera enumeration for dev tooling

## Configuration

**Environment:**
- `src/mtg_recognition/settings.py` — Pydantic `BaseSettings`; reads `.env` file
- Key settings: `DATA_DIR`, `MTGJSON_URL`, `BATCH_SIZE=20`, `MAX_UPLOAD_SIZE_MB=25`, `MAX_IMAGE_DIMENSION=2048`, `JPEG_QUALITY=90`
- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` and `U2NET_HOME` are pushed to `os.environ` at settings instantiation
- No secrets required at runtime — all data is local

**Build:**
- `pyproject.toml` — project metadata, dependencies, uv index configuration, pytest markers
- `uv.lock` — pinned dependency graph
- `Dockerfile` — multi-stage build: `base` → `deps` (dependency-only layer) → `runtime`
- `docker-compose.yml` — single service, port `8501:8501`, named volume `mtg_ultra_cache` for Ultralytics assets

## Platform Requirements

**Development:**
- Python 3.12+, `uv` package manager
- System libs for OpenCV: `libgl1`, `libglib2.0-0`, `libgomp1`, `libsm6`, `libxext6`, `libxrender1`
- Reference data under `data/reference/` (not in git): `MobileCache.sqlite`, `images/` (~1M JPEGs), pHash index (`.npy` + `.json`), YOLO `.pt` models, OCR models

**Production:**
- Docker (recommended); Streamlit listens on `0.0.0.0:8501`
- CPU-only: no GPU required
- Runs as unprivileged user `appuser` (uid 10001) inside container
- PaddleX font `simfang.ttf` fetched from Baidu CDN at Docker image build time and bundled into the image layer

## Key Observations

- All ML inference is CPU-only — PyTorch and PaddlePaddle are installed from CPU-specific indices; GPU migration would require significant dependency changes.
- ONNX Runtime and paddle2onnx are transitive PaddleOCR dependencies and are not directly invoked by application code.
- OpenAI CLIP and `DeepCardMatcher` (MobileNetV3 embeddings) are implemented but not wired into the active `Pipeline` — only `PHashCardMatcher` is used in production.
- No linter or formatter is configured in `pyproject.toml`; the project convention is to match surrounding code style.
