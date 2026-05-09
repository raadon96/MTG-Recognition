# External Integrations
_Generated: 2026-05-09_

## Summary

The application is designed for offline/local operation at runtime. All ML models and the full card database are bundled or pre-downloaded before startup. The only live network call during recognition is an optional Scryfall image URL fetch for display purposes. Data ingestion (a separate offline job) pulls from MTGJson and Scryfall bulk data APIs, which are not invoked during normal app operation.

## APIs & External Services

**Scryfall (runtime UI only):**
- Purpose: Fetch card detail JSON to extract image URL for display in the Streamlit result view
- Endpoint: `https://api.scryfall.com/cards/{scryfall_id}`
- Client: `requests.get()` in `src/mtg_recognition/app/app.py`
- Auth: None (public API)
- Failure handling: Returns `None` on error; UI shows a placeholder instead of crashing

**MTGJson (offline data ingestion only):**
- Purpose: Download `AllPrintings.sqlite` as source data for the database build job
- URL: `https://mtgjson.com/api/v5/AllPrintings.sqlite.zip` (configurable via `settings.MTGJSON_URL`)
- Client: `urllib.request` in `src/mtg_recognition/jobs/build_database.py`
- Auth: None (public)
- Invoked by: `python -m mtg_recognition.jobs.build_database` (offline, not part of the app)

**Scryfall CDN (offline data ingestion only):**
- Purpose: Download card images by Scryfall UUID for building the reference image library
- URL pattern: `https://cards.scryfall.io/{type}/{face}/{id[0]}/{id[1]}/{id}.jpg`
- Client: `aiohttp` (async, batched) in `src/mtg_recognition/jobs/build_database.py`
- Auth: None
- Header: `User-Agent: mtg-recognition/1.0 (contact: adnen.boumessouer@gmail.com)`

**Ultralytics Asset CDN (Docker first-run only):**
- Purpose: Download `mobileclip_blt.ts` and `mobileclip2_b.ts` model assets used by the YOLO model
- Cache location: Docker named volume `mtg_ultra_cache` → `/home/appuser/.cache/ultralytics-assets`
- Mechanism: `docker-entrypoint.sh` creates symlinks at `/app/mobileclip*.ts` pointing to the cached files on each container start; files are downloaded on first run if absent

**PaddleX Font CDN (Docker image build time only):**
- Purpose: Pre-bundle the `simfang.ttf` font required by PaddleOCR/PaddleX
- Source: `https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/simfang.ttf`
- Destination: `/home/appuser/.paddlex/fonts/simfang.ttf` (baked into the Docker image layer during `docker build`)

**PyPI / Package Registries (build time only):**
- Standard: `https://pypi.org`
- PaddlePaddle: `https://www.paddlepaddle.org.cn/packages/stable/cpu/` (explicit index, CPU-only)
- PyTorch CPU: `https://download.pytorch.org/whl/cpu` (explicit index, CPU-only)
- OpenAI CLIP: `git+https://github.com/openai/CLIP` (installed from GitHub)

## Data Storage

**Primary Database:**
- SQLite — `data/reference/MobileCache.sqlite` (~35 MB, ~1M+ card rows)
- Opened read-only: `sqlite:///...?mode=ro&nolock=1`
- Shared engine singleton: `src/mtg_recognition/database.py`
- ORM: SQLModel (`src/mtg_recognition/jobs/schemas.py`)
  - `MobileCache` — denormalized view/table used for all recognition queries (name, setCode, number, baseSetSize, artist, setReleaseYear, url, uuid)
  - `Cards`, `CardIdentifiers`, `Sets`, `Meta` — used only in the offline build job
- Sessions opened per-query: `with Session(engine) as session:`

**pHash Index:**
- `data/reference/index/phash/hashes.npy` — numpy array of shape `[N, 256]`, dtype `uint8`; one row per card
- `data/reference/index/phash/index_to_uuid.json` — ordered list of card UUIDs mapping index position to UUID
- Loaded into memory at `Pipeline` / `PHashCardMatcher` initialization; persists for the lifetime of the process

**Reference Card Images:**
- `data/reference/images/{uuid}.jpg` — ~1M JPEG files, indexed by Scryfall UUID
- Used only during offline index build (`PHashCardMatcher.build_index()`), not at recognition time
- Card image URLs for display are fetched live from Scryfall CDN via the stored `url` column in `MobileCache`

**ML Model Weights (local files, not in git):**
- YOLO segmentation: `src/mtg_recognition/engine/segmentation/models/prod/yoloe-11s-seg.pt`
- U2Net (rembg): controlled by `U2NET_HOME` env var (default: `src/mtg_recognition/engine/segmentation/models/prod/`)
- PaddleOCR: `src/mtg_recognition/engine/ocr/models/prod/` (PP-OCRv5 mobile det + rec models)

**Temporary Files:**
- Warped card images written to `tempfile.NamedTemporaryFile` (`.png`) during batch processing in `src/mtg_recognition/app/batch_processor.py`; caller responsible for cleanup

## Authentication & Identity

**Auth Provider:** None — no authentication or user identity on the Streamlit app.

## Monitoring & Observability

**Error Tracking:** None — errors are caught and displayed inline in the Streamlit UI, or propagated to the CLI.

**Logs:**
- `logging` module used at INFO level in `src/mtg_recognition/engine/ocr/ocr_extraction.py`
- `print()` / `rich.print()` used throughout the pipeline and CLI for status output
- `PYTHONUNBUFFERED=1` set in Docker ensures immediate log flushing to stdout/stderr
- Download errors written to `data/reference/download_errors.log` (data ingestion job only)

## CI/CD & Deployment

**Hosting:**
- Docker container, single-service via `docker-compose.yml`
- Streamlit exposed on port `8501`; Docker healthcheck polls `http://127.0.0.1:8501/_stcore/health` every 30 s (5 s timeout, 40 s start period, 3 retries)

**CI Pipeline:** None configured — no `.github/workflows/`, `.gitlab-ci.yml`, or equivalent.

## Environment Configuration

**No required env vars for runtime.** The app uses default paths that resolve correctly when run from the project root. All paths in `Settings` are relative (e.g. `data/reference/MobileCache.sqlite`).

**Optional env vars (read by `src/mtg_recognition/settings.py`):**
- Any field on `Settings` can be overridden via env var or `.env` file
- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` — skip Paddle model source validation (set automatically)
- `U2NET_HOME` — directory for rembg U2Net weights (set automatically to segmentation models prod dir)
- `ULTRALYTICS_ASSET_CACHE_DIR` — set in Docker to the named volume mount path

**Secrets:** None required — no API keys, tokens, or credentials.

## Webhooks & Callbacks

**Incoming:** None

**Outgoing:** None (the Scryfall API call is a synchronous GET request, not a webhook or push notification)

## Key Observations

- The application is intentionally self-contained at inference time: Scryfall is only called for image display, not for card identification.
- Reference images (`data/reference/images/`) are not accessed at recognition time — only the pHash index files and `MobileCache.sqlite` are read during inference.
- The MobileNetV3 deep-embedding index path (`data/reference/index/mobilenet_v3_small/`) is configured in `src/mtg_recognition/engine/matching/deep_matcher.py` but that matcher is not used in the production `Pipeline`.
- All three external ML model sources (Ultralytics, PaddlePaddle, rembg/U2Net) download models from their respective CDNs on first use; Docker build pre-fetches the PaddleX font but not the model weights themselves.
