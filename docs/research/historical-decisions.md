# Historical decisions recovered from git history and local notes

Research for [#9](https://github.com/decksray/MTG-Recognition/issues/9) (map [#2](https://github.com/decksray/MTG-Recognition/issues/2)).
The question: which load-bearing decisions, and the reasons for them, can be recovered so they can become ADRs?

Researched 2026-10-17 against all 106 commits (`git log --all`).

## Sources

| Source | Where | Notes |
|---|---|---|
| Commit history | `git log --all`, 106 commits, 2026-03-06 → 2026-10-17 | Most commits before May have a subject line only, with no body. |
| Deleted `.planning/` (GSD workflow) | Last full state at `9ff0064^`; deleted in `9ff0064` | `PROJECT.md`, `REQUIREMENTS.md`, `ROADMAP.md`, `STATE.md`, `codebase/*.md` (7 files, generated 2026-05-09), `phases/01-dataset-versioning/*`, `phases/02-cli-sync-command/*`. Decision IDs D-01…D-15 are **per phase**: phase 1 and phase 2 each have their own D-01…, so the IDs collide. |
| Deleted `.memsearch/memory/*.md` | Recoverable at `aefda34`; deleted in `9ff0064` | Session summaries from 2026-05-09/10/16/17. Holds the owner's own arguments for the sync design. |
| `AGENTS.md` | `506b9f7:AGENTS.md`; untracked in `b6aeb23` | The original project description. |
| Local notes (gitignored, owner's machine only) | `.notes/` in the main checkout | `next/grill-explain-test-dataset.md`, `next/grill-architecture-improvement.md`, `next/notes-mobilecache-versioning.md`, `BRAINSTORMING.md`, `2026-03-22-145256-decksray-product-vision.txt`, `failure_artist.md`, `PADDLEOCR_ONNX_*`, `notes_onnx.md`. `next/notes-dvc-cred.md` holds credentials. It was deliberately not read, and nothing from it is reproduced here. |
| Current code | `src/`, `pyproject.toml`, `Dockerfile`, `.gitignore`, `data/reference/.mtg-sync-state.json` | Constants and comments that encode decisions. |

**Confidence levels used below**

- **Stated**: a primary source states the decision and its reason (an owner note, a discussion log, a commit body, or a code comment).
- **Partial**: the decision is visible in code or commits, but its reason is stated only in part.
- **Inferred**: the decision is visible, but its reason is my reconstruction. Each one is listed under [Questions for the owner](#questions-for-the-owner).

A note on GSD-generated text: `.planning/codebase/*.md` was written by an agent (gsd-codebase-mapper) on 2026-05-09. It describes the code; it does not record the owner's intent, and at least one of its claims is wrong (see HD-10). It is cited here as a snapshot of the code, never as a source of rationale.

---

## Summary

| ID | Decision | Status | What | Why |
|---|---|---|---|---|
| HD-01 | Two-stage retrieval: pHash coarse top-k, then OCR fuzzy re-rank | Active | Stated | Partial |
| HD-02 | Deep-embedding (MobileNetV3-small) matcher kept but disabled; CLIP not used as a matcher | Active (dead code) | Stated | **Inferred** |
| HD-03 | CPU-only inference; Raspberry Pi 5 was the original target | Active / target unclear | Stated | Partial |
| HD-04 | Three-path segmentation: bypass → YOLOE → CV box → rembg | Active | Stated | Stated (measured) |
| HD-05 | YOLOE open-vocabulary classes `"book cover"`, `"art work"` plus NMS settings | Active | Stated | Stated (empirical) |
| HD-06 | Decreasing-confidence YOLO loop | **Superseded** by HD-04/05 | Stated | **Inferred** (why it was replaced) |
| HD-07 | Fallback fires only when YOLO finds zero cards | Active | Stated | **Inferred**: owner's proposed rule not implemented |
| HD-08 | YOLO weights loaded as `.pt`, not ONNX | Active | Stated | **Inferred** |
| HD-09 | Warped size clamped to 488×680 … 976×1360 | Active | Stated | Stated |
| HD-10 | MobileCache.sqlite and pHash index committed to git; reference images synced | Active | Stated | Partial |
| HD-11 | `mtg sync`: version state file keyed on the MTGJSON version; pinned by default, `--update-db` opt-in | Active | Stated | Stated |
| HD-12 | All reference-data building in one module (`jobs/sync.py`); CLI is a thin registration file | Active | Stated | Stated |
| HD-13 | DVC + Google Drive for test data; single root `data/test/`; no history rewrite | Active | Stated | Stated |
| HD-14 | Input assumption: cards photographed on a bright, uniform surface | Active | Stated | Stated |
| HD-15 | PaddleOCR PP-OCRv5 *mobile* det/rec, CPU, fp16, HPI disabled | Active | Stated | **Inferred** (why HPI and server_det were dropped) |
| HD-16 | OCR fuzzy weights (name 2.0, artist 0.5, rest 1.0) | Active | Stated | **Inferred** |
| HD-17 | OCR failures degrade to the pHash ranking instead of failing | Active | Stated | Stated |
| HD-18 | Candidate counts: k_phash 100 → 20 (CLI) / 50 (app), top 5 in app | Active, inconsistent | Stated | **Inferred** |
| HD-19 | MobileCache: one denormalized, read-only table | Active | Stated | **Inferred** |
| HD-20 | `BatchResult` stores the full top-K and always runs with `debug=True` | Active | Stated | Stated |
| HD-21 | Product direction: recognition runs on the user's side; in-browser (WASM) is the long-term ideal | Direction | Stated | Stated |
| HD-22 | Docker: non-root user, healthcheck, fonts and model caches baked in or on a volume | Active | Stated | Partial |
| HD-23 | GSD planning workflow abandoned; `.planning/` and `.memsearch/` deleted; phase 3 (docs) never run | Process | Stated | **Inferred** |

Smaller heuristics whose reasons are already written down in `CLAUDE.md` or code comments are listed in [Already documented](#already-documented-heuristics) rather than repeated as records.

---

## Decision records

### HD-01: Two-stage retrieval (pHash coarse → OCR fine)

- **Decided**: a warped card is first hashed (`imagehash.phash`, 16×16 = 256 bits) and compared by Hamming distance against a precomputed index of the reference images to get top-k candidates. PaddleOCR then reads the name ROI and the info ROI, and `rapidfuzz` re-ranks only those candidates.
- **Why (stated)**: the docstring in the first commit says *"1. Perceptual hash … 2. OCR-based fuzzy matching for accurate refinement"*, and the Stage 2 comment says *"Accurate OCR search on reduced search space"* (`506b9f7:src/mtg_recognition/engine/pipeline.py`, lines 47, 84). The product-vision note gives the business reason: *"Decksray uses local OCR + procedural hash recognition that runs directly on the user's machine — resulting in much lower operating costs, better margins"*, set against a competitor's *"cloud-based AI inference"* (`.notes/2026-03-22-145256-decksray-product-vision.txt`).
- **Alternatives**: a deep-embedding matcher (HD-02). No other alternative is recorded.
- **Evidence**: `506b9f7` (the design was already in place at the first commit), current `engine/pipeline.py`, `engine/matching/phash_matcher.py` (`HASH_SIZE = 16  # … higher = more precise, but slower`).
- **Confidence**: the design is stated. The reason is partly stated: cost and local execution are explicit. Using pHash only to shrink the search space is implied by the comment rather than argued.

### HD-02: Deep-embedding matcher disabled; CLIP not a matcher

- **Decided**: `DeepCardMatcher` (torchvision MobileNetV3-small embeddings, index at `data/reference/index/mobilenet_v3_small/`) exists but is not wired in. Its calls in `pipeline.py` are commented out with `# Commented - keeping for future use`. CLIP is a dependency only because Ultralytics' YOLOE text encoder needs it (`set_classes`; MobileCLIP `mobileclip_blt.ts`). It was never used as a matcher.
- **Why**: **not stated anywhere.** The calls were already commented out in the very first commit (`506b9f7:src/mtg_recognition/engine/pipeline.py`, lines 96 and 101), so the switch happened before this repository's history began. `AGENTS.md` (`506b9f7`) still described *"Deep learning embeddings (MobileNetV3) for card matching"* and a `hybrid_matcher` module that is not in the repo.
- **Inferred reasons**: pHash plus OCR was accurate enough; embeddings cost more CPU time and index storage and need torch at inference time; pHash is easier to port.
- **Evidence**: `506b9f7`; `.planning/codebase/CONCERNS.md` ("Dead module: deep_matcher.py"); `.planning/codebase/STACK.md` ("ML — Deep Embedding (inactive)").
- **Confidence**: what = stated; why = **inferred**.

### HD-03: CPU-only inference; Raspberry Pi 5 as the original target

- **Decided**: every model runs on CPU. torch, torchvision, and paddlepaddle are pinned to CPU-only indices (`pyproject.toml` `[[tool.uv.index]]` `pytorch-cpu` and `paddle`, both `explicit = true`). PaddleOCR is called with `device="cpu"` and YOLO inference with `device="cpu"`.
- **Why (partial)**: `deep_matcher.py:26` says `DEVICE = "cpu"  # RPi5 has no CUDA, so CPU.` This is the **only** Raspberry Pi mention in code, history, or notes. The product-vision note (HD-21) wants compute on the user's machine. The `MobileCache` name points the same way, toward a mobile or edge target.
- **Alternatives**: GPU was never considered in any recorded source.
- **Evidence**: `src/mtg_recognition/engine/matching/deep_matcher.py:26`; `pyproject.toml`; `.planning/codebase/STACK.md` ("The entire inference stack is CPU-only").
- **Confidence**: CPU-only is stated by the config. RPi5 as a target is stated once, in a comment in dead code; whether it is **still** a target is unknown (question Q3).

### HD-04: Three-path segmentation (bypass → YOLOE → CV box → rembg)

- **Decided** (`12494e8`, 2026-03-26):
  1. **Bypass**: aspect ratio within 63/88 ± 3 % **and** a dark outer border (mean grey level < 100) → resize only.
  2. **YOLOE** `yoloe-11s-seg`, batched in mini-batches of 20.
  3. **Fallback**: classical Otsu/contour detection (`box.py`), then rembg (`silueta`).
- **Why (stated, measured)**: the owner's evaluation on about 216 test images across four scenarios (`.notes/next/grill-explain-test-dataset.md`):
  - rembg/silueta: *"very reliable and accurate in the single and multi card natural image scenarios"*, but *"fails at single and multi card scan scenarios (probably because the scans do not have a 'background')"*, and *"500-700 ms per image"*.
  - yoloe-11s-seg with the new classes: *"reliably segment[s] the card(s) in all cases except the single card scan"*, at *"~120 ms per image"*.
  - Owner's plan: *"implement a yoloe-11s-seg based segmentation as the baseline … and relegate the rembg silueta based approach to fallback"*, plus batch inference because *"app users are most likely uploading a batch of images at once"*.
  - Bypass: the owner proposed detecting single-card scans by *"running Paddleocr on the border region"*. What was built is an aspect-ratio and dark-border test. Its constants are justified in code: *"scans-without-background have a dark card border (mean ≈ 23–52); scans-with-background have a bright scanner bed (mean ≈ 148–240). A threshold of 100 sits in the middle of the 96-point gap"* (`engine/segmentation/yolo.py:47-52`).
  - CV box before rembg: *"sub-millisecond, good for uniform backgrounds"*; rembg *"very slow to load and run, so only used as a last resort"* (`yolo.py`, `_fallback_segment`).
- **Alternatives rejected**: rembg as the primary path (too slow, fails on scans: `7056922` made it primary on 2026-03-24, and `12494e8` demoted it two days later); YOLO alone (misses single-card scans); detecting scans with OCR (not adopted, reason not stated, see Q5).
- **Evidence**: `7056922`, `12494e8`, `6ebf3ee` (mini-batches), `.notes/next/grill-explain-test-dataset.md`, `engine/segmentation/yolo.py` module docstring (*"covers the rare <1 % of images YOLO misses"*).
- **Confidence**: stated, for the overall shape and for the bypass threshold. Why the bypass uses geometry rather than OCR is not stated.

### HD-05: YOLOE open-vocabulary classes and predict settings

- **Decided**: `YOLO_SEG_CLASSES = ["book cover", "art work"]`; `conf=0.1`, `iou=0.0`, `agnostic_nms=True`, `overlap_mask=False`, `max_det=50` (`engine/segmentation/yolo.py:54-63`).
- **Why (stated, empirical)**: from the grill note: *"instead of using the classes 'book' and 'card', I changed those to 'book cover' and 'art work'. Furthermore, I set agnostic_nms to True (to prevent overlaps between different classes) and overlap_mask to False (to ensure masks do not overlap at the pixel level)"*. It reports confidences *"in most cases >0.2 … even reaching 0.6-0.7"*. The earlier prompt experiments are kept as comments in `run_yolo.py` (added `3844af6`, deleted `fef89e6`): for yolov8s-worldv2, `"card"` was marked `# best` and `"game card"` `# shitty`; for yoloe-26n-seg, `"game card"` and `"card"` were both `# shitty`. `parameters.json` (deleted `7056922`) used `["book"]` and `["game card"]` for yoloe-11s-seg.
- **Alternatives rejected**: the class names `"card"`, `"game card"`, `"MTG card"`, `"Magic: The Gathering Card"`, `"book"`; the models `yolov8s-worldv2`, `yolov8m-worldv2`, `yoloe-26n-seg`.
- **Leftover**: `yolov8s-worldv2.pt` (about 26 MB) is still committed under `models/prod/` but nothing references it (`settings.YOLO_SEG_MODEL_PATH` points only at `yoloe-11s-seg.pt`).
- **Confidence**: stated.

### HD-06: Decreasing-confidence inference loop (superseded)

- **Decided** (`fef89e6`, 2026-03-23): `yolomodel.py` ran YOLO with `max_det=1`, starting at `conf` (0.2 or 0.1 from `parameters.json`) and halving it until `conf_min = 0.02`. It returned the first mask found. The commit subject gives the reason: *"Increased segmentation reliability by running inference with decreasing confidence until min conf is reached"*.
- **Superseded by**: `7056922` (rembg replaced YOLO and `yolomodel.py` was deleted), then `12494e8` (one batched YOLOE pass at `conf=0.1`, `max_det=50`).
- **Why it was replaced**: **not stated.** The loop allowed at most one detection per image, so it could not support multi-card images; it ran up to four sequential inferences per image, which conflicts with batching. Both problems are raised in the grill note, but the note does not mention the loop.
- **Confidence**: what = stated; why it was replaced = **inferred** (Q6).

### HD-07: Fallback triggers on zero detections only

- **Decided**: in `segment_batch`, an image goes to the CV/rembg fallback only if YOLO returns **no** valid segment (`if segments: … else: fallback_indices.append(...)`).
- **Owner's proposal that was not adopted**: *"An image could have 8 cards and yoloe-11s-seg could detect 6 of them … Suggestion: run yoloe-11s-seg inference with confidence of 0.1 and check results. If at least one of the result has a confidence lower than say 0.2 or 0.3, switch to rembg fallback"* (`.notes/next/grill-explain-test-dataset.md`).
- **Why it was dropped**: **not stated.** A multi-card image with partial detections now silently returns only the cards that were found.
- **Confidence**: what = stated (code); why = **inferred** (Q7).

### HD-08: YOLO weights as `.pt`, not ONNX

- **Decided** (`3844af6`, 2026-03-17, *"Reverted to .pt models for segmentation"*): the loader stopped exporting to and loading `.onnx`. `yoloe-11s-seg.onnx` was replaced with `.pt`, and Docker compose gained an Ultralytics asset cache volume (`mtg_ultra_cache`) for the MobileCLIP `.ts` files.
- **Why**: **not stated.** Likely reason: YOLOE's `set_classes()` needs the PyTorch model and its MobileCLIP text encoder. An ONNX export freezes the class embeddings, so changing classes means re-exporting.
- **Portability note**: a WASM port needs ONNX. Finding out why `.pt` won directly affects whether the WASM port is feasible.
- **Confidence**: what = stated; why = **inferred** (Q8).

### HD-09: Warped output clamped between 488×680 and 976×1360

- **Decided** (`36f144d`, 2026-03-28): replace the fixed-size resize with clamping.
- **Why (stated)**: the commit subject says *"avoids unnecessary visual loss in segmented warped cards"*. `seg_utils.py` explains the minimum as *"consistent with Scryfall 'medium' card size and the precomputed pHash index"* and the maximum as *"2× WARPED_MIN_SIZE keeps per-card memory and temp-file sizes predictable"*.
- **Discrepancy**: Scryfall has no "medium" size. The reference images are downloaded as `fileType="normal"` (`jobs/sync.py`, `generate_image_uri`), and Scryfall's "normal" is 488×680, so the code comment should say "normal".
- **Confidence**: stated.

### HD-10: MobileCache.sqlite and pHash index committed to git; reference images synced

- **Decided**: `data/reference/MobileCache.sqlite` (about 35 MB) and `data/reference/index/phash/{hashes.npy,index_to_uuid.json}` (about 28 MB) have been in git **since the first commit** (`506b9f7`). They were updated in `6cd2104` and `507183e`. The roughly 1M reference images in `data/reference/images/` are not in git and are downloaded by `mtg sync`. Since `507183e` the sync state file `data/reference/.mtg-sync-state.json` is also committed (it was gitignored in `60a894c`).
- **Why (partial)**: `4cb66f8` is titled *"add --update-db flag to pin contributors to committed database"*, and its body says *"Database updates require an explicit opt-in via --update-db, making the update deliberate for maintainers"*. The README describes three tiers: `mtg sync` (contributors), `mtg sync --update-db` (maintainers), `mtg sync --force` (troubleshooting). Reproducibility is implied, since everyone matches against the same DB and index, but never written down.
- **The record contradicts itself**: the milestone plan said the opposite. `PROJECT.md` lists *"Storing MobileCache.sqlite in DVC — the sync command rebuilds it from MTGJSON on demand; no versioned binary tracking needed"* as **out of scope**. `ARCHITECTURE.md`, `STACK.md`, and the phase-1 `SKELETON.md` all say reference data is *"not in git"*, and `REQUIREMENTS.md` v2 defers *"DVC tracking of data/reference/ … collaborators rebuild from source"*. The files were in git the whole time, so the GSD codebase map was wrong. The pinned model in `4cb66f8` (2026-05-17) is the **final** decision and replaces the "rebuild on demand" plan.
- **Alternatives rejected**: rebuilding from MTGJSON on every machine (the original phase-2 plan); DVC for `data/reference/` (deferred in `REQUIREMENTS.md`, never revisited); distributing the images through DVC (*"too large; collaborators download from Scryfall"*, `REQUIREMENTS.md` Out of Scope).
- **Not recorded**: why git rather than DVC for the DB and index. Inferred reasons: small enough; works without the Google OAuth setup that DVC needs; the Docker image can `COPY` them. Also not recorded: the repo-growth cost, since every DB update adds about 60 MB of binary history.
- **Confidence**: what = stated; why = **partial** (Q9, Q10).

### HD-11: `mtg sync` versioning model

- **Decided** (phase 2, `02-CONTEXT.md` D-01…D-15; implemented `60a894c`; revised `aefda34`, `9ff0064`, `507183e`, `4cb66f8`):
  - One state file `{mobilecache_version, images_version, phash_version}`. The MTGJSON version string (for example `5.3.0+20260517`) is *"the single source of truth for all three artifacts"* (02-CONTEXT, Specific Ideas).
  - D-02: write `null` when a step starts and the version when it succeeds, so a crash leaves `null` and the next run retries automatically. D-03: a missing or corrupt state file counts as all-null.
  - D-04: check freshness through `https://mtgjson.com/api/v5/Meta.json` instead of downloading the roughly 500 MB ZIP.
  - D-05/D-06: images are compared with mobilecache_version, and the pHash index with images_version.
  - **Final form** (`4cb66f8`): `Meta.json` is fetched only with `--update-db` or `--force`. Otherwise `remote_version` is `None` and the DB is never rebuilt ("pinned mode"). `507183e` added a guard so that a fresh clone (committed state file, empty `images/`) still downloads the images.
- **Why (stated)**: owner's note (`.notes/next/notes-mobilecache-versioning.md`): *"The checks should be done efficiently … cache the version somehow … no need to open the MobileCache database"*. Claude pointed out a chicken-and-egg problem, which led to `Meta.json` (`02-DISCUSSION-LOG.md`, `.memsearch/memory/2026-05-16.md` 11:50). Claude also argued that an explicit image-version check added little, because the download is already incremental; the owner *"chose explicit version check anyway for consistency and predictability"*. The full pHash rebuild was accepted: *"Simpler code; acceptable latency for infrequent manual runs"* (`PROJECT.md` Key Decisions). Manual triggering only: *"CI scheduling is overkill for quarterly MTG releases"*.
- **Superseded sub-decisions**: D-04/D-08 `--skip-db-check` (removed in `aefda34`; replaced in effect by pinned-by-default); `PHashCardMatcher(auto_build=…)` (removed in `507183e`).
- **Confidence**: stated.

### HD-12: Reference data built in one module; thin CLI

- **Decided** (`d62cb88`, 2026-05-16): `jobs/build_database.py` was renamed to `jobs/sync.py`, which owns MobileCache, the images, and the pHash index. `build-index` is a `@click.command()` in `phash_matcher.py`. `cli.py` contains only `add_command()` calls.
- **Why (stated)**: the owner *"advocated for centralizing the building of MobileCache, reference images, and pHash index in a single module, arguing that these three artifacts constitute 'reference data' that should be built together and versioned together"* (`.memsearch/memory/2026-05-16.md` 21:29). Claude had argued against putting Click inside the library module `phash_matcher.py` (*"would violate layer separation"*). The owner pointed to the existing pattern (`pipeline.py`, `segmentation/cli.py` each have a Click `main`), and Claude agreed (21:12).
- **Supersedes**: phase-2 D-12 (*"All CLI code lives in src/mtg_recognition/cli.py"*).
- **Rejected**: separate `*_cli.py` adapter modules (Claude's middle-ground suggestion); a `build_database.py` re-export shim.
- **Confidence**: stated. Relevant to the import-linter ticket: the owner accepted Click inside engine modules on purpose.

### HD-13: DVC + Google Drive for the test dataset

- **Decided** (phase 1; `2853860`): `dvc[gdrive]` in the `dev` group (D-01); `benchmarks/dataset/` moved to `data/test/6_benchmark/` and all of `benchmarks/` gitignored (D-02, D-03); `git rm --cached` with no history rewrite (D-05); remote `origin` at `gdrive://root/decksray/MTG-Recognition/data/test` (D-07, D-08); personal Google accounts, no service account (D-09); `data/test/` as the single DVC root.
- **Why (stated)**: `PROJECT.md`: *"Free, simple auth for small team, DVC handles versioning"*; constraint *"No new cloud costs: Google Drive (free tier)"*. `SKELETON.md`: DVC is the *"industry standard for ML data versioning … deduplicates by content hash"*. A single root gives *"one pull command"*. No history rewrite *"to avoid the complexity and rebase-pain of git filter-branch / git filter-repo"*, since the 73 JPGs are *"non-sensitive"* and *"bounded"*. DVC is a dev dependency because it *"Prioritizes frictionless collaborator setup over env cleanliness"* (`01-DISCUSSION-LOG.md`).
- **Alternatives rejected** (`SKELETON.md`, `01-RESEARCH.md`): git-lfs (*"charges for bandwidth, no Google Drive backend"*); S3/Azure/GCS (cost); custom sync scripts (*"reinvents pointer-files + dedup"*); a DVC HTTP remote (needs a server); a local-only remote; a service account; DVC's default OAuth app (*"blocked by Google … since late 2023"*, treeverse/dvc#10516, so each developer creates their own GCP Desktop OAuth app); a standalone DVC install; keeping non-image benchmark output in git.
- **Secret handling**: `gdrive_client_id` goes in the committed `.dvc/config`; `gdrive_client_secret` goes only in the gitignored `.dvc/config.local` (`d4e6f00`, `STATE.md`).
- **CI implication** (not discussed then): the constraint was *"No CI/CD changes"*. A CI benchmark on `data/test/` (destination of #2) would need a non-interactive Drive credential such as a service account, which reopens D-09.
- **Confidence**: stated.

### HD-14: Input assumption: bright, uniform background

- **Decided**: the photo scenarios assume *"the card(s) are placed on a white surface. This reflects accurately usecase of the application since we will indicate to the user in the app that the cards need to be taken in good lighting conditions with a clear bright background (e.g. white desk table)"* (`.notes/next/grill-explain-test-dataset.md`).
- **Test categories** (from `.notes/cmds_OCR_CLI.md` and `SKELETON.md`): `0_error`, `1_photos_single_card`, `2_photos_multiple_cards`, `3_scans_single_card_with_bg`, `4_scans_single_card_no_bg`, `5_scans_multiple_cards`, `6_benchmark` (36 raw + 36 ready UUID-named images, 20 OCR images).
- **Why it matters**: the rembg and CV-box fallbacks and the bypass threshold all depend on this assumption. It is a product constraint, not only a test fixture.
- **Confidence**: stated. Whether the app actually tells users this is not verified (Q11).

### HD-15: PaddleOCR PP-OCRv5 mobile, CPU, fp16, HPI disabled

- **Decided**: `PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="PP-OCRv5_mobile_rec", device="cpu", enable_hpi=False, precision="fp16", lang="en")`, with document unwarping and textline orientation off (`engine/ocr/ocr.py`). The models are committed under `engine/ocr/models/prod/`.
- **History**:
  - `6b08687` (2026-03-15) *"enabled hpi mode for ocr text-detection"*: ONNX Runtime backend for detection with `PP-OCRv5_server_det`; recognition kept on Paddle because, per the code comment, *"onnx backend does not work for rec"*. The Dockerfile ran `paddlex --install hpi-cpu`.
  - `9350146` (2026-03-30) added `PP-OCRv5_mobile_det`; `38d517f` (2026-04-14) deleted `PP-OCRv5_server_det`.
  - `96ef7eb`/`bb9da90` (2026-04-15/16) *"removed hpi mode from ocr"*: the `enable_hpi` parameter, the HPI config, the Dockerfile plugin install, and the README step were all removed.
  - `.notes/PADDLEOCR_ONNX_*` (2026-03-06 investigation) claims ONNX Runtime is *"+5-20% faster"* on CPU, with a *"+30-60s"* first-run conversion.
- **Why HPI and server_det were dropped**: **not stated.** Likely reasons: the HPI plugin made Docker builds and setup more complex for a small gain, and the mobile detector is faster on CPU with enough accuracy for small ROIs.
- **Alternatives**: `rapidocr>=3.7.0` sits in the dev dependencies (*"alternative OCR library for dev comparison"*, `STACK.md`), but no evaluation is recorded anywhere. `paddle2onnx`/`onnxslim` are leftovers (the map already marks them removable).
- **Portability**: `BRAINSTORMING.md` Annex identifies PaddleOCR's lack of a WASM build as the main blocker for in-browser use, with ONNX export of PP-OCRv5 det/rec as the way around it.
- **Confidence**: what = stated; why = **inferred** (Q12, Q13).

### HD-16: OCR fuzzy-score weights

- **Decided**: `ATTRIBUTE_MAPPINGS` gives name 2.0, artist 0.5, and setCode, number, baseSetSize, setReleaseYear, power, toughness 1.0 each. Only name and artist use the `default_process` preprocessor.
- **History**: in `3ebd0d8` (2026-04-14) setCode went from 2.0 to 1.0 and setReleaseYear from 0.5 to 1.0. Artist has been 0.5 since `506b9f7`.
- **Why**: **not stated.** A low artist weight fits the unreliable artist OCR documented in `.notes/failure_artist.md` (`"CMR·ENJACK WANG"`, where the set-code artifact gets glued onto the artist name). No benchmark accompanies any weight change.
- **Confidence**: what = stated; why = **inferred** (Q14).

### HD-17: OCR failure degrades to the pHash ranking

- **Decided**: `predict` is wrapped in `try/except RuntimeError`. On failure the texts are empty, the scores become zero vectors, and the pHash order stands.
- **Why (stated)**: *"PaddlePaddle's C++ backend throws intermittent `RuntimeError: std::exception` on some inputs"* (`CLAUDE.md`; `.planning/codebase/ARCHITECTURE.md` "Error Handling: Fail-safe degradation").
- **Confidence**: stated.

### HD-18: Candidate counts

- **Decided**: `k_phash_candidates` defaulted to 100 at `506b9f7`. Today the default in `Pipeline.find_best_matches` (and the CLI `--k-phash`) is 20, while the app uses `K_PHASH_CANDIDATES = 50` and `K_TOP_MATCHES = 5` (`app/batch_processor.py:18-19`, introduced `e89b827`, *"minor improvments"*).
- **Why**: **not stated.** The CLI and the app use different candidate counts, so benchmark results from the CLI do not describe app behaviour.
- **Confidence**: what = stated; why = **inferred** (Q15). This matters for the CI benchmark ticket.

### HD-19: MobileCache: one denormalized, read-only table

- **Decided**: `create_database` joins MTGJSON `cards × cardIdentifiers × sets` into one `MobileCache` table. Columns: uuid, name, manaCost, type, number, baseSetSize, setCode, artist, power, toughness, borderColor, url (Scryfall `normal` image URI), setReleaseYear (taken from `sets.releaseDate`). Every column is a string; the only index is the `uuid` primary key, and the composite `Index("number", "baseSetSize")` is commented out. Rows are streamed with `yield_per(1000)` and inserted in batches of 5000 *"crucial for memory"*. The app opens the database with `sqlite:///…?mode=ro&nolock=1` (`database.py`). `setcodes.json` was dropped on day one (`9138ecd`).
- **Why**: memory-aware streaming is stated in a comment. **Not stated**: why denormalize; why only these columns (text and flavorText are commented out); why `nolock=1`; why `setReleaseYear` comes from the set release date rather than the printed copyright year (they can differ); what "Mobile" in the name refers to.
- **Inferred**: a small, read-only, single-table lookup keyed by UUID suits a constrained or in-browser client (sql.js is mentioned in `BRAINSTORMING.md`), and the matcher only ever filters candidates by UUID.
- **Confidence**: what = stated; why = **inferred** (Q16, Q17).

### HD-20: `BatchResult` keeps the full top-K and debug data

- **Decided**: `BatchResult` stores `matches`, `scores`, `ocr_debug`, and `timings_ms`, and batch runs always pass `debug=True`, so the detail view never re-runs the pipeline.
- **Why (stated)**: the root cause was that *"BatchResult stores only matches[0] … This forces display_batch_detail_view … to re-run the entire pipeline"*. The owner chose Option A, *"always capture debug data during batch processing"*, over computing it on demand. The memory cost was accepted *"for typical batch sizes (20–100 cards)"* (`.notes/next/grill-architecture-improvement.md`; implemented around `0261758`/`4e5cc61`/`afb17de`, 2026-04-13).
- **Note**: the same note flags `K_PHASH_CANDIDATES`/`K_TOP_MATCHES` living in `batch_processor.py` as *"an odd dependency direction"*, left for later. The map's BatchProcessor exemplar refactoring builds on this.
- **Confidence**: stated.

### HD-21: Product direction: compute on the user's side; WASM as the long-term goal

- **Decided (direction)**: `.notes/BRAINSTORMING.md` compares six deployment approaches (server web app, in-browser WASM/WebGPU, desktop, hybrid shell, native mobile, local Docker). In-browser inference is *"the long-term ideal — zero install, zero server cost … not in scope for v1"*. Local Docker is *"dominated"*. The vision note frames local recognition as the cost advantage over a cloud-inference competitor, with an EU and collector focus (Cardmarket).
- **Feasibility notes (stated)**: PyTorch YOLO and the PaddleOCR library cannot run in the browser; the PP-OCRv5 det/rec models and YOLO can be exported to ONNX for ONNX Runtime Web; DB post-processing, CTC decoding, the pipeline, and the extraction logic would have to be rewritten; the pHash index (about 32 MB) would be cached in IndexedDB; SQLite works through sql.js.
- **Relation to the map**: this is where #2's "portability lens" comes from. The note did not settle between server-side (A) and on-device (B/C); it left that as *"at expected scan volume, is your per-scan server cost higher than your margin?"*.
- **Confidence**: stated (as direction, not as a final decision).

### HD-22: Docker runtime choices

- **Decided**: multi-stage build (`base → deps → runtime`); non-root `appuser` (uid 10001) and a healthcheck on `/_stcore/health` (`260414b`); `docker compose` with a named volume for Ultralytics MobileCLIP assets, symlinked by `docker-entrypoint.sh` (`3844af6`); PaddleX `simfang.ttf` downloaded at build time (`59b216a`).
- **Why**: partly stated in Dockerfile comments (*"so downloads can write there"*). Not stated: why the font is pre-bundled. Likely: PaddleX fetches it from a Baidu CDN at first use, which fails or is slow for a non-root, possibly offline container.
- **Confidence**: partial (Q18).

### HD-23: GSD planning workflow abandoned

- **Facts**: the GSD milestone (`960e331`…`f05261d`, 2026-05-09…16) finished phases 1 and 2. Phase 3 (Documentation: DOC-01…03) never started, although `CONTRIBUTING.md` and README updates partly covered it. On 2026-05-17 `9ff0064` (*"implemented explicit predicates for rebuilding artifacts"*) deleted almost all of `.planning/` and `.memsearch/` without saying so in the message. The `D-NN` comments in `jobs/sync.py` and `phash_matcher.py:391` now point at documents that exist only in history.
- **Why**: **not stated.**
- **Consequence**: the `D-NN` citations in code are dangling, and they are ambiguous because the IDs are per phase. When the ADRs are written, these comments should cite ADR numbers instead.
- **Confidence**: facts stated; why = **inferred** (Q19).

---

## Already documented heuristics

These already have their reasons in `CLAUDE.md` or code comments, so they need no ADR. They are listed so nobody hunts for them again.

- Flavor-text zone filter: drop tokens with `y1 < 0.08·H and x2 < 0.90·W`; deliberately conservative to keep artist names; returns the input unchanged if everything would be dropped (`a4503bb`, `8b5147d`).
- `extract_setcode_candidates`: language-code separators, a `[A-Z]{2,5}` fallback, `_SETCODE_BLOCKLIST`; returns `[]` rather than noise (`788625d`, `3ebd0d8`, `370f6ca`).
- `extract_number_and_base_set_size`: two-stage extractor, slash guard, rejects copyright years, current-year ceiling (`1b7b35e`). Open concern: `extract_set_release_year` still hard-codes `<= 2030` (`.planning/codebase/CONCERNS.md`).
- `extract_power_toughness` handles star P/T (`9350146`).
- `enhance_contrast` CLAHE and gamma bands (`yolo.py:66-83`). The thresholds have no recorded reason or tests (`CONCERNS.md`).
- YOLO mask aspect-ratio tolerance 0.2 vs bypass tolerance 0.03 (`yolo.py`). The 0.2 has no recorded reason.

---

## Superseded decisions (timeline)

| When | Commit | Decision | Replaced by |
|---|---|---|---|
| before 2026-03-06 | (pre-history) | Deep MobileNetV3 matcher, `hybrid_matcher` | pHash + OCR (HD-01/02) |
| 2026-03-06 | `9138ecd` | `setcodes.json` lookup | set code taken from OCR |
| 2026-03-09…17 | `272926f` → `3844af6` | YOLO as ONNX (`yoloe-11s-seg.onnx`, `yolov8s-worldv2.onnx`) | `.pt` (HD-08) |
| 2026-03-15 | `6b08687` | PaddleOCR HPI/ONNX Runtime for detection, `PP-OCRv5_server_det` | mobile det, HPI off (HD-15) |
| 2026-03-23 | `fef89e6` | Decreasing-confidence loop, `max_det=1` | one batched pass (HD-06) |
| 2026-03-24 | `7056922` | rembg (`silueta`, replacing `u2netp`) as primary segmentor | YOLOE primary (HD-04) |
| 2026-03-26 | `12494e8` | Fixed-size warp target | min/max clamp (`36f144d`, HD-09) |
| 2026-05-09 | `PROJECT.md` | Rebuild MobileCache from MTGJSON on demand; reference data "not in git" | committed and pinned (HD-10) |
| 2026-05-16 | `02-CONTEXT.md` D-04/D-08 | `--skip-db-check` | removed `aefda34`; pinned by default `4cb66f8` |
| 2026-05-16 | `02-CONTEXT.md` D-12 | All CLI code in `cli.py` | thin `cli.py` + domain commands (HD-12) |
| 2026-05-16 | `02-CONTEXT.md` D-01 / `60a894c` | State file gitignored | committed (`507183e`) |
| 2026-05-17 | `507183e` | `PHashCardMatcher(auto_build=…)` | removed |
| 2026-05-09 | `STACK.md` | *"No linter or formatter is configured"* | ruff, ty, import-linter, pre-commit (`b26a35f`, 2026-10-17) |

---

## Discrepancies found

1. **Reference data in git**: `.planning/` says MobileCache and the pHash index are "not in git". They have been committed since `506b9f7` (HD-10).
2. **Scryfall "medium"**: the `seg_utils.py` comment and `ARCHITECTURE.md` say "medium"; the code downloads `normal` (488×680) (HD-09).
3. **k_phash**: CLI default 20 vs app 50 (HD-18).
4. **Unused weights**: `yolov8s-worldv2.pt` is still committed under `models/prod/` (HD-05).
5. **Dangling citations**: `D-NN` comments in `jobs/sync.py` and `phash_matcher.py:391` point at deleted, phase-scoped documents (HD-23).
6. **Owner's fallback rule not implemented**: the low-confidence fallback rule from the grill note was never built (HD-07).

---

## Questions for the owner

Each question corresponds to a rationale I inferred and need you to confirm.

- **Q1 (HD-01)**: Is pHash only there to cut the search space for OCR, or is it also meant to decide on its own when OCR is weak? Did you ever measure OCR over the full DB, without pHash?
- **Q2 (HD-02)**: Why was the MobileNetV3 matcher switched off before the first commit: accuracy, speed, index size, or the torch dependency? Is there a record of the comparison?
- **Q3 (HD-03)**: Is a Raspberry Pi 5 (or any ARM/edge device) still a target, or has the WASM/in-browser port replaced it as the constrained target?
- **Q4 (HD-03)**: Is CPU-only a hard requirement for the engine, or just what the dev and Docker setup happens to use?
- **Q5 (HD-04)**: Why did single-card scan detection end up as aspect ratio plus dark border instead of the OCR-on-border idea from your note?
- **Q6 (HD-06)**: Was the decreasing-confidence loop dropped because of `max_det=1` and multi-card support, because of batching, or because of the "book cover"/"art work" classes?
- **Q7 (HD-07)**: Your note proposed falling back when any detection is below 0.2–0.3 confidence, to catch partially detected multi-card images. Was this tried and rejected, or deferred? Is silently missing some cards acceptable?
- **Q8 (HD-08)**: Why `.pt` over ONNX for YOLO: `set_classes` needing PyTorch and MobileCLIP, accuracy, or Ultralytics ONNX bugs? This affects the WASM port.
- **Q9 (HD-10)**: Why git rather than DVC for MobileCache and the pHash index: size, no OAuth needed, or Docker `COPY`? Is about 60 MB of binary history per DB update acceptable long term?
- **Q10 (HD-10)**: Is the main purpose of pinning reproducibility, so that all contributors and benchmarks match against the same DB and index?
- **Q11 (HD-14)**: Does the app tell users to photograph on a bright, uniform background? Should this assumption be in an ADR as a product constraint?
- **Q12 (HD-15)**: Why was HPI/ONNX Runtime removed in `96ef7eb`/`bb9da90`: setup or Docker complexity, errors, no measurable speed-up, or reproducibility?
- **Q13 (HD-15)**: Why mobile_det over server_det, and was RapidOCR ever evaluated (it is still in the dev dependencies)?
- **Q14 (HD-16)**: How were the fuzzy weights chosen (name 2.0, artist 0.5; setCode 2.0→1.0 and year 0.5→1.0 in `3ebd0d8`)? By hand, or against the benchmark?
- **Q15 (HD-18)**: Which k_phash is canonical, 20 (CLI) or 50 (app)? Should the CI benchmark use the app's values?
- **Q16 (HD-19)**: What does "Mobile" in MobileCache mean, and why denormalize into one table with only these columns?
- **Q17 (HD-19)**: Why `nolock=1`, and why take `setReleaseYear` from the set release date rather than the printed copyright year?
- **Q18 (HD-22)**: Was the PaddleX font pre-bundled because the runtime download failed (non-root user, offline, or the Baidu CDN)?
- **Q19 (HD-23)**: Why was `.planning/` deleted in `9ff0064`? Was GSD dropped on purpose, and should the `D-NN` comments be replaced by ADR references?

---

## Glossary candidates

Terms defined or used consistently in the sources:

- **Reference data**: the three artifacts built and versioned together: MobileCache, reference images, pHash index (owner, `.memsearch/memory/2026-05-16.md`).
- **MobileCache**: the denormalized, read-only SQLite table and file of card printings derived from MTGJSON AllPrintings (`jobs/schemas.py`).
- **Reference images**: Scryfall `normal` card images, one per UUID, in `data/reference/images/`. Used only to build the pHash index, never at recognition time.
- **pHash index**: `hashes.npy` (N × 256 bits) plus `index_to_uuid.json`.
- **MTGJSON version**: for example `5.3.0+20260517`; the single version identifier for all reference data.
- **Sync state file** / **pinned mode**: `.mtg-sync-state.json`; `mtg sync` without `--update-db` never rebuilds the DB.
- **Segment** / **card segment**: one card cut out of an input image (`CardSegment`); an image may contain several.
- **Warped card** / **warped image**: a perspective-corrected card image, clamped to 488×680…976×1360.
- **Bypass**, **YOLO path**, **fallback** (CV box, rembg): the three segmentation paths.
- **Scan with / without background**, **photo**: input scenarios (test categories 1–5).
- **Candidates**: the pHash top-k passed to OCR; **matches**: the final re-ranked top-k.
- **Name ROI** / **info ROI**: the top (4–11 %) and bottom (about 13 %) card regions read by OCR.
- **Properties**: the attributes extracted by OCR (`name`, `setcode_candidates`, `number`, `base_set_size`, `artist`, `set_release_year`, `power`, `toughness`).
- **Collector number** / **base set size**: the `011/264` pair at the bottom of a card.
- **Language-code artifact**: the `SET·LANG` token (for example `TDM·DE`) used to find the set code.
- **Flavor-text zone**: the top strip of the info ROI where flavor text bleeds in.
- **Benchmark set**: `data/test/6_benchmark/`, with UUID filenames as ground truth (`success` flag in the CLI JSON).
