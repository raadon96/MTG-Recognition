# Accuracy reconnaissance: top-1, run-to-run variance and runtime

Answers [Measure today's top-1 accuracy, run-to-run variance and runtime on the UUID-named images](https://github.com/decksray/MTG-Recognition/issues/6). This is reconnaissance for "Define the accuracy benchmark" (#7), **not** the official CI baseline.

## Setup

- Commit `d8033f9` (`dev`), no code changes. Python 3.12.15, PaddleOCR 3.4.0, ultralytics 8.4.21.
- CPU: AMD Ryzen AI 7 350 (16 threads), 30 GB RAM, Linux.
- Input: every UUID-named image under `data/test/` excluding `masks/` and `ready/`: **260 images, 205 unique UUIDs**. Only five directories hold UUID-named files; `0_error`, `0_failure_cases`, `2_photos_multiple_cards`, `3_scans_single_card_with_bg` and `5_scans_multiple_cards` have none.
- Each category was symlinked into its own directory and run with the existing CLI: `uv run python -m mtg_recognition.engine.pipeline -i <dir> -o <cat>.json` (defaults `--k-phash 20 -k 1`, always segments). That is one process per category, so each category pays one model load. See `accuracy-recon/run.sh`.
- The full set was run twice, back to back. The `success` flag was read from segment 0.

## Results

| Category | n | Run 1 top-1 | Run 2 top-1 | Wall (s) r1 / r2 | s/img (r1, incl. load) |
|---|---|---|---|---|---|
| 1_photos_single_card | 200 | 188/200 (94.0 %) | 188/200 (94.0 %) | 121 / 116 | 0.61 |
| 4_scans_single_card_no_bg | 4 | 3/4 (75.0 %) | 3/4 (75.0 %) | 8 / 8 | 1.88 |
| 6_benchmark/ocr/common_layout | 4 | 4/4 (100 %) | 4/4 (100 %) | 8 / 8 | 2.09 |
| 6_benchmark/ocr/uncommon_layout | 16 | 15/16 (93.8 %) | 15/16 (93.8 %) | 15 / 14 | 0.93 |
| 6_benchmark/images/raw | 36 | 36/36 (100 %) | 36/36 (100 %) | 22 / 21 | 0.61 |
| **Total** | **260** | **246/260 (94.6 %)** | **246/260 (94.6 %)** | **174 / 167** | **0.67** |

- **Run-to-run variance: none.** Both runs gave the same top-1 UUID for all 260 images, with zero flips. Neither run logged `OCR prediction failed` (the PaddleOCR `RuntimeError` fallback never fired).
- **Runtime:** about 3 minutes for the full set. A one-image CLI run takes 6.7 s, so start-up (imports and model load) is about 6 s and steady-state cost is about **0.55 s/image** on CPU. Five category processes put about 30 s of the 174 s into start-up.
- **Segments:** every image produced exactly one segment, so the UUID check was meaningful everywhere. No multi-card images are UUID-named.
- **Duplicates:** 55 UUIDs appear in two or three categories (all pairs are `1_photos_single_card` with a `6_benchmark` subfolder). 18 pairs are byte-identical and 35 differ (different files of the same card). Duplicates always agreed with each other, and 192/205 unique UUIDs were matched correctly.

## Misses (identical in both runs)

| Category | UUID | Expected | Got | Kind |
|---|---|---|---|---|
| photos | 156dd28e | Selhoff Occultist JMP 174 | Selhoff Occultist SIS 21 | right card, wrong printing |
| photos | 96851213 | Lunarch Veteran // Luminous Phantom INR 32 (front) | same card, back-face UUID `fa15d8b3` | **other face, same image** |
| photos | fa15d8b3 | Lunarch Veteran // Luminous Phantom INR 32 (back) | Burnout Bashtronaut PDFT 115s | wrong card (DFC) |
| photos | 23c2e576 | Ecstatic Awakener // Awoken Demon INR 107 | Savanti Romero TMT 72 | wrong card (DFC) |
| photos | 3b78d1a8 | Shrill Howler // Howling Chorus INR 214 | Jin-Gitaxias MOM 65 | wrong card (DFC) |
| photos | 8dbff4d6 | Soulcipher Board // Cipherbound Spirit INR 85 | Epitaph Golem INR 262 | wrong card (DFC) |
| photos | ab62f9e1 | Hinterland Logger // Timber Shredder INR 203 | Vilespawn Spider INR 251 | wrong card (DFC) |
| photos | d3451c0f | Grizzled Angler // Grisly Anglerfish INR 67 | Gird for Battle JMP 106 | wrong card (DFC) |
| photos | d63e8a74 | Hanweir Watchkeep // Bane of Hanweir INR 158 | Amazing Acrobatics SPM 25 | wrong card (DFC) |
| photos | de33b83e | Smoldering Werewolf // Erupting Dreadwolf INR 171 | Oran-Rief NEC 173 | wrong card (DFC) |
| photos + ocr/uncommon | d2f3f21f | Harvest Hand // Scrounged Scythe INR 265 | Ugin's Construct FRF 164 | wrong card (DFC) |
| photos | c4db37a8 | Traveler's Amulet INR 273 | Fiend Hunter INR 22 | wrong card |
| scans_no_bg | 0a0ebcb7 | Phelia PMH3 40s (promo) | Phelia MH3 40 | right card, wrong printing |

### Double-faced cards are the main failure mode

- MTGJSON gives **each face of a double-faced card (DFC) its own UUID, and both rows share the same image URL**. In `MobileCache.sqlite`, 4,531 of 109,018 rows share their URL with another row. The ground-truth label for a DFC photo is therefore ambiguous: the filename names one face, and the pHash index holds two identical hashes.
- In `1_photos_single_card`, single-faced cards score **178/180 (98.9 %)**. DFCs score **10/20 (50 %)**: one miss is the other face of the right card, and nine are genuinely wrong cards.
- If the benchmark counted "same card, other face" as correct, photos would be 189/200 and the total 247/260.

## Implications for "Define the accuracy benchmark"

- **Tolerance:** results are deterministic on this machine, so a tight ratchet (exact count, zero tolerance) is feasible locally. Determinism across machines (CI runner vs. laptop, different CPU/BLAS) has **not** been tested.
- **CI cost:** about 3 minutes of compute for all 260 images (probably slower on a 2- or 4-core GitHub runner, untested), plus the dataset pull. Running the full set in CI looks affordable.
- **Correctness definition:** decide how DFC faces count (UUID-exact, or any UUID sharing the image/scryfall id) and whether "right card, wrong printing" is a separate metric.
- **Category coverage:** `6_benchmark/*` scores 55/56 and adds no signal not already in `1_photos_single_card` (all its UUIDs also appear there). Several categories have **no UUID ground truth at all**, which matters for "Decide the benchmark dataset's layout and naming".

## Side observations

- `CLAUDE.md` says the DB holds "~1M+ cards", but `mobilecache` has **109,018 rows**.
- The CLI writes `raw_texts`/`raw_scores`/`raw_boxes` twice per segment (`debug` and `debug_info`).
- The OCR fallback goes to `print` rather than a logger and isn't recorded in the JSON output, so a benchmark can't currently count fallbacks from the results file.

## Artifacts

`docs/research/accuracy-recon/`: `run1/` and `run2/` (per-category JSON outputs, CLI logs, `times.txt`), `run.sh` (runner) and `analyse.py` (summary script; expects the outputs under `/tmp/acc6`).
