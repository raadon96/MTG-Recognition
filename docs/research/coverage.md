# Scoring prototype: coverage of the reading, and the weights

Findings for "Prototype scoring: coverage of the reading, and the weights" (#234). Throwaway branch `research/coverage`, never merged. Script: `scripts/coverage_scoring.py`. Raw output: `output/coverage/report.md` (not committed; rerun to get it).

**Recommendation: add coverage (count digit runs, weight 1) and keep today's weights.** This is variant 1. Variant 3 isn't clearly better, and it loses a benchmark image. The maintainer reviews this before deciding; adopting it is a follow-up ticket.

## Setup

- **Readings with the current configuration.** dev's engine as it is after #217, #218 and #219: `read_card` (PP-OCRv5_mobile detection, PP-OCRv6 tiny recognition, 960 / max, 64 px minimum crop side) and `scoring.rank`. OCR runs once (`prepare`). Every variant is then replayed on the saved text (`report`).
- **Sample**: #196's 1,291 card images (`scripts/ocr_sample.json`), with tune and confirm halves as listed there. Each card image's competitors are its 20 pHash candidates plus every printing of its name.
- **Benchmark, end to end**: each query image is segmented, shortlisted and read as `Pipeline.recognize` does, then ranked by each variant. Today's variant reproduces the baseline exactly (255 / 257 / 257), so the replay matches the real pipeline.
- **Measures**
  - **Same name first** (#197): the right printing scores strictly above every other printing of its name. Names with one printing are left out: 1,112 cards, 561 tune, 563 confirm.
  - **First among candidates**: the right printing scores strictly above every other competitor, pHash candidates included, out of 656 tune / 658 confirm. The search tunes on this measure (see Finding 4).
  - **Benchmark**: exact printing / card count / same card name, and the images that change.
- **Coverage**: the share of digit runs in the collector information that equal one of the candidate's shown values (number, base set size, year). It also counts a number and size glued together (`751350`), and an open `1993` for any printing that shows a year, as `rank` does (#217). Coverage is left out when the text holds no run of digits, which is true for 72 of the 1,291 card images. Four forms were tried: each run counted once or weighted by its length, with or without power/toughness digits counting as explained for creatures.
- **Search**: each of the 7 weights from {0, 0.5, 1, 2}, and coverage from {0.5, 1, 2}. The search runs on the tune half only. Among equal results it picks the vector closest to today's weights. The benchmark is only a final check.

## Results

| Variant | Benchmark | Changed images | First among candidates, tune / confirm | Same name first, all / tune / confirm |
|---|---|---|---|---|
| today: name 2, artist 0.5, the rest 1 | 255 / 257 / 257 | 0 | 401 / 425 | 631 / 307 / 333 |
| **1: coverage ×1 (runs), today's weights** | **255 / 257 / 257** | **0** | **410 / 433** | **650 / 318 / 341** |
| 1, coverage ×2 (best on tune) | 255 / 257 / 257 | 0 | 412 / 431 | 652 / 320 / 341 |
| 2: weights only: artist 2, year 2, the rest as today | 255 / 257 / 257 | 0 | 407 / 431 | 642 / 313 / 338 |
| 3: set code 0.5, artist 1, P/T 2, coverage 2, the rest as today | 254 / 257 / 257 | 1 lost | 415 / 437 | 656 / 321 / 344 |

Same name first, cards gained / lost against today:

| Variant | tune | confirm |
|---|---|---|
| 1: coverage ×1 | +11 / −0 | +12 / −4 |
| 1, coverage ×2 | +13 / −0 | +15 / −7 |
| 2: weights only | +6 / −0 | +6 / −1 |
| 3: both | +14 / −0 | +17 / −6 |

Coverage form, with today's weights, on tune (first among candidates / same name first):

| Form | ×0.5 | ×1 | ×2 |
|---|---|---|---|
| runs | 411 / 317 | 410 / 318 | 412 / 320 |
| runs + P/T | 409 / 315 | 410 / 316 | 412 / 318 |
| length | 411 / 317 | 411 / 319 | 411 / 319 |
| length + P/T | 409 / 315 | 411 / 317 | 411 / 317 |

Per group, same name first:

| Variant | benchmark | 1993 | 1997 | 2003 | 2015 | future | adventure | modal_dfc | saga | split | transform | uncommon |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| today | 180 | 33 | 50 | 54 | 67 | 70 | 30 | 51 | 42 | 0 | 42 | 12 |
| 1: coverage ×1 | 183 | 33 | 54 | 55 | 67 | 72 | 38 | 52 | 42 | 0 | 42 | 12 |
| 2: weights only | 183 | 32 | 52 | 56 | 67 | 71 | 32 | 52 | 42 | 0 | 43 | 12 |
| 3: both | 184 | 33 | 54 | 56 | 68 | 72 | 40 | 52 | 42 | 0 | 43 | 12 |
| out of | 200 | 90 | 91 | 87 | 93 | 88 | 100 | 91 | 70 | 100 | 88 | 14 |

The same-name numbers aren't comparable with #197's 702: that used PP-OCRv6 small and the prototype's own scoring.

## Findings

1. **Coverage fixes the weakness ADR 0008 records, and costs nothing on the benchmark.** With today's weights it adds 19 same-name first (631 → 650) and 17 first among candidates. The gains hold on both halves, and no benchmark image changes. The 23 cards it gains are the expected case: a printing that shows only a name and artist (CED, FBB, ATQ, LEA) or no number (TMP) beat the right one when OCR missed something. It includes a Commander Legends card against Antiquities on `6_photos…`, the same pattern as the issue's Jalum Tome example. For example, `266/281` and `AFR*EN` were read, yet CED won with 90.0 against AFR's 84.2. With coverage, CED drops to 64.3 and AFR wins with 81.9.
2. **Its 4 losses (confirm half) are wrong printed values in the catalog, not flaws in coverage.** Three are SDCC promos (PSDC, PS14, PS15). The card prints the parent set's `172/249`, `134/269` or `135/272`, but the catalog says it shows no number or no size. The fourth is the French FBB Counterspell, which prints `©1994` though the catalog says it shows no year. Today these printings win only because they "show" fewer details. Coverage counts the digits they don't explain against them. Fixing the printed-detail rules (#201) for these sets would turn these losses into gains.
3. **The coverage form hardly matters.** Each of the four forms, at each weight, lies within 3 cards of the others on tune. Counting runs is the simplest form. Explaining P/T digits never helps. ×2 is 2 cards better than ×1 on tune but loses 3 more cards on confirm, so ×1 is preferred: the round value, and the weight the issue proposed.
4. **The weights alone add little, and the grid has no clear optimum.** The best vector without coverage raises artist and year to 2 (+6 tune, +6 confirm). Fifteen vectors tie on tune. Most of the best ones keep the number, size and set code near today's values. Before tuning on the strict measure, a first attempt credited ties to the right printing. Every reference image is its own first pHash candidate, so that search set the number and size to 0 and lost 2 benchmark images. Tuning on same name first alone would be blind in the same way to the name's job, which is telling different cards apart.
5. **Weights and coverage together (variant 3) isn't clearly better, and it overfits.** It gains 3–6 cards over variant 1 on each measure. But it loses a benchmark image (`5_scans_multiple_cards/with_inner_gaps_with_border_gaps.jpg`) and loses 6 cards on confirm. It also sets power/toughness to 2, which the issue argued should count less, since every printing of a card has the same P/T. 42 vectors tie for best on tune. The issue's rule applies: the simpler change, variant 1, is preferred.

## Open points

- **Misread numbers** (`300` for `308`) are unexplained for every candidate, the right one included; not measured separately.
- **Digits that belong to no detail** (P/T, flavor text) lower every candidate's coverage alike. Counting P/T digits as explained didn't help (Finding 3).
- **Catalog printed values for SDCC promos and FBB** (Finding 2): a separate fix to #201's rules.
- 72 card images hold no run of digits. Coverage leaves them out, as intended, so cards from before the year was printed keep today's scores.

## How to rerun

```bash
# on research/coverage; the 16 uncommon-layout photos from research/ocr-settings'
# output/ocr_compare/uncommon_cards are copied to the same path here
uv run python scripts/coverage_scoring.py prepare   # OCR once, about 2 min
uv run python scripts/coverage_scoring.py report > output/coverage/report.md
```
