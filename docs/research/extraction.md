# Extraction prototype: current, candidate-driven, and with positions

Findings for the ticket "Prototype extraction: current, candidate-driven, and with positions" (#197) of the map #189. Throwaway branch `research/extraction`, never merged. Scripts: `scripts/extraction.py` (cases and report) and `scripts/extraction_candidate.py` (the candidate-driven extraction).

## Setup

- **No OCR rerun.** The OCR tokens and boxes saved by #196 (`output/ocr_compare/<run>/readings.json`) are replayed through each extraction: `v6small-own` (PP-OCRv6 small with its own detection settings, the prototype's model), `v5det-v6tinyrec` (the fast option) and `v5-mobile` (today's model, for reference).
- **Sample**: `scripts/ocr_sample.json` from #196, 1,291 card images: the benchmark's 275, 100 reference images per frame (1993, 1997, 2003, 2015, future) and per common layout (transform, adventure, modal_dfc, saga, split), and the 16 uncommon-layout photos. Tune and confirm halves as listed there.
- **What a card shows**: the printed values from `printed_rules.py` (#201). A detail the rules mark as not shown, or not derivable, is left out.
- **Three measures**:
  1. **Benchmark, end to end**: each benchmark card image's 20 pHash candidates, shortlisted again from the saved card images of #195's run, ranked by each variant; exact printing / card count / card name per query image. A card image whose correct printing isn't a candidate keeps #195's result, since no extraction can change it.
  2. **Same name first**: the correct printing scores strictly above every other printing of its name (shuffled with seed 189), leaving out names with one printing. 1,112 card images.
  3. **Field score of the correct printing**: the mean 0-100 score per field, over the cards that show the field.

### The variants

| Variant | Extraction | Values compared | Average |
|---|---|---|---|
| current, catalog values, all | today's `ocr_extraction.py` (the reading saved with the tokens) | the catalog's | over all 8 fields, as today's `rank` |
| current, printed values, all / shown | today's | printed (rules) | all 8 fields (a field not shown scores 0) / only the fields the candidate shows |
| candidate, all / shown | for each candidate, whether each detail it shows appears in the raw text | printed | as above |
| positions, all / shown | as candidate, looking for the collector details on the bottom two lines only, and for power/toughness in the rightmost token above the bottom line | printed | as above |

Every printed-values variant breaks ties the same way (see Finding 1). Weights are today's: name 2, artist 0.5, the rest 1.

### Candidate-driven extraction, in short

For one candidate, with the tokens of both crops (`O`/`o` next to a digit read as `0` first):

- **name**: `fuzz.ratio` against the name crop's text, the best of the card's names: each face of `A // B`, and the flavor name when there is one.
- **set code**: an upper-case run of letters and digits equal to the code, or the code followed by a two-letter language (`THBDE`, `WAREN`).
- **collector number, year**: a run of digits equal to the value, leading zeros dropped. `14` isn't found in `149`, because whole runs are compared.
- **base set size**: the run right after the number, or glued to it when the slash is lost or read as `1` (`751350`, `7511350`).
- **artist**: `fuzz.partial_ratio` against the collector information's text.
- **power/toughness**: `p/t` with the slash read as `/ | \ I l 1 )`, only for candidates with a power.

There is no blocklist, and no rule for flavor text: flavor text and copyright words only matter when they hold a candidate's value, and they hold it for every candidate alike.

## Results

### PP-OCRv6 small, own detection settings (the prototype's model)

| Variant | Benchmark exact / count / name | Same name first | tune | confirm | name | set code | number | size | artist | year | P/T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| current, catalog values, all (today) | 255 / 257 / 257 | 643 | 317 | 336 | 82.6 | 74.9 | 81.5 | 66.7 | 71.0 | 82.1 | 92.6 |
| current, printed values, all | 255 / 257 / 257 | 643 | 321 | 332 | 82.6 | 77.9 | 84.4 | 82.9 | 71.0 | 84.5 | 92.6 |
| current, printed values, shown | 254 / 257 / 257 | 672 | 332 | 349 | | | | | | | |
| candidate, all | 255 / 257 / 257 | 674 | 333 | 351 | 84.2 | 86.4 | 86.8 | 83.5 | 93.0 | 82.2 | 96.5 |
| **candidate, shown** | **255 / 257 / 257** | **702** | **347** | **365** | | | | | | | |
| positions, all | 255 / 257 / 257 | 671 | 329 | 352 | 84.2 | 86.4 | 86.7 | 83.3 | 92.8 | 82.1 | 95.1 |
| positions, shown | 255 / 257 / 257 | 702 | 344 | 368 | | | | | | | |

Field scores don't depend on the average, so they are given once per extraction. Same name first, out of 1,112 (561 tune, 563 confirm).

Per group, same name first (v6 small):

| Variant | benchmark | 1993 | 1997 | 2003 | 2015 | future | transform | adventure | modal_dfc | saga | split | uncommon |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| today | 196 | 20 | 48 | 51 | 69 | 69 | 47 | 35 | 49 | 45 | 0 | 14 |
| candidate, shown | 196 | 41 | 55 | 60 | 72 | 70 | 52 | 37 | 57 | 48 | 0 | 14 |
| positions, shown | 196 | 41 | 55 | 60 | 72 | 70 | 53 | 36 | 57 | 48 | 0 | 14 |
| out of | 200 | 90 | 91 | 87 | 93 | 88 | 88 | 100 | 91 | 70 | 100 | 14 |
| same details as another printing | 34 | 64 | 28 | 26 | 19 | 22 | 35 | 32 | 35 | 29 | 93 | 2 |

The last row counts the cards that show exactly the same details as another printing of their name: OCR alone can't separate them, whatever the extraction.

### v5_mobile detection + v6 tiny recognition (the fast option)

| Variant | Benchmark exact / count / name | Same name first | tune | confirm |
|---|---|---|---|---|
| today | 253 / 257 / 256 | 601 | 298 | 309 |
| **candidate, shown** | **255 / 257 / 257** | **681** | **338** | **352** |
| positions, shown | 255 / 257 / 257 | 681 | 335 | 355 |

Candidate-driven extraction recovers both benchmark images the fast option loses today, including `1_photos_single_card/b86a6181…`, the card name #196 lost to the `01997` year. With it, the fast option no longer loses a card name or an exact printing against today (253 / 257 / 256 → 255 / 257 / 257), and none is worse. It still ranks same-name printings below v6 small (681 against 702).

### v5_mobile (today's model), for reference

Today 254 / 257 / 257, 612 same name first; candidate, shown 254 / 257 / 257 (one image gained, one lost on the tune half), 675; positions, shown 255 / 257 / 257, 676.

### Code

| | Today (`ocr_extraction.py`) | Candidate-driven (`extraction_candidate.py`) |
|---|---|---|
| Hand-kept rules: blocklist entries + regexes | 28 + 22 | 0 + 5 |
| Highest function complexity (`uvx radon cc`) | 12 | 38 (one prototype function; see Assumptions) |
| Lines (information only) | 484 | 120 |
| Scoring time per card image, 20 candidates | 0.16 ms | 0.41 ms |

## Findings

1. **Scoring against printed values needs a tie-break, and today's code has one by accident.** Many printings show exactly the same details as another: The List and promos reprint the original's set code, number and size, and suffix variants (`254★`, `71z`, `240a`) print the same number as the plain one. Scored against printed values they tie, and pHash order then picks, often wrongly: without a tie-break, v5_mobile fell from 254 to 239 on the benchmark. Today's code breaks these ties by comparing with the catalog's values (`PLST`, `WAR-47`, `5075`, `254★`), which never match the text, so the plain printing wins. The prototype makes this explicit: among equal scores, prefer the printing that prints its own details (not in `printed_sources`) and whose number has no suffix.
2. **Candidate-driven extraction beats today's code on every measure and every model.** With v6 small, the benchmark is unchanged (255 / 257 / 257, no image changed) and same name first rises from 643 to 702 (tune 317 → 347, confirm 336 → 365). Old frames gain the most (1993: 20 → 41, 2003: 51 → 60), because there is no longer a step that guesses the field before comparing. The largest field gain is the artist (71 → 93), where today's blocklist and token filter drop or add words.
3. **Count only the details the candidate shows ("shown") rather than averaging over all 8 fields.** It adds 28 same-name first with v6 small (674 → 702) and doesn't change the benchmark. It has one known weakness: when OCR misses details, a printing that shows fewer of them can win. On `6_photos…/bd5fdd3b…` (Jalum Tome, CMR), the copyright line wasn't read and the artist only partly, so the Antiquities printing, which shows only name and artist, scores higher. The benchmark still picks the right printing there: the Antiquities printing isn't among its pHash candidates.
4. **Positions add nothing consistent: drop them**, as the ticket asked unless they beat the candidate-driven variant. With v6 small: 702 both overall, tune 344 against 347, confirm 368 against 365, benchmark identical. The first rule tried (power/toughness in the rightmost token) failed outright, because the copyright line reaches further right than the P/T box. Whole-run digit comparison already handles the main risk positions were meant to cover (`14` inside `149`).
5. **Flavor names**: candidates match the flavor name as well as the card name, so the 2.9% of printings whose large name is a flavor name aren't scored against a name they don't show. The sample holds too few of them to measure.
6. **Split cards stay at 0 / 100** on same name first for every variant. Their names are sideways, outside the name crop, and 93 of 100 show the same details as another printing anyway, so a rotated crop would help the name (telling split cards apart from other cards) but not the printing.

## What this changes

- **Reading** (`GLOSSARY.md`): today, "the details read off one card image before they are compared with any printing". Candidate-driven, nothing is extracted before comparing: the reading becomes the text and its boxes OCR finds in the name and collector information crops, and field scores come from looking for each candidate's values in it. `Reading`'s 8 fields and `ocr_extraction.py` go; `FieldScore.read` can no longer hold "what was read" per field, only whether and how closely the value was found.
- **Scoring needs the printed values**: the five fields #201 defined (`printedSetCode`, `printedNumber`, `printedBaseSetSize`, `printedYear`, plus `layout`), whether each is shown, the flavor name, and whether the printing prints its own details with a plain number (for the tie-break). This settles the fog item "Frame and layout in the catalog" in favour of adding them, except `layout`, which the prototype didn't use.
- **Crops**: no change needed for upright cards. Rotating the name crop for split cards, battles and planes stays open (Finding 6).

## Assumptions and things not measured

- **A1. Tuning.** The candidate-driven rules were written from #193 and #201's findings, not tuned. Three decisions were made on results that include the confirm half: the tie-break (from the benchmark losses, which span both halves), the P/T position rule, and "shown" against "all". Each holds on both halves.
- **A2. Complexity.** The prototype's `candidate_field_scores` is one function holding every field (complexity 38). Split by field, as a production version would be, it should be well below today's 12; not measured.
- **A3. The 35 sets the rules get wrong** (#201) score as if the rules were right. Jumpstart (`JMP 174`) is one: the rules say no base set size is printed. Not counted separately.
- **A4. Benchmark card images are from #195's segmentation** (`conf` 0.05), as in #196: today's numbers here are 255 / 257 / 257 with v6 small and 254 / 257 / 257 with v5_mobile, not the baseline's 253 / 256 / 256.
- **A5. Name scoring** uses `fuzz.ratio` against the whole name crop's text. `partial_ratio` wasn't tried; short names inside longer ones (`Shock` in `Shocker`) would score 100 with it.
- **A6. Not measured**: the time to load the printed values into the catalog, and the uncommon layouts beyond the 16 photos.

## How to rerun

```bash
# on research/extraction, with output/ocr_compare/<run>/readings.json from #196
uv run python scripts/extraction.py prepare
uv run python scripts/extraction.py report v6small-own v5det-v6tinyrec v5-mobile [--subset tune|confirm]
```
