# What each printing prints, derived from AllPrintings

Research for issue #201 (map #189). Question: for each printing, what does the card print for the details scoring uses (set code, collector number, base set size, copyright year), and can it be derived from `AllPrintings.sqlite`? Which printings print nothing for a detail? How often do the derived values match what OCR read? What would the catalog need, and at what size?

Researched 2026-11-08 against MTGJSON release `5.3.0+20261103`, the release the committed catalog is built from. Starting point: [card-details.md](card-details.md) (#193).

## Sources and method

Primary sources:

- **AllPrintings**: `data/reference/mtgjson/AllPrintings.sqlite`, tables `cards` and `sets`, opened read-only.
- **MTGJSON docs**, read 2026-11-08: [card (set) model](https://mtgjson.com/data-models/card/card-set/) and [set model](https://mtgjson.com/data-models/set/). Quoted: `originalReleaseDate` is "the original release date ... for a promotional card printed outside of a cycle window, such as Secret Lair Drop promotions"; `baseSetSize` "will default to totalSetSize if not available"; `parentCode` is "the parent printing set code for set variations like promotions, guild kits, etc."; `number` "can have a variety of numbers, letters and/or symbols for promotional qualities". The docs list no values for `promoTypes` or set `type`; the values used below are the ones in the file.
- **Card images**: the reference images in `data/reference/images` (Scryfall `normal` scans, one per catalog row), read by the production OCR model.
- **The code**: `jobs/sync.py` `build_catalog` and `extract_year`, `reference/catalog.py`, `engine/ocr/scoring.py` and `engine/ocr/ocr_extraction.py` on `95b7acb`.

Scripts, all throwaway, in `docs/research/printed-details/`:

- `rules.py`: the rules, as one function per printing on plain rows. `printed_sources` finds whose details a card prints; `printed_details` returns the printed set code, number, base set size and year, or `None` when nothing is printed or the value can't be derived, with the rule that decided each.
- `survey.py`: a second, wider set of readings. For each paper set and frame, up to 4 normal-layout printings with a reference image (seed 201): 2,734 cards over 760 set-and-frame pairs. The bottom 15% of the card is upscaled 2x and read by `load_paddleocr()` with `text_det_thresh=0.05`, as #193 read whole cards. Raw output: `survey.json`.
- `check.py`: compares, for each reading and detail, the derived value and today's catalog value with the OCR tokens. Readings: the 540 of #193 (`card-details/positions.json`, tokens whose box starts at 0.85 of the card's height or lower, plus tokens read on a rotated card) and the 2,734 of the survey. Output: `check.txt`.
- `coverage.py`: how many catalog rows each rule decides, and the size of a catalog with the added fields. Output: `coverage.txt`, `coverage.json`.

Two match measures, both reported:

- **token**: the tokens hold the value. Normalised: `O`/`o` next to a digit read as `0`; leading zeros dropped; the number must stand as a collector number (followed by `/`, three digits or more, a leading zero, a rarity letter next to it, alone in its token, or after `Coast`/`Inc`/`LLC`); the size may follow the number with the slash read as `1`, `I`, `l` or `)` or lost (`751350` for `75/350`); the year may be either end of a range.
- **extracted**: the production extractor returns exactly the value (a field score of 100): `extract_possible_set_codes`, `extract_collector_number_and_base_set_size`, `extract_copyright_year`, run on the same tokens without the flavor text filter.

"Catalog value" is the value scoring compares against today: `setCode`, `number`, `baseSetSize` (the set's), and `setReleaseYear` (the year of the set's `releaseDate`, `sync.py` `extract_year`). Scoring compares the reading with these strings by fuzzy ratio (`scoring.py` `FIELDS`); the number is compared without leading zeros.

## Summary

- For about 98% of paper printings the rules derive each detail, or that it isn't printed, from AllPrintings alone. The rules use `setCode`, `number`, `frameVersion`, `originalReleaseDate`, `promoTypes`, the set's `releaseDate`, `type`, `parentCode`, `baseSetSize`, and the name, artist and number of other printings. No hand-kept table is needed for the main rules.
- On the 540 readings of #193, the derived values match what OCR read far more often than today's catalog values: set code 345/347 against 250, number 464/490 against 346, base set size 217/251 against 142, year 477/530 against 415. On the 2,734 survey readings: 1,080/1,123 against 776, 1,869/2,148 against 1,464, 1,040/1,317 against 719, 2,209/2,697 against 2,074.
- Where the rules differ from the catalog, OCR almost always shows the rule's value: on the 540 readings, 95 of 95 set codes, 118 of 123 numbers, 75 of 81 sizes and 62 of 69 years, against 0 for the catalog's value.
- The rules leave out a detail on 34% of paper printings for the set code, 9% for the number, 48% for the base set size and 2% for the year. OCR found the catalog's value on almost none of these readings, except on the promo sets the rules call not derivable.
- What can't be derived: the printed code of event promo sets (`PRM`, `FNM`, `H19`; 815 paper rows in 91 sets); the details of old-frame promos (898 paper rows); and 35 sets (about 5,500 paper rows, 5.2%) whose printed size, number or year differs from what the fields give (listed below). The rules leave the first two out; the 35 sets would need a hand-kept table.
- Adding the four printed fields and `layout` costs 2.3 MB (+5.8%) on a VACUUMed catalog, or 1.6 MB when only values that differ from the catalog's are stored. The committed catalog (43.8 MB) isn't VACUUMed; VACUUM alone saves 3.6 MB, so the catalog with every added field (42.5 MB) is still smaller than today's file.

## 1. The rules

### Whose details a card prints (`printed_sources`)

Checked in this order; the first that applies decides.

| Source | Rule | Rows (paper) |
| --- | --- | --- |
| The List | `setCode = 'PLST'`: the set code and number in the number, `HOP-128` → `HOP`, `128`. Every prefix is a set in AllPrintings. | 5,764 (5,764) |
| Parent set's card | The set has a `parentCode`, and the parent set has a card with the same name and the same number digits: `PKHM 214p` → `KHM 214`, `PSOI 245s` → `SOI 245`, `OC16 50` → `C16 50`. | 7,959 (7,901) |
| World Championship decks | `PTC`, `WC97` to `WC04`: the latest expansion or core set printing released before the deck with the same name, artist and number digits (`WC01 jt191sb` → `7ED 191`). Found for 1,216 of 1,244 rows. | 1,216 (1,216) |
| Own set | Every other card. | |

A parent link alone is not enough: event promo sets with a parent (`PPP1`, `PSS2`, `PRWK`) number their cards on their own (`001/005 P Promo Pack`) and print `PRM`. The name-and-number match separates the two.

The date that decides the format is the source set's `releaseDate` for the first three, and `originalReleaseDate` when present, else the set's `releaseDate`, for the own set.

### Rules per detail

Shares are of the 115,875 catalog rows (106,466 paper). "Match" is token match on the readings the rule decided: the 540 of #193, then the survey, with the catalog value's match in brackets.

| Detail | Rule | Rows (paper) | Differs from catalog | Match, #193 | Match, survey |
| --- | --- | --- | --- | --- | --- |
| Set code | Printed only on the 2015 frame (`frameVersion = '2015'`). Value: the source's code. | 75,580 (70,764) | 11,215 | 345/347 (250) | 1,080/1,123 (776) |
| | of which own set | 64,365 | 0 | 250/252 | 773/814 |
| | of which parent set's card: the parent's code (`POGW` → `OGW`) | 7,231 | 7,231 | 58/58 (0) | 303/305 (3) |
| | of which The List: the original set's code | 3,864 | 3,864 | 32/32 (0) | 4/4 (0) |
| | of which oversized (`setCode = 'O' + parentCode`): the parent's code (`OPCA` → `PCA`) | 120 | 120 | 5/5 (0) | |
| Collector number | Printed from Exodus (source date ≥ 1998-06-15), or on a retro reprint. Value: the first run of digits, without leading zeros (`214p` → `214`, `0254` → `254`, `58★` → `58`). | 104,976 (96,514) | 17,128 | 464/490 (346) | 1,869/2,148 (1,464) |
| | of which own set | 88,112 | 3,587 (suffixes only) | 355/377 (344) | 1,198/1,356 (1,158) |
| | of which parent set's card | 7,451 | 7,063 | 59/60 (1) | 388/449 (101) |
| | of which The List | 5,551 | 5,551 | 46/49 (0) | 13/14 (0) |
| | of which promo set's own numbering (2015 frame) | 3,174 | 239 | 1/1 | 251/305 (205) |
| | of which World Championship | 688 | 688 | 3/3 (0) | 19/24 (0) |
| Base set size | Printed when the number is, the date is before 2023-04-01, the card isn't a retro reprint, and, from 2019-10-01, the number isn't above the size. Value: the source set's `baseSetSize`. | 59,563 (54,991) | 10,512 | 217/251 (142) | 1,040/1,317 (719) |
| | of which own set | 49,038 | 0 | 142/170 | 714/917 |
| | of which The List | 4,988 | 4,988 | 39/43 (0) | 10/11 (0) |
| | of which parent set's card (`214p` → `/285`, catalog 1) | 4,849 | 4,836 | 34/35 (0) | 299/365 (5) |
| | of which World Championship | 688 | 688 | 2/3 (0) | 17/24 (0) |
| Copyright year | Printed from Legends (date ≥ 1994-06-01), and on retro reprints. Value: the year of the date above. | 113,558 (104,149) | 14,436 | 477/530 (415) | 2,209/2,697 (2,074) |
| | of which own set, no `originalReleaseDate` | 89,448 | 0 | 348/388 | 1,587/1,897 |
| | of which `originalReleaseDate` (Secret Lair, promos) | 9,177 | 8,041 | 18/25 (4) | 213/288 (104) |
| | of which parent set's card | 7,959 | 37 | 59/63 (59) | 375/460 (370) |
| | of which The List: the original set's year | 5,764 | 5,465 | 47/49 (3) | 12/16 (1) |
| | of which World Championship | 1,210 | 893 | 5/5 (1) | 22/36 (12) |

"Retro reprint": `frameVersion` 1993, 1997 or 2003 and a date from Magic 2015 (2014-07-18) on. Time Spiral's timeshifted cards (2006, 1997 frame) still print `70/121`, so the boundary is Magic 2015, not Eighth Edition.

The dates come from AllPrintings set release dates, checked on the survey:

- **Base set size until 2023-04-01**: Dominaria Remastered (2023-01-13) and Phyrexia: All Will Be One (2023-02-10) print `114/271`; March of the Machine (2023-04-21) and every later set read prints the rarity-first number alone (`R 0293`). No paper set falls between them. #193 used 2023-01-01, which wrongly left out DMR and ONE.
- **Extras print the number alone from 2019-10-01**: Core Set 2020 (2019-07-12) prints `312/280`; Throne of Eldraine (2019-10-04) prints `286` for a card above 269, and so do Commander decks from 2020 for reprints numbered above their new cards (`AFC 267`, `C21 302`). #193 called this "Commander and Jumpstart print no size"; the survey shows it is the number above the size that decides: `C20 271/322`, `AFC 024/062`, `MIC 026/038` print the size.
- **Copyright year from Legends**: Alpha, Beta, Unlimited, Arabian Nights, Antiquities, Revised and the Collectors' Editions print `Illus. © Artist` with no year. Legends, The Dark and Summer Magic print `Illus. © 1994 Artist`, which the extractor reads as the year. Fallen Empires (1994-11) is the first with `© 1994 Wizards of the Coast`.

### Printings that print nothing, or whose value can't be derived

Scoring should leave these details out rather than score them 0. "Found anyway" counts readings where OCR found such a detail all the same.

| Detail | Left out because | Rows (paper) | Found anyway, #193 / survey |
| --- | --- | --- | --- |
| Set code | 2003, 1997, 1993 or future frame | 36,149 (33,915) | 0 / 0 |
| | playtest card (`promoTypes` has `playtest`: Mystery Booster playtest cards print `TEST CARD`) | 972 (972) | 0 / 0 |
| | **not derivable**: promo set with its own numbering (`type = 'promo'` and not a parent set's card, or `keyruneCode = 'PMEI'` and type `funny` or `memorabilia`). They print `PRM`, `FNM`, `H19`, `J21`, which no field holds. | 3,174 (815; 91 sets) | 1 / 294 |
| Collector number | before Exodus (1998-06-15) | 8,092 (8,080) | 0 / 1 |
| | playtest card | 973 (973) | 1 / 2 |
| | **not derivable**: old-frame promo with its own numbering. They print another set's number (`FNM 45/350`), an event mark (`PT 2008`, `C10 CHaMPS`), their own number and code (`4/12 F10`), or nothing. | 1,833 (898; 122 sets) | 2 / 158 |
| Base set size | no number printed (the three above) | 10,899 (9,952) | 0 / 36 |
| | from April 2023 | 34,041 (32,538) | 0 / 0 |
| | number above the base set size, from October 2019 | 9,379 (7,534) | 0 / 7 |
| | retro reprint | 1,121 (1,087) | 0 / 1 |
| | **not derivable**: promo set with its own numbering. They print a size of their own (`001/003`, `004/005`) that `baseSetSize` holds on about half the readings (97 of 182). | 872 (364) | 0 / 139 |
| Copyright year | before Legends (1994-06-01) | 2,317 (2,317) | 0 / 2 |

Per the #193 table of what's printed: the leave-outs agree with it, with the two corrections above (the April 2023 boundary, and extras rather than supplemental sets).

### Exceptions the rules get wrong

Found on the readings; none is derivable from AllPrintings fields. Each would need a hand-kept entry; the rules leave them as they are.

| Set | What it prints | What the rules give | Paper rows |
| --- | --- | --- | --- |
| TSR | year 2020 | 2021 | 414 |
| VIS | year 1996 | 1997 | 167 |
| RQS, PLGM | year 1995 | 1996 | 67 |
| ULST, CC2, MPS | the year before (2020, 2021, 2017) | the set's year | 125 |
| Secret Lair drops, some event promos | the year before `originalReleaseDate` on some cards (`SLD 888`: 2024 against 2025) | `originalReleaseDate`'s year | part of 8,041 |
| WC03, WC04 | the deck's year | the original set's year | 225 |
| FBB | `1994 Wizards` | not printed | 307 |
| DBL, J22, SCD | the number alone (`338`) | the number and `baseSetSize` | 1,820 |
| 40K, GN3, BOT, PSAL, Theros hero decks (THP1-3, TFTH, TBTH, TDAG) | another size (`050/168`, `017/136`, `001/015`, `16/60`, `3/9`) than `baseSetSize` | `baseSetSize` | 1,584 |
| 30A, 2015 frame | rarity-first number alone (`R0099`) in 2022 | number and size | 297 |
| P02, ATH, BRB, S00 | no number | number and size | 406 |
| PS14-PS19 (SDCC) | the original card's code, number and size | own code, number and size | 37 |
| MP2 | `MPS` | `MP2` | 54 |

Sizing a hand-kept table for these: 35 set entries (one value or one rule each) cover about 5,500 paper rows (5.2%), plus the part of Secret Lair printed a year early, which would need per-card entries. The map's rules discourage hand-kept lists; the cheaper option is to leave these details as the rules give them, since scoring loses only the one detail on these cards.

### Mismatches on the 540 readings of #193

Where the rules give a value and OCR didn't read it. Most are OCR errors or the measurement, not the rule:

- **Set code (2 of 347)**: J22 prints no set code on the line OCR read; 30A is read `JOA`. Both OCR.
- **Number (26 of 490)**: older frames read badly (`6o/350` for 66, `b4/156`, `IB3/350`, the line cut off); split cards in the 2003 frame or older print sideways, and `ocr_positions.py` kept rotated tokens only when they matched the catalog, so their collector information is mostly missing (10 readings); 30A retro reprints read as noise (3); one flip and one leveler card read without the number. No rule error.
- **Base set size (34 of 251)**: the same OCR and sideways cases (27); rule errors: DBL (3) and J22 print the number alone, BOT prints `001/015`, PSAL prints `16/60` (3).
- **Year (53 of 530)**: OCR (`1906`, `199o`, `O1097`, the range start only, sideways split and planar cards: 34); rule errors: TSR (4), VIS (1), Secret Lair a year early (3), playtest cards printed 2019 for a 2021 set (2), 30A retro noise.

Per #193 group (derived value read / readings where the rules say it's printed; catalog value read / all readings):

| Group | Set code | Number | Base set size | Year |
| --- | --- | --- | --- | --- |
| frame 1993 | - ; 0/20 | - ; 0/20 | - ; 0/20 | 7/10 ; 6/20 |
| frame 1997, before Exodus | - ; 0/20 | - ; 0/20 | - ; 0/20 | 14/20 ; 13/20 |
| frame 1997, 1998-2003 | - ; 0/20 | 17/20 ; 15/20 | 17/20 ; 16/20 | 19/20 ; 18/20 |
| frame 1997, retro reprint | - ; 0/20 | 15/19 ; 14/20 | - ; 0/20 | 12/20 ; 8/20 |
| frame 2003 | - ; 0/20 | 15/20 ; 15/20 | 15/20 ; 15/20 | 19/20 ; 19/20 |
| frame 2015, 2014-2022 | 18/20 ; 15/20 | 20/20 ; 18/20 | 14/17 ; 11/20 | 17/20 ; 17/20 |
| frame 2015, 2023+ | 19/19 ; 18/20 | 19/19 ; 18/20 | - ; 0/20 | 20/20 ; 20/20 |
| frame 2015, borderless | 20/20 ; 20/20 | 20/20 ; 19/20 | - ; 0/20 | 16/20 ; 12/20 |
| frame future | - ; 0/20 | 19/19 ; 19/20 | 8/8 ; 7/20 | 19/20 ; 18/20 |
| The List | 17/17 ; 0/20 | 20/20 ; 0/20 | 16/17 ; 0/20 | 20/20 ; 0/20 |
| transform | 16/16 ; 15/20 | 19/20 ; 16/20 | 7/11 ; 4/20 | 19/20 ; 17/20 |
| battle | 20/20 ; 11/20 | 20/20 ; 11/20 | - ; 0/20 | 20/20 ; 20/20 |
| modal_dfc | 20/20 ; 14/20 | 20/20 ; 14/20 | 13/13 ; 7/20 | 19/20 ; 17/20 |
| adventure | 20/20 ; 15/20 | 20/20 ; 15/20 | 4/4 ; 1/20 | 20/20 ; 18/20 |
| split, 2003 frame or older | - ; 0/20 | 11/18 ; 6/20 | 8/18 ; 5/20 | 11/20 ; 8/20 |
| split, 2015 frame | 16/16 ; 15/20 | 16/16 ; 14/20 | 6/6 ; 6/20 | 16/20 ; 15/20 |
| aftermath | 20/20 ; 14/20 | 20/20 ; 14/20 | 13/13 ; 7/20 | 20/20 ; 17/20 |
| flip | 1/1 ; 1/20 | 17/20 ; 11/20 | 14/20 ; 10/20 | 18/20 ; 15/20 |
| prepare | 20/20 ; 19/20 | 20/20 ; 19/20 | - ; 0/20 | 20/20 ; 20/20 |
| saga | 20/20 ; 15/20 | 20/20 ; 15/20 | 5/5 ; 2/20 | 19/20 ; 19/20 |
| class | 20/20 ; 9/20 | 20/20 ; 9/20 | 10/10 ; 2/20 | 20/20 ; 17/20 |
| case | 20/20 ; 13/20 | 20/20 ; 12/20 | - ; 0/20 | 20/20 ; 20/20 |
| mutate | 20/20 ; 16/20 | 20/20 ; 16/20 | 11/11 ; 8/20 | 20/20 ; 20/20 |
| leveler | 5/5 ; 3/20 | 18/20 ; 12/20 | 16/18 ; 10/20 | 19/20 ; 11/20 |
| prototype | 20/20 ; 15/20 | 20/20 ; 15/20 | 16/16 ; 11/20 | 20/20 ; 19/20 |
| meld | 20/20 ; 16/20 | 20/20 ; 12/20 | 13/13 ; 9/20 | 20/20 ; 18/20 |
| planar | 13/13 ; 8/20 | 18/19 ; 18/20 | 11/11 ; 11/20 | 13/20 ; 13/20 |

The layout groups gain most because many of their samples are prerelease and promo-pack promos or List cards, which print their parent's or original set's details.

The production extractor, on the same tokens, returns the derived value about as often as the tokens hold it, a few points lower (set code 324/347, number 422/490, size 203/251, year 487/530 on the #193 readings), and the catalog value correspondingly less often (239, 311, 133, 424). Its losses are its own (the slash read as `1`, numbers glued to the copyright line), the subject of the extraction prototype.

## 2. What the catalog would need

Measured with `coverage.py`: the committed catalog copied to `/tmp`, columns added, filled by the rules, `VACUUM`ed, and compared with a copy VACUUMed without changes.

| Catalog | Size | Added |
| --- | --- | --- |
| Committed file | 43,753,472 B (43.8 MB) | |
| Same, VACUUMed | 40,177,664 B (40.2 MB) | |
| + `layout` | 40,861,696 B | +0.68 MB (+1.7%) |
| + `printedSetCode`, `printedNumber`, `printedBaseSetSize`, `printedYear`, `layout`, every value stored | 42,500,096 B | +2.32 MB (+5.8%) |
| Same, storing only values that differ from the catalog's (NULL: the catalog's value; empty: not printed) | 41,816,064 B | +1.64 MB (+4.1%) |

All columns are text. `NULL` in the full layout means not printed or not derivable; scoring would leave the detail out. A `printedBaseSetSize` of `NULL` also answers "whether the base set size is printed", so no separate flag is needed.

The fields replace nothing: `setCode`, `number`, `baseSetSize` and `setReleaseYear` still identify the printing and are shown in the app. Scoring would compare the reading with the printed fields instead.

Building them adds to `build_catalog`: the `sets` columns `type`, `parentCode`, `keyruneCode`, and the `cards` columns `frameVersion`, `originalReleaseDate`, `promoTypes`, `layout`; plus one pass over all cards to find parent-set and World Championship sources by name, artist and number. It needs no download and no hand-kept table.

## 3. Verified and assumed

### Verified

- Every rule's coverage and its difference from the catalog: counted on all 115,875 rows of AllPrintings (`coverage.txt`).
- Match rates: measured on 540 + 2,734 readings of reference images by the production OCR model (`check.txt`).
- The boundary dates (Legends, Exodus, Magic 2015, October 2019, April 2023): the last set before and the first set after each boundary were read on the survey.
- The List's number prefix is always a set in AllPrintings; its `originalReleaseDate` is the date the card was added to The List (2019 to 2026), not the original set's date, so it can't give the year.
- No field in `sets` holds the code event promos print (`PRM`, `FNM`, `H19`): checked `keyruneCode`, `mtgoCode`, `tokenSetCode`, `parentCode` on 28 such sets.
- How catalog values are made: `build_catalog` copies `setCode`, `number` and the set's `baseSetSize`, and `extract_year` takes the year of the set's `releaseDate`; `scoring.py` compares these, unchanged, with the reading.

### Assumed

- **B1.** Reference images show what the card prints. For The List, Scryfall's image is sometimes the original printing's scan; the planeswalker symbol aside, the collector information is the original's, which is what the rule gives. Not checked on physical List cards.
- **B2.** Four cards per set and frame represent the set. Sets that mix formats (Secret Lair's years, 40K's sizes) can hide exceptions the survey didn't sample.
- **B3.** A number's letter suffix (`87a`, `229a`, `77z`) is not printed, or OCR doesn't need it: the rule keeps the digits only. 3,587 own-set rows differ from the catalog only by such a suffix or a `★`. Seen on samples for `★` and `s`/`p`; not checked for `a`/`b` variants.
- **B4.** The survey reads the bottom 15% of the card; some 2003-frame lines sit at the edge of that crop and were cut. This lowers match rates for the rules and the catalog alike.

## 4. What it means for the extraction prototype

- Score against the derived values, not the catalog's: they match what OCR reads on 86% to 98% of readings per detail where today's values match on 57% to 78%, and every difference observed goes the rules' way.
- Leave a detail out of the score when the rules give `None`, rather than scoring 0. That changes the weighted average's denominator per candidate, so candidates are compared on the details they print.
- The prototype can compute the derived values from AllPrintings directly, with `rules.py`, without changing the catalog. The catalog change (four text columns and `layout`, +1.6 to 2.3 MB) is only needed once extraction is adopted.
- The rules don't make printings separable that print the same details. An expansion card and its prerelease, promo-pack and List versions now get the same printed values, so OCR scores them equally (#193 found 29% of paper printings in such groups). pHash decides those, as before.
- Open points:
  1. Whether the 35 exception sets are worth a hand-kept table (5.2% of paper rows score one detail against a wrong value without it).
  2. Whether The List's reference images and the physical cards print the same year (B1).
  3. Whether to keep World Championship decks as candidates at all: they are gold-bordered, not tournament-legal, and the rule finds the original for 98% of them.
  4. Whether scoring should use the copyright year at all for sets whose cards print a range (`1993-2001`): the extractor takes the end, which the rule gives, but OCR often reads only the start.
