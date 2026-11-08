# Where each detail sits across card frames and layouts

Research for issue #193 (map #189). Question: where does each detail sit on a card, for each frame and layout, and does it fall inside the two crops `read_card` reads today (`NAME_ROI`, `COLLECTOR_INFORMATION_ROI` in `engine/ocr/reading.py`)?

Researched 2026-11-08 against MTGJSON release `5.3.0+20261103`, the release the committed catalog is built from. The page `card-details.html` next to this file shows one card per frame and layout with each detail boxed, today's two crops drawn dashed, the table below and the traps; open it in a browser (it loads the card pictures from Scryfall).

## Sources and method

Primary sources:

- **CR**: Magic: The Gathering Comprehensive Rules, effective 2026-11-06 (`https://media.wizards.com/2026/downloads/MagicCompRules 20261106.txt`). Rules quoted by number.
- **Scryfall docs**: `https://scryfall.com/docs/api/frames` and `https://scryfall.com/docs/api/layouts`, read 2026-11-08.
- **MTGJSON docs**: `https://mtgjson.com/data-models/card/card-set/` and `https://mtgjson.com/data-models/set/`, read 2026-11-08.
- **AllPrintings**: `data/reference/mtgjson/AllPrintings.sqlite`. The catalog has one row per row of its `cards` table (115,875 in both; `jobs/sync.py` `build_catalog` joins `cards`, `cardIdentifiers` and `sets` with no filter).

Measurements (scripts in `docs/research/card-details/`, all throwaway):

- `catalog_stats.py`: shares and trap counts from AllPrintings.
- `ocr_positions.py`: for 27 groups (frames, eras within a frame, layouts), 20 randomly sampled paper printings each (seed 193; 540 cards) whose reference image is in `data/reference/images`. Each reference image (488x680, Scryfall `normal`) is read whole, upscaled 2x, by the production model: `load_paddleocr()` and `predict(..., text_det_thresh=0.05)` as in `read_card`. Layouts with sideways or upside-down text are also read rotated (90° both ways, or 180°), and the boxes mapped back onto the upright card. Each OCR text line is matched against the sampled printing's own details (fuzzy name and artist, exact number, set code, P/T, copyright keywords) and its box stored as fractions of the card. Raw output: `positions.json`.
- `summarise.py`: per group and detail, on how many samples the detail was read, its median box, and whether that box lies at least 90% inside today's crop (the name crop for names, the collector information crop for the rest). Writes `summary.json` and `../card-details.html`.
- `inseparable.py`: printings that share every detail scoring uses, or every printed detail.

Today's crops, checked in `_crop` (`card_image[h*top : h*bottom, w*left : w*right]`, so the tuples are (top, bottom, left, right)):

| Crop | Top | Bottom | Left | Right |
| --- | --- | --- | --- | --- |
| `NAME_ROI` | 0.04 | 0.11 | 0.06 | 0.90 |
| `COLLECTOR_INFORMATION_ROI` | 0.88 | 1.00 | 0.05 | 0.95 |

Limits of the measurement: "read" means OCR found the detail; a detail not read is either not printed or misread, and the notes say which where it matters. Reference images are clean scans; card images from photos are not (assumption A1 below). 20 samples per group show where details usually sit, not every exception.

## Summary

- On every upright frame and layout, all collector information details sit between 0.89 and 0.98 of the card's height, inside `COLLECTOR_INFORMATION_ROI`, and the name sits between 0.03 and 0.11, inside `NAME_ROI` (1993 frame: just clipped at the top). For about 97% of the catalog the two crops hold every detail that is printed.
- The real gaps are names and details printed sideways or elsewhere: the second name of adventure, prepare, aftermath, flip and meld cards; both names of split cards (Rooms included), battles and planes; P/T of flip and leveler cards; and the whole collector information of pre-2015-frame split cards and planes, printed sideways along the right edge.
- What is printed changes more by **era and set type** than by frame: set code only from Magic 2015 (July 2014); no collector number before Exodus (June 1998); no base set size from 2023, nor on retro-frame reprints, Commander and other supplemental cards; promos and The List print another set's code, number and size than the catalog holds.
- Scoring compares details that are often not what is printed: `setReleaseYear` against the copyright year (wrong on all 20 List samples and on most Secret Lair samples), `baseSetSize` where none is printed or another is printed, and catalog set codes and numbers (`POGW`, `63s`, `HOP-128`) that never appear on the card.
- OCR alone can't separate 29% of paper printings even when read perfectly: the original and its prerelease, promo-pack and List versions print identical details, and pre-Exodus reprints share name and artist across many sets. 909 catalog rows (split, adventure, aftermath halves) are identical in every scored field and share one image.

## 1. Share of the catalog

### Verified (AllPrintings, `catalog_stats.py`)

| Frame | Rows | Share |
| --- | --- | --- |
| 2015 | 79,726 | 68.8% |
| 2003 | 17,927 | 15.5% |
| 1997 | 12,325 | 10.6% |
| 1993 | 5,645 | 4.9% |
| future | 252 | 0.2% |

Scryfall's frames page lists the same five values: 1993 "The original Magic card frame, starting from Limited Edition Alpha", 1997 "from Mirage block", 2003 "introduced in Eighth Edition and Mirrodin block", 2015 "introduced in Magic 2015", future "The frame used on cards from the future".

| Layout | Rows | Share | | Layout | Rows | Share |
| --- | --- | --- | --- | --- | --- | --- |
| normal | 109,526 | 94.52% | | aftermath | 194 | 0.17% |
| transform | 2,120 | 1.83% | | reversible_card | 164 | 0.14% |
| adventure | 914 | 0.79% | | mutate | 146 | 0.13% |
| modal_dfc | 654 | 0.56% | | vanguard | 119 | 0.10% |
| split | 517 | 0.45% | | scheme | 110 | 0.09% |
| saga | 426 | 0.37% | | flip | 78 | 0.07% |
| planar | 330 | 0.28% | | meld, class | 72 each | 0.06% |
| prepare | 248 | 0.21% | | leveler, prototype, host, case, augment | 63, 49, 29, 27, 17 | ≤0.05% |

Rows count faces: a split, adventure, aftermath, flip or prepare card is two rows (three to five for a few split cards), so the number of physical cards in those layouts is half the rows. 9,409 rows (8.1%) are online-only (`sets.isOnlineOnly` or `cards.isOnlineOnly`); 106,466 have `paper` in `availability`.

Cross-tabulated, every non-normal layout except split, transform, planar, scheme, flip, leveler, prepare and vanguard is 2015 frame only; split cards exist in 1997 (38), 2003 (113) and 2015 (366) frames, flip cards almost all in 2003 (74).

## 2. Where each detail sits

### Verified: what the rules say

- Name: "printed on its upper left corner" (CR 201.1).
- Power/toughness: "two numbers separated by a slash printed in its lower right corner" (CR 208.1). A leveler has "three power/toughness boxes" (CR 711.1); a prototype card has "a second set of power, toughness, and mana cost" in an inset frame (CR 719.1). A transform front "whose back face is a creature has the back face's power and toughness printed in gray above the power and toughness box" (CR 712.2c).
- Collector information (CR 213.1): collector number "in the form [card number]/[total cards in the set] or simply [card number]" (213.1a); rarity "a single letter following the collector number" (213.1b); set code and language "separated by a bullet point", or by a star on premium cards (213.1e); illustration credit after "the paintbrush icon or, on older cards, the abbreviation 'Illus.'" (213.1f); legal text "at the bottom or bottom-right" (213.1g). "Not all card sets were printed with all of the information listed below on each card" (213.1).
- Two-name layouts: split cards "have two card faces on a single card" (CR 709.1); flip cards print the alternative name, text, type, power and toughness "upside down" in the bottom half (CR 710.1, 710.1b); adventurer cards inset the adventure "on the left" of the text box (CR 716.1-716.2); preparation cards inset the prepare spell "on the right" (CR 723.1-723.2); a modal DFC face has "a hint bar in the lower left corner with information about the opposite face" (CR 712.3c).
- Expansion symbol: "normally printed below the right edge of the illustration"; it was black on every card before Exodus, and core sets before Sixth Edition had none (CR 206.1, 206.2).

### Verified: measured positions

Median boxes as (left, top, right, bottom) fractions of the card, from `summary.json`. "in" = at least 90% inside today's crop on every sample where it was read.

| Group (share of catalog) | Name | Set code | Collector number | Base set size | Artist | Copyright year | P/T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1993 frame (4.8%) | (0.07, 0.04, 0.38, 0.09); top clipped on 9 of 18 | not printed | not printed | not printed | in, after "Illus." at (0.09, 0.90, 0.52, 0.94) | in, (0.09, 0.93, 0.62, 0.95), only from 1994 sets | in, (0.84, 0.90, 0.93, 0.95) |
| 1997 frame before Exodus (1.9%) | in | not printed | not printed | not printed | in | in | in |
| 1997 frame 1998-2003 (5.5%) | in | not printed | in, end of the copyright line (0.23, 0.92, 0.76, 0.95) | in, same line | in, centred (0.32, 0.89, 0.67, 0.93) | in | in, often misread |
| 1997 frame retro reprints 2014+ (1.5%) | in | not printed | in, number alone | not printed | in | in | in |
| 2003 frame (13.5%) | in | not printed | in, end of the copyright line (0.08, 0.94, 0.61, 0.97) | in, same line | in (0.11, 0.91, 0.37, 0.94) | in | in |
| 2015 frame 2014-2022 (33.5%) | in | in, line 2 left (0.06, 0.95, 0.38, 0.97) | in, line 1 left (0.06, 0.93, 0.21, 0.95) | in, on 10 of 19; missing on supplemental sets | in, after the set code | in, right (0.59, 0.94, 0.94, 0.97) | in, (0.82, 0.89, 0.91, 0.94) |
| 2015 frame 2023+ (22.8%) | in (19 of 20) | in | in, rarity first: `C 0254` | not printed | in | in | in |
| 2015 frame borderless (5.4%) | in (15 of 16) | in | in | not printed on any sample | in | in | in |
| future frame (0.2%) | in, starts at x 0.17 | not printed | in, end of the copyright line | in on FUT; MB2 (2024) prints the number alone | in, right (0.61, 0.90, 0.83, 0.93) | in | in |
| The List (5.0%) | in | in, the original set's | in, the original set's | printed, the original set's (catalog: 5,075) | in | printed, the original set's year | in |
| transform (1.8%) | in, starts at x 0.16 | in | in | in, older sets | in | in | in |
| battle (0.05%) | **sideways**, left edge (0.05, 0.47, 0.12, 0.82) | in | in | not printed | in | in | none |
| modal_dfc (0.6%) | in | in | in | in, older sets | in | in | in |
| adventure (0.8%) | in | in | in | in, older sets | in | in | in |
| adventure name | **outside**, left inset (0.08, 0.63, 0.34, 0.67) | | | | | | |
| split, 2003 frame or older (0.1%) | **sideways**, both names along the left edge | not printed | **sideways**, right edge (0.92, 0.55, 0.95, 0.91) | **sideways** | **sideways** | **sideways** | none |
| split, 2015 frame incl. Rooms (0.3%) | **sideways**, both names | in | in | in, older sets | in | in | none |
| aftermath (0.2%) | in | in | in | in | in | in | none |
| aftermath second name | **outside**, sideways in the bottom half (0.86, 0.56, 0.92, 0.66) | | | | | | |
| flip (0.07%) | in | in, 2015-frame reprints only | in | in | in | in | **outside**, mid-card (0.82, 0.25, 0.92, 0.30) |
| flip second name | **outside**, upside down (0.32, 0.86, 0.92, 0.91) | | | | | | |
| prepare (0.2%) | in | in | in, rarity first | not printed | in | in | in |
| prepare spell name | **outside**, right inset (0.51, 0.63, 0.74, 0.67) | | | | | | |
| saga, class, case, mutate, prototype (0.6%) | in | in | in | in, older sets | in | in | in (prototype's second P/T is mid-card, not compared) |
| leveler (0.05%) | in | in, 2015-frame reprints | in | in | in | in | **outside**: the uppermost P/T box (the catalog's) is mid-card (0.80, 0.65, 0.89, 0.70) |
| meld (0.06%) | in | in | in | in | in | in | in |
| planar (0.3%) | **sideways** (0.05, 0.40, 0.11, 0.60) | **sideways**, right edge | **sideways** | **sideways** | **sideways** | **sideways** | none |

Positions are the same across upright layouts because the 2015 frame keeps one collector information block: line 1 holds the collector number, base set size and rarity; line 2 holds set code, language and artist; the copyright sits at the right of both lines; P/T sits just above it on the right. In the 1997, 2003 and future frames the artist line is above the copyright line, and the collector number and base set size end the copyright line, so OCR returns them glued to it (`Wizards of the Coast LLC197/318`).

What appears in the collector information crop besides the details, measured on samples (`other_text_examples` in `summary.json`): the last line of flavor or rules text (some unmatched text on 5 to 12 of 20 samples in most 2015-frame groups), the MDFC hint bar (`Land`, `C: Add`, on 20 of 20 MDFC samples), licensing lines (`©MARVEL`, `FFO SQUARE ENIX`, `C2026 Viacom.`), and playtest marks (`CONFIDENTIAL`, `TEST CARD-Not for constructed play`).

### Verified: what production reads on 16 retro-frame photos

`mtg recognize` on `data/scratch/uncommon_layout` (16 photos: 9 in the 1997 frame, 6 of them INR and BRR retro reprints; 2 in the 2003 frame; 5 in the 2015 frame, 2 of them transform). All 16 best printings were right, with scores from 25 to 88. The collector information readings had the artist and copyright year, the number alone on retro reprints (`397`), no base set size, and three photos of foreign-language printings (`DMR·JP`, `CMM·DE`, `CLB DE`), which the catalog doesn't hold.

### Assumption

- **A1.** Card images from segmentation have the same proportions and margins as reference images, so the fractions above transfer. Segmentation crops to the card's edges, as Scryfall's images are; the benchmark doesn't measure how far a card image's edges drift.
- **A2.** Details read only rotated are not read by production: `load_paddleocr` sets `use_textline_orientation=False`, and PaddleOCR's text line orientation model only flips text by 180°. Reading sideways names needs the crop rotated before OCR.

## 3. Gaps in today's crops

Per catalog row, using the shares above:

| Gap | Rows | Share |
| --- | --- | --- |
| Second name outside `NAME_ROI` (adventure, prepare, aftermath, flip, meld) | 1,506 | 1.3% |
| Names sideways (split, battle, planar) | 901 | 0.8% |
| Collector information sideways (split before the 2015 frame, planar) | 481 | 0.4% |
| P/T outside the crop (flip, leveler) | 141 | 0.1% |
| Flavor name printed in the name crop instead of the name | 3,398 | 2.9% |
| 1993 frame name clipped at the top | 5,645 | 4.9% |

For split, adventure, aftermath, flip and prepare cards the missing second name matters less than it looks: the catalog's name is "A // B" for both rows, and scoring fuzzy-matches the read name against it, so the first name alone scores about half (see trap 5).

## 4. Where OCR alone can't separate printings

### Verified (`inseparable.py`)

- **Equal in every scored field** (name, set code, number, base set size, artist, set year, P/T): 909 rows (0.8%): the two rows of split (471), aftermath (194) and adventure cards whose main face isn't a creature (150), which share one reference image, so pHash ties them too and the catalog's order decides; and reversible cards printing the same card on both sides (94), which pHash can separate.
- **Equal in every printed detail**, under the rules measured above (no number before Exodus, set code only from Magic 2015, promos print their parent's code, The List its original set's): 30,955 of 106,466 paper printings (29.1%). By kind:
  - 2015 frame (20,691): an expansion card and its prerelease promo (`63s`, date-stamped), promo-pack promo (`63p`, planeswalker-stamped) and List reprint print the same name, number, set code and artist. They differ by a stamp on the art and by foil.
  - 1993 frame (4,707) and 1997/2003 core sets (about 2,900): reprints across Alpha, Beta, Unlimited, Revised, Fourth Edition, Collectors' Edition, Foreign Black Border and others share name and artist, and no number or set code is printed. `Plains` by Jesper Myrfors alone has 53 paper printings.
- For these, pHash has to decide; for stamped promos even pHash may not, since the stamp is a small change to the image (see questions).

## 5. Traps

### Verified

1. **Year does not predict frame.** 4,868 rows (4.2%) carry the 1993 or 1997 frame in a set released after July 2003, or the 2003 frame after July 2014: The List (1,886), 30th Anniversary Edition, Masters Editions I-IV, Ravnica, Innistrad and Dominaria Remastered, The Brothers' War Retro Artifacts and Commander, Secret Lair, Time Spiral Remastered, the Modern Horizons timeshifts. 2,276 Magic Online promos (PRM) carry the 2015 frame with a set date of 2002; Mystery Booster 2 (2024) prints the future frame. Within the 1997 frame, 2,579 rows come from sets released after July 2014.
2. **The copyright year is not `setReleaseYear`.** Scoring compares the read copyright year with the set's release year (`scoring.py` `FIELDS`, `sync.py` `extract_year`). Measured: all 20 List samples print another year than the catalog's 2020; Secret Lair samples print the drop's year (2023 to 2026) against 2019; TSR prints 2020 against 2021; Visions and WC97 print 1996 against 1997. 13,553 rows have an `originalReleaseDate` in another year than their set. Older frames print a range (`1993-2001`), where the extractor takes the second year. 1993-frame cards also print "Illus. © Artist" and sometimes "© 1994 Artist": a © that isn't the Wizards line.
3. **`baseSetSize` against what's printed.** No collector number before Exodus (7,910 rows in sets released before 1998-06-15). No base set size on 2015-frame cards from 2023 (32,737 rows in 2015 frame from 2023 on), on retro-frame reprints, nor on Commander, Jumpstart and other supplemental sets measured (`J22 161`, `AFC 267`, `C21 302`); MB2's future-frame cards print the number alone. Promo sets print their parent's number and size while the catalog holds the promo set's: PKHM `214p` prints `214/285`, the catalog's base set size for PKHM is 1. 30,561 rows have a number above their set's base set size (showcase, extended art, bundle cards); those print the slash with the main run's size (`279/269`). MTGJSON: `baseSetSize` "will default to totalSetSize if not available".
4. **The List.** 5,764 rows, 5.0% of the catalog, in four frames. The card prints its original set's code, number, base set size and copyright year, plus a planeswalker symbol; the catalog has set code `PLST`, number `HOP-128`, base set size 5,075 and year 2020. Only name, artist and P/T agree with what's printed.
5. **Two-name layouts have one name in the catalog.** For split, adventure, aftermath, flip and prepare cards both rows carry the full name `Bonecrusher Giant // Stomp` (`build_catalog` uses `faceName` only for transform, modal_dfc and reversible_card), with the same number and image. Only P/T tells the rows apart, and only when one face is a creature.
6. **Printed set code and number differ from the catalog's.** Set codes are printed from Magic 2015 on (verified on samples; none read on 2003 and older frames). Promo sets print their parent's code (`POGW` prints `OGW`) and the base number without its suffix (`63s` prints `063`). The rarity letter comes first from 2023 (`C 0254`, on 19 of 20 samples), though CR 213.1b says it follows the number.
7. **Flavor names.** 3,398 rows have a `flavorName`: the large name is the flavor name (`Archangel of Thune` reads as its Secret Lair name), and the card's name is printed small below it, at y 0.11-0.14, outside the name crop.
8. **Old-frame misreads.** In the 1997 frame OCR reads `2/2` as `212` or `2/12`, `75/350` as `751350`, and the number glued to the copyright line. On the 1997-frame 1998-2003 group OCR read P/T on 5 of the 9 creatures sampled; on every other upright group it read P/T on all or all but one.
9. **Online-only printings** (9,409 rows) are candidates though no card exists to photograph.
10. **Foreign-language printings**: photos of `JP` and `DE` cards exist in `data/scratch`; the catalog holds English printings only, so the name can't match and only the collector information helps.

### Assumption

- The date ranges used for groups (Exodus 1998-06-15, Eighth Edition 2003-07-28, Magic 2015 2014-07-18, 2023-01-01 for the rarity-first format) are set release dates from AllPrintings, not the dates each change happened; the 2023 boundary in particular is approximate (the rarity-first format appeared on 1 of the 20 samples from 2014-2022: 30A, 2022).

## 6. What it means for the extraction prototype

- Today's two crops hold every printed detail for upright layouts, about 97% of the catalog. Extraction doesn't need new crops for them; it needs to tell the details apart within the collector information crop, by position: the 2015 frame's two lines on the left and copyright on the right, against the older frames' artist line above a copyright line that ends with the number.
- Frame alone doesn't say which format to expect. The layout of the collector information follows the era and set type (2015 frame before 2023 with `123/280 R`, from 2023 with `R 0123`, supplemental with `123`), and retro-frame reprints print modern-era details in an old frame. Extraction that recognises the pattern of each line is more robust than extraction that selects rules by frame.
- Scoring should give no points, and take none, for a detail the printing can't show: base set size on post-2023, retro and supplemental cards; set code and number on old frames. Today these fields score 0 against every candidate alike, which lowers all scores but doesn't reorder them; the trouble is fields that are printed differently from the catalog (The List, promos, copyright year), which reward the wrong printing.
- Sideways names (split, battle, planar: 0.8%) and second names (1.3%) need a rotated or extra crop, or must be left to pHash. That is a decision for the prototype, measured on those layouts.
- The prototype's sample of 100 reference images per frame and layout should split the 2015 frame by era (2014-2022, 2023+), keep retro reprints, The List and promos as their own groups, and include the 1993 frame, which the benchmark doesn't cover.

## Questions only an experiment can answer

1. How often does extraction read each detail correctly on card images from photos, per group above, compared with reference images? (Accuracy on reference images is an upper bound.)
2. Does pHash separate an expansion card from its prerelease and promo-pack versions, and from its List reprint? If not, what share of benchmark-like photos are affected?
3. Does the 1993 frame's clipped name top cost name accuracy? Would moving `NAME_ROI`'s top to 0.03 hurt other frames?
4. Does rotating the name crop for split cards, battles and planes (detected by layout or by aspect of the name box) recover their names without hurting upright cards?
5. Would comparing the copyright year with each printing's own printed year (not in the catalog today) change rankings, or does the year add little once name, number and set code are read?
6. How do segmentation's card edges compare with the reference images' edges (assumption A1)? A few percent of drift would push the 1993 name and the bottom line of the collector information out of the crops.
7. Does `use_textline_orientation=True` read flip cards' upside-down second name and P/T, and at what cost in time?
