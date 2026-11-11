# What each card shows, and where (MTGJSON release 5.3.0+20261103)

A copy of the summary for issue #203 (map #189), built on 2026-11-08 from MTGJSON release `5.3.0+20261103`. Its figures come from that release and are not updated for later ones. It is kept here as the reference for scoring and for the catalog's printed values (#210).

It was written on the research branch `research/card-details-summary` ([05c9193](https://github.com/raadon96/MTG-Recognition/tree/05c9193db671683fcb3bd83035f9beac416e8ce9/docs/research)) and changed when copied here:

- the decisions map #189 made after it (#197, #206) are in [Decided after this summary](#decided-after-this-summary), and the text it contradicted now points there;
- domain things are named with the terms in [`GLOSSARY.md`](../../GLOSSARY.md) (collector style, shown on the card, reading, score);
- links to the research notes are permanent links to the research commits.

It brings together two research notes, plus the conclusions reached with the maintainer when reviewing them:

- [card-details.md](https://github.com/raadon96/MTG-Recognition/blob/95b7acb093b4cd164611e260bac583d2f3a93064/docs/research/card-details.md) (#193): where each detail sits, and whether it falls inside today's crops;
- [printed-details.md](https://github.com/raadon96/MTG-Recognition/blob/72d81b0789c25f2d459dc838c4a0b847ca0c1bb9/docs/research/printed-details.md) (#201): what each printing shows on the card, derived from AllPrintings.

It was the input to the extraction prototype (#197).

| Research branch | Commit | Tag |
| --- | --- | --- |
| `research/card-details` | [95b7acb](https://github.com/raadon96/MTG-Recognition/tree/95b7acb093b4cd164611e260bac583d2f3a93064/docs/research) | `archive/research/card-details` |
| `research/printed-details` | [72d81b0](https://github.com/raadon96/MTG-Recognition/tree/72d81b0789c25f2d459dc838c4a0b847ca0c1bb9/docs/research) | `archive/research/printed-details` |
| `research/card-details-summary` | [05c9193](https://github.com/raadon96/MTG-Recognition/tree/05c9193db671683fcb3bd83035f9beac416e8ce9/docs/research) | `archive/research/card-details-summary` |

The tags keep each commit reachable if its branch is deleted.

The page [card-details-summary.html](card-details-summary.html) shows the same tables with an annotated card per row; download it and open it in a browser (it loads the card pictures from Scryfall). This file is the complete version.

[`build.py`](https://github.com/raadon96/MTG-Recognition/blob/05c9193db671683fcb3bd83035f9beac416e8ce9/docs/research/card-details-summary/build.py) built the tables below and the HTML page. It also wrote [`summary.json`](https://github.com/raadon96/MTG-Recognition/blob/05c9193db671683fcb3bd83035f9beac416e8ce9/docs/research/card-details-summary/summary.json). Its inputs were:

- AllPrintings;
- the rules in [`rules.py`](https://github.com/raadon96/MTG-Recognition/blob/72d81b0789c25f2d459dc838c4a0b847ca0c1bb9/docs/research/printed-details/rules.py);
- the 540 readings of #193 ([`positions.json`](https://github.com/raadon96/MTG-Recognition/blob/95b7acb093b4cd164611e260bac583d2f3a93064/docs/research/card-details/positions.json)).

No new OCR was run. The scripts stay on the research branches; they aren't copied here.

## Summary

- **What the card shows is the reference, not the catalog.** A detail the card doesn't show can't be read, and a catalog value that differs from the one shown on the card misleads scoring. #201 measured it: where the two differ, OCR reads the value shown on the card (95 of 95 set codes, 118 of 123 collector numbers, 75 of 81 base set sizes, 62 of 69 years) and never the catalog's.
- **Frame and collector style together give clear answers.** The frame decides where the details sit; the **collector style** decides which details the collector information prints (artist only, artist and year, number and size, number alone, rarity first). Grouped by both (table A), every detail is always or never shown in each row, except power/toughness (creatures and vehicles only) and one cell at the 95% edge. The frame alone doesn't decide: the 2015 frame prints the base set size until March 2023 and not after, and the 1997 frame prints the collector number from Exodus on.
- **Today's crops hold every detail a card shows, on every upright layout.** The exceptions are layouts: sideways names (split cards, battles, planes), second names (adventure, prepare, aftermath, flip), power/toughness mid-card (flip, leveler), the whole collector information of planes and older split cards, and the 1993 frame's name, whose top the name crop cuts off.

## 1. How to read the tables

### Three layers

1. **Shown on the card: always, never or sometimes.** This is the truth about the card, and it decides the cell's colour. It comes from the rules in [`rules.py`](https://github.com/raadon96/MTG-Recognition/blob/72d81b0789c25f2d459dc838c4a0b847ca0c1bb9/docs/research/printed-details/rules.py) (set code, collector number, base set size, year) and from AllPrintings (artist, power/toughness, second name), counted over every paper printing in the group:
   - **always shown**: on at least 95% of the group's paper printings;
   - **never shown**: on at most 5%;
   - **sometimes shown**: in between, with the main reason it is missing.
2. **Inside today's crops: inside, clipped, outside or sideways.** Where the detail sits for today's extraction, measured on the sampled reference images where the card shows it and OCR read it:
   - **inside**: at least 90% of the detail's box lies inside the crop (the name crop for names, the collector information crop for everything else) on at least 90% of the readings;
   - **clipped**: 50% to 90% inside;
   - **outside**: inside on at most 10% of the readings;
   - **sideways**: printed sideways or upside down, read only after rotating the card;
   - **partly**: inside on some readings only (no cell has this today).
3. **Read rate**: on how many sampled reference images OCR found the detail, out of those where the card shows it. For information only: it never decides the colour.

### Why the read rate can't decide what a card shows

#193's tables counted a detail as "read" only when OCR found the *catalog's* value. A low rate mixes up three things:

- **Not printed.** Base set size on 2015-frame cards from April 2023: 0 of 20.
- **Printed but misread.** The 2003 frame's collector number: read 14 of 20, yet printed on every card. OCR misreads small old-frame text (`6o/350`, `212` for `2/2`).
- **Printed, but not the catalog's value.** A List card prints `128/169`; the catalog holds 5,075, so #193 counted it as not read.

The rules decide the first. The read rate measures the second, and only once the third is removed, which is why this summary matches tokens against the rules' values, not the catalog's.

### For each printing

The labels summarise a group. For one printing, `printed_details` in [`rules.py`](https://github.com/raadon96/MTG-Recognition/blob/72d81b0789c25f2d459dc838c4a0b847ca0c1bb9/docs/research/printed-details/rules.py) gives one of three answers per detail:

- **shown**, with the value the card prints;
- **not shown**, so scoring leaves the detail out rather than scoring 0;
- **shown, but the value can't be derived**: event promo sets that print `PRM`, `FNM` or `H19`, and old-frame promos that print another set's collector number or an event mark (about 1,700 paper printings together). These count as not shown in the tables, with the reason "value unknown". A further 35 sets (about 5,500 paper printings, 5.2%) print a value the rules get wrong (TSR prints 2020, the rules give 2021); the rules don't flag these ([printed-details.md](https://github.com/raadon96/MTG-Recognition/blob/72d81b0789c25f2d459dc838c4a0b847ca0c1bb9/docs/research/printed-details.md), "Exceptions the rules get wrong"). #206 decided to keep no hand-kept table for them: the catalog stores the rules' values, and the right printing misses that one detail.

"Sometimes" has two kinds of reason, and they need different handling:

- **It depends on the card.** Power/toughness is printed on creatures and vehicles only. The catalog knows each printing's power, so scoring can ask for it only where it applies.
- **It depends on the collector style.** Inside a group that mixes styles (table B), the base set size depends on the date. Table A splits by collector style, or a printed-detail field in the catalog answers it per printing.

### Groups: frame, collector style, layout

Three terms each answer one question:

- **Frame** (glossary): where the details sit, and whether a set code is printed (2015 frame only).
- **Collector style** (glossary): which details the collector information prints.
- **Layout** (glossary): which details move, turn sideways or come twice.

**Collector style** is defined in `GLOSSARY.md`: which details a printing's collector information prints, and in what form. It is decided by the date the printing's details come from, whether it is a retro reprint, and whether it is numbered above the base set size.

| Collector style | Prints | Example |
| --- | --- | --- |
| artist only | the illustration credit | `Illus. Mark Poole` |
| artist and year | + the copyright year | `Illus. … © 1995 Wizards of the Coast` |
| number and size | + collector number and base set size | `197/318` |
| number alone | + collector number, no base set size | `302` |
| rarity first | the rarity before the collector number, no base set size | `C 0254` |

How the rules decide it, in order (`frame_row` in [`build.py`](https://github.com/raadon96/MTG-Recognition/blob/05c9193db671683fcb3bd83035f9beac416e8ce9/docs/research/card-details-summary/build.py), using the dates in [`rules.py`](https://github.com/raadon96/MTG-Recognition/blob/72d81b0789c25f2d459dc838c4a0b847ca0c1bb9/docs/research/printed-details/rules.py)):

| Condition | Collector style |
| --- | --- |
| before Legends (1994-06-01), not a retro reprint | artist only |
| before Exodus (1998-06-15), not a retro reprint | artist and year |
| retro reprint: 1993, 1997 or 2003 frame from Magic 2015 (2014-07-18) on | number alone |
| from March of the Machine (2023-04-01): 2015 frame | rarity first |
| from March of the Machine: future frame (MB2) | number alone |
| numbered above the base set size, from Throne of Eldraine (2019-10-01) | number alone |
| otherwise | number and size |

The date is the one the printing's details come from: the original set's for The List, the parent set's for promos that reprint a parent card, `originalReleaseDate` for Secret Lair and similar, else the set's release date. So a List card sits in the row of the set whose details it prints, and The List has no row of its own.

- **Table A** sorts every paper printing (106,466) into exactly one row, by frame and collector style. "From" and "to" are the 1st and 99th percentile of the dates of the row's printings, so a few promos with odd dates don't stretch the range; "today" means the style is still printed. Where a row needs a condition besides the dates, the style says "only …".
- **Table B** answers only **where** the layout puts things; which details are printed is table A's job. These printings are also in table A. Its columns are the name, the second name, the collector information as a whole (its details move together: upright in one strip, or sideways together), and power/toughness. Transform, modal_dfc, saga, class, case, mutate, prototype and meld place every detail where a normal card does, so they have no row (see below the table). Rarer layouts (reversible, scheme, vanguard, host, augment: 0.3%) are left out.

Merged: no paper 1993-frame card is a retro reprint, and the 6 paper 2003-frame retro reprints are counted in "2003, number and size". Borderless is not a boundary: it changes no detail, so borderless cards sit in their collector style's row.

## 2. The tables

Each cell: the label (with the share of the group's paper printings that show it, when not 100%), the main reason when sometimes shown, then the crop layer and the read rate on the samples.

### Table A: by frame and collector style

| Frame | Collector style | From | To | Share | name | set code | collector number | base set size | artist | year | power/toughness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1993 | artist only | August 1993 | April 1994 | 2.2% (2,317) | always; inside, read 10/10 | never (0%) | never (0%) | never (0%) | always; inside, read 10/10 | never (0%) | sometimes (33%), creatures and vehicles only; inside, read 2/2 |
| 1993 | artist and year | June 1994 | December 1996 | 3.1% (3,328) | always; clipped, read 8/10 | never (0%) | never (0%) | never (0%) | always; inside, read 9/10 | always; inside, read 7/10 | sometimes (41%), creatures and vehicles only; inside, read 4/4 |
| 1997 | artist and year | June 1996 | March 1998 | 2.3% (2,435) | always; inside, read 18/20 | never (0%) | never (0%) | never (0%) | always; inside, read 19/20 | always; inside, read 14/20 | sometimes (42%), creatures and vehicles only; inside, read 6/6 |
| 1997 | number and size | June 1998 | October 2006 | 6.3% (6,764) | always; inside, read 22/22 | never (0%) | always (97%); inside, read 19/22 | always (97%); inside, read 18/22 | always; inside, read 21/22 | always; inside, read 21/22 | sometimes (45%), creatures and vehicles only; inside, read 5/9 |
| 1997 | number alone (only retro reprints) | March 2021 | today | 1.7% (1,780) | always; inside, read 18/20 | never (0%) | sometimes (95%), old-frame promos: value unknown; inside, read 15/19 | never (0%) | always; inside, read 19/20 | always; inside, read 12/20 | sometimes (40%), creatures and vehicles only; inside, read 9/9 |
| 2003 | number and size | July 2003 | June 2014 | 16.0% (17,042) | always; inside, read 21/21 | never (0%) | always (96%); inside, read 16/21 | always (96%); inside, read 16/21 | always; inside, read 18/21 | always; inside, read 20/21 | sometimes (48%), creatures and vehicles only; inside, read 6/6 |
| 2015 | number and size | July 2014 | February 2023 | 30.6% (32,569) | always; inside, read 31/31 | always (98%); inside, read 29/31 | always (99%); inside, read 31/31 | always (98%); inside, read 28/31 | always (100%); inside, read 30/31 | always; inside, read 28/31 | sometimes (49%), creatures and vehicles only; inside, read 16/16 |
| 2015 | number alone (only numbered above the base set size) | October 2019 | February 2023 | 7.1% (7,534) | always; inside, read 4/5 | always (99%); inside, read 5/5 | always; inside, read 5/5 | never (0%) | always (100%); inside, read 2/5 | always; inside, read 5/5 | sometimes (50%), creatures and vehicles only; inside, read 1/1 |
| 2015 | rarity first | April 2023 | today | 30.5% (32,448) | always; inside, read 38/41 | always (97%); inside, read 40/40 | always (98%); inside, read 40/40 | never (0%) | always (98%); inside, read 39/41 | always; inside, read 37/41 | sometimes (50%), creatures and vehicles only; inside, read 19/21 |
| future | number and size | May 2007 | September 2022 | 0.1% (98) | always; inside, read 8/8 | never (0%) | always (99%); inside, read 8/8 | always (99%); inside, read 8/8 | always; inside, read 8/8 | always; inside, read 8/8 | sometimes (62%), creatures and vehicles only; inside, read 4/4 |
| future | number alone | August 2024 | today | 0.1% (151) | always; inside, read 12/12 | never (0%) | always (97%); inside, read 11/11 | never (0%) | always; inside, read 12/12 | always; inside, read 11/12 | sometimes (31%), creatures and vehicles only; inside, read 6/6 |

Samples per row: #193's frame groups and its 20 List samples, re-sorted by the rules above (5 to 41 per row; 5 in "2015, number alone", so its read rates are thin).

### Table B: layouts that move details

| Layout | Share | name | second name | collector information | power/toughness |
| --- | --- | --- | --- | --- | --- |
| adventure | 0.8% (802) | always; inside, read 20/20 | always; outside, read 20/20 | always; inside, read 20/20 | sometimes (41%), creatures and vehicles only; inside, read 18/19 |
| prepare | 0.2% (238) | always; inside, read 20/20 | always; outside, read 20/20 | always; inside, read 20/20 | sometimes (50%), creatures and vehicles only; inside, read 20/20 |
| aftermath | 0.1% (142) | always; inside, read 20/20 | always; outside, read 20/20 | always; inside, read 20/20 | never (0%) |
| flip | 0.1% (74) | always; inside, read 18/20 | always; sideways, read 18/20 | always; inside, read 19/20 | sometimes (90%), creatures and vehicles only; outside, read 20/20 |
| split, 2015 frame | 0.3% (338) | always; sideways, read 16/20 | always; sideways, read 20/20 | always; inside, read 17/20 | never (0%) |
| split, before the 2015 frame | 0.1% (145) | always; sideways, read 19/20 | always; sideways, read 20/20 | always; sideways, read 17/20 | never (0%) |
| battle | 0.1% (54) | always; sideways, read 20/20 | never (0%) | always; inside, read 20/20 | never (0%) |
| planar | 0.3% (324) | always; sideways, read 20/20 | never (0%) | always; sideways, read 19/20 | never (0%) |
| leveler | 0.1% (61) | always; inside, read 20/20 | never (0%) | always; inside, read 20/20 | always (98%); outside, read 20/20 |

Samples per row: #193's 20 per layout group. "Collector information" counts a sample as read when OCR read any of its details (set code, collector number, base set size, artist, year) that the card shows.

The other common layouts (transform, modal_dfc, saga, class, case, mutate, prototype, meld: 3.0% of paper printings) have no row: on #193's 160 samples of them, the name, the collector information and power/toughness were all inside today's crops.

### What the tables show

- **Table A.** Name and artist are always shown. The copyright year is shown from collector style "artist and year" on, the collector number from "number and size" on, and the base set size only in "number and size". The set code is shown on the 2015 frame only, whatever the collector style. Power/toughness is the only detail that is sometimes shown in every row. One cell is at the edge: the 1997-frame retro reprints' collector number, shown on 94.7%, the rest being old-frame promos whose value can't be derived.
- **Every detail a card shows sits inside today's crops on upright cards**, except the 1993 frame's name, whose top is cut off on 8 of 10 readings from Legends on (the name sits at 0.04 to 0.09 of the card's height, the crop starts at 0.04).
- **Table B: the gaps.**
  - Second names outside the name crop: adventure (left inset) and prepare (right inset), aftermath (sideways in the bottom half), flip (upside down).
  - Names printed sideways: split cards, battles, planes.
  - The whole collector information printed sideways: planes, and split cards before the 2015 frame.
  - Power/toughness mid-card: flip and leveler cards.
  - Share of the catalog's printings (from #193): second names 1.3%, sideways names 0.8%, sideways collector information 0.4%, mid-card power/toughness 0.1%.

## 3. Traps

1. **The catalog isn't what the card shows.** A List card is `PLST HOP-128` with base set size 5,075 in the catalog, but prints `HOP 128/169`. A prerelease promo `PKHM 214p` prints `KHM 214/285`; the catalog's base set size for PKHM is 1. Printings whose derived value differs from the catalog's: set code 11,215, collector number 17,128, base set size 10,512, year 14,436 (#201).
2. **The year on the card isn't the set's year.** Secret Lair prints 2023 to 2026, where the catalog has 2019; Time Spiral Remastered prints 2020, the catalog 2021; all 20 List samples print another year than the catalog's 2020. 13,553 printings have an `originalReleaseDate` in another year than their set. Older frames print a range (`1993-2001`), and OCR often reads only the start. #206 decided to store the end year and to count the year as found when the reading holds the end year or the start followed by a dash (`1993-`).
3. **The frame doesn't tell the collector style.** 4.2% of printings carry a frame older than their set's date: The List, Remasters, Masters Editions, 30A, Secret Lair. A 1997-frame card from 2023 prints a collector number but no base set size. 2,276 Magic Online promos carry the 2015 frame with a 2002 date.
4. **Two-name cards have one name in the catalog.** Both printings of `Bonecrusher Giant // Stomp` carry the full name, the same collector number and the same image (split, adventure, aftermath, flip, prepare). The catalog's name is "A // B", so reading only the first name scores about half.
5. **The big name may not be the card's name.** 3,231 paper printings (3,398 printings in the catalog, 2.9%) have a flavor name: the large name is the flavor name, and the card name is printed small below it, at 0.11 to 0.14 of the card's height, outside the name crop.
6. **Other text sits in the collector information crop.** The last line of flavor or rules text (on 5 to 12 of 20 samples in most 2015-frame groups), the MDFC hint bar (20 of 20), licensing lines (MARVEL, SQUARE ENIX), playtest marks, and the grey back-face power/toughness on transform fronts (CR 712.2c).
7. **Many printings show exactly the same details.** An expansion card and its prerelease, promo-pack and List versions differ only by a stamp; pre-Exodus reprints share name and artist across many sets. 29.1% of paper printings can only be told apart by the picture, and the rules make no difference there (issue #202).
8. **Old-frame misreads.** On the 1997 frame OCR reads `2/2` as `212`, `75/350` as `751350`, and returns the collector number stuck to the copyright line (`Wizards of the Coast LLC197/318`).
9. **Not every candidate exists on paper or in English.** 9,409 printings (8.1%) are online-only. Photos of Japanese and German printings exist in `data/scratch`; the catalog holds English printings only.

## 4. What #201 changed in #193's documents

[card-details.md](https://github.com/raadon96/MTG-Recognition/blob/95b7acb093b4cd164611e260bac583d2f3a93064/docs/research/card-details.md) and [card-details.html](https://github.com/raadon96/MTG-Recognition/blob/95b7acb093b4cd164611e260bac583d2f3a93064/docs/research/card-details.html) stay unchanged on their branch. These parts are out of date:

- **Base set size until April 2023, not January 2023.** Dominaria Remastered and Phyrexia: All Will Be One (January and February 2023) still print `114/271`. #193's group "2015 frame, 2023+" holds them.
- **A number above the base set size drops the size, not being a Commander or Jumpstart set.** `C20 271/322`, `AFC 024/062` and `MIC 026/038` print the size; `AFC 267` and `C21 302` don't, because they are numbered above it. J22 and DBL do print the number alone, but as exceptions the rules get wrong.
- **"Read" and "not read" in the HTML table and in `summary.json` compare with the catalog's values.** For The List and promos, "not read" for set code, collector number and base set size means "printed, but not the catalog's value". The tables above use the rules' values instead.
- **The groups.** #193's frame groups ("2015 frame, 2014-2022", "2015 frame, 2023+", "2015 frame, borderless" and the others) are replaced by table A's frame and collector style rows; "The List (PLST)" is spread over table A by the set whose details each card prints.

## 5. What this means for the extraction prototype

Written before the prototype ran. What #197 and #206 decided afterwards is in [Decided after this summary](#decided-after-this-summary).

- **Score only what the card shows.** For each candidate, take the values shown on the card from [`rules.py`](https://github.com/raadon96/MTG-Recognition/blob/72d81b0789c25f2d459dc838c4a0b847ca0c1bb9/docs/research/printed-details/rules.py) and leave out the details it marks as not shown, rather than scoring them 0. Power/toughness only for candidates with a power.
- **Keep the two crops for upright cards.** They hold every detail an upright card shows. Inside the collector information crop, tell the details apart by line and by the shape of the text, not by frame: the 2015 frame has two lines on the left and the copyright on the right; older frames have the artist line above a copyright line that ends with the collector number. The collector style says which of these lines to expect.
- **Decide what to do with table B's gaps.** Rotate or add a crop for sideways and second names (2.5% of the catalog's printings together), or leave those cards to pHash. Candidates are known before OCR, so the layout could choose the crop.
- **Sample by table A's rows.** The prototype's 100 reference images per group should follow the rows of table A, plus the layouts of table B, rather than the frame alone. The 1993 frame and "2015, number alone" are the rows the benchmark doesn't cover.
- **The 1993 name crop.** Measure whether moving `NAME_ROI`'s top from 0.04 to 0.03 recovers the 1993 frame's name without hurting other frames.

## 6. Verified and assumed

### Verified

- Every label: counted over all 106,466 paper printings in AllPrintings, with the rules of #201, which were checked against 540 + 2,734 OCR readings of reference images there.
- Every crop and read count: measured on #193's 540 readings of reference images by the production OCR model, with tokens matched against the rules' values for set code, collector number, base set size and year, and #193's matching for name, second name, artist and power/toughness.
- Power/toughness is in AllPrintings for creatures and vehicles: a card has `power` or `toughness` exactly when the card prints them.

### Assumed

- **C1. The thresholds.** 95% and 5% for always and never, and 90% of the box for inside, are choices, not measurements. Moving them changes one cell (the 1997-frame retro reprints' collector number, at 94.7%).
- **C2. Reference images stand for photos.** The crop layer is measured on clean scans (#193's assumption A1). Card images from segmentation can have drifted edges, which would push the 1993 name and the bottom line further out of the crops.
- **C3. Samples are small.** 5 to 41 samples per row of table A, 20 per layout. They show where details usually sit, not every exception.
- **C4. Second names.** A second name is counted as shown on every card of the two-name layouts (adventure, prepare, aftermath, flip, split). A meld card's melded name and a transform card's back face aren't counted: they are not printed on the face being read.

## Decided after this summary

Map #189 went on after this summary was written. Its decisions are recorded in #197, #206 and the spec #210, and they change how the sections above apply:

- **The catalog holds the values shown on the card** (#197, #210). It gains a printed set code, printed collector number, printed base set size and printed year, each empty when the card doesn't show the detail. It also gains the flavor name and whether the printing is the plain one. The values come from the rules of #201, from AllPrintings alone, and replace `rules.py` as the source scoring compares with.
- **No hand-kept table for the 35 sets the rules get wrong** (#206). The catalog stores the rules' values. Where the card shows something else, the right printing misses that one detail. Example: *Jhoira's Timebug* in TSR shows `© 2020`, the rule gives 2021, and it still wins 5 of 6 details against 2 of 5 for the TSP original.
- **World Championship decks stay candidates** (#206). They get the values of the original printing, found by name, artist and collector number. For the 28 printings whose original isn't found, the set code, collector number, base set size and year are empty, so they count as not shown. On a tie, the tie-break prefers the original, so a photo of a World Championship card comes back as its original printing.
- **Range years store the end year** (#206). For `© 1993-2003` the printed year is 2003. Scoring counts the year as found when the reading holds the end year, or the start year followed by a dash (`1993-`).
- **Scoring is candidate-driven, without positions** (#197). For each candidate, scoring looks for each detail the printing shows on the card in the raw text of the reading. A detail it doesn't show is left out. The score is the weighted average over the details it shows. Section 5's idea of telling details apart by line inside the collector information crop was not adopted.
- **Two names and flavor names** (#197). The name is scored as the best of the printing's names: each face of `A // B`, and the flavor name (traps 4 and 5).
- **Ties** (#197). Among equal scores, the plain printing comes first, then pHash order. The plain printing shows its own details (not The List, not a promo showing its parent's details, not a World Championship deck), and its collector number has no suffix (`254★`, `71z`).
- **The crops stay as they are** (#210), so table B's gaps and the 1993 frame's cut name remain. Rotating the name crop for split cards, battles and planes is out of scope (#189): 93 of 100 sampled split cards show the same details as another printing of their name, so it would help tell the card apart, but not the printing. Revisit it as a normal issue if a split-card photo fails the benchmark.

## Open questions

Still open at the end of map #189:

1. Do physical List cards print the same year as their reference images? (#201, B1). The rules use the List card's own year.
2. Does `use_textline_orientation=True` read the flip cards' upside-down second name and power/toughness, and at what cost in time? (#193)
3. How far do segmentation's card edges drift from the reference images' edges (C2)?
4. Does pHash separate printings that show the same details? (#202, outside the map)

Decided since: the 35 sets and World Championship decks (#206). Out of scope: rotating the name crop by layout (#189).
