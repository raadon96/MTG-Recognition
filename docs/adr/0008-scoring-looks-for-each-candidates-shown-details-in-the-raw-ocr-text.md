# Scoring looks for each candidate's shown details in the raw OCR text

The reading is the raw text OCR finds in the name crop and the collector information crop, with nothing extracted from it. Scoring takes each candidate in turn and looks in that text for each detail the candidate shows on the card: the values from the catalog's printed columns, and today's name, artist and power/toughness. A detail the candidate doesn't show is left out of its score instead of scoring 0. Before this, the reading guessed each field from the tokens before any candidate was known: 22 regular expressions, a 28-word blocklist and a flavor-text filter decided which token was the set code, the number or the artist, and every mistake they made counted against every candidate. When a candidate's value is known, it only has to be found in the text. Flavor text and copyright words then hold a candidate's value only by chance, and when they do, every candidate shares that chance, so no blocklist or filter is needed.

ADR 0003 still holds: matching is a pHash shortlist, then OCR scoring.

## The rules

Before searching, `O` or `o` next to a digit is read as `0` (`199o`, `6o/350`).

- **Name**: `fuzz.ratio` against the name crop's text, taking the best of the printing's names: each face of `A // B`, and the flavor name, which is the large name on the card when there is one.
- **Artist**: `fuzz.partial_ratio` against the collector information's text.
- **Set code**: found when an upper-case run of letters and digits equals the printed set code, or equals it followed by a two-letter language (`THBDE`, `WAREN`).
- **Collector number and year**: found when a whole run of digits equals the value, leading zeros dropped, so `14` isn't found in `149`. For a range year (`© 1993-2001`) the catalog holds the end year. OCR often reads only the start (`1993-`, `1993-20l0`). Every range starts at 1993, so a run `1993` that isn't followed by a year counts as the year for any printing that shows one; when the end was read (`1993-2003`), only that year counts. On #201's 2,734 survey readings, 465 of 471 range starts read `1993`, and the rule finds the year on 29 more of the 1,175 readings of older-frame cards (1997, 2003 and future frames). Counting every `1993`, even when the end was read, would find 75, but would credit printings whose range ends in another year on the 575 readings that hold a whole range; on the benchmark it lost an 8th Edition card to its 5th Edition printing. The catalog doesn't store ranges: which printings print one can't be told apart cleanly from the data.
- **Base set size**: found in the run right after the number, or glued to it when OCR lost the slash or read it as `1` (`751350`, `7511350`).
- **Power/toughness**: `p/t` with the slash read as any of `/ | \ I l 1 )`, only for candidates with a power.

A detail that is either found or not scores 0 or 100. The score is the weighted average over the details the candidate shows, with the weights of ADR 0003 (name 2, artist 0.5, the rest 1).

## Tie-break

Many printings show exactly the same details as another: The List and promos show their original's set code, number and size, and suffix variants (`254★`, `71z`) show the same number as the plain printing. Their scores tie. Among equal scores, the plain printing comes first (it shows its own details and its number has no suffix; the catalog's `isPlain`), then pHash order. Before, the old scoring broke these ties by accident: it compared the reading with the catalog's own values (`PLST`, `WAR-47`, `254★`), which never appear on the card, so the plain printing won. Without a tie-break, today's model fell from 254 to 239 exact printings on the benchmark (#197). The ranking records whether the tie-break decided each printing's place, and the app's debug view shows it.

## Considered Options

Measured on the 1,291 card images of #196 replayed through each variant (#197), with PP-OCRv6 small. "Same name first" counts the cards whose printing scores strictly above every other printing of its name, out of 1,112:

- **Guessing each field, as before**: 643 same name first.
- **Candidate-driven, averaging over all fields**, a detail not shown scoring 0: 674. A printing that shows fewer details was penalised for details it doesn't print.
- **Candidate-driven, over the details shown**, adopted: 702. Old frames gain most (1993: 20 to 41), and the artist's field score rises from 71 to 93, where the blocklist and token filter used to drop or add words.
- **Candidate-driven, with positions** (the collector details on the bottom two lines, power/toughness in the rightmost token): 702 too, with no consistent gain across the tune and confirm halves. The copyright line reaches further right than the power/toughness, so the rightmost-token rule failed. Not adopted.

The benchmark was unchanged by each variant with v6 small (255 / 257 / 257). With today's PP-OCRv5_mobile, the adopted rules leave the totals of exact printing, card count and same card name unchanged against the baseline.

## Consequences

- No list of OCR misreads is kept apart from the slash characters of power/toughness. The extraction module and its tests are gone.
- One token can count as two details. A power/toughness of `2/2` read as `212` is also collector number 212, so a printing numbered 212 gains a detail it isn't showing. This decides one benchmark photo (`6_photos_single_card_textured_bg/b845b1d7…`) wrongly, between two plain printings that tie.
- When OCR misses details, a printing that shows fewer of them can score higher, since a detail it doesn't show isn't counted against it.
- Split cards still can't be told apart by name: their names are sideways, outside the name crop.
- On an OCR failure there is no reading, and the ranking is the candidates in pHash order, every score 0.
