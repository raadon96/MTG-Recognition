# Scoring counts how much of the collector information's digits each candidate explains

ADR 0008 scores each candidate over the details it shows, so a printing that shows few details has few ways to lose. When OCR misses something on the right printing, a printing that shows only a name and artist (Alpha, Collectors' Edition, Antiquities) or no number (Tempest) can win. For example, `266/281` and `AFR*EN` were read from a Commander Legends card, yet Collectors' Edition won with 90.0 against 84.2.

Scoring now also gives each candidate a **coverage**: the share of the collector information's runs of digits it explains. Coverage is one more entry in the breakdown, with weight 1. The other weights are unchanged (name 2, artist 0.5, the rest 1), and ADR 0008 otherwise still holds.

## The rule

The runs are the ones scoring already looks for details in: `O` next to a digit read as `0`, leading zeros dropped. The name crop isn't used. A run is explained when it is, for the candidate:

- its printed collector number, printed year, or printed base set size; the size counts only when the number is shown;
- the number and size glued together (`751350`, `7511350`), as the base set size is found;
- `1993` not followed by a year, when the candidate shows a year, as the year is found.

Coverage is 100 × explained runs / all runs; each run counts once, whatever its length. When the collector information holds no run of digits, coverage isn't shown and is left out of the score, so cards that print no digits keep their scores. Power/toughness digits stay unexplained.

## Results

Measured in #234 on #196's 1,291 card images, replayed with dev's engine (PP-OCRv6 tiny, 960 / max). "Same name first" counts the cards whose printing scores strictly above every other printing of its name, out of 1,112 (561 tune, 563 confirm):

- Benchmark: 255 / 257 / 257, with no image changed.
- Same name first: 631 → 650 (tune 307 → 318, confirm 333 → 341).

The 4 cards it loses on the confirm half come from wrong printed values in the catalog: the SDCC promos PSDC, PS14 and PS15 print their parent set's number and size, which the catalog says they don't show, and the French FBB Counterspell prints `©1994`, though the catalog says it shows no year. Those printings won only because they showed fewer details. Fixing their printed values (#201's rules) is separate work.

## Considered Options

- **Weighting each run by its length**, and **counting power/toughness digits as explained** for creatures: every form, at weights 0.5, 1 and 2, lies within 3 cards of the others on tune. Counting P/T digits never helped. Counting runs is the simplest.
- **Coverage weight 2**: 2 cards better than weight 1 on tune, but 3 more cards lost on confirm.
- **Tuning the weights alone** (artist 2, year 2): +6 same name first on each half, less than coverage. The grid over {0, 0.5, 1, 2} has no clear best: fifteen weight vectors tie on tune.
- **Tuning the weights with coverage** (set code 0.5, artist 1, power/toughness 2, coverage 2): 3–6 more cards than coverage alone, but it loses a benchmark image (`5_scans_multiple_cards/with_inner_gaps_with_border_gaps.jpg`) and 6 cards on confirm, and 42 vectors tie for best on tune, so it fits the tune half rather than the cards. It also raises power/toughness, which every printing of a card shares. Not adopted.

## Consequences

- Digits that belong to no detail (power/toughness, flavor text) and misread numbers (`300` for `308`) lower every candidate's coverage alike, the right one included.
- A printing whose catalog values say it shows fewer details than it prints now loses to the printings that explain the digits (the SDCC promos and FBB above).
