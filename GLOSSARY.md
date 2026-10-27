# MTG Recognition

Identifies Magic: The Gathering cards in photos and scans by matching them against reference data covering every printing.

## Language

### Cards and printings

**Printing**:
One edition of a card in a specific set, with its own collector number, artist and image; identified by a UUID. Each face of a double-faced card is a separate printing. The result of recognition is always a printing, never just a card name.
_Avoid_: card (when the edition matters), row, match (for the thing matched)

**Card**:
The physical object in a photo or scan. For a card independent of edition (e.g. "Lightning Bolt", which has many printings), say **card name**.
_Avoid_: card (for the name), printing (when the edition is unknown or irrelevant)

### Printed details

**Collector information**:
The strip below a card's text box holding its set code, collector number, rarity, language, illustration credit and legal text.
_Avoid_: info line, info ROI, footer

**Set code**:
The short code of the set a printing belongs to (e.g. `MH3`), printed in the collector information.
_Avoid_: setcode, set, edition

**Collector number**:
A printing's number within its set: the `123` in `123/280`.
_Avoid_: number, card number

**Base set size**:
The number of cards in a set's main run: the `280` in `123/280`, printed after the collector number on many cards.
_Avoid_: set size, total

**Artist**:
The illustrator named in a card's illustration credit.
_Avoid_: illustrator

### Images

**Query image**:
The photo or scan submitted for recognition, holding zero, one or several cards.
_Avoid_: image (alone), upload, input

**Card image**:
One card cut out of a query image and straightened into a flat, upright rectangle.
_Avoid_: segment, warped image, crop

**Reference image**:
The official image of a printing, as published by Scryfall, that a card image is compared with.
_Avoid_: card image, Scryfall image

### Recognition

**Recognition**:
Turning a query image into a ranking for each card in it: segmentation, then matching.
_Avoid_: identification, detection, prediction

**Segmentation**:
The stage that finds each card in a query image and turns it into a card image.
_Avoid_: instance segmentation (one technique within it), detection, extraction

**Matching**:
The stage that turns a card image into a ranking: candidates are shortlisted by similarity, then each is scored against the card's reading.
_Avoid_: match (as a noun)

**Candidate**:
A printing shortlisted because its reference image looks like the card image, so that it can be inspected more closely.
_Avoid_: match, hit

**Ranking**:
The candidates for one card after closer inspection, ordered from most to least likely, each with a score.
_Avoid_: matches, results, top-k

**Best printing**:
The first entry of a card's ranking: the printing the card is recognised as.
_Avoid_: best match, top match, top printing

**Reading**:
The details read off one card image before they are compared with any printing: its name, power/toughness and its collector information (set code, collector number, illustration credit, copyright year). A reading can be wrong or incomplete, which is why it is scored rather than looked up.
_Avoid_: properties, OCR result, printing properties

**Score**:
How well a card's reading agrees with a printing's details, from 0 to 100; higher is better.
_Avoid_: confidence (it is not a probability), distance

**Similarity**:
How alike a card image looks to a printing's reference image, from 0 to 1; higher is more alike. It shortlists candidates and is never shown as a score.
_Avoid_: score, confidence

### Reference data

**Reference data**:
Everything built from one MTGJSON release and used to recognise cards: the catalog and the pHash index (both committed), and the reference images the index is built from (a local cache, never versioned).
_Avoid_: assets

**Catalog**:
The details of every printing (name, set, collector number, artist, and so on), looked up by the printing's UUID. Part of the reference data.
_Avoid_: MobileCache, cache, database

**pHash index**:
The perceptual hash of every printing's reference image, searched by similarity to shortlist candidates. Part of the reference data.
_Avoid_: index (alone), hash database, catalogue

### Benchmark

**Benchmark**:
The fixed set of labelled query images that recognition is measured against, covering every case the engine must handle: one or several cards, on a photo or a scan.
_Avoid_: test images, test data

**Label**:
What a query image should be recognised as: the expected printing of each card in it, and so how many cards it holds.
_Avoid_: ground truth (in names), UUID image

**Baseline**:
The benchmark results the maintainer has adopted as the reference that new results are compared with.
_Avoid_: reference (that is the reference data)
