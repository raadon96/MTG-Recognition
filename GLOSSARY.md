# MTG Recognition

Identifies Magic: The Gathering cards in photos and scans by matching them against reference data covering every printing.

## Language

**Printing**:
One edition of a card in a specific set, with its own collector number, artist and image; identified by a UUID. The result of recognition is always a printing, never just a card name.
_Avoid_: card (when the edition matters), row, match (for the thing matched)

**Card**:
The physical object in a photo or scan, or a card name independent of edition (e.g. "Lightning Bolt", which has many printings).
_Avoid_: printing (when the edition is unknown or irrelevant)

**Candidate**:
A printing shortlisted because its image looks like the card, so that it can be inspected more closely.
_Avoid_: match, hit

**Ranking**:
The candidates for one card after closer inspection, ordered from most to least likely, each with a score.
_Avoid_: matches, results, top-k

**Reading**:
The details read off one card's image before they are compared with any printing: its name, power/toughness and the information below the text box (set code, collector number, illustration credit, copyright year). A reading can be wrong or incomplete, which is why it is scored rather than looked up.
_Avoid_: properties, OCR result, printing properties

**Score**:
How well a card's reading agrees with a printing's details, from 0 to 100; higher is better.
_Avoid_: confidence (it is not a probability), distance

**Similarity**:
How alike a card's image looks to a printing's reference image, from 0 to 1; higher is more alike. It shortlists candidates and is never shown as a score.
_Avoid_: score, confidence

**Reference data**:
Everything built from one MTGJSON release and used to recognise cards: the MobileCache database and the pHash index (both committed), and the reference images the index is built from (a local cache, never versioned).
_Avoid_: reference catalogue, assets
