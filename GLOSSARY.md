# MTG Recognition

Identifies Magic: The Gathering cards in photos and scans by matching them against a reference catalogue of every printing.

## Language

**Printing**:
One edition of a card in a specific set, with its own collector number, artist and image; identified by a UUID. The result of recognition is always a printing, never just a card name.
_Avoid_: card (when the edition matters), row, match (for the thing matched)

**Card**:
The physical object in a photo or scan, or a card name independent of edition (e.g. "Lightning Bolt", which has many printings).
_Avoid_: printing (when the edition is unknown or irrelevant)
