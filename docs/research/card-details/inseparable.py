# ruff: noqa: E501 -- throwaway research script; its SQL strings stay on one line
"""Throwaway: printings that OCR alone can't tell apart, so that pHash has to.

Two views:
- catalog: printings whose catalog fields used in scoring are all equal, so their
  scores are equal whatever is read.
- printed: printings whose printed details are equal, under the rules measured in
  ocr_positions.py (no number before Exodus, set code only from Magic 2015, promos
  print the parent's code, The List prints the original set's code and number).

uv run python docs/research/card-details/inseparable.py <AllPrintings.sqlite>
"""

import re
import sqlite3
import sys
from collections import Counter, defaultdict

DOUBLE_FACED_LAYOUTS = {"transform", "modal_dfc", "reversible_card"}


def main(allprintings: str) -> None:
    con = sqlite3.connect(f"file:{allprintings}?mode=ro", uri=True)
    rows = con.execute(
        """select c.uuid, c.name, c.faceName, c.layout, c.setCode, c.number, c.artist,
                  c.power, c.toughness, c.frameVersion, s.baseSetSize, s.releaseDate,
                  s.parentCode, s.type, c.availability
           from cards c join sets s on c.setCode = s.code"""
    ).fetchall()

    catalog_groups = defaultdict(list)
    printed_groups = defaultdict(list)
    for (
        uuid,
        name,
        face,
        layout,
        set_code,
        number,
        artist,
        power,
        toughness,
        frame,
        base_size,
        released,
        parent,
        set_type,
        availability,
    ) in rows:
        name = face if layout in DOUBLE_FACED_LAYOUTS else name
        year = released[:4]
        catalog_key = (
            name,
            set_code,
            number,
            base_size,
            artist,
            year,
            power,
            toughness,
        )
        catalog_groups[catalog_key].append((uuid, frame, layout, set_code))

        # printed details
        printed_set = None
        printed_number = None
        if "-" in number:  # The List: XLN-123
            printed_set, number = number.split("-", 1)
        digits = re.sub(r"\D", "", number).lstrip("0") or None
        if released >= "1998-06-15" and frame != "1993":
            printed_number = digits
        if frame == "2015" and released >= "2014-07-18":
            printed_set = printed_set or (
                parent if set_type == "promo" and parent else set_code
            )
        printed_key = (name, artist, power, toughness, printed_set, printed_number)
        printed_groups[printed_key].append(
            (uuid, frame, layout, set_code, set_type, "paper" in (availability or ""))
        )

    report("catalog: equal scoring fields", catalog_groups, lambda m: (m[1], m[2]))
    report("printed: equal printed details", printed_groups, lambda m: m[1])
    # which kinds of sets the printed collisions come from
    by_type = Counter(
        m[4] for members in printed_groups.values() if len(members) > 1 for m in members
    )
    print("printed collisions by set type:", by_type.most_common(12))
    paper_only = defaultdict(list)
    for key, members in printed_groups.items():
        paper = [m for m in members if m[5]]
        if len(paper) > 1:
            paper_only[key] = paper
    paper_total = sum(m[5] for members in printed_groups.values() for m in members)
    paper_shared = sum(len(m) for m in paper_only.values())
    print(
        f"## printed, paper printings only: {paper_shared} of {paper_total} ({100 * paper_shared / paper_total:.1f}%) share their key"
    )
    for k, n in Counter(
        (m[1], m[4]) for members in paper_only.values() for m in members
    ).most_common(15):
        print("   ", k, n)
    examples = sorted(paper_only.items(), key=lambda kv: -len(kv[1]))[:12]
    for key, members in examples:
        print("  ", key, len(members), sorted({m[3] for m in members})[:10])


def report(title: str, groups: dict, kind) -> None:
    total = sum(len(m) for m in groups.values())
    shared = [m for members in groups.values() if len(members) > 1 for m in members]
    print(
        f"## {title}: {len(shared)} of {total} printings ({100 * len(shared) / total:.1f}%) share their key"
    )
    for k, n in Counter(kind(m) for m in shared).most_common(12):
        print("   ", k, n)


if __name__ == "__main__":
    main(sys.argv[1])
