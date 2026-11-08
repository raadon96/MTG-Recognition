"""Throwaway: how many catalog rows each rule in rules.py decides, how often the
derived value differs from the catalog's, and what the added fields cost in size.

Size: copies the catalog to <tmp dir>, adds the printed-detail columns and
`layout`, fills them from the rules, VACUUMs, and compares the file size with an
unchanged copy, VACUUMed the same way. Measured: `layout` alone; every value
stored ("full"); and only values that differ from the catalog's ("sparse": NULL
means the catalog's value, '' means not printed), both with `layout`.

uv run python docs/research/printed-details/coverage.py \
    <AllPrintings.sqlite> <catalog.sqlite> <tmp dir> <out.json>
"""

import json
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from rules import printed_details, printed_sources

DETAILS = ["set_code", "number", "base_set_size", "year"]
COLUMNS = {
    "set_code": "printedSetCode",
    "number": "printedNumber",
    "base_set_size": "printedBaseSetSize",
    "year": "printedYear",
}


def main(allprintings: str, catalog: str, tmp_dir: str, out_path: str) -> None:
    ap = sqlite3.connect(f"file:{allprintings}?mode=ro", uri=True)
    ap.row_factory = sqlite3.Row
    sets = {r["code"]: dict(r) for r in ap.execute("select * from sets")}
    cards = [
        dict(r)
        for r in ap.execute(
            """select c.uuid, c.setCode, c.number, c.frameVersion,
                      c.originalReleaseDate, c.layout, c.promoTypes, c.name, c.artist,
                      c.availability like '%paper%' as paper
               from cards c"""
        )
    ]
    cat = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True)
    catalog_values = {
        uuid: {"set_code": code, "number": number, "base_set_size": size, "year": year}
        for uuid, code, number, size, year in cat.execute(
            "select uuid, setCode, number, baseSetSize, setReleaseYear from catalog"
        )
    }
    cat.close()

    # coverage: per detail and rule, rows and paper rows, and how many differ
    sources = printed_sources(cards, sets)
    derived = {c["uuid"]: printed_details(c, sets, sources) for c in cards}
    coverage: dict = {}
    for detail in DETAILS:
        counts: Counter = Counter()
        for c in cards:
            d = derived[c["uuid"]]
            value, rule = d[detail], d["rule"][detail]
            catalog_value = catalog_values[c["uuid"]][detail]
            same = value is not None and str(catalog_value) == value
            for scope in ("rows", "paper") if c["paper"] else ("rows",):
                counts[(rule, scope, "all")] += 1
                counts[(rule, scope, "same" if same else "differs")] += 1
        coverage[detail] = {
            rule: {
                scope: {
                    "rows": counts[(rule, scope, "all")],
                    "same as catalog": counts[(rule, scope, "same")],
                }
                for scope in ("rows", "paper")
            }
            for rule in sorted({k[0] for k in counts})
        }

    # size: one copy as it is, one with the added columns, both VACUUMed
    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    sizes = {"catalog on disk": Path(catalog).stat().st_size}
    for variant in ("unchanged", "layout only", "full", "sparse"):
        path = tmp / f"catalog-{variant.replace(' ', '-')}.sqlite"
        shutil.copyfile(catalog, path)
        con = sqlite3.connect(path)
        if variant == "layout only":
            con.execute("alter table catalog add column layout text")
            con.executemany(
                "update catalog set layout = ? where uuid = ?",
                [(c["layout"], c["uuid"]) for c in cards],
            )
            con.commit()
        elif variant != "unchanged":
            for column in [*COLUMNS.values(), "layout"]:
                con.execute(f"alter table catalog add column {column} text")
            assignments = ", ".join(f"{c} = ?" for c in COLUMNS.values())
            con.executemany(
                f"update catalog set {assignments}, layout = ? where uuid = ?",
                [
                    (
                        *(
                            stored(
                                variant,
                                derived[c["uuid"]][d],
                                catalog_values[c["uuid"]][d],
                            )
                            for d in DETAILS
                        ),
                        c["layout"],
                        c["uuid"],
                    )
                    for c in cards
                ],
            )
            con.commit()
        con.execute("vacuum")
        con.close()
        sizes[variant] = path.stat().st_size

    result = {"coverage": coverage, "sizes": sizes, "rows": len(cards)}
    Path(out_path).write_text(json.dumps(result, indent=1))
    print(json.dumps(sizes, indent=1))
    for detail in DETAILS:
        print(f"\n{detail}: rule | rows | same as catalog | paper rows | same")
        for rule, c in sorted(
            coverage[detail].items(), key=lambda kv: -kv[1]["rows"]["rows"]
        ):
            print(
                f"  {rule} | {c['rows']['rows']} | {c['rows']['same as catalog']} | "
                f"{c['paper']['rows']} | {c['paper']['same as catalog']}"
            )


def stored(variant: str, value: str | None, catalog_value) -> str | None:
    if variant == "full":
        return value
    # sparse: NULL when the catalog's value is printed, '' when nothing is
    if value is None:
        return ""
    return None if str(catalog_value) == value else value


if __name__ == "__main__":
    main(*sys.argv[1:5])
