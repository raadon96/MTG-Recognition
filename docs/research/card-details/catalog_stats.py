# ruff: noqa: E501 -- throwaway research script; its SQL strings stay on one line
"""Throwaway: shares and traps measured on AllPrintings (the catalog's rows are
AllPrintings' cards table, one row per printing).

uv run python docs/research/card-details/catalog_stats.py <AllPrintings.sqlite>
"""

import sqlite3
import sys

QUERIES = {
    "rows": "select count(*) from cards",
    "frame share": """
        select frameVersion, count(*), round(100.0*count(*)/(select count(*) from cards),2)
        from cards group by 1 order by 2 desc""",
    "layout share": """
        select layout, count(*), round(100.0*count(*)/(select count(*) from cards),2)
        from cards group by 1 order by 2 desc""",
    # Year against frame: the set's release year per frame
    "frame by set release year (min, max)": """
        select c.frameVersion, min(s.releaseDate), max(s.releaseDate)
        from cards c join sets s on c.setCode = s.code group by 1""",
    "printings per (frame, release year bucket)": """
        select c.frameVersion,
          case when s.releaseDate < '1997-10-01' then 'a <1997-10'
               when s.releaseDate < '2003-07-28' then 'b 1997-10..2003-07'
               when s.releaseDate < '2014-07-18' then 'c 2003-07..2014-07'
               else 'd >=2014-07' end as bucket, count(*)
        from cards c join sets s on c.setCode = s.code group by 1, 2 order by 1, 2""",
    "old frame in a later era, by set (top 25)": """
        select c.setCode, s.name, substr(s.releaseDate,1,4), c.frameVersion, count(*)
        from cards c join sets s on c.setCode = s.code
        where (c.frameVersion in ('1993','1997') and s.releaseDate >= '2003-07-28')
           or (c.frameVersion = '2003' and s.releaseDate >= '2014-07-18')
        group by 1, 4 order by 5 desc limit 25""",
    "old frame in a later era, total": """
        select count(*) from cards c join sets s on c.setCode = s.code
        where (c.frameVersion in ('1993','1997') and s.releaseDate >= '2003-07-28')
           or (c.frameVersion = '2003' and s.releaseDate >= '2014-07-18')""",
    "2015 frame before July 2014, by set (top 10)": """
        select c.setCode, s.name, s.releaseDate, count(*)
        from cards c join sets s on c.setCode = s.code
        where c.frameVersion = '2015' and s.releaseDate < '2014-07-18'
        group by 1 order by 4 desc limit 10""",
    # The catalog's year is the set's release year; promos keep their original card
    "originalReleaseDate differs from the set's year": """
        select count(*) from cards c join sets s on c.setCode = s.code
        where c.originalReleaseDate is not null
          and substr(c.originalReleaseDate,1,4) != substr(s.releaseDate,1,4)""",
    "set released in Oct-Dec (copyright year may be the previous year)": """
        select count(*) from cards c join sets s on c.setCode = s.code
        where substr(s.releaseDate,6,2) in ('10','11','12')""",
    # Collector numbers
    "numbers that are not plain digits, by pattern (top 20)": """
        select case when number glob '*[0-9]' and number not glob '*[^0-9]*' then 'digits'
                    when number like '%-%' then 'has -'
                    when number glob '*[0-9][a-z]' then 'digits+letter'
                    when number glob '*★' then 'digits+star'
                    when number glob '[A-Z]*' then 'starts with letter'
                    else 'other' end, count(*)
        from cards group by 1 order by 2 desc""",
    "number above baseSetSize (out of the main run)": """
        select count(*) from cards c join sets s on c.setCode = s.code
        where cast(c.number as integer) > s.baseSetSize""",
    "baseSetSize = 0 or null": """
        select count(*) from cards c join sets s on c.setCode = s.code
        where s.baseSetSize is null or s.baseSetSize = 0""",
    "sets before Exodus (1998-06-15), printings": """
        select count(*) from cards c join sets s on c.setCode = s.code
        where s.releaseDate < '1998-06-15'""",
    # Set codes that can't be printed as they are
    "set type shares": """
        select s.type, count(*), round(100.0*count(*)/(select count(*) from cards),2)
        from cards c join sets s on c.setCode = s.code group by 1 order by 2 desc""",
    "printings in sets with a parentCode (promo/token/etc), top 15": """
        select s.type, count(*) from cards c join sets s on c.setCode = s.code
        where s.parentCode is not null group by 1 order by 2 desc limit 15""",
    "The List (PLST) sample numbers": """
        select c.number, c.name, c.frameVersion from cards c where c.setCode = 'PLST' limit 8""",
    "The List (PLST) rows and frames": """
        select c.frameVersion, count(*) from cards c where c.setCode = 'PLST' group by 1""",
    "online-only printings": """
        select count(*) from cards c join sets s on c.setCode = s.code
        where s.isOnlineOnly = 1 or c.isOnlineOnly = 1""",
    "borderless / showcase / extendedart / fullart shares": """
        select sum(borderColor='borderless'), sum(frameEffects like '%showcase%'),
               sum(frameEffects like '%extendedart%'), sum(isFullArt=1),
               sum(isTextless=1), sum(flavorName is not null),
               sum(printedName is not null)
        from cards""",
    # Faces: layouts whose printings print two names on one front
    "faces per layout": """
        select layout, side, count(*) from cards
        where layout in ('split','adventure','aftermath','flip','prepare','transform',
                         'modal_dfc','meld','reversible_card') group by 1, 2""",
    "power present by frame": """
        select frameVersion, sum(power is not null), count(*) from cards group by 1""",
}


def main(allprintings_path: str) -> None:
    con = sqlite3.connect(f"file:{allprintings_path}?mode=ro", uri=True)
    for title, sql in QUERIES.items():
        print(f"## {title}")
        for row in con.execute(sql):
            print("  ", row)


if __name__ == "__main__":
    main(sys.argv[1])
