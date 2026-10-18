"""Summarise /tmp/acc6/run{1,2}: per-category top-1, flips, timings, OCR failures."""

import json
import pathlib
import re

R = pathlib.Path("/tmp/acc6")


def load(run):
    res, segs = {}, {}
    for f in sorted((R / run).glob("*.json")):
        cat = f.stem
        for img in json.loads(f.read_text()):
            key = (cat, pathlib.Path(img["image"]).stem)
            s = img["segments"]
            segs[key] = len(s)
            # single-card image: success iff any segment... use segment 0 as the CLI does per segment
            res[key] = bool(s) and s[0]["success"] is True
            res[key + ("top1",)] = s[0]["matches"][0]["uuid"] if s and s[0]["matches"] else None
    return res, segs


def times(run):
    return {
        line.split()[0]: (int(line.split()[1]), float(line.split()[2]))
        for line in (R / run / "times.txt").read_text().splitlines()
    }


def ocr_fail(run):
    return {f.stem: len(re.findall("OCR prediction failed", f.read_text())) for f in (R / run).glob("*.log")}


runs = {r: load(r) for r in ("run1", "run2")}
cats = sorted({k[0] for k in runs["run1"][0] if len(k) == 2})
print("| Category | n | Run 1 top-1 | Run 2 top-1 | Run 1 time (s) | s/img | Run 2 time (s) | OCR failures r1/r2 |")
print("|---|---|---|---|---|---|---|---|")
tot = {"run1": [0, 0], "run2": [0, 0]}
t = {r: times(r) for r in runs}
of = {r: ocr_fail(r) for r in runs}
for c in cats:
    row = [c]
    keys = [k for k in runs["run1"][0] if len(k) == 2 and k[0] == c]
    row.append(str(len(keys)))
    for r in runs:
        ok = sum(runs[r][0].get(k, False) for k in keys)
        tot[r][0] += ok
        tot[r][1] += len(keys)
        row.append(f"{ok}/{len(keys)} ({100 * ok / len(keys):.1f} %)")
    n, s1 = t["run1"][c]
    row += [f"{s1:.0f}", f"{s1 / n:.2f}", f"{t['run2'][c][1]:.0f}", f"{of['run1'].get(c, 0)}/{of['run2'].get(c, 0)}"]
    print("| " + " | ".join(row) + " |")
print("| **Total** |", tot["run1"][1], "|", *[f"{tot[r][0]}/{tot[r][1]} ({100 * tot[r][0] / tot[r][1]:.1f} %) |" for r in runs],
      f"{sum(v[1] for v in t['run1'].values()):.0f} |", f"{sum(v[1] for v in t['run1'].values()) / tot['run1'][1]:.2f} |",
      f"{sum(v[1] for v in t['run2'].values()):.0f} |", f"{sum(of['run1'].values())}/{sum(of['run2'].values())} |")

print("\nFlips (result or top-1 uuid changed between runs):")
for k in sorted(k for k in runs["run1"][0] if len(k) == 2):
    a, b = runs["run1"][0][k], runs["run2"][0].get(k)
    ta, tb = runs["run1"][0][k + ("top1",)], runs["run2"][0].get(k + ("top1",))
    if a != b or ta != tb:
        print(" ", k, a, "->", b, ta, "->", tb)

print("\nMulti-segment / zero-segment images (run1, run2):")
for k in sorted(runs["run1"][1]):
    a, b = runs["run1"][1][k], runs["run2"][1].get(k)
    if a != 1 or b != 1:
        print(" ", k, a, b)

print("\nMisses in run 1:")
for k in sorted(k for k in runs["run1"][0] if len(k) == 2 and not runs["run1"][0][k]):
    print(" ", k)

# same UUID across categories: does the result agree?
by = {}
for k in runs["run1"][0]:
    if len(k) == 2:
        by.setdefault(k[1], []).append((k[0], runs["run1"][0][k]))
print("\nDuplicate UUIDs with disagreeing results (run1):")
for u, v in by.items():
    if len(v) > 1 and len({x[1] for x in v}) > 1:
        print(" ", u, v)
uniq_ok = sum(any(x[1] for x in v) for v in by.values())
print(f"\nUnique UUIDs: {len(by)}; any-copy correct: {uniq_ok}")
