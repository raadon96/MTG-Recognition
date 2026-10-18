"""Compare two runs under /tmp/acc6: top-1 count, top-1 UUID flips, segment-count changes, timings.

usage: uv run python docs/research/accuracy-recon/compare.py <baseline-run> <new-run>
Committed runs (run1, run2, issue23) can be copied back to /tmp/acc6/<name>/ first.
"""

import json
import pathlib
import sys

R = pathlib.Path("/tmp/acc6")


def load(run):
    res = {}
    for f in sorted((R / run).glob("*.json")):
        for img in json.loads(f.read_text()):
            s = img["segments"]
            top1 = s[0]["matches"][0]["uuid"] if s and s[0]["matches"] else None
            res[(f.stem, pathlib.Path(img["image"]).stem)] = (bool(s) and s[0]["success"] is True, top1, len(s))
    return res


base_run, new_run = sys.argv[1], sys.argv[2]
a, b = load(base_run), load(new_run)
print(f"{base_run}: {sum(v[0] for v in a.values())}/{len(a)} | {new_run}: {sum(v[0] for v in b.values())}/{len(b)}")
print("top-1 uuid flips:", [k for k in a if a[k][1] != b.get(k, (None, None, None))[1]])
print("segment-count changes:", [k for k in a if a[k][2] != b.get(k, (None, None, None))[2]])
times = [line.split() for line in (R / new_run / "times.txt").read_text().splitlines()]
print("times:", [(c, n, f"{float(s):.0f}s") for c, n, s in times], f"total {sum(float(t[2]) for t in times):.0f}s")
