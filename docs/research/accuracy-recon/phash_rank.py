"""Rank of the expected UUID among the top-200 pHash candidates for selected misses."""
import pathlib, sys
from mtg_recognition.engine.pipeline import Pipeline
from mtg_recognition.core.image_utils import load_and_resize_image

p = Pipeline(auto_build=True)
for path in sys.argv[1:]:
    path = pathlib.Path(path)
    img = load_and_resize_image(path)
    seg = p.segmentor.segment_batch([img])[0]
    warped = seg.segments[0].warped_image
    m, s = p.phash_matcher.find_best_matches(warped, k=200)
    uu = [r.uuid for r in m]
    r = uu.index(path.stem) + 1 if path.stem in uu else None
    print(path.stem[:8], "pHash rank of expected:", r, "| top1:", m[0].name, m[0].setCode, m[0].number)
