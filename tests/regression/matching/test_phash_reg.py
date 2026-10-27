"""pHash coarse search on pre-warped UUID-named benchmark images (requires
`dvc pull` and the pHash index).
"""

from pathlib import Path

import cv2
import pytest

from mtg_recognition.engine.matching.phash import (
    PHashIndex,
    load_phash_index,
    shortlist,
)
from mtg_recognition.reference import ReferenceData

READY_BENCHMARK = Path("data/test/6_benchmark/images/ready")
# The Pipeline's default k_phash_candidates: the printing must survive into OCR.
K = 20

IMG_PATHS = sorted(READY_BENCHMARK.glob("*.jpg"))


@pytest.fixture(scope="module")
def index() -> PHashIndex:
    return load_phash_index(ReferenceData.default())


@pytest.mark.slow
@pytest.mark.skipif(
    not IMG_PATHS, reason=f"{READY_BENCHMARK} missing; run `uv run dvc pull`"
)
@pytest.mark.parametrize("img_path", IMG_PATHS, ids=lambda p: p.stem)
def test_top_k_contains_the_printing_named_by_the_file(index, img_path):
    image = cv2.imread(str(img_path))
    assert image is not None
    uuids, _ = shortlist(index, image, k=K)

    assert img_path.stem in uuids
