"""pHash shortlisting on card images named by the UUID of their printing,
committed as fixtures so pHash is tested without segmentation (requires the
pHash index).
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

FIXTURE_DIR = Path("tests/regression/matching/fixtures")
# The Pipeline's default k_phash_candidates: the printing must survive into OCR.
K = 20

IMG_PATHS = sorted(FIXTURE_DIR.glob("*.jpg"))


@pytest.fixture(scope="module")
def index() -> PHashIndex:
    return load_phash_index(ReferenceData.default())


@pytest.mark.slow
@pytest.mark.parametrize("img_path", IMG_PATHS, ids=lambda p: p.stem)
def test_top_k_contains_the_printing_named_by_the_file(index, img_path):
    image = cv2.imread(str(img_path))
    assert image is not None
    uuids, _ = shortlist(index, image, k=K)

    assert img_path.stem in uuids
