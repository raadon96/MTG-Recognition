"""End-to-end recognition on real UUID-named images (requires `dvc pull` and the reference data)."""

from pathlib import Path

import pytest

from mtg_recognition.core.image_utils import load_image
from mtg_recognition.engine.pipeline import Pipeline

RAW_BENCHMARK = Path("data/test/6_benchmark/images/raw")
UUIDS = [
    "01013cab-be73-532f-b6bf-0f24d0c3792f",
    "0a2f85cd-f66d-5330-a9d5-314554f871a1",
    "0c941678-45c6-5aab-a863-136c1e0b7a7f",
]


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return Pipeline(auto_build=False)


@pytest.mark.slow
@pytest.mark.parametrize("uuid", UUIDS)
def test_recognizes_the_printing_named_by_the_file(pipeline, uuid):
    img_path = RAW_BENCHMARK / f"{uuid}.jpg"
    if not img_path.exists():
        pytest.skip(f"{img_path} missing; run `uv run dvc pull`")

    result = pipeline.recognize(load_image(img_path))

    assert result.error is None
    [segment] = result.segments
    assert segment.ranked[0].printing.uuid == uuid
