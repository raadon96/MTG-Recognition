import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from mtg_recognition.engine.matching.phash import (
    PHashIndex,
    _phash_bits,
    build_phash_index,
    load_phash_index,
    shortlist,
)
from mtg_recognition.reference.layout import ReferenceData

from ..sync_fakes import jpeg


def write_jpegs(directory: Path, names: list[str]) -> dict[str, Path]:
    """One distinct JPEG per name, `<name>.jpg`, by printing UUID `uuid-<name>`."""
    images = {}
    for seed, name in enumerate(names):
        images[f"uuid-{name}"] = directory / f"{name}.jpg"
        images[f"uuid-{name}"].write_bytes(jpeg(seed))
    return images


@pytest.mark.unit
class TestBuildPHashIndex:
    def test_unhashable_image_is_reported_and_left_out(self, tmp_path) -> None:
        good, truncated = tmp_path / "good.jpg", tmp_path / "truncated.jpg"
        good.write_bytes(jpeg(0))
        truncated.write_bytes(jpeg(1)[:200])
        reference = ReferenceData(tmp_path / "reference")

        failures = build_phash_index(
            {"uuid-good": good, "uuid-truncated": truncated}, reference, workers=1
        )

        assert list(failures) == ["uuid-truncated"]
        assert load_phash_index(reference).uuids == ["uuid-good"]


@pytest.mark.unit
class TestShortlist:
    def test_an_indexed_image_shortlists_its_own_printing_first(self, tmp_path) -> None:
        reference = ReferenceData(tmp_path / "reference")
        build_phash_index(write_jpegs(tmp_path, ["a", "b", "c"]), reference, workers=1)
        query = cv2.imread(str(tmp_path / "b.jpg"))
        assert query is not None

        uuids, similarities = shortlist(load_phash_index(reference), query, k=5)

        assert len(uuids) == 3
        assert uuids[0] == "uuid-b"
        assert similarities[0] == pytest.approx(1.0)

    def test_similarities_match_the_unpacked_hamming_distance(self, tmp_path) -> None:
        reference = ReferenceData(tmp_path / "reference")
        build_phash_index(write_jpegs(tmp_path, ["a", "b", "c"]), reference, workers=1)
        query = cv2.imread(str(tmp_path / "a.jpg"))
        assert query is not None
        index_path, mapping_path = reference.phash_files
        hashes, uuids = np.load(index_path), json.loads(mapping_path.read_text())
        bits = _phash_bits(Image.fromarray(query[:, :, ::-1]))
        expected = dict(zip(uuids, 1 - (hashes != bits).sum(axis=1) / 256, strict=True))

        shortlisted, similarities = shortlist(load_phash_index(reference), query, k=3)

        assert similarities == [expected[uuid] for uuid in shortlisted]

    def test_tied_printings_keep_index_order(self, tmp_path) -> None:
        # Hamming distances are whole bits, so ties are common; their order must
        # not depend on the CPU (#146)
        reference = ReferenceData(tmp_path / "reference")
        build_phash_index(write_jpegs(tmp_path, ["a"]), reference, workers=1)
        bits = np.unpackbits(load_phash_index(reference).hashes[0])
        one_bit_off = bits.copy()
        one_bit_off[0] ^= 1
        # 1000 tied printings, then the printing with the identical fingerprint
        rows = [one_bit_off] * 1000 + [bits]
        index = PHashIndex(
            hashes=np.packbits(np.array(rows), axis=1),
            uuids=[f"uuid-{i}" for i in range(len(rows))],
        )
        query = cv2.imread(str(tmp_path / "a.jpg"))
        assert query is not None

        uuids, similarities = shortlist(index, query, k=20)

        assert uuids == ["uuid-1000"] + [f"uuid-{i}" for i in range(19)]
        assert similarities == [1.0] + [255 / 256] * 19


@pytest.mark.unit
class TestLoadPHashIndex:
    def test_missing_files_raise(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="mtg build-index"):
            load_phash_index(ReferenceData(tmp_path / "reference"))

    def test_more_hashes_than_uuids_raise(self, tmp_path) -> None:
        reference = ReferenceData(tmp_path / "reference")
        build_phash_index(write_jpegs(tmp_path, ["a", "b"]), reference, workers=1)
        _, mapping_path = reference.phash_files
        mapping_path.write_text(json.dumps(json.loads(mapping_path.read_text())[:-1]))

        with pytest.raises(ValueError, match="mtg build-index"):
            load_phash_index(reference)
