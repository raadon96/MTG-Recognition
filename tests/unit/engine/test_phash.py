import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from mtg_recognition.engine.matching.phash import (
    PHashIndex,
    _phash_bits,
    compute_phashes,
    load_phash_index,
    phash_index,
    shortlist,
)
from mtg_recognition.jobs.sync import generate_image_uri, update
from mtg_recognition.reference.layout import ReferenceData

from ..sync_fakes import PRINTINGS_V1, jpeg


def write_jpegs(directory: Path, names: list[str]) -> dict[str, Path]:
    """One distinct JPEG per name, `<name>.jpg`, by printing UUID `uuid-<name>`."""
    images = {}
    for seed, name in enumerate(names):
        images[f"uuid-{name}"] = directory / f"{name}.jpg"
        images[f"uuid-{name}"].write_bytes(jpeg(seed))
    return images


def index_of(images: dict[str, Path]) -> PHashIndex:
    """The pHashes of `images`, in the order given, as the catalog would hold them."""
    phashes, failures = compute_phashes(images, workers=1)
    assert failures == {}
    return phash_index([(uuid, phashes[uuid]) for uuid in images])


@pytest.mark.unit
class TestComputePHashes:
    def test_unhashable_image_is_reported_and_left_out(self, tmp_path) -> None:
        good, truncated = tmp_path / "good.jpg", tmp_path / "truncated.jpg"
        good.write_bytes(jpeg(0))
        truncated.write_bytes(jpeg(1)[:200])

        phashes, failures = compute_phashes(
            {"uuid-good": good, "uuid-truncated": truncated}, workers=1
        )

        assert list(phashes) == ["uuid-good"]
        assert list(failures) == ["uuid-truncated"]

    def test_phash_is_256_bits_packed_into_32_bytes(self, tmp_path) -> None:
        phashes, _ = compute_phashes(write_jpegs(tmp_path, ["a"]), workers=1)

        assert len(phashes["uuid-a"]) == 32


@pytest.mark.unit
class TestShortlist:
    def test_an_indexed_image_shortlists_its_own_printing_first(self, tmp_path) -> None:
        index = index_of(write_jpegs(tmp_path, ["a", "b", "c"]))
        query = cv2.imread(str(tmp_path / "b.jpg"))
        assert query is not None

        uuids, similarities = shortlist(index, query, k=5)

        assert len(uuids) == 3
        assert uuids[0] == "uuid-b"
        assert similarities[0] == pytest.approx(1.0)

    def test_similarities_match_the_unpacked_hamming_distance(self, tmp_path) -> None:
        images = write_jpegs(tmp_path, ["a", "b", "c"])
        query = cv2.imread(str(tmp_path / "a.jpg"))
        assert query is not None
        bits = _phash_bits(Image.fromarray(query[:, :, ::-1]))
        expected = {
            uuid: 1 - (_phash_bits(Image.open(path).convert("RGB")) != bits).sum() / 256
            for uuid, path in images.items()
        }

        shortlisted, similarities = shortlist(index_of(images), query, k=3)

        assert similarities == [expected[uuid] for uuid in shortlisted]

    def test_tied_printings_keep_index_order(self, tmp_path) -> None:
        # Hamming distances are whole bits, so ties are common; their order must
        # not depend on the CPU (#146)
        bits = np.unpackbits(index_of(write_jpegs(tmp_path, ["a"])).hashes[0])
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
    def test_phashes_load_in_catalog_order_without_the_printings_lacking_one(
        self, tmp_path, network
    ) -> None:
        reference = ReferenceData(tmp_path / "reference")
        del network.images[generate_image_uri(PRINTINGS_V1[1][2], "front")]
        with network.client() as client:
            update(reference, client, workers=1)

        index = load_phash_index(reference)

        assert index.uuids == ["uuid-bolt", "uuid-aberration"]
        assert index.hashes.shape == (2, 32)

    def test_catalog_without_phashes_raises(self, tmp_path, network) -> None:
        reference = ReferenceData(tmp_path / "reference")
        with network.client() as client:
            update(reference, client, workers=1)
        with sqlite3.connect(reference.catalog_path) as connection:
            connection.execute("UPDATE catalog SET phash = NULL")

        with pytest.raises(ValueError, match="mtg update"):
            load_phash_index(reference)
