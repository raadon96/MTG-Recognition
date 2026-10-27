import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

from mtg_recognition.reference import ReferenceData

HASH_SIZE = 16  # 16x16 = 256-bit hash
DEFAULT_NUM_WORKERS = max(1, cpu_count() - 1)


@dataclass(slots=True)
class PHashIndex:
    hashes: np.ndarray  # (printings, 32) uint8, one bit-packed fingerprint per row
    uuids: list[str]  # the printing UUID of each row


def load_phash_index(reference: ReferenceData) -> PHashIndex:
    """Read the pHash index of `reference`.

    Raises:
        FileNotFoundError: If either index file is missing
        ValueError: If the number of hashes differs from the number of UUIDs
    """
    index_path, mapping_path = reference.phash_files
    for path in (index_path, mapping_path):
        if not path.exists():
            raise FileNotFoundError(
                f"pHash index file {path} not found. Run `mtg build-index` first."
            )

    hashes = np.load(index_path)
    uuids = json.loads(mapping_path.read_text())
    if hashes.shape[0] != len(uuids):
        raise ValueError(
            f"pHash index holds {hashes.shape[0]} hashes but {len(uuids)} UUIDs. "
            "Run `mtg build-index` to rebuild it."
        )
    # the file stores one byte per bit; pack 8 per byte for `shortlist`
    return PHashIndex(hashes=np.packbits(hashes, axis=1), uuids=uuids)


def build_phash_index(
    images: Mapping[str, Path],
    reference: ReferenceData,
    workers: int = DEFAULT_NUM_WORKERS,
) -> dict[str, str]:
    """Hash `images` (printing UUID -> image file) and write the pHash index of
    `reference`, replacing any earlier one.

    Args:
        images: The image of each printing to index
        reference: Reference data the index files are written to
        workers: Number of parallel processes; 1 hashes in this process, without
            a process pool

    Returns:
        The error of each image that couldn't be hashed, by UUID; those
        printings are left out of the index

    Raises:
        ValueError: If no image could be hashed
    """
    hashes: list[np.ndarray] = []
    uuids: list[str] = []
    failures: dict[str, str] = {}
    for uuid, bits, error in tqdm(
        _compute_hashes(list(images.items()), workers),
        total=len(images),
        desc="Computing pHashes",
    ):
        if error is not None:
            failures[uuid] = error
        else:
            uuids.append(uuid)
            hashes.append(bits)

    if not hashes:
        raise ValueError("No hashes were extracted.")

    index_path, mapping_path = reference.phash_files
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(index_path, np.array(hashes, dtype=np.uint8))
    mapping_path.write_text(json.dumps(uuids))
    return failures


def shortlist(
    index: PHashIndex, image: np.ndarray, k: int
) -> tuple[list[str], list[float]]:
    """The `k` printings of `index` whose fingerprint is closest to `image`'s.

    Args:
        index: The pHash index to search
        image: The card, as a BGR array
        k: Number of printings to return; clamped to the index size

    Returns:
        `(uuids, similarities)`, best first; a similarity is `1 - hamming / 256`,
        from 0 to 1, where 1 means identical fingerprints
    """
    bits = _phash_bits(Image.fromarray(image[:, :, ::-1]))
    # Hamming distance: XOR sets the bits that differ, bitwise_count counts them
    differing = np.bitwise_xor(index.hashes, np.packbits(bits))
    distances = np.bitwise_count(differing).sum(axis=1, dtype=np.int32)
    similarities = 1.0 - distances / bits.size

    # top k, best first: argpartition then a stable sort of the k only
    k = min(k, len(index.uuids))
    top = np.argpartition(-similarities, k - 1)[:k]
    top = top[np.argsort(-similarities[top], kind="stable")]
    return [index.uuids[i] for i in top], similarities[top].tolist()


def _compute_hashes(
    items: list[tuple[str, Path]], workers: int
) -> Iterator[tuple[str, np.ndarray | None, str | None]]:
    """Yield `_compute_hash_worker` results, in this process when `workers` is 1,
    else from a process pool.
    """
    if workers == 1:
        yield from map(_compute_hash_worker, items)
        return

    with Pool(processes=workers) as pool:
        # imap_unordered yields results as they complete; about four chunks
        # per worker balances the load
        chunksize = max(1, len(items) // (workers * 4))
        yield from pool.imap_unordered(_compute_hash_worker, items, chunksize=chunksize)


def _compute_hash_worker(
    item: tuple[str, Path],
) -> tuple[str, np.ndarray | None, str | None]:
    """Hash one image: `(uuid, bits, None)`, or `(uuid, None, error)` when the
    image can't be read. Module-level so `multiprocessing.Pool` can pickle it.
    """
    uuid, path = item
    try:
        return uuid, _phash_bits(Image.open(path).convert("RGB")), None
    except Exception as e:  # PIL raises OSError, SyntaxError, ValueError...
        return uuid, None, f"{type(e).__name__}: {e}"


def _phash_bits(image: Image.Image) -> np.ndarray:
    """The fingerprint of `image`: its pHash as a flat uint8 array of bits."""
    return imagehash.phash(image, hash_size=HASH_SIZE).hash.flatten().astype(np.uint8)
