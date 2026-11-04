from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

from mtg_recognition.reference.catalog import read_phashes
from mtg_recognition.reference.layout import ReferenceData

HASH_SIZE = 16  # 16x16 = 256-bit hash
DEFAULT_NUM_WORKERS = max(1, cpu_count() - 1)


@dataclass(slots=True)
class PHashes:
    hashes: np.ndarray  # (printings, 32) uint8, one bit-packed pHash per row
    uuids: list[str]  # the printing UUID of each row


def load_phashes(reference: ReferenceData) -> PHashes:
    """Read every pHash in the catalog of `reference`. A printing without a pHash
    is left out, so it is never a candidate.

    Raises:
        ValueError: If the catalog holds no pHash
    """
    rows = read_phashes(reference.catalog_path)
    if not rows:
        raise ValueError(
            f"The catalog {reference.catalog_path} holds no pHash. "
            "Run `mtg reference update` first."
        )
    return phashes_from_rows(rows)


def phashes_from_rows(rows: list[tuple[str, bytes]]) -> PHashes:
    """The `PHashes` of `(uuid, phash)` rows, each pHash packed 8 bits per byte
    as the catalog stores it; the rows keep their order.
    """
    hashes = np.frombuffer(b"".join(phash for _, phash in rows), dtype=np.uint8)
    return PHashes(
        hashes=hashes.reshape(len(rows), -1), uuids=[uuid for uuid, _ in rows]
    )


def compute_phashes(
    images: Mapping[str, Path], workers: int = DEFAULT_NUM_WORKERS
) -> tuple[dict[str, bytes], dict[str, str]]:
    """Hash `images` (printing UUID -> image file).

    Args:
        images: The reference image of each printing to hash
        workers: Number of parallel processes; 1 hashes in this process, without
            a process pool

    Returns:
        `(phashes, failures)`: the pHash of each printing, packed 8 bits per byte
        as the catalog stores it, and the error of each image that couldn't be
        hashed, by UUID
    """
    phashes: dict[str, bytes] = {}
    failures: dict[str, str] = {}
    for uuid, bits, error in tqdm(
        _compute_hashes(list(images.items()), workers),
        total=len(images),
        desc="Computing pHashes",
    ):
        if error is not None:
            failures[uuid] = error
        else:
            phashes[uuid] = np.packbits(bits).tobytes()
    return phashes, failures


def shortlist(
    phashes: PHashes, image: np.ndarray, k: int
) -> tuple[list[str], list[float]]:
    """The `k` printings in `phashes` whose pHash is closest to `image`'s.

    Args:
        phashes: The pHashes to search
        image: The card, as a BGR array
        k: Number of printings to return; clamped to the number of pHashes

    Returns:
        `(uuids, similarities)`, best first; a similarity is `1 - hamming / 256`,
        from 0 to 1, where 1 means identical pHashes
    """
    bits = _phash_bits(Image.fromarray(image[:, :, ::-1]))
    # Hamming distance: XOR sets the bits that differ, bitwise_count counts them
    differing = np.bitwise_xor(phashes.hashes, np.packbits(bits))
    distances = np.bitwise_count(differing).sum(axis=1, dtype=np.int32)
    similarities = 1.0 - distances / bits.size

    # top k, best first, ties in catalog order: distances are whole bits, so ties
    # are common, and argpartition orders them differently on different CPUs
    # (#146). A stable sort of every pHash takes ~3 ms per card, so find the
    # k-th distance and stable-sort only the printings at least that close.
    k = min(k, len(phashes.uuids))
    kth = np.partition(distances, k - 1)[k - 1]
    top = np.flatnonzero(distances <= kth)
    top = top[np.argsort(distances[top], kind="stable")][:k]
    return [phashes.uuids[i] for i in top], similarities[top].tolist()


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
    """The pHash of `image` as a flat uint8 array of bits."""
    return imagehash.phash(image, hash_size=HASH_SIZE).hash.flatten().astype(np.uint8)
