import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
from numpydantic import NDArray, Shape
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.engine.row import Row
from sqlmodel import Session, col, select
from tqdm import tqdm

from mtg_recognition.engine.ocr.ocr_utils import sort_scores_with_indices
from mtg_recognition.jobs.schemas import MobileCache

# --- Configuration ---
IMG_DIR = "data/reference/images"
INDEX_PATH = "data/reference/index/phash/hashes.npy"
MAPPING_PATH = "data/reference/index/phash/index_to_uuid.json"
HASH_SIZE = 16  # 16x16 = 256-bit hash (higher = more precise, but slower)


class PHashCardMatcher:
    """
    Initializes the pHash matcher.
    If database files exist, they are loaded.
    If not, the matcher is ready for build_index() to be called.

    Args:
        index_path: Path to save/load the numpy array of hash values
        mapping_path: Path to save/load the UUID mapping
        img_dir: Directory containing reference card images
        hash_size: Size of the perceptual hash (default 16 = 256-bit hash)
        auto_build: If True, automatically build index if missing
    """

    def __init__(
        self,
        index_path: str = INDEX_PATH,
        mapping_path: str = MAPPING_PATH,
        img_dir: str = IMG_DIR,
        hash_size: int = HASH_SIZE,
        auto_build: bool = True,
    ):

        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.img_dir_path = Path(img_dir)
        self.hash_size = hash_size

        # type and shape validation using numpydantic (https://github.com/p2p-ld/numpydantic)
        self.hashes: (
            NDArray[Shape[f"* x, {hash_size * hash_size} y"], np.uint8] | None
        ) = None
        self.uuid_mapping: list | None = None
        self.uuid_to_idx: dict[str, int] | None = (
            None  # Reverse mapping for O(1) lookup
        )
        self.auto_build = auto_build

        self._load_index()

    # def model_post_init(self, __context: Any) -> None:
    #     # Try to load existing index
    #     self._load_index()

    @property
    def _index_is_loaded(self):
        """Check if the pHash index is loaded."""
        return self.hashes is not None and self.uuid_mapping is not None

    @staticmethod
    def _compute_hash_worker(args):
        """
        Worker function for parallel hash computation.
        Must be a static method or top-level function for pickling.

        Args:
            args: Tuple of (image_path, hash_size)

        Returns:
            Tuple of (uuid, hash_array) or None if failed
        """
        image_path, hash_size = args

        try:
            # Open image
            img = Image.open(image_path).convert("RGB")

            # Compute perceptual hash
            phash = imagehash.phash(img, hash_size=hash_size)

            # Convert to numpy array
            hash_array = np.array(phash.hash).flatten().astype(np.uint8)

            # Extract UUID from filename
            uuid = image_path.stem

            return (uuid, hash_array)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def _load_index(self):
        """Internal method to load hashes and mapping from disk."""
        if self.index_path.exists() and self.mapping_path.exists():
            print(f"Loading pHashes from {self.index_path}...")
            try:
                # Load hash array
                self.hashes = np.load(self.index_path)

                # Load UUID mapping
                with open(self.mapping_path) as f:
                    self.uuid_mapping = json.load(f)

                # Build reverse mapping for O(1) UUID lookups
                self.uuid_to_idx = {
                    uuid: idx for idx, uuid in enumerate(self.uuid_mapping)
                }

                print(f"pHash index loaded. Contains {len(self.uuid_mapping)} cards.")

                assert self.hashes.shape[0] == len(self.uuid_mapping)  # Sanity check

            except Exception as e:
                print(f"Error loading pHash index: {e}")
        else:
            if self.auto_build:  # <-- Only build if flag is True
                print("Warning: pHash index files not found. Building index...")
                self.build_index()
            else:
                print(
                    "Warning: pHash index files not found. Call build_index() to create."
                )

    def build_index(self, force=False, num_workers=cpu_count() - 1):
        """
        Scans the reference directory and builds a pHash index using parallel processing.

        Args:
            force: If True, rebuilds even if index files exist
            num_workers: Number of parallel processes (default: cpu_count())
        """
        if self.index_path.exists() and self.mapping_path.exists():
            print(
                f"pHash index already exists at {self.index_path} and {self.mapping_path}..."
            )
            if not force:
                return

        print(f"Building pHash index from {self.img_dir}...")

        # Ensure output directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        assert self.index_path.parent.exists()

        # 1. Find all images
        image_paths = list(self.img_dir.glob("*.jpg"))
        if not image_paths:
            raise ValueError(f"Error: No images found in {self.img_dir}")

        print(f"Found {len(image_paths)} images.")
        print(f"Using {num_workers} parallel workers...")

        # 2. Prepare arguments for workers
        # Each worker gets (image_path, hash_size)
        worker_args = [(path, self.hash_size) for path in image_paths]

        # 3. Process in parallel with progress bar
        hash_list = []
        uuid_list = []

        with Pool(processes=num_workers) as pool:
            # imap_unordered returns results as they complete (faster than map)
            # chunksize determines how many tasks each worker gets at once
            # Optimal chunksize: total_images / (num_workers * 4)
            chunksize = max(1, len(image_paths) // (num_workers * 4))

            results = pool.imap_unordered(
                self._compute_hash_worker, worker_args, chunksize=chunksize
            )

            # Collect results with progress bar
            for result in tqdm(
                results, total=len(image_paths), desc="Computing pHashes"
            ):
                if result is not None:
                    uuid, hash_array = result
                    uuid_list.append(uuid)
                    hash_list.append(hash_array)

        if not hash_list:
            raise ValueError("No hashes were extracted.")

        # 4. Convert to numpy array
        all_hashes = np.array(hash_list, dtype=np.uint8)

        print(f"Saving {all_hashes.shape[0]} hashes to {self.index_path}...")
        np.save(self.index_path, all_hashes)

        print(f"Saving mapping to {self.mapping_path}...")
        with open(self.mapping_path, "w") as f:
            json.dump(uuid_list, f)

        # 5. Load into memory
        self.hashes = all_hashes
        self.uuid_mapping = uuid_list

        # Build reverse mapping for O(1) UUID lookups
        self.uuid_to_idx = {uuid: idx for idx, uuid in enumerate(uuid_list)}

        print("pHash index build complete and loaded!")

    def _compute_hash(self, image: np.ndarray | Image.Image) -> np.ndarray:
        """
        Compute perceptual hash for a given image.

        Args:
            image: PIL Image or numpy array (BGR from OpenCV)

        Returns:
            numpy array of the hash
        """
        # Convert OpenCV BGR to PIL RGB if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image[:, :, ::-1])

        phash = imagehash.phash(image, hash_size=self.hash_size)
        return np.array(phash.hash).flatten().astype(np.uint8)

    def _get_uuids_from_indices(self, indices: list[int]) -> list[str]:
        """Convert list of indices to UUIDs."""
        if self.uuid_mapping is None:
            raise ValueError("UUID mapping not loaded")
        return [self.uuid_mapping[idx] for idx in indices]

    def _get_indices_from_uuids(self, uuids: list[str]) -> list[int]:
        """Convert list of UUIDs to indices using O(1) lookup."""
        if self.uuid_to_idx is None:
            raise ValueError("UUID to index mapping not loaded")
        return [self.uuid_to_idx[uuid] for uuid in uuids]

    def _filter_by_uuids(self, uuids: list[str]) -> tuple[list[str], np.ndarray]:
        """
        Filter the hash index by specific UUIDs.

        Args:
            uuids: List of UUIDs to filter by (empty list = return all)

        Returns:
            Tuple of (filtered_uuid_mapping, filtered_hashes)
        """
        if not uuids:
            # No filtering - return everything
            if self.uuid_mapping is None or self.hashes is None:
                raise ValueError("Index not loaded")
            return self.uuid_mapping, self.hashes

        # Filter by UUIDs using O(1) lookup
        indices = self._get_indices_from_uuids(uuids)
        if self.hashes is None:
            raise ValueError("Hashes not loaded")
        filtered_hashes = self.hashes[indices]

        return uuids, filtered_hashes

    def find_best_matches(
        self,
        query_image,
        candidates: list[Row] = [],
        k=5,
    ) -> tuple[list[Row], list[float]]:
        """
        Finds the top k matching cards using Hamming distance.

        Args:
            query_image: Query image (numpy array in BGR format from OpenCV,
                        or PIL Image in RGB)
            candidates: Optional list of candidate Row objects to restrict search to
            k: Number of top matches to return

        Returns:
            Tuple of (matches, scores) where matches are Row objects and scores are floats (0-1)
        """
        if not self._index_is_loaded:
            raise ValueError("Error: pHash index not loaded. Call build_index() first.")

        # Filter by candidates if provided
        subset_uuids = (
            [candidate.uuid for candidate in candidates] if candidates else []
        )
        uuid_mapping, hashes = self._filter_by_uuids(subset_uuids)

        # Compute hash for query image
        query_hash = self._compute_hash(query_image)

        # Compute Hamming distances to all reference hashes (lower is more similar)
        # XOR operation: bits that differ = 1, same = 0
        # Sum gives total number of different bits (Hamming distance)
        distances = np.sum(np.bitwise_xor(hashes, query_hash), axis=1)
        # Convert distances to similarity scores (0 to 1, where 1 is identical)
        total_bits = self.hash_size * self.hash_size
        scores = 1.0 - (distances.astype(float) / float(total_bits))

        # Get top K indices (highest similarities = best matches)
        if k > len(scores):
            k = len(scores)

        top_k_scores, top_k_indices = sort_scores_with_indices(scores, k)
        top_k_uuids = [uuid_mapping[idx] for idx in top_k_indices]

        # Query database for Row objects
        from mtg_recognition.database import engine

        statement = select(MobileCache).where(col(MobileCache.uuid).in_(top_k_uuids))
        with Session(engine) as session:
            results = session.exec(statement).all()

        # Reorder results based on score (since IN clause does not guarantee return order)
        uuid_to_match = {row.uuid: row for row in results}
        top_k_matches = [
            uuid_to_match[uuid] for uuid in top_k_uuids if uuid in uuid_to_match
        ]

        return top_k_matches, top_k_scores

    def get_distance(
        self, query_image: np.ndarray | Image.Image, reference_uuid: str
    ) -> int:
        """
        Get the Hamming distance between a query image and a specific reference card.
        Useful for verification of a candidate match.

        Args:
            query_image: Query image
            reference_uuid: UUID of the reference card

        Returns:
            Hamming distance (int)
        """
        if self.uuid_to_idx is None:
            raise ValueError("Index not loaded.")
        if reference_uuid not in self.uuid_to_idx:
            raise ValueError(f"UUID {reference_uuid} not found in index.")

        # Find index of the reference UUID using O(1) lookup
        ref_idx = self.uuid_to_idx[reference_uuid]
        # Compute query hash
        query_hash = self._compute_hash(query_image)
        # Compute distance
        if self.hashes is None:
            raise ValueError("Hashes not loaded")
        distance = np.sum(np.bitwise_xor(self.hashes[ref_idx], query_hash))

        return int(distance)

    def get_similarity(
        self, query_image: np.ndarray | Image.Image, reference_uuid: str
    ) -> float:
        """
        Get the similarity score between a query image and a specific reference card.
        Similarity is in range [0, 1] where 1 is identical.

        Args:
            query_image: Query image
            reference_uuid: UUID of the reference card

        Returns:
            Similarity score (float, 0-1)
        """
        distance = self.get_distance(query_image, reference_uuid)
        total_bits = self.hash_size * self.hash_size
        similarity = 1.0 - (float(distance) / float(total_bits))
        return similarity


def main():
    # Test with a sample segmented test image
    query_image_path = Path(
        "benchmarks/dataset/images/ready/0a2f85cd-f66d-5330-a9d5-314554f871a1.jpg",
        # "benchmarks/dataset/images/ready/40bc1031-6cc8-5e10-9ff3-12f8a7b70c78.jpg"
    )
    assert query_image_path.exists()

    matcher = PHashCardMatcher(auto_build=True)
    matcher.build_index(force=False)

    import cv2

    k = 10
    query_img = cv2.imread(str(query_image_path))
    best_matches, best_scores = matcher.find_best_matches(query_img, k=k)

    print(f"\nTop {k} matches:")
    for i, match in enumerate(best_matches):
        print(f"  UUID: {match.uuid}, name: {match.name}, Score: {best_scores[i]}")


# --- Example Usage ---
if __name__ == "__main__":
    main()


# uv run python -m mtg_recognition.engine.matching.phash_matcher
