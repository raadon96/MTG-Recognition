import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
import imagehash
import numpy as np
from numpydantic import NDArray, Shape
from PIL import Image
from tqdm import tqdm

from mtg_recognition.core.ranking import sort_scores_with_indices
from mtg_recognition.settings import settings

# --- Configuration ---
HASH_SIZE = 16  # 16x16 = 256-bit hash (higher = more precise, but slower)
DEFAULT_NUM_WORKERS = cpu_count() - 1


class PHashCardMatcher:
    """
    Initializes the pHash matcher.
    If index files exist on disk, they are loaded automatically.
    If not, the matcher is ready for build_index() to be called explicitly.

    Args:
        index_path: Path to save/load the numpy array of hash values
        mapping_path: Path to save/load the UUID mapping
        img_dir: Directory containing reference card images
        hash_size: Size of the perceptual hash (default 16 = 256-bit hash)
    """

    def __init__(
        self,
        index_path: Path = settings.PHASH_INDEX_FILE,
        mapping_path: Path = settings.PHASH_MAPPING_FILE,
        img_dir: Path = settings.IMG_DIR,
        hash_size: int = HASH_SIZE,
    ):

        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.img_dir = img_dir
        self.hash_size = hash_size

        # type and shape validation using numpydantic
        # (https://github.com/p2p-ld/numpydantic)
        self.hashes: (
            NDArray[Shape[f"* x, {hash_size * hash_size} y"], np.uint8] | None
        ) = None
        self.uuid_mapping: list | None = None

        self._load_index()

    @property
    def index_loaded(self):
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

                print(f"pHash index loaded. Contains {len(self.uuid_mapping)} cards.")

                assert self.hashes.shape[0] == len(self.uuid_mapping)  # Sanity check

            except Exception as e:
                print(f"Error loading pHash index: {e}")
        else:
            print("Warning: pHash index files not found. Call build_index() to create.")

    def build_index(self, force=False, num_workers=DEFAULT_NUM_WORKERS):
        """
        Scans the reference directory and builds a pHash index using parallel
        processing.

        Args:
            force: If True, rebuilds even if index files exist
            num_workers: Number of parallel processes (default: cpu_count())
        """
        if self.index_path.exists() and self.mapping_path.exists():
            print(
                f"pHash index already exists at {self.index_path} and "
                f"{self.mapping_path}..."
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

    def find_best_matches(
        self,
        query_image,
        k=5,
    ) -> tuple[list[str], list[float]]:
        """
        Finds the top k matching printings using Hamming distance.

        Args:
            query_image: Query image (numpy array in BGR format from OpenCV,
                        or PIL Image in RGB)
            k: Number of top matches to return

        Returns:
            Tuple of (uuids, scores), best first; scores are floats (0-1)
        """
        hashes, uuid_mapping = self.hashes, self.uuid_mapping
        if hashes is None or uuid_mapping is None:
            raise ValueError("Error: pHash index not loaded. Call build_index() first.")

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
        top_k_scores, top_k_indices = sort_scores_with_indices(scores, k)
        top_k_uuids = [uuid_mapping[idx] for idx in top_k_indices]

        return top_k_uuids, top_k_scores


@click.command()
@click.option(
    "--force",
    is_flag=True,
    default=False,
    show_default=True,
    help="Rebuild pHash index even if it already exists.",
)
@click.option(
    "--workers",
    type=click.INT,
    default=cpu_count() - 1,
    show_default=True,
    help="Number of parallel worker processes for hash computation.",
)
def build_index(force: bool, workers: int) -> None:
    """Rebuild the pHash index from reference images already on disk.

    Requires data/reference/images/ to be populated first (run 'mtg sync'
    to download reference images if you haven't already).
    """
    # Preflight check (D-10): image directory must exist and be non-empty.
    img_dir = settings.IMG_DIR
    if not img_dir.exists() or not any(img_dir.iterdir()):
        click.echo(
            "Error: data/reference/images/ is missing or empty. Run 'mtg sync' first.",
            err=True,
        )
        raise SystemExit(1)

    matcher = PHashCardMatcher()
    matcher.build_index(force=force, num_workers=workers)
