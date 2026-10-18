import json
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpydantic import NDArray, Shape
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from mtg_recognition.engine.ocr.ocr_extraction import sort_scores_with_indices

# --- Configuration ---
IMG_DIR = "data/reference/images"
INDEX_PATH = "data/reference/index/mobilenet_v3_small/embeddings.npy"
MAPPING_PATH = "data/reference/index/mobilenet_v3_small/index_to_uuid.json"
EMBEDDING_BATCH_SIZE = 32  # Sweet spot for MobileNetV3 on CPU.
DEVICE = "cpu"  # RPi5 has no CUDA, so CPU.


class CardDataset(Dataset):
    """
    Custom Dataset to load images efficiently for PyTorch DataLoader.
    """

    def __init__(self, img_dir, transform=None):
        self.img_dir: Path = Path(img_dir)
        self.transform = transform
        self.image_files = list(self.img_dir.glob("*.jpg"))

        if not self.image_files:
            raise ValueError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        try:
            # Loading with PIL is standard for torchvision transforms
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            # Extract UUID from filename (e.g. "abc-123.jpg" -> "abc-123")
            # uuid = img_name.rsplit(".", 1)[0]
            uuid = img_path.stem  # Using pathlib
            return image, uuid

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy tensor and None UUID to handle errors gracefully
            # (In production, you might want to filter these out beforehand)
            return torch.zeros((3, 224, 224)), "ERROR"

    # def __getitems__(self, idxs):
    #     # TODO: test this version
    #     # NOTE: might be more efficient with numpy array of strings
    #     getter = operator.itemgetter(*idxs)
    #     img_paths = list(getter(self.image_files))

    #     try:
    #         # Loading with PIL is standard for torchvision transforms
    #         images = [Image.open(img_path).convert("RGB") for img_path in img_paths]

    #         if self.transform:
    #             images = [self.transform(image) for image in images]

    #         # Extract UUID from filename (e.g. "abc-123.jpg" -> "abc-123")
    #         uuids = [img_path.stem for img_path in img_paths]  # Using pathlib
    #         return images, uuids

    #     except Exception as e:
    #         print(f"Error loading {img_paths[-1]}: {e}")  # TODO: return correct path
    #         # Return a dummy tensor and None UUID to handle errors gracefully
    #         # (In production, you might want to filter these out beforehand)
    #         dummy_images = [torch.zeros((3, 224, 224)) for _ in idxs]
    #         dummy_errors = ["ERROR" for _ in idxs]
    #         return dummy_images, dummy_errors


class DeepCardMatcher(BaseModel):
    """
    Initializes the deep matcher.
    If database files exist, they are loaded.
    If not, the matcher is ready for build_index() to be called.

    Args:
        index_path: Path to save/load the numpy array of embeddings
        mapping_path: Path to save/load the UUID mapping
        img_dir: Directory containing reference card images
        auto_build: If True, automatically build database if missing
    """

    index_path: Path = Path(INDEX_PATH)
    mapping_path: Path = Path(MAPPING_PATH)
    img_dir: Path = Path(IMG_DIR)
    device: str = DEVICE
    embeddings: NDArray | None = None
    uuid_mapping: list | None = None
    auto_build: bool = True
    _model: Any = None
    _transform: Any = None

    def model_post_init(self, __context: Any) -> None:
        # 1. Initialize Feature Extractor (MobileNetV3-Small)
        print("Initializing feature extractor...")
        self._model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        # Remove classifier to get raw feature embeddings (576-dim)
        self._model.classifier = torch.nn.Identity()
        self._model.eval()
        self._model.to(self.device)

        # 2. Define Preprocessing
        self._transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 3. Try to load index
        self._load_index()

    @property
    def index_loaded(self):
        """Check if the pHash index is loaded."""
        return self.embeddings is not None and self.uuid_mapping is not None

    def _load_index(self):
        if self.index_path.exists() and self.mapping_path.exists():
            print(f"Loading index from {self.index_path}...")
            try:
                self.embeddings = np.load(self.index_path)
                with open(self.mapping_path) as f:
                    self.uuid_mapping = json.load(f)
                print(f"Index loaded. Contains {len(self.uuid_mapping)} cards.")
            except Exception as e:
                print(f"Error loading index: {e}")
        else:
            if self.auto_build:
                print("Index files not found. Building index...")
                self.build_index()
            else:
                print("Warning: Index files not found. Call build_index() to create.")

    def build_index(self, force=False, num_workers=cpu_count() - 1):
        """
        Scans the reference directory, extracts and saves embeddings using optimized DataLoader.
        """
        if self.index_path.exists() and self.mapping_path.exists():
            print("Embeddings already exist. Use force=True to rebuild.")
            if not force:
                return

        print(f"Building index from {self.img_dir} with {num_workers} workers...")

        # Ensure output directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        assert self.index_path.parent.exists()

        # 1. Create Dataset and DataLoader
        dataset = CardDataset(self.img_dir, transform=self._transform)
        dataloader = DataLoader(
            dataset,
            batch_size=EMBEDDING_BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
        )

        embeddings_list = []
        uuid_list = []

        # 2. Process in Batches
        with torch.no_grad():  # Disable gradient calculation for speed
            for batch_imgs, batch_uuids in tqdm(
                dataloader, desc="Extracting embeddings"
            ):
                # Filter out any errors during loading
                valid_mask = [u != "ERROR" for u in batch_uuids]
                if not any(valid_mask):
                    continue

                # Filter tensors and uuids
                batch_imgs = batch_imgs[valid_mask].to(self.device)
                valid_uuids = [u for i, u in enumerate(batch_uuids) if valid_mask[i]]

                # Inference
                embeddings = self._model(batch_imgs)
                embeddings = embeddings.view(embeddings.size(0), -1)

                # L2 Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                embeddings_list.append(embeddings.cpu().numpy())
                uuid_list.extend(valid_uuids)

        if not embeddings_list:
            raise ValueError("No embeddings were extracted.")

        # 3. Aggregate and Save
        all_embeddings = np.concatenate(embeddings_list, axis=0)

        print(f"Saving {all_embeddings.shape[0]} vectors to {self.index_path}...")
        np.save(self.index_path, all_embeddings)

        print(f"Saving mapping to {self.mapping_path}...")
        with open(self.mapping_path, "w") as f:
            json.dump(uuid_list, f)

        # 4. Refresh Memory
        self.embeddings = all_embeddings
        self.uuid_mapping = uuid_list
        print("Index build complete and loaded!")

    def _get_uuids_from_idxs(self, idx_list):
        uuids = [self.uuid_mapping[idx] for idx in idx_list]
        return uuids

    def _get_idxs_from_uuids(self, uuid_list):
        idxs = [self.uuid_mapping.index(uuid) for uuid in uuid_list]
        return idxs

    def _filter_by_uuids_or_idxs(self, subset_uuids: list[str], subset_idxs: list[int]):
        """
        Filter the index by UUIDs or indices.

        If both UUIDs and indices are given, raises a ValueError.
        If only UUIDs are given, returns the corresponding indices and UUIDs.
        If only indices are given, returns the corresponding UUIDs and embeddings.
        If neither UUIDs nor indices are given, returns the entire UUID mapping and embeddings.
        """
        if subset_uuids and subset_idxs:
            raise ValueError(
                "Error: Cannot filter by both UUIDs and indices at the same time."
            )
        elif subset_uuids:
            subset_idxs = self._get_idxs_from_uuids(subset_uuids)
            uuid_mapping = subset_uuids
        elif subset_idxs:
            uuid_mapping = self._get_uuids_from_idxs(subset_idxs)
        else:
            uuid_mapping = self.uuid_mapping
        embeddings = self.embeddings[subset_idxs] if subset_idxs else self.embeddings
        return uuid_mapping, embeddings

    def find_best_matches(
        self,
        query_image_bgr: np.ndarray,
        candidate_uuids: list[str] | None = None,
        k: int = 5,
    ) -> tuple[list[str], list[float]]:
        if not self.index_loaded:
            raise ValueError("Error: Index not loaded.")

        # Filter if subset specified
        subset_uuids = candidate_uuids or []
        subset_idxs = []
        uuid_mapping, embeddings = self._filter_by_uuids_or_idxs(
            subset_uuids, subset_idxs
        )

        # Convert OpenCV (BGR) to PIL (RGB)
        img_rgb = Image.fromarray(query_image_bgr[:, :, ::-1])

        # Preprocess
        input_tensor = self._transform(img_rgb).unsqueeze(0).to(self.device)

        # Extract Query Embedding
        with torch.no_grad():
            f = self._model(input_tensor)
            f = f.view(f.size(0), -1)
            f = torch.nn.functional.normalize(f, p=2, dim=1)
            query_vector = f.cpu().numpy().flatten()

        # Vectorized Cosine Similarity (higher is more similar)
        scores = np.dot(embeddings, query_vector)

        # Get top K indices (highest similarities = best matches)
        if k > len(scores):
            k = len(scores)

        top_k_scores, top_k_indices = sort_scores_with_indices(scores, k)
        top_k_uuids = [uuid_mapping[idx] for idx in top_k_indices]

        return top_k_uuids, top_k_scores


def main():
    # Test with a sample segmented test image
    query_image_path = Path(
        # "benchmarks/dataset/images/ready/cd9aa095-f999-5a2c-91e5-4acd94d45416.jpg",
        "benchmarks/dataset/images/ready/40bc1031-6cc8-5e10-9ff3-12f8a7b70c78.jpg",
        # "benchmarks/dataset/images/raw/0a2f85cd-f66d-5330-a9d5-314554f871a1.jpg"
    )
    assert query_image_path.exists()

    matcher = DeepCardMatcher(auto_build=True)
    matcher.build_index(force=False)

    k = 5
    query_img = cv2.imread(str(query_image_path))
    uuids, scores = matcher.find_best_matches(query_img, k=k)

    print(f"\nTop {k} matches:")
    for uuid, score in zip(uuids, scores, strict=True):
        print(f"  UUID: {uuid}, Score: {score}")


if __name__ == "__main__":
    main()

# uv run python -m mtg_recognition.engine.matching.deep_matcher
