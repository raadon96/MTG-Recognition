from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class CardMatcher(ABC):
    @abstractmethod
    def find_best_matches(
        self,
        query_image: np.ndarray,
        subset_uuids: list[str] = [],
        subset_idxs: list[int] = [],
        k: int = 1,
    ) -> list[dict]: ...

    @abstractmethod
    def _load_index(self) -> None: ...

    @abstractmethod
    def build_index(self, force: bool) -> None: ...
