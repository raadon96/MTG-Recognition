from abc import ABC, abstractmethod

import numpy as np


class CardMatcher(ABC):
    @abstractmethod
    def find_best_matches(
        self,
        query_image: np.ndarray,
        candidate_uuids: list[str] | None = None,
        k: int = 1,
    ) -> tuple[list[str], list[float]]:
        """Return the UUIDs of the k most similar printings and their scores, best first."""
        ...

    @abstractmethod
    def _load_index(self) -> None: ...

    @abstractmethod
    def build_index(self, force: bool) -> None: ...
