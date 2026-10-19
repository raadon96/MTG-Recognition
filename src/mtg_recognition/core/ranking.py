import numpy as np


def sort_scores_with_indices(
    scores: np.ndarray, k: int | None = None, threshold: float = 0.0
) -> tuple[list[float], list[int]]:
    """
    Return top-k scores (descending) and their indices, using argpartition for
    efficiency.

    Parameters:
    - scores: 1-D numpy array of numeric scores
    - k: number of top elements to return (default: None, returns all valid scores)
    - threshold: only scores >= threshold are considered (default: 0.0)

    Returns:
    - top_scores_desc: 1-D numpy array of the top-k scores in descending order
    - top_indices_desc: 1-D numpy array of the corresponding indices in the original
      scores
    """
    # Ensure input is a NumPy array
    scores = np.asarray(scores)

    if scores.ndim != 1:
        raise ValueError("scores must be a 1-D array")

    if k is not None and k <= 0:
        raise ValueError("k must be > 0")

    # Filter by threshold
    valid_mask = scores >= threshold
    valid_scores = scores[valid_mask]
    valid_indices = np.nonzero(valid_mask)[0]

    if valid_scores.size == 0:
        return [], []

    # If k is None, return all valid scores sorted
    if k is None:
        sort_order = np.argsort(-valid_scores)
        return valid_scores[sort_order].tolist(), valid_indices[sort_order].tolist()

    # If k is larger than number of valid scores, clamp it
    k = min(k, valid_scores.size)

    # Use argpartition to get indices of the top-k values in descending order.
    # We partition based on the (k-1)th largest position in the *negative* scores
    # to emulate top-k in descending order.
    # Compute indices within the valid_scores array
    topk_pos_in_valid = np.argpartition(-valid_scores, k - 1)[:k]

    # Gather the top-k scores and their original indices
    topk_scores = valid_scores[topk_pos_in_valid]
    topk_indices = valid_indices[topk_pos_in_valid]

    # Sort the top-k locally in descending order
    sort_order = np.argsort(-topk_scores, kind="stable")
    top_scores_desc = topk_scores[sort_order]
    top_indices_desc = topk_indices[sort_order]

    return top_scores_desc.tolist(), top_indices_desc.tolist()
