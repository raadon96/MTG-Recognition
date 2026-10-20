import numpy as np
import pytest

from mtg_recognition.core.ranking import sort_scores_with_indices


class TestSortScoresWithIndices:
    def test_default_parameters_returns_all_sorted(self):
        """Test that k=None and threshold=0.0 returns all scores in descending order"""
        scores = np.array([0.5, 0.9, 0.3, 0.95, 0.7])
        top_scores, top_indices = sort_scores_with_indices(scores)
        assert top_scores == [0.95, 0.9, 0.7, 0.5, 0.3]
        assert top_indices == [3, 1, 4, 0, 2]

    def test_top_k_selection(self):
        """Test that k parameter correctly limits the number of results"""
        scores = np.array([0.5, 0.9, 0.3, 0.95, 0.7])
        top_scores, top_indices = sort_scores_with_indices(scores, k=3)
        assert len(top_scores) == 3
        assert top_scores == [0.95, 0.9, 0.7]
        assert top_indices == [3, 1, 4]

    def test_threshold_filtering(self):
        """Test that threshold parameter filters out lower scores"""
        scores = np.array([0.5, 0.9, 0.3, 0.95, 0.7])
        top_scores, top_indices = sort_scores_with_indices(scores, threshold=0.6)
        assert top_scores == [0.95, 0.9, 0.7]
        assert top_indices == [3, 1, 4]

    def test_k_and_threshold_combined(self):
        """Test that k and threshold work together correctly"""
        scores = np.array([0.5, 0.9, 0.3, 0.95, 0.7, 0.8])
        top_scores, top_indices = sort_scores_with_indices(scores, k=2, threshold=0.6)
        assert len(top_scores) == 2
        assert top_scores == [0.95, 0.9]
        assert top_indices == [3, 1]

    def test_empty_result_all_below_threshold(self):
        """Test that empty lists are returned when all scores are below threshold"""
        scores = np.array([0.1, 0.2, 0.3])
        top_scores, top_indices = sort_scores_with_indices(scores, threshold=0.5)
        assert top_scores == []
        assert top_indices == []

    def test_k_larger_than_valid_scores(self):
        """Test that k larger than available scores returns all valid scores"""
        scores = np.array([0.5, 0.9, 0.3])
        top_scores, top_indices = sort_scores_with_indices(scores, k=10)
        assert len(top_scores) == 3
        assert top_scores == [0.9, 0.5, 0.3]

    def test_invalid_k_raises_error(self):
        """Test that k <= 0 raises ValueError"""
        scores = np.array([0.5, 0.9, 0.3])
        with pytest.raises(ValueError, match="k must be > 0"):
            sort_scores_with_indices(scores, k=0)
        with pytest.raises(ValueError, match="k must be > 0"):
            sort_scores_with_indices(scores, k=-1)

    def test_multidimensional_array_raises_error(self):
        """Test that non-1D arrays raise ValueError"""
        scores = np.array([[0.5, 0.9], [0.3, 0.7]])
        with pytest.raises(ValueError, match="scores must be a 1-D array"):
            sort_scores_with_indices(scores)

    def test_single_element_array(self):
        """Test edge case with single element"""
        scores = np.array([0.75])
        top_scores, top_indices = sort_scores_with_indices(scores)
        assert top_scores == [0.75]
        assert top_indices == [0]

    def test_duplicate_scores(self):
        """Test that duplicate scores are handled correctly"""
        scores = np.array([0.9, 0.7, 0.9, 0.5])
        top_scores, top_indices = sort_scores_with_indices(scores, k=2)
        assert len(top_scores) == 2
        assert top_scores == [0.9, 0.9]
        # Both indices 0 and 2 should be present
        assert 0 in top_indices
        assert 2 in top_indices
