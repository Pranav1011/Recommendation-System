"""
Unit tests for recommendation metrics.
"""

import pytest

from src.training.metrics import (
    compute_all_metrics,
    coverage,
    hit_rate_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRecallAtK:
    """Test Recall@K metric."""

    def test_perfect_recall(self):
        """Test perfect recall (all relevant items in top-K)."""
        predictions = [[1, 2, 3, 4, 5]]
        ground_truth = [{1, 2, 3}]
        k = 5

        recall = recall_at_k(predictions, ground_truth, k)
        assert recall == 1.0

    def test_zero_recall(self):
        """Test zero recall (no relevant items in top-K)."""
        predictions = [[4, 5, 6, 7, 8]]
        ground_truth = [{1, 2, 3}]
        k = 5

        recall = recall_at_k(predictions, ground_truth, k)
        assert recall == 0.0

    def test_partial_recall(self):
        """Test partial recall."""
        predictions = [[1, 5, 2, 7, 8]]
        ground_truth = [{1, 2, 3}]
        k = 5

        recall = recall_at_k(predictions, ground_truth, k)
        assert recall == pytest.approx(2 / 3, abs=1e-6)

    def test_multiple_users(self):
        """Test recall with multiple users."""
        predictions = [
            [1, 5, 2, 7, 8],
            [4, 5, 6, 7, 8],
        ]
        ground_truth = [
            {1, 2, 3},
            {4, 5},
        ]
        k = 5

        recall = recall_at_k(predictions, ground_truth, k)
        # User 1: 2/3, User 2: 2/2 => avg = 5/6
        assert recall == pytest.approx(5 / 6, abs=1e-6)

    def test_empty_ground_truth(self):
        """Test recall with empty ground truth (skip user)."""
        predictions = [[1, 2, 3]]
        ground_truth = [set()]
        k = 5

        recall = recall_at_k(predictions, ground_truth, k)
        assert recall == 0.0


class TestPrecisionAtK:
    """Test Precision@K metric."""

    def test_perfect_precision(self):
        """Test perfect precision (all predictions relevant)."""
        predictions = [[1, 2, 3, 4, 5]]
        ground_truth = [{1, 2, 3, 4, 5, 6, 7}]
        k = 5

        precision = precision_at_k(predictions, ground_truth, k)
        assert precision == 1.0

    def test_zero_precision(self):
        """Test zero precision (no predictions relevant)."""
        predictions = [[4, 5, 6, 7, 8]]
        ground_truth = [{1, 2, 3}]
        k = 5

        precision = precision_at_k(predictions, ground_truth, k)
        assert precision == 0.0

    def test_partial_precision(self):
        """Test partial precision."""
        predictions = [[1, 5, 2, 7, 8]]
        ground_truth = [{1, 2, 3}]
        k = 5

        precision = precision_at_k(predictions, ground_truth, k)
        assert precision == pytest.approx(2 / 5, abs=1e-6)

    def test_k_smaller_than_predictions(self):
        """Test precision with K smaller than prediction list."""
        predictions = [[1, 5, 2, 7, 8, 9, 10]]
        ground_truth = [{1, 2}]
        k = 3

        precision = precision_at_k(predictions, ground_truth, k)
        # Top 3: [1, 5, 2], hits: 2 => 2/3
        assert precision == pytest.approx(2 / 3, abs=1e-6)


class TestNDCGAtK:
    """Test NDCG@K metric."""

    def test_perfect_ranking(self):
        """Test perfect ranking (highest relevance first)."""
        predictions = [[1, 2, 3]]
        ground_truth = [{1, 2, 3}]
        relevance_scores = [{1: 5.0, 2: 4.0, 3: 3.0}]
        k = 3

        ndcg = ndcg_at_k(predictions, ground_truth, relevance_scores, k)
        assert ndcg == pytest.approx(1.0, abs=1e-6)

    def test_reversed_ranking(self):
        """Test reversed ranking (lowest relevance first)."""
        predictions = [[3, 2, 1]]  # Reversed order
        ground_truth = [{1, 2, 3}]
        relevance_scores = [{1: 5.0, 2: 4.0, 3: 3.0}]
        k = 3

        ndcg = ndcg_at_k(predictions, ground_truth, relevance_scores, k)
        # Should be < 1.0 due to suboptimal ranking
        assert 0.0 < ndcg < 1.0

    def test_empty_ground_truth(self):
        """Test NDCG with empty ground truth."""
        predictions = [[1, 2, 3]]
        ground_truth = [set()]
        relevance_scores = [{}]
        k = 3

        ndcg = ndcg_at_k(predictions, ground_truth, relevance_scores, k)
        assert ndcg == 0.0


class TestHitRateAtK:
    """Test Hit Rate@K metric."""

    def test_all_hits(self):
        """Test when all users have at least one hit."""
        predictions = [
            [1, 5, 2, 7, 8],
            [4, 5, 6, 7, 8],
        ]
        ground_truth = [
            {1, 2, 3},
            {4, 5},
        ]
        k = 5

        hit_rate = hit_rate_at_k(predictions, ground_truth, k)
        assert hit_rate == 1.0

    def test_no_hits(self):
        """Test when no users have hits."""
        predictions = [
            [10, 11, 12],
            [20, 21, 22],
        ]
        ground_truth = [
            {1, 2, 3},
            {4, 5, 6},
        ]
        k = 3

        hit_rate = hit_rate_at_k(predictions, ground_truth, k)
        assert hit_rate == 0.0

    def test_partial_hits(self):
        """Test when some users have hits."""
        predictions = [
            [1, 5, 2],  # Has hits
            [10, 11, 12],  # No hits
            [4, 5, 6],  # Has hits
        ]
        ground_truth = [
            {1, 2, 3},
            {4, 5, 6},
            {4, 5, 6},
        ]
        k = 3

        hit_rate = hit_rate_at_k(predictions, ground_truth, k)
        assert hit_rate == pytest.approx(2 / 3, abs=1e-6)


class TestMAPAtK:
    """Test Mean Average Precision@K metric."""

    def test_perfect_map(self):
        """Test perfect MAP (all relevant items at top)."""
        predictions = [[1, 2, 3, 4, 5]]
        ground_truth = [{1, 2, 3}]
        k = 5

        map_score = map_at_k(predictions, ground_truth, k)
        assert map_score == 1.0

    def test_zero_map(self):
        """Test zero MAP (no relevant items)."""
        predictions = [[4, 5, 6, 7, 8]]
        ground_truth = [{1, 2, 3}]
        k = 5

        map_score = map_at_k(predictions, ground_truth, k)
        assert map_score == 0.0

    def test_map_ordering_matters(self):
        """Test that MAP considers ranking order."""
        # Relevant items at positions 1, 3, 5
        predictions1 = [[1, 5, 2, 7, 3]]
        # Relevant items at positions 1, 2, 3
        predictions2 = [[1, 2, 3, 7, 8]]

        ground_truth = [{1, 2, 3}]
        k = 5

        map1 = map_at_k([predictions1[0]], ground_truth, k)
        map2 = map_at_k([predictions2[0]], ground_truth, k)

        # Better ranking should have higher MAP
        assert map2 > map1


class TestCoverage:
    """Test Coverage metric."""

    def test_full_coverage(self):
        """Test full catalog coverage."""
        predictions = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ]
        n_items = 12

        cov = coverage(predictions, n_items)
        assert cov == 1.0

    def test_zero_coverage(self):
        """Test zero coverage (shouldn't happen in practice)."""
        predictions = [[], [], []]
        n_items = 10

        cov = coverage(predictions, n_items)
        assert cov == 0.0

    def test_partial_coverage(self):
        """Test partial coverage."""
        predictions = [
            [1, 2, 3],
            [1, 2, 3],  # Duplicates
            [4, 5, 6],
        ]
        n_items = 10

        cov = coverage(predictions, n_items)
        # Unique items: {1, 2, 3, 4, 5, 6} = 6 items
        assert cov == pytest.approx(6 / 10, abs=1e-6)


class TestComputeAllMetrics:
    """Test compute_all_metrics function."""

    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        predictions = [
            [1, 5, 2, 7, 8],
            [4, 5, 6, 7, 8],
        ]
        ground_truth = [
            {1, 2, 3},
            {4, 5},
        ]
        relevance_scores = [
            {1: 5.0, 2: 4.5, 3: 4.0},
            {4: 5.0, 5: 4.5},
        ]
        n_items = 10

        metrics = compute_all_metrics(
            predictions, ground_truth, relevance_scores, n_items, k_values=[5]
        )

        # Check all expected metrics are present
        assert "recall@5" in metrics
        assert "precision@5" in metrics
        assert "ndcg@5" in metrics
        assert "hit_rate@5" in metrics
        assert "map@5" in metrics
        assert "coverage" in metrics

        # Check all values are floats
        for value in metrics.values():
            assert isinstance(value, float)

        # Check all values are in valid range [0, 1]
        for value in metrics.values():
            assert 0.0 <= value <= 1.0

    def test_multiple_k_values(self):
        """Test computing metrics at multiple K values."""
        predictions = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        ground_truth = [{1, 2, 3, 4, 5}]
        relevance_scores = [{i: 5.0 for i in range(1, 6)}]
        n_items = 10

        metrics = compute_all_metrics(
            predictions, ground_truth, relevance_scores, n_items, k_values=[3, 5, 10]
        )

        # Check metrics for all K values
        for k in [3, 5, 10]:
            assert f"recall@{k}" in metrics
            assert f"precision@{k}" in metrics
            assert f"ndcg@{k}" in metrics

        # Recall should increase with K
        assert metrics["recall@3"] <= metrics["recall@5"] <= metrics["recall@10"]


class TestEdgeCases:
    """Test edge cases for metrics."""

    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise errors."""
        predictions = [[1, 2, 3]]
        ground_truth = [{1}, {2}]  # Different length
        k = 3

        with pytest.raises(ValueError, match="must have same length"):
            recall_at_k(predictions, ground_truth, k)

    def test_k_larger_than_predictions(self):
        """Test K larger than prediction list."""
        predictions = [[1, 2]]  # Only 2 items
        ground_truth = [{1, 2, 3}]
        k = 10  # K is 10

        recall = recall_at_k(predictions, ground_truth, k)
        precision = precision_at_k(predictions, ground_truth, k)

        # Should still work correctly
        assert recall == pytest.approx(2 / 3, abs=1e-6)
        assert precision == pytest.approx(2 / 10, abs=1e-6)
