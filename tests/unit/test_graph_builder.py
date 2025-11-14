"""
Unit tests for Graph Builder.

Tests user-item bipartite graph construction for LightGCN.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from src.data.graph_builder import UserItemGraph, build_graph


@pytest.fixture
def mock_ratings_df():
    """Create mock ratings DataFrame for testing."""
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3, 3],
            "movieId": [10, 20, 10, 30, 20, 30, 40],
            "rating": [5.0, 4.5, 4.0, 5.0, 3.5, 4.5, 4.0],
        }
    )


@pytest.fixture
def mock_ratings_df_large():
    """Create larger mock ratings DataFrame."""
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
            "movieId": [10, 20, 30, 10, 40, 20, 30, 40, 10, 50, 50],
            "rating": [5.0, 4.5, 4.0, 5.0, 3.5, 4.5, 4.0, 5.0, 4.0, 4.5, 3.0],
        }
    )


@pytest.fixture
def mock_ratings_df_low_ratings():
    """Create mock ratings DataFrame with all low ratings."""
    return pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [10, 20, 30],
            "rating": [2.0, 2.5, 3.0],
        }
    )


@pytest.fixture
def mock_ratings_df_single_user():
    """Create mock ratings DataFrame with single user."""
    return pd.DataFrame(
        {
            "userId": [1, 1, 1],
            "movieId": [10, 20, 30],
            "rating": [5.0, 4.5, 4.0],
        }
    )


@pytest.fixture
def mock_ratings_df_single_item():
    """Create mock ratings DataFrame with single item."""
    return pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [10, 10, 10],
            "rating": [5.0, 4.5, 4.0],
        }
    )


@pytest.fixture
def mock_ratings_df_empty():
    """Create empty mock ratings DataFrame."""
    return pd.DataFrame({"userId": [], "movieId": [], "rating": []})


class TestUserItemGraph:
    """Test UserItemGraph class."""

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_initialization(self, mock_exists, mock_read_parquet, mock_ratings_df):
        """Test UserItemGraph initialization."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        assert graph.n_users == 5
        assert graph.n_items == 6
        assert graph.min_rating == 4.0
        assert hasattr(graph, "ratings_df")
        assert hasattr(graph, "adjacency")
        assert hasattr(graph, "norm_adj")

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_initialization_file_not_found(self, mock_exists, mock_read_parquet):
        """Test UserItemGraph initialization with missing file."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="Ratings file not found"):
            UserItemGraph(
                n_users=5,
                n_items=6,
                train_ratings_path=Path("missing.parquet"),
                min_rating=4.0,
            )

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_load_data_basic(self, mock_exists, mock_read_parquet, mock_ratings_df):
        """Test data loading and filtering."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # Should filter out rating 3.5
        assert len(graph.ratings_df) == 6  # 7 total - 1 with rating 3.5

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_load_data_user_item_mappings(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test user and item ID to index mappings."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # Check user mappings
        assert len(graph.user_to_idx) == 3  # Users 1, 2, 3
        assert graph.user_to_idx[1] == 0
        assert graph.user_to_idx[2] == 1
        assert graph.user_to_idx[3] == 2

        # Check item mappings
        assert len(graph.item_to_idx) == 4  # Items 10, 20, 30, 40
        assert graph.item_to_idx[10] == 0
        assert graph.item_to_idx[20] == 1
        assert graph.item_to_idx[30] == 2
        assert graph.item_to_idx[40] == 3

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_load_data_unique_ids(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test unique user and item ID extraction."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # Check unique IDs are sorted
        assert graph.user_ids == [1, 2, 3]
        assert graph.item_ids == [10, 20, 30, 40]

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_build_adjacency_matrix_shape(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test adjacency matrix has correct shape."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # Adjacency should be (n_users + n_items) x (n_users + n_items)
        expected_shape = (5 + 6, 5 + 6)
        assert graph.adjacency.shape == expected_shape

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_build_adjacency_matrix_structure(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test adjacency matrix has correct bipartite structure."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        adj = graph.adjacency.toarray()

        # Top-left (user-user) should be zero
        assert np.allclose(adj[:5, :5], 0)

        # Bottom-right (item-item) should be zero
        assert np.allclose(adj[5:, 5:], 0)

        # Top-right and bottom-left should be transposes
        assert np.allclose(adj[:5, 5:], adj[5:, :5].T)

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_build_adjacency_matrix_values(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test adjacency matrix has correct edge values."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        adj = graph.adjacency.toarray()

        # All edges should be 1.0 (binary interactions)
        non_zero = adj[adj > 0]
        assert np.allclose(non_zero, 1.0)

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_normalize_adjacency_shape(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test normalized adjacency has correct shape."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        assert graph.norm_adj.shape == graph.adjacency.shape

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_normalize_adjacency_symmetry(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test normalized adjacency is symmetric."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        norm_adj_dense = graph.norm_adj.toarray()
        assert np.allclose(norm_adj_dense, norm_adj_dense.T)

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_normalize_adjacency_has_self_loops(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test normalized adjacency has self-loops added."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        norm_adj_dense = graph.norm_adj.toarray()
        # Diagonal should be non-zero (self-loops)
        assert np.all(np.diag(norm_adj_dense) > 0)

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_get_sparse_graph_types(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test get_sparse_graph returns correct types."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        indices, values, size = graph.get_sparse_graph()

        assert isinstance(indices, torch.Tensor)
        assert isinstance(values, torch.Tensor)
        assert isinstance(size, torch.Size)

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_get_sparse_graph_shapes(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test get_sparse_graph returns correct shapes."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        indices, values, size = graph.get_sparse_graph()

        # Indices should be [2, num_edges]
        assert indices.shape[0] == 2
        assert indices.dtype == torch.long

        # Values should be [num_edges]
        assert values.ndim == 1
        assert values.dtype == torch.float32

        # Size should match adjacency shape
        assert size == torch.Size([11, 11])  # 5 users + 6 items

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_get_sparse_graph_num_edges(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test get_sparse_graph has correct number of edges."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        indices, values, size = graph.get_sparse_graph()

        # Number of edges should match norm_adj.nnz
        assert indices.shape[1] == graph.norm_adj.nnz
        assert values.shape[0] == graph.norm_adj.nnz

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_get_dense_graph_shape(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test get_dense_graph returns correct shape."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        dense_graph = graph.get_dense_graph()

        assert dense_graph.shape == torch.Size([11, 11])
        assert dense_graph.dtype == torch.float32

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_get_dense_graph_values(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test get_dense_graph matches norm_adj."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        dense_graph = graph.get_dense_graph()
        expected = torch.FloatTensor(graph.norm_adj.toarray())

        assert torch.allclose(dense_graph, expected)

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_get_user_interactions(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test get_user_interactions returns correct items."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # User 0 (ID=1) should have items 10, 20 (indices 0, 1)
        user_items = graph.get_user_interactions(0)
        assert len(user_items) == 2
        assert 0 in user_items  # Item 10
        assert 1 in user_items  # Item 20

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_get_user_interactions_multiple_items(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test get_user_interactions for user with multiple items."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # User 2 (ID=3) should have items 30, 40 (indices 2, 3)
        # Rating 3.5 for item 20 is filtered out
        user_items = graph.get_user_interactions(2)
        assert len(user_items) == 2
        assert set(user_items) == {2, 3}

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_get_item_interactions(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test get_item_interactions returns correct users."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # Item 0 (ID=10) should have users 1, 2 (indices 0, 1)
        item_users = graph.get_item_interactions(0)
        assert len(item_users) == 2
        assert set(item_users) == {0, 1}

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_get_item_interactions_multiple_users(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test get_item_interactions for item with multiple users."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # Item 2 (ID=30) should have users 2, 3 (indices 1, 2)
        item_users = graph.get_item_interactions(2)
        assert len(item_users) == 2
        assert set(item_users) == {1, 2}

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_get_statistics_basic(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test get_statistics returns correct structure."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        stats = graph.get_statistics()

        # Check all required keys are present
        required_keys = {
            "n_users",
            "n_items",
            "n_interactions",
            "avg_user_degree",
            "avg_item_degree",
            "max_user_degree",
            "max_item_degree",
            "sparsity",
        }
        assert set(stats.keys()) == required_keys

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_get_statistics_values(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test get_statistics returns correct values."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        stats = graph.get_statistics()

        assert stats["n_users"] == 5
        assert stats["n_items"] == 6
        assert stats["n_interactions"] == 6  # After filtering
        assert stats["avg_user_degree"] > 0
        assert stats["avg_item_degree"] > 0
        assert stats["max_user_degree"] >= stats["avg_user_degree"]
        assert stats["max_item_degree"] >= stats["avg_item_degree"]
        assert 0 <= stats["sparsity"] <= 1

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_different_min_rating_threshold(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test graph with different rating threshold."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.5,
        )

        # Should only keep ratings >= 4.5
        assert len(graph.ratings_df) == 4
        assert all(graph.ratings_df["rating"] >= 4.5)

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_min_rating_3_point_0(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test graph with min_rating=3.0."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=3.0,
        )

        # Should keep all ratings
        assert len(graph.ratings_df) == 7
        assert all(graph.ratings_df["rating"] >= 3.0)

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_empty_ratings_after_filter(
        self, mock_exists, mock_read_parquet, mock_ratings_df_low_ratings
    ):
        """Test graph with no ratings above threshold."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df_low_ratings

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # All ratings filtered out
        assert len(graph.ratings_df) == 0
        assert len(graph.user_ids) == 0
        assert len(graph.item_ids) == 0

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_single_user_graph(
        self, mock_exists, mock_read_parquet, mock_ratings_df_single_user
    ):
        """Test graph with single user."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df_single_user

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        assert len(graph.user_ids) == 1
        assert len(graph.item_ids) == 3
        stats = graph.get_statistics()
        assert stats["n_users"] == 5
        assert stats["n_items"] == 6

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_single_item_graph(
        self, mock_exists, mock_read_parquet, mock_ratings_df_single_item
    ):
        """Test graph with single item."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df_single_item

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        assert len(graph.user_ids) == 3
        assert len(graph.item_ids) == 1
        stats = graph.get_statistics()
        assert stats["n_users"] == 5
        assert stats["n_items"] == 6

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_large_graph(self, mock_exists, mock_read_parquet, mock_ratings_df_large):
        """Test graph with more interactions."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df_large

        graph = UserItemGraph(
            n_users=10,
            n_items=10,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # Should filter out rating 3.5 and 3.0
        assert len(graph.ratings_df) == 9
        stats = graph.get_statistics()
        assert stats["n_interactions"] == 9


class TestBuildGraph:
    """Test build_graph factory function."""

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_build_graph_basic(self, mock_exists, mock_read_parquet, mock_ratings_df):
        """Test build_graph factory function."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = build_graph(
            n_users=5,
            n_items=6,
            train_ratings_path="dummy.parquet",
            min_rating=4.0,
        )

        assert isinstance(graph, UserItemGraph)
        assert graph.n_users == 5
        assert graph.n_items == 6
        assert graph.min_rating == 4.0

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_build_graph_returns_statistics(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test build_graph logs statistics."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = build_graph(
            n_users=5,
            n_items=6,
            train_ratings_path="dummy.parquet",
            min_rating=4.0,
        )

        # Should be able to get statistics
        stats = graph.get_statistics()
        assert stats is not None
        assert "n_users" in stats
        assert "n_items" in stats

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_build_graph_default_min_rating(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test build_graph with default min_rating."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = build_graph(n_users=5, n_items=6, train_ratings_path="dummy.parquet")

        # Default min_rating should be 4.0
        assert graph.min_rating == 4.0

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_build_graph_custom_min_rating(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test build_graph with custom min_rating."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = build_graph(
            n_users=5,
            n_items=6,
            train_ratings_path="dummy.parquet",
            min_rating=3.5,
        )

        assert graph.min_rating == 3.5


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_graph_with_empty_dataframe(
        self, mock_exists, mock_read_parquet, mock_ratings_df_empty
    ):
        """Test graph construction with empty DataFrame."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df_empty

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # Should handle empty data gracefully
        assert len(graph.ratings_df) == 0
        assert len(graph.user_ids) == 0
        assert len(graph.item_ids) == 0
        assert graph.adjacency.shape == (11, 11)

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_adjacency_matrix_is_sparse(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test adjacency matrix is stored as sparse."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # Adjacency should be scipy sparse matrix
        assert sp.issparse(graph.adjacency)
        assert sp.issparse(graph.norm_adj)

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_norm_adj_is_coo_format(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test normalized adjacency is in COO format."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # norm_adj should be COO format
        assert isinstance(graph.norm_adj, sp.coo_matrix)

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_sparse_graph_can_build_sparse_tensor(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test sparse graph output can build PyTorch sparse tensor."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        indices, values, size = graph.get_sparse_graph()

        # Should be able to create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size)
        assert sparse_tensor.is_sparse
        assert sparse_tensor.shape == size

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_user_item_id_sorting(
        self, mock_exists, mock_read_parquet, mock_ratings_df
    ):
        """Test user and item IDs are sorted."""
        mock_exists.return_value = True
        # Create unsorted DataFrame
        unsorted_df = mock_ratings_df.copy()
        unsorted_df = unsorted_df.sample(frac=1, random_state=42)  # Shuffle
        mock_read_parquet.return_value = unsorted_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # IDs should still be sorted
        assert graph.user_ids == sorted(graph.user_ids)
        assert graph.item_ids == sorted(graph.item_ids)

    @patch("src.data.graph_builder.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_degree_computation(self, mock_exists, mock_read_parquet, mock_ratings_df):
        """Test user and item degree computation."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        stats = graph.get_statistics()

        # Degrees should be positive
        assert stats["avg_user_degree"] > 0
        assert stats["avg_item_degree"] > 0
        assert stats["max_user_degree"] > 0
        assert stats["max_item_degree"] > 0

        # Max should be >= average
        assert stats["max_user_degree"] >= stats["avg_user_degree"]
        assert stats["max_item_degree"] >= stats["avg_item_degree"]


@pytest.mark.parametrize("min_rating", [3.0, 3.5, 4.0, 4.5, 5.0])
@patch("src.data.graph_builder.pd.read_parquet")
@patch("pathlib.Path.exists")
class TestParametrizedRatingThresholds:
    """Test graph with different rating thresholds."""

    def test_different_thresholds(
        self, mock_exists, mock_read_parquet, min_rating, mock_ratings_df
    ):
        """Test graph construction with parametrized rating thresholds."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=5,
            n_items=6,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=min_rating,
        )

        # All kept ratings should be >= threshold
        assert all(graph.ratings_df["rating"] >= min_rating)
        assert graph.min_rating == min_rating


@pytest.mark.parametrize("n_users,n_items", [(10, 20), (50, 100), (100, 50)])
@patch("src.data.graph_builder.pd.read_parquet")
@patch("pathlib.Path.exists")
class TestParametrizedGraphSizes:
    """Test graph with different sizes."""

    def test_different_sizes(
        self, mock_exists, mock_read_parquet, n_users, n_items, mock_ratings_df
    ):
        """Test graph construction with parametrized sizes."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = mock_ratings_df

        graph = UserItemGraph(
            n_users=n_users,
            n_items=n_items,
            train_ratings_path=Path("dummy.parquet"),
            min_rating=4.0,
        )

        # Adjacency shape should be (n_users + n_items, n_users + n_items)
        expected_shape = (n_users + n_items, n_users + n_items)
        assert graph.adjacency.shape == expected_shape
        assert graph.norm_adj.shape == expected_shape
