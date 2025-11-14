"""Unit tests for QdrantManager."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.vector_store.qdrant_client import QdrantManager


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    with patch("src.vector_store.qdrant_client.QdrantClient") as mock:
        yield mock


@pytest.fixture
def qdrant_manager(mock_qdrant_client):
    """Create QdrantManager with mocked client."""
    manager = QdrantManager(host="localhost", port=6333)
    return manager


class TestQdrantManagerInit:
    """Test QdrantManager initialization."""

    def test_init_with_defaults(self, mock_qdrant_client):
        """Test initialization with default parameters."""
        manager = QdrantManager()

        assert manager.host == "localhost"
        assert manager.port == 6333
        assert manager.timeout == 30

        mock_qdrant_client.assert_called_once()

    def test_init_with_custom_params(self, mock_qdrant_client):
        """Test initialization with custom parameters."""
        manager = QdrantManager(
            host="qdrant.example.com",
            port=6334,
            timeout=60,
            prefer_grpc=True,
        )

        assert manager.host == "qdrant.example.com"
        assert manager.port == 6334
        assert manager.timeout == 60

    @patch.dict("os.environ", {"QDRANT_HOST": "test-host", "QDRANT_PORT": "9999"})
    def test_from_env(self, mock_qdrant_client):
        """Test initialization from environment variables."""
        manager = QdrantManager.from_env()

        assert manager.host == "test-host"
        assert manager.port == 9999


class TestCreateCollections:
    """Test collection creation."""

    def test_create_collections_default(self, qdrant_manager):
        """Test creating collections with default settings."""
        qdrant_manager.client.get_collection = Mock(side_effect=ValueError)

        qdrant_manager.create_collections(embedding_dim=128)

        # Should create both collections
        assert qdrant_manager.client.create_collection.call_count == 2

    def test_create_collections_recreate(self, qdrant_manager):
        """Test recreating existing collections."""
        mock_info = MagicMock(points_count=1000)
        qdrant_manager.client.get_collection = Mock(return_value=mock_info)

        qdrant_manager.create_collections(embedding_dim=128, recreate=True)

        # Should delete and recreate
        assert qdrant_manager.client.delete_collection.call_count == 2
        assert qdrant_manager.client.create_collection.call_count == 2

    def test_create_collections_skip_existing(self, qdrant_manager):
        """Test skipping existing collections."""
        mock_info = MagicMock(points_count=1000)
        qdrant_manager.client.get_collection = Mock(return_value=mock_info)

        qdrant_manager.create_collections(embedding_dim=128, recreate=False)

        # Should not delete or recreate
        qdrant_manager.client.delete_collection.assert_not_called()
        qdrant_manager.client.create_collection.assert_not_called()


class TestIndexEmbeddings:
    """Test embedding indexing."""

    def test_index_user_embeddings(self, qdrant_manager):
        """Test indexing user embeddings."""
        user_embeddings = np.random.randn(100, 128).astype(np.float32)

        qdrant_manager.index_user_embeddings(user_embeddings, batch_size=50)

        # Should upsert in 2 batches (100 users / 50 per batch)
        assert qdrant_manager.client.upsert.call_count == 2

    def test_index_movie_embeddings(self, qdrant_manager):
        """Test indexing movie embeddings with metadata."""
        movie_embeddings = np.random.randn(50, 128).astype(np.float32)
        movie_metadata = pd.DataFrame(
            {
                "movieId": range(1, 51),
                "title": [f"Movie {i}" for i in range(1, 51)],
                "genres": ["Action|Comedy"] * 50,
                "year": [2020] * 50,
                "avg_rating": [4.0] * 50,
                "popularity": [100] * 50,
            }
        )

        qdrant_manager.index_movie_embeddings(
            movie_embeddings, movie_metadata, batch_size=25
        )

        # Should upsert in 2 batches (50 movies / 25 per batch)
        assert qdrant_manager.client.upsert.call_count == 2

    def test_index_movie_embeddings_minimal_metadata(self, qdrant_manager):
        """Test indexing with minimal metadata."""
        movie_embeddings = np.random.randn(10, 128).astype(np.float32)
        movie_metadata = pd.DataFrame(
            {
                "movieId": range(1, 11),
                "title": [f"Movie {i}" for i in range(1, 11)],
                "genres": ["Action"] * 10,
            }
        )

        qdrant_manager.index_movie_embeddings(
            movie_embeddings, movie_metadata, batch_size=10
        )

        qdrant_manager.client.upsert.assert_called_once()


class TestSearch:
    """Test search functionality."""

    def test_search_similar_movies_no_filters(self, qdrant_manager):
        """Test searching without filters."""
        query_vector = np.random.randn(128).astype(np.float32)

        # Mock search results
        mock_result = MagicMock()
        mock_result.payload = {
            "movie_id": 123,
            "title": "Test Movie",
            "genres": ["Action"],
        }
        mock_result.score = 0.95

        qdrant_manager.client.search = Mock(return_value=[mock_result])

        results = qdrant_manager.search_similar_movies(query_vector, k=10)

        assert len(results) == 1
        assert results[0][0] == 123  # movie_id
        assert results[0][1] == 0.95  # score

        # Verify search was called
        qdrant_manager.client.search.assert_called_once()

    def test_search_similar_movies_with_filters(self, qdrant_manager):
        """Test searching with filters."""
        query_vector = np.random.randn(128).astype(np.float32)

        qdrant_manager.client.search = Mock(return_value=[])

        results = qdrant_manager.search_similar_movies(
            query_vector,
            k=10,
            genre_filter=["Action", "Comedy"],
            min_year=2000,
            max_year=2020,
            min_rating=4.0,
        )

        # Verify filter was passed
        call_args = qdrant_manager.client.search.call_args
        assert call_args.kwargs["query_filter"] is not None


class TestGetEmbeddings:
    """Test embedding retrieval."""

    def test_get_user_embedding_success(self, qdrant_manager):
        """Test successful user embedding retrieval."""
        mock_point = MagicMock()
        mock_point.vector = [0.1] * 128

        qdrant_manager.client.retrieve = Mock(return_value=[mock_point])

        embedding = qdrant_manager.get_user_embedding(user_id=42)

        assert embedding is not None
        assert embedding.shape == (128,)

    def test_get_user_embedding_not_found(self, qdrant_manager):
        """Test user not found."""
        qdrant_manager.client.retrieve = Mock(return_value=[])

        embedding = qdrant_manager.get_user_embedding(user_id=9999)

        assert embedding is None

    def test_get_movie_embedding_success(self, qdrant_manager):
        """Test successful movie embedding retrieval."""
        mock_point = MagicMock()
        mock_point.vector = [0.2] * 128

        qdrant_manager.client.retrieve = Mock(return_value=[mock_point])

        embedding = qdrant_manager.get_movie_embedding(movie_idx=10)

        assert embedding is not None
        assert embedding.shape == (128,)


class TestRecommendations:
    """Test recommendation generation."""

    def test_get_recommendations_success(self, qdrant_manager):
        """Test getting recommendations for a user."""
        # Mock user embedding retrieval
        mock_point = MagicMock()
        mock_point.vector = [0.1] * 128
        qdrant_manager.client.retrieve = Mock(return_value=[mock_point])

        # Mock search results
        mock_result = MagicMock()
        mock_result.payload = {"movie_id": 456, "title": "Recommended Movie"}
        mock_result.score = 0.88
        qdrant_manager.client.search = Mock(return_value=[mock_result])

        results = qdrant_manager.get_recommendations(user_id=1, k=5)

        assert len(results) == 1
        assert results[0][0] == 456

    def test_get_recommendations_user_not_found(self, qdrant_manager):
        """Test recommendations when user not found."""
        qdrant_manager.client.retrieve = Mock(return_value=[])

        results = qdrant_manager.get_recommendations(user_id=9999, k=5)

        assert len(results) == 0


class TestUtilities:
    """Test utility methods."""

    def test_get_collection_info(self, qdrant_manager):
        """Test getting collection information."""
        mock_info = MagicMock()
        mock_info.points_count = 10000
        mock_info.indexed_vectors_count = 10000
        mock_info.segments_count = 1
        mock_info.status = "green"

        qdrant_manager.client.get_collection = Mock(return_value=mock_info)

        info = qdrant_manager.get_collection_info("test_collection")

        assert info["points_count"] == 10000
        assert info["status"] == "green"

    def test_health_check_success(self, qdrant_manager):
        """Test successful health check."""
        qdrant_manager.client.get_collections = Mock(return_value=[])

        assert qdrant_manager.health_check() is True

    def test_health_check_failure(self, qdrant_manager):
        """Test failed health check."""
        qdrant_manager.client.get_collections = Mock(side_effect=Exception("Connection failed"))

        assert qdrant_manager.health_check() is False

    def test_close(self, qdrant_manager):
        """Test closing connection."""
        qdrant_manager.close()

        qdrant_manager.client.close.assert_called_once()
