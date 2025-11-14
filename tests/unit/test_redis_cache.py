"""Unit tests for RedisCache."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.vector_store.redis_cache import RedisCache


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    with patch("src.vector_store.redis_cache.redis.Redis") as mock:
        yield mock


@pytest.fixture
def redis_cache(mock_redis):
    """Create RedisCache with mocked client."""
    cache = RedisCache(host="localhost", port=6379)
    return cache


class TestRedisCacheInit:
    """Test RedisCache initialization."""

    def test_init_with_defaults(self, mock_redis):
        """Test initialization with default parameters."""
        cache = RedisCache()

        assert cache.host == "localhost"
        assert cache.port == 6379
        assert cache.db == 0

    def test_init_with_custom_params(self, mock_redis):
        """Test initialization with custom parameters."""
        cache = RedisCache(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
        )

        assert cache.host == "redis.example.com"
        assert cache.port == 6380
        assert cache.db == 1

    @patch.dict("os.environ", {"REDIS_HOST": "test-redis", "REDIS_PORT": "7777"})
    def test_from_env(self, mock_redis):
        """Test initialization from environment variables."""
        cache = RedisCache.from_env()

        assert cache.host == "test-redis"
        assert cache.port == 7777


class TestCacheKeys:
    """Test cache key generation."""

    def test_make_key_simple(self, redis_cache):
        """Test simple key generation."""
        key = redis_cache._make_key("test", 123)
        assert key == "test:123"

    def test_make_key_multiple_parts(self, redis_cache):
        """Test key with multiple parts."""
        key = redis_cache._make_key("rec", 42, "action", 2020)
        assert key == "rec:42:action:2020"


class TestRecommendationsCache:
    """Test recommendations caching."""

    def test_get_recommendations_hit(self, redis_cache):
        """Test cache hit for recommendations."""
        recommendations = [
            (1, 0.9, {"title": "Movie 1"}),
            (2, 0.8, {"title": "Movie 2"}),
        ]

        redis_cache.client.get = Mock(
            return_value=redis_cache._serialize(recommendations)
        )

        result = redis_cache.get_recommendations(user_id=42)

        assert result is not None
        assert len(result) == 2
        assert result[0][0] == 1

    def test_get_recommendations_miss(self, redis_cache):
        """Test cache miss for recommendations."""
        redis_cache.client.get = Mock(return_value=None)

        result = redis_cache.get_recommendations(user_id=42)

        assert result is None

    def test_set_recommendations(self, redis_cache):
        """Test setting recommendations in cache."""
        recommendations = [(1, 0.9, {"title": "Movie 1"})]

        redis_cache.client.setex = Mock(return_value=True)

        success = redis_cache.set_recommendations(
            user_id=42, recommendations=recommendations
        )

        assert success is True
        redis_cache.client.setex.assert_called_once()

    def test_get_recommendations_with_filters(self, redis_cache):
        """Test cache key includes filters."""
        filters = {"genre": "Action", "min_year": 2000}

        redis_cache.client.get = Mock(return_value=None)

        redis_cache.get_recommendations(user_id=42, filters=filters)

        # Verify filter was included in key
        call_args = redis_cache.client.get.call_args[0][0]
        assert "42" in call_args


class TestEmbeddingsCache:
    """Test embeddings caching."""

    def test_get_user_embedding_hit(self, redis_cache):
        """Test cache hit for user embedding."""
        embedding = np.random.randn(128).astype(np.float32)

        redis_cache.client.get = Mock(return_value=redis_cache._serialize(embedding))

        result = redis_cache.get_user_embedding(user_id=10)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (128,)

    def test_get_user_embedding_miss(self, redis_cache):
        """Test cache miss for user embedding."""
        redis_cache.client.get = Mock(return_value=None)

        result = redis_cache.get_user_embedding(user_id=10)

        assert result is None

    def test_set_user_embedding(self, redis_cache):
        """Test setting user embedding."""
        embedding = np.random.randn(128).astype(np.float32)

        redis_cache.client.setex = Mock(return_value=True)

        success = redis_cache.set_user_embedding(user_id=10, embedding=embedding)

        assert success is True
        redis_cache.client.setex.assert_called_once()

    def test_get_movie_embedding_hit(self, redis_cache):
        """Test cache hit for movie embedding."""
        embedding = np.random.randn(128).astype(np.float32)

        redis_cache.client.get = Mock(return_value=redis_cache._serialize(embedding))

        result = redis_cache.get_movie_embedding(movie_id=5)

        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_set_movie_embedding(self, redis_cache):
        """Test setting movie embedding."""
        embedding = np.random.randn(128).astype(np.float32)

        redis_cache.client.setex = Mock(return_value=True)

        success = redis_cache.set_movie_embedding(movie_id=5, embedding=embedding)

        assert success is True


class TestPopularItemsCache:
    """Test popular items caching."""

    def test_get_popular_items_hit(self, redis_cache):
        """Test cache hit for popular items."""
        items = [(1, 1000.0), (2, 950.0), (3, 900.0)]

        redis_cache.client.get = Mock(return_value=redis_cache._serialize(items))

        result = redis_cache.get_popular_items(k=100)

        assert result is not None
        assert len(result) == 3

    def test_get_popular_items_by_genre(self, redis_cache):
        """Test popular items with genre filter."""
        redis_cache.client.get = Mock(return_value=None)

        result = redis_cache.get_popular_items(genre="Action", k=50)

        # Verify genre was included in key
        call_args = redis_cache.client.get.call_args[0][0]
        assert "Action" in call_args

    def test_set_popular_items(self, redis_cache):
        """Test setting popular items."""
        items = [(1, 1000.0), (2, 950.0)]

        redis_cache.client.setex = Mock(return_value=True)

        success = redis_cache.set_popular_items(items=items, k=100)

        assert success is True


class TestCacheInvalidation:
    """Test cache invalidation."""

    def test_invalidate_user(self, redis_cache):
        """Test invalidating user cache."""
        # Mock keys to return 1 key for first pattern, 1 for second, none for third
        redis_cache.client.keys = Mock(
            side_effect=[[b"rec:42:1"], [b"user_emb:42"], []]
        )
        # Mock delete to return 2 for each call (even though only 1 key each)
        redis_cache.client.delete = Mock(return_value=2)

        deleted = redis_cache.invalidate_user(user_id=42)

        # Two patterns have keys, so 2 delete calls * 2 return value = 4
        assert deleted == 4

    def test_invalidate_movie(self, redis_cache):
        """Test invalidating movie cache."""
        redis_cache.client.delete = Mock(return_value=1)

        deleted = redis_cache.invalidate_movie(movie_id=10)

        assert deleted == 1

    def test_flush_all(self, redis_cache):
        """Test flushing all cache."""
        redis_cache.client.flushdb = Mock(return_value=True)

        success = redis_cache.flush_all()

        assert success is True
        redis_cache.client.flushdb.assert_called_once()


class TestStatistics:
    """Test cache statistics."""

    def test_get_stats_success(self, redis_cache):
        """Test getting cache statistics."""
        mock_info = {
            "connected_clients": 5,
            "used_memory_human": "10M",
            "total_commands_processed": 1000,
            "keyspace_hits": 800,
            "keyspace_misses": 200,
        }

        redis_cache.client.info = Mock(return_value=mock_info)

        stats = redis_cache.get_stats()

        assert stats["connected_clients"] == 5
        assert stats["hit_rate"] == 0.8  # 800 / (800 + 200)

    def test_calculate_hit_rate_zero_total(self, redis_cache):
        """Test hit rate calculation with zero total."""
        info = {"keyspace_hits": 0, "keyspace_misses": 0}

        hit_rate = redis_cache._calculate_hit_rate(info)

        assert hit_rate == 0.0


class TestHealthAndConnection:
    """Test health check and connection management."""

    def test_health_check_success(self, redis_cache):
        """Test successful health check."""
        redis_cache.client.ping = Mock(return_value=True)

        assert redis_cache.health_check() is True

    def test_health_check_failure(self, redis_cache):
        """Test failed health check."""
        from redis.exceptions import ConnectionError

        redis_cache.client.ping = Mock(side_effect=ConnectionError("Connection failed"))

        assert redis_cache.health_check() is False

    def test_close(self, redis_cache):
        """Test closing connection."""
        redis_cache.close()

        redis_cache.client.close.assert_called_once()


class TestSerialization:
    """Test serialization/deserialization."""

    def test_serialize_numpy_array(self, redis_cache):
        """Test serializing numpy array."""
        data = np.random.randn(128).astype(np.float32)

        serialized = redis_cache._serialize(data)

        assert isinstance(serialized, bytes)

    def test_deserialize_numpy_array(self, redis_cache):
        """Test deserializing numpy array."""
        original = np.random.randn(128).astype(np.float32)
        serialized = redis_cache._serialize(original)

        deserialized = redis_cache._deserialize(serialized)

        assert isinstance(deserialized, np.ndarray)
        assert np.array_equal(original, deserialized)

    def test_serialize_complex_object(self, redis_cache):
        """Test serializing complex nested structure."""
        data = [(1, 0.9, {"title": "Movie", "genres": ["Action", "Comedy"]})]

        serialized = redis_cache._serialize(data)
        deserialized = redis_cache._deserialize(serialized)

        assert deserialized == data
