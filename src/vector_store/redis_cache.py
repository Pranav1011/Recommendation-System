"""
Redis Cache Manager

Production-grade caching layer for recommendation system.
Supports:
- Recommendation caching with TTL
- User/movie embedding caching
- Popular items caching
- Cache invalidation strategies
- Automatic serialization/deserialization
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis
from redis.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger(__name__)


class RedisCache:
    """Manage Redis cache for recommendation system."""

    # Cache key prefixes
    PREFIX_RECOMMENDATIONS = "rec"
    PREFIX_USER_EMBEDDING = "user_emb"
    PREFIX_MOVIE_EMBEDDING = "movie_emb"
    PREFIX_POPULAR_ITEMS = "popular"
    PREFIX_USER_HISTORY = "history"

    # Default TTLs (in seconds)
    TTL_RECOMMENDATIONS = 3600  # 1 hour
    TTL_USER_EMBEDDING = 86400  # 24 hours
    TTL_MOVIE_EMBEDDING = 86400  # 24 hours
    TTL_POPULAR_ITEMS = 604800  # 1 week
    TTL_USER_HISTORY = 3600  # 1 hour

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
    ):
        """
        Initialize Redis cache client.

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number (0-15)
            password: Redis password (optional)
            max_connections: Maximum connection pool size
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connect timeout in seconds
        """
        self.host = host
        self.port = port
        self.db = db

        # Create connection pool for better performance
        pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=False,  # We'll handle encoding ourselves
        )

        self.client = redis.Redis(connection_pool=pool)

        logger.info(f"Connected to Redis at {host}:{port} (db={db})")

    @classmethod
    def from_env(cls) -> "RedisCache":
        """Create RedisCache from environment variables."""
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        db = int(os.getenv("REDIS_DB", "0"))
        password = os.getenv("REDIS_PASSWORD")

        return cls(host=host, port=port, db=db, password=password)

    def _make_key(self, prefix: str, *parts: Any) -> str:
        """
        Create cache key from prefix and parts.

        Args:
            prefix: Key prefix
            *parts: Key parts to join

        Returns:
            Cache key string
        """
        parts_str = ":".join(str(p) for p in parts)
        return f"{prefix}:{parts_str}"

    def _serialize(self, data: Any) -> bytes:
        """
        Serialize data for storage.

        Args:
            data: Data to serialize

        Returns:
            Serialized bytes
        """
        # Use pickle for Python objects (numpy arrays, complex types)
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize data from storage.

        Args:
            data: Serialized bytes

        Returns:
            Deserialized data
        """
        return pickle.loads(data)

    def get_recommendations(
        self, user_id: int, filters: Optional[Dict] = None
    ) -> Optional[List[Tuple[int, float, Dict]]]:
        """
        Get cached recommendations for a user.

        Args:
            user_id: User ID
            filters: Optional filters applied (for cache key)

        Returns:
            List of (movie_id, score, metadata) or None if not cached
        """
        # Create cache key with filters
        filter_key = json.dumps(filters, sort_keys=True) if filters else "none"
        key = self._make_key(self.PREFIX_RECOMMENDATIONS, user_id, filter_key)

        try:
            data = self.client.get(key)
            if data:
                return self._deserialize(data)
            return None
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis get error: {e}")
            return None

    def set_recommendations(
        self,
        user_id: int,
        recommendations: List[Tuple[int, float, Dict]],
        filters: Optional[Dict] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache recommendations for a user.

        Args:
            user_id: User ID
            recommendations: List of (movie_id, score, metadata)
            filters: Optional filters applied
            ttl: Time to live in seconds (default: TTL_RECOMMENDATIONS)

        Returns:
            True if cached successfully
        """
        filter_key = json.dumps(filters, sort_keys=True) if filters else "none"
        key = self._make_key(self.PREFIX_RECOMMENDATIONS, user_id, filter_key)
        ttl = ttl or self.TTL_RECOMMENDATIONS

        try:
            data = self._serialize(recommendations)
            self.client.setex(key, ttl, data)
            return True
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis set error: {e}")
            return False

    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """
        Get cached user embedding.

        Args:
            user_id: User ID

        Returns:
            User embedding or None if not cached
        """
        key = self._make_key(self.PREFIX_USER_EMBEDDING, user_id)

        try:
            data = self.client.get(key)
            if data:
                return self._deserialize(data)
            return None
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis get error: {e}")
            return None

    def set_user_embedding(
        self, user_id: int, embedding: np.ndarray, ttl: Optional[int] = None
    ) -> bool:
        """
        Cache user embedding.

        Args:
            user_id: User ID
            embedding: User embedding vector
            ttl: Time to live in seconds

        Returns:
            True if cached successfully
        """
        key = self._make_key(self.PREFIX_USER_EMBEDDING, user_id)
        ttl = ttl or self.TTL_USER_EMBEDDING

        try:
            data = self._serialize(embedding)
            self.client.setex(key, ttl, data)
            return True
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis set error: {e}")
            return False

    def get_movie_embedding(self, movie_id: int) -> Optional[np.ndarray]:
        """
        Get cached movie embedding.

        Args:
            movie_id: Movie ID

        Returns:
            Movie embedding or None if not cached
        """
        key = self._make_key(self.PREFIX_MOVIE_EMBEDDING, movie_id)

        try:
            data = self.client.get(key)
            if data:
                return self._deserialize(data)
            return None
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis get error: {e}")
            return None

    def set_movie_embedding(
        self, movie_id: int, embedding: np.ndarray, ttl: Optional[int] = None
    ) -> bool:
        """
        Cache movie embedding.

        Args:
            movie_id: Movie ID
            embedding: Movie embedding vector
            ttl: Time to live in seconds

        Returns:
            True if cached successfully
        """
        key = self._make_key(self.PREFIX_MOVIE_EMBEDDING, movie_id)
        ttl = ttl or self.TTL_MOVIE_EMBEDDING

        try:
            data = self._serialize(embedding)
            self.client.setex(key, ttl, data)
            return True
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis set error: {e}")
            return False

    def get_popular_items(
        self, genre: Optional[str] = None, k: int = 100
    ) -> Optional[List[Tuple[int, float]]]:
        """
        Get cached popular items.

        Args:
            genre: Optional genre filter
            k: Number of items

        Returns:
            List of (movie_id, popularity_score) or None
        """
        genre_key = genre or "all"
        key = self._make_key(self.PREFIX_POPULAR_ITEMS, genre_key, k)

        try:
            data = self.client.get(key)
            if data:
                return self._deserialize(data)
            return None
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis get error: {e}")
            return None

    def set_popular_items(
        self,
        items: List[Tuple[int, float]],
        genre: Optional[str] = None,
        k: int = 100,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache popular items.

        Args:
            items: List of (movie_id, popularity_score)
            genre: Optional genre filter
            k: Number of items
            ttl: Time to live in seconds

        Returns:
            True if cached successfully
        """
        genre_key = genre or "all"
        key = self._make_key(self.PREFIX_POPULAR_ITEMS, genre_key, k)
        ttl = ttl or self.TTL_POPULAR_ITEMS

        try:
            data = self._serialize(items)
            self.client.setex(key, ttl, data)
            return True
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis set error: {e}")
            return False

    def invalidate_user(self, user_id: int) -> int:
        """
        Invalidate all cache entries for a user.

        Args:
            user_id: User ID

        Returns:
            Number of keys deleted
        """
        try:
            # Find all keys for this user
            patterns = [
                self._make_key(self.PREFIX_RECOMMENDATIONS, user_id, "*"),
                self._make_key(self.PREFIX_USER_EMBEDDING, user_id),
                self._make_key(self.PREFIX_USER_HISTORY, user_id, "*"),
            ]

            deleted = 0
            for pattern in patterns:
                keys = self.client.keys(pattern)
                if keys:
                    deleted += self.client.delete(*keys)

            logger.info(f"Invalidated {deleted} cache entries for user {user_id}")
            return deleted

        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis invalidation error: {e}")
            return 0

    def invalidate_movie(self, movie_id: int) -> int:
        """
        Invalidate all cache entries for a movie.

        Args:
            movie_id: Movie ID

        Returns:
            Number of keys deleted
        """
        try:
            key = self._make_key(self.PREFIX_MOVIE_EMBEDDING, movie_id)
            deleted = self.client.delete(key)

            logger.info(f"Invalidated {deleted} cache entries for movie {movie_id}")
            return deleted

        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis invalidation error: {e}")
            return 0

    def flush_all(self) -> bool:
        """
        Flush all cache entries (use with caution).

        Returns:
            True if successful
        """
        try:
            self.client.flushdb()
            logger.warning("Flushed all cache entries")
            return True
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis flush error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        try:
            info = self.client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
            }
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis stats error: {e}")
            return {}

    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0

    def health_check(self) -> bool:
        """
        Check if Redis is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            return self.client.ping()
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def close(self) -> None:
        """Close Redis connection."""
        try:
            self.client.close()
            logger.info("Closed Redis connection")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
