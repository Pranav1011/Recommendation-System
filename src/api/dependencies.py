"""
FastAPI Dependency Injection

Provides dependency injection for:
- Qdrant vector database client
- Redis cache client
- Movie metadata loader
- Proper connection lifecycle management
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import HTTPException

from src.vector_store.qdrant_client import QdrantManager
from src.vector_store.redis_cache import RedisCache

logger = logging.getLogger(__name__)

# Global instances (singletons)
_qdrant_manager: Optional[QdrantManager] = None
_redis_cache: Optional[RedisCache] = None
_movie_metadata: Optional[pd.DataFrame] = None


# ==================== Qdrant Dependency ====================


async def get_qdrant_manager() -> QdrantManager:
    """
    Get Qdrant manager instance (singleton).

    Raises:
        HTTPException: If Qdrant is not available
    """
    global _qdrant_manager

    if _qdrant_manager is None:
        try:
            logger.info("Initializing Qdrant manager...")
            _qdrant_manager = QdrantManager.from_env()

            # Health check
            if not _qdrant_manager.health_check():
                raise ConnectionError("Qdrant health check failed")

            logger.info("Qdrant manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Vector database unavailable",
                    "detail": str(e),
                    "error_code": "QDRANT_UNAVAILABLE",
                },
            )

    return _qdrant_manager


async def close_qdrant_manager():
    """Close Qdrant manager connection."""
    global _qdrant_manager
    if _qdrant_manager is not None:
        _qdrant_manager.close()
        _qdrant_manager = None
        logger.info("Closed Qdrant manager")


# ==================== Redis Dependency ====================


async def get_redis_cache() -> RedisCache:
    """
    Get Redis cache instance (singleton).

    Raises:
        HTTPException: If Redis is not available
    """
    global _redis_cache

    if _redis_cache is None:
        try:
            logger.info("Initializing Redis cache...")
            _redis_cache = RedisCache.from_env()

            # Health check
            if not _redis_cache.health_check():
                raise ConnectionError("Redis health check failed")

            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Cache unavailable",
                    "detail": str(e),
                    "error_code": "REDIS_UNAVAILABLE",
                },
            )

    return _redis_cache


async def close_redis_cache():
    """Close Redis cache connection."""
    global _redis_cache
    if _redis_cache is not None:
        _redis_cache.close()
        _redis_cache = None
        logger.info("Closed Redis cache")


# ==================== Movie Metadata Dependency ====================


@lru_cache(maxsize=1)
def _load_movie_metadata() -> pd.DataFrame:
    """
    Load movie metadata from parquet file (cached).

    Returns:
        DataFrame with movie metadata
    """
    metadata_path = Path("data/processed/movies.parquet")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Movie metadata not found at {metadata_path}")

    logger.info(f"Loading movie metadata from {metadata_path}")
    df = pd.read_parquet(metadata_path)

    # Parse year from title if not already present
    if "year" not in df.columns:
        df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype("Int64")

    logger.info(f"Loaded {len(df):,} movies")
    return df


async def get_movie_metadata() -> pd.DataFrame:
    """
    Get movie metadata DataFrame.

    Raises:
        HTTPException: If metadata cannot be loaded
    """
    global _movie_metadata

    if _movie_metadata is None:
        try:
            _movie_metadata = _load_movie_metadata()
        except Exception as e:
            logger.error(f"Failed to load movie metadata: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Movie metadata unavailable",
                    "detail": str(e),
                    "error_code": "METADATA_UNAVAILABLE",
                },
            )

    return _movie_metadata


# ==================== Movie Lookup Helper ====================


async def get_movie_by_id(movie_id: int) -> Optional[dict]:
    """
    Get movie metadata by ID.

    Args:
        movie_id: Movie ID to lookup

    Returns:
        Movie metadata dict or None if not found
    """
    metadata = await get_movie_metadata()
    movie = metadata[metadata["movieId"] == movie_id]

    if movie.empty:
        return None

    row = movie.iloc[0]
    return {
        "movie_id": int(row["movieId"]),
        "title": str(row["title"]),
        "genres": row["genres"].split("|") if isinstance(row["genres"], str) else [],
        "year": int(row["year"]) if pd.notna(row.get("year")) else None,
    }


# ==================== Lifecycle Management ====================


async def startup_event():
    """
    Initialize all dependencies on application startup.

    This is called by FastAPI's lifespan event.
    """
    logger.info("Starting up recommendation API...")

    # Initialize connections
    try:
        await get_qdrant_manager()
        logger.info("✓ Qdrant connected")
    except Exception as e:
        logger.warning(f"✗ Qdrant connection failed: {e}")

    try:
        await get_redis_cache()
        logger.info("✓ Redis connected")
    except Exception as e:
        logger.warning(f"✗ Redis connection failed: {e}")

    try:
        await get_movie_metadata()
        logger.info("✓ Movie metadata loaded")
    except Exception as e:
        logger.warning(f"✗ Movie metadata loading failed: {e}")

    logger.info("Startup complete")


async def shutdown_event():
    """
    Cleanup all dependencies on application shutdown.

    This is called by FastAPI's lifespan event.
    """
    logger.info("Shutting down recommendation API...")

    await close_qdrant_manager()
    await close_redis_cache()

    logger.info("Shutdown complete")


# ==================== Optional: Ensemble Dependency ====================


async def get_ensemble_recommender():
    """
    Get ensemble recommender (optional, for advanced use cases).

    This can be implemented if you want to use the ensemble model
    directly in the API instead of just Qdrant search.
    """
    # TODO: Implement if needed
    # from src.models.ensemble import EnsembleRecommender
    # return EnsembleRecommender(...)
    raise NotImplementedError("Ensemble recommender not yet implemented in API")
