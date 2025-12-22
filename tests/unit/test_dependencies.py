"""Unit tests for the FastAPI dependency injection."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi import HTTPException

from src.api.dependencies import (
    close_qdrant_manager,
    close_redis_cache,
    get_ensemble_recommender,
    get_movie_by_id,
    get_movie_metadata,
    get_qdrant_manager,
    get_redis_cache,
    shutdown_event,
    startup_event,
)

# ==================== Fixtures ====================


@pytest.fixture
def mock_movie_df():
    """Mock movie metadata DataFrame."""
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["Movie 1 (2020)", "Movie 2 (2019)", "Movie 3 (2021)"],
            "genres": ["Action|Adventure", "Drama", "Comedy"],
            "year": [2020, 2019, 2021],
        }
    )


@pytest.fixture
def mock_movie_df_no_year():
    """Mock movie metadata DataFrame without year column."""
    return pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["Movie 1 (2020)", "Movie 2 (2019)"],
            "genres": ["Action", "Drama"],
        }
    )


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global singleton instances before each test."""
    import src.api.dependencies as deps

    deps._qdrant_manager = None
    deps._redis_cache = None
    deps._movie_metadata = None

    # Clear LRU cache
    deps._load_movie_metadata.cache_clear()

    yield

    # Reset again after test
    deps._qdrant_manager = None
    deps._redis_cache = None
    deps._movie_metadata = None
    deps._load_movie_metadata.cache_clear()


# ==================== Qdrant Dependency Tests ====================


class TestGetQdrantManager:
    """Tests for get_qdrant_manager dependency."""

    @pytest.mark.asyncio
    async def test_get_qdrant_first_time(self):
        """Test getting Qdrant manager for the first time."""
        mock_qdrant = MagicMock()
        mock_qdrant.health_check.return_value = True

        with patch("src.api.dependencies.QdrantManager") as mock_qdrant_class:
            mock_qdrant_class.from_env.return_value = mock_qdrant

            manager = await get_qdrant_manager()

            assert manager is mock_qdrant
            mock_qdrant_class.from_env.assert_called_once()
            mock_qdrant.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_qdrant_singleton(self):
        """Test that Qdrant manager is a singleton."""
        mock_qdrant = MagicMock()
        mock_qdrant.health_check.return_value = True

        with patch("src.api.dependencies.QdrantManager") as mock_qdrant_class:
            mock_qdrant_class.from_env.return_value = mock_qdrant

            manager1 = await get_qdrant_manager()
            manager2 = await get_qdrant_manager()

            assert manager1 is manager2
            # Should only be called once (singleton)
            mock_qdrant_class.from_env.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_qdrant_health_check_fails(self):
        """Test when Qdrant health check fails."""
        mock_qdrant = MagicMock()
        mock_qdrant.health_check.return_value = False

        with patch("src.api.dependencies.QdrantManager") as mock_qdrant_class:
            mock_qdrant_class.from_env.return_value = mock_qdrant

            with pytest.raises(HTTPException) as exc_info:
                await get_qdrant_manager()

            assert exc_info.value.status_code == 503
            assert "QDRANT_UNAVAILABLE" in exc_info.value.detail["error_code"]

    @pytest.mark.asyncio
    async def test_get_qdrant_connection_error(self):
        """Test when Qdrant connection fails."""
        with patch("src.api.dependencies.QdrantManager") as mock_qdrant_class:
            mock_qdrant_class.from_env.side_effect = ConnectionError(
                "Connection refused"
            )

            with pytest.raises(HTTPException) as exc_info:
                await get_qdrant_manager()

            assert exc_info.value.status_code == 503
            assert "Vector database unavailable" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_close_qdrant_manager(self):
        """Test closing Qdrant manager."""
        mock_qdrant = MagicMock()
        mock_qdrant.health_check.return_value = True

        with patch("src.api.dependencies.QdrantManager") as mock_qdrant_class:
            mock_qdrant_class.from_env.return_value = mock_qdrant

            # Initialize manager
            await get_qdrant_manager()

            # Close it
            await close_qdrant_manager()

            mock_qdrant.close.assert_called_once()

            # Verify it's reset (next call creates new instance)
            await get_qdrant_manager()
            assert mock_qdrant_class.from_env.call_count == 2

    @pytest.mark.asyncio
    async def test_close_qdrant_when_none(self):
        """Test closing Qdrant manager when it's None."""
        # Should not raise error
        await close_qdrant_manager()


# ==================== Redis Dependency Tests ====================


class TestGetRedisCache:
    """Tests for get_redis_cache dependency."""

    @pytest.mark.asyncio
    async def test_get_redis_first_time(self):
        """Test getting Redis cache for the first time."""
        mock_redis = MagicMock()
        mock_redis.health_check.return_value = True

        with patch("src.api.dependencies.RedisCache") as mock_redis_class:
            mock_redis_class.from_env.return_value = mock_redis

            cache = await get_redis_cache()

            assert cache is mock_redis
            mock_redis_class.from_env.assert_called_once()
            mock_redis.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_redis_singleton(self):
        """Test that Redis cache is a singleton."""
        mock_redis = MagicMock()
        mock_redis.health_check.return_value = True

        with patch("src.api.dependencies.RedisCache") as mock_redis_class:
            mock_redis_class.from_env.return_value = mock_redis

            cache1 = await get_redis_cache()
            cache2 = await get_redis_cache()

            assert cache1 is cache2
            mock_redis_class.from_env.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_redis_health_check_fails(self):
        """Test when Redis health check fails."""
        mock_redis = MagicMock()
        mock_redis.health_check.return_value = False

        with patch("src.api.dependencies.RedisCache") as mock_redis_class:
            mock_redis_class.from_env.return_value = mock_redis

            with pytest.raises(HTTPException) as exc_info:
                await get_redis_cache()

            assert exc_info.value.status_code == 503
            assert "REDIS_UNAVAILABLE" in exc_info.value.detail["error_code"]

    @pytest.mark.asyncio
    async def test_get_redis_connection_error(self):
        """Test when Redis connection fails."""
        with patch("src.api.dependencies.RedisCache") as mock_redis_class:
            mock_redis_class.from_env.side_effect = Exception("Redis connection error")

            with pytest.raises(HTTPException) as exc_info:
                await get_redis_cache()

            assert exc_info.value.status_code == 503
            assert "Cache unavailable" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_close_redis_cache(self):
        """Test closing Redis cache."""
        mock_redis = MagicMock()
        mock_redis.health_check.return_value = True

        with patch("src.api.dependencies.RedisCache") as mock_redis_class:
            mock_redis_class.from_env.return_value = mock_redis

            # Initialize cache
            await get_redis_cache()

            # Close it
            await close_redis_cache()

            mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_redis_when_none(self):
        """Test closing Redis cache when it's None."""
        # Should not raise error
        await close_redis_cache()


# ==================== Movie Metadata Tests ====================


class TestGetMovieMetadata:
    """Tests for get_movie_metadata dependency."""

    @pytest.mark.asyncio
    async def test_get_metadata_success(self, mock_movie_df, tmp_path):
        """Test getting movie metadata successfully."""
        # Create temporary parquet file
        metadata_path = tmp_path / "data" / "processed"
        metadata_path.mkdir(parents=True)
        parquet_file = metadata_path / "movies.parquet"
        mock_movie_df.to_parquet(parquet_file)

        with patch("src.api.dependencies.Path") as mock_path:
            mock_path.return_value = parquet_file

            df = await get_movie_metadata()

            assert len(df) == 3
            assert "year" in df.columns
            assert list(df["movieId"]) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_get_metadata_adds_year_column(self, mock_movie_df_no_year, tmp_path):
        """Test that year column is extracted from title if missing."""
        # Create temporary parquet file without year column
        metadata_path = tmp_path / "data" / "processed"
        metadata_path.mkdir(parents=True)
        parquet_file = metadata_path / "movies.parquet"
        mock_movie_df_no_year.to_parquet(parquet_file)

        with patch("src.api.dependencies.Path") as mock_path:
            mock_path.return_value = parquet_file

            df = await get_movie_metadata()

            assert "year" in df.columns
            assert df.iloc[0]["year"] == 2020
            assert df.iloc[1]["year"] == 2019

    @pytest.mark.asyncio
    async def test_get_metadata_file_not_found(self):
        """Test when metadata file doesn't exist."""
        with patch("src.api.dependencies.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = False
            mock_path.return_value = mock_path_instance

            with pytest.raises(HTTPException) as exc_info:
                await get_movie_metadata()

            assert exc_info.value.status_code == 503
            assert "METADATA_UNAVAILABLE" in exc_info.value.detail["error_code"]

    @pytest.mark.asyncio
    async def test_get_metadata_singleton(self, mock_movie_df, tmp_path):
        """Test that metadata is cached (singleton pattern)."""
        metadata_path = tmp_path / "data" / "processed"
        metadata_path.mkdir(parents=True)
        parquet_file = metadata_path / "movies.parquet"
        mock_movie_df.to_parquet(parquet_file)

        with (
            patch("src.api.dependencies.Path") as mock_path,
            patch("src.api.dependencies.pd.read_parquet") as mock_read,
        ):
            mock_path.return_value = parquet_file
            mock_read.return_value = mock_movie_df

            df1 = await get_movie_metadata()
            df2 = await get_movie_metadata()

            # Should only read once (cached)
            assert mock_read.call_count == 1
            assert df1 is df2


class TestGetMovieById:
    """Tests for get_movie_by_id helper function."""

    @pytest.mark.asyncio
    async def test_get_movie_by_id_found(self, mock_movie_df):
        """Test getting movie by ID when it exists."""
        with patch(
            "src.api.dependencies.get_movie_metadata", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_movie_df

            movie = await get_movie_by_id(1)

            assert movie is not None
            assert movie["movie_id"] == 1
            assert movie["title"] == "Movie 1 (2020)"
            assert movie["genres"] == ["Action", "Adventure"]
            assert movie["year"] == 2020

    @pytest.mark.asyncio
    async def test_get_movie_by_id_not_found(self, mock_movie_df):
        """Test getting movie by ID when it doesn't exist."""
        with patch(
            "src.api.dependencies.get_movie_metadata", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_movie_df

            movie = await get_movie_by_id(999999)

            assert movie is None

    @pytest.mark.asyncio
    async def test_get_movie_by_id_genres_parsing(self, mock_movie_df):
        """Test that genres are properly parsed from pipe-delimited string."""
        with patch(
            "src.api.dependencies.get_movie_metadata", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_movie_df

            movie = await get_movie_by_id(1)

            assert movie["genres"] == ["Action", "Adventure"]

    @pytest.mark.asyncio
    async def test_get_movie_by_id_no_year(self):
        """Test getting movie with missing year."""
        df = pd.DataFrame(
            {
                "movieId": [1],
                "title": ["Movie 1"],
                "genres": ["Action"],
                "year": [pd.NA],
            }
        )

        with patch(
            "src.api.dependencies.get_movie_metadata", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = df

            movie = await get_movie_by_id(1)

            assert movie["year"] is None


# ==================== Lifecycle Management Tests ====================


class TestStartupEvent:
    """Tests for startup_event."""

    @pytest.mark.asyncio
    async def test_startup_all_services_success(self):
        """Test startup when all services connect successfully."""
        mock_qdrant = MagicMock()
        mock_qdrant.health_check.return_value = True
        mock_redis = MagicMock()
        mock_redis.health_check.return_value = True
        mock_df = pd.DataFrame(
            {"movieId": [1], "title": ["Movie 1"], "genres": ["Action"]}
        )

        with (
            patch(
                "src.api.dependencies.get_qdrant_manager", new_callable=AsyncMock
            ) as mock_get_qdrant,
            patch(
                "src.api.dependencies.get_redis_cache", new_callable=AsyncMock
            ) as mock_get_redis,
            patch(
                "src.api.dependencies.get_movie_metadata", new_callable=AsyncMock
            ) as mock_get_metadata,
        ):

            mock_get_qdrant.return_value = mock_qdrant
            mock_get_redis.return_value = mock_redis
            mock_get_metadata.return_value = mock_df

            # Should not raise
            await startup_event()

            mock_get_qdrant.assert_called_once()
            mock_get_redis.assert_called_once()
            mock_get_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_qdrant_fails(self):
        """Test startup when Qdrant connection fails."""
        mock_redis = MagicMock()
        mock_redis.health_check.return_value = True
        mock_df = pd.DataFrame(
            {"movieId": [1], "title": ["Movie 1"], "genres": ["Action"]}
        )

        with (
            patch(
                "src.api.dependencies.get_qdrant_manager", new_callable=AsyncMock
            ) as mock_get_qdrant,
            patch(
                "src.api.dependencies.get_redis_cache", new_callable=AsyncMock
            ) as mock_get_redis,
            patch(
                "src.api.dependencies.get_movie_metadata", new_callable=AsyncMock
            ) as mock_get_metadata,
        ):

            mock_get_qdrant.side_effect = HTTPException(
                status_code=503, detail="Qdrant unavailable"
            )
            mock_get_redis.return_value = mock_redis
            mock_get_metadata.return_value = mock_df

            # Should not raise, just log warning
            await startup_event()

            # Other services should still be initialized
            mock_get_redis.assert_called_once()
            mock_get_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_all_services_fail(self):
        """Test startup when all services fail."""
        with (
            patch(
                "src.api.dependencies.get_qdrant_manager", new_callable=AsyncMock
            ) as mock_get_qdrant,
            patch(
                "src.api.dependencies.get_redis_cache", new_callable=AsyncMock
            ) as mock_get_redis,
            patch(
                "src.api.dependencies.get_movie_metadata", new_callable=AsyncMock
            ) as mock_get_metadata,
        ):

            mock_get_qdrant.side_effect = Exception("Qdrant error")
            mock_get_redis.side_effect = Exception("Redis error")
            mock_get_metadata.side_effect = Exception("Metadata error")

            # Should not raise, just log warnings
            await startup_event()


class TestShutdownEvent:
    """Tests for shutdown_event."""

    @pytest.mark.asyncio
    async def test_shutdown_event(self):
        """Test shutdown event closes all connections."""
        with (
            patch(
                "src.api.dependencies.close_qdrant_manager", new_callable=AsyncMock
            ) as mock_close_qdrant,
            patch(
                "src.api.dependencies.close_redis_cache", new_callable=AsyncMock
            ) as mock_close_redis,
        ):

            await shutdown_event()

            mock_close_qdrant.assert_called_once()
            mock_close_redis.assert_called_once()


# ==================== Ensemble Recommender Tests ====================


class TestGetEnsembleRecommender:
    """Tests for get_ensemble_recommender."""

    @pytest.mark.asyncio
    async def test_ensemble_not_implemented(self):
        """Test that ensemble recommender raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            await get_ensemble_recommender()

        assert "not yet implemented" in str(exc_info.value)
