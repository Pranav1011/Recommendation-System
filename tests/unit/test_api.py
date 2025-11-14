"""Unit tests for the FastAPI application."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

# ==================== Fixtures ====================


@pytest.fixture
def mock_qdrant():
    """Mock QdrantManager for testing."""
    mock = MagicMock()
    mock.health_check.return_value = True
    mock.get_collection_info.return_value = {
        "points_count": 1000,
        "status": "green",
    }
    mock.USER_COLLECTION = "users"
    mock.MOVIE_COLLECTION = "movies"
    mock.get_recommendations.return_value = [
        (
            1,
            0.95,
            {
                "title": "Movie 1",
                "genres": ["Action"],
                "avg_rating": 4.5,
                "popularity": 100,
            },
        ),
        (
            2,
            0.90,
            {
                "title": "Movie 2",
                "genres": ["Drama"],
                "avg_rating": 4.2,
                "popularity": 80,
            },
        ),
    ]
    mock.get_movie_embedding.return_value = np.random.rand(128).astype(np.float32)
    mock.search_similar_movies.return_value = [
        (1, 0.95, {"title": "Movie 1", "genres": ["Action"], "avg_rating": 4.5}),
        (2, 0.90, {"title": "Movie 2", "genres": ["Drama"], "avg_rating": 4.2}),
    ]
    return mock


@pytest.fixture
def mock_redis():
    """Mock RedisCache for testing."""
    mock = MagicMock()
    mock.health_check.return_value = True
    mock.get_stats.return_value = {
        "hit_rate": 0.85,
        "used_memory_human": "256MB",
        "total_keys": 1000,
    }
    mock.get_recommendations.return_value = None
    mock.get_popular_items.return_value = None
    return mock


@pytest.fixture
def mock_movie_metadata():
    """Mock movie metadata DataFrame."""
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 100],
            "title": [
                "Movie 1 (2020)",
                "Movie 2 (2019)",
                "Movie 3 (2021)",
                "Test Movie (2000)",
            ],
            "genres": ["Action|Adventure", "Drama", "Comedy|Romance", "Action"],
            "year": [2020, 2019, 2021, 2000],
            "avg_rating": [4.5, 4.2, 3.8, 4.0],
            "popularity": [1000, 800, 500, 200],
        }
    )


@pytest.fixture
def client(mock_qdrant, mock_redis, mock_movie_metadata):
    """Create a test client with mocked dependencies."""
    from src.api.dependencies import (
        get_movie_metadata,
        get_qdrant_manager,
        get_redis_cache,
    )
    from src.api.main import app

    # Override dependencies
    async def override_qdrant():
        return mock_qdrant

    async def override_redis():
        return mock_redis

    async def override_metadata():
        return mock_movie_metadata

    app.dependency_overrides[get_qdrant_manager] = override_qdrant
    app.dependency_overrides[get_redis_cache] = override_redis
    app.dependency_overrides[get_movie_metadata] = override_metadata

    client = TestClient(app)
    yield client

    # Clean up overrides
    app.dependency_overrides.clear()


# ==================== Root & Health Endpoints ====================


class TestRootEndpoints:
    """Tests for root and health check endpoints."""

    def test_root_endpoint(self, client):
        """Test the root endpoint returns correct response."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "running"
        assert data["version"] == "1.0.0"
        assert "/docs" in data["docs_url"]
        assert "/health" in data["health_url"]

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "recommendation-api"
        assert "timestamp" in data


class TestDeepHealthCheck:
    """Tests for deep health check endpoint."""

    def test_deep_health_all_healthy(self, client, mock_qdrant, mock_redis):
        """Test deep health check when all services are healthy."""
        response = client.get("/api/v1/health/deep")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert len(data["services"]) == 3
        assert "timestamp" in data

        # Check Qdrant service
        qdrant_service = next(s for s in data["services"] if s["service"] == "qdrant")
        assert qdrant_service["healthy"] is True
        assert "latency_ms" in qdrant_service
        assert qdrant_service["details"]["points_count"] == 1000

    def test_deep_health_qdrant_unhealthy(self, client, mock_qdrant):
        """Test deep health check when Qdrant is unhealthy."""
        mock_qdrant.health_check.return_value = False

        response = client.get("/api/v1/health/deep")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"

        qdrant_service = next(s for s in data["services"] if s["service"] == "qdrant")
        assert qdrant_service["healthy"] is False

    def test_deep_health_redis_unhealthy(self, client, mock_redis):
        """Test deep health check when Redis is unhealthy."""
        mock_redis.health_check.return_value = False

        response = client.get("/api/v1/health/deep")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"

        redis_service = next(s for s in data["services"] if s["service"] == "redis")
        assert redis_service["healthy"] is False


# ==================== Recommendation Endpoints ====================


class TestRecommendationsPOST:
    """Tests for POST /api/v1/recommend endpoint."""

    def test_recommend_success_no_cache(self, client, mock_qdrant, mock_redis):
        """Test successful recommendations without cache."""
        mock_redis.get_recommendations.return_value = None

        response = client.post("/api/v1/recommend", json={"user_id": 123, "k": 10})

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 123
        assert len(data["recommendations"]) == 2
        assert data["cached"] is False
        assert data["count"] == 2
        assert data["recommendations"][0]["movie_id"] == 1
        assert data["recommendations"][0]["title"] == "Movie 1"

        # Verify Qdrant was called
        mock_qdrant.get_recommendations.assert_called_once()
        mock_redis.set_recommendations.assert_called_once()

    def test_recommend_success_with_cache(self, client, mock_redis):
        """Test successful recommendations from cache."""
        cached_results = [
            (
                1,
                0.95,
                {
                    "title": "Cached Movie 1",
                    "genres": ["Action"],
                    "avg_rating": 4.5,
                    "popularity": 100,
                },
            ),
            (
                2,
                0.90,
                {
                    "title": "Cached Movie 2",
                    "genres": ["Drama"],
                    "avg_rating": 4.2,
                    "popularity": 80,
                },
            ),
        ]
        mock_redis.get_recommendations.return_value = cached_results

        response = client.post("/api/v1/recommend", json={"user_id": 123, "k": 10})

        assert response.status_code == 200
        data = response.json()
        assert data["cached"] is True
        assert data["recommendations"][0]["title"] == "Cached Movie 1"

    def test_recommend_with_filters(self, client, mock_qdrant, mock_redis):
        """Test recommendations with genre and rating filters."""
        mock_redis.get_recommendations.return_value = None

        response = client.post(
            "/api/v1/recommend",
            json={
                "user_id": 123,
                "k": 5,
                "min_rating": 4.0,
                "genres": ["Action", "Sci-Fi"],
                "min_year": 2000,
                "max_year": 2023,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filters_applied"] is not None
        assert "min_rating" in data["filters_applied"]
        assert "genres" in data["filters_applied"]

        # Verify filters passed to Qdrant
        mock_qdrant.get_recommendations.assert_called_once()
        call_kwargs = mock_qdrant.get_recommendations.call_args.kwargs
        assert call_kwargs["min_rating"] == 4.0
        assert set(call_kwargs["genre_filter"]) == {"Action", "Sci-Fi"}

    def test_recommend_user_not_found(self, client, mock_qdrant, mock_redis):
        """Test recommendation for non-existent user."""
        mock_redis.get_recommendations.return_value = None
        mock_qdrant.get_recommendations.return_value = []

        response = client.post("/api/v1/recommend", json={"user_id": 999999, "k": 10})

        assert response.status_code == 404
        data = response.json()
        assert "error" in data["detail"]
        assert "USER_NOT_FOUND" in data["detail"]["error_code"]

    def test_recommend_qdrant_error(self, client, mock_qdrant, mock_redis):
        """Test recommendation when Qdrant fails."""
        mock_redis.get_recommendations.return_value = None
        mock_qdrant.get_recommendations.side_effect = Exception(
            "Qdrant connection error"
        )

        response = client.post("/api/v1/recommend", json={"user_id": 123, "k": 10})

        assert response.status_code == 500
        data = response.json()
        assert "QDRANT_QUERY_FAILED" in data["detail"]["error_code"]

    def test_recommend_invalid_user_id(self, client):
        """Test recommendation with invalid user ID."""
        response = client.post("/api/v1/recommend", json={"user_id": -1, "k": 10})

        assert response.status_code == 422  # Validation error

    def test_recommend_invalid_k(self, client):
        """Test recommendation with invalid k value."""
        response = client.post("/api/v1/recommend", json={"user_id": 123, "k": 0})

        assert response.status_code == 422

    def test_recommend_k_too_large(self, client):
        """Test recommendation with k > 100."""
        response = client.post("/api/v1/recommend", json={"user_id": 123, "k": 150})

        assert response.status_code == 422


class TestRecommendationsGET:
    """Tests for GET /api/v1/recommend/{user_id} endpoint."""

    def test_recommend_get_basic(self, client, mock_qdrant, mock_redis):
        """Test GET recommendations with basic parameters."""
        mock_redis.get_recommendations.return_value = None

        response = client.get("/api/v1/recommend/123")

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 123
        assert len(data["recommendations"]) == 2

    def test_recommend_get_with_query_params(self, client, mock_qdrant, mock_redis):
        """Test GET recommendations with query parameters."""
        mock_redis.get_recommendations.return_value = None

        response = client.get(
            "/api/v1/recommend/123?k=5&min_rating=4.0&genres=Action,Sci-Fi&min_year=2000"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2

        # Verify genre parsing
        call_kwargs = mock_qdrant.get_recommendations.call_args.kwargs
        assert set(call_kwargs["genre_filter"]) == {"Action", "Sci-Fi"}

    def test_recommend_get_empty_genres(self, client, mock_qdrant, mock_redis):
        """Test GET recommendations with empty genres string."""
        mock_redis.get_recommendations.return_value = None

        response = client.get("/api/v1/recommend/123?genres=")

        assert response.status_code == 200


class TestSimilarMovies:
    """Tests for GET /api/v1/similar/{movie_id} endpoint."""

    def test_similar_movies_success(self, client, mock_qdrant, mock_movie_metadata):
        """Test getting similar movies successfully."""
        with patch(
            "src.api.main.get_movie_by_id", new_callable=AsyncMock
        ) as mock_get_movie:
            mock_get_movie.return_value = {"movie_id": 1, "title": "Movie 1"}

            response = client.get("/api/v1/similar/1?k=5")

            assert response.status_code == 200
            data = response.json()
            assert data["movie_id"] == 1
            assert data["query_title"] == "Movie 1"
            assert len(data["similar_movies"]) == 1  # Filters out query movie
            assert data["count"] == 1

    def test_similar_movies_not_found(self, client):
        """Test similar movies for non-existent movie."""
        with patch(
            "src.api.main.get_movie_by_id", new_callable=AsyncMock
        ) as mock_get_movie:
            mock_get_movie.return_value = None

            response = client.get("/api/v1/similar/999999")

            assert response.status_code == 404
            data = response.json()
            assert "MOVIE_NOT_FOUND" in data["detail"]["error_code"]

    def test_similar_movies_no_embedding(
        self, client, mock_qdrant, mock_movie_metadata
    ):
        """Test similar movies when embedding not found."""
        with patch(
            "src.api.main.get_movie_by_id", new_callable=AsyncMock
        ) as mock_get_movie:
            mock_get_movie.return_value = {"movie_id": 1, "title": "Movie 1"}
            mock_qdrant.get_movie_embedding.return_value = None

            response = client.get("/api/v1/similar/1")

            assert response.status_code == 404
            data = response.json()
            assert "NO_EMBEDDING" in data["detail"]["error_code"]

    def test_similar_movies_search_error(
        self, client, mock_qdrant, mock_movie_metadata
    ):
        """Test similar movies when search fails."""
        with patch(
            "src.api.main.get_movie_by_id", new_callable=AsyncMock
        ) as mock_get_movie:
            mock_get_movie.return_value = {"movie_id": 1, "title": "Movie 1"}
            mock_qdrant.search_similar_movies.side_effect = Exception("Search failed")

            response = client.get("/api/v1/similar/1")

            assert response.status_code == 500


# ==================== Cold Start Endpoint ====================


class TestColdStart:
    """Tests for POST /api/v1/cold-start/rate endpoint."""

    def test_cold_start_success(self, client, mock_qdrant, mock_movie_metadata):
        """Test cold start recommendations successfully."""
        # Mock search to return different movies than rated ones
        mock_qdrant.search_similar_movies.return_value = [
            (1, 0.95, {"title": "Movie 1", "genres": ["Action"], "avg_rating": 4.5}),
            (2, 0.90, {"title": "Movie 2", "genres": ["Drama"], "avg_rating": 4.2}),
            (
                100,
                0.85,
                {"title": "Test Movie", "genres": ["Action"], "avg_rating": 4.0},
            ),
        ]

        response = client.post(
            "/api/v1/cold-start/rate",
            json={
                "ratings": [
                    {"movie_id": 1, "rating": 5.0},
                    {"movie_id": 2, "rating": 4.5},
                    {"movie_id": 3, "rating": 4.0},
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "temp_user_id" in data
        assert data["temp_user_id"].startswith("temp_")
        assert len(data["recommendations"]) > 0
        assert data["ratings_processed"] == 3

        # Verify rated movies are filtered out
        rec_ids = [r["movie_id"] for r in data["recommendations"]]
        assert 1 not in rec_ids
        assert 2 not in rec_ids
        assert 3 not in rec_ids

    def test_cold_start_no_valid_movies(self, client, mock_movie_metadata):
        """Test cold start with no valid movie IDs."""
        response = client.post(
            "/api/v1/cold-start/rate",
            json={"ratings": [{"movie_id": 999999, "rating": 5.0}]},
        )

        assert response.status_code == 400
        data = response.json()
        assert "INVALID_RATINGS" in data["detail"]["error_code"]

    def test_cold_start_invalid_rating_value(self, client):
        """Test cold start with invalid rating value."""
        response = client.post(
            "/api/v1/cold-start/rate",
            json={"ratings": [{"movie_id": 1, "rating": 6.0}]},  # Too high
        )

        assert response.status_code == 422

    def test_cold_start_missing_fields(self, client):
        """Test cold start with missing rating fields."""
        response = client.post(
            "/api/v1/cold-start/rate",
            json={"ratings": [{"movie_id": 1}]},  # Missing rating
        )

        assert response.status_code == 422

    def test_cold_start_empty_ratings(self, client):
        """Test cold start with empty ratings list."""
        response = client.post("/api/v1/cold-start/rate", json={"ratings": []})

        assert response.status_code == 422

    def test_cold_start_search_error(self, client, mock_qdrant, mock_movie_metadata):
        """Test cold start when search fails."""
        mock_qdrant.search_similar_movies.side_effect = Exception("Search failed")

        response = client.post(
            "/api/v1/cold-start/rate",
            json={"ratings": [{"movie_id": 1, "rating": 5.0}]},
        )

        assert response.status_code == 500


# ==================== Popular Movies Endpoint ====================


class TestPopularMovies:
    """Tests for GET /api/v1/movies/popular endpoint."""

    def test_popular_movies_from_cache(self, client, mock_redis, mock_movie_metadata):
        """Test getting popular movies from cache."""
        mock_redis.get_popular_items.return_value = [(1, 1000.0), (2, 800.0)]

        response = client.get("/api/v1/movies/popular?limit=20")

        assert response.status_code == 200
        data = response.json()
        assert data["cached"] is True
        assert len(data["popular_movies"]) == 2
        assert data["count"] == 2

    def test_popular_movies_no_cache(self, client, mock_redis, mock_movie_metadata):
        """Test getting popular movies without cache."""
        mock_redis.get_popular_items.return_value = None

        response = client.get("/api/v1/movies/popular?limit=3")

        assert response.status_code == 200
        data = response.json()
        assert data["cached"] is False
        assert len(data["popular_movies"]) <= 3  # May be less if fewer movies available

        # Verify sorted by popularity if multiple results
        if len(data["popular_movies"]) > 1:
            assert (
                data["popular_movies"][0]["popularity"]
                >= data["popular_movies"][1]["popularity"]
            )

    def test_popular_movies_with_genre_filter(
        self, client, mock_redis, mock_movie_metadata
    ):
        """Test getting popular movies filtered by genre."""
        mock_redis.get_popular_items.return_value = None

        response = client.get("/api/v1/movies/popular?limit=10&genre=Action")

        assert response.status_code == 200
        data = response.json()
        assert data["genre_filter"] == "Action"

        # Verify all results contain Action genre (if any results)
        if len(data["popular_movies"]) > 0:
            for movie in data["popular_movies"]:
                assert "Action" in movie["genres"]

    def test_popular_movies_no_popularity_data(self, client, mock_redis):
        """Test popular movies when popularity data missing."""
        mock_redis.get_popular_items.return_value = None

        with patch(
            "src.api.dependencies.get_movie_metadata", new_callable=AsyncMock
        ) as mock_metadata:
            # Return DataFrame without popularity column
            mock_metadata.return_value = pd.DataFrame(
                {
                    "movieId": [1, 2],
                    "title": ["Movie 1", "Movie 2"],
                    "genres": ["Action", "Drama"],
                }
            )

            response = client.get("/api/v1/movies/popular")

            assert response.status_code == 503
            data = response.json()
            assert "NO_POPULARITY_DATA" in data["detail"]["error_code"]


# ==================== Movie Search Endpoint ====================


class TestMovieSearch:
    """Tests for GET /api/v1/movies/search endpoint."""

    def test_search_movies_success(self, client, mock_movie_metadata):
        """Test searching movies successfully."""
        response = client.get("/api/v1/movies/search?query=Movie&limit=10")

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "Movie"
        assert len(data["results"]) > 0
        assert data["count"] == len(data["results"])
        assert all("Movie" in r["title"] for r in data["results"])

    def test_search_movies_case_insensitive(self, client, mock_movie_metadata):
        """Test search is case-insensitive."""
        response = client.get("/api/v1/movies/search?query=movie")

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) > 0

    def test_search_movies_relevance_ranking(self, client, mock_movie_metadata):
        """Test search results are ranked by relevance."""
        response = client.get("/api/v1/movies/search?query=Test&limit=10")

        assert response.status_code == 200
        data = response.json()

        if len(data["results"]) > 1:
            # Earlier matches should have higher relevance
            assert (
                data["results"][0]["relevance_score"]
                >= data["results"][-1]["relevance_score"]
            )

    def test_search_movies_no_results(self, client, mock_movie_metadata):
        """Test search with no matching results."""
        response = client.get("/api/v1/movies/search?query=XyZabc123NonExistent")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        # Allow empty results or no results
        assert len(data["results"]) == 0

    def test_search_movies_missing_query(self, client):
        """Test search without query parameter."""
        response = client.get("/api/v1/movies/search")

        assert response.status_code == 422

    def test_search_movies_empty_query(self, client):
        """Test search with empty query."""
        response = client.get("/api/v1/movies/search?query=")

        assert response.status_code == 422


# ==================== Monitoring Endpoints ====================


class TestMetrics:
    """Tests for /api/v1/metrics endpoint."""

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/api/v1/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]


class TestSystemStats:
    """Tests for /api/v1/stats endpoint."""

    def test_system_stats_success(
        self, client, mock_qdrant, mock_redis, mock_movie_metadata
    ):
        """Test getting system statistics."""
        response = client.get("/api/v1/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_users" in data
        assert "total_movies" in data
        assert data["total_movies"] == len(mock_movie_metadata)  # From mock metadata
        assert "cache_stats" in data
        assert "qdrant_stats" in data
        assert "hit_rate" in data["cache_stats"]
        assert "user_points" in data["qdrant_stats"]


# ==================== Legacy Endpoints ====================


class TestValidationEndpoints:
    """Tests for validation endpoints."""

    def test_validate_valid_user(self, client):
        """Test validating a valid user ID."""
        response = client.get("/api/v1/validate/user/user123")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user123"
        assert data["is_valid"] is True

    def test_validate_invalid_user(self, client):
        """Test validating an invalid user ID with whitespace."""
        response = client.get("/api/v1/validate/user/ user123")
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False

    def test_validate_valid_item(self, client):
        """Test validating a valid item ID."""
        response = client.get("/api/v1/validate/item/item456")
        assert response.status_code == 200
        data = response.json()
        assert data["item_id"] == "item456"
        assert data["is_valid"] is True

    def test_validate_invalid_item(self, client):
        """Test validating an invalid item ID with whitespace."""
        response = client.get("/api/v1/validate/item/item456 ")
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False


class TestNormalizeScoreEndpoint:
    """Tests for score normalization endpoint."""

    def test_normalize_score_default_range(self, client):
        """Test normalizing a score with default range."""
        response = client.post("/api/v1/normalize-score?score=0.5")
        assert response.status_code == 200
        data = response.json()
        assert data["original_score"] == 0.5
        assert data["normalized_score"] == 0.5
        assert data["range"]["min"] == 0.0
        assert data["range"]["max"] == 1.0

    def test_normalize_score_above_range(self, client):
        """Test normalizing a score above the range."""
        response = client.post("/api/v1/normalize-score?score=1.5")
        assert response.status_code == 200
        data = response.json()
        assert data["normalized_score"] == 1.0

    def test_normalize_score_custom_range(self, client):
        """Test normalizing a score with custom range."""
        response = client.post(
            "/api/v1/normalize-score?score=5.0&min_val=0.0&max_val=10.0"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["normalized_score"] == 5.0
