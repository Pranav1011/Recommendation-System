"""
Pydantic Models for API Request/Response Validation

Defines all request and response schemas for the recommendation API.
Includes validation, examples, and proper error messages.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ==================== Request Models ====================


class RecommendationRequest(BaseModel):
    """Request body for POST /api/v1/recommend endpoint."""

    user_id: int = Field(
        ..., description="User ID to get recommendations for", ge=0, example=123
    )
    k: int = Field(
        10, description="Number of recommendations to return", ge=1, le=100
    )
    min_rating: Optional[float] = Field(
        None, description="Minimum average rating filter", ge=0.0, le=5.0, example=4.0
    )
    genres: Optional[List[str]] = Field(
        None, description="Filter by genres (OR condition)", example=["Action", "Sci-Fi"]
    )
    min_year: Optional[int] = Field(
        None, description="Minimum release year", ge=1900, le=2100, example=2000
    )
    max_year: Optional[int] = Field(
        None, description="Maximum release year", ge=1900, le=2100, example=2023
    )

    @field_validator("genres")
    @classmethod
    def validate_genres(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate genre list."""
        if v is not None:
            # Remove empty strings and duplicates
            v = [g.strip() for g in v if g.strip()]
            if not v:
                return None
            return list(set(v))
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "k": 10,
                "min_rating": 4.0,
                "genres": ["Action", "Sci-Fi"],
                "min_year": 2000,
            }
        }


class ColdStartRatingRequest(BaseModel):
    """Request body for POST /api/v1/cold-start/rate endpoint."""

    ratings: List[Dict[str, float]] = Field(
        ...,
        description="List of movie ratings from new user",
        min_length=1,
        max_length=50,
        example=[{"movie_id": 1, "rating": 5.0}, {"movie_id": 2, "rating": 4.0}],
    )

    @field_validator("ratings")
    @classmethod
    def validate_ratings(cls, v: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Validate rating format."""
        for rating in v:
            if "movie_id" not in rating or "rating" not in rating:
                raise ValueError("Each rating must have 'movie_id' and 'rating' fields")
            if not isinstance(rating["movie_id"], (int, float)):
                raise ValueError("movie_id must be a number")
            if not isinstance(rating["rating"], (int, float)):
                raise ValueError("rating must be a number")
            if not 0.5 <= rating["rating"] <= 5.0:
                raise ValueError("rating must be between 0.5 and 5.0")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "ratings": [
                    {"movie_id": 1, "rating": 5.0},
                    {"movie_id": 296, "rating": 4.5},
                    {"movie_id": 318, "rating": 5.0},
                ]
            }
        }


# ==================== Response Models ====================


class MovieRecommendation(BaseModel):
    """Single movie recommendation with metadata."""

    movie_id: int = Field(..., description="Movie ID")
    title: str = Field(..., description="Movie title")
    score: float = Field(..., description="Recommendation score/confidence")
    genres: List[str] = Field(default_factory=list, description="Movie genres")
    year: Optional[int] = Field(None, description="Release year")
    avg_rating: Optional[float] = Field(None, description="Average user rating")
    popularity: Optional[int] = Field(None, description="Number of ratings")

    class Config:
        json_schema_extra = {
            "example": {
                "movie_id": 318,
                "title": "Shawshank Redemption, The (1994)",
                "score": 0.89,
                "genres": ["Crime", "Drama"],
                "year": 1994,
                "avg_rating": 4.5,
                "popularity": 81491,
            }
        }


class RecommendationResponse(BaseModel):
    """Response for recommendation endpoints."""

    user_id: int = Field(..., description="User ID")
    recommendations: List[MovieRecommendation] = Field(
        ..., description="List of recommended movies"
    )
    count: int = Field(..., description="Number of recommendations returned")
    cached: bool = Field(
        False, description="Whether results were served from cache"
    )
    filters_applied: Optional[Dict[str, Any]] = Field(
        None, description="Filters that were applied"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "recommendations": [
                    {
                        "movie_id": 318,
                        "title": "Shawshank Redemption, The (1994)",
                        "score": 0.89,
                        "genres": ["Crime", "Drama"],
                        "year": 1994,
                        "avg_rating": 4.5,
                        "popularity": 81491,
                    }
                ],
                "count": 1,
                "cached": False,
                "filters_applied": {"min_rating": 4.0},
            }
        }


class SimilarMoviesResponse(BaseModel):
    """Response for similar movies endpoint."""

    movie_id: int = Field(..., description="Query movie ID")
    query_title: str = Field(..., description="Query movie title")
    similar_movies: List[MovieRecommendation] = Field(
        ..., description="List of similar movies"
    )
    count: int = Field(..., description="Number of similar movies returned")

    class Config:
        json_schema_extra = {
            "example": {
                "movie_id": 1,
                "query_title": "Toy Story (1995)",
                "similar_movies": [
                    {
                        "movie_id": 2,
                        "title": "Jumanji (1995)",
                        "score": 0.78,
                        "genres": ["Adventure", "Children", "Fantasy"],
                    }
                ],
                "count": 1,
            }
        }


class ColdStartResponse(BaseModel):
    """Response for cold start recommendation endpoint."""

    temp_user_id: str = Field(
        ..., description="Temporary user ID for this session"
    )
    recommendations: List[MovieRecommendation] = Field(
        ..., description="Initial recommendations based on ratings"
    )
    count: int = Field(..., description="Number of recommendations returned")
    ratings_processed: int = Field(
        ..., description="Number of ratings processed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "temp_user_id": "temp_12345",
                "recommendations": [
                    {
                        "movie_id": 260,
                        "title": "Star Wars: Episode IV - A New Hope (1977)",
                        "score": 0.85,
                        "genres": ["Action", "Adventure", "Sci-Fi"],
                    }
                ],
                "count": 1,
                "ratings_processed": 3,
            }
        }


class PopularMoviesResponse(BaseModel):
    """Response for popular movies endpoint."""

    popular_movies: List[MovieRecommendation] = Field(
        ..., description="List of popular movies"
    )
    count: int = Field(..., description="Number of movies returned")
    genre_filter: Optional[str] = Field(
        None, description="Genre filter applied"
    )
    cached: bool = Field(
        True, description="Whether results were served from cache"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "popular_movies": [
                    {
                        "movie_id": 318,
                        "title": "Shawshank Redemption, The (1994)",
                        "score": 81491.0,
                        "genres": ["Crime", "Drama"],
                        "popularity": 81491,
                    }
                ],
                "count": 1,
                "genre_filter": None,
                "cached": True,
            }
        }


class MovieSearchResult(BaseModel):
    """Single movie search result."""

    movie_id: int = Field(..., description="Movie ID")
    title: str = Field(..., description="Movie title")
    genres: List[str] = Field(default_factory=list, description="Movie genres")
    year: Optional[int] = Field(None, description="Release year")
    relevance_score: float = Field(
        ..., description="Search relevance score"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "movie_id": 1,
                "title": "Toy Story (1995)",
                "genres": ["Adventure", "Animation", "Children"],
                "year": 1995,
                "relevance_score": 1.0,
            }
        }


class MovieSearchResponse(BaseModel):
    """Response for movie search endpoint."""

    query: str = Field(..., description="Search query")
    results: List[MovieSearchResult] = Field(
        ..., description="Search results"
    )
    count: int = Field(..., description="Number of results")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "toy story",
                "results": [
                    {
                        "movie_id": 1,
                        "title": "Toy Story (1995)",
                        "genres": ["Adventure", "Animation", "Children"],
                        "year": 1995,
                        "relevance_score": 1.0,
                    }
                ],
                "count": 1,
            }
        }


# ==================== Health & Monitoring Models ====================


class ServiceHealth(BaseModel):
    """Health status of a single service."""

    service: str = Field(..., description="Service name")
    healthy: bool = Field(..., description="Health status")
    latency_ms: Optional[float] = Field(
        None, description="Response latency in milliseconds"
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional details"
    )


class DeepHealthResponse(BaseModel):
    """Response for deep health check endpoint."""

    status: str = Field(..., description="Overall health status")
    services: List[ServiceHealth] = Field(
        ..., description="Health status of all services"
    )
    timestamp: str = Field(..., description="Health check timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "services": [
                    {
                        "service": "qdrant",
                        "healthy": True,
                        "latency_ms": 5.2,
                        "details": {"points_count": 62423},
                    },
                    {
                        "service": "redis",
                        "healthy": True,
                        "latency_ms": 1.1,
                        "details": {"hit_rate": 0.85},
                    },
                ],
                "timestamp": "2025-01-29T12:00:00Z",
            }
        }


class SystemStats(BaseModel):
    """Response for system statistics endpoint."""

    total_users: int = Field(..., description="Total number of users")
    total_movies: int = Field(..., description="Total number of movies")
    cache_stats: Dict[str, Any] = Field(
        ..., description="Cache statistics"
    )
    qdrant_stats: Dict[str, Any] = Field(
        ..., description="Qdrant statistics"
    )
    timestamp: str = Field(..., description="Statistics timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "total_users": 137883,
                "total_movies": 62423,
                "cache_stats": {
                    "hit_rate": 0.85,
                    "used_memory_human": "256MB",
                },
                "qdrant_stats": {"points_count": 62423, "status": "green"},
                "timestamp": "2025-01-29T12:00:00Z",
            }
        }


# ==================== Error Models ====================


class ErrorDetail(BaseModel):
    """Detailed error information."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error description")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: Optional[str] = Field(None, description="Error timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "User not found",
                "detail": "User ID 999999 does not exist in the database",
                "error_code": "USER_NOT_FOUND",
                "timestamp": "2025-01-29T12:00:00Z",
            }
        }
