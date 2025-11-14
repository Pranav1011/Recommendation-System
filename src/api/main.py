"""
Main FastAPI Application for Recommendation System

Production-grade API with:
- Personalized recommendations (Qdrant + Redis)
- Similar movie search
- Cold start handling
- Popular movies
- Movie search
- Health checks and monitoring
"""

import hashlib
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.api.dependencies import (
    get_movie_by_id,
    get_movie_metadata,
    get_qdrant_manager,
    get_redis_cache,
    shutdown_event,
    startup_event,
)
from src.api.middleware import recommendation_counter, setup_middleware
from src.api.models import (
    ColdStartRatingRequest,
    ColdStartResponse,
    DeepHealthResponse,
    MovieRecommendation,
    MovieSearchResponse,
    MovieSearchResult,
    PopularMoviesResponse,
    RecommendationRequest,
    RecommendationResponse,
    ServiceHealth,
    SimilarMoviesResponse,
    SystemStats,
)
from src.utils.helpers import normalize_score, validate_item_id, validate_user_id
from src.vector_store.qdrant_client import QdrantManager
from src.vector_store.redis_cache import RedisCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==================== Lifespan Management ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup/shutdown)."""
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()


# ==================== FastAPI App ====================

app = FastAPI(
    title="Movie Recommendation API",
    description=(
        "Production-grade recommendation system using collaborative filtering, "
        "vector search (Qdrant), and Redis caching. "
        "Built with Two-Tower neural network and LightGCN ensemble."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Setup middleware
setup_middleware(app)


# ==================== Root & Health Endpoints ====================


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Movie Recommendation API",
        "status": "running",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "recommendation-api",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get(
    "/api/v1/health/deep",
    response_model=DeepHealthResponse,
    tags=["Health"],
)
async def deep_health_check(
    qdrant: QdrantManager = Depends(get_qdrant_manager),
    redis: RedisCache = Depends(get_redis_cache),
):
    """
    Deep health check for all services.

    Checks:
    - Qdrant vector database
    - Redis cache
    - Movie metadata
    """
    services = []
    overall_healthy = True

    # Check Qdrant
    start = time.time()
    qdrant_healthy = qdrant.health_check()
    qdrant_latency = (time.time() - start) * 1000

    if qdrant_healthy:
        movie_info = qdrant.get_collection_info(qdrant.MOVIE_COLLECTION)
        qdrant_details = {"points_count": movie_info.get("points_count", 0)}
    else:
        qdrant_details = None
        overall_healthy = False

    services.append(
        ServiceHealth(
            service="qdrant",
            healthy=qdrant_healthy,
            latency_ms=round(qdrant_latency, 2),
            details=qdrant_details,
        )
    )

    # Check Redis
    start = time.time()
    redis_healthy = redis.health_check()
    redis_latency = (time.time() - start) * 1000

    if redis_healthy:
        redis_stats = redis.get_stats()
        redis_details = {
            "hit_rate": round(redis_stats.get("hit_rate", 0), 3),
            "used_memory": redis_stats.get("used_memory_human", "N/A"),
        }
    else:
        redis_details = None
        overall_healthy = False

    services.append(
        ServiceHealth(
            service="redis",
            healthy=redis_healthy,
            latency_ms=round(redis_latency, 2),
            details=redis_details,
        )
    )

    # Check movie metadata
    try:
        metadata = await get_movie_metadata()
        metadata_healthy = len(metadata) > 0
        metadata_details = {"movie_count": len(metadata)}
    except Exception:
        metadata_healthy = False
        metadata_details = None
        overall_healthy = False

    services.append(
        ServiceHealth(
            service="movie_metadata",
            healthy=metadata_healthy,
            latency_ms=0.0,
            details=metadata_details,
        )
    )

    return DeepHealthResponse(
        status="healthy" if overall_healthy else "degraded",
        services=services,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# ==================== Recommendation Endpoints ====================


@app.post(
    "/api/v1/recommend",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    status_code=status.HTTP_200_OK,
)
async def get_recommendations_post(
    request: RecommendationRequest,
    qdrant: QdrantManager = Depends(get_qdrant_manager),
    redis: RedisCache = Depends(get_redis_cache),
):
    """
    Get personalized movie recommendations for a user.

    This endpoint uses vector similarity search in Qdrant to find movies
    similar to the user's preferences. Results are cached in Redis for
    fast repeated queries.

    **Example Request:**
    ```json
    {
      "user_id": 123,
      "k": 10,
      "min_rating": 4.0,
      "genres": ["Action", "Sci-Fi"],
      "min_year": 2000
    }
    ```
    """
    # Build filters dict for cache key
    filters = {}
    if request.min_rating is not None:
        filters["min_rating"] = request.min_rating
    if request.genres:
        filters["genres"] = sorted(request.genres)
    if request.min_year is not None:
        filters["min_year"] = request.min_year
    if request.max_year is not None:
        filters["max_year"] = request.max_year

    # Check cache
    cached_results = redis.get_recommendations(request.user_id, filters or None)
    if cached_results:
        logger.info(f"Cache hit for user {request.user_id}")
        recommendation_counter.labels(endpoint="recommend", cached="true").inc()

        # Convert to response format
        recommendations = [
            MovieRecommendation(
                movie_id=movie_id,
                title=metadata.get("title", "Unknown"),
                score=score,
                genres=metadata.get("genres", []),
                year=metadata.get("year"),
                avg_rating=metadata.get("avg_rating"),
                popularity=metadata.get("popularity"),
            )
            for movie_id, score, metadata in cached_results[: request.k]
        ]

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            count=len(recommendations),
            cached=True,
            filters_applied=filters or None,
        )

    # Get recommendations from Qdrant
    try:
        results = qdrant.get_recommendations(
            user_id=request.user_id,
            k=request.k,
            genre_filter=request.genres,
            min_year=request.min_year,
            max_year=request.max_year,
            min_rating=request.min_rating,
        )
    except Exception as e:
        logger.error(f"Qdrant query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to generate recommendations",
                "detail": str(e),
                "error_code": "QDRANT_QUERY_FAILED",
            },
        )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "User not found",
                "detail": f"User ID {request.user_id} does not exist",
                "error_code": "USER_NOT_FOUND",
            },
        )

    # Convert to response format
    recommendations = [
        MovieRecommendation(
            movie_id=movie_id,
            title=metadata.get("title", "Unknown"),
            score=round(score, 4),
            genres=metadata.get("genres", []),
            year=metadata.get("year"),
            avg_rating=metadata.get("avg_rating"),
            popularity=metadata.get("popularity"),
        )
        for movie_id, score, metadata in results
    ]

    # Cache results
    redis.set_recommendations(request.user_id, results, filters or None)
    recommendation_counter.labels(endpoint="recommend", cached="false").inc()

    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        count=len(recommendations),
        cached=False,
        filters_applied=filters or None,
    )


@app.get(
    "/api/v1/recommend/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
)
async def get_recommendations_get(
    user_id: int,
    k: int = Query(10, ge=1, le=100, description="Number of recommendations"),
    min_rating: Optional[float] = Query(
        None, ge=0.0, le=5.0, description="Minimum average rating"
    ),
    genres: Optional[str] = Query(
        None, description="Comma-separated genres (e.g., 'Action,Sci-Fi')"
    ),
    min_year: Optional[int] = Query(None, ge=1900, description="Minimum release year"),
    max_year: Optional[int] = Query(None, le=2100, description="Maximum release year"),
    qdrant: QdrantManager = Depends(get_qdrant_manager),
    redis: RedisCache = Depends(get_redis_cache),
):
    """
    Get personalized movie recommendations (GET version).

    Same as POST /api/v1/recommend but using query parameters.

    **Example:**
    ```
    GET /api/v1/recommend/123?k=10&min_rating=4.0&genres=Action,Sci-Fi
    ```
    """
    # Parse genres
    genre_list = None
    if genres:
        genre_list = [g.strip() for g in genres.split(",") if g.strip()]

    # Create request object
    request = RecommendationRequest(
        user_id=user_id,
        k=k,
        min_rating=min_rating,
        genres=genre_list,
        min_year=min_year,
        max_year=max_year,
    )

    # Reuse POST endpoint logic
    return await get_recommendations_post(request, qdrant, redis)


@app.get(
    "/api/v1/similar/{movie_id}",
    response_model=SimilarMoviesResponse,
    tags=["Recommendations"],
)
async def get_similar_movies(
    movie_id: int,
    k: int = Query(10, ge=1, le=100, description="Number of similar movies"),
    qdrant: QdrantManager = Depends(get_qdrant_manager),
):
    """
    Find similar movies based on vector similarity.

    Uses the movie's embedding to find other movies with similar
    characteristics (genre, themes, user preferences).

    **Example:**
    ```
    GET /api/v1/similar/1?k=10
    ```
    """
    # Get movie metadata
    movie_info = await get_movie_by_id(movie_id)
    if not movie_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Movie not found",
                "detail": f"Movie ID {movie_id} does not exist",
                "error_code": "MOVIE_NOT_FOUND",
            },
        )

    # Get movie embedding
    # Note: We need to find the movie's index in the embedding matrix
    # This requires a mapping from movie_id to index
    metadata = await get_movie_metadata()
    movie_idx = metadata[metadata["movieId"] == movie_id].index
    if len(movie_idx) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Movie embedding not found", "error_code": "NO_EMBEDDING"},
        )

    movie_idx = movie_idx[0]
    movie_embedding = qdrant.get_movie_embedding(movie_idx)

    if movie_embedding is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Movie embedding not found", "error_code": "NO_EMBEDDING"},
        )

    # Search for similar movies (k+1 because first result is the query movie itself)
    try:
        results = qdrant.search_similar_movies(movie_embedding, k=k + 1)
    except Exception as e:
        logger.error(f"Similar movie search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Search failed", "detail": str(e)},
        )

    # Filter out the query movie and convert to response
    similar_movies = [
        MovieRecommendation(
            movie_id=mid,
            title=meta.get("title", "Unknown"),
            score=round(score, 4),
            genres=meta.get("genres", []),
            year=meta.get("year"),
            avg_rating=meta.get("avg_rating"),
            popularity=meta.get("popularity"),
        )
        for mid, score, meta in results
        if mid != movie_id
    ][:k]

    recommendation_counter.labels(endpoint="similar", cached="false").inc()

    return SimilarMoviesResponse(
        movie_id=movie_id,
        query_title=movie_info["title"],
        similar_movies=similar_movies,
        count=len(similar_movies),
    )


# ==================== Cold Start Endpoint ====================


@app.post(
    "/api/v1/cold-start/rate",
    response_model=ColdStartResponse,
    tags=["Cold Start"],
)
async def cold_start_recommendations(
    request: ColdStartRatingRequest,
    qdrant: QdrantManager = Depends(get_qdrant_manager),
):
    """
    Handle cold start for new users.

    New users provide initial ratings, and we generate a temporary
    user embedding by averaging the embeddings of rated movies
    (weighted by rating).

    **Example Request:**
    ```json
    {
      "ratings": [
        {"movie_id": 1, "rating": 5.0},
        {"movie_id": 296, "rating": 4.5},
        {"movie_id": 318, "rating": 5.0}
      ]
    }
    ```
    """
    metadata = await get_movie_metadata()

    # Create temporary user embedding from rated movies
    import numpy as np

    weighted_embeddings = []
    total_weight = 0

    for rating_data in request.ratings:
        movie_id = int(rating_data["movie_id"])
        rating = float(rating_data["rating"])

        # Find movie index
        movie_idx = metadata[metadata["movieId"] == movie_id].index
        if len(movie_idx) == 0:
            logger.warning(f"Movie {movie_id} not found, skipping")
            continue

        movie_idx = movie_idx[0]
        movie_embedding = qdrant.get_movie_embedding(movie_idx)

        if movie_embedding is not None:
            # Weight by rating (normalize to 0-1 range)
            weight = (rating - 0.5) / 4.5  # 0.5-5.0 -> 0-1
            weighted_embeddings.append(movie_embedding * weight)
            total_weight += weight

    if not weighted_embeddings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "No valid movie ratings",
                "detail": "None of the provided movie IDs were found",
                "error_code": "INVALID_RATINGS",
            },
        )

    # Average weighted embeddings
    temp_user_embedding = np.sum(weighted_embeddings, axis=0) / total_weight

    # Normalize
    temp_user_embedding = temp_user_embedding / np.linalg.norm(temp_user_embedding)

    # Search for recommendations
    try:
        results = qdrant.search_similar_movies(temp_user_embedding, k=20)
    except Exception as e:
        logger.error(f"Cold start search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Search failed", "detail": str(e)},
        )

    # Filter out movies the user already rated
    rated_movie_ids = {int(r["movie_id"]) for r in request.ratings}
    recommendations = [
        MovieRecommendation(
            movie_id=mid,
            title=meta.get("title", "Unknown"),
            score=round(score, 4),
            genres=meta.get("genres", []),
            year=meta.get("year"),
            avg_rating=meta.get("avg_rating"),
            popularity=meta.get("popularity"),
        )
        for mid, score, meta in results
        if mid not in rated_movie_ids
    ][:10]

    # Generate temporary user ID
    ratings_hash = hashlib.md5(str(sorted(rated_movie_ids)).encode()).hexdigest()[:8]
    temp_user_id = f"temp_{ratings_hash}"

    recommendation_counter.labels(endpoint="cold_start", cached="false").inc()

    return ColdStartResponse(
        temp_user_id=temp_user_id,
        recommendations=recommendations,
        count=len(recommendations),
        ratings_processed=len(request.ratings),
    )


# ==================== Popular Movies Endpoint ====================


@app.get(
    "/api/v1/movies/popular",
    response_model=PopularMoviesResponse,
    tags=["Movies"],
)
async def get_popular_movies(
    limit: int = Query(20, ge=1, le=100, description="Number of movies to return"),
    genre: Optional[str] = Query(None, description="Filter by genre"),
    redis: RedisCache = Depends(get_redis_cache),
):
    """
    Get popular movies (cached).

    Returns the most popular movies based on number of ratings.
    Results are heavily cached since popularity changes slowly.

    **Example:**
    ```
    GET /api/v1/movies/popular?limit=20&genre=Action
    ```
    """
    # Check cache first
    cached_popular = redis.get_popular_items(genre=genre, k=limit)

    if cached_popular:
        # Convert to response format
        metadata = await get_movie_metadata()
        popular_movies = []

        for movie_id, popularity_score in cached_popular:
            movie_info = metadata[metadata["movieId"] == movie_id]
            if not movie_info.empty:
                row = movie_info.iloc[0]
                popular_movies.append(
                    MovieRecommendation(
                        movie_id=int(row["movieId"]),
                        title=str(row["title"]),
                        score=float(popularity_score),
                        genres=(
                            row["genres"].split("|")
                            if isinstance(row["genres"], str)
                            else []
                        ),
                        year=int(row["year"]) if pd.notna(row.get("year")) else None,
                        avg_rating=(
                            float(row["avg_rating"]) if "avg_rating" in row else None
                        ),
                        popularity=(
                            int(row["popularity"]) if "popularity" in row else None
                        ),
                    )
                )

        return PopularMoviesResponse(
            popular_movies=popular_movies,
            count=len(popular_movies),
            genre_filter=genre,
            cached=True,
        )

    # Compute popular movies from metadata
    metadata = await get_movie_metadata()

    if "popularity" not in metadata.columns:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Popularity data not available",
                "error_code": "NO_POPULARITY_DATA",
            },
        )

    # Filter by genre if specified
    if genre:
        metadata = metadata[
            metadata["genres"].str.contains(genre, case=False, na=False)
        ]

    # Sort by popularity and get top N
    top_movies = metadata.nlargest(limit, "popularity")

    popular_movies = [
        MovieRecommendation(
            movie_id=int(row["movieId"]),
            title=str(row["title"]),
            score=float(row["popularity"]),
            genres=row["genres"].split("|") if isinstance(row["genres"], str) else [],
            year=int(row["year"]) if pd.notna(row.get("year")) else None,
            avg_rating=float(row["avg_rating"]) if "avg_rating" in row else None,
            popularity=int(row["popularity"]) if "popularity" in row else None,
        )
        for _, row in top_movies.iterrows()
    ]

    # Cache results
    cache_data = [(m.movie_id, m.score) for m in popular_movies]
    redis.set_popular_items(cache_data, genre=genre, k=limit)

    return PopularMoviesResponse(
        popular_movies=popular_movies,
        count=len(popular_movies),
        genre_filter=genre,
        cached=False,
    )


# ==================== Movie Search Endpoint ====================


@app.get(
    "/api/v1/movies/search",
    response_model=MovieSearchResponse,
    tags=["Movies"],
)
async def search_movies(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
):
    """
    Search movies by title.

    Simple text search on movie titles. Returns movies ranked by relevance.

    **Example:**
    ```
    GET /api/v1/movies/search?query=star%20wars&limit=10
    ```
    """
    metadata = await get_movie_metadata()

    # Simple case-insensitive search
    query_lower = query.lower()
    matches = metadata[
        metadata["title"].str.lower().str.contains(query_lower, na=False)
    ]

    # Rank by how early the query appears in the title
    def get_relevance(title):
        title_lower = title.lower()
        if query_lower in title_lower:
            # Earlier match = higher relevance
            position = title_lower.index(query_lower)
            return 1.0 / (1 + position)
        return 0.0

    matches["relevance"] = matches["title"].apply(get_relevance)
    matches = matches.nlargest(limit, "relevance")

    # Convert to response
    results = [
        MovieSearchResult(
            movie_id=int(row["movieId"]),
            title=str(row["title"]),
            genres=row["genres"].split("|") if isinstance(row["genres"], str) else [],
            year=int(row["year"]) if pd.notna(row.get("year")) else None,
            relevance_score=round(float(row["relevance"]), 4),
        )
        for _, row in matches.iterrows()
    ]

    return MovieSearchResponse(
        query=query,
        results=results,
        count=len(results),
    )


# ==================== Monitoring Endpoints ====================


@app.get("/api/v1/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus format for scraping.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get(
    "/api/v1/stats",
    response_model=SystemStats,
    tags=["Monitoring"],
)
async def system_stats(
    qdrant: QdrantManager = Depends(get_qdrant_manager),
    redis: RedisCache = Depends(get_redis_cache),
):
    """
    Get system statistics.

    Returns statistics about the recommendation system including
    cache hit rates, database sizes, etc.
    """
    # Get metadata stats
    metadata = await get_movie_metadata()
    total_movies = len(metadata)

    # Get Qdrant stats
    user_info = qdrant.get_collection_info(qdrant.USER_COLLECTION)
    total_users = user_info.get("points_count", 0)

    movie_info = qdrant.get_collection_info(qdrant.MOVIE_COLLECTION)
    qdrant_stats = {
        "movie_points": movie_info.get("points_count", 0),
        "user_points": total_users,
        "status": movie_info.get("status", "unknown"),
    }

    # Get Redis stats
    cache_stats = redis.get_stats()

    return SystemStats(
        total_users=total_users,
        total_movies=total_movies,
        cache_stats=cache_stats,
        qdrant_stats=qdrant_stats,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# ==================== Legacy Endpoints (Backward Compatibility) ====================


@app.get("/api/v1/validate/user/{user_id}", tags=["Legacy"])
async def validate_user(user_id: str):
    """Validate a user ID (legacy endpoint)."""
    is_valid = validate_user_id(user_id)
    return {"user_id": user_id, "is_valid": is_valid}


@app.get("/api/v1/validate/item/{item_id}", tags=["Legacy"])
async def validate_item(item_id: str):
    """Validate an item ID (legacy endpoint)."""
    is_valid = validate_item_id(item_id)
    return {"item_id": item_id, "is_valid": is_valid}


@app.post("/api/v1/normalize-score", tags=["Legacy"])
async def normalize_score_endpoint(
    score: float, min_val: float = 0.0, max_val: float = 1.0
):
    """Normalize a score to a specified range (legacy endpoint)."""
    normalized = normalize_score(score, min_val, max_val)
    return {
        "original_score": score,
        "normalized_score": normalized,
        "range": {"min": min_val, "max": max_val},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
