"""Main FastAPI application for the recommendation system."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.utils.helpers import normalize_score, validate_item_id, validate_user_id

app = FastAPI(
    title="Recommendation System API",
    description="A scalable recommendation system using collaborative filtering and vector search",
    version="0.1.0",
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Recommendation System API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "recommendation-api"}


@app.get("/api/v1/validate/user/{user_id}")
async def validate_user(user_id: str):
    """Validate a user ID."""
    is_valid = validate_user_id(user_id)
    return {"user_id": user_id, "is_valid": is_valid}


@app.get("/api/v1/validate/item/{item_id}")
async def validate_item(item_id: str):
    """Validate an item ID."""
    is_valid = validate_item_id(item_id)
    return {"item_id": item_id, "is_valid": is_valid}


@app.post("/api/v1/normalize-score")
async def normalize_score_endpoint(
    score: float, min_val: float = 0.0, max_val: float = 1.0
):
    """Normalize a score to a specified range."""
    normalized = normalize_score(score, min_val, max_val)
    return {
        "original_score": score,
        "normalized_score": normalized,
        "range": {"min": min_val, "max": max_val},
    }


@app.get("/api/v1/recommendations/{user_id}")
async def get_recommendations(user_id: str, limit: int = 10):
    """Get recommendations for a user (placeholder implementation)."""
    if not validate_user_id(user_id):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid user ID"},
        )

    # Placeholder - will be implemented with actual recommendation logic
    return {
        "user_id": user_id,
        "recommendations": [],
        "limit": limit,
        "message": "Recommendation engine not yet implemented",
    }
