"""Unit tests for the FastAPI application."""

import pytest
from starlette.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


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

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "recommendation-api"


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


class TestRecommendationsEndpoint:
    """Tests for recommendations endpoint."""

    def test_get_recommendations_valid_user(self, client):
        """Test getting recommendations for a valid user."""
        response = client.get("/api/v1/recommendations/user123")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user123"
        assert "recommendations" in data
        assert "limit" in data

    def test_get_recommendations_invalid_user(self, client):
        """Test getting recommendations for an invalid user."""
        response = client.get("/api/v1/recommendations/ ")
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_get_recommendations_with_limit(self, client):
        """Test getting recommendations with custom limit."""
        response = client.get("/api/v1/recommendations/user123?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 5
