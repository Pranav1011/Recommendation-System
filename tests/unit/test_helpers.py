"""Unit tests for helper utility functions."""
import pytest
from src.utils.helpers import validate_user_id, validate_item_id, normalize_score


class TestValidateUserId:
    """Tests for validate_user_id function."""

    def test_valid_user_id(self):
        """Test that valid user IDs return True."""
        assert validate_user_id("user123") is True
        assert validate_user_id("a") is True
        assert validate_user_id("user_123") is True

    def test_empty_user_id(self):
        """Test that empty user IDs return False."""
        assert validate_user_id("") is False

    def test_none_user_id(self):
        """Test that None user ID returns False."""
        assert validate_user_id(None) is False

    def test_user_id_with_whitespace(self):
        """Test that user IDs with leading/trailing whitespace return False."""
        assert validate_user_id(" user123") is False
        assert validate_user_id("user123 ") is False
        assert validate_user_id(" user123 ") is False

    def test_non_string_user_id(self):
        """Test that non-string user IDs return False."""
        assert validate_user_id(123) is False
        assert validate_user_id([]) is False


class TestValidateItemId:
    """Tests for validate_item_id function."""

    def test_valid_item_id(self):
        """Test that valid item IDs return True."""
        assert validate_item_id("item123") is True
        assert validate_item_id("a") is True
        assert validate_item_id("item_456") is True

    def test_empty_item_id(self):
        """Test that empty item IDs return False."""
        assert validate_item_id("") is False

    def test_none_item_id(self):
        """Test that None item ID returns False."""
        assert validate_item_id(None) is False

    def test_item_id_with_whitespace(self):
        """Test that item IDs with leading/trailing whitespace return False."""
        assert validate_item_id(" item123") is False
        assert validate_item_id("item123 ") is False

    def test_non_string_item_id(self):
        """Test that non-string item IDs return False."""
        assert validate_item_id(456) is False
        assert validate_item_id({}) is False


class TestNormalizeScore:
    """Tests for normalize_score function."""

    def test_score_within_range(self):
        """Test that scores within range remain unchanged."""
        assert normalize_score(0.5) == 0.5
        assert normalize_score(0.0) == 0.0
        assert normalize_score(1.0) == 1.0

    def test_score_below_range(self):
        """Test that scores below range are clamped to minimum."""
        assert normalize_score(-0.5) == 0.0
        assert normalize_score(-100.0) == 0.0

    def test_score_above_range(self):
        """Test that scores above range are clamped to maximum."""
        assert normalize_score(1.5) == 1.0
        assert normalize_score(100.0) == 1.0

    def test_custom_range(self):
        """Test normalization with custom min/max values."""
        assert normalize_score(5.0, min_val=0.0, max_val=10.0) == 5.0
        assert normalize_score(-1.0, min_val=0.0, max_val=10.0) == 0.0
        assert normalize_score(15.0, min_val=0.0, max_val=10.0) == 10.0

    def test_inverted_range(self):
        """Test normalization with inverted range (min > max)."""
        # Should still clamp correctly
        result = normalize_score(0.5, min_val=1.0, max_val=0.0)
        assert result == 1.0  # max(1.0, min(0.0, 0.5)) = max(1.0, 0.0) = 1.0
