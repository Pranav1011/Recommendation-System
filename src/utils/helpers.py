"""Helper utility functions for the recommendation system."""


def validate_user_id(user_id: str) -> bool:
    """
    Validate that a user ID is properly formatted.

    Args:
        user_id: The user ID to validate

    Returns:
        True if valid, False otherwise
    """
    if not user_id or not isinstance(user_id, str):
        return False
    return len(user_id) > 0 and user_id.strip() == user_id


def validate_item_id(item_id: str) -> bool:
    """
    Validate that an item ID is properly formatted.

    Args:
        item_id: The item ID to validate

    Returns:
        True if valid, False otherwise
    """
    if not item_id or not isinstance(item_id, str):
        return False
    return len(item_id) > 0 and item_id.strip() == item_id


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize a score to be within a specified range.

    Args:
        score: The score to normalize
        min_val: Minimum value of the range (default: 0.0)
        max_val: Maximum value of the range (default: 1.0)

    Returns:
        Normalized score clamped to [min_val, max_val]
    """
    return max(min_val, min(max_val, score))
