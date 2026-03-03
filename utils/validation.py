"""
Validation utilities for API request data.
"""

from typing import List, Tuple

from flask import jsonify


# These will be set by the app during initialization
MAX_TOP_K = 100
DEFAULT_TOP_K = 10


def init_validation_config(max_top_k: int, default_top_k: int):
    """Initialize validation config from app config."""
    global MAX_TOP_K, DEFAULT_TOP_K
    MAX_TOP_K = max_top_k
    DEFAULT_TOP_K = default_top_k


def validate_top_k(data: dict) -> int:
    """
    Validate and return top_k from request data.
    Returns clamped value between 1 and MAX_TOP_K, defaults to DEFAULT_TOP_K.

    Raises:
        ValueError: If top_k is not a valid positive integer.
    """
    raw = data.get("top_k", DEFAULT_TOP_K)
    try:
        top_k = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"top_k must be a valid integer, got: {raw}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got: {top_k}")
    return min(top_k, MAX_TOP_K)


def validate_required_fields(data: dict, required_fields: List[str]) -> Tuple:
    """
    Validate that all required fields are present in the request data.

    Returns:
        None if valid, or (response, status_code) tuple if invalid.
    """
    if data is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Request body must be valid JSON",
                }
            ),
            400,
        )

    for field in required_fields:
        if field not in data:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Missing required field: {field}",
                    }
                ),
                400,
            )
    return None
