"""
Validation utilities for API request data.
"""

import os
from typing import Any, Dict, List, Tuple

from flask import jsonify


# These will be set by the app during initialization
MAX_TOP_K = 100
DEFAULT_TOP_K = 10
MAX_TEXT_LENGTH = 10_000
MAX_IMAGE_SIZE_BYTES = 50 * 1024 * 1024


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


def deduplicate_text_results(search_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Keep only the best score per product for text results."""
    product_scores: Dict[str, float] = {}
    for result in search_results:
        product_id = result["product_id"]
        score = result["score"]
        if product_id not in product_scores:
            product_scores[product_id] = score
        else:
            product_scores[product_id] = max(product_scores[product_id], score)
    return product_scores


def deduplicate_visual_results(
    search_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Keep best visual score and its image number per product."""
    product_scores: Dict[str, Dict[str, Any]] = {}
    for result in search_results:
        product_id = result["product_id"]
        score = result["score"]
        image_no = result.get("image_no", 0)

        if product_id not in product_scores:
            product_scores[product_id] = {
                "score": score,
                "image_no": image_no,
            }
        else:
            if score > product_scores[product_id]["score"]:
                product_scores[product_id] = {
                    "score": score,
                    "image_no": image_no,
                }
    return product_scores


def deduplicate_fused_results(
    search_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Keep best fused score and its image number per product."""
    product_scores: Dict[str, Dict[str, Any]] = {}
    for result in search_results:
        product_id = result["product_id"]
        score = result["score"]
        image_no = result.get("image_no", 0)

        if product_id not in product_scores:
            product_scores[product_id] = {
                "score": score,
                "image_no": image_no,
            }
        else:
            if score > product_scores[product_id]["score"]:
                product_scores[product_id] = {
                    "score": score,
                    "image_no": image_no,
                }
    return product_scores


def validate_clip_model(model_name: str):
    """Raise ValueError if the model is not a CLIP model.

    Cross-modal search requires CLIP because text and image
    must share the same embedding space.
    """
    from services.manager_service import MODEL_REGISTRY

    model_info = MODEL_REGISTRY.get(model_name)
    if not model_info or model_info["type"] != "clip":
        model_type = model_info["type"] if model_info else "unknown"
        raise ValueError(
            f"Cross-modal search requires a CLIP model. "
            f"'{model_name}' is type '{model_type}'."
        )


def validate_text_length(text: str, max_length: int = MAX_TEXT_LENGTH):
    """Validate text payload size to prevent overly large inputs."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if len(text) > max_length:
        raise ValueError(f"Text exceeds maximum length of {max_length} characters")


def validate_image_file_size(
    image_path: str, max_size_bytes: int = MAX_IMAGE_SIZE_BYTES
):
    """
    Validate image file size when the image path exists locally.

    Missing files are validated later by embedding/model loading logic.
    """
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("image must be a non-empty file path")

    if not os.path.exists(image_path):
        return

    file_size = os.path.getsize(image_path)
    if file_size > max_size_bytes:
        raise ValueError("Image size exceeds maximum limit of 50MB")
