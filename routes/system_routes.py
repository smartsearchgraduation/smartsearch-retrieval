"""
System Routes — health check, index stats endpoints.
"""

from flask import Blueprint, jsonify

from services.manager_service import get_available_models, get_all_index_stats


system_bp = Blueprint("system", __name__)


@system_bp.route("/api/retrieval/index-stats", methods=["GET"])
def get_index_stats():
    """
    Get per-model statistics about the FAISS indices.

    Response:
    {
        "status": "success",
        "indices": {
            "bge-large-en-v1.5_1024_embeddings": {"textual": 50, "visual": 0, "fused": 0},
            "ViT-B-32_512_embeddings": {"textual": 0, "visual": 120, "fused": 0}
        }
    }
    """
    try:
        stats = get_all_index_stats()

        return (
            jsonify(
                {
                    "status": "success",
                    "indices": stats,
                }
            ),
            200,
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                }
            ),
            500,
        )


@system_bp.route("/api/retrieval/models", methods=["GET"])
def get_models():
    """
    List available embedding models and defaults.

    Response:
    {
        "status": "success",
        "data": {
            "textual_models": [{"name": "...", "dimension": 512}, ...],
            "visual_models": [...],
            "defaults": {"textual": "...", "visual": "..."}
        }
    }
    """
    try:
        models_data = get_available_models()

        return (
            jsonify(
                {
                    "status": "success",
                    "data": models_data,
                }
            ),
            200,
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                }
            ),
            500,
        )


@system_bp.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return (
        jsonify(
            {
                "status": "healthy",
                "service": "E-Commerce Product Retrieval System",
            }
        ),
        200,
    )
