"""
System Routes — health check, index stats endpoints.
"""

from flask import Blueprint, jsonify

from services.manager_service import get_faiss_manager, get_available_models


system_bp = Blueprint("system", __name__)


@system_bp.route("/api/retrieval/index-stats", methods=["GET"])
def get_index_stats():
    """
    Get statistics about the FAISS indices.

    Response:
    {
        "status": "success",
        "indices": {
            "textual": 100,
            "visual": 250,
            "fused": 0
        }
    }
    """
    try:
        faiss_manager = get_faiss_manager()
        sizes = faiss_manager.get_all_sizes()

        return (
            jsonify(
                {
                    "status": "success",
                    "indices": sizes,
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
