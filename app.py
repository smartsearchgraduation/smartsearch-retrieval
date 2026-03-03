"""
E-Commerce Product Retrieval System
Flask API for managing products and performing similarity search.
"""

from flask import Flask, request, jsonify
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Import managers
from models.textual_models import TextModelManager
from models.visual_models import VisualModelManager
from models.fused_models import FusedModelManager
from vector_db import FAISSManager


app = Flask(__name__)

# Global managers (initialized lazily)
_textual_managers: Dict[str, TextModelManager] = {}
_visual_managers: Dict[str, VisualModelManager] = {}
_fused_managers: Dict[str, FusedModelManager] = {}
_faiss_manager: Optional[FAISSManager] = None

import json
import os


def _load_config() -> dict:
    """Load configuration from config.json."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in configuration file: {e}")


# Load configuration
_config = _load_config()
MODEL_REGISTRY = _config["models"]

# Defaults from config
MAX_TOP_K = _config["defaults"].get("max_top_k", 100)
DEFAULT_TOP_K = _config["defaults"].get("top_k", 10)
INDEX_PATH = _config["defaults"].get("index_path", "./data/faiss_indices")
HOST = _config["defaults"].get("host", "0.0.0.0")
PORT = _config["defaults"].get("port", 5002)
DEFAULT_DIMENSION = _config["defaults"].get("dimension", 512)


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


def get_faiss_manager(
    textual_model_name: str = None,
    visual_model_name: str = None,
    fused_model_name: str = None,
) -> FAISSManager:
    """
    Get or initialize the FAISS manager.

    Dimensions are determined from the MODEL_REGISTRY based on the provided model names.
    The first call with model names sets the dimensions for each index type.
    """
    global _faiss_manager
    if _faiss_manager is None:
        # Get dimensions from MODEL_REGISTRY based on provided model names
        textual_dim = (
            MODEL_REGISTRY.get(textual_model_name, {}).get("dimension", DEFAULT_DIMENSION)
            if textual_model_name
            else DEFAULT_DIMENSION
        )
        visual_dim = (
            MODEL_REGISTRY.get(visual_model_name, {}).get("dimension", DEFAULT_DIMENSION)
            if visual_model_name
            else DEFAULT_DIMENSION
        )
        fused_dim = (
            MODEL_REGISTRY.get(fused_model_name, {}).get("dimension", DEFAULT_DIMENSION)
            if fused_model_name
            else DEFAULT_DIMENSION
        )

        _faiss_manager = FAISSManager(
            dimension=DEFAULT_DIMENSION,
            index_path=INDEX_PATH,
            dimensions={
                "textual": textual_dim,
                "visual": visual_dim,
                "fused": fused_dim,
            },
        )
    return _faiss_manager


def get_textual_manager(model_name: str) -> TextModelManager:
    """Get or initialize a textual model manager."""
    if model_name not in _textual_managers:
        # Get model type from MODEL_REGISTRY, fallback to detection logic
        if model_name in MODEL_REGISTRY:
            model_type = MODEL_REGISTRY[model_name]["type"]
        elif "bge" in model_name.lower() or model_name.startswith("BAAI/"):
            model_type = "bge"
        elif "qwen" in model_name.lower() or model_name.startswith("Qwen/"):
            model_type = "qwen"
        else:
            model_type = "clip"

        _textual_managers[model_name] = TextModelManager(
            model_type=model_type,
            model_config={"model_name": model_name},
        )
    return _textual_managers[model_name]


def get_visual_manager(model_name: str) -> VisualModelManager:
    """Get or initialize a visual model manager."""
    if model_name not in _visual_managers:
        _visual_managers[model_name] = VisualModelManager(
            model_type="clip",
            model_config={"model_name": model_name},
        )
    return _visual_managers[model_name]


def get_fused_manager(model_name: str) -> FusedModelManager:
    """Get or initialize a fused model manager."""
    if model_name not in _fused_managers:
        _fused_managers[model_name] = FusedModelManager(
            model_type="clip",
            model_config={"model_name": model_name},
        )
    return _fused_managers[model_name]


def combine_product_text(
    name: str,
    description: str,
    brand: str,
    category: str,
    price: Any,
) -> str:
    """Combine product fields into a single text for embedding."""
    parts = []
    if name:
        parts.append(name)
    if description:
        parts.append(description)
    if brand:
        parts.append(brand)
    if category:
        parts.append(category)
    if price:
        parts.append(f"Price: {price}")
    return " ".join(parts)


@app.route("/api/retrieval/add-product", methods=["POST"])
def add_product():
    """
    Add a product to the retrieval system.

    Request JSON:
    {
        "id": "product_id",
        "name": "Product Name",
        "description": "Product description",
        "brand": "Brand Name",
        "category": "Category",
        "price": 99.99,
        "images": ["C:/path/to/image1.jpg", "C:/path/to/image2.jpg"],
        "textual_model_name": "ViT-B/32",
        "visual_model_name": "ViT-B/32",
        "fused_model_name": "ViT-B/32"  # Optional, not used for now
    }

    Response:
    {
        "status": "success" | "error",
        "message": "...",
        "details": {...}
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["id", "name", "textual_model_name", "visual_model_name"]
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

        product_id = data["id"]
        name = data.get("name", "")
        description = data.get("description", "")
        brand = data.get("brand", "")
        category = data.get("category", "")
        price = data.get("price", "")
        images = data.get("images", [])
        textual_model_name = data["textual_model_name"]
        visual_model_name = data["visual_model_name"]
        # fused_model_name = data.get("fused_model_name")  # Not used for now

        # Get managers
        faiss_manager = get_faiss_manager(
            textual_model_name=textual_model_name,
            visual_model_name=visual_model_name,
        )
        textual_manager = get_textual_manager(textual_model_name)
        visual_manager = get_visual_manager(visual_model_name)

        # Combine text fields and create textual embedding
        combined_text = combine_product_text(name, description, brand, category, price)
        textual_embedding = textual_manager.get_document_embedding(combined_text)

        # Add to textual index
        textual_vector_id = faiss_manager.add_to_textual(
            embedding=textual_embedding,
            product_id=product_id,
            model_name=textual_model_name,
        )

        # Process images and add to visual index
        visual_vector_ids = []
        for image_no, image_path in enumerate(images):
            try:
                visual_embedding = visual_manager.get_embedding(image_path)
                visual_vector_id = faiss_manager.add_to_visual(
                    embedding=visual_embedding,
                    product_id=product_id,
                    image_no=image_no,
                    model_name=visual_model_name,
                )
                visual_vector_ids.append(visual_vector_id)
            except FileNotFoundError as e:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Image not found: {image_path}",
                        }
                    ),
                    400,
                )
            except ValueError as e:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": str(e),
                        }
                    ),
                    400,
                )

        faiss_manager.save()
        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Product {product_id} added successfully",
                    "details": {
                        "product_id": product_id,
                        "textual_vector_id": textual_vector_id,
                        "visual_vector_ids": visual_vector_ids,
                        "images_processed": len(visual_vector_ids),
                    },
                }
            ),
            201,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        error_message = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
        return (
            jsonify(
                {
                    "status": "error",
                    "message": error_message,
                }
            ),
            500,
        )


@app.route("/api/retrieval/index-stats", methods=["GET"])
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


@app.route("/api/health", methods=["GET"])
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


@app.route("/api/retrieval/search/late", methods=["POST"])
def late_fusion_search():
    """
    Late fusion search combining textual and visual search results.

    Performs parallel searches on Textual and Visual indices,
    then combines results using weighted ranking.

    Request JSON:
    {
        "text": "search query text",
        "textual_model_name": "ViT-B/32",
        "text_weight": 0.5,
        "image": "C:/path/to/query/image.jpg",
        "visual_model_name": "ViT-B/32"
    }

    Response:
    {
        "status": "success",
        "results": [
            {
                "product_id": "prod_001",
                "combined_score": 0.85,
                "text_score": 0.9,
                "image_score": 0.8,
                "best_image_no": 0
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = [
            "text",
            "textual_model_name",
            "text_weight",
            "image",
            "visual_model_name",
        ]
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

        text = data["text"]
        textual_model_name = data["textual_model_name"]
        text_weight = float(data["text_weight"])
        image_path = data["image"]
        visual_model_name = data["visual_model_name"]
        top_k = validate_top_k(data)

        # Validate text_weight
        if not 0 <= text_weight <= 1:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "text_weight must be between 0 and 1",
                    }
                ),
                400,
            )

        image_weight = 1 - text_weight

        # Get managers
        faiss_manager = get_faiss_manager(
            textual_model_name=textual_model_name,
            visual_model_name=visual_model_name,
        )
        textual_manager = get_textual_manager(textual_model_name)
        visual_manager = get_visual_manager(visual_model_name)

        # Generate embeddings
        text_embedding = textual_manager.get_embedding(text)
        image_embedding = visual_manager.get_embedding(image_path)

        # Perform parallel searches using ThreadPoolExecutor
        textual_results = []
        visual_results = []

        def search_textual():
            return faiss_manager.search_textual(
                query_embedding=text_embedding,
                top_k=top_k * 5,  # Get more results for better fusion
                model_name=textual_model_name,
            )

        def search_visual():
            return faiss_manager.search_visual(
                query_embedding=image_embedding,
                top_k=top_k * 5,  # Get more results for better fusion
                model_name=visual_model_name,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_textual = executor.submit(search_textual)
            future_visual = executor.submit(search_visual)

            textual_results = future_textual.result()
            visual_results = future_visual.result()

        # Process textual results: one entry per product
        text_scores: Dict[str, float] = {}
        for result in textual_results:
            product_id = result["product_id"]
            score = result["score"]
            if product_id not in text_scores:
                text_scores[product_id] = score
            else:
                # Keep the highest score
                text_scores[product_id] = max(text_scores[product_id], score)

        # Process visual results: keep best score and image_no per product
        visual_scores: Dict[str, Dict[str, Any]] = {}
        for result in visual_results:
            product_id = result["product_id"]
            score = result["score"]
            image_no = result.get("image_no", 0)

            if product_id not in visual_scores:
                visual_scores[product_id] = {
                    "score": score,
                    "image_no": image_no,
                }
            else:
                # Keep the highest score and its corresponding image_no
                if score > visual_scores[product_id]["score"]:
                    visual_scores[product_id] = {
                        "score": score,
                        "image_no": image_no,
                    }

        # Combine results using late fusion
        # Get all unique product IDs from both indices
        all_product_ids = set(text_scores.keys()) | set(visual_scores.keys())

        combined_results = []
        for product_id in all_product_ids:
            t_score = text_scores.get(product_id, 0.0)
            v_data = visual_scores.get(product_id, {"score": 0.0, "image_no": -1})
            v_score = v_data["score"]
            best_image_no = v_data["image_no"]

            # Calculate combined score using weighted average
            combined_score = (text_weight * t_score) + (image_weight * v_score)

            combined_results.append(
                {
                    "product_id": product_id,
                    "combined_score": round(combined_score, 6),
                    "text_score": round(t_score, 6),
                    "image_score": round(v_score, 6),
                    "best_image_no": best_image_no,
                }
            )

        # Sort by combined score (descending)
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)

        # Return top_k results
        combined_results = combined_results[:top_k]

        return (
            jsonify(
                {
                    "status": "success",
                    "results": combined_results,
                    "meta": {
                        "text_weight": text_weight,
                        "image_weight": image_weight,
                        "total_results": len(combined_results),
                    },
                }
            ),
            200,
        )

    except FileNotFoundError as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Image not found: {str(e)}",
                }
            ),
            400,
        )
    except ValueError as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                }
            ),
            400,
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


@app.route("/api/retrieval/search/text", methods=["POST"])
def text_search():
    """
    Text-only search in the Textual index.

    Request JSON:
    {
        "text": "search query text",
        "textual_model_name": "ViT-B/32",
        "top_k": 10  // optional, default 10
    }

    Response:
    {
        "status": "success",
        "results": [
            {
                "product_id": "prod_001",
                "score": 0.85
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["text", "textual_model_name"]
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

        text = data["text"]
        textual_model_name = data["textual_model_name"]
        top_k = validate_top_k(data)

        # Get managers
        faiss_manager = get_faiss_manager(
            textual_model_name=textual_model_name,
        )
        textual_manager = get_textual_manager(textual_model_name)

        # Generate text embedding
        text_embedding = textual_manager.get_embedding(text)

        # Search textual index
        search_results = faiss_manager.search_textual(
            query_embedding=text_embedding,
            top_k=top_k * 2,  # Get more for deduplication
            model_name=textual_model_name,
        )

        # Deduplicate: keep only the best score per product
        product_scores: Dict[str, float] = {}
        for result in search_results:
            product_id = result["product_id"]
            score = result["score"]
            if product_id not in product_scores:
                product_scores[product_id] = score
            else:
                product_scores[product_id] = max(product_scores[product_id], score)

        # Build results list
        results = [
            {
                "product_id": product_id,
                "score": round(score, 6),
            }
            for product_id, score in product_scores.items()
        ]

        # Sort by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Return top_k results
        results = results[:top_k]

        return (
            jsonify(
                {
                    "status": "success",
                    "results": results,
                    "meta": {
                        "total_results": len(results),
                        "model_name": textual_model_name,
                    },
                }
            ),
            200,
        )

    except ValueError as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                }
            ),
            400,
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


@app.route("/api/retrieval/delete-product/<product_id>", methods=["DELETE"])
def delete_product(product_id: str):
    """
    Delete a product and all its embeddings from all FAISS indices.

    Path Parameters:
        product_id: The ID of the product to delete.

    Response:
    {
        "status": "success" | "error",
        "message": "...",
        "details": {
            "product_id": "prod_001",
            "removed_counts": {
                "textual": 1,
                "visual": 3,
                "fused": 3
            },
            "total_removed": 7
        }
    }
    """
    try:
        if not product_id:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Product ID is required",
                    }
                ),
                400,
            )

        # Get FAISS manager
        faiss_manager = get_faiss_manager()

        # Remove product from all indices
        removed_counts = faiss_manager.remove_product_from_all(product_id)
        total_removed = sum(removed_counts.values())

        if total_removed == 0:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Product {product_id} not found in any index",
                    }
                ),
                404,
            )

        # Save indices after deletion
        faiss_manager.save()

        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Product {product_id} deleted successfully",
                    "details": {
                        "product_id": product_id,
                        "removed_counts": removed_counts,
                        "total_removed": total_removed,
                    },
                }
            ),
            200,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        error_message = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
        return (
            jsonify(
                {
                    "status": "error",
                    "message": error_message,
                }
            ),
            500,
        )


if __name__ == "__main__":
    app.run(debug=True, host=HOST, port=PORT)
