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


def get_faiss_manager() -> FAISSManager:
    """Get or initialize the FAISS manager."""
    global _faiss_manager
    if _faiss_manager is None:
        _faiss_manager = FAISSManager(
            dimension=512,
            index_path="./data/faiss_indices",
        )
    return _faiss_manager


def get_textual_manager(model_name: str) -> TextModelManager:
    """Get or initialize a textual model manager."""
    if model_name not in _textual_managers:
        _textual_managers[model_name] = TextModelManager(
            model_type="clip",
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
        faiss_manager = get_faiss_manager()
        textual_manager = get_textual_manager(textual_model_name)
        visual_manager = get_visual_manager(visual_model_name)

        # Combine text fields and create textual embedding
        combined_text = combine_product_text(name, description, brand, category, price)
        textual_embedding = textual_manager.get_embedding(combined_text)

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
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
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
        top_k = data.get("top_k", 10)

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
        faiss_manager = get_faiss_manager()
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
        top_k = data.get("top_k", 10)

        # Get managers
        faiss_manager = get_faiss_manager()
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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
