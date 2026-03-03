"""
Search Routes — text search, image search, late fusion search endpoints.
"""

from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, request, jsonify

from services.manager_service import (
    get_faiss_manager,
    get_textual_manager,
    get_visual_manager,
)
from utils.validation import validate_required_fields, validate_top_k


search_bp = Blueprint("search", __name__)


@search_bp.route("/api/retrieval/search/late", methods=["POST"])
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
        error = validate_required_fields(
            data,
            ["text", "textual_model_name", "text_weight", "image", "visual_model_name"],
        )
        if error:
            return error

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


@search_bp.route("/api/retrieval/search/text", methods=["POST"])
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
        error = validate_required_fields(data, ["text", "textual_model_name"])
        if error:
            return error

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


@search_bp.route("/api/retrieval/search/image", methods=["POST"])
def image_search():
    """
    Image-only search in the Visual index.

    Request JSON:
    {
        "image": "C:/path/to/query/image.jpg",
        "visual_model_name": "ViT-B/32",
        "top_k": 10  // optional, default 10
    }

    Response:
    {
        "status": "success",
        "results": [
            {
                "product_id": "prod_001",
                "score": 0.85,
                "best_image_no": 0
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        error = validate_required_fields(data, ["image", "visual_model_name"])
        if error:
            return error

        image_path = data["image"]
        visual_model_name = data["visual_model_name"]
        top_k = validate_top_k(data)

        # Get managers
        faiss_manager = get_faiss_manager(
            visual_model_name=visual_model_name,
        )
        visual_manager = get_visual_manager(visual_model_name)

        # Generate image embedding
        image_embedding = visual_manager.get_embedding(image_path)

        # Search visual index
        search_results = faiss_manager.search_visual(
            query_embedding=image_embedding,
            top_k=top_k * 2,  # Get more for deduplication
            model_name=visual_model_name,
        )

        # Deduplicate: keep best score and image_no per product
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

        # Build results list
        results = [
            {
                "product_id": product_id,
                "score": round(scores["score"], 6),
                "best_image_no": scores["image_no"],
            }
            for product_id, scores in product_scores.items()
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
                        "model_name": visual_model_name,
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
