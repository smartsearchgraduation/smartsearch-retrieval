"""
Search Routes — text search, image search, late fusion search, early fusion search,
               cross-modal search (image-by-text, text-by-image) endpoints.
"""

from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, request, jsonify

from services.manager_service import (
    get_faiss_manager,
    get_fused_manager,
    get_textual_manager,
    get_visual_manager,
)
from utils.validation import (
    deduplicate_fused_results,
    deduplicate_text_results,
    deduplicate_visual_results,
    validate_clip_model,
    validate_image_file_size,
    validate_required_fields,
    validate_text_length,
    validate_top_k,
)


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

        validate_text_length(text)
        validate_image_file_size(image_path)

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

        # Get separate FAISS managers per model
        textual_faiss = get_faiss_manager(textual_model_name)
        visual_faiss = get_faiss_manager(visual_model_name)
        textual_manager = get_textual_manager(textual_model_name)
        visual_manager = get_visual_manager(visual_model_name)

        # Generate embeddings
        text_embedding = textual_manager.get_embedding(text)
        image_embedding = visual_manager.get_embedding(image_path)

        # Perform parallel searches using ThreadPoolExecutor
        textual_results = []
        visual_results = []

        def search_textual():
            return textual_faiss.search_textual(
                query_embedding=text_embedding,
                top_k=top_k * 5,  # Get more results for better fusion
                model_name=textual_model_name,
            )

        def search_visual():
            return visual_faiss.search_visual(
                query_embedding=image_embedding,
                top_k=top_k * 5,  # Get more results for better fusion
                model_name=visual_model_name,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_textual = executor.submit(search_textual)
            future_visual = executor.submit(search_visual)

            textual_results = future_textual.result()
            visual_results = future_visual.result()

        # Keep one best result per product for each modality
        text_scores = deduplicate_text_results(textual_results)
        visual_scores = deduplicate_visual_results(visual_results)

        # Combine results using late fusion
        # Only combine products that appear in BOTH modality results
        # to avoid unfairly penalizing strong single-modality matches with a 0.0 score
        all_product_ids = set(text_scores.keys()) & set(visual_scores.keys())

        combined_results = []
        for product_id in all_product_ids:
            t_score = text_scores[product_id]
            v_data = visual_scores[product_id]
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

        validate_text_length(text)

        # Get managers
        faiss_manager = get_faiss_manager(textual_model_name)
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
        product_scores = deduplicate_text_results(search_results)

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

        validate_image_file_size(image_path)

        # Get managers
        faiss_manager = get_faiss_manager(visual_model_name)
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
        product_scores = deduplicate_visual_results(search_results)

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


@search_bp.route("/api/retrieval/search/early", methods=["POST"])
def early_fusion_search():
    """
    Early fusion search using a fused text+image query embedding.

    Fuses the text and image query into a single embedding using CLIP's
    shared embedding space (weighted average fusion), then searches the
    Fused index.

    Requires that products were indexed with fused embeddings
    (fused_model_name provided during add-product).

    Request JSON:
    {
        "text": "search query text",
        "image": "C:/path/to/query/image.jpg",
        "fused_model_name": "ViT-B/32",
        "text_weight": 0.5,
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
        error = validate_required_fields(
            data, ["text", "image", "fused_model_name"]
        )
        if error:
            return error

        text = data["text"]
        image_path = data["image"]
        fused_model_name = data["fused_model_name"]
        text_weight = float(data.get("text_weight", 0.5))
        top_k = validate_top_k(data)

        validate_text_length(text)
        validate_image_file_size(image_path)

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

        # Get managers
        faiss_manager = get_faiss_manager(fused_model_name)
        fused_manager = get_fused_manager(fused_model_name)

        # Set fusion weights (average fusion with configurable text_weight)
        fused_manager.set_fusion_method("weighted", text_weight=text_weight)

        # Generate fused query embedding
        fused_embedding = fused_manager.get_embedding(text, image_path)

        # Search fused index
        search_results = faiss_manager.search_fused(
            query_embedding=fused_embedding,
            top_k=top_k * 2,  # Get more for deduplication
            model_name=fused_model_name,
        )

        # Deduplicate: keep best score and image_no per product
        product_scores = deduplicate_fused_results(search_results)

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
                        "model_name": fused_model_name,
                        "text_weight": text_weight,
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


@search_bp.route("/api/retrieval/search/image-by-text", methods=["POST"])
def image_by_text_search():
    """
    Cross-modal search: find products using a text query against the fused index.

    Encodes the text query with CLIP's text encoder, then searches the
    Fused FAISS index. Works because CLIP text embeddings and fused
    embeddings (text+image blend) share the same vector space.

    Only CLIP models are supported (fused index is built with CLIP).

    Request JSON:
    {
        "text": "search query text",
        "fused_model_name": "ViT-B/32",
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
        error = validate_required_fields(data, ["text", "fused_model_name"])
        if error:
            return error

        text = data["text"]
        fused_model_name = data["fused_model_name"]
        top_k = validate_top_k(data)

        validate_text_length(text)
        validate_clip_model(fused_model_name)

        # Encode text with the same CLIP model used for the fused index
        textual_manager = get_textual_manager(fused_model_name)
        text_embedding = textual_manager.get_embedding(text)

        # Search the fused index with the text embedding (cross-modal)
        faiss_manager = get_faiss_manager(fused_model_name)
        search_results = faiss_manager.search_fused(
            query_embedding=text_embedding,
            top_k=top_k * 2,  # Get more for deduplication
            model_name=fused_model_name,
        )

        # Deduplicate: keep best score and image_no per product
        product_scores = deduplicate_fused_results(search_results)

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
                        "model_name": fused_model_name,
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


@search_bp.route("/api/retrieval/search/text-by-image", methods=["POST"])
def text_by_image_search():
    """
    Cross-modal search: find products using an image query against the fused index.

    Encodes the image with CLIP's image encoder, then searches the
    Fused FAISS index. Works because CLIP image embeddings and fused
    embeddings (text+image blend) share the same vector space.

    Only CLIP models are supported (fused index is built with CLIP).

    Request JSON:
    {
        "image": "C:/path/to/query/image.jpg",
        "fused_model_name": "ViT-B/32",
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
        error = validate_required_fields(data, ["image", "fused_model_name"])
        if error:
            return error

        image_path = data["image"]
        fused_model_name = data["fused_model_name"]
        top_k = validate_top_k(data)

        validate_image_file_size(image_path)
        validate_clip_model(fused_model_name)

        # Encode image with the same CLIP model used for the fused index
        visual_manager = get_visual_manager(fused_model_name)
        image_embedding = visual_manager.get_embedding(image_path)

        # Search the fused index with the image embedding (cross-modal)
        faiss_manager = get_faiss_manager(fused_model_name)
        search_results = faiss_manager.search_fused(
            query_embedding=image_embedding,
            top_k=top_k * 2,  # Get more for deduplication
            model_name=fused_model_name,
        )

        # Deduplicate: keep best score and image_no per product
        product_scores = deduplicate_fused_results(search_results)

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
                        "model_name": fused_model_name,
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
