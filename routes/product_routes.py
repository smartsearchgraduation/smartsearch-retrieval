"""
Product Routes — add-product, update-product, delete-product endpoints.
"""

import traceback

from flask import Blueprint, request, jsonify

from services.manager_service import (
    get_faiss_manager,
    get_fused_manager,
    get_textual_manager,
    get_visual_manager,
    combine_product_text,
    remove_product_from_all_models,
)
from vector_db.faiss_manager import IndexType
from utils.validation import (
    validate_image_file_size,
    validate_required_fields,
    validate_text_length,
)


product_bp = Blueprint("product", __name__)


@product_bp.route("/api/retrieval/add-product", methods=["POST"])
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
        "fused_model_name": "ViT-B/32"  // Optional, enables early fusion indexing (CLIP only)
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
        error = validate_required_fields(
            data, ["id", "name", "textual_model_name", "visual_model_name"]
        )
        if error:
            return error

        product_id = data["id"]
        name = data.get("name", "")
        description = data.get("description", "")
        brand = data.get("brand", "")
        category = data.get("category", "")
        price = data.get("price", "")
        images = data.get("images", [])
        textual_model_name = data["textual_model_name"]
        visual_model_name = data["visual_model_name"]
        fused_model_name = data.get("fused_model_name")

        # Get separate FAISS managers per model
        textual_faiss = get_faiss_manager(textual_model_name)
        visual_faiss = get_faiss_manager(visual_model_name)

        # Check if product already has embeddings for the active model
        if textual_faiss.has_product(IndexType.TEXTUAL, product_id):
            return (
                jsonify(
                    {
                        "status": "success",
                        "message": f"Product {product_id} already has embeddings for this model, skipping",
                        "details": {
                            "product_id": product_id,
                            "skipped": True,
                        },
                    }
                ),
                200,
            )

        textual_manager = get_textual_manager(textual_model_name)
        visual_manager = get_visual_manager(visual_model_name)

        # Combine text fields and create textual embedding
        combined_text = combine_product_text(name, description, brand, category, price)
        validate_text_length(combined_text)
        textual_embedding = textual_manager.get_document_embedding(combined_text)

        # Add to textual index
        textual_vector_id = textual_faiss.add_to_textual(
            embedding=textual_embedding,
            product_id=product_id,
            model_name=textual_model_name,
        )

        # Process images and add to visual index
        visual_vector_ids = []
        for image_no, image_path in enumerate(images):
            try:
                validate_image_file_size(image_path)
                visual_embedding = visual_manager.get_embedding(image_path)
                visual_vector_id = visual_faiss.add_to_visual(
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

        # Process fused embeddings (early fusion indexing) if fused_model_name provided
        fused_vector_ids = []
        if fused_model_name and images:
            fused_faiss = get_faiss_manager(fused_model_name)
            fused_mgr = get_fused_manager(fused_model_name)

            for image_no, image_path in enumerate(images):
                try:
                    fused_embedding = fused_mgr.get_embedding(combined_text, image_path)
                    fused_vector_id = fused_faiss.add_to_fused(
                        embedding=fused_embedding,
                        product_id=product_id,
                        image_no=image_no,
                        model_name=fused_model_name,
                    )
                    fused_vector_ids.append(fused_vector_id)
                except FileNotFoundError:
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
            fused_faiss.save()

        textual_faiss.save()
        if visual_faiss is not textual_faiss:
            visual_faiss.save()

        details = {
            "product_id": product_id,
            "textual_vector_id": textual_vector_id,
            "visual_vector_ids": visual_vector_ids,
            "images_processed": len(visual_vector_ids),
        }
        if fused_vector_ids:
            details["fused_vector_ids"] = fused_vector_ids

        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Product {product_id} added successfully",
                    "details": details,
                }
            ),
            201,
        )

    except Exception as e:
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


@product_bp.route("/api/retrieval/delete-product/<product_id>", methods=["DELETE"])
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

        # Remove product from ALL model folders
        all_removed = remove_product_from_all_models(product_id)

        total_removed = sum(
            sum(counts.values()) for counts in all_removed.values()
        )

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

        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Product {product_id} deleted successfully",
                    "details": {
                        "product_id": product_id,
                        "removed_counts": all_removed,
                        "total_removed": total_removed,
                    },
                }
            ),
            200,
        )

    except Exception as e:
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


@product_bp.route("/api/retrieval/update-product/<product_id>", methods=["PUT"])
def update_product(product_id: str):
    """
    Update a product's embeddings in the retrieval system.

    Atomically removes old embeddings and re-indexes with new data.

    Path Parameters:
        product_id: The ID of the product to update.

    Request JSON:
    {
        "name": "Updated Product Name",
        "description": "Updated description",
        "brand": "Brand Name",
        "category": "Category",
        "price": 149.99,
        "images": ["C:/path/to/new_image.jpg"],
        "textual_model_name": "ViT-B/32",
        "visual_model_name": "ViT-B/32",
        "fused_model_name": "ViT-B/32"  // Optional, enables early fusion indexing (CLIP only)
    }

    Response:
    {
        "status": "success" | "error",
        "message": "...",
        "details": {
            "product_id": "prod_001",
            "removed_counts": {...},
            "textual_vector_id": 5,
            "visual_vector_ids": [6, 7],
            "images_processed": 2
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

        data = request.get_json()

        # Validate required fields
        error = validate_required_fields(
            data, ["name", "textual_model_name", "visual_model_name"]
        )
        if error:
            return error

        name = data.get("name", "")
        description = data.get("description", "")
        brand = data.get("brand", "")
        category = data.get("category", "")
        price = data.get("price", "")
        images = data.get("images", [])
        textual_model_name = data["textual_model_name"]
        visual_model_name = data["visual_model_name"]
        fused_model_name = data.get("fused_model_name")

        # Step 1: Remove old embeddings from ALL model folders
        all_removed = remove_product_from_all_models(product_id)

        # Get separate FAISS managers per model
        textual_faiss = get_faiss_manager(textual_model_name)
        visual_faiss = get_faiss_manager(visual_model_name)
        textual_manager = get_textual_manager(textual_model_name)
        visual_manager = get_visual_manager(visual_model_name)

        # Step 2: Generate and add new textual embedding
        combined_text = combine_product_text(name, description, brand, category, price)
        validate_text_length(combined_text)
        textual_embedding = textual_manager.get_document_embedding(combined_text)

        textual_vector_id = textual_faiss.add_to_textual(
            embedding=textual_embedding,
            product_id=product_id,
            model_name=textual_model_name,
        )

        # Step 3: Generate and add new visual embeddings
        visual_vector_ids = []
        for image_no, image_path in enumerate(images):
            try:
                validate_image_file_size(image_path)
                visual_embedding = visual_manager.get_embedding(image_path)
                visual_vector_id = visual_faiss.add_to_visual(
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

        # Step 4: Generate and add fused embeddings if fused_model_name provided
        fused_vector_ids = []
        if fused_model_name and images:
            fused_faiss = get_faiss_manager(fused_model_name)
            fused_mgr = get_fused_manager(fused_model_name)

            for image_no, image_path in enumerate(images):
                try:
                    fused_embedding = fused_mgr.get_embedding(combined_text, image_path)
                    fused_vector_id = fused_faiss.add_to_fused(
                        embedding=fused_embedding,
                        product_id=product_id,
                        image_no=image_no,
                        model_name=fused_model_name,
                    )
                    fused_vector_ids.append(fused_vector_id)
                except FileNotFoundError:
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
            fused_faiss.save()

        # Step 5: Save indices
        textual_faiss.save()
        if visual_faiss is not textual_faiss:
            visual_faiss.save()

        details = {
            "product_id": product_id,
            "removed_counts": all_removed,
            "textual_vector_id": textual_vector_id,
            "visual_vector_ids": visual_vector_ids,
            "images_processed": len(visual_vector_ids),
        }
        if fused_vector_ids:
            details["fused_vector_ids"] = fused_vector_ids

        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Product {product_id} updated successfully",
                    "details": details,
                }
            ),
            200,
        )

    except Exception as e:
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
