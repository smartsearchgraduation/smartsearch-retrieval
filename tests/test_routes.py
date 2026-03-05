"""
Route / endpoint tests for search, product, and system blueprints.

All service-layer dependencies are mocked so no real models or FAISS
indices are needed.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Search routes
# ---------------------------------------------------------------------------

class TestTextSearch:

    def test_text_search_success(
        self, client, mock_faiss_manager, mock_textual_manager
    ):
        with patch("routes.search_routes.get_faiss_manager", return_value=mock_faiss_manager), \
             patch("routes.search_routes.get_textual_manager", return_value=mock_textual_manager):
            resp = client.post(
                "/api/retrieval/search/text",
                json={
                    "text": "red shoes",
                    "textual_model_name": "ViT-B/32",
                },
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert len(data["results"]) >= 1

    def test_text_search_missing_field(self, client):
        resp = client.post(
            "/api/retrieval/search/text",
            json={"textual_model_name": "ViT-B/32"},  # missing "text"
        )
        assert resp.status_code == 400
        assert "Missing required field" in resp.get_json()["message"]


class TestImageSearch:

    def test_image_search_success(
        self, client, mock_faiss_manager, mock_visual_manager
    ):
        with patch("routes.search_routes.get_faiss_manager", return_value=mock_faiss_manager), \
             patch("routes.search_routes.get_visual_manager", return_value=mock_visual_manager):
            resp = client.post(
                "/api/retrieval/search/image",
                json={
                    "image": "C:/fake/image.jpg",
                    "visual_model_name": "ViT-B/32",
                },
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"

    def test_image_search_missing_field(self, client):
        resp = client.post(
            "/api/retrieval/search/image",
            json={"visual_model_name": "ViT-B/32"},  # missing "image"
        )
        assert resp.status_code == 400
        assert "Missing required field" in resp.get_json()["message"]


class TestLateFusionSearch:

    def test_late_fusion_search_success(
        self, client, mock_faiss_manager, mock_textual_manager, mock_visual_manager
    ):
        with patch("routes.search_routes.get_faiss_manager", return_value=mock_faiss_manager), \
             patch("routes.search_routes.get_textual_manager", return_value=mock_textual_manager), \
             patch("routes.search_routes.get_visual_manager", return_value=mock_visual_manager):
            resp = client.post(
                "/api/retrieval/search/late",
                json={
                    "text": "red shoes",
                    "textual_model_name": "ViT-B/32",
                    "text_weight": 0.5,
                    "image": "C:/fake/image.jpg",
                    "visual_model_name": "ViT-B/32",
                },
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "results" in data

    def test_late_fusion_invalid_weight(
        self, client, mock_faiss_manager, mock_textual_manager, mock_visual_manager
    ):
        with patch("routes.search_routes.get_faiss_manager", return_value=mock_faiss_manager), \
             patch("routes.search_routes.get_textual_manager", return_value=mock_textual_manager), \
             patch("routes.search_routes.get_visual_manager", return_value=mock_visual_manager):
            resp = client.post(
                "/api/retrieval/search/late",
                json={
                    "text": "red shoes",
                    "textual_model_name": "ViT-B/32",
                    "text_weight": 1.5,  # invalid
                    "image": "C:/fake/image.jpg",
                    "visual_model_name": "ViT-B/32",
                },
            )
        assert resp.status_code == 400
        assert "text_weight" in resp.get_json()["message"]


# ---------------------------------------------------------------------------
# Product routes
# ---------------------------------------------------------------------------

class TestAddProduct:

    def test_add_product_success(
        self, client, mock_faiss_manager, mock_textual_manager, mock_visual_manager
    ):
        with patch("routes.product_routes.get_faiss_manager", return_value=mock_faiss_manager), \
             patch("routes.product_routes.get_textual_manager", return_value=mock_textual_manager), \
             patch("routes.product_routes.get_visual_manager", return_value=mock_visual_manager), \
             patch("routes.product_routes.combine_product_text", return_value="Test Product"):
            resp = client.post(
                "/api/retrieval/add-product",
                json={
                    "id": "prod_001",
                    "name": "Test Product",
                    "textual_model_name": "ViT-B/32",
                    "visual_model_name": "ViT-B/32",
                    "images": [],
                },
            )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["details"]["product_id"] == "prod_001"

    def test_add_product_missing_field(self, client):
        resp = client.post(
            "/api/retrieval/add-product",
            json={"name": "Test"},  # missing "id", model names
        )
        assert resp.status_code == 400


class TestDeleteProduct:

    def test_delete_product_success(self, client, mock_faiss_manager):
        with patch("routes.product_routes.get_faiss_manager", return_value=mock_faiss_manager):
            resp = client.delete("/api/retrieval/delete-product/prod_001")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"

    def test_delete_product_not_found(self, client, mock_faiss_manager):
        mock_faiss_manager.remove_product_from_all.return_value = {
            "textual": 0, "visual": 0, "fused": 0,
        }
        with patch("routes.product_routes.get_faiss_manager", return_value=mock_faiss_manager):
            resp = client.delete("/api/retrieval/delete-product/nonexistent")
        assert resp.status_code == 404


class TestUpdateProduct:

    def test_update_product_success(
        self, client, mock_faiss_manager, mock_textual_manager, mock_visual_manager
    ):
        with patch("routes.product_routes.get_faiss_manager", return_value=mock_faiss_manager), \
             patch("routes.product_routes.get_textual_manager", return_value=mock_textual_manager), \
             patch("routes.product_routes.get_visual_manager", return_value=mock_visual_manager), \
             patch("routes.product_routes.combine_product_text", return_value="Updated"):
            resp = client.put(
                "/api/retrieval/update-product/prod_001",
                json={
                    "name": "Updated Product",
                    "textual_model_name": "ViT-B/32",
                    "visual_model_name": "ViT-B/32",
                    "images": [],
                },
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"


# ---------------------------------------------------------------------------
# System routes
# ---------------------------------------------------------------------------

class TestSystemRoutes:

    def test_health_check(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"

    def test_index_stats(self, client, mock_faiss_manager):
        with patch("routes.system_routes.get_faiss_manager", return_value=mock_faiss_manager):
            resp = client.get("/api/retrieval/index-stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "indices" in data
        assert data["indices"]["textual"] == 10
