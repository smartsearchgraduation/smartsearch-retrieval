"""Route-level unit tests (RU-RT-NN).

These tests exercise the Flask handlers via the test client. The
manager_service / model-pool boundary is stubbed via the ``stub_managers``
fixture; FAISS itself is real, in-memory, rooted at ``tmp_index_dir``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest


# ---------- Helpers ----------


def _add_product(client, payload: Dict[str, Any]):
    return client.post(
        "/api/retrieval/add-product",
        data=json.dumps(payload),
        content_type="application/json",
    )


def _post_json(client, url: str, payload: Dict[str, Any]):
    return client.post(url, data=json.dumps(payload), content_type="application/json")


def _put_json(client, url: str, payload: Dict[str, Any]):
    return client.put(url, data=json.dumps(payload), content_type="application/json")


# ---------- System routes ----------


def test_ru_rt_01_health_returns_healthy(flask_client):
    """RU-RT-01: GET /api/health returns healthy status."""
    resp = flask_client.get("/api/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "healthy"
    assert body["service"] == "E-Commerce Product Retrieval System"


def test_ru_rt_02_models_happy_path(flask_client, monkeypatch):
    """RU-RT-02: GET /api/retrieval/models happy path returns stubbed data."""
    fake = {
        "textual_models": [{"name": "ViT-B/32", "dimension": 512}],
        "visual_models": [],
        "defaults": {"textual": "ViT-B/32", "visual": "ViT-B/32"},
    }
    import routes.system_routes as sysr

    monkeypatch.setattr(sysr, "get_available_models", lambda: fake)
    resp = flask_client.get("/api/retrieval/models")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["data"] == fake


def test_ru_rt_03_models_error_path(flask_client, monkeypatch):
    """RU-RT-03: GET /api/retrieval/models returns 500 if backend raises."""
    import routes.system_routes as sysr

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(sysr, "get_available_models", boom)
    resp = flask_client.get("/api/retrieval/models")
    assert resp.status_code == 500
    body = resp.get_json()
    assert body["status"] == "error"
    assert "boom" in body["message"]


def test_ru_rt_04_index_stats_happy_path(flask_client, monkeypatch):
    """RU-RT-04: GET /api/retrieval/index-stats happy path."""
    fake = {"ViT-B-32_512_embeddings": {"textual": 3, "visual": 2, "fused": 0}}
    import routes.system_routes as sysr

    monkeypatch.setattr(sysr, "get_all_index_stats", lambda: fake)
    resp = flask_client.get("/api/retrieval/index-stats")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["indices"] == fake


def test_ru_rt_05_index_stats_error_path(flask_client, monkeypatch):
    """RU-RT-05: GET /api/retrieval/index-stats returns 500 on exception."""
    import routes.system_routes as sysr

    def boom():
        raise Exception("kapow")

    monkeypatch.setattr(sysr, "get_all_index_stats", boom)
    resp = flask_client.get("/api/retrieval/index-stats")
    assert resp.status_code == 500
    body = resp.get_json()
    assert body["status"] == "error"
    assert "kapow" in body["message"]


# ---------- Add product ----------


def test_ru_rt_06_add_product_happy_path(flask_client, stub_managers):
    """RU-RT-06: POST /add-product happy path with no images."""
    resp = _add_product(
        flask_client,
        {
            "id": "p1",
            "name": "Shoe",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
        },
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["status"] == "success"
    details = body["details"]
    assert details["product_id"] == "p1"
    assert isinstance(details["textual_vector_id"], int)
    assert details["visual_vector_ids"] == []
    assert details["images_processed"] == 0


def test_ru_rt_07_add_product_missing_required_field(flask_client):
    """RU-RT-07: POST /add-product missing textual_model_name returns 400."""
    resp = _add_product(flask_client, {"id": "p2", "name": "X"})
    assert resp.status_code == 400
    body = resp.get_json()
    assert "Missing required field: textual_model_name" in body["message"]


def test_ru_rt_08_add_product_missing_json_body(flask_client):
    """RU-RT-08: POST /add-product with empty body returns non-2xx error."""
    resp = flask_client.post("/api/retrieval/add-product", data="")
    assert resp.status_code >= 400
    # Body may or may not be JSON depending on Flask's response for missing CT
    body = resp.get_json(silent=True)
    if body is not None:
        assert body.get("status") == "error"


def test_ru_rt_09_add_product_image_not_found(flask_client, stub_managers, monkeypatch):
    """RU-RT-09: POST /add-product with image not found returns 400."""
    from services import manager_service

    orig_visual_cls = manager_service.VisualModelManager

    class RaisingVisual(orig_visual_cls):
        def get_embedding(self, image_path):
            raise FileNotFoundError(image_path)

    monkeypatch.setattr(manager_service, "VisualModelManager", RaisingVisual)

    resp = _add_product(
        flask_client,
        {
            "id": "p9",
            "name": "ShoeX",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "images": ["C:/no/such/file.jpg"],
        },
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["message"].startswith("Image not found:")


def test_ru_rt_10_add_product_oversize_image(flask_client, stub_managers, oversize_image_path):
    """RU-RT-10: POST /add-product with oversize image returns 400."""
    resp = _add_product(
        flask_client,
        {
            "id": "p10",
            "name": "Big",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "images": [oversize_image_path],
        },
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert "exceeds maximum limit of 50MB" in body["message"]


def test_ru_rt_11_add_product_duplicate_skip(flask_client, stub_managers):
    """RU-RT-11: POST /add-product duplicate skip path returns 200 skipped."""
    payload = {
        "id": "p3",
        "name": "Shoe3",
        "textual_model_name": "ViT-B/32",
        "visual_model_name": "ViT-B/32",
    }
    r1 = _add_product(flask_client, payload)
    assert r1.status_code == 201
    # Re-add same id without images: textual already indexed.
    # Visual was added with empty images list, so visual NOT yet indexed.
    # To trigger pure skip path, we must seed both. Add again with no images;
    # since visual_already_indexed will be False (no visual added), the
    # skip-all condition isn't met. So we instead add with an image first
    # then re-add.
    # Simpler: call has_product manually after first add for visual=False.
    # Re-add will go through: not skipped. To match spec RU-RT-11 exactly,
    # we instead add textual+visual (image) then re-add.
    # CLARIFY: spec says "Pre-seed real FAISSManager (via add) so product 'p3'
    # already in textual+visual." Without a real image, visual_vector_ids is
    # empty after the first add. We accept either: skipped True OR a fresh
    # success. Implementation: add with no images twice — second call should
    # still detect textual_already_indexed but visual_already_indexed=False,
    # so it proceeds. Therefore we directly re-add and assert skipped behavior
    # only when both indices contain p3.
    # Use tmp_image fixture path via flask_client's fixture? Not available
    # here. Use a placeholder file.
    import tempfile, os as _os
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"\xff\xd8\xff\xe0" + b"0" * 64)
        img = f.name
    try:
        seed_payload = {**payload, "images": [img]}
        flask_client.post(
            "/api/retrieval/add-product",
            data=json.dumps(seed_payload),
            content_type="application/json",
        )
        # Second call — same id, no images => fused not enabled, both
        # textual & visual already indexed => skip
        r2 = flask_client.post(
            "/api/retrieval/add-product",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert r2.status_code == 200
        body = r2.get_json()
        assert body["status"] == "success"
        assert "already has embeddings" in body["message"]
        assert body["details"]["skipped"] is True
    finally:
        _os.unlink(img)


def test_ru_rt_12_add_product_fused_enabled(flask_client, stub_managers, tmp_image_path):
    """RU-RT-12: POST /add-product with fused_model_name + 1 image."""
    resp = _add_product(
        flask_client,
        {
            "id": "p4",
            "name": "Shoe4",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "fused_model_name": "ViT-B/32",
            "images": [tmp_image_path],
        },
    )
    assert resp.status_code == 201
    body = resp.get_json()
    details = body["details"]
    assert "fused_vector_ids" in details
    assert len(details["fused_vector_ids"]) == 1
    assert details["fused_skipped"] is False


# ---------- Delete ----------


def test_ru_rt_13_delete_product_happy_path(flask_client, stub_managers):
    """RU-RT-13: DELETE /delete-product/<id> happy path."""
    _add_product(
        flask_client,
        {
            "id": "p5",
            "name": "Shoe5",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
        },
    )
    resp = flask_client.delete("/api/retrieval/delete-product/p5")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["details"]["total_removed"] > 0
    assert body["details"]["removed_counts"]


def test_ru_rt_14_delete_product_unknown_returns_404(flask_client, stub_managers):
    """RU-RT-14: DELETE /delete-product/<unknown> returns 404."""
    resp = flask_client.delete("/api/retrieval/delete-product/ghost")
    assert resp.status_code == 404
    body = resp.get_json()
    assert "not found in any index" in body["message"]


# ---------- Update ----------


def test_ru_rt_15_update_product_happy_path(flask_client, stub_managers):
    """RU-RT-15: PUT /update-product/<id> happy path."""
    _add_product(
        flask_client,
        {
            "id": "p6",
            "name": "Old",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
        },
    )
    resp = _put_json(
        flask_client,
        "/api/retrieval/update-product/p6",
        {
            "name": "newname",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    details = body["details"]
    assert "removed_counts" in details
    assert isinstance(details["textual_vector_id"], int)


def test_ru_rt_16_update_product_missing_required_field(flask_client, stub_managers):
    """RU-RT-16: PUT /update-product missing textual_model_name returns 400."""
    # Need any product id in URL — backend validates body before lookup.
    resp = _put_json(
        flask_client, "/api/retrieval/update-product/anything", {"name": "x"}
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert "Missing required field: textual_model_name" in body["message"]


def test_ru_rt_17_update_product_image_not_found(flask_client, stub_managers, monkeypatch):
    """RU-RT-17: PUT /update-product with missing image file returns 400."""
    from services import manager_service

    orig_visual_cls = manager_service.VisualModelManager

    class RaisingVisual(orig_visual_cls):
        def get_embedding(self, image_path):
            raise FileNotFoundError(image_path)

    monkeypatch.setattr(manager_service, "VisualModelManager", RaisingVisual)

    resp = _put_json(
        flask_client,
        "/api/retrieval/update-product/p17",
        {
            "name": "Shoe",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "images": ["nope.jpg"],
        },
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["message"].startswith("Image not found:")


# ---------- Text search ----------


def test_ru_rt_18_search_text_happy_path(flask_client, stub_managers):
    """RU-RT-18: POST /search/text happy path."""
    _add_product(
        flask_client,
        {
            "id": "p1",
            "name": "shoe",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
        },
    )
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/text",
        {"text": "shoe", "textual_model_name": "ViT-B/32"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert isinstance(body["results"], list)
    assert body["meta"]["total_results"] >= 0


def test_ru_rt_19_search_text_missing_text(flask_client):
    """RU-RT-19: POST /search/text missing text returns 400."""
    resp = _post_json(
        flask_client, "/api/retrieval/search/text", {"textual_model_name": "ViT-B/32"}
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert "Missing required field: text" in body["message"]


def test_ru_rt_20_search_text_top_k_zero(flask_client, stub_managers):
    """RU-RT-20: POST /search/text top_k=0 returns 400."""
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/text",
        {"text": "x", "textual_model_name": "ViT-B/32", "top_k": 0},
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert "top_k must be >= 1" in body["message"]


def test_ru_rt_21_search_text_top_k_non_integer(flask_client, stub_managers):
    """RU-RT-21: POST /search/text top_k='abc' returns 400."""
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/text",
        {"text": "x", "textual_model_name": "ViT-B/32", "top_k": "abc"},
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert "top_k must be a valid integer" in body["message"]


def test_ru_rt_22_search_text_top_k_default(flask_client, stub_managers):
    """RU-RT-22: POST /search/text without top_k uses DEFAULT_TOP_K * 2 = 20."""
    from services import manager_service

    # Force the manager to be created so we can wrap its method.
    flask_client.post(
        "/api/retrieval/add-product",
        data=json.dumps(
            {
                "id": "p22",
                "name": "x",
                "textual_model_name": "ViT-B/32",
                "visual_model_name": "ViT-B/32",
            }
        ),
        content_type="application/json",
    )

    mgr = manager_service.get_faiss_manager("ViT-B/32")
    observed: List[int] = []
    orig = mgr.search_textual

    def spy(query_embedding, top_k=10, model_name=None):
        observed.append(top_k)
        return orig(query_embedding=query_embedding, top_k=top_k, model_name=model_name)

    mgr.search_textual = spy  # type: ignore[assignment]

    resp = _post_json(
        flask_client,
        "/api/retrieval/search/text",
        {"text": "any", "textual_model_name": "ViT-B/32"},
    )
    assert resp.status_code == 200
    assert observed and observed[0] == manager_service.DEFAULT_TOP_K * 2 == 20


def test_ru_rt_23_search_text_top_k_clamped(flask_client, stub_managers):
    """RU-RT-23: POST /search/text top_k=10000 clamps to MAX_TOP_K * 2 = 200."""
    from services import manager_service

    flask_client.post(
        "/api/retrieval/add-product",
        data=json.dumps(
            {
                "id": "p23",
                "name": "x",
                "textual_model_name": "ViT-B/32",
                "visual_model_name": "ViT-B/32",
            }
        ),
        content_type="application/json",
    )

    mgr = manager_service.get_faiss_manager("ViT-B/32")
    observed: List[int] = []
    orig = mgr.search_textual

    def spy(query_embedding, top_k=10, model_name=None):
        observed.append(top_k)
        return orig(query_embedding=query_embedding, top_k=top_k, model_name=model_name)

    mgr.search_textual = spy  # type: ignore[assignment]

    resp = _post_json(
        flask_client,
        "/api/retrieval/search/text",
        {"text": "x", "textual_model_name": "ViT-B/32", "top_k": 10000},
    )
    assert resp.status_code == 200
    assert observed and observed[0] == manager_service.MAX_TOP_K * 2 == 200
    body = resp.get_json()
    assert len(body["results"]) <= manager_service.MAX_TOP_K


# ---------- Image search ----------


def test_ru_rt_24_search_image_happy_path(flask_client, stub_managers, tmp_image_path):
    """RU-RT-24: POST /search/image happy path."""
    _add_product(
        flask_client,
        {
            "id": "p24",
            "name": "shoe",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "images": [tmp_image_path],
        },
    )
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/image",
        {"image": tmp_image_path, "visual_model_name": "ViT-B/32"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert isinstance(body["results"], list)
    if body["results"]:
        entry = body["results"][0]
        assert "product_id" in entry and "score" in entry and "best_image_no" in entry


def test_ru_rt_25_search_image_missing_image(flask_client):
    """RU-RT-25: POST /search/image missing image field returns 400."""
    resp = _post_json(
        flask_client, "/api/retrieval/search/image", {"visual_model_name": "ViT-B/32"}
    )
    assert resp.status_code == 400
    assert "Missing required field: image" in resp.get_json()["message"]


def test_ru_rt_26_search_image_oversize(flask_client, stub_managers, oversize_image_path):
    """RU-RT-26: POST /search/image with oversize image returns 400."""
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/image",
        {"image": oversize_image_path, "visual_model_name": "ViT-B/32"},
    )
    assert resp.status_code == 400
    assert "exceeds maximum limit of 50MB" in resp.get_json()["message"]


def test_ru_rt_27_search_image_top_k_negative(flask_client, stub_managers, tmp_image_path):
    """RU-RT-27: POST /search/image with top_k=-1 returns 400."""
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/image",
        {"image": tmp_image_path, "visual_model_name": "ViT-B/32", "top_k": -1},
    )
    assert resp.status_code == 400
    assert "top_k must be >= 1" in resp.get_json()["message"]


# ---------- Late fusion ----------


def test_ru_rt_28_search_late_happy_path(flask_client, stub_managers, tmp_image_path):
    """RU-RT-28: POST /search/late happy path."""
    _add_product(
        flask_client,
        {
            "id": "p28",
            "name": "shoe",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "images": [tmp_image_path],
        },
    )
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/late",
        {
            "text": "shoe",
            "textual_model_name": "ViT-B/32",
            "text_weight": 0.5,
            "image": tmp_image_path,
            "visual_model_name": "ViT-B/32",
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["meta"]["text_weight"] == 0.5
    assert body["meta"]["image_weight"] == 0.5


def test_ru_rt_29_search_late_missing_text_weight(flask_client, tmp_image_path):
    """RU-RT-29: POST /search/late missing text_weight returns 400."""
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/late",
        {
            "text": "shoe",
            "textual_model_name": "ViT-B/32",
            "image": tmp_image_path,
            "visual_model_name": "ViT-B/32",
        },
    )
    assert resp.status_code == 400
    assert "Missing required field: text_weight" in resp.get_json()["message"]


def test_ru_rt_30_search_late_text_weight_out_of_range(flask_client, stub_managers, tmp_image_path):
    """RU-RT-30: POST /search/late text_weight=1.5 returns 400."""
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/late",
        {
            "text": "shoe",
            "textual_model_name": "ViT-B/32",
            "text_weight": 1.5,
            "image": tmp_image_path,
            "visual_model_name": "ViT-B/32",
        },
    )
    assert resp.status_code == 400
    assert "text_weight must be between 0 and 1" in resp.get_json()["message"]


# ---------- Early fusion ----------


def test_ru_rt_31_search_early_happy_path(flask_client, stub_managers, tmp_image_path):
    """RU-RT-31: POST /search/early happy path."""
    _add_product(
        flask_client,
        {
            "id": "p31",
            "name": "shoe",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "fused_model_name": "ViT-B/32",
            "images": [tmp_image_path],
        },
    )
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/early",
        {
            "text": "shoe",
            "image": tmp_image_path,
            "fused_model_name": "ViT-B/32",
            "text_weight": 0.5,
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert "text_weight" in body["meta"]


def test_ru_rt_32_search_early_missing_fused_model_name(flask_client, tmp_image_path):
    """RU-RT-32: POST /search/early missing fused_model_name returns 400."""
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/early",
        {"text": "shoe", "image": tmp_image_path},
    )
    assert resp.status_code == 400
    assert "Missing required field: fused_model_name" in resp.get_json()["message"]


def test_ru_rt_33_search_early_top_k_zero(flask_client, stub_managers, tmp_image_path):
    """RU-RT-33: POST /search/early invalid top_k=0 returns 400."""
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/early",
        {
            "text": "shoe",
            "image": tmp_image_path,
            "fused_model_name": "ViT-B/32",
            "top_k": 0,
        },
    )
    assert resp.status_code == 400
    assert "top_k must be >= 1" in resp.get_json()["message"]


# ---------- Cross-modal: image-by-text ----------


def test_ru_rt_34_image_by_text_non_multimodal(flask_client, stub_managers):
    """RU-RT-34: POST /search/image-by-text with non-multimodal model returns 400."""
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/image-by-text",
        {"text": "shoe", "fused_model_name": "BAAI/bge-large-en-v1.5"},
    )
    assert resp.status_code == 400
    msg = resp.get_json()["message"]
    assert ("multimodal" in msg) or ("Cross-modal search requires" in msg)


def test_ru_rt_35_image_by_text_happy_path(flask_client, stub_managers, tmp_image_path):
    """RU-RT-35: POST /search/image-by-text happy path."""
    _add_product(
        flask_client,
        {
            "id": "p35",
            "name": "shoe",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "fused_model_name": "ViT-B/32",
            "images": [tmp_image_path],
        },
    )
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/image-by-text",
        {"text": "shoe", "fused_model_name": "ViT-B/32"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert isinstance(body["results"], list)


def test_ru_rt_36_image_by_text_missing_text(flask_client):
    """RU-RT-36: POST /search/image-by-text missing text returns 400."""
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/image-by-text",
        {"fused_model_name": "ViT-B/32"},
    )
    assert resp.status_code == 400
    assert "Missing required field: text" in resp.get_json()["message"]


def test_ru_rt_37_image_by_text_top_k_clamp(flask_client, stub_managers, tmp_image_path):
    """RU-RT-37: POST /search/image-by-text top_k=999 clamps fused search to 200."""
    from services import manager_service

    _add_product(
        flask_client,
        {
            "id": "p37",
            "name": "shoe",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "fused_model_name": "ViT-B/32",
            "images": [tmp_image_path],
        },
    )

    mgr = manager_service.get_faiss_manager("ViT-B/32")
    observed: List[int] = []
    orig = mgr.search_fused

    def spy(query_embedding, top_k=10, model_name=None):
        observed.append(top_k)
        return orig(query_embedding=query_embedding, top_k=top_k, model_name=model_name)

    mgr.search_fused = spy  # type: ignore[assignment]

    resp = _post_json(
        flask_client,
        "/api/retrieval/search/image-by-text",
        {"text": "shoe", "fused_model_name": "ViT-B/32", "top_k": 999},
    )
    assert resp.status_code == 200
    assert observed and observed[0] == manager_service.MAX_TOP_K * 2 == 200


# ---------- Cross-modal: text-by-image ----------


def test_ru_rt_38_text_by_image_non_multimodal(flask_client, stub_managers, tmp_image_path):
    """RU-RT-38: POST /search/text-by-image with non-multimodal model returns 400."""
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/text-by-image",
        {"image": tmp_image_path, "fused_model_name": "BAAI/bge-large-en-v1.5"},
    )
    assert resp.status_code == 400
    msg = resp.get_json()["message"]
    assert ("multimodal" in msg) or ("Cross-modal search requires" in msg)


def test_ru_rt_39_text_by_image_happy_path(flask_client, stub_managers, tmp_image_path):
    """RU-RT-39: POST /search/text-by-image happy path."""
    _add_product(
        flask_client,
        {
            "id": "p39",
            "name": "shoe",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "fused_model_name": "ViT-B/32",
            "images": [tmp_image_path],
        },
    )
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/text-by-image",
        {"image": tmp_image_path, "fused_model_name": "ViT-B/32"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"


def test_ru_rt_40_text_by_image_missing_image(flask_client):
    """RU-RT-40: POST /search/text-by-image missing image returns 400."""
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/text-by-image",
        {"fused_model_name": "ViT-B/32"},
    )
    assert resp.status_code == 400
    assert "Missing required field: image" in resp.get_json()["message"]


def test_ru_rt_41_text_by_image_oversize(flask_client, stub_managers, oversize_image_path):
    """RU-RT-41: POST /search/text-by-image oversize image returns 400."""
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/text-by-image",
        {"image": oversize_image_path, "fused_model_name": "ViT-B/32"},
    )
    assert resp.status_code == 400
    assert "exceeds maximum limit of 50MB" in resp.get_json()["message"]


def test_ru_rt_42_search_text_empty_index(flask_client, stub_managers):
    """RU-RT-42: POST /search/text on empty index returns empty results."""
    resp = _post_json(
        flask_client,
        "/api/retrieval/search/text",
        {"text": "shoe", "textual_model_name": "ViT-B/32"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["results"] == []
    assert body["meta"]["total_results"] == 0
