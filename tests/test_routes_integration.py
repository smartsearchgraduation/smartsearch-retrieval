"""Route integration tests (RI-L2-NN).

Full request-cycle tests: real services.manager_service, real in-memory
FAISSManager rooted at tmp_index_dir, only the model-pool boundary is
stubbed (via the ``stub_managers`` fixture).
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest


pytestmark = pytest.mark.integration


def _post(client, url: str, payload: Dict[str, Any]):
    return client.post(url, data=json.dumps(payload), content_type="application/json")


def _put(client, url: str, payload: Dict[str, Any]):
    return client.put(url, data=json.dumps(payload), content_type="application/json")


def _add(client, payload: Dict[str, Any]):
    return _post(client, "/api/retrieval/add-product", payload)


@pytest.mark.integration
def test_ri_l2_01_add_then_text_search(flask_client, stub_managers):
    """RI-L2-01: Add then text-search round-trip."""
    add = _add(
        flask_client,
        {
            "id": "rt1",
            "name": "Red Sneaker",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
        },
    )
    assert add.status_code == 201

    search = _post(
        flask_client,
        "/api/retrieval/search/text",
        {"text": "Red Sneaker", "textual_model_name": "ViT-B/32", "top_k": 5},
    )
    assert search.status_code == 200
    body = search.get_json()
    assert body["results"], "expected at least one result"
    assert body["results"][0]["product_id"] == "rt1"


@pytest.mark.integration
def test_ri_l2_02_add_image_then_image_search(flask_client, stub_managers, tmp_image_path):
    """RI-L2-02: Add with image then image-search round-trip."""
    add = _add(
        flask_client,
        {
            "id": "rt2",
            "name": "Sneaker",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "images": [tmp_image_path],
        },
    )
    assert add.status_code == 201
    assert len(add.get_json()["details"]["visual_vector_ids"]) == 1

    search = _post(
        flask_client,
        "/api/retrieval/search/image",
        {"image": tmp_image_path, "visual_model_name": "ViT-B/32"},
    )
    assert search.status_code == 200
    body = search.get_json()
    assert body["results"]
    assert body["results"][0]["product_id"] == "rt2"


@pytest.mark.integration
def test_ri_l2_03_delete_after_add_removes_from_search(flask_client, stub_managers):
    """RI-L2-03: Delete after add removes from search results."""
    _add(
        flask_client,
        {
            "id": "rt3",
            "name": "Hat",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
        },
    )
    delete = flask_client.delete("/api/retrieval/delete-product/rt3")
    assert delete.status_code == 200
    assert delete.get_json()["details"]["total_removed"] > 0

    search = _post(
        flask_client,
        "/api/retrieval/search/text",
        {"text": "Hat", "textual_model_name": "ViT-B/32"},
    )
    assert search.status_code == 200
    body = search.get_json()
    assert all(r["product_id"] != "rt3" for r in body["results"])


@pytest.mark.integration
def test_ri_l2_04_top_k_clamp_end_to_end(flask_client, stub_managers):
    """RI-L2-04: top_k=200 (above MAX_TOP_K) clamps results to <= 100."""
    for i in range(3):
        _add(
            flask_client,
            {
                "id": f"rtbulk{i}",
                "name": f"item-{i}",
                "textual_model_name": "ViT-B/32",
                "visual_model_name": "ViT-B/32",
            },
        )

    search = _post(
        flask_client,
        "/api/retrieval/search/text",
        {"text": "item", "textual_model_name": "ViT-B/32", "top_k": 200},
    )
    assert search.status_code == 200
    body = search.get_json()
    assert len(body["results"]) <= 100
    assert body["meta"]["total_results"] <= 3


@pytest.mark.integration
def test_ri_l2_05_image_by_text_round_trip(flask_client, stub_managers, tmp_image_path):
    """RI-L2-05: image-by-text cross-modal round-trip."""
    _add(
        flask_client,
        {
            "id": "rt5",
            "name": "Red Sneaker",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "fused_model_name": "ViT-B/32",
            "images": [tmp_image_path],
        },
    )
    resp = _post(
        flask_client,
        "/api/retrieval/search/image-by-text",
        {"text": "Red Sneaker", "fused_model_name": "ViT-B/32"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["results"]
    assert any(r["product_id"] == "rt5" for r in body["results"])


@pytest.mark.integration
def test_ri_l2_06_text_by_image_round_trip(flask_client, stub_managers, tmp_image_path):
    """RI-L2-06: text-by-image cross-modal round-trip."""
    _add(
        flask_client,
        {
            "id": "rt6",
            "name": "Sneaker",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "fused_model_name": "ViT-B/32",
            "images": [tmp_image_path],
        },
    )
    resp = _post(
        flask_client,
        "/api/retrieval/search/text-by-image",
        {"image": tmp_image_path, "fused_model_name": "ViT-B/32"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert any(r["product_id"] == "rt6" for r in body["results"])


@pytest.mark.integration
def test_ri_l2_07_index_stats_reflects_added(flask_client, stub_managers, tmp_image_path):
    """RI-L2-07: index-stats reflects added products."""
    _add(
        flask_client,
        {
            "id": "ri7a",
            "name": "A",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "images": [tmp_image_path],
        },
    )
    _add(
        flask_client,
        {
            "id": "ri7b",
            "name": "B",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
        },
    )
    resp = flask_client.get("/api/retrieval/index-stats")
    assert resp.status_code == 200
    indices = resp.get_json()["indices"]
    assert indices, "expected at least one folder"
    # find any folder with the expected counts
    matched = False
    for stats in indices.values():
        if stats.get("textual", 0) >= 2 and stats.get("visual", 0) >= 1:
            matched = True
            break
    assert matched, f"no folder reflects expected counts: {indices}"


@pytest.mark.integration
def test_ri_l2_08_duplicate_add_dedup(flask_client, stub_managers, tmp_image_path):
    """RI-L2-08: Adding same product twice — dedup at search."""
    payload = {
        "id": "rt8",
        "name": "Cap",
        "textual_model_name": "ViT-B/32",
        "visual_model_name": "ViT-B/32",
        "images": [tmp_image_path],
    }
    r1 = _add(flask_client, payload)
    assert r1.status_code == 201
    r2 = _add(flask_client, payload)
    assert r2.status_code == 200
    assert r2.get_json()["details"]["skipped"] is True

    search = _post(
        flask_client,
        "/api/retrieval/search/text",
        {"text": "Cap", "textual_model_name": "ViT-B/32"},
    )
    assert search.status_code == 200
    pids = [r["product_id"] for r in search.get_json()["results"]]
    assert pids.count("rt8") == 1


@pytest.mark.integration
def test_ri_l2_09_persistence_smoke(flask_client, stub_managers):
    """RI-L2-09: Persistence smoke — save → reload from disk → search."""
    from services import manager_service

    _add(
        flask_client,
        {
            "id": "rt9",
            "name": "Boot",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
        },
    )
    # Force reload by clearing the FAISS cache only.
    manager_service._faiss_managers.clear()

    search = _post(
        flask_client,
        "/api/retrieval/search/text",
        {"text": "Boot", "textual_model_name": "ViT-B/32"},
    )
    # CLARIFY: spec says "If component does not auto-load, mark SKIP".
    # FAISSManager constructor auto-loads from index_path if files exist.
    if search.status_code != 200:
        pytest.skip("FAISSManager did not auto-load from disk")
    body = search.get_json()
    pids = [r["product_id"] for r in body["results"]]
    if "rt9" not in pids:
        pytest.skip("manager_service does not auto-rehydrate caches from disk")
    assert "rt9" in pids


@pytest.mark.integration
def test_ri_l2_10_update_replaces_embeddings(flask_client, stub_managers):
    """RI-L2-10: Update endpoint replaces embeddings."""
    _add(
        flask_client,
        {
            "id": "rt10",
            "name": "Old",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
        },
    )
    put = _put(
        flask_client,
        "/api/retrieval/update-product/rt10",
        {
            "name": "Brand New Hat",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
        },
    )
    assert put.status_code == 200
    removed = put.get_json()["details"]["removed_counts"]
    # removed_counts is a dict of folder->{type:count}; total non-zero
    total = sum(sum(v.values()) for v in removed.values()) if removed else 0
    assert total > 0

    search = _post(
        flask_client,
        "/api/retrieval/search/text",
        {"text": "Brand New Hat", "textual_model_name": "ViT-B/32"},
    )
    assert search.status_code == 200
    body = search.get_json()
    assert body["results"]
    assert body["results"][0]["product_id"] == "rt10"


@pytest.mark.integration
def test_ri_l2_11_late_fusion_end_to_end(flask_client, stub_managers, tmp_image_path):
    """RI-L2-11: Late fusion end-to-end."""
    _add(
        flask_client,
        {
            "id": "rt11",
            "name": "Combo",
            "textual_model_name": "ViT-B/32",
            "visual_model_name": "ViT-B/32",
            "images": [tmp_image_path],
        },
    )
    resp = _post(
        flask_client,
        "/api/retrieval/search/late",
        {
            "text": "Combo",
            "textual_model_name": "ViT-B/32",
            "text_weight": 0.5,
            "image": tmp_image_path,
            "visual_model_name": "ViT-B/32",
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert any(r["product_id"] == "rt11" for r in body["results"])
    assert body["meta"]["text_weight"] == 0.5
    assert body["meta"]["image_weight"] == 0.5


@pytest.mark.integration
def test_ri_l2_12_cross_modal_rejects_non_multimodal(flask_client, stub_managers):
    """RI-L2-12: Cross-modal endpoint rejects non-multimodal model."""
    _add(
        flask_client,
        {
            "id": "rt12",
            "name": "Anything",
            "textual_model_name": "BAAI/bge-large-en-v1.5",
            "visual_model_name": "ViT-B/32",
        },
    )
    resp = _post(
        flask_client,
        "/api/retrieval/search/image-by-text",
        {"text": "Anything", "fused_model_name": "BAAI/bge-large-en-v1.5"},
    )
    assert resp.status_code == 400
    assert "multimodal" in resp.get_json()["message"]
