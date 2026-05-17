"""
Unit tests for services.manager_service.

Mocking strategy:
- The real FAISSManager (vector_db.faiss_manager) is used for every cache /
  persistence test. faiss is never mocked.
- TextModelManager / VisualModelManager / FusedModelManager are replaced with
  lightweight recording fakes (`_FakeTextManager`, `_FakeVisualManager`,
  `_FakeFusedManager`) at the boundary visible to services.manager_service so
  that no real model weights are ever loaded.
- DATA_BASE_PATH is monkeypatched to a tempfile.TemporaryDirectory before any
  FAISSManager is created.
- An autouse fixture clears the four module-level caches
  (`_textual_managers`, `_visual_managers`, `_fused_managers`,
  `_faiss_managers`) before AND after every test.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading

import numpy as np
import pytest

from services import manager_service as ms
from vector_db.faiss_manager import FAISSManager, IndexType, make_folder_name


# --------------------------------------------------------------------------- #
# Fake managers (record constructor args, don't load weights)
# --------------------------------------------------------------------------- #


class _FakeTextManager:
    def __init__(self, model_type=None, model_config=None):
        self.model_type = model_type
        self.model_config = model_config or {}
        self.model = object()


class _FakeVisualManager:
    def __init__(self, model_type=None, model_config=None):
        self.model_type = model_type
        self.model_config = model_config or {}
        self.model = object()


class _FakeFusedManager:
    def __init__(self, model_type=None, model_config=None):
        self.model_type = model_type
        self.model_config = model_config or {}
        self.model = object()


# --------------------------------------------------------------------------- #
# Autouse fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_caches():
    """Clear the four module-level caches before and after each test."""
    ms._textual_managers.clear()
    ms._visual_managers.clear()
    ms._fused_managers.clear()
    ms._faiss_managers.clear()
    yield
    ms._textual_managers.clear()
    ms._visual_managers.clear()
    ms._fused_managers.clear()
    ms._faiss_managers.clear()


@pytest.fixture(autouse=True)
def _stub_managers(monkeypatch):
    """Replace heavyweight managers with fakes inside services.manager_service."""
    monkeypatch.setattr(ms, "TextModelManager", _FakeTextManager)
    monkeypatch.setattr(ms, "VisualModelManager", _FakeVisualManager)
    monkeypatch.setattr(ms, "FusedModelManager", _FakeFusedManager)
    yield


@pytest.fixture
def configured(monkeypatch, tmp_index_dir):
    """load_config() then point DATA_BASE_PATH at the temp dir."""
    ms.load_config()
    monkeypatch.setattr(ms, "DATA_BASE_PATH", tmp_index_dir)
    return tmp_index_dir


# --------------------------------------------------------------------------- #
# Configuration loading
# --------------------------------------------------------------------------- #


def test_ru_ms_01_load_config_populates_model_registry():
    """RU-MS-01: load_config registers every model with type + dimension."""
    ms.load_config()
    expected = {
        "ViT-B/32",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-large-finetuned",
        "Qwen/Qwen3-Embedding-8B",
        "Marqo/marqo-ecommerce-embeddings-L",
        "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    }
    assert set(ms.MODEL_REGISTRY.keys()) == expected
    assert ms.MODEL_REGISTRY["ViT-B/32"]["dimension"] == 512
    assert ms.MODEL_REGISTRY["ViT-B/32"]["type"] == "clip"
    assert ms.MODEL_REGISTRY["BAAI/bge-large-en-v1.5"]["dimension"] == 1024


def test_ru_ms_02_load_config_exposes_defaults():
    """RU-MS-02: HOST/PORT/MAX_TOP_K/DEFAULT_TOP_K/DEFAULT_DIMENSION/DATA_BASE_PATH."""
    ms.load_config()
    assert ms.HOST == "0.0.0.0"
    assert ms.PORT == 5002
    assert ms.MAX_TOP_K == 100
    assert ms.DEFAULT_TOP_K == 10
    assert ms.DEFAULT_DIMENSION == 512
    assert ms.DATA_BASE_PATH == "./data"


def test_ru_ms_03_load_config_default_models():
    """RU-MS-03: DEFAULT_MODELS has correct textual + visual."""
    ms.load_config()
    assert ms.DEFAULT_MODELS["textual"] == "BAAI/bge-large-en-v1.5"
    assert ms.DEFAULT_MODELS["visual"] == "ViT-B/32"


def test_ru_ms_04_load_config_missing_file_raises(monkeypatch):
    """RU-MS-04: missing config.json -> RuntimeError 'Configuration file not found'."""
    real_open = open

    def fake_open(path, *args, **kwargs):
        if str(path).endswith("config.json"):
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    with pytest.raises(RuntimeError, match=r"Configuration file not found"):
        ms.load_config()


def test_ru_ms_05_load_config_invalid_json_raises(monkeypatch):
    """RU-MS-05: malformed JSON -> RuntimeError 'Invalid JSON'."""
    def fake_json_load(_f):
        raise json.JSONDecodeError("bad", "{not json", 0)

    monkeypatch.setattr(ms.json, "load", fake_json_load)
    with pytest.raises(RuntimeError, match=r"Invalid JSON"):
        ms.load_config()


# --------------------------------------------------------------------------- #
# get_faiss_manager / caching / dimension routing
# --------------------------------------------------------------------------- #


def test_ru_ms_06_get_faiss_manager_dimension_and_path(configured):
    """RU-MS-06: dimension == 512 and folder lives under DATA_BASE_PATH."""
    mgr = ms.get_faiss_manager("ViT-B/32")
    assert isinstance(mgr, FAISSManager)
    assert mgr.dimension == 512
    expected_folder = make_folder_name("ViT-B/32", 512)
    assert mgr.index_path.endswith(expected_folder)
    assert os.path.normpath(mgr.index_path).startswith(os.path.normpath(configured))


def test_ru_ms_07_get_faiss_manager_caches_same_instance(configured):
    """RU-MS-07: second call returns the same cached FAISSManager."""
    m1 = ms.get_faiss_manager("ViT-B/32")
    m2 = ms.get_faiss_manager("ViT-B/32")
    assert m1 is m2


def test_ru_ms_08_get_faiss_manager_distinct_per_model(configured):
    """RU-MS-08: different models -> distinct managers and dimensions."""
    m_clip = ms.get_faiss_manager("ViT-B/32")
    m_bge = ms.get_faiss_manager("BAAI/bge-large-en-v1.5")
    assert m_clip is not m_bge
    assert m_clip.dimension == 512
    assert m_bge.dimension == 1024
    assert m_clip.index_path != m_bge.index_path


def test_ru_ms_09_get_faiss_manager_none_falls_back_to_default_textual(configured):
    """RU-MS-09: get_faiss_manager(None) with empty cache uses DEFAULT_MODELS textual."""
    mgr = ms.get_faiss_manager(None)
    assert mgr.dimension == 1024  # BGE
    expected_folder = make_folder_name("BAAI/bge-large-en-v1.5", 1024)
    assert mgr.index_path.endswith(expected_folder)


def test_ru_ms_10_get_faiss_manager_none_returns_first_cached(configured):
    """RU-MS-10: with at least one cached manager, None returns the first cached value."""
    cached = ms.get_faiss_manager("ViT-B/32")
    mgr = ms.get_faiss_manager(None)
    assert mgr is cached


def test_ru_ms_11_unknown_model_uses_default_dimension(configured):
    """RU-MS-11: unknown model name falls back to DEFAULT_DIMENSION (512)."""
    mgr = ms.get_faiss_manager("not/a/registered-model")
    assert mgr.dimension == 512


# --------------------------------------------------------------------------- #
# get_all_faiss_managers / discover_model_folders / get_or_load_all_faiss_managers
# --------------------------------------------------------------------------- #


def test_ru_ms_12_get_all_faiss_managers_returns_live_dict(configured):
    """RU-MS-12: dict keyed by folder name with the live FAISSManager."""
    m = ms.get_faiss_manager("ViT-B/32")
    d = ms.get_all_faiss_managers()
    expected_key = make_folder_name("ViT-B/32", 512)
    assert isinstance(d, dict)
    assert list(d.keys()) == [expected_key]
    assert d[expected_key] is m


def test_ru_ms_13_discover_model_folders_missing_path(monkeypatch, tmp_index_dir):
    """RU-MS-13: returns [] when DATA_BASE_PATH does not exist."""
    ms.load_config()
    missing = os.path.join(tmp_index_dir, "does_not_exist")
    monkeypatch.setattr(ms, "DATA_BASE_PATH", missing)
    assert ms.discover_model_folders() == []


def test_ru_ms_14_discover_model_folders_filters_embeddings(monkeypatch, tmp_index_dir):
    """RU-MS-14: returns only directories ending with '_embeddings'."""
    ms.load_config()
    monkeypatch.setattr(ms, "DATA_BASE_PATH", tmp_index_dir)
    os.mkdir(os.path.join(tmp_index_dir, "ViT-B-32_512_embeddings"))
    os.mkdir(os.path.join(tmp_index_dir, "bge_1024_embeddings"))
    os.mkdir(os.path.join(tmp_index_dir, "random_other"))
    with open(os.path.join(tmp_index_dir, "not_a_dir_embeddings"), "w") as f:
        f.write("file not dir")
    found = set(ms.discover_model_folders())
    assert found == {"ViT-B-32_512_embeddings", "bge_1024_embeddings"}


def test_ru_ms_15_get_or_load_parses_dimension_from_folder(monkeypatch, tmp_index_dir):
    """RU-MS-15: parses dimension from <name>_<dim>_embeddings folder name."""
    ms.load_config()
    monkeypatch.setattr(ms, "DATA_BASE_PATH", tmp_index_dir)
    folder = "bge_1024_embeddings"
    os.mkdir(os.path.join(tmp_index_dir, folder))
    d = ms.get_or_load_all_faiss_managers()
    assert folder in d
    assert d[folder].dimension == 1024


def test_ru_ms_16_get_or_load_unparseable_dim_falls_back(monkeypatch, tmp_index_dir):
    """RU-MS-16: unparseable folder dimension -> DEFAULT_DIMENSION."""
    ms.load_config()
    monkeypatch.setattr(ms, "DATA_BASE_PATH", tmp_index_dir)
    folder = "weird_embeddings"
    os.mkdir(os.path.join(tmp_index_dir, folder))
    d = ms.get_or_load_all_faiss_managers()
    assert folder in d
    assert d[folder].dimension == ms.DEFAULT_DIMENSION == 512


# --------------------------------------------------------------------------- #
# remove_product_from_all_models
# --------------------------------------------------------------------------- #


def test_ru_ms_17_remove_product_from_all_models_returns_only_affected(
    configured, clip_vec, bge_vec
):
    """RU-MS-17: returns only folders where >=1 vector was actually removed."""
    clip_mgr = ms.get_faiss_manager("ViT-B/32")
    bge_mgr = ms.get_faiss_manager("BAAI/bge-large-en-v1.5")

    # CLARIFY: spec uses `add_textual` (does not exist on FAISSManager). Use the
    # real public API `add_to_textual(embedding, product_id, model_name)`.
    clip_mgr.add_to_textual(clip_vec("P1"), "P1", "ViT-B/32")

    result = ms.remove_product_from_all_models("P1")
    clip_folder = make_folder_name("ViT-B/32", 512)
    bge_folder = make_folder_name("BAAI/bge-large-en-v1.5", 1024)
    assert clip_folder in result
    assert bge_folder not in result
    assert sum(result[clip_folder].values()) >= 1
    # CLIP textual now empty
    assert clip_mgr.get_all_sizes()["textual"] == 0
    # save() should have created on-disk artifacts
    assert os.path.isdir(clip_mgr.index_path)
    assert len(os.listdir(clip_mgr.index_path)) > 0


def test_ru_ms_18_remove_product_from_all_models_unknown_returns_empty(configured):
    """RU-MS-18: unknown product across all managers -> {}."""
    ms.get_faiss_manager("ViT-B/32")
    ms.get_faiss_manager("BAAI/bge-large-en-v1.5")
    assert ms.remove_product_from_all_models("nope") == {}


# --------------------------------------------------------------------------- #
# Textual / visual / fused manager getters
# --------------------------------------------------------------------------- #


def test_ru_ms_19_get_textual_manager_caches(configured):
    """RU-MS-19: get_textual_manager returns the same cached instance."""
    t1 = ms.get_textual_manager("BAAI/bge-large-en-v1.5")
    t2 = ms.get_textual_manager("BAAI/bge-large-en-v1.5")
    assert t1 is t2
    assert isinstance(t1, _FakeTextManager)


def test_ru_ms_20_get_textual_manager_resolves_type_via_registry(configured):
    """RU-MS-20: type comes from MODEL_REGISTRY when present."""
    t = ms.get_textual_manager("Qwen/Qwen3-Embedding-8B")
    assert t.model_type == "qwen"


@pytest.mark.parametrize(
    "model_name,expected_type",
    [
        ("BAAI/bge-something-new", "bge"),
        ("Qwen/foo", "qwen"),
        ("Marqo/foo", "marqo"),
        ("openai/clip-vit", "clip"),
    ],
)
def test_ru_ms_21_get_textual_manager_name_fallback(monkeypatch, model_name, expected_type):
    """RU-MS-21: when registry is empty, type detected from model name patterns."""
    ms.load_config()
    monkeypatch.setattr(ms, "MODEL_REGISTRY", {})
    t = ms.get_textual_manager(model_name)
    assert t.model_type == expected_type


@pytest.mark.parametrize(
    "model_name,expected_type,clear_registry",
    [
        ("facebook/dinov3-vit7b16-pretrain-lvd1689m", "dinov3", False),
        ("facebook/dinov3-something", "dinov3", True),
        ("Marqo/x", "marqo", True),
        ("some-clip", "clip", True),
    ],
)
def test_ru_ms_22_get_visual_manager_type_resolution(
    monkeypatch, model_name, expected_type, clear_registry
):
    """RU-MS-22: visual manager type via registry first, then name patterns."""
    ms.load_config()
    if clear_registry:
        monkeypatch.setattr(ms, "MODEL_REGISTRY", {})
    v = ms.get_visual_manager(model_name)
    assert v.model_type == expected_type
    # caching
    assert ms.get_visual_manager(model_name) is v


def test_ru_ms_23_get_fused_manager_caches_and_resolves_marqo(configured):
    """RU-MS-23: fused manager caches and uses _get_visual_model_type."""
    f1 = ms.get_fused_manager("Marqo/marqo-ecommerce-embeddings-L")
    f2 = ms.get_fused_manager("Marqo/marqo-ecommerce-embeddings-L")
    assert f1 is f2
    assert isinstance(f1, _FakeFusedManager)
    assert f1.model_type == "marqo"


# --------------------------------------------------------------------------- #
# combine_product_text
# --------------------------------------------------------------------------- #


def test_ru_ms_24_combine_product_text_full():
    """RU-MS-24: full inputs joined with spaces, price prefixed 'Price: '."""
    out = ms.combine_product_text("Shoe", "Comfy", "Nike", "Footwear", 99.9)
    assert out == "Shoe Comfy Nike Footwear Price: 99.9"


def test_ru_ms_25_combine_product_text_skips_falsy():
    """RU-MS-25: empty/None/0 fields skipped."""
    out = ms.combine_product_text("Hat", "", "BrandX", None, 0)
    assert out == "Hat BrandX"


def test_ru_ms_26_combine_product_text_all_empty():
    """RU-MS-26: all-falsy inputs -> empty string."""
    assert ms.combine_product_text("", "", None, None, 0) == ""


# --------------------------------------------------------------------------- #
# get_all_index_stats / get_available_models
# --------------------------------------------------------------------------- #


def test_ru_ms_27_get_all_index_stats_reflects_real_sizes(configured, clip_vec):
    """RU-MS-27: stats dict mirrors FAISSManager.get_all_sizes() per folder."""
    mgr = ms.get_faiss_manager("ViT-B/32")
    mgr.add_to_textual(clip_vec("P1"), "P1", "ViT-B/32")
    mgr.add_to_textual(clip_vec("P2"), "P2", "ViT-B/32")
    stats = ms.get_all_index_stats()
    folder = make_folder_name("ViT-B/32", 512)
    assert folder in stats
    assert stats[folder]["textual"] == 2


def test_ru_ms_28_get_available_models_categorizes(configured):
    """RU-MS-28: textual/visual_models categorized by TEXTUAL_TYPES/VISUAL_TYPES."""
    out = ms.get_available_models()
    textual_names = {e["name"] for e in out["textual_models"]}
    visual_names = {e["name"] for e in out["visual_models"]}

    assert "ViT-B/32" in textual_names
    assert "BAAI/bge-large-en-v1.5" in textual_names
    assert "Qwen/Qwen3-Embedding-8B" in textual_names
    assert "Marqo/marqo-ecommerce-embeddings-L" in textual_names
    assert "facebook/dinov3-vit7b16-pretrain-lvd1689m" not in textual_names

    assert "ViT-B/32" in visual_names
    assert "Marqo/marqo-ecommerce-embeddings-L" in visual_names
    assert "facebook/dinov3-vit7b16-pretrain-lvd1689m" in visual_names
    assert "BAAI/bge-large-en-v1.5" not in visual_names
    assert "Qwen/Qwen3-Embedding-8B" not in visual_names

    assert out["defaults"] == {
        "textual": "BAAI/bge-large-en-v1.5",
        "visual": "ViT-B/32",
    }


def test_ru_ms_29_get_available_models_empty(monkeypatch):
    """RU-MS-29: empty registry -> empty lists, empty default strings."""
    monkeypatch.setattr(ms, "MODEL_REGISTRY", {})
    monkeypatch.setattr(ms, "DEFAULT_MODELS", {})
    out = ms.get_available_models()
    assert out == {
        "textual_models": [],
        "visual_models": [],
        "defaults": {"textual": "", "visual": ""},
    }


# --------------------------------------------------------------------------- #
# Integration smoke
# --------------------------------------------------------------------------- #


def test_ru_ms_30_end_to_end_add_and_search(configured, bge_vec):
    """RU-MS-30: add_to_textual + search_textual returns the seeded product as top hit."""
    mgr = ms.get_faiss_manager("BAAI/bge-large-en-v1.5")
    # CLARIFY: spec used `add_textual` / `search_textual` with positional vec; the
    # real API takes (embedding, product_id, model_name). Using real API.
    mgr.add_to_textual(bge_vec("hello"), "P1", "BAAI/bge-large-en-v1.5")
    hits = mgr.search_textual(bge_vec("hello"), top_k=1)
    assert hits, "expected at least one hit"
    assert hits[0]["product_id"] == "P1"


def test_ru_ms_31_persistence_smoke(configured, bge_vec):
    """RU-MS-31: save() leaves at least one file under index_path."""
    mgr = ms.get_faiss_manager("BAAI/bge-large-en-v1.5")
    mgr.add_to_textual(bge_vec("hello"), "P1", "BAAI/bge-large-en-v1.5")
    mgr.save()
    assert os.path.isdir(mgr.index_path)
    files = os.listdir(mgr.index_path)
    assert len(files) > 0


def test_ru_ms_32_concurrency_smoke(configured, clip_vec):
    """RU-MS-32: two threads share the same cached manager and both add succeeds."""
    seen = {}

    def worker(pid):
        mgr = ms.get_faiss_manager("ViT-B/32")
        seen[pid] = mgr
        mgr.add_to_textual(clip_vec(pid), pid, "ViT-B/32")

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert seen["A"] is seen["B"]
    assert seen["A"].get_all_sizes()["textual"] >= 2
