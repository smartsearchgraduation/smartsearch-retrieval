"""Unit tests for vector_db.faiss_manager.

Implements spec tests/specs/faiss_manager.md (RU-FM-01 through RU-FM-26).
Real in-memory faiss is used; faiss is never mocked. Persistence paths come
from the tmp_index_dir fixture only. Embeddings are produced by deterministic
helpers — no real model weights are loaded.
"""

from __future__ import annotations

import json
import os
import threading

import faiss
import numpy as np
import pytest

from vector_db.faiss_manager import (
    FAISSManager,
    IndexType,
    make_folder_name,
    sanitize_model_name,
)


# ---------------- RU-FM-01 ----------------


def test_ru_fm_01_default_dimension_init(faiss_manager_factory, tmp_index_dir):
    """RU-FM-01: Default-dimension initialization creates three empty IndexFlatIP-wrapped indices of dim 512."""
    mgr = faiss_manager_factory(dimension=512)
    for t in IndexType:
        assert mgr.dimensions[t] == 512
    assert mgr.get_all_sizes() == {"textual": 0, "visual": 0, "fused": 0}
    for t in IndexType:
        assert isinstance(mgr.indices[t], faiss.IndexIDMap)


# ---------------- RU-FM-02 ----------------


def test_ru_fm_02_per_index_dimension_override(
    faiss_manager_factory, bge_vec, dinov3_vec, clip_vec
):
    """RU-FM-02: Per-index dimension dict overrides the default for specified types and falls back to default for the rest."""
    mgr = faiss_manager_factory(
        dimension=512, dimensions={"textual": 1024, "visual": 4096}
    )
    assert mgr.dimensions[IndexType.TEXTUAL] == 1024
    assert mgr.dimensions[IndexType.VISUAL] == 4096
    assert mgr.dimensions[IndexType.FUSED] == 512

    tid = mgr.add_to_textual(bge_vec("s1"), "p1", "m")
    vid = mgr.add_to_visual(dinov3_vec("s1"), "p1", 0, "m")
    fid = mgr.add_to_fused(clip_vec("s1"), "p1", 0, "m")
    assert isinstance(tid, int) and isinstance(vid, int) and isinstance(fid, int)
    assert mgr.get_all_sizes() == {"textual": 1, "visual": 1, "fused": 1}


# ---------------- RU-FM-03 ----------------


def test_ru_fm_03_add_to_textual_returns_id_and_metadata(faiss_manager_factory, clip_vec):
    """RU-FM-03: add_to_textual returns a vector_id and stores model_name + product_id in metadata."""
    mgr = faiss_manager_factory(dimension=512)
    rid = mgr.add_to_textual(clip_vec("7"), "prodA", "clip-vit-b-32")
    assert rid == 0
    meta_list = mgr.metadata[IndexType.TEXTUAL]
    assert len(meta_list) == 1
    m = meta_list[0]
    assert m["product_id"] == "prodA"
    assert m["model_name"] == "clip-vit-b-32"
    assert m["_vector_id"] == 0


# ---------------- RU-FM-04 ----------------


def test_ru_fm_04_visual_and_fused_image_no(faiss_manager_factory, clip_vec):
    """RU-FM-04: add_to_visual and add_to_fused persist image_no alongside other metadata."""
    mgr = faiss_manager_factory(dimension=512)
    vid = mgr.add_to_visual(clip_vec("2"), "p1", 3, "clip")
    fid = mgr.add_to_fused(clip_vec("3"), "p1", 5, "clip")
    assert vid == 0
    assert fid == 0

    vmeta = mgr.metadata[IndexType.VISUAL][0]
    fmeta = mgr.metadata[IndexType.FUSED][0]
    assert vmeta["image_no"] == 3
    assert vmeta["product_id"] == "p1"
    assert vmeta["model_name"] == "clip"
    assert fmeta["image_no"] == 5
    assert fmeta["product_id"] == "p1"
    assert fmeta["model_name"] == "clip"


# ---------------- RU-FM-05 ----------------


def test_ru_fm_05_l2_normalization_on_add(faiss_manager_factory):
    """RU-FM-05: L2-normalization is applied to inputs at add time (non-unit input still produces unit-norm stored vector)."""
    mgr = faiss_manager_factory(dimension=512)
    raw = np.zeros(512, dtype=np.float32)
    raw[0] = 3.0
    raw[1] = 4.0  # norm = 5
    mgr.add_to_textual(raw.tolist(), "p", "m")

    # IndexIDMap wraps an IndexFlatIP — reconstruct via the underlying index,
    # since reconstruct() on the IDMap wrapper is not implemented for IndexFlatIP.
    idx = mgr.indices[IndexType.TEXTUAL].index
    rec = idx.reconstruct(0)
    rec = np.asarray(rec, dtype=np.float32)
    assert abs(float(np.linalg.norm(rec)) - 1.0) < 1e-5
    assert abs(rec[0] - 0.6) < 1e-5
    assert abs(rec[1] - 0.8) < 1e-5


# ---------------- RU-FM-06 ----------------


def test_ru_fm_06_id_counters_independent(faiss_manager_factory, clip_vec):
    """RU-FM-06: Custom-ID assignment via IndexIDMap: id_counter increases monotonically per index type and is independent across types."""
    mgr = faiss_manager_factory(dimension=512)
    t_ids = [mgr.add_to_textual(clip_vec(f"t{i}"), f"p{i}", "m") for i in range(3)]
    f_ids = [mgr.add_to_fused(clip_vec(f"f{i}"), f"p{i}", 0, "m") for i in range(2)]
    assert t_ids == [0, 1, 2]
    assert f_ids == [0, 1]
    assert mgr.id_counters[IndexType.TEXTUAL] == 3
    assert mgr.id_counters[IndexType.FUSED] == 2
    assert mgr.id_counters[IndexType.VISUAL] == 0


# ---------------- RU-FM-07 ----------------


def test_ru_fm_07_cosine_search_exact_match_first(faiss_manager_factory, clip_vec):
    """RU-FM-07: Cosine-similarity search via IndexFlatIP returns the exact-match vector with score ≈ 1.0 first."""
    mgr = faiss_manager_factory(dimension=512)
    for i in range(1, 6):
        mgr.add_to_textual(clip_vec(str(i)), f"p{i}", "m")
    results = mgr.search_textual(clip_vec("3"), top_k=5)
    assert len(results) == 5
    assert results[0]["product_id"] == "p3"
    assert abs(results[0]["score"] - 1.0) < 1e-4
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------- RU-FM-08 ----------------


def test_ru_fm_08_top_k_truncation(faiss_manager_factory, clip_vec):
    """RU-FM-08: top_k truncation: requesting fewer results than indexed returns exactly top_k items."""
    mgr = faiss_manager_factory(dimension=512)
    for i in range(6):
        mgr.add_to_textual(clip_vec(f"v{i}"), f"p{i}", "m")
    results = mgr.search_textual(clip_vec("v0"), top_k=3)
    assert len(results) == 3


# ---------------- RU-FM-09 ----------------


def test_ru_fm_09_model_name_filter(faiss_manager_factory, clip_vec):
    """RU-FM-09: model_name filter excludes non-matching results from search output."""
    mgr = faiss_manager_factory(dimension=512)
    for i in range(3):
        mgr.add_to_textual(clip_vec(f"a{i}"), f"pA{i}", "A")
    for i in range(3):
        mgr.add_to_textual(clip_vec(f"b{i}"), f"pB{i}", "B")
    results = mgr.search_textual(clip_vec("a0"), top_k=10, model_name="A")
    assert len(results) <= 3
    for r in results:
        assert r["model_name"] == "A"


# ---------------- RU-FM-10 ----------------


def test_ru_fm_10_convenience_wrappers_route_correctly(faiss_manager_factory, clip_vec):
    """RU-FM-10: Convenience wrappers search_textual / search_visual / search_fused route to their respective indices."""
    mgr = faiss_manager_factory(dimension=512)
    q = clip_vec("1")
    mgr.add_to_textual(q, "T", "m")
    mgr.add_to_visual(q, "V", 0, "m")
    mgr.add_to_fused(q, "F", 0, "m")
    rt = mgr.search_textual(q, top_k=1)
    rv = mgr.search_visual(q, top_k=1)
    rf = mgr.search_fused(q, top_k=1)
    assert rt[0]["product_id"] == "T"
    assert rv[0]["product_id"] == "V"
    assert rf[0]["product_id"] == "F"


# ---------------- RU-FM-11 ----------------


def test_ru_fm_11_has_product_scoping(faiss_manager_factory, clip_vec):
    """RU-FM-11: has_product returns True only for the index type the product was added to."""
    mgr = faiss_manager_factory(dimension=512)
    mgr.add_to_textual(clip_vec("1"), "p1", "m")
    assert mgr.has_product(IndexType.TEXTUAL, "p1") is True
    assert mgr.has_product(IndexType.VISUAL, "p1") is False
    assert mgr.has_product(IndexType.FUSED, "p1") is False
    assert mgr.has_product(IndexType.TEXTUAL, "ghost") is False


# ---------------- RU-FM-12 ----------------


def test_ru_fm_12_get_sizes_reflect_adds(faiss_manager_factory, clip_vec):
    """RU-FM-12: get_index_size and get_all_sizes reflect cumulative adds."""
    mgr = faiss_manager_factory(dimension=512)
    for i in range(4):
        mgr.add_to_textual(clip_vec(f"t{i}"), f"pt{i}", "m")
    for i in range(2):
        mgr.add_to_visual(clip_vec(f"v{i}"), f"pv{i}", 0, "m")
    mgr.add_to_fused(clip_vec("f0"), "pf0", 0, "m")
    assert mgr.get_index_size(IndexType.TEXTUAL) == 4
    assert mgr.get_all_sizes() == {"textual": 4, "visual": 2, "fused": 1}


# ---------------- RU-FM-13 ----------------


def test_ru_fm_13_remove_by_product_id(faiss_manager_factory, clip_vec):
    """RU-FM-13: remove_by_product_id removes all vectors for a product, returns the count, and rebuilds the index."""
    mgr = faiss_manager_factory(dimension=512)
    for i in range(3):
        mgr.add_to_textual(clip_vec(f"p1-{i}"), "p1", "m")
    for i in range(2):
        mgr.add_to_textual(clip_vec(f"p2-{i}"), "p2", "m")
    removed = mgr.remove_by_product_id(IndexType.TEXTUAL, "p1")
    assert removed == 3
    assert mgr.get_index_size(IndexType.TEXTUAL) == 2
    for meta in mgr.metadata[IndexType.TEXTUAL]:
        assert meta["product_id"] != "p1"

    results = mgr.search_textual(clip_vec("p1-0"), top_k=10)
    for r in results:
        assert r["product_id"] == "p2"


# ---------------- RU-FM-14 ----------------


def test_ru_fm_14_remove_nonexistent_product(faiss_manager_factory, clip_vec):
    """RU-FM-14: remove_by_product_id for non-existent product returns 0 and leaves index unchanged."""
    mgr = faiss_manager_factory(dimension=512)
    for i in range(2):
        mgr.add_to_textual(clip_vec(f"r{i}"), "real", "m")
    before = list(mgr.metadata[IndexType.TEXTUAL])
    removed = mgr.remove_by_product_id(IndexType.TEXTUAL, "missing")
    assert removed == 0
    assert mgr.get_index_size(IndexType.TEXTUAL) == 2
    assert mgr.metadata[IndexType.TEXTUAL] == before


# ---------------- RU-FM-15 ----------------


def test_ru_fm_15_remove_product_from_all(faiss_manager_factory, clip_vec):
    """RU-FM-15: remove_product_from_all sweeps every index type and returns per-type counts."""
    mgr = faiss_manager_factory(dimension=512)
    for i in range(2):
        mgr.add_to_textual(clip_vec(f"t{i}"), "p", "m")
    mgr.add_to_visual(clip_vec("v0"), "p", 0, "m")
    for i in range(3):
        mgr.add_to_fused(clip_vec(f"f{i}"), "p", i, "m")
    counts = mgr.remove_product_from_all("p")
    assert counts == {"textual": 2, "visual": 1, "fused": 3}
    assert mgr.get_all_sizes() == {"textual": 0, "visual": 0, "fused": 0}


# ---------------- RU-FM-16 ----------------


def test_ru_fm_16_save_writes_six_files(faiss_manager_factory, tmp_index_dir, clip_vec):
    """RU-FM-16: save() writes <type>_index.faiss and <type>_metadata.json for every IndexType."""
    mgr = faiss_manager_factory(dimension=512)
    mgr.add_to_textual(clip_vec("t"), "p", "m")
    mgr.add_to_visual(clip_vec("v"), "p", 0, "m")
    mgr.add_to_fused(clip_vec("f"), "p", 0, "m")
    mgr.save()
    files = set(os.listdir(tmp_index_dir))
    expected = {
        "textual_index.faiss",
        "textual_metadata.json",
        "visual_index.faiss",
        "visual_metadata.json",
        "fused_index.faiss",
        "fused_metadata.json",
    }
    assert expected.issubset(files)


# ---------------- RU-FM-17 ----------------


def test_ru_fm_17_save_no_path_raises(clip_vec):
    """RU-FM-17: save() raises ValueError when no path is provided and index_path is None."""
    mgr = FAISSManager(dimension=512, index_path=None, use_gpu=False)
    mgr.add_to_textual(clip_vec("x"), "p", "m")
    with pytest.raises(ValueError, match=r"No save path"):
        mgr.save()


# ---------------- RU-FM-18 ----------------


def test_ru_fm_18_round_trip_persistence(tmp_index_dir, clip_vec):
    """RU-FM-18: Round-trip persistence: save then build a fresh manager pointing at the same dir restores indices, metadata, and id_counters."""
    mgr1 = FAISSManager(dimension=512, index_path=tmp_index_dir, use_gpu=False)
    mgr1.add_to_textual(clip_vec("1"), "a", "m1")
    mgr1.add_to_textual(clip_vec("2"), "b", "m1")
    mgr1.add_to_visual(clip_vec("3"), "c", 0, "m1")
    mgr1.add_to_fused(clip_vec("4"), "d", 0, "m1")
    mgr1.save()

    mgr2 = FAISSManager(dimension=512, index_path=tmp_index_dir, use_gpu=False)
    assert mgr2.get_all_sizes() == mgr1.get_all_sizes()
    for t in IndexType:
        assert mgr2.metadata[t] == mgr1.metadata[t]
        assert mgr2.id_counters[t] == mgr1.id_counters[t]

    results = mgr2.search_textual(clip_vec("1"), top_k=2)
    assert results[0]["product_id"] == "a"
    assert abs(results[0]["score"] - 1.0) < 1e-4


# ---------------- RU-FM-19 ----------------


def test_ru_fm_19_load_empty_dir_no_raise(tmp_index_dir):
    """RU-FM-19: load() with no saved files leaves indices empty and does not raise."""
    sub = os.path.join(tmp_index_dir, "empty")
    os.makedirs(sub, exist_ok=True)
    mgr = FAISSManager(dimension=512, index_path=sub, use_gpu=False)
    assert mgr.get_all_sizes() == {"textual": 0, "visual": 0, "fused": 0}
    for t in IndexType:
        assert mgr.id_counters[t] == 0


# ---------------- RU-FM-20 ----------------


def test_ru_fm_20_clear_specific_index(faiss_manager_factory, clip_vec):
    """RU-FM-20: clear(index_type) resets only the specified index, preserving the others."""
    mgr = faiss_manager_factory(dimension=512)
    for i in range(2):
        mgr.add_to_textual(clip_vec(f"t{i}"), f"pt{i}", "m")
        mgr.add_to_visual(clip_vec(f"v{i}"), f"pv{i}", i, "m")
        mgr.add_to_fused(clip_vec(f"f{i}"), f"pf{i}", i, "m")
    mgr.clear(IndexType.TEXTUAL)
    assert mgr.get_all_sizes() == {"textual": 0, "visual": 2, "fused": 2}
    assert mgr.id_counters[IndexType.TEXTUAL] == 0
    assert mgr.id_counters[IndexType.VISUAL] == 2
    assert mgr.id_counters[IndexType.FUSED] == 2


# ---------------- RU-FM-21 ----------------


def test_ru_fm_21_clear_all(faiss_manager_factory, clip_vec):
    """RU-FM-21: clear() without argument resets every index and counter."""
    mgr = faiss_manager_factory(dimension=512)
    mgr.add_to_textual(clip_vec("a"), "p", "m")
    mgr.add_to_visual(clip_vec("b"), "p", 0, "m")
    mgr.add_to_fused(clip_vec("c"), "p", 0, "m")
    mgr.clear()
    assert mgr.get_all_sizes() == {"textual": 0, "visual": 0, "fused": 0}
    for t in IndexType:
        assert mgr.id_counters[t] == 0
        assert mgr.metadata[t] == []


# ---------------- RU-FM-22 ----------------


def test_ru_fm_22_search_empty_index(faiss_manager_factory, clip_vec):
    """RU-FM-22: Searching an empty index returns an empty list (no exception, no -1 IDs leaked)."""
    mgr = faiss_manager_factory(dimension=512)
    results = mgr.search_textual(clip_vec("0"), top_k=5)
    assert results == []


# ---------------- RU-FM-23 ----------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BAAI/bge-large-en-v1.5", "bge-large-en-v1.5"),
        ("Qwen/Qwen3-Embedding-8B", "Qwen3-Embedding-8B"),
        ("ViT-B/32", "ViT-B-32"),
        ("plainname", "plainname"),
    ],
)
def test_ru_fm_23_sanitize_model_name_patterns(raw, expected):
    """RU-FM-23: sanitize_model_name handles all three documented patterns: HF org-prefixed, ViT slash form, and plain."""
    assert sanitize_model_name(raw) == expected


# ---------------- RU-FM-24 ----------------


@pytest.mark.parametrize(
    "model,dim,expected",
    [
        ("BAAI/bge-large-en-v1.5", 1024, "bge-large-en-v1.5_1024_embeddings"),
        ("ViT-B/32", 512, "ViT-B-32_512_embeddings"),
        ("Qwen/Qwen3-Embedding-8B", 4096, "Qwen3-Embedding-8B_4096_embeddings"),
    ],
)
def test_ru_fm_24_make_folder_name(model, dim, expected):
    """RU-FM-24: make_folder_name composes sanitized model name with dimension and _embeddings suffix."""
    assert make_folder_name(model, dim) == expected


# ---------------- RU-FM-25 ----------------


def test_ru_fm_25_thread_safety_smoke(faiss_manager_factory, clip_vec):
    """RU-FM-25: Thread-safety smoke test: concurrent add_to_textual calls do not produce duplicate IDs or corrupt the counter."""
    mgr = faiss_manager_factory(dimension=512)
    n_threads = 8
    per_thread = 25
    results: list[int] = []
    results_lock = threading.Lock()

    def worker(thread_id: int):
        local_ids = []
        for i in range(per_thread):
            seed = f"t{thread_id}-i{i}"
            vid = mgr.add_to_textual(clip_vec(seed), f"p-{thread_id}-{i}", "m")
            local_ids.append(vid)
        with results_lock:
            results.extend(local_ids)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(results) == n_threads * per_thread
    assert len(set(results)) == n_threads * per_thread
    assert mgr.id_counters[IndexType.TEXTUAL] == n_threads * per_thread
    assert mgr.get_index_size(IndexType.TEXTUAL) == n_threads * per_thread
    assert len(mgr.metadata[IndexType.TEXTUAL]) == n_threads * per_thread


# ---------------- RU-FM-26 ----------------


def test_ru_fm_26_search_after_partial_removal(faiss_manager_factory, clip_vec):
    """RU-FM-26: Search after partial removal returns only surviving vectors with correct metadata."""
    mgr = faiss_manager_factory(dimension=512)
    for i in range(1, 5):
        mgr.add_to_textual(clip_vec(str(i)), f"p{i}", "m")
    mgr.remove_by_product_id(IndexType.TEXTUAL, "p2")

    results = mgr.search_textual(clip_vec("2"), top_k=4)
    assert len(results) <= 3
    for r in results:
        assert r["product_id"] != "p2"

    surviving_ids = {m["_vector_id"] for m in mgr.metadata[IndexType.TEXTUAL]}
    returned_ids = {r["vector_id"] for r in results}
    assert returned_ids.issubset(surviving_ids)
