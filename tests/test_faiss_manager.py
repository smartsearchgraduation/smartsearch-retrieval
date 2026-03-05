"""
Unit tests for FAISSManager.

Uses real FAISS in-memory indices (no mocking) to ensure correct
vector add / search / remove / save / load / clear behaviour.
"""

import os
import tempfile
import pytest

from vector_db.faiss_manager import FAISSManager, IndexType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_embedding(dim=512, val=0.1):
    """Return a constant embedding vector."""
    return [val] * dim


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestFAISSManagerInit:

    def test_init_default_dimension(self):
        """All three indices should be created with the default dimension."""
        mgr = FAISSManager(dimension=512)
        for idx_type in IndexType:
            assert mgr.dimensions[idx_type] == 512
            assert mgr.get_index_size(idx_type) == 0

    def test_init_custom_dimensions(self):
        """Per-index dimensions override the default."""
        dims = {"textual": 768, "visual": 512, "fused": 1024}
        mgr = FAISSManager(dimension=512, dimensions=dims)
        assert mgr.dimensions[IndexType.TEXTUAL] == 768
        assert mgr.dimensions[IndexType.VISUAL] == 512
        assert mgr.dimensions[IndexType.FUSED] == 1024


# ---------------------------------------------------------------------------
# Adding vectors
# ---------------------------------------------------------------------------

class TestFAISSManagerAdd:

    def test_add_to_textual(self):
        mgr = FAISSManager(dimension=512)
        vec_id = mgr.add_to_textual(
            embedding=_dummy_embedding(),
            product_id="prod_001",
            model_name="ViT-B/32",
        )
        assert vec_id == 0
        assert mgr.get_index_size(IndexType.TEXTUAL) == 1

    def test_add_to_visual(self):
        mgr = FAISSManager(dimension=512)
        vec_id = mgr.add_to_visual(
            embedding=_dummy_embedding(),
            product_id="prod_001",
            image_no=0,
            model_name="ViT-B/32",
        )
        assert vec_id == 0
        assert mgr.get_index_size(IndexType.VISUAL) == 1

    def test_add_to_fused(self):
        mgr = FAISSManager(dimension=512)
        vec_id = mgr.add_to_fused(
            embedding=_dummy_embedding(),
            product_id="prod_001",
            image_no=0,
            model_name="ViT-B/32",
        )
        assert vec_id == 0
        assert mgr.get_index_size(IndexType.FUSED) == 1


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------

class TestFAISSManagerSearch:

    def test_search_textual(self):
        mgr = FAISSManager(dimension=512)
        mgr.add_to_textual(_dummy_embedding(), "prod_001", "ViT-B/32")
        results = mgr.search_textual(_dummy_embedding(), top_k=5)
        assert len(results) == 1
        assert results[0]["product_id"] == "prod_001"
        assert results[0]["score"] > 0

    def test_search_visual(self):
        mgr = FAISSManager(dimension=512)
        mgr.add_to_visual(_dummy_embedding(), "prod_001", 0, "ViT-B/32")
        results = mgr.search_visual(_dummy_embedding(), top_k=5)
        assert len(results) == 1
        assert results[0]["image_no"] == 0

    def test_search_with_model_filter(self):
        mgr = FAISSManager(dimension=512)
        mgr.add_to_textual(_dummy_embedding(), "prod_001", "ViT-B/32")
        mgr.add_to_textual(_dummy_embedding(val=0.2), "prod_002", "other-model")

        results = mgr.search_textual(
            _dummy_embedding(), top_k=10, model_name="ViT-B/32"
        )
        product_ids = [r["product_id"] for r in results]
        assert "prod_001" in product_ids
        assert "prod_002" not in product_ids

    def test_search_empty_index(self):
        mgr = FAISSManager(dimension=512)
        results = mgr.search_textual(_dummy_embedding(), top_k=5)
        assert results == []

    def test_top_k_limit(self):
        mgr = FAISSManager(dimension=512)
        for i in range(20):
            mgr.add_to_textual(
                _dummy_embedding(val=0.1 + i * 0.01), f"prod_{i:03d}", "ViT-B/32"
            )
        results = mgr.search_textual(_dummy_embedding(), top_k=5)
        assert len(results) <= 5


# ---------------------------------------------------------------------------
# Removing vectors
# ---------------------------------------------------------------------------

class TestFAISSManagerRemove:

    def test_remove_by_product_id(self):
        mgr = FAISSManager(dimension=512)
        mgr.add_to_textual(_dummy_embedding(), "prod_001", "ViT-B/32")
        mgr.add_to_textual(_dummy_embedding(val=0.2), "prod_002", "ViT-B/32")

        removed = mgr.remove_by_product_id(IndexType.TEXTUAL, "prod_001")
        assert removed == 1
        assert mgr.get_index_size(IndexType.TEXTUAL) == 1

    def test_remove_product_from_all(self):
        mgr = FAISSManager(dimension=512)
        mgr.add_to_textual(_dummy_embedding(), "prod_001", "ViT-B/32")
        mgr.add_to_visual(_dummy_embedding(), "prod_001", 0, "ViT-B/32")
        mgr.add_to_fused(_dummy_embedding(), "prod_001", 0, "ViT-B/32")

        counts = mgr.remove_product_from_all("prod_001")
        assert counts == {"textual": 1, "visual": 1, "fused": 1}
        sizes = mgr.get_all_sizes()
        assert all(v == 0 for v in sizes.values())


# ---------------------------------------------------------------------------
# get_all_sizes
# ---------------------------------------------------------------------------

class TestFAISSManagerSizes:

    def test_get_all_sizes(self):
        mgr = FAISSManager(dimension=512)
        mgr.add_to_textual(_dummy_embedding(), "p1", "m")
        mgr.add_to_textual(_dummy_embedding(), "p2", "m")
        mgr.add_to_visual(_dummy_embedding(), "p1", 0, "m")

        sizes = mgr.get_all_sizes()
        assert sizes == {"textual": 2, "visual": 1, "fused": 0}


# ---------------------------------------------------------------------------
# Save / Load persistence
# ---------------------------------------------------------------------------

class TestFAISSManagerPersistence:

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate a manager
            mgr = FAISSManager(dimension=512, index_path=tmpdir)
            mgr.add_to_textual(_dummy_embedding(), "prod_001", "ViT-B/32")
            mgr.add_to_visual(_dummy_embedding(), "prod_001", 0, "ViT-B/32")
            mgr.save()

            # Create a new manager and load
            mgr2 = FAISSManager(dimension=512, index_path=tmpdir)
            assert mgr2.get_index_size(IndexType.TEXTUAL) == 1
            assert mgr2.get_index_size(IndexType.VISUAL) == 1

            # Search should still work
            results = mgr2.search_textual(_dummy_embedding(), top_k=5)
            assert len(results) == 1
            assert results[0]["product_id"] == "prod_001"


# ---------------------------------------------------------------------------
# Clearing indices
# ---------------------------------------------------------------------------

class TestFAISSManagerClear:

    def test_clear_single_index(self):
        mgr = FAISSManager(dimension=512)
        mgr.add_to_textual(_dummy_embedding(), "p1", "m")
        mgr.add_to_visual(_dummy_embedding(), "p1", 0, "m")

        mgr.clear(IndexType.TEXTUAL)
        assert mgr.get_index_size(IndexType.TEXTUAL) == 0
        assert mgr.get_index_size(IndexType.VISUAL) == 1  # untouched

    def test_clear_all(self):
        mgr = FAISSManager(dimension=512)
        mgr.add_to_textual(_dummy_embedding(), "p1", "m")
        mgr.add_to_visual(_dummy_embedding(), "p1", 0, "m")
        mgr.add_to_fused(_dummy_embedding(), "p1", 0, "m")

        mgr.clear()
        sizes = mgr.get_all_sizes()
        assert all(v == 0 for v in sizes.values())
