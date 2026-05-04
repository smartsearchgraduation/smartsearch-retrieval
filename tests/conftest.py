"""
Shared pytest fixtures for the Retrieval service test suite.

Conventions enforced by these fixtures (see tests/CONVENTIONS.md):
- Real in-memory faiss is used; faiss is never mocked.
- FAISS persistence paths always live under tempfile.TemporaryDirectory.
- Embedder/model layers are stubbed with deterministic, L2-normalized vectors.
- No deep-learning weights are ever loaded.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

# Make repo root importable as a package root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._helpers import (  # noqa: E402
    DIM_BGE,
    DIM_CLIP,
    DIM_DINOV3,
    DIM_MARQO,
    DIM_QWEN,
    StubFusedEmbedder,
    StubImageEmbedder,
    StubTextEmbedder,
    deterministic_vector,
)


# ---------- Filesystem ----------


@pytest.fixture
def tmp_index_dir() -> Iterator[str]:
    """Yield an isolated temporary directory for FAISS persistence."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def tmp_image_path(tmp_path: Path) -> str:
    """Create a tiny placeholder image file and return its path.

    The file content is not a real image; tests that need genuine image
    decoding should construct their own. This fixture exists for tests
    that only need a path that exists on disk (e.g. file-size checks).
    """
    p = tmp_path / "placeholder.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 64)  # 4-byte JPEG magic + filler
    return str(p)


# ---------- Deterministic vectors ----------


@pytest.fixture
def vec_factory():
    """Return ``deterministic_vector`` for ad-hoc vector generation in tests."""
    return deterministic_vector


@pytest.fixture
def clip_vec():
    return lambda seed: deterministic_vector(seed, DIM_CLIP)


@pytest.fixture
def bge_vec():
    return lambda seed: deterministic_vector(seed, DIM_BGE)


@pytest.fixture
def qwen_vec():
    return lambda seed: deterministic_vector(seed, DIM_QWEN)


@pytest.fixture
def marqo_vec():
    return lambda seed: deterministic_vector(seed, DIM_MARQO)


@pytest.fixture
def dinov3_vec():
    return lambda seed: deterministic_vector(seed, DIM_DINOV3)


# ---------- Stub embedders ----------


@pytest.fixture
def stub_text_embedder():
    return StubTextEmbedder()


@pytest.fixture
def stub_image_embedder():
    return StubImageEmbedder()


@pytest.fixture
def stub_fused_embedder():
    return StubFusedEmbedder()


# ---------- FAISS ----------


@pytest.fixture
def faiss_manager_factory(tmp_index_dir):
    """Return a callable that builds FAISSManager with a temp index_path.

    Usage:
        mgr = faiss_manager_factory(dimension=512)
        mgr = faiss_manager_factory(dimensions={"textual": 1024, "visual": 512, "fused": 512})
    """
    from vector_db.faiss_manager import FAISSManager

    def _factory(**kwargs):
        kwargs.setdefault("index_path", tmp_index_dir)
        kwargs.setdefault("use_gpu", False)
        return FAISSManager(**kwargs)

    return _factory


# ---------- Numerics ----------


@pytest.fixture(autouse=True)
def _numpy_print_options():
    """Stable numpy print options for assertion error messages."""
    with np.printoptions(precision=4, suppress=True):
        yield


# ---------- Routes: Flask client + manager-service stubs ----------


@pytest.fixture
def oversize_image_path(tmp_path: Path) -> str:
    """Create a >50MB file used to trigger validate_image_file_size."""
    p = tmp_path / "huge.jpg"
    # 50 MB + 1 byte
    size = 50 * 1024 * 1024 + 1
    with open(p, "wb") as f:
        f.seek(size - 1)
        f.write(b"\0")
    return str(p)


@pytest.fixture
def flask_client(tmp_index_dir, monkeypatch):
    """Yield a fresh Flask test client with isolated manager-service state.

    Resets the four module caches, points DATA_BASE_PATH at tmp_index_dir,
    initializes validation config so MAX_TOP_K/DEFAULT_TOP_K reflect config.
    """
    # Make sure config is loaded so MODEL_REGISTRY, MAX_TOP_K, etc. exist.
    from services import manager_service
    if not manager_service.MODEL_REGISTRY:
        manager_service.load_config()

    monkeypatch.setattr(manager_service, "DATA_BASE_PATH", tmp_index_dir)

    # Reset caches before
    manager_service._faiss_managers.clear()
    manager_service._textual_managers.clear()
    manager_service._visual_managers.clear()
    manager_service._fused_managers.clear()

    # Re-init validation config to match loaded values
    from utils import validation as _validation
    _validation.init_validation_config(
        manager_service.MAX_TOP_K, manager_service.DEFAULT_TOP_K
    )

    # Build app fresh
    from app import create_app
    app = create_app()
    # The freshly-created app re-imports manager_service; but DATA_BASE_PATH
    # is already monkeypatched on the same module object, so subsequent
    # get_faiss_manager calls will use tmp_index_dir.
    monkeypatch.setattr(manager_service, "DATA_BASE_PATH", tmp_index_dir)

    client = app.test_client()
    try:
        yield client
    finally:
        manager_service._faiss_managers.clear()
        manager_service._textual_managers.clear()
        manager_service._visual_managers.clear()
        manager_service._fused_managers.clear()


@pytest.fixture
def stub_managers(monkeypatch):
    """Patch TextModelManager / VisualModelManager / FusedModelManager in
    services.manager_service with deterministic fakes that honor each model's
    documented dimension via MODEL_REGISTRY.
    """
    from services import manager_service
    if not manager_service.MODEL_REGISTRY:
        manager_service.load_config()

    registry = manager_service.MODEL_REGISTRY

    def _dim_for(model_name: str) -> int:
        return registry.get(model_name, {}).get("dimension", 512)

    class FakeTextManager:
        def __init__(self, model_type=None, model_config=None):
            self.model_config = model_config or {}
            self.model_name = self.model_config.get("model_name", "stub-text")
            self.dimension = _dim_for(self.model_name)

        def get_embedding(self, text: str):
            return deterministic_vector(
                f"text::{self.model_name}::{text}", self.dimension
            )

        def get_document_embedding(self, text: str):
            return deterministic_vector(
                f"doc::{self.model_name}::{text}", self.dimension
            )

    class FakeVisualManager:
        def __init__(self, model_type=None, model_config=None):
            self.model_config = model_config or {}
            self.model_name = self.model_config.get("model_name", "stub-visual")
            self.dimension = _dim_for(self.model_name)

        def get_embedding(self, image_path: str):
            return deterministic_vector(
                f"image::{self.model_name}::{image_path}", self.dimension
            )

    class FakeFusedManager:
        def __init__(self, model_type=None, model_config=None):
            self.model_config = model_config or {}
            self.model_name = self.model_config.get("model_name", "stub-fused")
            self.dimension = _dim_for(self.model_name)
            self._fusion = ("weighted", 0.5)

        def set_fusion_method(self, method, text_weight=0.5):
            self._fusion = (method, text_weight)

        def get_embedding(self, text: str, image_path: str = None):
            if image_path is None:
                return deterministic_vector(
                    f"fused-text::{self.model_name}::{text}", self.dimension
                )
            return deterministic_vector(
                f"fused::{self.model_name}::{text}::{image_path}", self.dimension
            )

    monkeypatch.setattr(manager_service, "TextModelManager", FakeTextManager)
    monkeypatch.setattr(manager_service, "VisualModelManager", FakeVisualManager)
    monkeypatch.setattr(manager_service, "FusedModelManager", FakeFusedManager)

    return {
        "TextModelManager": FakeTextManager,
        "VisualModelManager": FakeVisualManager,
        "FusedModelManager": FakeFusedManager,
    }
