"""
Unit tests for the embedding model layer.

Covers:
- Model pools: CLIPModelPool, DINOv3ModelPool, OpenCLIPModelPool
- Managers:    TextModelManager, VisualModelManager, FusedModelManager
- Per-backend embedders: CLIP, BGE, Marqo, Qwen, DINOv3 (text/image/fused)

Mocking strategy:
- No real model weights are ever loaded.
- Pool tests inject fake `clip` / `open_clip` / `transformers` modules into
  `sys.modules` BEFORE calling the pool's `.get(...)` method, so the inner
  `import` resolves to the stub.
- Embedder tests patch `CLIPModelPool.get` / `DINOv3ModelPool.get` /
  `OpenCLIPModelPool.get` via monkeypatch and return stub objects whose
  encode_* methods produce deterministic L2-normalized torch tensors of the
  documented dimension.
- `PIL.Image.open` is patched where the production code tries to open the
  placeholder file from `tmp_image_path` (it isn't a real JPEG).
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests._helpers import (
    DIM_BGE,
    DIM_CLIP,
    DIM_DINOV3,
    DIM_MARQO,
    DIM_QWEN,
    deterministic_vector,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _stub_tensor(
    seed: str, dim: int, batch: int = 1, normalize: bool = True
) -> torch.Tensor:
    """Deterministic torch tensor of shape (batch, dim), optionally L2-normalized."""
    vec = np.array(deterministic_vector(seed, dim), dtype=np.float32)
    if not normalize:
        # Multiply to break unit norm
        vec = vec * 3.0
    rows = np.tile(vec, (batch, 1))
    return torch.tensor(rows, dtype=torch.float32)


def _make_fake_clip_module(load_counter: dict, raise_exc: Exception | None = None):
    """Create a fake `clip` module with `load(name, device=...)` and `tokenize(...)`."""
    mod = types.ModuleType("clip")

    def load(name, device=None, **kwargs):
        load_counter["count"] = load_counter.get("count", 0) + 1
        if raise_exc is not None:
            raise raise_exc
        model = MagicMock(name=f"clip-model-{load_counter['count']}")
        model.eval.return_value = model
        # encode_text / encode_image return deterministic L2-normalized tensors
        model.encode_text.side_effect = lambda toks: _stub_tensor(
            f"clip-text::{load_counter['count']}", DIM_CLIP
        )
        model.encode_image.side_effect = lambda t: _stub_tensor(
            f"clip-image::{load_counter['count']}", DIM_CLIP
        )
        preprocess = MagicMock(name=f"clip-preprocess-{load_counter['count']}")
        preprocess.side_effect = lambda img: torch.zeros(3, 224, 224)
        return model, preprocess

    def tokenize(texts, truncate=True):
        # Just return a tensor; embedders move it to device with .to(...)
        return torch.zeros(
            len(texts) if isinstance(texts, list) else 1, 77, dtype=torch.long
        )

    mod.load = load
    mod.tokenize = tokenize
    return mod


def _make_fake_open_clip_module(load_counter: dict, raise_exc: Exception | None = None):
    mod = types.ModuleType("open_clip")

    def create_model_and_transforms(name):
        load_counter["count"] = load_counter.get("count", 0) + 1
        if raise_exc is not None:
            raise raise_exc
        model = MagicMock(name=f"oc-model-{load_counter['count']}")
        model.eval.return_value = model
        model.to.return_value = model
        model.encode_text.side_effect = lambda toks, normalize=False: _stub_tensor(
            f"oc-text::{load_counter['count']}", DIM_MARQO
        )
        model.encode_image.side_effect = lambda t, normalize=False: _stub_tensor(
            f"oc-image::{load_counter['count']}", DIM_MARQO
        )
        preprocess = MagicMock(name=f"oc-preprocess-{load_counter['count']}")
        preprocess.side_effect = lambda img: torch.zeros(3, 224, 224)
        return model, None, preprocess

    def get_tokenizer(name):
        tok = MagicMock(name="oc-tokenizer")
        tok.side_effect = lambda texts: _FakeTensorWithTo(
            torch.zeros(len(texts), 77, dtype=torch.long)
        )
        return tok

    mod.create_model_and_transforms = create_model_and_transforms
    mod.get_tokenizer = get_tokenizer
    return mod


def _make_fake_transformers_module(
    load_counter: dict,
    last_hidden_dim: int = DIM_BGE,
    raise_in_model: Exception | None = None,
):
    """Fake `transformers` module with AutoTokenizer/AutoModel/AutoImageProcessor."""
    mod = types.ModuleType("transformers")

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            inst = cls()
            return inst

        def __call__(
            self,
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ):
            seq_len = 4
            return _FakeTokenizerOutput(
                {
                    "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
                    "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
                }
            )

    class _FakeModelOutput:
        def __init__(self, last_hidden_state, pooler_output=None):
            self.last_hidden_state = last_hidden_state
            self.pooler_output = pooler_output

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            load_counter["count"] = load_counter.get("count", 0) + 1
            if raise_in_model is not None:
                raise raise_in_model
            inst = cls()
            return inst

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            seq_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 4
            # Build a deterministic last_hidden_state of shape (1, seq_len, dim)
            base_vec = np.array(
                deterministic_vector("transformers-fake", last_hidden_dim),
                dtype=np.float32,
            )
            lhs = torch.tensor(np.tile(base_vec, (1, seq_len, 1)), dtype=torch.float32)
            pooler = torch.tensor(np.tile(base_vec, (1, 1)), dtype=torch.float32)
            return _FakeModelOutput(last_hidden_state=lhs, pooler_output=pooler)

    class _FakeImageProcessor:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return cls()

        def __call__(self, images=None, return_tensors="pt"):
            return _FakeTokenizerOutput(
                {"pixel_values": torch.zeros(1, 3, 224, 224, dtype=torch.float32)}
            )

    mod.AutoTokenizer = _FakeTokenizer
    mod.AutoModel = _FakeModel
    mod.AutoImageProcessor = _FakeImageProcessor
    return mod


class _FakeTensorWithTo:
    """Wraps a tensor so .to(device) returns the same wrapper (works with `.to(...)`)."""

    def __init__(self, t):
        self._t = t

    def to(self, device):
        return self._t


class _FakeTokenizerOutput(dict):
    """Dict subclass so `**inputs` works; supports .to(device) returning itself."""

    def to(self, device):
        return self


# --------------------------------------------------------------------------- #
# Autouse fixture: clear pools and clean fake modules around every test       #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _clear_pools_and_modules():
    """Clear all model pools and remove fake modules before AND after every test."""
    from models.clip_model_pool import CLIPModelPool
    from models.dinov3_model_pool import DINOv3ModelPool
    from models.open_clip_model_pool import OpenCLIPModelPool

    fake_module_names = (
        "clip",
        "open_clip",
    )  # transformers is real; don't pop globally
    saved = {n: sys.modules.get(n, None) for n in fake_module_names}

    CLIPModelPool.clear()
    DINOv3ModelPool.clear()
    OpenCLIPModelPool.clear()
    for n in fake_module_names:
        sys.modules.pop(n, None)
    yield
    CLIPModelPool.clear()
    DINOv3ModelPool.clear()
    OpenCLIPModelPool.clear()
    for n, v in saved.items():
        if v is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = v


# =========================================================================== #
# CLIPModelPool                                                               #
# =========================================================================== #


def test_ru_em_01_clip_pool_get_caches_per_key(monkeypatch):
    """RU-EM-01: CLIPModelPool.get caches per (model_name, device)."""
    from models.clip_model_pool import CLIPModelPool

    counter = {"count": 0}
    fake_clip = _make_fake_clip_module(counter)
    monkeypatch.setitem(sys.modules, "clip", fake_clip)

    r1 = CLIPModelPool.get("ViT-B/32", "cpu")
    r2 = CLIPModelPool.get("ViT-B/32", "cpu")

    assert r1 is r2
    assert counter["count"] == 1


def test_ru_em_02_clip_pool_different_device_separate_entry(monkeypatch):
    """RU-EM-02: Different device produces separate cache entry."""
    from models.clip_model_pool import CLIPModelPool

    counter = {"count": 0}
    monkeypatch.setitem(sys.modules, "clip", _make_fake_clip_module(counter))

    r_cpu = CLIPModelPool.get("ViT-B/32", "cpu")
    r_cuda = CLIPModelPool.get("ViT-B/32", "cuda")

    assert r_cpu is not r_cuda
    assert counter["count"] == 2


def test_ru_em_03_clip_pool_different_model_name_separate_entry(monkeypatch):
    """RU-EM-03: Different model_name produces separate cache entry."""
    from models.clip_model_pool import CLIPModelPool

    counter = {"count": 0}
    monkeypatch.setitem(sys.modules, "clip", _make_fake_clip_module(counter))

    r_b32 = CLIPModelPool.get("ViT-B/32", "cpu")
    r_l14 = CLIPModelPool.get("ViT-L/14", "cpu")

    assert r_b32 is not r_l14
    assert counter["count"] == 2


def test_ru_em_04_clip_pool_clear_forces_reload(monkeypatch):
    """RU-EM-04: clear() empties the cache; subsequent get reloads."""
    from models.clip_model_pool import CLIPModelPool

    counter = {"count": 0}
    monkeypatch.setitem(sys.modules, "clip", _make_fake_clip_module(counter))

    CLIPModelPool.get("ViT-B/32", "cpu")
    CLIPModelPool.clear()
    CLIPModelPool.get("ViT-B/32", "cpu")

    assert counter["count"] == 2


def test_ru_em_05_clip_pool_import_error(monkeypatch):
    """RU-EM-05: Missing `clip` package raises ImportError."""
    from models.clip_model_pool import CLIPModelPool

    monkeypatch.setitem(sys.modules, "clip", None)

    with pytest.raises(ImportError, match=r"CLIP is not installed"):
        CLIPModelPool.get("ViT-B/32", "cpu")


def test_ru_em_06_clip_pool_load_failure_runtimeerror(monkeypatch):
    """RU-EM-06: Generic load failure is wrapped in RuntimeError."""
    from models.clip_model_pool import CLIPModelPool

    counter = {"count": 0}
    monkeypatch.setitem(
        sys.modules,
        "clip",
        _make_fake_clip_module(counter, raise_exc=Exception("boom")),
    )

    with pytest.raises(RuntimeError, match=r"Failed to load CLIP model"):
        CLIPModelPool.get("ViT-B/32", "cpu")


# =========================================================================== #
# DINOv3ModelPool                                                             #
# =========================================================================== #


def test_ru_em_07_dinov3_pool_get_caches_per_key(monkeypatch):
    """RU-EM-07: DINOv3ModelPool.get caches per (model_name, device)."""
    from models.dinov3_model_pool import DINOv3ModelPool

    counter = {"count": 0}
    fake = _make_fake_transformers_module(counter, last_hidden_dim=DIM_DINOV3)
    monkeypatch.setitem(sys.modules, "transformers", fake)

    r1 = DINOv3ModelPool.get("facebook/dinov3-vit7b16-pretrain-lvd1689m", "cpu")
    r2 = DINOv3ModelPool.get("facebook/dinov3-vit7b16-pretrain-lvd1689m", "cpu")

    assert r1 is r2
    assert counter["count"] == 1


def test_ru_em_08_dinov3_pool_clear_reloads(monkeypatch):
    """RU-EM-08: DINOv3ModelPool reloads after clear()."""
    from models.dinov3_model_pool import DINOv3ModelPool

    counter = {"count": 0}
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _make_fake_transformers_module(counter, last_hidden_dim=DIM_DINOV3),
    )

    DINOv3ModelPool.get("facebook/dinov3-vit7b16-pretrain-lvd1689m", "cpu")
    DINOv3ModelPool.clear()
    DINOv3ModelPool.get("facebook/dinov3-vit7b16-pretrain-lvd1689m", "cpu")

    assert counter["count"] == 2


def test_ru_em_09_dinov3_pool_import_error(monkeypatch):
    """RU-EM-09: Missing transformers raises ImportError."""
    from models.dinov3_model_pool import DINOv3ModelPool

    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(ImportError, match=r"transformers is not installed"):
        DINOv3ModelPool.get("facebook/dinov3-vit7b16-pretrain-lvd1689m", "cpu")


def test_ru_em_10_dinov3_pool_load_failure_runtimeerror(monkeypatch):
    """RU-EM-10: Generic DINOv3 load failure is wrapped in RuntimeError."""
    from models.dinov3_model_pool import DINOv3ModelPool

    counter = {"count": 0}
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _make_fake_transformers_module(
            counter, last_hidden_dim=DIM_DINOV3, raise_in_model=Exception("nope")
        ),
    )

    with pytest.raises(RuntimeError, match=r"Failed to load DINOv3 model"):
        DINOv3ModelPool.get("facebook/dinov3-vit7b16-pretrain-lvd1689m", "cpu")


# =========================================================================== #
# OpenCLIPModelPool                                                           #
# =========================================================================== #


def test_ru_em_11_open_clip_pool_get_caches_per_key(monkeypatch):
    """RU-EM-11: OpenCLIPModelPool.get caches per (model_name, device)."""
    from models.open_clip_model_pool import OpenCLIPModelPool

    counter = {"count": 0}
    monkeypatch.setitem(sys.modules, "open_clip", _make_fake_open_clip_module(counter))

    r1 = OpenCLIPModelPool.get("Marqo/marqo-ecommerce-embeddings-L", "cpu")
    r2 = OpenCLIPModelPool.get("Marqo/marqo-ecommerce-embeddings-L", "cpu")

    assert r1 is r2
    assert counter["count"] == 1


def test_ru_em_12_open_clip_pool_different_device_separate_entry(monkeypatch):
    """RU-EM-12: Different device produces separate cache entry for OpenCLIP pool."""
    from models.open_clip_model_pool import OpenCLIPModelPool

    counter = {"count": 0}
    monkeypatch.setitem(sys.modules, "open_clip", _make_fake_open_clip_module(counter))

    r_cpu = OpenCLIPModelPool.get("Marqo/marqo-ecommerce-embeddings-L", "cpu")
    r_cuda = OpenCLIPModelPool.get("Marqo/marqo-ecommerce-embeddings-L", "cuda")

    assert r_cpu is not r_cuda
    assert counter["count"] == 2


def test_ru_em_13_open_clip_pool_import_error(monkeypatch):
    """RU-EM-13: Missing open_clip raises ImportError."""
    from models.open_clip_model_pool import OpenCLIPModelPool

    monkeypatch.setitem(sys.modules, "open_clip", None)

    with pytest.raises(ImportError, match=r"open_clip is not installed"):
        OpenCLIPModelPool.get("Marqo/marqo-ecommerce-embeddings-L", "cpu")


def test_ru_em_14_open_clip_pool_load_failure_runtimeerror(monkeypatch):
    """RU-EM-14: Generic OpenCLIP load failure is wrapped in RuntimeError."""
    from models.open_clip_model_pool import OpenCLIPModelPool

    counter = {"count": 0}
    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        _make_fake_open_clip_module(counter, raise_exc=Exception("oops")),
    )

    with pytest.raises(RuntimeError, match=r"Failed to load OpenCLIP model"):
        OpenCLIPModelPool.get("Marqo/marqo-ecommerce-embeddings-L", "cpu")


# =========================================================================== #
# Stub builders for embedder-level tests                                      #
# =========================================================================== #


def _patch_clip_pool_for_embedders(
    monkeypatch, text_seed="clip-text", image_seed="clip-image"
):
    """Patch CLIPModelPool.get + inject fake `clip` module so embedders work."""
    from models import clip_model_pool

    model = MagicMock(name="clip-stub-model")
    model.encode_text.side_effect = lambda toks: _stub_tensor(text_seed, DIM_CLIP)
    model.encode_image.side_effect = lambda t: _stub_tensor(image_seed, DIM_CLIP)
    preprocess = MagicMock(name="clip-stub-preprocess")
    preprocess.side_effect = lambda img: torch.zeros(3, 224, 224)

    monkeypatch.setattr(
        clip_model_pool.CLIPModelPool,
        "get",
        classmethod(lambda cls, name, device: (model, preprocess)),
    )

    # Inject a minimal fake `clip` module for `clip.tokenize` calls inside embedders.
    fake_clip = types.ModuleType("clip")
    fake_clip.tokenize = lambda texts, truncate=True: torch.zeros(
        len(texts) if isinstance(texts, list) else 1, 77, dtype=torch.long
    )
    monkeypatch.setitem(sys.modules, "clip", fake_clip)

    return model, preprocess


def _patch_open_clip_pool_for_embedders(
    monkeypatch, text_seed="oc-text", image_seed="oc-image"
):
    from models import open_clip_model_pool

    model = MagicMock(name="oc-stub-model")
    model.encode_text.side_effect = lambda toks, normalize=False: _stub_tensor(
        text_seed, DIM_MARQO
    )
    model.encode_image.side_effect = lambda t, normalize=False: _stub_tensor(
        image_seed, DIM_MARQO
    )
    preprocess = MagicMock(name="oc-stub-preprocess")
    preprocess.side_effect = lambda img: torch.zeros(3, 224, 224)
    tokenizer = MagicMock(name="oc-stub-tokenizer")
    tokenizer.side_effect = lambda texts: _FakeTensorWithTo(
        torch.zeros(len(texts) if isinstance(texts, list) else 1, 77, dtype=torch.long)
    )

    monkeypatch.setattr(
        open_clip_model_pool.OpenCLIPModelPool,
        "get",
        classmethod(lambda cls, name, device: (model, preprocess, tokenizer)),
    )
    return model, preprocess, tokenizer


def _patch_dinov3_pool_for_embedders(monkeypatch):
    from models import dinov3_model_pool

    model = MagicMock(name="dinov3-stub-model")

    class _Out:
        def __init__(self, pooler):
            self.pooler_output = pooler
            self.last_hidden_state = pooler.unsqueeze(1)

    def _call(**inputs):
        pooler = _stub_tensor("dinov3-pooler", DIM_DINOV3)
        return _Out(pooler)

    model.side_effect = _call

    processor = MagicMock(name="dinov3-stub-processor")
    processor.side_effect = (
        lambda images=None, return_tensors="pt": _FakeTokenizerOutput(
            {"pixel_values": torch.zeros(1, 3, 224, 224)}
        )
    )

    monkeypatch.setattr(
        dinov3_model_pool.DINOv3ModelPool,
        "get",
        classmethod(lambda cls, name, device: (model, processor)),
    )
    return model, processor


def _patch_transformers_for_bge(monkeypatch):
    """Inject fake transformers module sized for BGE (1024-d)."""
    counter = {"count": 0}
    fake = _make_fake_transformers_module(counter, last_hidden_dim=DIM_BGE)
    monkeypatch.setitem(sys.modules, "transformers", fake)


def _patch_transformers_for_qwen(monkeypatch):
    counter = {"count": 0}
    fake = _make_fake_transformers_module(counter, last_hidden_dim=DIM_QWEN)
    monkeypatch.setitem(sys.modules, "transformers", fake)


def _patch_pil_open_with_dummy(monkeypatch):
    """Patch PIL.Image.open globally to return a dummy RGB image (placeholder fixture isn't a real JPEG)."""
    from PIL import Image as _PILImage

    real_new = _PILImage.new

    def fake_open(path, *args, **kwargs):
        return real_new("RGB", (4, 4), color="red")

    monkeypatch.setattr(_PILImage, "open", fake_open)


# =========================================================================== #
# TextModelManager                                                            #
# =========================================================================== #


def test_ru_em_15_text_manager_clip(monkeypatch):
    """RU-EM-15: model_type='clip' instantiates CLIPTextEmbedder; embeds 512-d."""
    from models.textual_models.text_model_manager import TextModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = TextModelManager(model_type="clip", model_config={"device": "cpu"})

    assert manager.model.__class__.__name__ == "CLIPTextEmbedder"
    out = manager.get_embedding("hello")
    assert isinstance(out, list)
    assert len(out) == DIM_CLIP
    assert all(isinstance(x, float) for x in out)


def test_ru_em_16_text_manager_bge(monkeypatch):
    """RU-EM-16: model_type='bge' instantiates BGEBaseEmbedder; output 1024-d L2-normalized."""
    from models.textual_models.text_model_manager import TextModelManager

    _patch_transformers_for_bge(monkeypatch)
    manager = TextModelManager(model_type="bge", model_config={"device": "cpu"})

    assert manager.model.__class__.__name__ == "BGEBaseEmbedder"
    out = manager.get_embedding("text")
    assert len(out) == DIM_BGE
    assert abs(np.linalg.norm(out) - 1.0) < 1e-4


def test_ru_em_17_text_manager_marqo(monkeypatch):
    """RU-EM-17: model_type='marqo' instantiates MarqoTextEmbedder; output 1024-d."""
    from models.textual_models.text_model_manager import TextModelManager

    _patch_open_clip_pool_for_embedders(monkeypatch)
    manager = TextModelManager(model_type="marqo", model_config={"device": "cpu"})

    assert manager.model.__class__.__name__ == "MarqoTextEmbedder"
    out = manager.get_embedding("hello")
    assert len(out) == DIM_MARQO


def test_ru_em_18_text_manager_qwen(monkeypatch):
    """RU-EM-18: model_type='qwen' instantiates Qwen8BEmbedder; 4096-d L2-normalized."""
    from models.textual_models.text_model_manager import TextModelManager

    _patch_transformers_for_qwen(monkeypatch)
    manager = TextModelManager(model_type="qwen", model_config={"device": "cpu"})

    assert manager.model.__class__.__name__ == "Qwen8BEmbedder"
    out = manager.get_embedding("query")
    assert len(out) == DIM_QWEN
    assert abs(np.linalg.norm(out) - 1.0) < 1e-4


def test_ru_em_19_text_manager_unknown_type():
    """RU-EM-19: Unknown model_type string raises ValueError."""
    from models.textual_models.text_model_manager import TextModelManager

    with pytest.raises(ValueError, match=r"Unknown model type"):
        TextModelManager(model_type="banana")


def test_ru_em_20_text_manager_non_string_type():
    """RU-EM-20: Non-string non-enum model_type raises TypeError."""
    from models.textual_models.text_model_manager import TextModelManager

    with pytest.raises(TypeError):
        TextModelManager(model_type=123)


def test_ru_em_21_text_manager_get_embeddings_batch(monkeypatch):
    """RU-EM-21: get_embeddings([list]) delegates to embed_documents and returns N vectors."""
    from models.textual_models.text_model_manager import TextModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = TextModelManager(model_type="clip", model_config={"device": "cpu"})

    out = manager.get_embeddings(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == DIM_CLIP for v in out)


def test_ru_em_22_text_manager_embed_product(monkeypatch):
    """RU-EM-22: embed_product joins fields and produces a 512-d vector."""
    from models.textual_models.text_model_manager import TextModelManager

    captured = {"texts": []}

    fake_clip = types.ModuleType("clip")

    def tokenize(texts, truncate=True):
        captured["texts"].extend(texts if isinstance(texts, list) else [texts])
        return torch.zeros(
            len(texts) if isinstance(texts, list) else 1, 77, dtype=torch.long
        )

    fake_clip.tokenize = tokenize
    monkeypatch.setitem(sys.modules, "clip", fake_clip)

    from models import clip_model_pool

    model = MagicMock()
    model.encode_text.side_effect = lambda toks: _stub_tensor("clip-text", DIM_CLIP)
    preprocess = MagicMock()
    monkeypatch.setattr(
        clip_model_pool.CLIPModelPool,
        "get",
        classmethod(lambda cls, name, device: (model, preprocess)),
    )

    manager = TextModelManager(model_type="clip", model_config={"device": "cpu"})
    out = manager.embed_product({"name": "X", "description": "Y", "tags": ["t1", "t2"]})

    assert len(out) == DIM_CLIP
    combined = " ".join(captured["texts"])
    assert "X" in combined and "Y" in combined and "t1" in combined and "t2" in combined


def test_ru_em_23_text_manager_get_model_info(monkeypatch):
    """RU-EM-23: get_model_info returns a dict with expected keys/values."""
    from models.textual_models.text_model_manager import TextModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = TextModelManager(model_type="clip", model_config={"device": "cpu"})

    info = manager.get_model_info()
    for k in ("model_type", "model_config", "embedding_dimension", "model_class"):
        assert k in info
    assert info["model_type"] == "clip"
    assert info["embedding_dimension"] == DIM_CLIP
    assert info["model_class"] == "CLIPTextEmbedder"


# =========================================================================== #
# VisualModelManager                                                          #
# =========================================================================== #


def test_ru_em_24_visual_manager_clip(monkeypatch, tmp_image_path):
    """RU-EM-24: model_type='clip' instantiates CLIPImageEmbedder; 512-d L2-normalized."""
    from models.visual_models.visual_model_manager import VisualModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)

    manager = VisualModelManager("clip", {"device": "cpu"})
    assert manager.model.__class__.__name__ == "CLIPImageEmbedder"

    out = manager.get_embedding(tmp_image_path)
    assert len(out) == DIM_CLIP
    assert abs(np.linalg.norm(out) - 1.0) < 1e-4


def test_ru_em_25_visual_manager_dinov3(monkeypatch, tmp_image_path):
    """RU-EM-25: model_type='dinov3' instantiates DINOv3ImageEmbedder; 4096-d L2-normalized."""
    from models.visual_models.visual_model_manager import VisualModelManager

    _patch_dinov3_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)

    manager = VisualModelManager("dinov3", {"device": "cpu"})
    assert manager.model.__class__.__name__ == "DINOv3ImageEmbedder"

    out = manager.get_embedding(tmp_image_path)
    assert len(out) == DIM_DINOV3
    assert abs(np.linalg.norm(out) - 1.0) < 1e-4


def test_ru_em_26_visual_manager_marqo(monkeypatch, tmp_image_path):
    """RU-EM-26: model_type='marqo' instantiates MarqoImageEmbedder; 1024-d."""
    from models.visual_models.visual_model_manager import VisualModelManager

    _patch_open_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)

    manager = VisualModelManager("marqo", {"device": "cpu"})
    assert manager.model.__class__.__name__ == "MarqoImageEmbedder"

    out = manager.get_embedding(tmp_image_path)
    assert len(out) == DIM_MARQO


def test_ru_em_27_visual_manager_unknown_type():
    """RU-EM-27: Unknown visual model type raises ValueError."""
    from models.visual_models.visual_model_manager import VisualModelManager

    with pytest.raises(ValueError, match=r"Unknown model type"):
        VisualModelManager("hologram")


def test_ru_em_28_visual_manager_relative_path_rejected(monkeypatch):
    """RU-EM-28: get_embedding rejects relative path with ValueError."""
    from models.visual_models.visual_model_manager import VisualModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = VisualModelManager("clip", {"device": "cpu"})

    with pytest.raises(ValueError, match=r"absolute"):
        manager.get_embedding("relative/path.jpg")


def test_ru_em_29_visual_manager_missing_file_raises(monkeypatch):
    """RU-EM-29: Absolute path that doesn't exist raises FileNotFoundError."""
    from models.visual_models.visual_model_manager import VisualModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = VisualModelManager("clip", {"device": "cpu"})

    if os.name == "nt":
        bogus = "C:\\nonexistent\\xyz.jpg"
    else:
        bogus = "/nonexistent/xyz.jpg"

    with pytest.raises(FileNotFoundError):
        manager.get_embedding(bogus)


def test_ru_em_30_visual_manager_embed_product_image_requires_key(monkeypatch):
    """RU-EM-30: embed_product_image requires 'image_path' key."""
    from models.visual_models.visual_model_manager import VisualModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = VisualModelManager("clip", {"device": "cpu"})

    with pytest.raises(ValueError, match=r"image_path"):
        manager.embed_product_image({"name": "x"})


# =========================================================================== #
# FusedModelManager                                                           #
# =========================================================================== #


def test_ru_em_31_fused_manager_clip_default_average(monkeypatch, tmp_image_path):
    """RU-EM-31: 'clip' fused, default average fusion → 512-d L2-normalized."""
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)

    manager = FusedModelManager("clip", {"device": "cpu"})
    assert manager.model.__class__.__name__ == "CLIPFusedEmbedder"
    assert manager.model.fusion_method == "average"

    out = manager.get_embedding("hello", tmp_image_path)
    assert len(out) == DIM_CLIP
    assert abs(np.linalg.norm(out) - 1.0) < 1e-4


def test_ru_em_32_fused_manager_marqo(monkeypatch, tmp_image_path):
    """RU-EM-32: 'marqo' fused → 1024-d L2-normalized."""
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_open_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)

    manager = FusedModelManager("marqo", {"device": "cpu"})
    assert manager.model.__class__.__name__ == "MarqoFusedEmbedder"

    out = manager.get_embedding("hello", tmp_image_path)
    assert len(out) == DIM_MARQO
    assert abs(np.linalg.norm(out) - 1.0) < 1e-4


def test_ru_em_33_fused_manager_unknown_type():
    """RU-EM-33: Unknown fused model type raises ValueError."""
    from models.fused_models.fused_model_manager import FusedModelManager

    with pytest.raises(ValueError, match=r"Unknown model type"):
        FusedModelManager("xyz")


def test_ru_em_34_fused_manager_concat_doubles_dim(monkeypatch, tmp_image_path):
    """RU-EM-34: set_fusion_method('concat') doubles output dimension."""
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)

    manager = FusedModelManager("clip", {"device": "cpu"})
    manager.set_fusion_method("concat")
    out = manager.get_embedding("hello", tmp_image_path)

    assert len(out) == 2 * DIM_CLIP


def test_ru_em_35_fused_manager_set_weighted(monkeypatch):
    """RU-EM-35: set_fusion_method('weighted', text_weight=0.7) updates weights and config."""
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_clip_pool_for_embedders(monkeypatch)

    manager = FusedModelManager("clip", {"device": "cpu"})
    manager.set_fusion_method("weighted", text_weight=0.7)

    assert manager.model.text_weight == pytest.approx(0.7, abs=1e-9)
    assert manager.model.image_weight == pytest.approx(0.3, abs=1e-9)
    assert manager.model_config["fusion_method"] == "weighted"
    assert manager.model_config["text_weight"] == 0.7


def test_ru_em_36_fused_manager_unknown_fusion_method(monkeypatch):
    """RU-EM-36: Unknown fusion method raises ValueError."""
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_clip_pool_for_embedders(monkeypatch)

    manager = FusedModelManager("clip", {"device": "cpu"})
    with pytest.raises(ValueError, match=r"Unknown fusion method"):
        manager.set_fusion_method("xor")


def test_ru_em_37_fused_manager_individual_embeddings(monkeypatch, tmp_image_path):
    """RU-EM-37: get_individual_embeddings returns two distinct 512-d vectors."""
    from models.fused_models.fused_model_manager import FusedModelManager

    # Use distinct seeds for text vs image so the two vectors differ.
    _patch_clip_pool_for_embedders(
        monkeypatch, text_seed="distinct-text-seed", image_seed="distinct-image-seed"
    )
    _patch_pil_open_with_dummy(monkeypatch)

    manager = FusedModelManager("clip", {"device": "cpu"})
    t_emb, i_emb = manager.get_individual_embeddings("hello", tmp_image_path)

    assert len(t_emb) == DIM_CLIP
    assert len(i_emb) == DIM_CLIP
    assert t_emb != i_emb


def test_ru_em_38_fused_manager_embed_product_validations(monkeypatch, tmp_image_path):
    """RU-EM-38: embed_product requires image_path and at least one text field."""
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)

    manager = FusedModelManager("clip", {"device": "cpu"})

    with pytest.raises(ValueError, match=r"at least one text field"):
        manager.embed_product({"image_path": tmp_image_path})

    with pytest.raises(ValueError, match=r"image_path"):
        manager.embed_product({"name": "x"})


# =========================================================================== #
# Embedder L2 normalization sanity                                            #
# =========================================================================== #


def test_ru_em_39_clip_fused_average_unit_norm(monkeypatch, tmp_image_path):
    """RU-EM-39: CLIPFusedEmbedder._fuse_embeddings('average') yields unit-norm output."""
    from models.fused_models.clip_fused_embedder import CLIPFusedEmbedder

    # Use stubs that return non-unit-norm vectors; the embedder's _get_*_embedding
    # normalizes internally before fusion, so the fused result must still be unit-norm.
    from models import clip_model_pool

    model = MagicMock()
    model.encode_text.side_effect = lambda toks: _stub_tensor(
        "raw-text", DIM_CLIP, normalize=False
    )
    model.encode_image.side_effect = lambda t: _stub_tensor(
        "raw-image", DIM_CLIP, normalize=False
    )
    preprocess = MagicMock()
    preprocess.side_effect = lambda img: torch.zeros(3, 224, 224)
    monkeypatch.setattr(
        clip_model_pool.CLIPModelPool,
        "get",
        classmethod(lambda cls, name, device: (model, preprocess)),
    )

    fake_clip = types.ModuleType("clip")
    fake_clip.tokenize = lambda texts, truncate=True: torch.zeros(
        len(texts) if isinstance(texts, list) else 1, 77, dtype=torch.long
    )
    monkeypatch.setitem(sys.modules, "clip", fake_clip)
    _patch_pil_open_with_dummy(monkeypatch)

    embedder = CLIPFusedEmbedder(fusion_method="average", device="cpu")
    result = embedder.embed_text_and_image("text", tmp_image_path)

    assert abs(np.linalg.norm(result) - 1.0) < 1e-5


def test_ru_em_40_marqo_fused_weighted_unit_norm(monkeypatch, tmp_image_path):
    """RU-EM-40: MarqoFusedEmbedder('weighted', 0.6) yields 1024-d unit-norm output."""
    from models.fused_models.marqo_fused_embedder import MarqoFusedEmbedder

    _patch_open_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)

    embedder = MarqoFusedEmbedder(
        fusion_method="weighted", text_weight=0.6, device="cpu"
    )
    result = embedder.embed_text_and_image("hello", tmp_image_path)

    assert len(result) == DIM_MARQO
    assert abs(np.linalg.norm(result) - 1.0) < 1e-5


# =========================================================================== #
# Expansion round 2 — RU-EM-41..RU-EM-83                                      #
# =========================================================================== #


# --- FusedModelManager extra coverage ------------------------------------- #


def test_ru_em_41_fused_manager_accepts_enum(monkeypatch):
    """RU-EM-41: FusedModelManager accepts FusedModelType enum directly."""
    from models.fused_models.fused_model_manager import (
        FusedModelManager,
        FusedModelType,
    )

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = FusedModelManager(
        model_type=FusedModelType.CLIP, model_config={"device": "cpu"}
    )
    assert manager.model_type is FusedModelType.CLIP
    assert manager.model.__class__.__name__ == "CLIPFusedEmbedder"


def test_ru_em_42_fused_manager_non_str_non_enum_typeerror():
    """RU-EM-42: FusedModelManager(model_type=3.14) raises TypeError."""
    from models.fused_models.fused_model_manager import FusedModelManager

    with pytest.raises(TypeError, match=r"FusedModelType or str"):
        FusedModelManager(model_type=3.14)


def test_ru_em_43_fused_manager_get_embedding_null_model(monkeypatch, tmp_image_path):
    """RU-EM-43: get_embedding raises RuntimeError when model is None."""
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)
    manager = FusedModelManager("clip", {"device": "cpu"})
    manager.model = None
    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.get_embedding("hi", tmp_image_path)


def test_ru_em_44_fused_manager_get_embedding_from_pil(monkeypatch):
    """RU-EM-44: get_embedding_from_pil returns 512-d unit-norm; null model raises."""
    from PIL import Image
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = FusedModelManager("clip", {"device": "cpu"})

    pil_image = Image.new("RGB", (224, 224), color="red")
    out = manager.get_embedding_from_pil("hello", pil_image)
    assert len(out) == DIM_CLIP
    assert abs(np.linalg.norm(out) - 1.0) < 1e-4

    manager.model = None
    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.get_embedding_from_pil("hello", pil_image)


def test_ru_em_45_fused_manager_get_embeddings_batch(monkeypatch, tmp_image_path):
    """RU-EM-45: get_embeddings(pairs) validates paths, delegates, errors on null model."""
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)
    manager = FusedModelManager("clip", {"device": "cpu"})

    out = manager.get_embeddings([("a", tmp_image_path), ("b", tmp_image_path)])
    assert len(out) == 2
    assert all(len(v) == DIM_CLIP for v in out)

    with pytest.raises(ValueError, match=r"absolute"):
        manager.get_embeddings([("a", tmp_image_path), ("b", "relative.jpg")])

    manager.model = None
    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.get_embeddings([("a", tmp_image_path)])


def test_ru_em_46_fused_manager_individual_null_model(monkeypatch, tmp_image_path):
    """RU-EM-46: get_individual_embeddings raises RuntimeError when model None."""
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = FusedModelManager("clip", {"device": "cpu"})
    manager.model = None
    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.get_individual_embeddings("hi", tmp_image_path)


def test_ru_em_47_fused_manager_dimension_getters_and_null(monkeypatch):
    """RU-EM-47: get_embedding_dimension/base_dimension happy & null-model paths."""
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = FusedModelManager("clip", {"device": "cpu"})

    assert manager.get_embedding_dimension() == DIM_CLIP
    assert manager.get_base_embedding_dimension() == DIM_CLIP

    manager.model = None
    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.get_embedding_dimension()
    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.get_base_embedding_dimension()


def test_ru_em_48_fused_manager_get_model_info_keys(monkeypatch):
    """RU-EM-48: get_model_info contains fusion_method and base_embedding_dimension."""
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = FusedModelManager("clip", {"device": "cpu"})

    info = manager.get_model_info()
    assert info["model_type"] == "clip"
    assert info["fusion_method"] == "average"
    assert info["model_class"] == "CLIPFusedEmbedder"
    assert info["embedding_dimension"] == DIM_CLIP
    assert info["base_embedding_dimension"] == DIM_CLIP
    assert isinstance(info["model_config"], dict)


def test_ru_em_49_fused_manager_set_fusion_method_null(monkeypatch):
    """RU-EM-49: set_fusion_method raises RuntimeError when model is None."""
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = FusedModelManager("clip", {"device": "cpu"})
    manager.model = None
    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.set_fusion_method("concat")


def test_ru_em_50_fused_manager_embed_product_full(monkeypatch, tmp_image_path):
    """RU-EM-50: embed_product joins all text fields (name, desc, category, brand, tags)."""
    from models.fused_models.fused_model_manager import FusedModelManager
    from models import clip_model_pool

    captured = {"texts": []}
    fake_clip = types.ModuleType("clip")

    def tokenize(texts, truncate=True):
        captured["texts"].extend(texts if isinstance(texts, list) else [texts])
        return torch.zeros(
            len(texts) if isinstance(texts, list) else 1, 77, dtype=torch.long
        )

    fake_clip.tokenize = tokenize
    monkeypatch.setitem(sys.modules, "clip", fake_clip)

    model = MagicMock()
    model.encode_text.side_effect = lambda toks: _stub_tensor("clip-text", DIM_CLIP)
    model.encode_image.side_effect = lambda t: _stub_tensor("clip-image", DIM_CLIP)
    preprocess = MagicMock()
    preprocess.side_effect = lambda img: torch.zeros(3, 224, 224)
    monkeypatch.setattr(
        clip_model_pool.CLIPModelPool,
        "get",
        classmethod(lambda cls, name, device: (model, preprocess)),
    )
    _patch_pil_open_with_dummy(monkeypatch)

    manager = FusedModelManager("clip", {"device": "cpu"})
    out = manager.embed_product(
        {
            "image_path": tmp_image_path,
            "name": "X",
            "description": "Y",
            "category": "C",
            "brand": "B",
            "tags": ["t1", "t2"],
        }
    )
    assert len(out) == DIM_CLIP
    combined = " ".join(captured["texts"])
    for tok in ("X", "Y", "C", "B", "t1", "t2"):
        assert tok in combined


def test_ru_em_51_fused_manager_embed_product_string_tags(monkeypatch, tmp_image_path):
    """RU-EM-51: embed_product accepts non-list tags as a single string."""
    from models.fused_models.fused_model_manager import FusedModelManager
    from models import clip_model_pool

    captured = {"texts": []}
    fake_clip = types.ModuleType("clip")

    def tokenize(texts, truncate=True):
        captured["texts"].extend(texts if isinstance(texts, list) else [texts])
        return torch.zeros(
            len(texts) if isinstance(texts, list) else 1, 77, dtype=torch.long
        )

    fake_clip.tokenize = tokenize
    monkeypatch.setitem(sys.modules, "clip", fake_clip)

    model = MagicMock()
    model.encode_text.side_effect = lambda toks: _stub_tensor("clip-text", DIM_CLIP)
    model.encode_image.side_effect = lambda t: _stub_tensor("clip-image", DIM_CLIP)
    preprocess = MagicMock()
    preprocess.side_effect = lambda img: torch.zeros(3, 224, 224)
    monkeypatch.setattr(
        clip_model_pool.CLIPModelPool,
        "get",
        classmethod(lambda cls, name, device: (model, preprocess)),
    )
    _patch_pil_open_with_dummy(monkeypatch)

    manager = FusedModelManager("clip", {"device": "cpu"})
    out = manager.embed_product(
        {"image_path": tmp_image_path, "name": "X", "tags": "single-tag-string"}
    )
    assert len(out) == DIM_CLIP
    combined = " ".join(captured["texts"])
    assert "single-tag-string" in combined


def test_ru_em_52_fused_manager_embed_products_batch(monkeypatch, tmp_image_path):
    """RU-EM-52: embed_products iterates and returns N vectors."""
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)
    manager = FusedModelManager("clip", {"device": "cpu"})

    out = manager.embed_products(
        [
            {"image_path": tmp_image_path, "name": "a"},
            {"image_path": tmp_image_path, "name": "b"},
        ]
    )
    assert len(out) == 2
    assert all(len(v) == DIM_CLIP for v in out)


def test_ru_em_53_fused_manager_switch_model(monkeypatch, tmp_image_path):
    """RU-EM-53: switch_model('marqo', cfg) reinitializes with Marqo backend."""
    from models.fused_models.fused_model_manager import FusedModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_open_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)

    manager = FusedModelManager("clip", {"device": "cpu"})
    assert manager.model.__class__.__name__ == "CLIPFusedEmbedder"

    manager.switch_model("marqo", {"device": "cpu"})
    assert manager.model_type.value == "marqo"
    assert manager.model.__class__.__name__ == "MarqoFusedEmbedder"

    out = manager.get_embedding("t", tmp_image_path)
    assert len(out) == DIM_MARQO


# --- VisualModelManager extra coverage ------------------------------------ #


def test_ru_em_54_visual_manager_enum_and_bad_type(monkeypatch):
    """RU-EM-54: enum accepted; list raises TypeError."""
    from models.visual_models.visual_model_manager import (
        VisualModelManager,
        VisualModelType,
    )

    _patch_dinov3_pool_for_embedders(monkeypatch)
    manager = VisualModelManager(model_type=VisualModelType.DINOV3)
    assert manager.model_type is VisualModelType.DINOV3

    with pytest.raises(TypeError, match=r"VisualModelType or str"):
        VisualModelManager(model_type=[1, 2])


def test_ru_em_55_visual_manager_null_model_runtimeerror(monkeypatch, tmp_image_path):
    """RU-EM-55: get_embedding/get_embeddings/get_embedding_from_pil all raise when model None."""
    from PIL import Image
    from models.visual_models.visual_model_manager import VisualModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = VisualModelManager("clip", {"device": "cpu"})
    manager.model = None

    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.get_embedding(tmp_image_path)
    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.get_embeddings([tmp_image_path])
    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.get_embedding_from_pil(Image.new("RGB", (8, 8)))


def test_ru_em_56_visual_manager_get_embeddings_batch(monkeypatch, tmp_image_path):
    """RU-EM-56: get_embeddings validates paths, returns N vectors; bad path raises."""
    from models.visual_models.visual_model_manager import VisualModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)
    manager = VisualModelManager("clip", {"device": "cpu"})

    out = manager.get_embeddings([tmp_image_path, tmp_image_path])
    assert len(out) == 2
    assert all(len(v) == DIM_CLIP for v in out)

    with pytest.raises(ValueError, match=r"absolute"):
        manager.get_embeddings(["relative.jpg"])


def test_ru_em_57_visual_manager_get_embedding_from_pil(monkeypatch):
    """RU-EM-57: get_embedding_from_pil returns 512-d list[float] for CLIP."""
    from PIL import Image
    from models.visual_models.visual_model_manager import VisualModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = VisualModelManager("clip", {"device": "cpu"})

    pil = Image.new("RGB", (32, 32), color="green")
    out = manager.get_embedding_from_pil(pil)
    assert len(out) == DIM_CLIP
    assert isinstance(out, list)
    assert all(isinstance(x, float) for x in out)


def test_ru_em_58_visual_manager_dimension_fallback(monkeypatch):
    """RU-EM-58: get_embedding_dimension falls back to PIL sample when missing; null raises."""
    from models.visual_models.visual_model_manager import VisualModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = VisualModelManager("clip", {"device": "cpu"})

    class _Shim:
        def __init__(self, inner):
            self._inner = inner

        def embed_pil_image(self, image):
            return self._inner.embed_pil_image(image)

    manager.model = _Shim(manager.model)
    assert not hasattr(manager.model, "get_embedding_dimension")
    assert manager.get_embedding_dimension() == DIM_CLIP

    manager.model = None
    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.get_embedding_dimension()


def test_ru_em_59_visual_manager_get_model_info(monkeypatch):
    """RU-EM-59: get_model_info returns the four expected keys/values."""
    from models.visual_models.visual_model_manager import VisualModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = VisualModelManager("clip", {"device": "cpu"})

    info = manager.get_model_info()
    assert info["model_type"] == "clip"
    assert isinstance(info["model_config"], dict)
    assert info["embedding_dimension"] == DIM_CLIP
    assert info["model_class"] == "CLIPImageEmbedder"


def test_ru_em_60_visual_manager_embed_product_images(monkeypatch, tmp_image_path):
    """RU-EM-60: embed_product_images returns N vectors; missing key raises."""
    from models.visual_models.visual_model_manager import VisualModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)
    manager = VisualModelManager("clip", {"device": "cpu"})

    out = manager.embed_product_images(
        [{"image_path": tmp_image_path}, {"image_path": tmp_image_path}]
    )
    assert len(out) == 2
    assert all(len(v) == DIM_CLIP for v in out)

    with pytest.raises(ValueError, match=r"image_path"):
        manager.embed_product_images([{"name": "no-image"}])


def test_ru_em_61_visual_manager_switch_model(monkeypatch, tmp_image_path):
    """RU-EM-61: switch_model('dinov3', cfg) replaces backend; new dim 4096."""
    from models.visual_models.visual_model_manager import VisualModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_dinov3_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)

    manager = VisualModelManager("clip", {"device": "cpu"})
    assert manager.model.__class__.__name__ == "CLIPImageEmbedder"

    manager.switch_model("dinov3", {"device": "cpu"})
    assert manager.model_type.value == "dinov3"
    assert manager.model.__class__.__name__ == "DINOv3ImageEmbedder"

    out = manager.get_embedding(tmp_image_path)
    assert len(out) == DIM_DINOV3


# --- TextModelManager extra coverage -------------------------------------- #


def test_ru_em_62_text_manager_enum_input(monkeypatch):
    """RU-EM-62: TextModelManager accepts TextModelType enum directly."""
    from models.textual_models.text_model_manager import (
        TextModelManager,
        TextModelType,
    )

    _patch_transformers_for_bge(monkeypatch)
    manager = TextModelManager(
        model_type=TextModelType.BGE, model_config={"device": "cpu"}
    )
    assert manager.model_type is TextModelType.BGE
    assert manager.model.__class__.__name__ == "BGEBaseEmbedder"


def test_ru_em_63_text_manager_null_model_runtime(monkeypatch):
    """RU-EM-63: All four accessors raise RuntimeError when model is None."""
    from models.textual_models.text_model_manager import TextModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = TextModelManager("clip", {"device": "cpu"})
    manager.model = None

    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.get_embedding("x")
    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.get_document_embedding("x")
    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.get_embeddings(["x"])
    with pytest.raises(RuntimeError, match=r"Model not initialized"):
        manager.get_embedding_dimension()


def test_ru_em_64_text_manager_dimension_fallback(monkeypatch):
    """RU-EM-64: get_embedding_dimension falls back to sample when attr missing."""
    from models.textual_models.text_model_manager import TextModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    manager = TextModelManager("clip", {"device": "cpu"})

    class _Shim:
        def __init__(self, inner):
            self._inner = inner

        def embed_query(self, text):
            return self._inner.embed_query(text)

        def embed_documents(self, texts):
            return self._inner.embed_documents(texts)

    manager.model = _Shim(manager.model)
    assert not hasattr(manager.model, "get_embedding_dimension")
    assert manager.get_embedding_dimension() == DIM_CLIP


def test_ru_em_65_text_manager_embed_product_brand_string_tags(monkeypatch):
    """RU-EM-65: embed_product handles brand and string tags."""
    from models.textual_models.text_model_manager import TextModelManager
    from models import clip_model_pool

    captured = {"texts": []}
    fake_clip = types.ModuleType("clip")

    def tokenize(texts, truncate=True):
        captured["texts"].extend(texts if isinstance(texts, list) else [texts])
        return torch.zeros(
            len(texts) if isinstance(texts, list) else 1, 77, dtype=torch.long
        )

    fake_clip.tokenize = tokenize
    monkeypatch.setitem(sys.modules, "clip", fake_clip)

    model = MagicMock()
    model.encode_text.side_effect = lambda toks: _stub_tensor("clip-text", DIM_CLIP)
    preprocess = MagicMock()
    monkeypatch.setattr(
        clip_model_pool.CLIPModelPool,
        "get",
        classmethod(lambda cls, name, device: (model, preprocess)),
    )

    manager = TextModelManager("clip", {"device": "cpu"})
    out = manager.embed_product(
        {
            "name": "N",
            "description": "D",
            "category": "C",
            "brand": "B",
            "tags": "single",
        }
    )
    assert len(out) == DIM_CLIP
    combined = " ".join(captured["texts"])
    for tok in ("N", "D", "C", "B", "single"):
        assert tok in combined


def test_ru_em_66_text_manager_embed_products_mixed_tags(monkeypatch):
    """RU-EM-66: embed_products joins fields with mixed list/string tags."""
    from models.textual_models.text_model_manager import TextModelManager
    from models import clip_model_pool

    captured = {"texts": []}
    fake_clip = types.ModuleType("clip")

    def tokenize(texts, truncate=True):
        captured["texts"].extend(texts if isinstance(texts, list) else [texts])
        return torch.zeros(
            len(texts) if isinstance(texts, list) else 1, 77, dtype=torch.long
        )

    fake_clip.tokenize = tokenize
    monkeypatch.setitem(sys.modules, "clip", fake_clip)

    model = MagicMock()
    model.encode_text.side_effect = lambda toks: _stub_tensor("clip-text", DIM_CLIP)
    preprocess = MagicMock()
    monkeypatch.setattr(
        clip_model_pool.CLIPModelPool,
        "get",
        classmethod(lambda cls, name, device: (model, preprocess)),
    )

    manager = TextModelManager("clip", {"device": "cpu"})
    out = manager.embed_products(
        [
            {"name": "a", "tags": ["t1", "t2"]},
            {"description": "d", "tags": "single"},
        ]
    )
    assert len(out) == 2
    assert all(len(v) == DIM_CLIP for v in out)
    combined = " | ".join(captured["texts"])
    assert "a t1 t2" in combined
    assert "d single" in combined


def test_ru_em_67_text_manager_switch_model(monkeypatch):
    """RU-EM-67: switch_model('qwen', cfg) reinitializes with Qwen backend."""
    from models.textual_models.text_model_manager import TextModelManager

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_transformers_for_qwen(monkeypatch)

    manager = TextModelManager("clip", {"device": "cpu"})
    assert manager.model.__class__.__name__ == "CLIPTextEmbedder"

    manager.switch_model("qwen", {"device": "cpu"})
    assert manager.model_type.value == "qwen"
    assert manager.model.__class__.__name__ == "Qwen8BEmbedder"

    out = manager.get_embedding("hello")
    assert len(out) == DIM_QWEN


# --- CLIPImageEmbedder direct coverage ------------------------------------ #


def test_ru_em_68_clip_image_embedder_batch(monkeypatch, tmp_image_path):
    """RU-EM-68: CLIPImageEmbedder.embed_images returns N 512-d L2-normalized vectors."""
    from models.visual_models.clip_image_embedder import CLIPImageEmbedder

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)

    embedder = CLIPImageEmbedder(device="cpu")
    out = embedder.embed_images([tmp_image_path, tmp_image_path, tmp_image_path])
    assert len(out) == 3
    assert all(len(v) == DIM_CLIP for v in out)
    assert all(abs(np.linalg.norm(v) - 1.0) < 1e-4 for v in out)


def test_ru_em_69_clip_image_embedder_pil_non_rgb(monkeypatch):
    """RU-EM-69: embed_pil_image converts non-RGB modes to RGB then embeds 512-d."""
    from PIL import Image
    from models.visual_models.clip_image_embedder import CLIPImageEmbedder

    _patch_clip_pool_for_embedders(monkeypatch)
    embedder = CLIPImageEmbedder(device="cpu")

    grayscale = Image.new("L", (32, 32), color=128)
    out = embedder.embed_pil_image(grayscale)
    assert len(out) == DIM_CLIP
    assert abs(np.linalg.norm(out) - 1.0) < 1e-4


def test_ru_em_70_clip_image_embedder_path_validation(monkeypatch):
    """RU-EM-70: _load_image rejects relative path and missing absolute path."""
    from models.visual_models.clip_image_embedder import CLIPImageEmbedder

    _patch_clip_pool_for_embedders(monkeypatch)
    embedder = CLIPImageEmbedder(device="cpu")

    with pytest.raises(ValueError, match=r"absolute"):
        embedder.embed_image("relative.jpg")

    bogus = "C:\\nonexistent\\zzz.jpg" if os.name == "nt" else "/nonexistent/zzz.jpg"
    with pytest.raises(FileNotFoundError):
        embedder.embed_image(bogus)


def test_ru_em_71_clip_image_embedder_dimension(monkeypatch):
    """RU-EM-71: CLIPImageEmbedder.get_embedding_dimension returns 512."""
    from models.visual_models.clip_image_embedder import CLIPImageEmbedder

    _patch_clip_pool_for_embedders(monkeypatch)
    embedder = CLIPImageEmbedder(device="cpu")
    assert embedder.get_embedding_dimension() == DIM_CLIP


# --- CLIPFusedEmbedder direct coverage ------------------------------------ #


def test_ru_em_72_clip_fused_pil_non_rgb(monkeypatch):
    """RU-EM-72: embed_text_and_pil_image with non-RGB image returns 512-d unit-norm."""
    from PIL import Image
    from models.fused_models.clip_fused_embedder import CLIPFusedEmbedder

    _patch_clip_pool_for_embedders(monkeypatch)
    embedder = CLIPFusedEmbedder(fusion_method="average", device="cpu")

    rgba = Image.new("RGBA", (16, 16))
    out = embedder.embed_text_and_pil_image("hello", rgba)
    assert len(out) == DIM_CLIP
    assert abs(np.linalg.norm(out) - 1.0) < 1e-4


def test_ru_em_73_clip_fused_embed_pairs(monkeypatch, tmp_image_path):
    """RU-EM-73: embed_pairs returns N fused 512-d unit-norm vectors."""
    from models.fused_models.clip_fused_embedder import CLIPFusedEmbedder

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)
    embedder = CLIPFusedEmbedder(device="cpu")

    out = embedder.embed_pairs([("a", tmp_image_path), ("b", tmp_image_path)])
    assert len(out) == 2
    assert all(len(v) == DIM_CLIP for v in out)
    assert all(abs(np.linalg.norm(v) - 1.0) < 1e-4 for v in out)


def test_ru_em_74_clip_fused_dimensions(monkeypatch):
    """RU-EM-74: get_embedding_dimension and get_base_embedding_dimension across fusions."""
    from models.fused_models.clip_fused_embedder import CLIPFusedEmbedder

    _patch_clip_pool_for_embedders(monkeypatch)
    embedder = CLIPFusedEmbedder(fusion_method="average", device="cpu")

    assert embedder.get_embedding_dimension() == DIM_CLIP
    assert embedder.get_base_embedding_dimension() == DIM_CLIP

    embedder.set_fusion_method("concat")
    assert embedder.get_embedding_dimension() == 2 * DIM_CLIP
    assert embedder.get_base_embedding_dimension() == DIM_CLIP


def test_ru_em_75_clip_fused_path_validation(monkeypatch):
    """RU-EM-75: _load_image rejects relative path and missing absolute path."""
    from models.fused_models.clip_fused_embedder import CLIPFusedEmbedder

    _patch_clip_pool_for_embedders(monkeypatch)
    embedder = CLIPFusedEmbedder(device="cpu")

    with pytest.raises(ValueError, match=r"absolute"):
        embedder.embed_text_and_image("t", "rel.jpg")

    bogus = "C:\\nope\\x.jpg" if os.name == "nt" else "/nope/x.jpg"
    with pytest.raises(FileNotFoundError):
        embedder.embed_text_and_image("t", bogus)


def test_ru_em_76_clip_fused_unknown_fusion_at_call(monkeypatch, tmp_image_path):
    """RU-EM-76: _fuse_embeddings raises ValueError for unknown method via embed_text_and_image."""
    from models.fused_models.clip_fused_embedder import CLIPFusedEmbedder

    _patch_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)
    embedder = CLIPFusedEmbedder(fusion_method="average", device="cpu")
    embedder.fusion_method = "garbage"  # bypass set_fusion_method validation

    with pytest.raises(ValueError, match=r"Unknown fusion method"):
        embedder.embed_text_and_image("t", tmp_image_path)


# --- Marqo embedders direct coverage -------------------------------------- #


def test_ru_em_77_marqo_image_embedder_full(monkeypatch, tmp_image_path):
    """RU-EM-77: MarqoImageEmbedder batch + non-RGB + path errors + dimension."""
    from PIL import Image
    from models.visual_models.marqo_image_embedder import MarqoImageEmbedder

    _patch_open_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)
    embedder = MarqoImageEmbedder(device="cpu")

    out = embedder.embed_images([tmp_image_path, tmp_image_path])
    assert len(out) == 2
    assert all(len(v) == DIM_MARQO for v in out)

    gray = Image.new("L", (8, 8))
    out2 = embedder.embed_pil_image(gray)
    assert len(out2) == DIM_MARQO

    with pytest.raises(ValueError, match=r"absolute"):
        embedder.embed_image("rel.jpg")

    bogus = "C:\\nope\\z.jpg" if os.name == "nt" else "/nope/z.jpg"
    with pytest.raises(FileNotFoundError):
        embedder.embed_image(bogus)

    assert embedder.get_embedding_dimension() == DIM_MARQO


def test_ru_em_78_marqo_text_embedder_batch_and_dim(monkeypatch):
    """RU-EM-78: MarqoTextEmbedder.embed_documents batch + get_embedding_dimension."""
    from models.textual_models.marqo_text_embedder import MarqoTextEmbedder

    _patch_open_clip_pool_for_embedders(monkeypatch)
    embedder = MarqoTextEmbedder(device="cpu")

    out = embedder.embed_documents(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == DIM_MARQO for v in out)

    assert embedder.get_embedding_dimension() == DIM_MARQO


def test_ru_em_79_marqo_fused_embedder_full(monkeypatch, tmp_image_path):
    """RU-EM-79: MarqoFusedEmbedder batch/individual/dims/set_fusion_method/path errors."""
    from PIL import Image
    from models.fused_models.marqo_fused_embedder import MarqoFusedEmbedder

    _patch_open_clip_pool_for_embedders(monkeypatch)
    _patch_pil_open_with_dummy(monkeypatch)
    embedder = MarqoFusedEmbedder(device="cpu", fusion_method="average")

    gray = Image.new("L", (16, 16))
    out_pil = embedder.embed_text_and_pil_image("t", gray)
    assert len(out_pil) == DIM_MARQO
    assert abs(np.linalg.norm(out_pil) - 1.0) < 1e-4

    out_pairs = embedder.embed_pairs([("a", tmp_image_path), ("b", tmp_image_path)])
    assert len(out_pairs) == 2
    assert all(len(v) == DIM_MARQO for v in out_pairs)

    t_emb, i_emb = embedder.get_individual_embeddings("t", tmp_image_path)
    assert len(t_emb) == DIM_MARQO
    assert len(i_emb) == DIM_MARQO

    assert embedder.get_embedding_dimension() == DIM_MARQO
    assert embedder.get_base_embedding_dimension() == DIM_MARQO

    embedder.set_fusion_method("concat")
    assert embedder.get_embedding_dimension() == 2 * DIM_MARQO

    with pytest.raises(ValueError, match=r"Unknown fusion method"):
        embedder.set_fusion_method("garbage")

    with pytest.raises(ValueError, match=r"absolute"):
        embedder.embed_text_and_image("t", "relative.jpg")

    bogus = "C:\\nope\\zz.jpg" if os.name == "nt" else "/nope/zz.jpg"
    with pytest.raises(FileNotFoundError):
        embedder.embed_text_and_image("t", bogus)


# --- DINOv3 embedder direct coverage -------------------------------------- #


def test_ru_em_80_dinov3_embedder_full(monkeypatch, tmp_image_path):
    """RU-EM-80: DINOv3 batch + non-RGB + path errors + _pool_cls last_hidden_state fallback."""
    from PIL import Image
    from models import dinov3_model_pool
    from models.visual_models.dinov3_image_embedder import DINOv3ImageEmbedder

    class _Out:
        def __init__(self, lhs):
            self.pooler_output = None
            self.last_hidden_state = lhs

    model = MagicMock(name="dinov3-stub-model-no-pooler")

    def _call(**inputs):
        seq_len = 5
        cls_vec = np.array(
            deterministic_vector("dinov3-cls", DIM_DINOV3), dtype=np.float32
        )
        lhs = np.zeros((1, seq_len, DIM_DINOV3), dtype=np.float32)
        lhs[0, 0, :] = cls_vec
        return _Out(torch.tensor(lhs, dtype=torch.float32))

    model.side_effect = _call
    processor = MagicMock(name="dinov3-stub-processor")
    processor.side_effect = (
        lambda images=None, return_tensors="pt": _FakeTokenizerOutput(
            {"pixel_values": torch.zeros(1, 3, 224, 224)}
        )
    )

    monkeypatch.setattr(
        dinov3_model_pool.DINOv3ModelPool,
        "get",
        classmethod(lambda cls, name, device: (model, processor)),
    )
    _patch_pil_open_with_dummy(monkeypatch)

    embedder = DINOv3ImageEmbedder(device="cpu")

    out_batch = embedder.embed_images([tmp_image_path, tmp_image_path])
    assert len(out_batch) == 2
    assert all(len(v) == DIM_DINOV3 for v in out_batch)
    assert all(abs(np.linalg.norm(v) - 1.0) < 1e-4 for v in out_batch)

    gray = Image.new("L", (8, 8))
    out_gray = embedder.embed_pil_image(gray)
    assert len(out_gray) == DIM_DINOV3

    with pytest.raises(ValueError, match=r"absolute"):
        embedder.embed_image("relative.jpg")

    bogus = "C:\\nope\\xx.jpg" if os.name == "nt" else "/nope/xx.jpg"
    with pytest.raises(FileNotFoundError):
        embedder.embed_image(bogus)


def test_ru_em_81_dinov3_dimension_constant(monkeypatch):
    """RU-EM-81: DINOv3.get_embedding_dimension returns 4096 without invoking model."""
    from models import dinov3_model_pool
    from models.visual_models.dinov3_image_embedder import DINOv3ImageEmbedder

    model = MagicMock(name="dinov3-stub-model-untouched")
    processor = MagicMock(name="dinov3-stub-processor-untouched")

    monkeypatch.setattr(
        dinov3_model_pool.DINOv3ModelPool,
        "get",
        classmethod(lambda cls, name, device: (model, processor)),
    )

    embedder = DINOv3ImageEmbedder(device="cpu")
    model.reset_mock()
    assert embedder.get_embedding_dimension() == DIM_DINOV3
    assert not model.called


# --- BGE & Qwen direct coverage ------------------------------------------- #


def test_ru_em_82_bge_embedder_full(monkeypatch):
    """RU-EM-82: BGE ImportError + RuntimeError + batch embed + dimension."""
    import importlib
    from models.textual_models import bge_base_embedder as bge_mod

    # (a) Missing transformers → ImportError when constructing.
    monkeypatch.setitem(sys.modules, "transformers", None)
    importlib.reload(bge_mod)
    with pytest.raises(ImportError, match=r"Transformers is not installed"):
        bge_mod.BGEBaseEmbedder(device="cpu")

    # (b) Generic load failure → RuntimeError.
    fake_bad = types.ModuleType("transformers")

    class _BadTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            raise Exception("kaboom")

    class _BadModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return cls()

    fake_bad.AutoTokenizer = _BadTokenizer
    fake_bad.AutoModel = _BadModel
    monkeypatch.setitem(sys.modules, "transformers", fake_bad)
    importlib.reload(bge_mod)
    with pytest.raises(RuntimeError, match=r"Failed to load BGE model"):
        bge_mod.BGEBaseEmbedder(device="cpu")

    # (c) Successful BGE-shaped stubs → batch + dimension.
    counter = {"count": 0}
    fake_good = _make_fake_transformers_module(counter, last_hidden_dim=DIM_BGE)
    monkeypatch.setitem(sys.modules, "transformers", fake_good)
    importlib.reload(bge_mod)

    embedder = bge_mod.BGEBaseEmbedder(device="cpu")
    out = embedder.embed_documents(["a", "b"])
    assert len(out) == 2
    assert all(len(v) == DIM_BGE for v in out)
    assert all(abs(np.linalg.norm(v) - 1.0) < 1e-4 for v in out)
    assert embedder.get_embedding_dimension() == DIM_BGE


def test_ru_em_83_qwen_embedder_full(monkeypatch):
    """RU-EM-83: Qwen ImportError + RuntimeError + batch embed + dimension."""
    import importlib
    from models.textual_models import qwen_8b_model as qwen_mod

    # (a) Missing transformers.
    monkeypatch.setitem(sys.modules, "transformers", None)
    importlib.reload(qwen_mod)
    with pytest.raises(ImportError, match=r"Transformers is not installed"):
        qwen_mod.Qwen8BEmbedder(device="cpu")

    # (b) Generic load failure.
    fake_bad = types.ModuleType("transformers")

    class _BadTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            raise Exception("crash")

    class _BadModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return cls()

    fake_bad.AutoTokenizer = _BadTokenizer
    fake_bad.AutoModel = _BadModel
    monkeypatch.setitem(sys.modules, "transformers", fake_bad)
    importlib.reload(qwen_mod)
    with pytest.raises(RuntimeError, match=r"Failed to load Qwen model"):
        qwen_mod.Qwen8BEmbedder(device="cpu")

    # (c) Successful stubs sized for Qwen (4096-d).
    counter = {"count": 0}
    fake_good = _make_fake_transformers_module(counter, last_hidden_dim=DIM_QWEN)
    monkeypatch.setitem(sys.modules, "transformers", fake_good)
    importlib.reload(qwen_mod)

    embedder = qwen_mod.Qwen8BEmbedder(device="cpu")
    out = embedder.embed_documents(["x", "y"])
    assert len(out) == 2
    assert all(len(v) == DIM_QWEN for v in out)
    assert all(abs(np.linalg.norm(v) - 1.0) < 1e-4 for v in out)
    assert embedder.get_embedding_dimension() == DIM_QWEN
