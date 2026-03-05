"""
Unit tests for CLIPModelPool.

All tests inject a mock `clip` module via sys.modules to avoid
downloading real model weights.
"""

import sys
import pytest
from unittest.mock import MagicMock

from models.clip_model_pool import CLIPModelPool


@pytest.fixture(autouse=True)
def _clear_pool():
    """Ensure a clean pool before and after every test."""
    CLIPModelPool.clear()
    yield
    CLIPModelPool.clear()


@pytest.fixture()
def mock_clip():
    """
    Inject a fake 'clip' module into sys.modules so that the lazy
    `import clip` inside CLIPModelPool.get() picks it up.
    """
    fake_clip = MagicMock()
    fake_model = MagicMock()
    fake_preprocess = MagicMock()
    fake_clip.load.return_value = (fake_model, fake_preprocess)

    original = sys.modules.get("clip")
    sys.modules["clip"] = fake_clip
    yield fake_clip, fake_model, fake_preprocess

    # Restore original state
    if original is None:
        sys.modules.pop("clip", None)
    else:
        sys.modules["clip"] = original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCLIPModelPool:

    def test_get_loads_model(self, mock_clip):
        """First get() should call clip.load and return (model, preprocess)."""
        fake_clip, fake_model, fake_preprocess = mock_clip

        model, preprocess = CLIPModelPool.get("ViT-B/32", "cpu")

        fake_clip.load.assert_called_once_with("ViT-B/32", device="cpu")
        assert model is fake_model
        assert preprocess is fake_preprocess

    def test_get_caches_model(self, mock_clip):
        """Second call with same args should reuse the cached model."""
        fake_clip, _, _ = mock_clip

        result1 = CLIPModelPool.get("ViT-B/32", "cpu")
        result2 = CLIPModelPool.get("ViT-B/32", "cpu")

        fake_clip.load.assert_called_once()
        assert result1 is result2

    def test_get_different_models(self, mock_clip):
        """Different model names should be cached separately."""
        fake_clip, _, _ = mock_clip
        fake_clip.load.side_effect = [
            (MagicMock(name="model_a"), MagicMock()),
            (MagicMock(name="model_b"), MagicMock()),
        ]

        m1, _ = CLIPModelPool.get("ViT-B/32", "cpu")
        m2, _ = CLIPModelPool.get("ViT-L/14", "cpu")

        assert m1 is not m2
        assert fake_clip.load.call_count == 2

    def test_get_different_devices(self, mock_clip):
        """Same model on different devices should be cached separately."""
        fake_clip, _, _ = mock_clip
        fake_clip.load.side_effect = [
            (MagicMock(name="cpu_model"), MagicMock()),
            (MagicMock(name="cuda_model"), MagicMock()),
        ]

        m_cpu, _ = CLIPModelPool.get("ViT-B/32", "cpu")
        m_cuda, _ = CLIPModelPool.get("ViT-B/32", "cuda")

        assert m_cpu is not m_cuda
        assert fake_clip.load.call_count == 2

    def test_clear(self, mock_clip):
        """After clear(), get() should reload the model."""
        fake_clip, _, _ = mock_clip

        CLIPModelPool.get("ViT-B/32", "cpu")
        CLIPModelPool.clear()
        CLIPModelPool.get("ViT-B/32", "cpu")

        assert fake_clip.load.call_count == 2
