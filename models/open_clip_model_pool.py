"""
Shared OpenCLIP Model Pool.

Thread-safe singleton pool that ensures each unique OpenCLIP model (identified by
model_name + device) is loaded only once into memory. All Marqo embedders
(text, image, fused) share the same model instance.

This is safe because:
- The model is used in .eval() mode (no training state)
- All inference uses torch.no_grad() context (no gradient accumulation)
- Model weights are read-only during inference
"""

import threading
from typing import Dict, Tuple, Any


class OpenCLIPModelPool:
    """
    Thread-safe singleton pool for OpenCLIP models.

    Prevents duplicate loading of the same model across
    text, image, and fused embedders.
    """

    _models: Dict[str, Tuple[Any, Any, Any]] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, model_name: str, device: str) -> Tuple[Any, Any, Any]:
        """
        Get or load an OpenCLIP model. Returns (model, preprocess, tokenizer) tuple.

        If the model is already loaded for the given (model_name, device),
        returns the cached instance. Otherwise loads it once.

        Args:
            model_name: HuggingFace model ID (e.g., "Marqo/marqo-ecommerce-embeddings-L").
            device: Device to load onto (e.g., "cpu", "cuda").

        Returns:
            Tuple of (model, preprocess, tokenizer) — shared references, do not mutate.
        """
        key = f"{model_name}:{device}"
        with cls._lock:
            if key not in cls._models:
                try:
                    import open_clip
                    import torch

                    hf_hub_name = f"hf-hub:{model_name}"
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        hf_hub_name
                    )
                    tokenizer = open_clip.get_tokenizer(hf_hub_name)
                    model = model.to(device)
                    model.eval()
                    cls._models[key] = (model, preprocess, tokenizer)
                    print(
                        f"[OpenCLIPModelPool] Loaded model: {model_name} on {device}"
                    )
                except ImportError:
                    raise ImportError(
                        "open_clip is not installed. Please install it via: "
                        "pip install open_clip_torch"
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to load OpenCLIP model: {e}")
            else:
                print(
                    f"[OpenCLIPModelPool] Reusing model: {model_name} on {device}"
                )
            return cls._models[key]

    @classmethod
    def clear(cls):
        """Clear all cached models (useful for testing)."""
        with cls._lock:
            cls._models.clear()
