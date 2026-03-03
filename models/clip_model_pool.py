"""
Shared CLIP Model Pool.

Thread-safe singleton pool that ensures each unique CLIP model (identified by
model_name + device) is loaded only once into memory. All CLIP embedders
(text, image, fused) share the same model instance.

This is safe because:
- The CLIP model is used in .eval() mode (no training state)
- All inference uses torch.no_grad() context (no gradient accumulation)
- Model weights are read-only during inference
"""

import threading
from typing import Dict, Tuple, Any


class CLIPModelPool:
    """
    Thread-safe singleton pool for CLIP models.

    Prevents duplicate loading of the same CLIP model across
    text, image, and fused embedders (~340MB saved per duplicate for ViT-B/32).
    """

    _models: Dict[str, Tuple[Any, Any]] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, model_name: str, device: str) -> Tuple[Any, Any]:
        """
        Get or load a CLIP model. Returns (model, preprocess) tuple.

        If the model is already loaded for the given (model_name, device),
        returns the cached instance. Otherwise loads it once.

        Args:
            model_name: CLIP model variant (e.g., "ViT-B/32").
            device: Device to load onto (e.g., "cpu", "cuda").

        Returns:
            Tuple of (model, preprocess) — shared references, do not mutate.
        """
        key = f"{model_name}:{device}"
        with cls._lock:
            if key not in cls._models:
                try:
                    import clip

                    model, preprocess = clip.load(model_name, device=device)
                    model.eval()
                    cls._models[key] = (model, preprocess)
                    print(
                        f"[CLIPModelPool] Loaded CLIP model: {model_name} on {device}"
                    )
                except ImportError:
                    raise ImportError(
                        "CLIP is not installed. Please install it via: "
                        "pip install git+https://github.com/openai/CLIP.git"
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to load CLIP model: {e}")
            else:
                print(
                    f"[CLIPModelPool] Reusing CLIP model: {model_name} on {device}"
                )
            return cls._models[key]

    @classmethod
    def clear(cls):
        """Clear all cached models (useful for testing)."""
        with cls._lock:
            cls._models.clear()
