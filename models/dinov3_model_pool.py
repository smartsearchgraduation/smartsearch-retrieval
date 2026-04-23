"""
Shared DINOv3 Model Pool.

Thread-safe singleton pool that ensures each unique DINOv3 model (identified by
model_name + device) is loaded only once into memory. All DINOv3 embedders
share the same model instance.

This is safe because:
- The model is used in .eval() mode (no training state)
- All inference uses torch.no_grad() context (no gradient accumulation)
- Model weights are read-only during inference
"""

import threading
from typing import Dict, Tuple, Any


class DINOv3ModelPool:
    """
    Thread-safe singleton pool for DINOv3 models.

    Prevents duplicate loading of the same DINOv3 model across embedders.
    """

    _models: Dict[str, Tuple[Any, Any]] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, model_name: str, device: str) -> Tuple[Any, Any]:
        """
        Get or load a DINOv3 model. Returns (model, processor) tuple.

        If the model is already loaded for the given (model_name, device),
        returns the cached instance. Otherwise loads it once.

        Args:
            model_name: HuggingFace model ID (e.g., "facebook/dinov3-vit7b16-pretrain-lvd1689m").
            device: Device to load onto (e.g., "cpu", "cuda").

        Returns:
            Tuple of (model, processor) — shared references, do not mutate.
        """
        key = f"{model_name}:{device}"
        with cls._lock:
            if key not in cls._models:
                try:
                    import torch
                    from transformers import AutoImageProcessor, AutoModel

                    processor = AutoImageProcessor.from_pretrained(model_name)
                    dtype = torch.bfloat16 if device == "cuda" else torch.float32
                    model = AutoModel.from_pretrained(model_name, dtype=dtype)
                    model = model.to(device)
                    model.eval()
                    cls._models[key] = (model, processor)
                    print(
                        f"[DINOv3ModelPool] Loaded model: {model_name} on {device} (dtype={dtype})"
                    )
                except ImportError:
                    raise ImportError(
                        "transformers is not installed. Please install it via: "
                        "pip install transformers"
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to load DINOv3 model: {e}")
            else:
                print(
                    f"[DINOv3ModelPool] Reusing model: {model_name} on {device}"
                )
            return cls._models[key]

    @classmethod
    def clear(cls):
        """Clear all cached models (useful for testing)."""
        with cls._lock:
            cls._models.clear()
