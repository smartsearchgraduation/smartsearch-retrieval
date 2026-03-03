"""
Manager Service — centralizes initialization and access to model managers and FAISS.

All lazy-initialization logic for TextModelManager, VisualModelManager,
FusedModelManager, and FAISSManager lives here.
"""

import json
import os
from typing import Dict, Any, Optional

from models.textual_models import TextModelManager
from models.visual_models import VisualModelManager
from models.fused_models import FusedModelManager
from vector_db import FAISSManager


# Global managers (initialized lazily)
_textual_managers: Dict[str, TextModelManager] = {}
_visual_managers: Dict[str, VisualModelManager] = {}
_fused_managers: Dict[str, FusedModelManager] = {}
_faiss_manager: Optional[FAISSManager] = None

# Configuration (loaded at module import)
_config = None
MODEL_REGISTRY = {}
DEFAULT_DIMENSION = 512
INDEX_PATH = "./data/faiss_indices"
HOST = "0.0.0.0"
PORT = 5002
MAX_TOP_K = 100
DEFAULT_TOP_K = 10


def load_config() -> dict:
    """Load configuration from config.json and initialize module-level variables."""
    global _config, MODEL_REGISTRY, DEFAULT_DIMENSION, INDEX_PATH
    global HOST, PORT, MAX_TOP_K, DEFAULT_TOP_K

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
    )
    try:
        with open(config_path, "r") as f:
            _config = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in configuration file: {e}")

    MODEL_REGISTRY = _config["models"]
    defaults = _config.get("defaults", {})
    DEFAULT_DIMENSION = defaults.get("dimension", 512)
    INDEX_PATH = defaults.get("index_path", "./data/faiss_indices")
    HOST = defaults.get("host", "0.0.0.0")
    PORT = defaults.get("port", 5002)
    MAX_TOP_K = defaults.get("max_top_k", 100)
    DEFAULT_TOP_K = defaults.get("top_k", 10)

    return _config


def get_faiss_manager(
    textual_model_name: str = None,
    visual_model_name: str = None,
    fused_model_name: str = None,
) -> FAISSManager:
    """
    Get or initialize the FAISS manager.

    Dimensions are determined from the MODEL_REGISTRY based on the provided model names.
    The first call with model names sets the dimensions for each index type.
    """
    global _faiss_manager
    if _faiss_manager is None:
        # Get dimensions from MODEL_REGISTRY based on provided model names
        textual_dim = (
            MODEL_REGISTRY.get(textual_model_name, {}).get("dimension", DEFAULT_DIMENSION)
            if textual_model_name
            else DEFAULT_DIMENSION
        )
        visual_dim = (
            MODEL_REGISTRY.get(visual_model_name, {}).get("dimension", DEFAULT_DIMENSION)
            if visual_model_name
            else DEFAULT_DIMENSION
        )
        fused_dim = (
            MODEL_REGISTRY.get(fused_model_name, {}).get("dimension", DEFAULT_DIMENSION)
            if fused_model_name
            else DEFAULT_DIMENSION
        )

        _faiss_manager = FAISSManager(
            dimension=DEFAULT_DIMENSION,
            index_path=INDEX_PATH,
            dimensions={
                "textual": textual_dim,
                "visual": visual_dim,
                "fused": fused_dim,
            },
        )
    return _faiss_manager


def get_textual_manager(model_name: str) -> TextModelManager:
    """Get or initialize a textual model manager."""
    if model_name not in _textual_managers:
        # Get model type from MODEL_REGISTRY, fallback to detection logic
        if model_name in MODEL_REGISTRY:
            model_type = MODEL_REGISTRY[model_name]["type"]
        elif "bge" in model_name.lower() or model_name.startswith("BAAI/"):
            model_type = "bge"
        elif "qwen" in model_name.lower() or model_name.startswith("Qwen/"):
            model_type = "qwen"
        else:
            model_type = "clip"

        _textual_managers[model_name] = TextModelManager(
            model_type=model_type,
            model_config={"model_name": model_name},
        )
    return _textual_managers[model_name]


def get_visual_manager(model_name: str) -> VisualModelManager:
    """Get or initialize a visual model manager."""
    if model_name not in _visual_managers:
        _visual_managers[model_name] = VisualModelManager(
            model_type="clip",
            model_config={"model_name": model_name},
        )
    return _visual_managers[model_name]


def get_fused_manager(model_name: str) -> FusedModelManager:
    """Get or initialize a fused model manager."""
    if model_name not in _fused_managers:
        _fused_managers[model_name] = FusedModelManager(
            model_type="clip",
            model_config={"model_name": model_name},
        )
    return _fused_managers[model_name]


def combine_product_text(
    name: str,
    description: str,
    brand: str,
    category: str,
    price: Any,
) -> str:
    """Combine product fields into a single text for embedding."""
    parts = []
    if name:
        parts.append(name)
    if description:
        parts.append(description)
    if brand:
        parts.append(brand)
    if category:
        parts.append(category)
    if price:
        parts.append(f"Price: {price}")
    return " ".join(parts)
