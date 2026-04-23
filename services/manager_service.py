"""
Manager Service — centralizes initialization and access to model managers and FAISS.

All lazy-initialization logic for TextModelManager, VisualModelManager,
FusedModelManager, and FAISSManager lives here.

Each model stores its embeddings in its own folder under DATA_BASE_PATH:
    <sanitized_model_name>_<dimension>_embeddings/
Callers use separate FAISSManager instances for different models (e.g. one
for the textual model, another for the visual model).
"""

import json
import os
from typing import Dict, Any, Optional, List

from models.textual_models import TextModelManager
from models.visual_models import VisualModelManager
from models.fused_models import FusedModelManager
from vector_db import FAISSManager
from vector_db.faiss_manager import make_folder_name


# Global managers (initialized lazily)
_textual_managers: Dict[str, TextModelManager] = {}
_visual_managers: Dict[str, VisualModelManager] = {}
_fused_managers: Dict[str, FusedModelManager] = {}
_faiss_managers: Dict[str, FAISSManager] = {}

# Configuration (loaded at module import)
_config = None
MODEL_REGISTRY = {}
DEFAULT_MODELS: Dict[str, str] = {}
DEFAULT_DIMENSION = 512
DATA_BASE_PATH = "./data"
HOST = "0.0.0.0"
PORT = 5002
MAX_TOP_K = 100
DEFAULT_TOP_K = 10


def load_config() -> dict:
    """Load configuration from config.json and initialize module-level variables."""
    global _config, MODEL_REGISTRY, DEFAULT_MODELS, DEFAULT_DIMENSION, DATA_BASE_PATH
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
    DEFAULT_MODELS = _config.get("default_models", {})
    defaults = _config.get("defaults", {})
    DEFAULT_DIMENSION = defaults.get("dimension", 512)
    DATA_BASE_PATH = defaults.get("data_base_path", "./data")
    HOST = defaults.get("host", "0.0.0.0")
    PORT = defaults.get("port", 5002)
    MAX_TOP_K = defaults.get("max_top_k", 100)
    DEFAULT_TOP_K = defaults.get("top_k", 10)

    return _config


def _get_model_dimension(model_name: str) -> int:
    """Get dimension for a model from registry."""
    return MODEL_REGISTRY.get(model_name, {}).get("dimension", DEFAULT_DIMENSION)


def get_faiss_manager(model_name: str = None) -> FAISSManager:
    """
    Get or create a FAISSManager for a single model.

    Each model gets its own folder under DATA_BASE_PATH:
        <sanitized_model_name>_<dimension>_embeddings/

    Callers that use different models for textual and visual embeddings
    should call this function separately for each model.
    """
    if model_name is None:
        if _faiss_managers:
            return next(iter(_faiss_managers.values()))
        model_name = DEFAULT_MODELS.get("textual", "")

    dimension = _get_model_dimension(model_name)
    folder_name = make_folder_name(model_name, dimension)

    if folder_name not in _faiss_managers:
        index_path = os.path.join(DATA_BASE_PATH, folder_name)
        _faiss_managers[folder_name] = FAISSManager(
            dimension=dimension,
            index_path=index_path,
        )
    return _faiss_managers[folder_name]


def get_all_faiss_managers() -> Dict[str, FAISSManager]:
    """Return all currently loaded FAISSManagers (keyed by folder name)."""
    return _faiss_managers


def discover_model_folders() -> List[str]:
    """Scan DATA_BASE_PATH for existing model embedding folders."""
    if not os.path.exists(DATA_BASE_PATH):
        return []
    return [
        d
        for d in os.listdir(DATA_BASE_PATH)
        if os.path.isdir(os.path.join(DATA_BASE_PATH, d)) and d.endswith("_embeddings")
    ]


def get_or_load_all_faiss_managers() -> Dict[str, FAISSManager]:
    """Ensure all on-disk model folders have a loaded FAISSManager."""
    for folder_name in discover_model_folders():
        if folder_name not in _faiss_managers:
            index_path = os.path.join(DATA_BASE_PATH, folder_name)
            # Parse dimension from folder name: <name>_<dim>_embeddings
            parts = folder_name.rsplit("_", 2)
            try:
                dim = int(parts[-2])
            except (IndexError, ValueError):
                dim = DEFAULT_DIMENSION
            _faiss_managers[folder_name] = FAISSManager(
                dimension=dim,
                index_path=index_path,
            )
    return _faiss_managers


def remove_product_from_all_models(product_id: str) -> Dict[str, Dict[str, int]]:
    """Remove a product's embeddings from ALL model folders.

    Returns dict of {folder_name: {index_type: removed_count}}.
    """
    all_managers = get_or_load_all_faiss_managers()
    all_removed = {}
    for folder_name, manager in all_managers.items():
        removed = manager.remove_product_from_all(product_id)
        if sum(removed.values()) > 0:
            manager.save()
            all_removed[folder_name] = removed
    return all_removed


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
        elif "marqo" in model_name.lower() or model_name.startswith("Marqo/"):
            model_type = "marqo"
        else:
            model_type = "clip"

        _textual_managers[model_name] = TextModelManager(
            model_type=model_type,
            model_config={"model_name": model_name},
        )
    return _textual_managers[model_name]


def _get_visual_model_type(model_name: str) -> str:
    """Determine visual model type from registry or name."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]["type"]
    if "marqo" in model_name.lower() or model_name.startswith("Marqo/"):
        return "marqo"
    if "dinov3" in model_name.lower() or model_name.startswith("facebook/dinov3"):
        return "dinov3"
    return "clip"


def get_visual_manager(model_name: str) -> VisualModelManager:
    """Get or initialize a visual model manager."""
    if model_name not in _visual_managers:
        model_type = _get_visual_model_type(model_name)
        _visual_managers[model_name] = VisualModelManager(
            model_type=model_type,
            model_config={"model_name": model_name},
        )
    return _visual_managers[model_name]


def get_fused_manager(model_name: str) -> FusedModelManager:
    """Get or initialize a fused model manager."""
    if model_name not in _fused_managers:
        model_type = _get_visual_model_type(model_name)
        _fused_managers[model_name] = FusedModelManager(
            model_type=model_type,
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


def get_all_index_stats() -> Dict[str, Dict[str, int]]:
    """Return per-model-folder index stats."""
    all_managers = get_or_load_all_faiss_managers()
    stats = {}
    for folder_name, manager in all_managers.items():
        stats[folder_name] = manager.get_all_sizes()
    return stats


TEXTUAL_TYPES = {"clip", "bge", "qwen", "marqo"}
VISUAL_TYPES = {"clip", "marqo", "dinov3"}


def get_available_models() -> dict:
    """Return available models categorized by type, with defaults."""
    textual_models = []
    visual_models = []

    for name, info in MODEL_REGISTRY.items():
        entry = {"name": name, "dimension": info["dimension"]}
        model_type = info["type"]
        if model_type in TEXTUAL_TYPES:
            textual_models.append(entry)
        if model_type in VISUAL_TYPES:
            visual_models.append(entry)

    return {
        "textual_models": textual_models,
        "visual_models": visual_models,
        "defaults": {
            "textual": DEFAULT_MODELS.get("textual", ""),
            "visual": DEFAULT_MODELS.get("visual", ""),
        },
    }
