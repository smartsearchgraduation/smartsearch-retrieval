"""
Shared test helpers for the Retrieval service test suite.

Provides deterministic, L2-normalized stub vectors for the supported
embedding dimensions, plus utilities for building fake embedders that
satisfy the mocking rules in CLAUDE.md (no real model weights, ever).
"""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np


# Per-model dimensions defined by config.json. Keep in sync with that file.
DIM_CLIP = 512
DIM_BGE = 1024
DIM_MARQO = 1024
DIM_QWEN = 4096
DIM_DINOV3 = 4096


def _seed_from_text(text: str) -> int:
    """Deterministic 32-bit seed derived from a hash of the input."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def deterministic_vector(seed_text: str, dimension: int) -> List[float]:
    """
    Return an L2-normalized vector of the given dimension whose values are
    deterministically derived from ``seed_text``. Same input always yields the
    same vector, regardless of process or platform.
    """
    rng = np.random.default_rng(_seed_from_text(seed_text))
    vec = rng.standard_normal(dimension).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def is_l2_normalized(vec, atol: float = 1e-5) -> bool:
    """Return True if ``vec`` has unit L2 norm within ``atol``."""
    arr = np.asarray(vec, dtype=np.float32)
    return abs(float(np.linalg.norm(arr)) - 1.0) <= atol


class StubTextEmbedder:
    """Deterministic, weights-free stand-in for a text embedder."""

    def __init__(self, model_name: str = "stub-text", dimension: int = DIM_BGE):
        self.model_name = model_name
        self.dimension = dimension

    def embed(self, text: str) -> List[float]:
        return deterministic_vector(f"text::{self.model_name}::{text}", self.dimension)


class StubImageEmbedder:
    """Deterministic, weights-free stand-in for an image embedder."""

    def __init__(self, model_name: str = "stub-image", dimension: int = DIM_CLIP):
        self.model_name = model_name
        self.dimension = dimension

    def embed(self, image_path: str) -> List[float]:
        return deterministic_vector(
            f"image::{self.model_name}::{image_path}", self.dimension
        )


class StubFusedEmbedder:
    """Deterministic, weights-free stand-in for a fused (multimodal) embedder."""

    def __init__(self, model_name: str = "stub-fused", dimension: int = DIM_CLIP):
        self.model_name = model_name
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        return deterministic_vector(
            f"fused-text::{self.model_name}::{text}", self.dimension
        )

    def embed_image(self, image_path: str) -> List[float]:
        return deterministic_vector(
            f"fused-image::{self.model_name}::{image_path}", self.dimension
        )

    def embed_fused(self, text: str, image_path: str) -> List[float]:
        return deterministic_vector(
            f"fused::{self.model_name}::{text}::{image_path}", self.dimension
        )
