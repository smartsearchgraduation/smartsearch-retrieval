"""
Marqo Fused Embedder
Generates fused embeddings combining text and image using Marqo's ecommerce model.
Uses the same fusion methods as CLIPFusedEmbedder (average, weighted, concat).
"""

import torch
import numpy as np
from typing import List, Tuple
from pathlib import Path
from PIL import Image


class MarqoFusedEmbedder:
    """
    Fused embedder using Marqo's ecommerce model.
    Combines text and image embeddings into a single fused embedding.
    """

    def __init__(
        self,
        model_name: str = "Marqo/marqo-ecommerce-embeddings-L",
        device: str = None,
        fusion_method: str = "average",
        text_weight: float = 0.5,
    ):
        self.model_name = model_name
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fusion_method = fusion_method
        self.text_weight = text_weight
        self.image_weight = 1.0 - text_weight
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the model from shared pool."""
        from models.open_clip_model_pool import OpenCLIPModelPool

        self.model, self.preprocess, self.tokenizer = OpenCLIPModelPool.get(
            self.model_name, self.device
        )
        print(
            f"[MarqoFusedEmbedder] Using model: {self.model_name} on {self.device}"
        )

    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image from an absolute file path."""
        path = Path(image_path)
        if not path.is_absolute():
            raise ValueError(f"Image path must be absolute. Got: {image_path}")
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return image

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        with torch.no_grad(), torch.amp.autocast(self.device):
            text_tokens = self.tokenizer([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens, normalize=True)
            embedding = text_features.cpu().float().numpy().flatten()
        return embedding

    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for an image from path."""
        image = self._load_image(image_path)
        return self._get_image_embedding_from_pil(image)

    def _get_image_embedding_from_pil(self, image: Image.Image) -> np.ndarray:
        """Generate embedding from a PIL Image object."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.amp.autocast(self.device):
            image_features = self.model.encode_image(image_tensor, normalize=True)
            embedding = image_features.cpu().float().numpy().flatten()
        return embedding

    def _fuse_embeddings(
        self, text_embedding: np.ndarray, image_embedding: np.ndarray
    ) -> np.ndarray:
        """Fuse text and image embeddings based on the fusion method."""
        if self.fusion_method == "average":
            fused = (text_embedding + image_embedding) / 2.0
        elif self.fusion_method == "weighted":
            fused = (
                self.text_weight * text_embedding + self.image_weight * image_embedding
            )
        elif self.fusion_method == "concat":
            fused = np.concatenate([text_embedding, image_embedding])
        else:
            raise ValueError(
                f"Unknown fusion method: {self.fusion_method}. "
                f"Available methods: 'average', 'weighted', 'concat'"
            )

        # Normalize the fused embedding to unit length for cosine similarity
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm

        return fused

    def embed_text_and_image(self, text: str, image_path: str) -> List[float]:
        """Generate fused embedding for a text-image pair."""
        text_embedding = self._get_text_embedding(text)
        image_embedding = self._get_image_embedding(image_path)
        fused_embedding = self._fuse_embeddings(text_embedding, image_embedding)
        return fused_embedding.tolist()

    def embed_text_and_pil_image(self, text: str, image: Image.Image) -> List[float]:
        """Generate fused embedding for a text and PIL Image pair."""
        text_embedding = self._get_text_embedding(text)
        image_embedding = self._get_image_embedding_from_pil(image)
        fused_embedding = self._fuse_embeddings(text_embedding, image_embedding)
        return fused_embedding.tolist()

    def embed_pairs(self, pairs: List[Tuple[str, str]]) -> List[List[float]]:
        """Generate fused embeddings for multiple text-image pairs."""
        embeddings = []
        for text, image_path in pairs:
            fused_embedding = self.embed_text_and_image(text, image_path)
            embeddings.append(fused_embedding)
        return embeddings

    def get_individual_embeddings(
        self, text: str, image_path: str
    ) -> Tuple[List[float], List[float]]:
        """Get individual text and image embeddings (useful for debugging)."""
        text_embedding = self._get_text_embedding(text)
        image_embedding = self._get_image_embedding(image_path)
        return text_embedding.tolist(), image_embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the fused embedding vectors."""
        dummy_text_emb = self._get_text_embedding("test")
        dummy_image = Image.new("RGB", (224, 224), color="white")
        dummy_image_emb = self._get_image_embedding_from_pil(dummy_image)
        fused = self._fuse_embeddings(dummy_text_emb, dummy_image_emb)
        return len(fused)

    def get_base_embedding_dimension(self) -> int:
        """Get the dimension of individual (non-fused) embeddings."""
        dummy_text_emb = self._get_text_embedding("test")
        return len(dummy_text_emb)

    def set_fusion_method(self, method: str, text_weight: float = 0.5):
        """Change the fusion method."""
        if method not in ["average", "weighted", "concat"]:
            raise ValueError(
                f"Unknown fusion method: {method}. "
                f"Available methods: 'average', 'weighted', 'concat'"
            )
        self.fusion_method = method
        self.text_weight = text_weight
        self.image_weight = 1.0 - text_weight
        print(f"[MarqoFusedEmbedder] Fusion method set to: {method}")
