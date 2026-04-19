"""
Marqo Image Embedder
Generates image embeddings using Marqo's ecommerce embedding model via OpenCLIP.
"""

import torch
import numpy as np
from typing import List
from pathlib import Path
from PIL import Image


class MarqoImageEmbedder:
    """
    Image embedder using Marqo's ecommerce model.
    Generates image embeddings for e-commerce product images.
    """

    def __init__(
        self,
        model_name: str = "Marqo/marqo-ecommerce-embeddings-L",
        device: str = None,
    ):
        self.model_name = model_name
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
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
            f"[MarqoImageEmbedder] Using model: {self.model_name} on {self.device}"
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

    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for a single image from path."""
        image = self._load_image(image_path)
        return self._get_embedding_from_pil(image)

    def _get_embedding_from_pil(self, image: Image.Image) -> np.ndarray:
        """Generate embedding from a PIL Image object."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.amp.autocast(self.device):
            image_features = self.model.encode_image(image_tensor, normalize=True)
            embedding = image_features.cpu().float().numpy().flatten()
        return embedding

    def embed_image(self, image_path: str) -> List[float]:
        """Generate embedding for a single image."""
        embedding = self._get_image_embedding(image_path)
        return embedding.tolist()

    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple images."""
        embeddings = []
        for image_path in image_paths:
            embedding = self._get_image_embedding(image_path)
            embeddings.append(embedding.tolist())
        return embeddings

    def embed_pil_image(self, image: Image.Image) -> List[float]:
        """Generate embedding for a PIL Image object."""
        embedding = self._get_embedding_from_pil(image)
        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        dummy_image = Image.new("RGB", (224, 224), color="white")
        embedding = self._get_embedding_from_pil(dummy_image)
        return len(embedding)
