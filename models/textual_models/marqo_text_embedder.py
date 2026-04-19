"""
Marqo Text Embedder
Generates text embeddings using Marqo's ecommerce embedding model via OpenCLIP.
"""

import torch
import numpy as np
from typing import List
from langchain.embeddings.base import Embeddings


class MarqoTextEmbedder(Embeddings):
    """
    A LangChain-compatible text embedder using Marqo's ecommerce model.
    Generates text embeddings for e-commerce product descriptions and queries.
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
            f"[MarqoTextEmbedder] Using model: {self.model_name} on {self.device}"
        )

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        with torch.no_grad(), torch.amp.autocast(self.device):
            text_tokens = self.tokenizer([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens, normalize=True)
            embedding = text_features.cpu().float().numpy().flatten()
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        embeddings = []
        for text in texts:
            embedding = self._get_text_embedding(text)
            embeddings.append(embedding.tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        embedding = self._get_text_embedding(text)
        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        dummy_embedding = self._get_text_embedding("test")
        return len(dummy_embedding)
