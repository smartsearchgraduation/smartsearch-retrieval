"""
CLIP Text Embedder using LangChain
Generates text embeddings using OpenAI's CLIP model for e-commerce product retrieval.
"""

import torch
import numpy as np
from typing import List
from langchain.embeddings.base import Embeddings


class CLIPTextEmbedder(Embeddings):
    """
    A LangChain-compatible text embedder using OpenAI's CLIP model.
    Generates text embeddings for e-commerce product descriptions and queries.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize the CLIP Text Embedder.

        Args:
            model_name: CLIP model variant to use. Options include:
                        - "ViT-B/32" (default, balanced speed/accuracy)
                        - "ViT-B/16" (higher accuracy)
                        - "ViT-L/14" (highest accuracy, slower)
                        - "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"
            device: Device to run the model on ('cuda' or 'cpu').
                    Auto-detected if None.
        """
        self.model_name = model_name
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.preprocess = None
        self._load_model()

    def _load_model(self):
        """Load the CLIP model from shared pool."""
        from models.clip_model_pool import CLIPModelPool

        self.model, self.preprocess = CLIPModelPool.get(self.model_name, self.device)
        print(
            f"[CLIPTextEmbedder] Using CLIP model: {self.model_name} on {self.device}"
        )

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text string.

        Returns:
            Numpy array of the text embedding vector.
        """
        import clip

        with torch.no_grad():
            # Tokenize and encode the text
            text_tokens = clip.tokenize([text], truncate=True).to(self.device)
            text_features = self.model.encode_text(text_tokens)

            # Normalize the features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Convert to numpy array
            embedding = text_features.cpu().numpy().flatten()

        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        Required by LangChain Embeddings interface.

        Args:
            texts: List of text documents to embed.

        Returns:
            List of embedding vectors as lists of floats.
        """
        embeddings = []
        for text in texts:
            embedding = self._get_text_embedding(text)
            embeddings.append(embedding.tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        Required by LangChain Embeddings interface.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        embedding = self._get_text_embedding(text)
        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Integer dimension of embeddings (e.g., 512 for ViT-B/32).
        """
        # Get dimension by encoding a dummy text
        dummy_embedding = self._get_text_embedding("test")
        return len(dummy_embedding)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the embedder
    embedder = CLIPTextEmbedder(model_name="ViT-B/32")

    # Test with sample e-commerce product descriptions
    sample_texts = [
        "Red leather handbag with gold chain strap",
        "Blue denim jacket with vintage wash",
        "White sneakers with comfortable sole",
    ]

    # Test embed_documents (LangChain interface)
    embeddings = embedder.embed_documents(sample_texts)
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # Test embed_query (LangChain interface)
    query = "stylish leather bag"
    query_embedding = embedder.embed_query(query)
    print(f"Query embedding dimension: {len(query_embedding)}")
