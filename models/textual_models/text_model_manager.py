"""
Text Model Manager
Main class for managing and directing text to the appropriate textual model.
Acts as a facade for all text embedding operations in the e-commerce retrieval system.
"""

import numpy as np
from typing import List, Dict, Union, Optional, Any
from enum import Enum


class TextModelType(Enum):
    """Enum for available text model types."""

    CLIP = "clip"
    BGE = "bge"
    # Future models can be added here
    # SENTENCE_TRANSFORMER = "sentence_transformer"
    # OPENAI = "openai"


class TextModelManager:
    """
    Main manager class for textual models.
    Directs text input to the appropriate text embedding model.
    """

    def __init__(
        self,
        model_type: Union[TextModelType, str] = TextModelType.CLIP,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Text Model Manager.

        Args:
            model_type: Type of text model to use (default: CLIP).
            model_config: Configuration dictionary for the model.
                         For CLIP: {"model_name": "ViT-B/32", "device": "cpu"}
        """
        self.model_config = model_config or {}
        self.model_type = self._parse_model_type(model_type)
        self.model = None
        self._initialize_model()

    def _parse_model_type(self, model_type: Union[TextModelType, str]) -> TextModelType:
        """Parse model type from string or enum."""
        if isinstance(model_type, TextModelType):
            return model_type
        elif isinstance(model_type, str):
            try:
                return TextModelType(model_type.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown model type: {model_type}. "
                    f"Available types: {[t.value for t in TextModelType]}"
                )
        else:
            raise TypeError(
                f"model_type must be TextModelType or str, got {type(model_type)}"
            )

    def _initialize_model(self):
        """Initialize the appropriate text model based on model_type."""
        if self.model_type == TextModelType.CLIP:
            self._initialize_clip_model()
        elif self.model_type == TextModelType.BGE:
            self._initialize_bge_model()
        else:
            raise NotImplementedError(
                f"Model type {self.model_type} is not implemented yet."
            )

    def _initialize_clip_model(self):
        """Initialize CLIP text embedder."""
        from .clip_text_embedder import CLIPTextEmbedder

        model_name = self.model_config.get("model_name", "ViT-B/32")
        device = self.model_config.get("device", None)

        self.model = CLIPTextEmbedder(model_name=model_name, device=device)
        print(f"[TextModelManager] Initialized CLIP model: {model_name}")

    def _initialize_bge_model(self):
        """Initialize BGE text embedder."""
        from .bge_base_embedder import BGEBaseEmbedder

        model_name = self.model_config.get("model_name", "BAAI/bge-base-en-v1.5")
        device = self.model_config.get("device", None)

        self.model = BGEBaseEmbedder(model_name=model_name, device=device)
        print(f"[TextModelManager] Initialized BGE model: {model_name}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for a single text.

        Args:
            text: Input text string.

        Returns:
            Embedding vector as a list of floats.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        return self.model.embed_query(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embedding vectors for multiple texts.

        Args:
            texts: List of input text strings.

        Returns:
            List of embedding vectors.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        return self.model.embed_documents(texts)

    def get_embeddings_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> np.ndarray:
        """
        Get embedding vectors for texts using batch processing.

        Args:
            texts: List of input text strings.
            batch_size: Batch size for processing.

        Returns:
            Numpy array of embeddings with shape (num_texts, embedding_dim).
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        if hasattr(self.model, "embed_batch"):
            return self.model.embed_batch(texts, batch_size=batch_size)
        else:
            # Fallback to regular embedding
            embeddings = self.model.embed_documents(texts)
            return np.array(embeddings)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embedding vectors.

        Returns:
            Integer dimension of the embedding vectors.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        else:
            # Get dimension from a sample embedding
            sample = self.get_embedding("test")
            return len(sample)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information.
        """
        return {
            "model_type": self.model_type.value,
            "model_config": self.model_config,
            "embedding_dimension": self.get_embedding_dimension(),
            "model_class": self.model.__class__.__name__ if self.model else None,
        }

    def embed_product(self, product: Dict[str, str]) -> List[float]:
        """
        Generate embedding for a product by combining its text fields.

        Args:
            product: Dictionary containing product information.
                    Expected keys: 'name', 'description', 'category', etc.

        Returns:
            Combined embedding vector for the product.
        """
        # Combine relevant text fields
        text_parts = []

        if "name" in product:
            text_parts.append(product["name"])
        if "description" in product:
            text_parts.append(product["description"])
        if "category" in product:
            text_parts.append(product["category"])
        if "brand" in product:
            text_parts.append(product["brand"])
        if "tags" in product:
            if isinstance(product["tags"], list):
                text_parts.extend(product["tags"])
            else:
                text_parts.append(product["tags"])

        combined_text = " ".join(text_parts)
        return self.get_embedding(combined_text)

    def embed_products(self, products: List[Dict[str, str]]) -> List[List[float]]:
        """
        Generate embeddings for multiple products.

        Args:
            products: List of product dictionaries.

        Returns:
            List of embedding vectors.
        """
        combined_texts = []
        for product in products:
            text_parts = []
            for key in ["name", "description", "category", "brand"]:
                if key in product:
                    text_parts.append(str(product[key]))
            if "tags" in product:
                if isinstance(product["tags"], list):
                    text_parts.extend(product["tags"])
                else:
                    text_parts.append(product["tags"])
            combined_texts.append(" ".join(text_parts))

        return self.get_embeddings(combined_texts)

    def switch_model(
        self,
        model_type: Union[TextModelType, str],
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Switch to a different text model.

        Args:
            model_type: New model type to use.
            model_config: Configuration for the new model.
        """
        self.model_type = self._parse_model_type(model_type)
        self.model_config = model_config or {}
        self.model = None
        self._initialize_model()
        print(f"[TextModelManager] Switched to model type: {self.model_type.value}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the manager with default CLIP model
    manager = TextModelManager(
        model_type="clip", model_config={"model_name": "ViT-B/32"}
    )

    # Test single text embedding
    text = "Elegant black dress with lace details"
    embedding = manager.get_embedding(text)
    print(f"Single embedding dimension: {len(embedding)}")

    # Test multiple text embeddings
    texts = [
        "Comfortable running shoes",
        "Vintage leather wallet",
        "Modern smartwatch with fitness tracking",
    ]
    embeddings = manager.get_embeddings(texts)
    print(f"Number of embeddings: {len(embeddings)}")

    # Test product embedding
    product = {
        "name": "Premium Wireless Headphones",
        "description": "High-quality noise-cancelling headphones with 30-hour battery life",
        "category": "Electronics",
        "brand": "AudioTech",
        "tags": ["wireless", "bluetooth", "noise-cancelling"],
    }
    product_embedding = manager.embed_product(product)
    print(f"Product embedding dimension: {len(product_embedding)}")

    # Get model info
    info = manager.get_model_info()
    print(f"Model info: {info}")
