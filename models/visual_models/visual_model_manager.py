"""
Visual Model Manager
Main class for managing and directing images to the appropriate visual model.
Acts as a facade for all image embedding operations in the e-commerce retrieval system.
"""

from typing import List, Dict, Union, Optional, Any
from enum import Enum
from pathlib import Path
from PIL import Image


class VisualModelType(Enum):
    """Enum for available visual model types."""

    CLIP = "clip"
    MARQO = "marqo"


class VisualModelManager:
    """
    Main manager class for visual models.
    Directs image input to the appropriate image embedding model.
    """

    def __init__(
        self,
        model_type: Union[VisualModelType, str] = VisualModelType.CLIP,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Visual Model Manager.

        Args:
            model_type: Type of visual model to use (default: CLIP).
            model_config: Configuration dictionary for the model.
                         For CLIP: {"model_name": "ViT-B/32", "device": "cpu"}
        """
        self.model_config = model_config or {}
        self.model_type = self._parse_model_type(model_type)
        self.model = None
        self._initialize_model()

    def _parse_model_type(
        self, model_type: Union[VisualModelType, str]
    ) -> VisualModelType:
        """Parse model type from string or enum."""
        if isinstance(model_type, VisualModelType):
            return model_type
        elif isinstance(model_type, str):
            try:
                return VisualModelType(model_type.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown model type: {model_type}. "
                    f"Available types: {[t.value for t in VisualModelType]}"
                )
        else:
            raise TypeError(
                f"model_type must be VisualModelType or str, got {type(model_type)}"
            )

    def _initialize_model(self):
        """Initialize the appropriate visual model based on model_type."""
        if self.model_type == VisualModelType.CLIP:
            self._initialize_clip_model()
        elif self.model_type == VisualModelType.MARQO:
            self._initialize_marqo_model()
        else:
            raise NotImplementedError(
                f"Model type {self.model_type} is not implemented yet."
            )

    def _initialize_clip_model(self):
        """Initialize CLIP image embedder."""
        from .clip_image_embedder import CLIPImageEmbedder

        model_name = self.model_config.get("model_name", "ViT-B/32")
        device = self.model_config.get("device", None)

        self.model = CLIPImageEmbedder(model_name=model_name, device=device)
        print(f"[VisualModelManager] Initialized CLIP model: {model_name}")

    def _initialize_marqo_model(self):
        """Initialize Marqo image embedder."""
        from .marqo_image_embedder import MarqoImageEmbedder

        model_name = self.model_config.get(
            "model_name", "Marqo/marqo-ecommerce-embeddings-L"
        )
        device = self.model_config.get("device", None)

        self.model = MarqoImageEmbedder(model_name=model_name, device=device)
        print(f"[VisualModelManager] Initialized Marqo model: {model_name}")

    def _validate_image_path(self, image_path: str) -> None:
        """
        Validate that the image path is absolute and exists.

        Args:
            image_path: Path to validate.

        Raises:
            ValueError: If path is not absolute.
            FileNotFoundError: If file does not exist.
        """
        path = Path(image_path)
        if not path.is_absolute():
            raise ValueError(f"Image path must be absolute. Got: {image_path}")
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

    def get_embedding(self, image_path: str) -> List[float]:
        """
        Get embedding vector for a single image.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            Embedding vector as a list of floats.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        self._validate_image_path(image_path)
        return self.model.embed_image(image_path)

    def get_embeddings(self, image_paths: List[str]) -> List[List[float]]:
        """
        Get embedding vectors for multiple images.

        Args:
            image_paths: List of absolute paths to image files.

        Returns:
            List of embedding vectors.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        for path in image_paths:
            self._validate_image_path(path)

        return self.model.embed_images(image_paths)

    def get_embedding_from_pil(self, image: Image.Image) -> List[float]:
        """
        Get embedding vector from a PIL Image object.

        Args:
            image: PIL Image object.

        Returns:
            Embedding vector as a list of floats.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        return self.model.embed_pil_image(image)

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
            dummy_image = Image.new("RGB", (224, 224), color="white")
            sample = self.get_embedding_from_pil(dummy_image)
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

    def embed_product_image(self, product: Dict[str, Any]) -> List[float]:
        """
        Generate embedding for a product image.

        Args:
            product: Dictionary containing product information.
                    Expected key: 'image_path' (absolute path to product image).

        Returns:
            Embedding vector for the product image.
        """
        if "image_path" not in product:
            raise ValueError("Product must contain 'image_path' key.")

        return self.get_embedding(product["image_path"])

    def embed_product_images(self, products: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate embeddings for multiple product images.

        Args:
            products: List of product dictionaries with 'image_path' key.

        Returns:
            List of embedding vectors.
        """
        image_paths = []
        for product in products:
            if "image_path" not in product:
                raise ValueError("Each product must contain 'image_path' key.")
            image_paths.append(product["image_path"])

        return self.get_embeddings(image_paths)

    def switch_model(
        self,
        model_type: Union[VisualModelType, str],
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Switch to a different visual model.

        Args:
            model_type: New model type to use.
            model_config: Configuration for the new model.
        """
        self.model_type = self._parse_model_type(model_type)
        self.model_config = model_config or {}
        self.model = None
        self._initialize_model()
        print(f"[VisualModelManager] Switched to model type: {self.model_type.value}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the manager with default CLIP model
    manager = VisualModelManager(
        model_type="clip", model_config={"model_name": "ViT-B/32"}
    )

    # Get model info
    info = manager.get_model_info()
    print(f"Model info: {info}")

    # Test with PIL image
    from PIL import Image

    test_image = Image.new("RGB", (224, 224), color="blue")
    embedding = manager.get_embedding_from_pil(test_image)
    print(f"PIL image embedding dimension: {len(embedding)}")

    # To test with actual images, uncomment and modify paths:
    # image_path = "C:\\path\\to\\your\\image.jpg"
    # embedding = manager.get_embedding(image_path)
    # print(f"Image embedding dimension: {len(embedding)}")

    # Test product image embedding
    # product = {
    #     "name": "Red Sneakers",
    #     "image_path": "C:\\path\\to\\sneakers.jpg"
    # }
    # product_embedding = manager.embed_product_image(product)
    # print(f"Product image embedding dimension: {len(product_embedding)}")
