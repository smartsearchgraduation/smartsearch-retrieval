"""
Fused Model Manager
Main class for managing and directing text+image pairs to the appropriate fused model.
Acts as a facade for all fused embedding operations in the e-commerce retrieval system.
"""

from typing import List, Dict, Union, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
from PIL import Image


class FusedModelType(Enum):
    """Enum for available fused model types."""

    CLIP = "clip"
    # Future models can be added here
    # BLIP = "blip"
    # FLAVA = "flava"


class FusedModelManager:
    """
    Main manager class for fused models.
    Directs text+image input to the appropriate fused embedding model.
    """

    def __init__(
        self,
        model_type: Union[FusedModelType, str] = FusedModelType.CLIP,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Fused Model Manager.

        Args:
            model_type: Type of fused model to use (default: CLIP).
            model_config: Configuration dictionary for the model.
                         For CLIP: {
                             "model_name": "ViT-B/32",
                             "device": "cpu",
                             "fusion_method": "average",
                             "text_weight": 0.5
                         }
        """
        self.model_config = model_config or {}
        self.model_type = self._parse_model_type(model_type)
        self.model = None
        self._initialize_model()

    def _parse_model_type(
        self, model_type: Union[FusedModelType, str]
    ) -> FusedModelType:
        """Parse model type from string or enum."""
        if isinstance(model_type, FusedModelType):
            return model_type
        elif isinstance(model_type, str):
            try:
                return FusedModelType(model_type.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown model type: {model_type}. "
                    f"Available types: {[t.value for t in FusedModelType]}"
                )
        else:
            raise TypeError(
                f"model_type must be FusedModelType or str, got {type(model_type)}"
            )

    def _initialize_model(self):
        """Initialize the appropriate fused model based on model_type."""
        if self.model_type == FusedModelType.CLIP:
            self._initialize_clip_model()
        else:
            raise NotImplementedError(
                f"Model type {self.model_type} is not implemented yet."
            )

    def _initialize_clip_model(self):
        """Initialize CLIP fused embedder."""
        from .clip_fused_embedder import CLIPFusedEmbedder

        model_name = self.model_config.get("model_name", "ViT-B/32")
        device = self.model_config.get("device", None)
        fusion_method = self.model_config.get("fusion_method", "average")
        text_weight = self.model_config.get("text_weight", 0.5)

        self.model = CLIPFusedEmbedder(
            model_name=model_name,
            device=device,
            fusion_method=fusion_method,
            text_weight=text_weight,
        )
        print(f"[FusedModelManager] Initialized CLIP model: {model_name}")

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

    def get_embedding(self, text: str, image_path: str) -> List[float]:
        """
        Get fused embedding vector for a text-image pair.

        Args:
            text: Input text string.
            image_path: Absolute path to the image file.

        Returns:
            Fused embedding vector as a list of floats.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        self._validate_image_path(image_path)
        return self.model.embed_text_and_image(text, image_path)

    def get_embedding_from_pil(self, text: str, image: Image.Image) -> List[float]:
        """
        Get fused embedding vector for a text and PIL Image pair.

        Args:
            text: Input text string.
            image: PIL Image object.

        Returns:
            Fused embedding vector as a list of floats.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        return self.model.embed_text_and_pil_image(text, image)

    def get_embeddings(self, pairs: List[Tuple[str, str]]) -> List[List[float]]:
        """
        Get fused embedding vectors for multiple text-image pairs.

        Args:
            pairs: List of (text, image_path) tuples.

        Returns:
            List of fused embedding vectors.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        for _, image_path in pairs:
            self._validate_image_path(image_path)

        return self.model.embed_pairs(pairs)

    def get_individual_embeddings(
        self, text: str, image_path: str
    ) -> Tuple[List[float], List[float]]:
        """
        Get individual text and image embeddings separately.

        Args:
            text: Input text string.
            image_path: Absolute path to the image file.

        Returns:
            Tuple of (text_embedding, image_embedding).
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        self._validate_image_path(image_path)
        return self.model.get_individual_embeddings(text, image_path)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of fused embedding vectors.

        Returns:
            Integer dimension of the fused embedding vectors.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        return self.model.get_embedding_dimension()

    def get_base_embedding_dimension(self) -> int:
        """
        Get the dimension of individual (non-fused) embeddings.

        Returns:
            Integer dimension of base embeddings.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        return self.model.get_base_embedding_dimension()

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
            "base_embedding_dimension": self.get_base_embedding_dimension(),
            "fusion_method": self.model.fusion_method if self.model else None,
            "model_class": self.model.__class__.__name__ if self.model else None,
        }

    def set_fusion_method(self, method: str, text_weight: float = 0.5):
        """
        Change the fusion method.

        Args:
            method: Fusion method ('average', 'weighted', 'concat').
            text_weight: Weight for text when using 'weighted' fusion.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        self.model.set_fusion_method(method, text_weight)
        # Update config
        self.model_config["fusion_method"] = method
        self.model_config["text_weight"] = text_weight

    def embed_product(self, product: Dict[str, Any]) -> List[float]:
        """
        Generate fused embedding for a product with text and image.

        Args:
            product: Dictionary containing product information.
                    Expected keys:
                        - 'image_path': Absolute path to product image
                        - At least one of: 'name', 'description', 'category', 'brand', 'tags'

        Returns:
            Fused embedding vector for the product.
        """
        if "image_path" not in product:
            raise ValueError("Product must contain 'image_path' key.")

        # Combine text fields
        text_parts = []
        for key in ["name", "description", "category", "brand"]:
            if key in product and product[key]:
                text_parts.append(str(product[key]))

        if "tags" in product:
            if isinstance(product["tags"], list):
                text_parts.extend(product["tags"])
            elif product["tags"]:
                text_parts.append(str(product["tags"]))

        if not text_parts:
            raise ValueError(
                "Product must contain at least one text field "
                "(name, description, category, brand, or tags)."
            )

        combined_text = " ".join(text_parts)
        return self.get_embedding(combined_text, product["image_path"])

    def embed_products(self, products: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate fused embeddings for multiple products.

        Args:
            products: List of product dictionaries.

        Returns:
            List of fused embedding vectors.
        """
        embeddings = []
        for product in products:
            embedding = self.embed_product(product)
            embeddings.append(embedding)
        return embeddings

    def switch_model(
        self,
        model_type: Union[FusedModelType, str],
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Switch to a different fused model.

        Args:
            model_type: New model type to use.
            model_config: Configuration for the new model.
        """
        self.model_type = self._parse_model_type(model_type)
        self.model_config = model_config or {}
        self.model = None
        self._initialize_model()
        print(f"[FusedModelManager] Switched to model type: {self.model_type.value}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the manager with default CLIP model
    manager = FusedModelManager(
        model_type="clip",
        model_config={
            "model_name": "ViT-B/32",
            "fusion_method": "average",
        },
    )

    # Get model info
    info = manager.get_model_info()
    print(f"Model info: {info}")

    # Test with PIL image
    from PIL import Image

    test_image = Image.new("RGB", (224, 224), color="green")
    text = "Green sneakers with white sole"

    embedding = manager.get_embedding_from_pil(text, test_image)
    print(f"Fused embedding dimension: {len(embedding)}")

    # Test changing fusion method
    manager.set_fusion_method("weighted", text_weight=0.6)
    embedding_weighted = manager.get_embedding_from_pil(text, test_image)
    print(f"Weighted embedding dimension: {len(embedding_weighted)}")

    # Test with concat fusion
    manager.set_fusion_method("concat")
    embedding_concat = manager.get_embedding_from_pil(text, test_image)
    print(f"Concat embedding dimension: {len(embedding_concat)}")

    # To test with actual images, uncomment and modify:
    # product = {
    #     "name": "Premium Sneakers",
    #     "description": "Comfortable running shoes",
    #     "category": "Footwear",
    #     "image_path": "C:\\path\\to\\sneakers.jpg"
    # }
    # product_embedding = manager.embed_product(product)
    # print(f"Product embedding dimension: {len(product_embedding)}")
