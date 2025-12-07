"""
CLIP Fused Embedder
Generates fused embeddings combining text and image using OpenAI's CLIP model.
Used for e-commerce product retrieval where both text and image are available.
"""

import torch
import numpy as np
from typing import List, Tuple
from pathlib import Path
from PIL import Image


class CLIPFusedEmbedder:
    """
    Fused embedder using OpenAI's CLIP model.
    Combines text and image embeddings into a single fused embedding.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = None,
        fusion_method: str = "average",
        text_weight: float = 0.5,
    ):
        """
        Initialize the CLIP Fused Embedder.

        Args:
            model_name: CLIP model variant to use. Options include:
                        - "ViT-B/32" (default, balanced speed/accuracy)
                        - "ViT-B/16" (higher accuracy)
                        - "ViT-L/14" (highest accuracy, slower)
                        - "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"
            device: Device to run the model on ('cuda' or 'cpu').
                    Auto-detected if None.
            fusion_method: Method to fuse text and image embeddings.
                          - "average": Simple average of both embeddings
                          - "weighted": Weighted average based on text_weight
                          - "concat": Concatenate embeddings (doubles dimension)
            text_weight: Weight for text embedding when using "weighted" fusion.
                        Image weight = 1 - text_weight.
        """
        self.model_name = model_name
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fusion_method = fusion_method
        self.text_weight = text_weight
        self.image_weight = 1.0 - text_weight
        self.model = None
        self.preprocess = None
        self._load_model()

    def _load_model(self):
        """Load the CLIP model and preprocessing function."""
        try:
            import clip

            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            print(
                f"[CLIPFusedEmbedder] Loaded CLIP model: {self.model_name} on {self.device}"
            )
        except ImportError:
            raise ImportError(
                "CLIP is not installed. Please install it via: "
                "pip install git+https://github.com/openai/CLIP.git"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")

    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from an absolute file path.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            PIL Image object.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the path is not absolute.
        """
        path = Path(image_path)

        # Validate absolute path
        if not path.is_absolute():
            raise ValueError(f"Image path must be absolute. Got: {image_path}")

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load and convert to RGB (in case of RGBA or grayscale)
        image = Image.open(image_path).convert("RGB")
        return image

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Input text string.

        Returns:
            Numpy array of the text embedding vector (normalized).
        """
        import clip

        with torch.no_grad():
            text_tokens = clip.tokenize([text], truncate=True).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embedding = text_features.cpu().numpy().flatten()

        return embedding

    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for an image.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            Numpy array of the image embedding vector (normalized).
        """
        image = self._load_image(image_path)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().flatten()

        return embedding

    def _get_image_embedding_from_pil(self, image: Image.Image) -> np.ndarray:
        """
        Generate embedding from a PIL Image object.

        Args:
            image: PIL Image object.

        Returns:
            Numpy array of the image embedding vector (normalized).
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().flatten()

        return embedding

    def _fuse_embeddings(
        self, text_embedding: np.ndarray, image_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Fuse text and image embeddings based on the fusion method.

        Args:
            text_embedding: Text embedding vector.
            image_embedding: Image embedding vector.

        Returns:
            Fused embedding vector.
        """
        if self.fusion_method == "average":
            # Simple average
            fused = (text_embedding + image_embedding) / 2.0

        elif self.fusion_method == "weighted":
            # Weighted average
            fused = (
                self.text_weight * text_embedding + self.image_weight * image_embedding
            )

        elif self.fusion_method == "concat":
            # Concatenation (doubles the dimension)
            fused = np.concatenate([text_embedding, image_embedding])

        else:
            raise ValueError(
                f"Unknown fusion method: {self.fusion_method}. "
                f"Available methods: 'average', 'weighted', 'concat'"
            )

        # Normalize the fused embedding (except for concat which may have different semantics)
        if self.fusion_method != "concat":
            fused = fused / np.linalg.norm(fused)

        return fused

    def embed_text_and_image(self, text: str, image_path: str) -> List[float]:
        """
        Generate fused embedding for a text-image pair.

        Args:
            text: Input text string.
            image_path: Absolute path to the image file.

        Returns:
            Fused embedding vector as a list of floats.
        """
        text_embedding = self._get_text_embedding(text)
        image_embedding = self._get_image_embedding(image_path)
        fused_embedding = self._fuse_embeddings(text_embedding, image_embedding)

        return fused_embedding.tolist()

    def embed_text_and_pil_image(self, text: str, image: Image.Image) -> List[float]:
        """
        Generate fused embedding for a text and PIL Image pair.

        Args:
            text: Input text string.
            image: PIL Image object.

        Returns:
            Fused embedding vector as a list of floats.
        """
        text_embedding = self._get_text_embedding(text)
        image_embedding = self._get_image_embedding_from_pil(image)
        fused_embedding = self._fuse_embeddings(text_embedding, image_embedding)

        return fused_embedding.tolist()

    def embed_pairs(self, pairs: List[Tuple[str, str]]) -> List[List[float]]:
        """
        Generate fused embeddings for multiple text-image pairs.

        Args:
            pairs: List of (text, image_path) tuples.

        Returns:
            List of fused embedding vectors.
        """
        embeddings = []
        for text, image_path in pairs:
            fused_embedding = self.embed_text_and_image(text, image_path)
            embeddings.append(fused_embedding)
        return embeddings

    def get_individual_embeddings(
        self, text: str, image_path: str
    ) -> Tuple[List[float], List[float]]:
        """
        Get individual text and image embeddings (useful for debugging).

        Args:
            text: Input text string.
            image_path: Absolute path to the image file.

        Returns:
            Tuple of (text_embedding, image_embedding) as lists of floats.
        """
        text_embedding = self._get_text_embedding(text)
        image_embedding = self._get_image_embedding(image_path)

        return text_embedding.tolist(), image_embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the fused embedding vectors.

        Returns:
            Integer dimension of embeddings.
        """
        # Create dummy embeddings to get dimension
        dummy_text_emb = self._get_text_embedding("test")
        dummy_image = Image.new("RGB", (224, 224), color="white")
        dummy_image_emb = self._get_image_embedding_from_pil(dummy_image)
        fused = self._fuse_embeddings(dummy_text_emb, dummy_image_emb)

        return len(fused)

    def get_base_embedding_dimension(self) -> int:
        """
        Get the dimension of individual (non-fused) embeddings.

        Returns:
            Integer dimension of base embeddings (e.g., 512 for ViT-B/32).
        """
        dummy_text_emb = self._get_text_embedding("test")
        return len(dummy_text_emb)

    def set_fusion_method(self, method: str, text_weight: float = 0.5):
        """
        Change the fusion method.

        Args:
            method: Fusion method ('average', 'weighted', 'concat').
            text_weight: Weight for text when using 'weighted' fusion.
        """
        if method not in ["average", "weighted", "concat"]:
            raise ValueError(
                f"Unknown fusion method: {method}. "
                f"Available methods: 'average', 'weighted', 'concat'"
            )
        self.fusion_method = method
        self.text_weight = text_weight
        self.image_weight = 1.0 - text_weight
        print(f"[CLIPFusedEmbedder] Fusion method set to: {method}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the embedder
    embedder = CLIPFusedEmbedder(
        model_name="ViT-B/32",
        fusion_method="average",
    )

    print(f"Base embedding dimension: {embedder.get_base_embedding_dimension()}")
    print(f"Fused embedding dimension: {embedder.get_embedding_dimension()}")

    # Test with PIL image
    from PIL import Image

    test_image = Image.new("RGB", (224, 224), color="red")
    text = "A red product image"

    embedding = embedder.embed_text_and_pil_image(text, test_image)
    print(f"Fused embedding dimension: {len(embedding)}")

    # Test weighted fusion
    embedder.set_fusion_method("weighted", text_weight=0.7)
    embedding_weighted = embedder.embed_text_and_pil_image(text, test_image)
    print(f"Weighted fused embedding dimension: {len(embedding_weighted)}")

    # Test concat fusion
    embedder.set_fusion_method("concat")
    embedding_concat = embedder.embed_text_and_pil_image(text, test_image)
    print(f"Concat fused embedding dimension: {len(embedding_concat)}")
