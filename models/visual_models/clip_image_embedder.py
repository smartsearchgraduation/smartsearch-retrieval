"""
CLIP Image Embedder
Generates image embeddings using OpenAI's CLIP model for e-commerce product retrieval.
"""

import torch
import numpy as np
from typing import List
from pathlib import Path
from PIL import Image


class CLIPImageEmbedder:
    """
    Image embedder using OpenAI's CLIP model.
    Generates image embeddings for e-commerce product images.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize the CLIP Image Embedder.

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
        """Load the CLIP model and preprocessing function."""
        try:
            import clip

            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            print(
                f"[CLIPImageEmbedder] Loaded CLIP model: {self.model_name} on {self.device}"
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

    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for a single image.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            Numpy array of the image embedding vector.
        """
        # Load and preprocess the image
        image = self._load_image(image_path)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Encode the image
            image_features = self.model.encode_image(image_tensor)

            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to numpy array
            embedding = image_features.cpu().numpy().flatten()

        return embedding

    def _get_embedding_from_pil(self, image: Image.Image) -> np.ndarray:
        """
        Generate embedding from a PIL Image object.

        Args:
            image: PIL Image object.

        Returns:
            Numpy array of the image embedding vector.
        """
        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().flatten()

        return embedding

    def embed_image(self, image_path: str) -> List[float]:
        """
        Generate embedding for a single image.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            Embedding vector as a list of floats.
        """
        embedding = self._get_image_embedding(image_path)
        return embedding.tolist()

    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple images.

        Args:
            image_paths: List of absolute paths to image files.

        Returns:
            List of embedding vectors as lists of floats.
        """
        embeddings = []
        for image_path in image_paths:
            embedding = self._get_image_embedding(image_path)
            embeddings.append(embedding.tolist())
        return embeddings

    def embed_pil_image(self, image: Image.Image) -> List[float]:
        """
        Generate embedding for a PIL Image object.

        Args:
            image: PIL Image object.

        Returns:
            Embedding vector as a list of floats.
        """
        embedding = self._get_embedding_from_pil(image)
        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Integer dimension of embeddings (e.g., 512 for ViT-B/32).
        """
        # Create a dummy image to get embedding dimension
        dummy_image = Image.new("RGB", (224, 224), color="white")
        embedding = self._get_embedding_from_pil(dummy_image)
        return len(embedding)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the embedder
    embedder = CLIPImageEmbedder(model_name="ViT-B/32")

    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")

    # To test with actual images, uncomment and modify paths:
    # image_path = "C:\\path\\to\\your\\image.jpg"
    # embedding = embedder.embed_image(image_path)
    # print(f"Image embedding dimension: {len(embedding)}")

    # Test with PIL image
    from PIL import Image

    test_image = Image.new("RGB", (224, 224), color="red")
    embedding = embedder.embed_pil_image(test_image)
    print(f"PIL image embedding dimension: {len(embedding)}")
