"""
DINOv3 Image Embedder
Generates image embeddings using Meta's DINOv3 ViT-7B pretrained model
(facebook/dinov3-vit7b16-pretrain-lvd1689m) for e-commerce product retrieval.

DINOv3 is a vision-only self-supervised model: it has no text encoder, so it
supports image search and the visual branch of late fusion, but not text,
early fusion, or cross-modal search.
"""

import torch
import numpy as np
from typing import List
from pathlib import Path
from PIL import Image


class DINOv3ImageEmbedder:
    """
    Image embedder using Meta's DINOv3 ViT-7B model.
    Generates image embeddings for e-commerce product images.

    Default pooling strategy: CLS token + L2 normalization.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        device: str = None,
    ):
        """
        Initialize the DINOv3 Image Embedder.

        Args:
            model_name: HuggingFace model ID. Default is
                        "facebook/dinov3-vit7b16-pretrain-lvd1689m".
            device: Device to run the model on ('cuda' or 'cpu').
                    Auto-detected if None.
        """
        self.model_name = model_name
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.processor = None
        self._embedding_dimension = 4096  # DINOv3 ViT-7B hidden size
        self._load_model()

    def _load_model(self):
        """Load the model from the shared pool."""
        from models.dinov3_model_pool import DINOv3ModelPool

        self.model, self.processor = DINOv3ModelPool.get(self.model_name, self.device)
        print(
            f"[DINOv3ImageEmbedder] Using model: {self.model_name} on {self.device}"
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

    def _pool_cls(self, outputs) -> torch.Tensor:
        """CLS-token pooling: prefer pooler_output if available, else first token."""
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        return outputs.last_hidden_state[:, 0, :]

    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for a single image from path."""
        image = self._load_image(image_path)
        return self._get_embedding_from_pil(image)

    def _get_embedding_from_pil(self, image: Image.Image) -> np.ndarray:
        """Generate embedding from a PIL Image object."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        if self.device == "cuda":
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                outputs = self.model(**inputs)
                pooled = self._pool_cls(outputs)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embedding = pooled.cpu().float().numpy().flatten()
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
                pooled = self._pool_cls(outputs)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embedding = pooled.cpu().float().numpy().flatten()

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
        return self._embedding_dimension


# Example usage and testing
if __name__ == "__main__":
    # Initialize the embedder
    embedder = DINOv3ImageEmbedder(
        model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m"
    )

    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")

    # Test with PIL image
    test_image = Image.new("RGB", (224, 224), color="red")
    embedding = embedder.embed_pil_image(test_image)
    print(f"PIL image embedding dimension: {len(embedding)}")

    # To test with actual images, uncomment and modify paths:
    # image_path = "C:\\path\\to\\your\\image.jpg"
    # embedding = embedder.embed_image(image_path)
    # print(f"Image embedding dimension: {len(embedding)}")
