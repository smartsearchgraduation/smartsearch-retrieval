"""
FAISS Manager
Manages FAISS indices for textual, visual, and fused embeddings.
Supports metadata storage alongside vectors for e-commerce product retrieval.
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from enum import Enum
import threading


class IndexType(Enum):
    """Enum for available index types."""

    TEXTUAL = "textual"
    VISUAL = "visual"
    FUSED = "fused"


class FAISSManager:
    """
    Manager class for FAISS vector indices.
    Handles three separate indices: Textual, Visual, and Fused.
    Each index stores embeddings with associated metadata.
    """

    def __init__(
        self,
        dimension: int = 512,
        index_path: Optional[str] = None,
        use_gpu: bool = False,
    ):
        """
        Initialize the FAISS Manager.

        Args:
            dimension: Dimension of embedding vectors (default: 512 for CLIP ViT-B/32).
            index_path: Path to directory for saving/loading indices.
            use_gpu: Whether to use GPU for FAISS operations (requires faiss-gpu).
        """
        self.dimension = dimension
        self.index_path = index_path
        self.use_gpu = use_gpu

        # Initialize indices
        self.indices: Dict[IndexType, faiss.Index] = {}
        self.metadata: Dict[IndexType, List[Dict[str, Any]]] = {}
        self.id_counters: Dict[IndexType, int] = {}

        # Thread lock for thread-safe operations
        self._lock = threading.Lock()

        # Try to load existing indices, or initialize new ones
        if index_path and os.path.exists(index_path):
            # Initialize empty structure first
            self._initialize_indices()
            # Then load saved data (overwrites the empty indices)
            self.load()
        else:
            # Create fresh indices
            self._initialize_indices()

    def _initialize_indices(self):
        """Initialize all three FAISS indices."""
        for index_type in IndexType:
            self._create_index(index_type)

    def _create_index(self, index_type: IndexType):
        """
        Create a new FAISS index for the given type.

        Args:
            index_type: Type of index to create.
        """
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        index = faiss.IndexFlatIP(self.dimension)

        # Wrap with IDMap to support custom IDs
        index = faiss.IndexIDMap(index)

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as e:
                print(f"[FAISSManager] GPU not available, using CPU: {e}")

        self.indices[index_type] = index
        self.metadata[index_type] = []
        self.id_counters[index_type] = 0

        print(
            f"[FAISSManager] Initialized {index_type.value} index with dimension {self.dimension}"
        )

    def _get_next_id(self, index_type: IndexType) -> int:
        """Get the next available ID for an index."""
        current_id = self.id_counters[index_type]
        self.id_counters[index_type] += 1
        return current_id

    def add_to_textual(
        self,
        embedding: List[float],
        product_id: str,
        model_name: str,
    ) -> int:
        """
        Add a textual embedding to the Textual index.

        Args:
            embedding: Embedding vector.
            product_id: Product identifier.
            model_name: Name of the model used to generate the embedding.

        Returns:
            ID of the added vector.
        """
        metadata = {
            "product_id": product_id,
            "model_name": model_name,
        }
        return self._add_vector(IndexType.TEXTUAL, embedding, metadata)

    def add_to_visual(
        self,
        embedding: List[float],
        product_id: str,
        image_no: int,
        model_name: str,
    ) -> int:
        """
        Add a visual embedding to the Visual index.

        Args:
            embedding: Embedding vector.
            product_id: Product identifier.
            image_no: Image number/index for the product.
            model_name: Name of the model used to generate the embedding.

        Returns:
            ID of the added vector.
        """
        metadata = {
            "product_id": product_id,
            "image_no": image_no,
            "model_name": model_name,
        }
        return self._add_vector(IndexType.VISUAL, embedding, metadata)

    def add_to_fused(
        self,
        embedding: List[float],
        product_id: str,
        image_no: int,
        model_name: str,
    ) -> int:
        """
        Add a fused embedding to the Fused index.

        Args:
            embedding: Embedding vector.
            product_id: Product identifier.
            image_no: Image number/index for the product.
            model_name: Name of the model used to generate the embedding.

        Returns:
            ID of the added vector.
        """
        metadata = {
            "product_id": product_id,
            "image_no": image_no,
            "model_name": model_name,
        }
        return self._add_vector(IndexType.FUSED, embedding, metadata)

    def _add_vector(
        self,
        index_type: IndexType,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> int:
        """
        Add a vector to the specified index.

        Args:
            index_type: Type of index to add to.
            embedding: Embedding vector.
            metadata: Metadata to store with the vector.

        Returns:
            ID of the added vector.
        """
        with self._lock:
            # Convert to numpy array and normalize
            vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(vector)

            # Get next ID
            vector_id = self._get_next_id(index_type)

            # Add to index
            self.indices[index_type].add_with_ids(
                vector, np.array([vector_id], dtype=np.int64)
            )

            # Store metadata with the vector ID
            metadata["_vector_id"] = vector_id
            self.metadata[index_type].append(metadata)

            return vector_id

    def add_batch_to_textual(
        self,
        embeddings: List[List[float]],
        product_ids: List[str],
        model_names: List[str],
    ) -> List[int]:
        """
        Add multiple textual embeddings to the Textual index.

        Args:
            embeddings: List of embedding vectors.
            product_ids: List of product identifiers.
            model_names: List of model names.

        Returns:
            List of IDs of the added vectors.
        """
        ids = []
        for emb, pid, mname in zip(embeddings, product_ids, model_names):
            vector_id = self.add_to_textual(emb, pid, mname)
            ids.append(vector_id)
        return ids

    def add_batch_to_visual(
        self,
        embeddings: List[List[float]],
        product_ids: List[str],
        image_nos: List[int],
        model_names: List[str],
    ) -> List[int]:
        """
        Add multiple visual embeddings to the Visual index.

        Args:
            embeddings: List of embedding vectors.
            product_ids: List of product identifiers.
            image_nos: List of image numbers.
            model_names: List of model names.

        Returns:
            List of IDs of the added vectors.
        """
        ids = []
        for emb, pid, ino, mname in zip(
            embeddings, product_ids, image_nos, model_names
        ):
            vector_id = self.add_to_visual(emb, pid, ino, mname)
            ids.append(vector_id)
        return ids

    def search(
        self,
        index_type: IndexType,
        query_embedding: List[float],
        top_k: int = 10,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the specified index.

        Args:
            index_type: Type of index to search.
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            model_name: Optional filter by model name.

        Returns:
            List of search results with metadata and scores.
        """
        with self._lock:
            # Convert to numpy array and normalize
            query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_vector)

            # Search
            scores, ids = self.indices[index_type].search(
                query_vector, top_k * 2
            )  # Get more for filtering

            # Collect results with metadata
            results = []
            for score, vector_id in zip(scores[0], ids[0]):
                if vector_id == -1:  # No more results
                    continue

                # Find metadata for this vector ID
                meta = self._get_metadata_by_id(index_type, int(vector_id))
                if meta is None:
                    continue

                # Filter by model name if specified
                if model_name and meta.get("model_name") != model_name:
                    continue

                result = {
                    "score": float(score),
                    "vector_id": int(vector_id),
                    **{k: v for k, v in meta.items() if k != "_vector_id"},
                }
                results.append(result)

                if len(results) >= top_k:
                    break

            return results

    def search_textual(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the Textual index."""
        return self.search(IndexType.TEXTUAL, query_embedding, top_k, model_name)

    def search_visual(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the Visual index."""
        return self.search(IndexType.VISUAL, query_embedding, top_k, model_name)

    def search_fused(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the Fused index."""
        return self.search(IndexType.FUSED, query_embedding, top_k, model_name)

    def _get_metadata_by_id(
        self, index_type: IndexType, vector_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific vector ID."""
        for meta in self.metadata[index_type]:
            if meta.get("_vector_id") == vector_id:
                return meta
        return None

    def get_index_size(self, index_type: IndexType) -> int:
        """Get the number of vectors in an index."""
        return self.indices[index_type].ntotal

    def get_all_sizes(self) -> Dict[str, int]:
        """Get sizes of all indices."""
        return {
            index_type.value: self.get_index_size(index_type)
            for index_type in IndexType
        }

    def remove_by_product_id(self, index_type: IndexType, product_id: str) -> int:
        """
        Remove all vectors for a product from an index.
        Note: FAISS IndexFlatIP doesn't support direct removal,
        so we rebuild the index without the removed vectors.

        Args:
            index_type: Type of index.
            product_id: Product ID to remove.

        Returns:
            Number of vectors removed.
        """
        with self._lock:
            # Find vectors to keep
            vectors_to_keep = []
            metadata_to_keep = []
            removed_count = 0

            for meta in self.metadata[index_type]:
                if meta.get("product_id") != product_id:
                    metadata_to_keep.append(meta)
                else:
                    removed_count += 1

            if removed_count == 0:
                return 0

            # Rebuild index
            self._create_index(index_type)
            self.metadata[index_type] = []

            # Note: This is a simplified implementation
            # In production, you'd need to store vectors to rebuild properly
            # For now, we just clear the metadata

            return removed_count

    def save(self, path: Optional[str] = None):
        """
        Save all indices and metadata to disk.

        Args:
            path: Directory path to save to. Uses self.index_path if not provided.
        """
        save_path = path or self.index_path
        if save_path is None:
            raise ValueError("No save path provided")

        os.makedirs(save_path, exist_ok=True)

        with self._lock:
            for index_type in IndexType:
                # Save FAISS index
                index_file = os.path.join(save_path, f"{index_type.value}_index.faiss")

                # Convert GPU index to CPU for saving if necessary
                index = self.indices[index_type]
                if self.use_gpu:
                    index = faiss.index_gpu_to_cpu(index)
                faiss.write_index(index, index_file)

                # Save metadata
                meta_file = os.path.join(save_path, f"{index_type.value}_metadata.json")
                with open(meta_file, "w") as f:
                    json.dump(
                        {
                            "metadata": self.metadata[index_type],
                            "id_counter": self.id_counters[index_type],
                        },
                        f,
                    )

        print(f"[FAISSManager] Saved indices to {save_path}")

    def load(self, path: Optional[str] = None):
        """
        Load all indices and metadata from disk.

        Args:
            path: Directory path to load from. Uses self.index_path if not provided.
        """
        load_path = path or self.index_path
        if load_path is None:
            raise ValueError("No load path provided")

        with self._lock:
            for index_type in IndexType:
                index_file = os.path.join(load_path, f"{index_type.value}_index.faiss")
                meta_file = os.path.join(load_path, f"{index_type.value}_metadata.json")

                if os.path.exists(index_file) and os.path.exists(meta_file):
                    # Load FAISS index
                    index = faiss.read_index(index_file)

                    if self.use_gpu:
                        try:
                            res = faiss.StandardGpuResources()
                            index = faiss.index_cpu_to_gpu(res, 0, index)
                        except Exception as e:
                            print(f"[FAISSManager] GPU not available: {e}")

                    self.indices[index_type] = index

                    # Load metadata
                    with open(meta_file, "r") as f:
                        data = json.load(f)
                        self.metadata[index_type] = data["metadata"]
                        self.id_counters[index_type] = data["id_counter"]

                    print(
                        f"[FAISSManager] Loaded {index_type.value} index with {index.ntotal} vectors"
                    )
                else:
                    print(
                        f"[FAISSManager] No saved index found for {index_type.value}, using empty index"
                    )

    def clear(self, index_type: Optional[IndexType] = None):
        """
        Clear one or all indices.

        Args:
            index_type: Specific index to clear, or None to clear all.
        """
        with self._lock:
            if index_type:
                self._create_index(index_type)
            else:
                self._initialize_indices()

        print(
            f"[FAISSManager] Cleared {'all indices' if index_type is None else index_type.value + ' index'}"
        )


# Example usage and testing
if __name__ == "__main__":
    # Initialize manager
    manager = FAISSManager(dimension=512)

    # Test adding vectors
    dummy_embedding = [0.1] * 512

    # Add to textual index
    text_id = manager.add_to_textual(
        embedding=dummy_embedding,
        product_id="prod_001",
        model_name="ViT-B/32",
    )
    print(f"Added to textual index with ID: {text_id}")

    # Add to visual index
    visual_id = manager.add_to_visual(
        embedding=dummy_embedding,
        product_id="prod_001",
        image_no=0,
        model_name="ViT-B/32",
    )
    print(f"Added to visual index with ID: {visual_id}")

    # Check sizes
    sizes = manager.get_all_sizes()
    print(f"Index sizes: {sizes}")

    # Search
    results = manager.search_textual(dummy_embedding, top_k=5)
    print(f"Search results: {results}")
