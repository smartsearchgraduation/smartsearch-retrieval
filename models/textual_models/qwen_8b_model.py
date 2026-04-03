"""
Qwen3 Embedding 8B Model
Generates text embeddings using Qwen/Qwen3-Embedding-8B model for e-commerce product retrieval.
"""

import torch
import numpy as np
from typing import List
from langchain.embeddings.base import Embeddings


class Qwen8BEmbedder(Embeddings):
    """
    A LangChain-compatible text embedder using Qwen/Qwen3-Embedding-8B model.
    Generates text embeddings for e-commerce product descriptions and queries.

    Qwen3-Embedding-8B produces 4096-dimensional embeddings optimized for
    semantic search and retrieval tasks.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B", device: str = None):
        """
        Initialize the Qwen 8B Embedder.

        Args:
            model_name: HuggingFace model name. Default is "Qwen/Qwen3-Embedding-8B".
            device: Device to run the model on ('cuda' or 'cpu').
                    Auto-detected if None.
        """
        self.model_name = model_name
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = None
        self.model = None
        self._embedding_dimension = 4096  # Qwen3-Embedding-8B produces 4096-dim embeddings
        self._load_model()

    def _load_model(self):
        """Load the Qwen model and tokenizer from HuggingFace."""
        try:
            from transformers import AutoTokenizer, AutoModel

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.model.to(self.device)
            self.model.eval()
            print(
                f"[Qwen8BEmbedder] Loaded Qwen model: {self.model_name} on {self.device}"
            )
        except ImportError:
            raise ImportError(
                "Transformers is not installed. Please install it via: "
                "pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen model: {e}")

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text string.

        Returns:
            Numpy array of the text embedding vector.
        """
        with torch.no_grad():
            # Tokenize the text
            encoded_input = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=8192,  # Qwen3-Embedding supports long context
                return_tensors="pt",
            ).to(self.device)

            # Get model output
            model_output = self.model(**encoded_input)

            # Use last token embedding for Qwen embedding models (decoder-based)
            last_hidden_state = model_output.last_hidden_state
            attention_mask = encoded_input["attention_mask"]

            # Last-token pooling: find the position of the last real token per sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1  # 0-indexed
            batch_indices = torch.arange(last_hidden_state.size(0), device=self.device)
            embedding = last_hidden_state[batch_indices, sequence_lengths]

            # Normalize the embedding
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            # Convert to numpy array
            embedding = embedding.cpu().numpy().flatten()

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
        # Qwen embedding models may benefit from query instruction prefix
        # Add instruction for better retrieval performance
        query_instruction = "Instruct: Given a query, retrieve relevant documents that answer the query.\nQuery: "
        prefixed_text = query_instruction + text
        embedding = self._get_text_embedding(prefixed_text)
        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Integer dimension of embeddings (4096 for Qwen3-Embedding-8B).
        """
        return self._embedding_dimension


# Example usage and testing
if __name__ == "__main__":
    # Initialize the embedder
    embedder = Qwen8BEmbedder()

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

    # Test get_embedding_dimension
    dim = embedder.get_embedding_dimension()
    print(f"Embedding dimension: {dim}")
