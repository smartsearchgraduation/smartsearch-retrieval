"""
BGE Base Embedder
Generates text embeddings using BAAI/ bge-large-en-v1.5 model for e-commerce product retrieval.
"""

import torch
import numpy as np
from typing import List
from langchain.embeddings.base import Embeddings


class BGEBaseEmbedder(Embeddings):
    """
    A LangChain-compatible text embedder using BAAI/ bge-large-en-v1.5 model.
    Generates text embeddings for e-commerce product descriptions and queries.

    BGE (BAAI General Embedding) models are optimized for retrieval tasks
    and produce 1024-dimensional embeddings.
    """

    def __init__(self, model_name: str = "BAAI/ bge-large-en-v1.5", device: str = None):
        """
        Initialize the BGE Base Embedder.

        Args:
            model_name: HuggingFace model name. Default is "BAAI/ bge-large-en-v1.5".
            device: Device to run the model on ('cuda' or 'cpu').
                    Auto-detected if None.
        """
        self.model_name = model_name
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = None
        self.model = None
        self._embedding_dimension = 1024  #  bge-large models produce 1024-dim embeddings
        self._load_model()

    def _load_model(self):
        """Load the BGE model and tokenizer from HuggingFace."""
        try:
            from transformers import AutoTokenizer, AutoModel

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(
                f"[BGEBaseEmbedder] Loaded BGE model: {self.model_name} on {self.device}"
            )
        except ImportError:
            raise ImportError(
                "Transformers is not installed. Please install it via: "
                "pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load BGE model: {e}")

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text string.

        Returns:
            Numpy array of the text embedding vector.
        """
        # BGE models recommend adding instruction prefix for queries
        # For retrieval, queries should use "Represent this sentence for searching relevant passages: "
        # But for document/passage encoding, no prefix is needed

        with torch.no_grad():
            # Tokenize the text
            encoded_input = self.tokenizer(
                text, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)

            # Get model output
            model_output = self.model(**encoded_input)

            # Use CLS token embedding (first token) as the sentence embedding
            # BGE models are trained to use CLS pooling
            embedding = model_output.last_hidden_state[:, 0, :]

            # Normalize the embedding
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            # Convert to numpy array
            embedding = embedding.cpu().numpy().flatten()

        return embedding

    def _get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input text strings.

        Returns:
            Numpy array of shape (num_texts, embedding_dim).
        """
        with torch.no_grad():
            # Tokenize all texts
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # Get model output
            model_output = self.model(**encoded_input)

            # Use CLS token embedding
            embeddings = model_output.last_hidden_state[:, 0, :]

            # Normalize the embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Convert to numpy array
            embeddings = embeddings.cpu().numpy()

        return embeddings

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

        For BGE models, queries can optionally be prefixed with an instruction
        for better retrieval performance. This method adds the recommended prefix.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        # BGE recommends adding instruction prefix for queries
        query_instruction = "Represent this sentence for searching relevant passages: "
        prefixed_text = query_instruction + text
        embedding = self._get_text_embedding(prefixed_text)
        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Integer dimension of embeddings (1024 for  bge-large models).
        """
        return self._embedding_dimension

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for texts in batches for better efficiency.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process at once.

        Returns:
            Numpy array of shape (num_texts, embedding_dim).
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self._get_batch_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the embedder
    embedder = BGEBaseEmbedder()

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

    # Test batch embedding
    batch_embeddings = embedder.embed_batch(sample_texts)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")

    # Test get_embedding_dimension
    dim = embedder.get_embedding_dimension()
    print(f"Embedding dimension: {dim}")
