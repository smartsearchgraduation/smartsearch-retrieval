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

            # Use last token embedding for Qwen embedding models (similar to decoder models)
            # Or use mean pooling over all tokens
            last_hidden_state = model_output.last_hidden_state
            attention_mask = encoded_input["attention_mask"]

            # Mean pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embedding = sum_embeddings / sum_mask

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
                max_length=8192,
                return_tensors="pt",
            ).to(self.device)

            # Get model output
            model_output = self.model(**encoded_input)

            # Mean pooling
            last_hidden_state = model_output.last_hidden_state
            attention_mask = encoded_input["attention_mask"]

            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

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

    def embed_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings for texts in batches for better efficiency.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process at once (smaller for large model).

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

    # Test batch embedding
    batch_embeddings = embedder.embed_batch(sample_texts)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")

    # Test get_embedding_dimension
    dim = embedder.get_embedding_dimension()
    print(f"Embedding dimension: {dim}")
