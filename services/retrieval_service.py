from models.clip_model import CLIPWrapper
from utils.preprocessing import clean_text
from functools import lru_cache
from config import Config
import torch

clip_wrapper = CLIPWrapper()


# Optional: in-memory cache
@lru_cache(maxsize=Config.CACHE_MAX_SIZE)
def get_text_embedding(text: str):
    text = clean_text(text)
    return clip_wrapper.encode_text(text)


def generate_embeddings(texts):
    """
    texts: list of strings
    returns: list of embeddings
    """
    embeddings = []
    for text in texts:
        emb = get_text_embedding(text)
        embeddings.append(emb)
    return embeddings


def compare_text_embeddings(text1, text2):
    """
    Compare two texts by generating their embeddings and computing similarity.

    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare

    Returns:
        dict: Contains embeddings and similarity metrics
    """
    # Generate embeddings using existing service function
    embeddings = generate_embeddings([text1, text2])

    embedding1 = embeddings[0]
    embedding2 = embeddings[1]

    # Compute cosine similarity (embeddings are already normalized)
    cosine_similarity = torch.nn.functional.cosine_similarity(
        embedding1, embedding2, dim=-1
    ).item()

    # Compute Euclidean distance
    euclidean_distance = torch.dist(embedding1, embedding2, p=2).item()

    return {
        "embedding1": embedding1,
        "embedding2": embedding2,
        "cosine_similarity": cosine_similarity,
        "euclidean_distance": euclidean_distance,
    }
