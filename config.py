import torch


class Config:
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # CLIP settings
    CLIP_MODEL_NAME = "ViT-B/32"  # CLIP ViT-Base with 32x32 patches (~350MB)
    CLIP_BATCH_SIZE = 16  # batch size for embeddings

    # Cache settings
    CACHE_ENABLED = True
    CACHE_MAX_SIZE = 1000  # max number of cached embeddings

    # Logging
    LOG_LEVEL = "INFO"
