import torch
from config import Config
from PIL import Image
import clip

from utils.logger import logger


class CLIPWrapper:
    def __init__(self):
        self.device = Config.DEVICE
        logger.info(f"Loading CLIP model on {self.device}...")
        self.model, self.preprocess = clip.load(
            Config.CLIP_MODEL_NAME, device=self.device
        )
        self.model.eval()
        logger.info("CLIP model loaded successfully.")

    @torch.no_grad()
    def encode_text(self, texts):
        """
        texts: list of strings
        returns: tensor of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        tokenized = clip.tokenize(texts).to(self.device)
        embeddings = self.model.encode_text(tokenized)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu()

    # For later extension: image embeddings
    @torch.no_grad()
    def encode_image(self, image: Image.Image):
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(image_input)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu()
