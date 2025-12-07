from models.visual_models import VisualModelManager

# Initialize
manager = VisualModelManager(model_type="clip", model_config={"model_name": "ViT-B/32"})

# Get image embedding (absolute path required)
embedding = manager.get_embedding("C:\\images\\product.jpg")

# Embed a product
product = {"name": "Red Sneakers", "image_path": "C:\\images\\sneakers.jpg"}
product_embedding = manager.embed_product_image(product)

# From PIL Image
from PIL import Image

img = Image.open("C:\\images\\product.jpg")
embedding = manager.get_embedding_from_pil(img)
print("Image Embedding:", embedding)
print("Product Image Embedding:", product_embedding)
