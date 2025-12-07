from models.textual_models import TextModelManager

# Initialize
manager = TextModelManager(model_type="clip", model_config={"model_name": "ViT-B/32"})

# Get text embedding
embedding = manager.get_embedding("Red leather handbag")

# Embed a product
product = {
    "name": "Premium Headphones",
    "description": "Wireless noise-cancelling",
    "category": "Electronics",
}
product_embedding = manager.embed_product(product)
print("Text Embedding:", embedding)
print("Product Embedding:", product_embedding)
