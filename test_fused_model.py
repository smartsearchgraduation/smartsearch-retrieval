from models.fused_models import FusedModelManager

# Initialize with average fusion
manager = FusedModelManager(
    model_type="clip",
    model_config={"model_name": "ViT-B/32", "fusion_method": "average"},
)

# Get fused embedding
embedding = manager.get_embedding(
    text="Red leather handbag", image_path="C:\\images\\product.jpg"
)

# Embed a complete product
product = {
    "name": "Premium Handbag",
    "description": "Elegant red leather",
    "category": "Accessories",
    "image_path": "C:\\images\\product.jpg",
}
product_embedding = manager.embed_product(product)

# Switch to weighted fusion (70% text, 30% image)
manager.set_fusion_method("weighted", text_weight=0.7)

print("Fused Embedding:", embedding)
print("Product Fused Embedding:", product_embedding)
