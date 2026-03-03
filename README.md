# SmartSearch Retrieval System

An E-Commerce Product Retrieval System built with Flask and OpenAI's CLIP model. This system enables semantic similarity search for products using textual descriptions, visual images, or a fusion of both modalities.

## 🚀 Features

- **Text-based Search**: Search products using natural language queries
- **Image-based Search**: Find similar products using image queries
- **Late Fusion Search**: Combine text and image searches with weighted scoring
- **Multi-modal Embeddings**: Generate embeddings for text, images, or fused text+image
- **FAISS Vector Database**: Fast similarity search with persistent storage
- **RESTful API**: Easy-to-use Flask-based API endpoints

## 📁 Project Structure

```
smartsearch-retrieval/
├── app.py                          # Flask app entry point (creates app, registers blueprints)
├── config.json                     # Model registry and application defaults
├── requirements.txt                # Python dependencies
├── data/                           # Data storage directory
│   └── faiss_indices/              # Persistent FAISS index storage
├── routes/
│   ├── __init__.py
│   ├── product_routes.py           # Add, update, delete product endpoints
│   ├── search_routes.py            # Text, image, late fusion search endpoints
│   └── system_routes.py            # Health check, index stats endpoints
├── services/
│   ├── __init__.py
│   └── manager_service.py          # Model manager initialization and config loading
├── utils/
│   ├── __init__.py
│   └── validation.py               # Request validation helpers
├── models/
│   ├── clip_model_pool.py          # Shared CLIP model pool (avoids duplicate loading)
│   ├── textual_models/
│   │   ├── __init__.py
│   │   ├── clip_text_embedder.py   # CLIP text embedding implementation
│   │   ├── bge_base_embedder.py    # BGE text embedding implementation
│   │   ├── qwen_8b_model.py        # Qwen text embedding implementation
│   │   └── text_model_manager.py   # Text model management facade
│   ├── visual_models/
│   │   ├── __init__.py
│   │   ├── clip_image_embedder.py  # CLIP image embedding implementation
│   │   └── visual_model_manager.py # Visual model management facade
│   └── fused_models/
│       ├── __init__.py
│       ├── clip_fused_embedder.py  # CLIP fused embedding implementation
│       └── fused_model_manager.py  # Fused model management facade
└── vector_db/
    ├── __init__.py
    └── faiss_manager.py            # FAISS index management
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/smartsearchgraduation/smartsearch-retrieval.git
   cd smartsearch-retrieval
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note**: The CLIP package is installed from GitHub: `git+https://github.com/openai/CLIP.git`

4. **For GPU support** (optional)
   - Replace `faiss-cpu` with `faiss-gpu` in requirements.txt
   - Install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/)

## 🚀 Running the Application

Start the Flask server:

```bash
python app.py
```

The server will start on `http://0.0.0.0:5000`

## 📡 API Endpoints

### Health Check

```http
GET /api/health
```

**Response:**
```json
{
    "status": "healthy",
    "service": "E-Commerce Product Retrieval System"
}
```

### Add Product

```http
POST /api/retrieval/add-product
```

**Request Body:**
```json
{
    "id": "product_001",
    "name": "Premium Leather Handbag",
    "description": "Elegant handcrafted leather bag",
    "brand": "LuxuryBrand",
    "category": "Accessories",
    "price": 299.99,
    "images": ["C:/absolute/path/to/image1.jpg", "C:/absolute/path/to/image2.jpg"],
    "textual_model_name": "ViT-B/32",
    "visual_model_name": "ViT-B/32"
}
```

**Response:**
```json
{
    "status": "success",
    "message": "Product product_001 added successfully",
    "details": {
        "product_id": "product_001",
        "textual_vector_id": 0,
        "visual_vector_ids": [0, 1],
        "images_processed": 2
    }
}
```

### Text Search

```http
POST /api/retrieval/search/text
```

**Request Body:**
```json
{
    "text": "red leather handbag",
    "textual_model_name": "ViT-B/32",
    "top_k": 10
}
```

**Response:**
```json
{
    "status": "success",
    "results": [
        {
            "product_id": "product_001",
            "score": 0.856432
        }
    ],
    "meta": {
        "total_results": 1,
        "model_name": "ViT-B/32"
    }
}
```

### Late Fusion Search

Combines text and image search with weighted scoring.

```http
POST /api/retrieval/search/late
```

**Request Body:**
```json
{
    "text": "red leather handbag",
    "textual_model_name": "ViT-B/32",
    "text_weight": 0.5,
    "image": "C:/absolute/path/to/query_image.jpg",
    "visual_model_name": "ViT-B/32",
    "top_k": 10
}
```

**Response:**
```json
{
    "status": "success",
    "results": [
        {
            "product_id": "product_001",
            "combined_score": 0.842156,
            "text_score": 0.856432,
            "image_score": 0.827880,
            "best_image_no": 0
        }
    ],
    "meta": {
        "text_weight": 0.5,
        "image_weight": 0.5,
        "total_results": 1
    }
}
```

### Update Product

```http
PUT /api/retrieval/update-product/<product_id>
```

Atomically removes old embeddings and re-indexes with new data.

**Request Body:**
```json
{
    "name": "Updated Leather Handbag",
    "description": "Premium handmade leather bag with gold buckle",
    "brand": "LuxuryBrand",
    "category": "Accessories",
    "price": 349.99,
    "images": ["C:/absolute/path/to/new_image.jpg"],
    "textual_model_name": "BAAI/bge-large-en-v1.5",
    "visual_model_name": "ViT-B/32"
}
```

**Response:**
```json
{
    "status": "success",
    "message": "Product product_001 updated successfully",
    "details": {
        "product_id": "product_001",
        "removed_counts": { "textual": 1, "visual": 2, "fused": 0 },
        "textual_vector_id": 2,
        "visual_vector_ids": [3],
        "images_processed": 1
    }
}
```

### Delete Product

```http
DELETE /api/retrieval/delete-product/<product_id>
```

Removes all embeddings for a product from all FAISS indices.

**Response:**
```json
{
    "status": "success",
    "message": "Product product_001 deleted successfully",
    "details": {
        "product_id": "product_001",
        "removed_counts": { "textual": 1, "visual": 2, "fused": 0 },
        "total_removed": 3
    }
}
```

### Image Search

Search products using an image query.

```http
POST /api/retrieval/search/image
```

**Request Body:**
```json
{
    "image": "C:/absolute/path/to/query_image.jpg",
    "visual_model_name": "ViT-B/32",
    "top_k": 10
}
```

**Response:**
```json
{
    "status": "success",
    "results": [
        {
            "product_id": "product_001",
            "score": 0.827880,
            "best_image_no": 0
        }
    ],
    "meta": {
        "total_results": 1,
        "model_name": "ViT-B/32"
    }
}
```

### Index Statistics

```http
GET /api/retrieval/index-stats
```

**Response:**
```json
{
    "status": "success",
    "indices": {
        "textual": 100,
        "visual": 250,
        "fused": 0
    }
}
```

## 🔧 Model Options

The system supports various CLIP model variants:

| Model | Description |
|-------|-------------|
| `ViT-B/32` | Default, balanced speed/accuracy |
| `ViT-B/16` | Higher accuracy |
| `ViT-L/14` | Highest accuracy, slower |
| `RN50` | ResNet-50 backbone |
| `RN101` | ResNet-101 backbone |
| `RN50x4` | ResNet-50 with 4x width |
| `RN50x16` | ResNet-50 with 16x width |
| `RN50x64` | ResNet-50 with 64x width |

## 📚 Usage Examples

### Text Embedding

```python
from models.textual_models import TextModelManager

manager = TextModelManager(model_type="clip", model_config={"model_name": "ViT-B/32"})

# Get embedding for a query
embedding = manager.get_embedding("Red leather handbag")

# Embed a product
product = {
    "name": "Premium Headphones",
    "description": "Wireless noise-cancelling",
    "category": "Electronics",
}
product_embedding = manager.embed_product(product)
```

### Visual Embedding

```python
from models.visual_models import VisualModelManager

manager = VisualModelManager(model_type="clip", model_config={"model_name": "ViT-B/32"})

# Get image embedding (absolute path required)
embedding = manager.get_embedding("C:/images/product.jpg")

# From PIL Image
from PIL import Image
img = Image.open("C:/images/product.jpg")
embedding = manager.get_embedding_from_pil(img)
```

### Fused Embedding

```python
from models.fused_models import FusedModelManager

manager = FusedModelManager(
    model_type="clip",
    model_config={"model_name": "ViT-B/32", "fusion_method": "average"},
)

# Get fused embedding
embedding = manager.get_embedding(
    text="Red leather handbag",
    image_path="C:/images/product.jpg"
)

# Switch to weighted fusion (70% text, 30% image)
manager.set_fusion_method("weighted", text_weight=0.7)
```

## 🔍 Fusion Methods

The fused embedder supports three fusion strategies:

| Method | Description |
|--------|-------------|
| `average` | Simple average of text and image embeddings |
| `weighted` | Weighted average based on `text_weight` parameter |
| `concat` | Concatenation of embeddings (doubles dimension) |

## 📝 Important Notes

- **Image paths must be absolute**: The system requires absolute file paths for all image inputs
- **Embedding dimension**: Default is 512 (for CLIP ViT-B/32)
- **Index persistence**: FAISS indices are saved to `./data/faiss_indices/`
- **Thread safety**: The FAISS manager uses thread locks for safe concurrent operations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

SmartSearch Graduation Team
