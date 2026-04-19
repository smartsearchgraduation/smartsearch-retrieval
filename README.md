# SmartSearch Retrieval System

An E-Commerce Product Retrieval System built with Flask, supporting multiple embedding models including OpenAI CLIP and Marqo's ecommerce-optimized model. This system enables semantic similarity search for products using textual descriptions, visual images, or a fusion of both modalities.

## 🚀 Features

- **Text-based Search**: Search products using natural language queries
- **Image-based Search**: Find similar products using image queries
- **Late Fusion Search**: Combine text and image searches with weighted scoring
- **Early Fusion Search**: Fuse text+image into a single query embedding using CLIP's shared space
- **Cross-Modal Search**: Search images with text or text with images using CLIP's shared embedding space
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
│   ├── bge-large-en-v1.5_1024_embeddings/   # BGE model indices
│   ├── ViT-B-32_512_embeddings/             # CLIP model indices
│   └── Qwen3-Embedding-8B_4096_embeddings/  # Qwen model indices
├── routes/
│   ├── __init__.py
│   ├── product_routes.py           # Add, update, delete product endpoints
│   ├── search_routes.py            # Text, image, late/early fusion, cross-modal search endpoints
│   └── system_routes.py            # Health check, index stats, models endpoints
├── services/
│   ├── __init__.py
│   └── manager_service.py          # Model manager initialization and config loading
├── utils/
│   ├── __init__.py
│   └── validation.py               # Request validation helpers
├── models/
│   ├── clip_model_pool.py          # Shared CLIP model pool (avoids duplicate loading)
│   ├── open_clip_model_pool.py     # Shared OpenCLIP model pool (for Marqo models)
│   ├── textual_models/
│   │   ├── __init__.py
│   │   ├── clip_text_embedder.py   # CLIP text embedding implementation
│   │   ├── marqo_text_embedder.py  # Marqo text embedding implementation
│   │   ├── bge_base_embedder.py    # BGE text embedding implementation
│   │   ├── qwen_8b_model.py        # Qwen text embedding implementation
│   │   └── text_model_manager.py   # Text model management facade
│   ├── visual_models/
│   │   ├── __init__.py
│   │   ├── clip_image_embedder.py  # CLIP image embedding implementation
│   │   ├── marqo_image_embedder.py # Marqo image embedding implementation
│   │   └── visual_model_manager.py # Visual model management facade
│   └── fused_models/
│       ├── __init__.py
│       ├── clip_fused_embedder.py  # CLIP fused embedding implementation
│       ├── marqo_fused_embedder.py # Marqo fused embedding implementation
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
   > The Marqo model requires `open_clip_torch` which is included in requirements.txt.

4. **For GPU support** (optional)
   - Replace `faiss-cpu` with `faiss-gpu` in requirements.txt
   - Install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/)

## 🚀 Running the Application

Start the Flask server:

```bash
python app.py
```

The server will start on `http://0.0.0.0:5002`

## 📡 API Endpoints

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

### Early Fusion Search

Fuses text and image into a single query embedding using a shared multimodal space, then searches the Fused index. Requires products to be indexed with `fused_model_name` during add-product. Only multimodal models (CLIP, Marqo) are supported since both modalities must share the same embedding space.

```http
POST /api/retrieval/search/early
```

**Request Body:**
```json
{
    "text": "red leather handbag",
    "image": "C:/absolute/path/to/query_image.jpg",
    "fused_model_name": "ViT-B/32",
    "text_weight": 0.5,
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
            "score": 0.842156,
            "best_image_no": 0
        }
    ],
    "meta": {
        "total_results": 1,
        "model_name": "ViT-B/32",
        "text_weight": 0.5
    }
}
```

### Image Search by Text (Cross-Modal)

Find products using a text query against the Fused index. Encodes the text with the multimodal model's text encoder and searches the Fused index. Only multimodal models (CLIP, Marqo) are supported. Requires products to be indexed with `fused_model_name` during add-product.

```http
POST /api/retrieval/search/image-by-text
```

**Request Body:**
```json
{
    "text": "red leather handbag",
    "fused_model_name": "Marqo/marqo-ecommerce-embeddings-L",
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
        "model_name": "Marqo/marqo-ecommerce-embeddings-L"
    }
}
```

### Text Search by Image (Cross-Modal)

Find products using an image query against the Fused index. Encodes the image with the multimodal model's image encoder and searches the Fused index. Only multimodal models (CLIP, Marqo) are supported. Requires products to be indexed with `fused_model_name` during add-product.

```http
POST /api/retrieval/search/text-by-image
```

**Request Body:**
```json
{
    "image": "C:/absolute/path/to/query_image.jpg",
    "fused_model_name": "Marqo/marqo-ecommerce-embeddings-L",
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
            "score": 0.812345,
            "best_image_no": 0
        }
    ],
    "meta": {
        "total_results": 1,
        "model_name": "Marqo/marqo-ecommerce-embeddings-L"
    }
}
```

### Add Product

Adds a product to the retrieval system. If the product already has embeddings for the active model, the request is skipped and a success response with `"skipped": true` is returned.

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
    "visual_model_name": "ViT-B/32",
    "fused_model_name": "ViT-B/32"
}
```

**Response (new product):**
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

**Response (already exists for active model):**
```json
{
    "status": "success",
    "message": "Product product_001 already has embeddings for this model, skipping",
    "details": {
        "product_id": "product_001",
        "skipped": true
    }
}
```

### Update Product

```http
PUT /api/retrieval/update-product/<product_id>
```

Removes old embeddings from **all** model folders, then re-indexes with the active model.

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
        "removed_counts": {
            "bge-large-en-v1.5_1024_embeddings": { "textual": 1, "visual": 0, "fused": 0 },
            "ViT-B-32_512_embeddings": { "textual": 0, "visual": 2, "fused": 0 }
        },
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

Removes all embeddings for a product from **all** model folders.

**Response:**
```json
{
    "status": "success",
    "message": "Product product_001 deleted successfully",
    "details": {
        "product_id": "product_001",
        "removed_counts": {
            "bge-large-en-v1.5_1024_embeddings": { "textual": 1, "visual": 0, "fused": 0 },
            "ViT-B-32_512_embeddings": { "textual": 0, "visual": 2, "fused": 0 }
        },
        "total_removed": 3
    }
}
```

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

### Index Statistics

Returns per-model index statistics.

```http
GET /api/retrieval/index-stats
```

**Response:**
```json
{
    "status": "success",
    "indices": {
        "bge-large-en-v1.5_1024_embeddings": {
            "textual": 100,
            "visual": 0,
            "fused": 0
        },
        "ViT-B-32_512_embeddings": {
            "textual": 0,
            "visual": 250,
            "fused": 0
        }
    }
}
```

### Available Models

List all available embedding models categorized by type, along with default selections.

```http
GET /api/retrieval/models
```

**Response:**
```json
{
    "status": "success",
    "data": {
        "textual_models": [
            { "name": "ViT-B/32", "dimension": 512 },
            { "name": "BAAI/bge-large-en-v1.5", "dimension": 1024 },
            { "name": "Qwen/Qwen3-Embedding-8B", "dimension": 4096 },
            { "name": "Marqo/marqo-ecommerce-embeddings-L", "dimension": 1024 }
        ],
        "visual_models": [
            { "name": "ViT-B/32", "dimension": 512 },
            { "name": "Marqo/marqo-ecommerce-embeddings-L", "dimension": 1024 }
        ],
        "defaults": {
            "textual": "BAAI/bge-large-en-v1.5",
            "visual": "ViT-B/32"
        }
    }
}
```

## 🔧 Model Options

The system supports the following models (configured in `config.json`):

| Model | Type | Dimension | Modality | Description |
|-------|------|-----------|----------|-------------|
| `ViT-B/32` | CLIP | 512 | Text + Image | Default, balanced speed/accuracy |
| `ViT-B/16` | CLIP | 512 | Text + Image | Higher accuracy |
| `BAAI/bge-large-en-v1.5` | BGE | 1024 | Text only | High-quality text embeddings |
| `Qwen/Qwen3-Embedding-8B` | Qwen | 4096 | Text only | Large-scale text embeddings |
| `Marqo/marqo-ecommerce-embeddings-L` | Marqo | 1024 | Text + Image | E-commerce optimized multimodal (652M params, SigLIP) |

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
- **Index persistence**: Each model stores its FAISS indices in a separate folder under `./data/` (e.g., `./data/bge-large-en-v1.5_1024_embeddings/`)
- **Cross-model operations**: Update and delete operations remove product embeddings from all model folders
- **Thread safety**: The FAISS manager uses thread locks for safe concurrent operations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

SmartSearch Graduation Team
