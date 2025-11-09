# smartsearch-retrieval

## Installation and Testing Guide

### 1. Activate Virtual Environment (if not already activated)

```powershell
# Navigate to project directory
cd <your path>

# Create the virtual environment
python -m venv retrieval-venv

# Activate the virtual environment
.\retrieval-venv\Scripts\Activate.ps1
```

**Note:** If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```


### 2. Install all packages
```poweshell
pip install -r requirements.txt
```


---

## 🚀 Running the Application

### Start the Flask Server

```powershell
python app.py
```

You should see output like:
```
INFO - Loading CLIP model on cpu...
INFO - CLIP model loaded successfully.
 * Running on http://0.0.0.0:5000
```

**Note:** First run will download the CLIP model (~350MB), which may take a few minutes.

---

## 🧪 Testing the Endpoints

### Method 1: Using the Test Script (Recommended)

Open a **new PowerShell terminal** (keep the server running in the first one):

```powershell
# Activate virtual environment
.\retrieval-venv\Scripts\Activate.ps1

# Run the test script
python test_endpoints.py
```

This will automatically test all endpoints with various test cases.

---


### Method 2: Using Postman or Insomnia

1. **Open Postman**
2. **Create a new POST request**

**For `/retrieval/send-text`:**
- URL: `http://localhost:5000/retrieval/send-text`
- Method: POST
- Headers: `Content-Type: application/json`
- Body (raw JSON):
  ```json
  {
    "text": "A cat sitting on a couch"
  }
  ```

**For `/retrieval/two-text-test`:**
- URL: `http://localhost:5000/retrieval/two-text-test`
- Method: POST
- Headers: `Content-Type: application/json`
- Body (raw JSON):
  ```json
  {
    "text1": "A dog playing in the park",
    "text2": "A puppy running in a garden"
  }
  ```

---

## 📊 Understanding the Results

### Endpoint 1: `/retrieval/send-text`
Returns a single embedding vector (512 dimensions for CLIP-ViT-Base).

**Response Structure:**
```json
{
  "embedding": [0.123, 0.456, ..., 0.789]  // 512 float values
}
```

### Endpoint 2: `/retrieval/two-text-test`
Returns embeddings for both texts plus similarity metrics.

**Response Structure:**
```json
{
  "text1": "First text",
  "text2": "Second text",
  "embedding1": [0.123, ...],  // 512 dimensions
  "embedding2": [0.456, ...],  // 512 dimensions
  "cosine_similarity": 0.85,   // Range: -1 to 1
  "euclidean_distance": 1.23   // Range: 0 to infinity
}
```

**Interpreting Similarity Metrics:**

| Metric | Range | Similar Texts | Dissimilar Texts |
|--------|-------|---------------|------------------|
| **Cosine Similarity** | -1 to 1 | Close to 1.0 | Close to 0.0 or negative |
| **Euclidean Distance** | 0 to ∞ | Close to 0.0 | Higher values |

**Examples:**
- Identical texts: `cosine_similarity ≈ 1.0`, `euclidean_distance ≈ 0.0`
- Similar concepts: `cosine_similarity ≈ 0.7-0.9`, `euclidean_distance ≈ 0.5-1.5`
- Unrelated texts: `cosine_similarity ≈ 0.0-0.4`, `euclidean_distance > 2.0`

---

## 🔍 Architecture Analysis

Based on codebase analysis, the system uses:

### **Imports & Dependencies:**
- `app.py`: Flask, jsonify, request
- `clip_model.py`: torch, clip, PIL, sentence_transformers
- `retrieval_service.py`: torch, functools.lru_cache
- `config.py`: torch (for CUDA detection)

### **Model Details:**
- **CLIP Model:** `openai/clip-vit-base-patch32`
- **Embedding Dimension:** 512
- **Device:** Auto-detects CUDA (GPU) or CPU
- **Caching:** LRU cache with max 1000 embeddings

### **Module Responsibilities:**
1. **app.py** - API routes and HTTP handling
2. **services/retrieval_service.py** - Business logic, embedding generation, comparison
3. **models/clip_model.py** - CLIP model wrapper, encoding operations
4. **utils/preprocessing.py** - Text cleaning
5. **utils/logger.py** - Logging configuration
6. **config.py** - Centralized configuration

---

## 🐛 Troubleshooting

**Issue: "Module not found"**
- Solution: Make sure virtual environment is activated and packages are installed

**Issue: "CUDA out of memory"**
- Solution: Model will automatically fall back to CPU if CUDA fails

**Issue: "Cannot connect to server"**
- Solution: Make sure Flask app is running (`python app.py`)

**Issue: "Model download fails"**
- Solution: Check internet connection; CLIP model downloads from GitHub on first run

**Issue: "Import error: No module named 'clip'"**
- Solution: Install CLIP from GitHub: `pip install git+https://github.com/openai/CLIP.git`

---

