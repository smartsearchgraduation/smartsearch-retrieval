"""
Compare DINOv3 vs CLIP on real product images.

Uses VisualModelManager from the Retrieval system to embed product images
with both models, then reports:
  - Intra-product similarity (same product's images — should be high)
  - Inter-product similarity (different products' images — should be low)
  - Ranking: for each image, how CLIP/DINOv3 rank the other images
"""

import os
import sys
import numpy as np
import torch
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.visual_models import VisualModelManager
from services.manager_service import load_config

BACKEND_UPLOADS = r"C:\Users\kaant\Desktop\Grad\Backend\uploads\products"

# Product ID -> list of image files (only picking first 3 per product for speed)
PRODUCTS = {
    "1": [
        "1_8ff7bcef3c304b868c4a3d7d27b39aa6.jpg",
        "1_b1e4d0c2b1cf41618fb408c854c09573.jpg",
        "1_17f5d89625f7446ab593ad41cbc4d981.jpg",
    ],
    "2": [
        "2_8f281ab5e589477092d63c404fd9bd76.jpg",
        "2_39503b7cf69a4730a2f2b4ae9e4286dd.jpg",
        "2_77b627d0f9ca4b28a42678a2709e68da.jpg",
    ],
    "3": [
        "3_a4f7723a5cf44c948d212bd00880ed6e.jpg",
        "3_ae0508fc0c5049a29812892bf8af3011.jpg",
        "3_bdda5113369447a082cfff7bd1e99a27.jpg",
    ],
    "4": [
        "4_82da5507b9f745b1b8180949410f36db.jpg",
        "4_4fa3c145bdb8498ab737efcc74f816fa.jpg",
        "4_ba616606ae514caba27bc10549b4e241.jpg",
    ],
    "5": [
        "5_21368e1d20d349c28e0df06d5f081f7b.jpg",
        "5_29e9d8c69593411b9a670f3aa9b8bc2c.jpg",
        "5_b186f95aea6545b9bf4c38830eb12505.jpg",
    ],
}

CLIP_MODEL = "ViT-B/32"
DINOV3_MODEL = "facebook/dinov3-vit7b16-pretrain-lvd1689m"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def embed_all(manager, flat_items):
    """flat_items: list of (product_id, filename, abs_path). Returns list of np arrays."""
    embeddings = []
    for i, (pid, fname, path) in enumerate(flat_items):
        print(f"  [{i+1}/{len(flat_items)}] product={pid} {fname[:30]}...", flush=True)
        emb = manager.get_embedding(path)
        embeddings.append(np.asarray(emb, dtype=np.float32))
    return embeddings


def analyze(embeddings, flat_items, model_label):
    """Compute intra-product, inter-product similarity stats and mean ranking."""
    n = len(flat_items)

    # Pairwise cosine similarity matrix
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            sim[i, j] = cosine_similarity(embeddings[i], embeddings[j])

    intra, inter = [], []
    for i in range(n):
        for j in range(i + 1, n):
            pid_i = flat_items[i][0]
            pid_j = flat_items[j][0]
            if pid_i == pid_j:
                intra.append(sim[i, j])
            else:
                inter.append(sim[i, j])

    intra = np.array(intra)
    inter = np.array(inter)

    # Ranking quality: for each query image, check if top-K nearest are same-product
    # K = (number of other same-product images)
    correct_at_top = 0
    total_top_slots = 0
    for i in range(n):
        pid_i = flat_items[i][0]
        same_prod = [j for j in range(n) if j != i and flat_items[j][0] == pid_i]
        k = len(same_prod)  # expected top-k
        # Get top-k nearest excluding self
        scores = [(sim[i, j], flat_items[j][0]) for j in range(n) if j != i]
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k_pids = [p for _, p in scores[:k]]
        correct_at_top += sum(1 for p in top_k_pids if p == pid_i)
        total_top_slots += k

    accuracy = correct_at_top / total_top_slots if total_top_slots > 0 else 0.0

    print(f"\n=== {model_label} ===")
    print(f"  Intra-product similarity (aynı ürün):  mean={intra.mean():.4f}  min={intra.min():.4f}  max={intra.max():.4f}  (n={len(intra)})")
    print(f"  Inter-product similarity (farklı):      mean={inter.mean():.4f}  min={inter.min():.4f}  max={inter.max():.4f}  (n={len(inter)})")
    gap = intra.mean() - inter.mean()
    print(f"  Ayrım gücü (intra-inter gap):           {gap:+.4f}")
    print(f"  Top-k recall (aynı ürün komşuları):     {correct_at_top}/{total_top_slots} = {accuracy*100:.1f}%")
    return {"intra_mean": intra.mean(), "inter_mean": inter.mean(), "gap": gap, "accuracy": accuracy}


def main():
    # Flatten items
    flat = []
    for pid, fnames in PRODUCTS.items():
        for fname in fnames:
            path = os.path.join(BACKEND_UPLOADS, fname)
            if not os.path.exists(path):
                print(f"MISSING: {path}")
                continue
            flat.append((pid, fname, path))

    print(f"Total images: {len(flat)} ({len(PRODUCTS)} products)\n")

    if not torch.cuda.is_available():
        print("UYARI: CUDA bulunamadi! Script CPU'da calisacak (cok yavas).")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_config()

    print(f"Loading CLIP ({CLIP_MODEL}) on {device}...")
    clip_manager = VisualModelManager(
        model_type="clip",
        model_config={"model_name": CLIP_MODEL, "device": device},
    )
    print("Embedding with CLIP...")
    clip_embs = embed_all(clip_manager, flat)

    print(f"\nLoading DINOv3 ({DINOV3_MODEL}) on {device}...")
    dino_manager = VisualModelManager(
        model_type="dinov3",
        model_config={"model_name": DINOV3_MODEL, "device": device},
    )
    print("Embedding with DINOv3...")
    dino_embs = embed_all(dino_manager, flat)

    clip_stats = analyze(clip_embs, flat, f"CLIP {CLIP_MODEL}")
    dino_stats = analyze(dino_embs, flat, f"DINOv3 {DINOV3_MODEL}")

    print("\n" + "=" * 60)
    print("KARSILASTIRMA ÖZETİ")
    print("=" * 60)
    print(f"  CLIP gap={clip_stats['gap']:+.4f}  accuracy={clip_stats['accuracy']*100:.1f}%")
    print(f"  DINO gap={dino_stats['gap']:+.4f}  accuracy={dino_stats['accuracy']*100:.1f}%")
    winner = "DINOv3" if dino_stats["gap"] > clip_stats["gap"] else "CLIP"
    print(f"  Daha yüksek ayrım gücü: {winner}")


if __name__ == "__main__":
    main()
