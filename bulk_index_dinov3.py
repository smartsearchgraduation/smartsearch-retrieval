"""
Bulk index ALL product images into the DINOv3 FAISS visual index.

Scans Backend/uploads/products/ for files matching <product_id>_<hash>.jpg,
groups by product_id, and adds each product's images to the DINOv3 visual
index. Skips products already indexed (resumable).

Only touches the DINOv3 visual index — existing textual/visual/fused
indexes for other models (BGE, CLIP, etc.) are untouched.

Usage:
    python bulk_index_dinov3.py              # index all products
    python bulk_index_dinov3.py --limit 50   # index only first 50 products (for testing)
    python bulk_index_dinov3.py --dry-run    # list what would be indexed, no model load
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.manager_service import load_config, get_faiss_manager
from vector_db.faiss_manager import IndexType

BACKEND_UPLOADS = r"C:\Users\kaant\Desktop\Grad\Backend\uploads\products"
DINOV3_MODEL = "facebook/dinov3-vit7b16-pretrain-lvd1689m"

FILENAME_PATTERN = re.compile(r"^(\d+)_.+\.(jpg|jpeg|png|gif|webp|avif|jfif)$", re.IGNORECASE)


def scan_products(uploads_folder: str):
    """Scan the uploads folder, group image files by product_id prefix."""
    if not os.path.isdir(uploads_folder):
        raise FileNotFoundError(f"Uploads folder not found: {uploads_folder}")

    groups = defaultdict(list)
    for fname in sorted(os.listdir(uploads_folder)):
        path = os.path.join(uploads_folder, fname)
        if not os.path.isfile(path):
            continue
        m = FILENAME_PATTERN.match(fname)
        if not m:
            continue
        pid = m.group(1)
        groups[pid].append(fname)

    return dict(sorted(groups.items(), key=lambda kv: int(kv[0])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process first N products")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be indexed without loading model")
    args = parser.parse_args()

    print(f"Scanning: {BACKEND_UPLOADS}")
    all_groups = scan_products(BACKEND_UPLOADS)
    total_products = len(all_groups)
    total_images = sum(len(v) for v in all_groups.values())
    print(f"Discovered {total_products} products, {total_images} image files\n")

    if args.limit:
        all_groups = dict(list(all_groups.items())[: args.limit])
        print(f"--limit {args.limit}: restricting to {len(all_groups)} products\n")

    load_config()
    print("Getting FAISS manager for DINOv3 folder...")
    faiss_mgr = get_faiss_manager(DINOV3_MODEL)
    print(f"  Index path: {faiss_mgr.index_path}")
    print(f"  Starting sizes: {faiss_mgr.get_all_sizes()}\n")

    # First, determine which products still need indexing (quick metadata check)
    pending = []
    already = 0
    for pid, fnames in all_groups.items():
        if faiss_mgr.has_product(IndexType.VISUAL, pid):
            already += 1
            continue
        pending.append((pid, fnames))

    pending_images = sum(len(f) for _, f in pending)
    print(f"Pending:       {len(pending)} products, {pending_images} images")
    print(f"Already done:  {already} products")
    if not pending:
        print("\nNothing to do. DINOv3 visual index is up to date.")
        return

    if args.dry_run:
        print("\n--dry-run: stopping before model load")
        for pid, fnames in pending[:5]:
            print(f"  would index product={pid} ({len(fnames)} images)")
        if len(pending) > 5:
            print(f"  ... and {len(pending) - 5} more products")
        return

    # Only import/load DINOv3 after we know there's work to do
    from models.visual_models import VisualModelManager

    print(f"\nLoading DINOv3 ({DINOV3_MODEL}) on cuda...")
    t0 = time.time()
    visual_mgr = VisualModelManager(
        model_type="dinov3",
        model_config={"model_name": DINOV3_MODEL, "device": "cuda"},
    )
    print(f"DINOv3 loaded in {time.time() - t0:.1f}s\n")

    added_products = 0
    added_images = 0
    skipped_missing = 0
    start = time.time()
    last_save = start

    for idx, (pid, fnames) in enumerate(pending, start=1):
        t_prod = time.time()
        product_images_added = 0
        for image_no, fname in enumerate(fnames):
            path = os.path.join(BACKEND_UPLOADS, fname)
            if not os.path.exists(path):
                skipped_missing += 1
                continue

            embedding = visual_mgr.get_embedding(path)
            faiss_mgr.add_to_visual(
                embedding=embedding,
                product_id=pid,
                image_no=image_no,
                model_name=DINOV3_MODEL,
            )
            product_images_added += 1
            added_images += 1

        if product_images_added > 0:
            added_products += 1
            dt = time.time() - t_prod
            elapsed = time.time() - start
            rate = added_images / elapsed if elapsed > 0 else 0
            eta = (pending_images - added_images) / rate if rate > 0 else 0
            print(
                f"[{idx}/{len(pending)}] product={pid:>4}  +{product_images_added} imgs  "
                f"({dt:.1f}s)  total={added_images}/{pending_images}  "
                f"rate={rate:.2f} img/s  ETA={eta/60:.1f} min"
            )

        # Periodic save every 60 seconds — minimize data loss on interruption
        if time.time() - last_save > 60:
            faiss_mgr.save()
            last_save = time.time()
            print(f"   [checkpoint saved]")

    # Final save
    faiss_mgr.save()

    elapsed = time.time() - start
    sizes = faiss_mgr.get_all_sizes()

    print()
    print("=" * 60)
    print(f"Products indexed:        {added_products}")
    print(f"Images embedded:         {added_images}")
    print(f"Missing files skipped:   {skipped_missing}")
    print(f"Elapsed:                 {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Final DINOv3 sizes:      {sizes}")
    print("=" * 60)


if __name__ == "__main__":
    main()
