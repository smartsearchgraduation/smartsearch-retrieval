"""Integration-test conftest for the Retrieval service.

Re-exports the existing fixtures from the parent `Retrieval/tests/conftest.py`
(`flask_client`, `stub_managers`, `tmp_index_dir`, `tmp_image_path`, etc.) so
that integration tests under `tests/integration/` can use them without
duplicating fixture setup. We also make the parent `tests` directory and the
Retrieval repo root importable.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT_TESTS = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

for p in (_REPO_ROOT, _PARENT_TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

# Re-export fixtures from the parent Retrieval/tests/conftest.py.
# pytest auto-discovers fixtures from any conftest.py up the tree, so simply
# importing the names here keeps them visible to integration tests via the
# usual fixture-resolution mechanism. We do NOT redefine them.
from conftest import (  # noqa: F401,E402  (re-export only)
    tmp_index_dir,
    tmp_image_path,
    flask_client,
    stub_managers,
    vec_factory,
    clip_vec,
    bge_vec,
    qwen_vec,
    marqo_vec,
    dinov3_vec,
    stub_text_embedder,
    stub_image_embedder,
    stub_fused_embedder,
    faiss_manager_factory,
    oversize_image_path,
)
