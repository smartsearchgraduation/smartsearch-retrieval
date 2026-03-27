"""
Shared pytest fixtures for the smartsearch-retrieval test suite.
"""

import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Fixtures: Flask app & test client
# ---------------------------------------------------------------------------

@pytest.fixture()
def app():
    """
    Create a Flask test app with mocked config loading.
    Patches load_config so it doesn't need a real config.json,
    then imports the app factory.
    """
    # Prepare a minimal in-memory config
    fake_config = {
        "models": {
            "ViT-B/32": {"type": "clip", "dimension": 512},
        },
        "defaults": {
            "dimension": 512,
            "top_k": 10,
            "max_top_k": 100,
            "data_base_path": "./data",
            "host": "0.0.0.0",
            "port": 5002,
        },
    }

    with patch("services.manager_service.load_config") as mock_load:
        # Simulate what load_config does: set module-level variables
        import services.manager_service as svc

        svc._config = fake_config
        svc.MODEL_REGISTRY = fake_config["models"]
        svc.DEFAULT_DIMENSION = 512
        svc.DATA_BASE_PATH = "./data"
        svc.HOST = "0.0.0.0"
        svc.PORT = 5002
        svc.MAX_TOP_K = 100
        svc.DEFAULT_TOP_K = 10

        mock_load.return_value = fake_config

        from utils.validation import init_validation_config
        init_validation_config(100, 10)

        from flask import Flask
        from routes.product_routes import product_bp
        from routes.search_routes import search_bp
        from routes.system_routes import system_bp

        flask_app = Flask(__name__)
        flask_app.register_blueprint(product_bp)
        flask_app.register_blueprint(search_bp)
        flask_app.register_blueprint(system_bp)
        flask_app.config["TESTING"] = True

        yield flask_app

    # Reset global managers after each test
    svc._faiss_managers.clear()
    svc._textual_managers.clear()
    svc._visual_managers.clear()
    svc._fused_managers.clear()


@pytest.fixture()
def client(app):
    """Flask test client."""
    return app.test_client()


# ---------------------------------------------------------------------------
# Fixtures: Mock managers (for route tests)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_faiss_manager():
    """A MagicMock standing in for FAISSManager."""
    fm = MagicMock()
    fm.search_textual.return_value = [
        {"product_id": "prod_001", "score": 0.95, "model_name": "ViT-B/32"},
    ]
    fm.search_visual.return_value = [
        {"product_id": "prod_001", "score": 0.90, "image_no": 0, "model_name": "ViT-B/32"},
    ]
    fm.add_to_textual.return_value = 0
    fm.add_to_visual.return_value = 1
    fm.remove_product_from_all.return_value = {"textual": 1, "visual": 2, "fused": 0}
    fm.get_all_sizes.return_value = {"textual": 10, "visual": 25, "fused": 0}
    fm.save.return_value = None
    return fm


@pytest.fixture()
def mock_textual_manager():
    """A MagicMock standing in for TextModelManager."""
    tm = MagicMock()
    tm.get_embedding.return_value = [0.1] * 512
    tm.get_document_embedding.return_value = [0.1] * 512
    return tm


@pytest.fixture()
def mock_visual_manager():
    """A MagicMock standing in for VisualModelManager."""
    vm = MagicMock()
    vm.get_embedding.return_value = [0.2] * 512
    return vm
