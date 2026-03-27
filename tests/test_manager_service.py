"""
Unit tests for services/manager_service.py.
"""

import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# combine_product_text
# ---------------------------------------------------------------------------

class TestCombineProductText:

    def test_combine_all_fields(self):
        from services.manager_service import combine_product_text

        result = combine_product_text(
            name="Sneakers",
            description="Comfortable running shoes",
            brand="Nike",
            category="Footwear",
            price=99.99,
        )
        assert "Sneakers" in result
        assert "Comfortable running shoes" in result
        assert "Nike" in result
        assert "Footwear" in result
        assert "Price: 99.99" in result

    def test_combine_partial_fields(self):
        from services.manager_service import combine_product_text

        result = combine_product_text(
            name="Sneakers",
            description="",
            brand="",
            category="Footwear",
            price="",
        )
        assert "Sneakers" in result
        assert "Footwear" in result
        # Empty fields should not appear
        assert "Price:" not in result


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:

    def test_load_config_success(self):
        """load_config should parse config.json and set module-level vars."""
        fake_config = {
            "models": {"test-model": {"type": "clip", "dimension": 256}},
            "defaults": {
                "dimension": 256,
                "top_k": 5,
                "max_top_k": 50,
                "data_base_path": "/tmp/idx",
                "host": "127.0.0.1",
                "port": 9999,
            },
        }

        import services.manager_service as svc

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(fake_config, f)
            tmp_path = f.name

        try:
            # Patch the config path resolution
            with patch.object(os.path, "join", return_value=tmp_path):
                cfg = svc.load_config()
            assert cfg["models"]["test-model"]["dimension"] == 256
            assert svc.DEFAULT_DIMENSION == 256
            assert svc.MAX_TOP_K == 50
            assert svc.PORT == 9999
        finally:
            os.unlink(tmp_path)

    def test_load_config_missing_file(self):
        """Should raise RuntimeError if config.json doesn't exist."""
        import services.manager_service as svc

        with patch.object(
            os.path, "join", return_value="/nonexistent/config.json"
        ):
            with pytest.raises(RuntimeError, match="Configuration file not found"):
                svc.load_config()
