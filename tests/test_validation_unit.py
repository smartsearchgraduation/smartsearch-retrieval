"""Unit tests for utils.validation (RU-VL-01..32)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import flask
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import validation  # noqa: E402
from utils.validation import (  # noqa: E402
    deduplicate_fused_results,
    deduplicate_text_results,
    deduplicate_visual_results,
    init_validation_config,
    validate_clip_model,
    validate_image_file_size,
    validate_required_fields,
    validate_text_length,
    validate_top_k,
)


# ---------- State-leakage guard ----------


@pytest.fixture(autouse=True)
def _restore_validation_globals():
    """Snapshot and restore module globals mutated by init_validation_config."""
    orig_max = validation.MAX_TOP_K
    orig_default = validation.DEFAULT_TOP_K
    orig_text = validation.MAX_TEXT_LENGTH
    orig_image = validation.MAX_IMAGE_SIZE_BYTES
    yield
    validation.MAX_TOP_K = orig_max
    validation.DEFAULT_TOP_K = orig_default
    validation.MAX_TEXT_LENGTH = orig_text
    validation.MAX_IMAGE_SIZE_BYTES = orig_image


# ---------- init_validation_config ----------


def test_ru_vl_01_init_validation_config_mutates_globals_and_clamps():
    """RU-VL-01: init_validation_config mutates module globals and changes clamping behavior."""
    init_validation_config(max_top_k=50, default_top_k=7)
    assert validation.MAX_TOP_K == 50
    assert validation.DEFAULT_TOP_K == 7
    assert validate_top_k({}) == 7
    assert validate_top_k({"top_k": 999}) == 50


# ---------- validate_top_k ----------


def test_ru_vl_02_missing_top_k_returns_default():
    """RU-VL-02: Missing top_k returns DEFAULT_TOP_K."""
    assert validate_top_k({}) == validation.DEFAULT_TOP_K
    assert validate_top_k({}) == 10


def test_ru_vl_03_string_coercible_top_k_returns_int():
    """RU-VL-03: String-coercible top_k returns the int."""
    assert validate_top_k({"top_k": "5"}) == 5


def test_ru_vl_04_non_numeric_string_raises():
    """RU-VL-04: Non-numeric string raises ValueError."""
    with pytest.raises(ValueError, match=r"top_k must be a valid integer"):
        validate_top_k({"top_k": "abc"})


def test_ru_vl_05_zero_raises_value_error():
    """RU-VL-05: Zero raises ValueError (boundary 1-1)."""
    with pytest.raises(ValueError, match=r"top_k must be >= 1"):
        validate_top_k({"top_k": 0})


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(1, 1, id="RU-VL-06"),
        pytest.param(100, 100, id="RU-VL-07"),
        pytest.param(999, 100, id="RU-VL-08"),
    ],
)
def test_ru_vl_06_07_08_top_k_boundaries(value, expected):
    """RU-VL-06/07/08: top_k boundary values (1, MAX, over MAX)."""
    assert validate_top_k({"top_k": value}) == expected


def test_ru_vl_09_top_k_none_raises():
    """RU-VL-09: top_k=None raises ValueError (TypeError reraised)."""
    with pytest.raises(ValueError, match=r"top_k must be a valid integer"):
        validate_top_k({"top_k": None})


# ---------- validate_required_fields ----------


def test_ru_vl_10_data_none_returns_400():
    """RU-VL-10: data is None returns 400 with JSON error."""
    with flask.Flask(__name__).test_request_context():
        result = validate_required_fields(None, ["query"])
        assert result is not None
        resp, status = result
        assert status == 400
        body = resp.get_json()
        assert body["status"] == "error"
        assert "Request body must be valid JSON" in body["message"]


def test_ru_vl_11_missing_field_returns_400_naming_field():
    """RU-VL-11: Missing one of two required fields returns 400 naming the field."""
    with flask.Flask(__name__).test_request_context():
        result = validate_required_fields({"query": "hi"}, ["query", "model_name"])
        assert result is not None
        resp, status = result
        assert status == 400
        body = resp.get_json()
        assert "Missing required field: model_name" in body["message"]


def test_ru_vl_12_all_required_present_returns_none():
    """RU-VL-12: All required fields present returns None."""
    with flask.Flask(__name__).test_request_context():
        result = validate_required_fields(
            {"query": "hi", "model_name": "ViT-B/32"}, ["query", "model_name"]
        )
        assert result is None


# ---------- deduplicate_text_results ----------


def test_ru_vl_13_dedup_text_empty():
    """RU-VL-13: Empty list yields empty dict."""
    assert deduplicate_text_results([]) == {}


def test_ru_vl_14_dedup_text_keeps_highest_score():
    """RU-VL-14: Multiple entries for one product keep the highest score; second product passes through."""
    out = deduplicate_text_results(
        [
            {"product_id": "A", "score": 0.5},
            {"product_id": "A", "score": 0.9},
            {"product_id": "A", "score": 0.7},
            {"product_id": "B", "score": 0.4},
        ]
    )
    assert out == {"A": 0.9, "B": 0.4}


# ---------- deduplicate_visual_results ----------


def test_ru_vl_15_dedup_visual_empty():
    """RU-VL-15: Empty list yields empty dict."""
    assert deduplicate_visual_results([]) == {}


def test_ru_vl_16_dedup_visual_best_score_and_image_no():
    """RU-VL-16: Best score wins and its image_no is kept; missing image_no defaults to 0."""
    out = deduplicate_visual_results(
        [
            {"product_id": "A", "score": 0.3, "image_no": 1},
            {"product_id": "A", "score": 0.8, "image_no": 2},
            {"product_id": "A", "score": 0.5, "image_no": 3},
            {"product_id": "B", "score": 0.4},
        ]
    )
    assert out == {
        "A": {"score": 0.8, "image_no": 2},
        "B": {"score": 0.4, "image_no": 0},
    }


# ---------- deduplicate_fused_results ----------


def test_ru_vl_17_dedup_fused_best_score_wins():
    """RU-VL-17: Best fused score wins; default image_no=0 when absent; lower-score later does not overwrite."""
    out = deduplicate_fused_results(
        [
            {"product_id": "X", "score": 0.9, "image_no": 5},
            {"product_id": "X", "score": 0.2, "image_no": 9},
            {"product_id": "Y", "score": 0.6},
        ]
    )
    assert out == {
        "X": {"score": 0.9, "image_no": 5},
        "Y": {"score": 0.6, "image_no": 0},
    }


# ---------- validate_clip_model ----------


def test_ru_vl_18_clip_model_passes(monkeypatch):
    """RU-VL-18: Multimodal CLIP model passes silently."""
    monkeypatch.setattr(
        "services.manager_service.MODEL_REGISTRY",
        {"ViT-B/32": {"type": "clip"}},
    )
    assert validate_clip_model("ViT-B/32") is None


def test_ru_vl_19_marqo_model_passes(monkeypatch):
    """RU-VL-19: Marqo model (multimodal) passes silently."""
    monkeypatch.setattr(
        "services.manager_service.MODEL_REGISTRY",
        {"Marqo/marqo-ecommerce-embeddings-L": {"type": "marqo"}},
    )
    assert validate_clip_model("Marqo/marqo-ecommerce-embeddings-L") is None


def test_ru_vl_20_bge_model_raises(monkeypatch):
    """RU-VL-20: Non-multimodal type (bge) raises ValueError mentioning the type."""
    monkeypatch.setattr(
        "services.manager_service.MODEL_REGISTRY",
        {"BAAI/bge-large-en-v1.5": {"type": "bge"}},
    )
    with pytest.raises(ValueError, match=r"'bge'") as excinfo:
        validate_clip_model("BAAI/bge-large-en-v1.5")
    assert "multimodal" in str(excinfo.value)


def test_ru_vl_21_unknown_model_raises(monkeypatch):
    """RU-VL-21: Unknown model raises ValueError mentioning 'unknown'."""
    monkeypatch.setattr("services.manager_service.MODEL_REGISTRY", {})
    with pytest.raises(ValueError, match=r"'unknown'"):
        validate_clip_model("does-not-exist")


# ---------- validate_text_length ----------


def test_ru_vl_22_non_string_raises():
    """RU-VL-22: Non-string input raises ValueError."""
    with pytest.raises(ValueError, match=r"text must be a string"):
        validate_text_length(123)


def test_ru_vl_23_length_at_default_limit_ok():
    """RU-VL-23: Length exactly at default limit (10_000) does not raise."""
    assert validate_text_length("a" * 10_000) is None


def test_ru_vl_24_length_one_over_default_raises():
    """RU-VL-24: Length one over default limit raises ValueError mentioning 10000."""
    with pytest.raises(ValueError, match=r"10000"):
        validate_text_length("a" * 10_001)


def test_ru_vl_25_custom_max_length_respected():
    """RU-VL-25: Custom max_length parameter is respected."""
    with pytest.raises(ValueError, match=r"3"):
        validate_text_length("hello", max_length=3)
    assert validate_text_length("abc", max_length=3) is None


# ---------- validate_image_file_size ----------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("", id="RU-VL-26"),
        pytest.param(12345, id="RU-VL-27"),
        pytest.param("   ", id="RU-VL-28"),
    ],
)
def test_ru_vl_26_27_28_invalid_path_raises(value):
    """RU-VL-26/27/28: Empty/non-string/whitespace path raises ValueError."""
    with pytest.raises(ValueError, match=r"non-empty file path"):
        validate_image_file_size(value)


def test_ru_vl_29_nonexistent_path_returns_silently():
    """RU-VL-29: Non-existent path returns silently (validated later)."""
    assert validate_image_file_size("Z:/definitely/does/not/exist.jpg") is None


def test_ru_vl_30_existing_small_file_returns_none(tmp_image_path):
    """RU-VL-30: Existing file under default limit returns None."""
    assert validate_image_file_size(tmp_image_path) is None


def test_ru_vl_31_existing_file_over_limit_raises(tmp_image_path):
    """RU-VL-31: Existing file with size over max_size_bytes raises ValueError."""
    with pytest.raises(ValueError, match=r"Image size exceeds"):
        validate_image_file_size(tmp_image_path, max_size_bytes=1)


def test_ru_vl_32_oversize_via_monkeypatched_getsize(tmp_image_path, monkeypatch):
    """RU-VL-32: Oversize via monkeypatched os.path.getsize raises ValueError mentioning 50MB."""
    monkeypatch.setattr(
        "utils.validation.os.path.getsize", lambda p: 100 * 1024 * 1024
    )
    with pytest.raises(ValueError, match=r"50MB"):
        validate_image_file_size(tmp_image_path)
