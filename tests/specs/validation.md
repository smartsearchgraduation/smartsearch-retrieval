# Validation Utilities — Test Specification

Source under test: `utils/validation.py`. ID prefix: `RU-VL-NN`. Available fixture: `tmp_image_path` (writes a small file to a temp dir and yields its path). Mocking notes: `validate_clip_model` performs a deferred `from services.manager_service import MODEL_REGISTRY` at call time — patch via `monkeypatch.setattr("services.manager_service.MODEL_REGISTRY", {...})`. `validate_required_fields` calls `flask.jsonify`, so its tests must run inside `with flask.Flask(__name__).test_request_context():`. No FAISS interaction. No real model weights are loaded. Module globals `MAX_TOP_K` and `DEFAULT_TOP_K` are mutated by `init_validation_config`; tests that change them must reset them in teardown (or use `monkeypatch.setattr` on the module attribute) so case ordering is irrelevant.

## init_validation_config

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-VL-01 | `init_validation_config` mutates module globals and changes `validate_top_k` clamping behavior | 1) Call `init_validation_config(max_top_k=50, default_top_k=7)`. 2) Assert `utils.validation.MAX_TOP_K == 50` and `utils.validation.DEFAULT_TOP_K == 7`. 3) Call `validate_top_k({})` and assert it returns `7`. 4) Call `validate_top_k({"top_k": 999})` and assert it returns `50`. | `max_top_k=50`, `default_top_k=7` | Save original `MAX_TOP_K`/`DEFAULT_TOP_K` and restore via `monkeypatch.setattr` so other tests see defaults. | Globals updated; default and clamp use the new values. |

## validate_top_k

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-VL-02 | Missing `top_k` returns `DEFAULT_TOP_K` | Call `validate_top_k({})` and assert it equals `utils.validation.DEFAULT_TOP_K` (10 by default). | `data = {}` | Module globals at defaults (`MAX_TOP_K=100`, `DEFAULT_TOP_K=10`). | Returns `10`. |
| RU-VL-03 | String-coercible `top_k` returns the int | Call `validate_top_k({"top_k": "5"})`. | `{"top_k": "5"}` | Defaults. | Returns `5`. |
| RU-VL-04 | Non-numeric string raises ValueError | Call `validate_top_k({"top_k": "abc"})` inside `pytest.raises(ValueError)`; assert message contains `"top_k must be a valid integer"`. | `{"top_k": "abc"}` | Defaults. | Raises `ValueError` with the integer-conversion message. |
| RU-VL-05 | Zero raises ValueError (boundary: 1−1) | Call `validate_top_k({"top_k": 0})` inside `pytest.raises(ValueError)`; assert message contains `"top_k must be >= 1"`. | `{"top_k": 0}` | Defaults. | Raises `ValueError` mentioning `>= 1`. |
| RU-VL-06 | `top_k=1` (boundary) returns 1 | Call `validate_top_k({"top_k": 1})`. | `{"top_k": 1}` | Defaults. | Returns `1`. |
| RU-VL-07 | `top_k=100` (boundary equal to MAX) returns 100 | Call `validate_top_k({"top_k": 100})`. | `{"top_k": 100}` | Defaults (`MAX_TOP_K=100`). | Returns `100`. |
| RU-VL-08 | `top_k=999` clamps to `MAX_TOP_K` | Call `validate_top_k({"top_k": 999})`. | `{"top_k": 999}` | Defaults. | Returns `100`. |
| RU-VL-09 | `top_k=None` raises ValueError | Call `validate_top_k({"top_k": None})` inside `pytest.raises(ValueError)`. | `{"top_k": None}` | Defaults. | Raises `ValueError` (TypeError path → reraised as ValueError with integer-conversion message). |

## validate_required_fields

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-VL-10 | `data is None` returns 400 with JSON error | Inside `flask.Flask(__name__).test_request_context()`, call `validate_required_fields(None, ["query"])`. Unpack the tuple `(resp, status)`. Assert `status == 400`. Assert `resp.get_json()["status"] == "error"` and `"Request body must be valid JSON"` in `resp.get_json()["message"]`. | `data=None`, `required_fields=["query"]` | Wrap call in Flask test request context. | Returns `(jsonify-response, 400)` with the JSON-error message. |
| RU-VL-11 | Missing one of two required fields returns 400 naming the field | Inside Flask test request context, call `validate_required_fields({"query": "hi"}, ["query", "model_name"])`. Assert `status == 400` and message contains `"Missing required field: model_name"`. | `data={"query":"hi"}`, `required=["query","model_name"]` | Flask test request context. | Returns `(response, 400)` mentioning the missing field `model_name`. |
| RU-VL-12 | All required fields present returns None | Inside Flask test request context, call `validate_required_fields({"query": "hi", "model_name": "ViT-B/32"}, ["query", "model_name"])`. | `data` has both fields | Flask test request context. | Returns `None`. |

## deduplicate_text_results

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-VL-13 | Empty list yields empty dict | Call `deduplicate_text_results([])`. | `[]` | None. | Returns `{}`. |
| RU-VL-14 | Multiple entries for one product keep the highest score; second product passes through | Call `deduplicate_text_results([{"product_id":"A","score":0.5},{"product_id":"A","score":0.9},{"product_id":"A","score":0.7},{"product_id":"B","score":0.4}])`. | See Steps. | None. | Returns `{"A": 0.9, "B": 0.4}`. |

## deduplicate_visual_results

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-VL-15 | Empty list yields empty dict | Call `deduplicate_visual_results([])`. | `[]` | None. | Returns `{}`. |
| RU-VL-16 | Best score wins and its `image_no` is kept; missing `image_no` defaults to 0 | Call `deduplicate_visual_results([{"product_id":"A","score":0.3,"image_no":1},{"product_id":"A","score":0.8,"image_no":2},{"product_id":"A","score":0.5,"image_no":3},{"product_id":"B","score":0.4}])`. | See Steps. | None. | Returns `{"A": {"score":0.8,"image_no":2}, "B": {"score":0.4,"image_no":0}}`. |

## deduplicate_fused_results

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-VL-17 | Best fused score wins; default image_no=0 when absent; lower-score later entry does not overwrite | Call `deduplicate_fused_results([{"product_id":"X","score":0.9,"image_no":5},{"product_id":"X","score":0.2,"image_no":9},{"product_id":"Y","score":0.6}])`. | See Steps. | None. | Returns `{"X": {"score":0.9,"image_no":5}, "Y": {"score":0.6,"image_no":0}}`. |

## validate_clip_model

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-VL-18 | Multimodal CLIP model passes silently | `monkeypatch.setattr("services.manager_service.MODEL_REGISTRY", {"ViT-B/32": {"type": "clip"}})`. Call `validate_clip_model("ViT-B/32")`. | model_name="ViT-B/32" | MODEL_REGISTRY patched as shown. | Returns `None` (no exception). |
| RU-VL-19 | Marqo model (multimodal) passes silently | Patch MODEL_REGISTRY to `{"Marqo/marqo-ecommerce-embeddings-L": {"type": "marqo"}}`. Call `validate_clip_model("Marqo/marqo-ecommerce-embeddings-L")`. | as above | MODEL_REGISTRY patched. | Returns `None`. |
| RU-VL-20 | Non-multimodal type (bge) raises ValueError mentioning the type | Patch MODEL_REGISTRY to `{"BAAI/bge-large-en-v1.5": {"type": "bge"}}`. Call `validate_clip_model("BAAI/bge-large-en-v1.5")` inside `pytest.raises(ValueError)`; assert message contains `"'bge'"` and `"multimodal"`. | model_name="BAAI/bge-large-en-v1.5" | MODEL_REGISTRY patched. | Raises `ValueError` whose message includes `'bge'`. |
| RU-VL-21 | Unknown model raises ValueError mentioning `unknown` | Patch MODEL_REGISTRY to `{}`. Call `validate_clip_model("does-not-exist")` inside `pytest.raises(ValueError)`; assert message contains `"'unknown'"`. | model_name="does-not-exist" | MODEL_REGISTRY patched to empty dict. | Raises `ValueError` whose message includes the literal `'unknown'`. |

## validate_text_length

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-VL-22 | Non-string input raises ValueError | Call `validate_text_length(123)` inside `pytest.raises(ValueError)`; assert message contains `"text must be a string"`. | `text=123` (int) | None. | Raises `ValueError`. |
| RU-VL-23 | Length exactly at default limit (10_000) does not raise (boundary) | Call `validate_text_length("a" * 10_000)`. | text length = 10_000 | None. | Returns `None` (no exception). |
| RU-VL-24 | Length one over default limit raises ValueError (boundary +1) | Call `validate_text_length("a" * 10_001)` inside `pytest.raises(ValueError)`; assert message contains `"10000"`. | text length = 10_001 | None. | Raises `ValueError` mentioning the limit. |
| RU-VL-25 | Custom `max_length` parameter is respected | Call `validate_text_length("hello", max_length=3)` inside `pytest.raises(ValueError)`; assert message contains `"3"`. Then call `validate_text_length("abc", max_length=3)` and assert it returns None. | text="hello", max_length=3; then text="abc", max_length=3 | None. | First call raises ValueError; second call returns None. |

## validate_image_file_size

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-VL-26 | Empty string path raises ValueError | Call `validate_image_file_size("")` inside `pytest.raises(ValueError)`; assert message contains `"non-empty file path"`. | `image_path=""` | None. | Raises `ValueError`. |
| RU-VL-27 | Non-string path raises ValueError | Call `validate_image_file_size(12345)` inside `pytest.raises(ValueError)`; assert message contains `"non-empty file path"`. | `image_path=12345` | None. | Raises `ValueError`. |
| RU-VL-28 | Whitespace-only path raises ValueError | Call `validate_image_file_size("   ")` inside `pytest.raises(ValueError)`; assert message contains `"non-empty file path"`. | `image_path="   "` | None. | Raises `ValueError`. |
| RU-VL-29 | Non-existent path returns silently (validated later) | Call `validate_image_file_size("Z:/definitely/does/not/exist.jpg")`. | non-existent path | None. | Returns `None` (no exception). |
| RU-VL-30 | Existing file under default limit returns None | Call `validate_image_file_size(tmp_image_path)`. | `tmp_image_path` fixture (small file) | `tmp_image_path` fixture writes a small file. | Returns `None`. |
| RU-VL-31 | Existing file with size over `max_size_bytes` raises ValueError | Call `validate_image_file_size(tmp_image_path, max_size_bytes=1)` inside `pytest.raises(ValueError)`; assert message contains `"Image size exceeds"`. | `tmp_image_path`, `max_size_bytes=1` | `tmp_image_path` fixture exists and is > 1 byte. | Raises `ValueError`. |
| RU-VL-32 | Oversize via monkeypatched `os.path.getsize` raises ValueError mentioning 50MB | `monkeypatch.setattr("utils.validation.os.path.getsize", lambda p: 100 * 1024 * 1024)` and ensure `os.path.exists` returns True for the path (use `tmp_image_path`). Call `validate_image_file_size(tmp_image_path)` inside `pytest.raises(ValueError)`; assert message contains `"50MB"`. | `tmp_image_path`, getsize patched to 100MB | Patch `os.path.getsize` on the validation module. | Raises `ValueError` whose message includes `"50MB"`. |

Wrote 32 test cases covering 10 feature bullets.

File written to: `C:\Users\user\Desktop\smartsearch-retrieval\tests\specs\validation.md`
