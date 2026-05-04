# Test Specification — manager_service

**Source under test:** `services/manager_service.py`
**ID prefix:** `RU-MS-NN` (sequential, zero-padded, starting at `RU-MS-01`).

## Mocking notes (apply to every case)

1. **Never mock `faiss.*` or `FAISSManager`.** Use the real in-memory FAISSManager produced by `services.manager_service.get_faiss_manager` / `get_or_load_all_faiss_managers`.
2. **Stub embedders at the model-pool boundary** before importing/using `TextModelManager`, `VisualModelManager`, `FusedModelManager`. Patch `CLIPModelPool.get`, `OpenCLIPModelPool.get` (Marqo), `DINOv3ModelPool.get`, plus the BGE / Qwen entry points used by `TextModelManager`. Stubs must return deterministic L2-normalized numpy float32 vectors of correct dimension (CLIP=512, BGE=1024, Marqo=1024, Qwen=4096, DINOv3=4096), seeded from `hash(input)`.
3. **Override `services.manager_service.DATA_BASE_PATH`** via `monkeypatch.setattr` to a `tempfile.TemporaryDirectory`. No tests may write under the repo.
4. **Reset module-level caches** between tests: at the start of each test (or via an autouse fixture defined in this test file, NOT in shared `conftest.py`), clear `_textual_managers`, `_visual_managers`, `_fused_managers`, `_faiss_managers` dicts on the `services.manager_service` module. Also call `load_config()` after pointing at controlled state when needed, or directly mutate `MODEL_REGISTRY` / `DEFAULT_MODELS` via monkeypatch.
5. **No real model weights are loaded.** If a sub-call would trigger weight download, the embedder boundary stubs must intercept it.

## Fixtures available

- `tmp_index_dir` — TemporaryDirectory path; tests monkeypatch `DATA_BASE_PATH` to this.
- `clip_vec`, `bge_vec`, `marqo_vec`, `qwen_vec`, `dinov3_vec` — deterministic L2-normalized vector factories.
- `tmp_image_path` — temp PNG file path.

Module-local helpers (defined in `tests/test_manager_service_unit.py`, not in shared conftest):
- `_reset_caches()` — clears the four `_*_managers` dicts.
- `_install_embedder_stubs(monkeypatch)` — patches all model-pool `.get` methods.
- `_seed_faiss(manager, n, dim)` — populates a real FAISSManager via `add_textual` / `add_visual` calls used by routes (call directly on the manager object).

## Test cases

### Configuration loading (`load_config`)

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-MS-01 | `load_config` populates MODEL_REGISTRY with every entry from config.json keyed by model name with `type` and `dimension`. | Import `services.manager_service as ms`. Call `ms.load_config()`. Assert `set(ms.MODEL_REGISTRY.keys())` equals the five model names from config.json. Assert `ms.MODEL_REGISTRY["ViT-B/32"]["dimension"] == 512` and `["type"] == "clip"`. Assert `ms.MODEL_REGISTRY["BAAI/bge-large-en-v1.5"]["dimension"] == 1024`. | Real `config.json` at repo root. | Caches cleared. | All five models present with correct dimension/type pairs. |
| RU-MS-02 | `load_config` exposes correct HOST, PORT, MAX_TOP_K, DEFAULT_TOP_K, DEFAULT_DIMENSION, DATA_BASE_PATH from `defaults`. | Call `ms.load_config()`. Assert `ms.HOST == "0.0.0.0"`, `ms.PORT == 5002`, `ms.MAX_TOP_K == 100`, `ms.DEFAULT_TOP_K == 10`, `ms.DEFAULT_DIMENSION == 512`, `ms.DATA_BASE_PATH == "./data"`. | Real `config.json`. | Caches cleared. | All six values match config.json `defaults` block. |
| RU-MS-03 | `load_config` populates DEFAULT_MODELS with textual + visual keys. | Call `ms.load_config()`. Assert `ms.DEFAULT_MODELS["textual"] == "BAAI/bge-large-en-v1.5"` and `ms.DEFAULT_MODELS["visual"] == "ViT-B/32"`. | Real `config.json`. | Caches cleared. | Both keys equal config.json values. |
| RU-MS-04 | `load_config` raises RuntimeError with "Configuration file not found" when config.json is missing. | Monkeypatch `os.path.dirname` chain OR temporarily rename config.json by monkeypatching `open` to raise FileNotFoundError. Call `ms.load_config()`. Assert `RuntimeError` raised, message contains "Configuration file not found". | n/a | `open` patched to raise `FileNotFoundError`. | RuntimeError with the expected substring. |
| RU-MS-05 | `load_config` raises RuntimeError with "Invalid JSON" when config.json contains malformed JSON. | Write `"{not json"` to a temp file. Monkeypatch the config path resolution so `load_config` reads that file (or patch `json.load` to raise `JSONDecodeError`). Call `load_config()`. Assert RuntimeError, message contains "Invalid JSON". | Malformed JSON content. | n/a | RuntimeError with "Invalid JSON" substring. |

### `get_faiss_manager` / caching / dimension routing

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-MS-06 | `get_faiss_manager(model_name)` creates a FAISSManager with the model's dimension and a folder named `<sanitized>_<dim>_embeddings` under DATA_BASE_PATH. | Call `ms.get_faiss_manager("ViT-B/32")`. Assert returned object is a real `FAISSManager` instance, `.dimension == 512`, `.index_path` ends with a folder name produced by `make_folder_name("ViT-B/32", 512)` and lives under `tmp_index_dir`. | model_name="ViT-B/32" | DATA_BASE_PATH monkeypatched to tmp_index_dir; load_config called; caches cleared. | FAISSManager with dim 512 in correct folder. |
| RU-MS-07 | Second call to `get_faiss_manager` with the same model returns the SAME cached instance. | `m1 = ms.get_faiss_manager("ViT-B/32")`; `m2 = ms.get_faiss_manager("ViT-B/32")`. Assert `m1 is m2`. | n/a | As RU-MS-06. | Identity equality. |
| RU-MS-08 | `get_faiss_manager` for two different models returns two distinct instances with different `index_path` folders and different dimensions. | `m_clip = ms.get_faiss_manager("ViT-B/32")`; `m_bge = ms.get_faiss_manager("BAAI/bge-large-en-v1.5")`. Assert `m_clip is not m_bge`, `m_clip.dimension == 512`, `m_bge.dimension == 1024`, and `m_clip.index_path != m_bge.index_path`. | Two registered models. | As RU-MS-06. | Two distinct managers, different dims, different paths. |
| RU-MS-09 | `get_faiss_manager(None)` with no managers loaded falls back to `DEFAULT_MODELS["textual"]`. | Caches cleared. Call `ms.get_faiss_manager(None)`. Assert `.dimension == 1024` (BGE) and folder name corresponds to BGE. | model_name=None | load_config done; default textual = BGE. | FAISSManager built for BGE. |
| RU-MS-10 | `get_faiss_manager(None)` with at least one cached manager returns the first cached value. | Pre-cache by calling `ms.get_faiss_manager("ViT-B/32")` first. Then call `ms.get_faiss_manager(None)`. Assert it returns the same instance as the CLIP one. | n/a | As RU-MS-06. | Identity equality with the previously cached CLIP manager. |
| RU-MS-11 | Unknown model name uses `DEFAULT_DIMENSION` (512) via `_get_model_dimension` fallback. | Call `ms.get_faiss_manager("not/a/registered-model")`. Assert returned manager has `.dimension == 512`. | model_name="not/a/registered-model" | load_config done. | dimension == 512 (DEFAULT_DIMENSION fallback). Notes: the source does NOT raise on unknown model — document this behaviour. |

### `get_all_faiss_managers`, `discover_model_folders`, `get_or_load_all_faiss_managers`

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-MS-12 | `get_all_faiss_managers` returns the live `_faiss_managers` dict keyed by folder name. | Call `ms.get_faiss_manager("ViT-B/32")`. Then `d = ms.get_all_faiss_managers()`. Assert it is a dict with exactly one key equal to `make_folder_name("ViT-B/32",512)` and value is the same FAISSManager instance. | n/a | tmp_index_dir; caches cleared. | Dict size 1 with correct mapping. |
| RU-MS-13 | `discover_model_folders` returns [] when DATA_BASE_PATH does not exist. | Set DATA_BASE_PATH to a non-existent subfolder of tmp_index_dir. Call `ms.discover_model_folders()`. Assert returns `[]`. | DATA_BASE_PATH=missing path | monkeypatch DATA_BASE_PATH. | Empty list. |
| RU-MS-14 | `discover_model_folders` returns only directories ending with `_embeddings`. | Inside tmp_index_dir create folders `ViT-B-32_512_embeddings/`, `bge_1024_embeddings/`, and `random_other/`, plus a file `not_a_dir_embeddings`. Call `ms.discover_model_folders()`. Assert returned set equals `{"ViT-B-32_512_embeddings","bge_1024_embeddings"}`. | Mixed folder fixture | DATA_BASE_PATH=tmp_index_dir. | Exactly the two `_embeddings` directories. |
| RU-MS-15 | `get_or_load_all_faiss_managers` instantiates a FAISSManager for each on-disk `_embeddings` folder, parsing dimension from the folder name. | Create folder `bge_1024_embeddings/` under tmp_index_dir. Caches cleared. Call `ms.get_or_load_all_faiss_managers()`. Assert returned dict has key `bge_1024_embeddings` whose manager has `.dimension == 1024`. | folder name with dim=1024 | tmp_index_dir set up. | Manager loaded with correct dimension. |
| RU-MS-16 | `get_or_load_all_faiss_managers` falls back to DEFAULT_DIMENSION when folder name dimension cannot be parsed. | Create folder `weird_embeddings/` (no parseable dim). Call `ms.get_or_load_all_faiss_managers()`. Assert manager for `weird_embeddings` has `.dimension == ms.DEFAULT_DIMENSION` (512). | folder="weird_embeddings" | tmp_index_dir set up. | Manager dim==512. |

### `remove_product_from_all_models`

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-MS-17 | `remove_product_from_all_models` calls `remove_product_from_all` on every loaded manager and returns only folders where ≥1 vector was removed. | Pre-load two managers (CLIP and BGE). Add a textual vector for product_id="P1" to the CLIP manager directly via `add_textual(...)` using a stubbed embedder vector. Call `ms.remove_product_from_all_models("P1")`. Assert returned dict has key only for CLIP folder, with value `{...: 1, ...}` summing to ≥1. Assert BGE folder NOT in result. Verify CLIP manager `get_all_sizes()` shows 0 textual entries. | product_id="P1" present in CLIP only | both managers loaded; embedder stubs installed; tmp_index_dir. | Result dict contains only CLIP folder; CLIP index now empty for P1; .save() persisted to disk (folder contains files). |
| RU-MS-18 | `remove_product_from_all_models` returns empty dict when product is unknown across all managers. | Pre-load CLIP and BGE managers (no products added). Call `ms.remove_product_from_all_models("nope")`. Assert returns `{}`. | product_id="nope" | empty managers. | `{}`. |

### Textual / visual / fused manager getters

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-MS-19 | `get_textual_manager` caches by model_name (second call returns same instance). | `t1 = ms.get_textual_manager("BAAI/bge-large-en-v1.5")`; `t2 = ms.get_textual_manager("BAAI/bge-large-en-v1.5")`. Assert `t1 is t2`. Assert `isinstance(t1, TextModelManager)`. | model="BAAI/bge-large-en-v1.5" | embedder stubs installed; load_config done. | Identity equality. |
| RU-MS-20 | `get_textual_manager` resolves model_type from MODEL_REGISTRY when present. | Call `ms.get_textual_manager("Qwen/Qwen3-Embedding-8B")`. Assert the underlying manager was constructed with `model_type="qwen"` (inspect `_textual_managers["Qwen/..."]` attribute or constructor recorded value via stub). | model="Qwen/Qwen3-Embedding-8B" | as RU-MS-19. | model_type=="qwen". |
| RU-MS-21 | `get_textual_manager` falls back to name-based detection for unknown registry entries (bge / qwen / marqo / clip). | Clear `MODEL_REGISTRY` (monkeypatch to `{}`). Call `ms.get_textual_manager("BAAI/bge-something-new")` and assert constructed model_type=="bge". Repeat for `"Qwen/foo"` → "qwen", `"Marqo/foo"` → "marqo", `"openai/clip-vit"` → "clip". | four name patterns | MODEL_REGISTRY emptied. | Detected types match the four expected fallbacks. |
| RU-MS-22 | `get_visual_manager` caches and resolves type via `_get_visual_model_type` (registry first, then name patterns for marqo/dinov3, default clip). | Call `ms.get_visual_manager("facebook/dinov3-vit7b16-pretrain-lvd1689m")` and assert type=="dinov3" and instance cached. Repeat with MODEL_REGISTRY emptied for `"facebook/dinov3-something"` → "dinov3", `"Marqo/x"` → "marqo", `"some-clip"` → "clip". | four model strings | embedder stubs installed. | Cached `VisualModelManager` with the expected `model_type` for each. |
| RU-MS-23 | `get_fused_manager` caches and resolves type using the same `_get_visual_model_type` rules. | `f1 = ms.get_fused_manager("Marqo/marqo-ecommerce-embeddings-L")`; `f2 = ms.get_fused_manager("Marqo/marqo-ecommerce-embeddings-L")`. Assert `f1 is f2`, `isinstance(f1, FusedModelManager)`, model_type=="marqo". | Marqo model | embedder stubs; load_config done. | Identity equality and correct model_type. |

### `combine_product_text`

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-MS-24 | `combine_product_text` joins all non-empty fields with spaces and prefixes price with "Price: ". | Call `ms.combine_product_text("Shoe","Comfy","Nike","Footwear",99.9)`. Assert result == "Shoe Comfy Nike Footwear Price: 99.9". | full inputs | n/a | Exact string match. |
| RU-MS-25 | `combine_product_text` skips empty/None/zero fields (price=0 falsy → skipped). | Call `ms.combine_product_text("Hat","","BrandX",None,0)`. Assert result == "Hat BrandX". | Mixed empties | n/a | "Hat BrandX" (no price token, no description, no category). |
| RU-MS-26 | `combine_product_text` returns empty string when ALL fields are falsy. | Call `ms.combine_product_text("","",None,None,0)`. Assert result == "". | All empty | n/a | Empty string. |

### `get_all_index_stats` and `get_available_models`

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-MS-27 | `get_all_index_stats` returns `{folder_name: get_all_sizes()}` for every loaded/discovered manager. | Pre-load CLIP manager; add 2 textual vectors for P1, P2 directly. Call `ms.get_all_index_stats()`. Assert dict has CLIP folder key whose value is a dict containing the textual count == 2. | n/a | embedder stubs; tmp_index_dir. | Mapping reflects the real FAISSManager.get_all_sizes() output. |
| RU-MS-28 | `get_available_models` categorizes models into textual_models / visual_models per TEXTUAL_TYPES / VISUAL_TYPES sets and returns defaults block. | Call `ms.get_available_models()`. Assert: textual_models contains entries for ViT-B/32 (clip), BGE, Qwen, Marqo (NOT dinov3). visual_models contains ViT-B/32, Marqo, dinov3 (NOT bge, NOT qwen). `defaults == {"textual":"BAAI/bge-large-en-v1.5","visual":"ViT-B/32"}`. | full registry | load_config done. | Categorization exactly matches the type sets; defaults correct. |
| RU-MS-29 | `get_available_models` returns empty lists when MODEL_REGISTRY is empty. | Monkeypatch `ms.MODEL_REGISTRY = {}` and `ms.DEFAULT_MODELS = {}`. Call `ms.get_available_models()`. Assert `textual_models==[]`, `visual_models==[]`, `defaults=={"textual":"","visual":""}`. | empty registry | n/a | Empty lists with empty default strings. |

### Integration smoke (real FAISSManager + stubbed embedders)

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-MS-30 | End-to-end add+search through a real FAISSManager obtained from manager_service. | `mgr = ms.get_faiss_manager("BAAI/bge-large-en-v1.5")`. Call `mgr.add_textual("P1", bge_vec("hello"))`. Call `mgr.search_textual(bge_vec("hello"), top_k=1)`. Assert top hit product_id == "P1". | bge_vec deterministic | embedder stubs installed; tmp_index_dir; load_config. | Search returns P1 as top result. Notes: exercises the manager_service → FAISSManager wiring without route logic. |
| RU-MS-31 | Persistence smoke: after `add_textual` + `mgr.save()`, the manager folder contains files on disk under DATA_BASE_PATH. | Same as RU-MS-30 then `mgr.save()`. List contents of `mgr.index_path`. Assert at least one file exists (e.g. textual index file). | n/a | tmp_index_dir. | Folder exists and is non-empty. |
| RU-MS-32 | Concurrency smoke: two threads each call `get_faiss_manager("ViT-B/32")` and add a vector for unique product IDs; final manager is the same instance and contains both products. | Use `threading.Thread` x2; each calls `ms.get_faiss_manager("ViT-B/32")` then `mgr.add_textual(pid, clip_vec(pid))` for pid in {"A","B"}. Join. Assert both threads got the same instance, and `mgr.get_all_sizes()["textual"] >= 2`. | pids "A","B" | embedder stubs; tmp_index_dir. | Single shared manager; both products present. Notes: same pattern as RU-FM-25; not a strict race-condition test. |

---

Wrote 32 test cases covering 14 feature bullets.
