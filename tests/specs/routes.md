# Routes — Test Specification

**Source files under test:**
- `routes/system_routes.py` — health/index-stats/models endpoints (`system_bp`).
- `routes/product_routes.py` — add/update/delete product endpoints (`product_bp`).
- `routes/search_routes.py` — text/image/late-fusion/early-fusion/cross-modal search endpoints (`search_bp`).

**Endpoints discovered (read from source):**

| Method | Path | Handler |
|--------|------|---------|
| GET | `/api/health` | `system_routes.health_check` |
| GET | `/api/retrieval/models` | `system_routes.get_models` |
| GET | `/api/retrieval/index-stats` | `system_routes.get_index_stats` |
| POST | `/api/retrieval/add-product` | `product_routes.add_product` |
| PUT | `/api/retrieval/update-product/<product_id>` | `product_routes.update_product` |
| DELETE | `/api/retrieval/delete-product/<product_id>` | `product_routes.delete_product` |
| POST | `/api/retrieval/search/text` | `search_routes.text_search` |
| POST | `/api/retrieval/search/image` | `search_routes.image_search` |
| POST | `/api/retrieval/search/late` | `search_routes.late_fusion_search` |
| POST | `/api/retrieval/search/early` | `search_routes.early_fusion_search` |
| POST | `/api/retrieval/search/image-by-text` | `search_routes.image_by_text_search` |
| POST | `/api/retrieval/search/text-by-image` | `search_routes.text_by_image_search` |

**ID prefixes:**
- `RU-RT-NN` — route-level UNIT tests (handler tested through Flask test client; manager_service / embedder boundary stubbed; real in-memory FAISSManager allowed). Drives `tests/test_routes_unit.py`.
- `RI-L2-NN` — route INTEGRATION tests (full request cycle, real `services.manager_service`, real in-memory FAISSManager rooted at a temp `DATA_BASE_PATH`; only model-pool boundary stubbed). Drives `tests/test_routes_integration.py`.

**Mocking rules (enforced):**
1. NEVER mock `faiss.*` or `FAISSManager`. Use real in-memory FAISSManagers (the manager constructs new IndexFlatIP via real `faiss`).
2. Persistence paths: monkeypatch `services.manager_service.DATA_BASE_PATH` to `tmp_index_dir` for every test that exercises caches or save/load.
3. Embedders are stubbed at the manager-pool boundary — patch the three manager classes (`TextModelManager`, `VisualModelManager`, `FusedModelManager`) inside `services.manager_service` with fake managers whose `get_embedding` / `get_document_embedding` return deterministic L2-normalized vectors of the right dimension (CLIP=512, BGE=1024, Marqo=1024, Qwen=4096, DINOv3=4096) keyed off a hash of the input.
4. Every test resets the module caches (`_faiss_managers`, `_textual_managers`, `_visual_managers`, `_fused_managers`) — the code-writer must add an autouse fixture `manager_service_clean` so each test starts from a clean slate. The Pre-condition column references this implicitly.
5. No real model weights ever loaded.

**Fixtures (existing + new, to be added by code-writer in conftest):**
- `tmp_index_dir` (existing) — temporary FAISS root.
- `tmp_image_path` (existing) — small valid PNG/JPG.
- `clip_vec`, `bge_vec`, `marqo_vec`, `qwen_vec`, `dinov3_vec` (existing) — deterministic L2-normalized vector generators.
- `flask_client` (NEW) — yields `app.test_client()`; autouse-paired with `manager_service_clean` which (a) clears the four module caches, (b) monkeypatches `DATA_BASE_PATH` to `tmp_index_dir`.
- `stub_managers` (NEW) — patches `services.manager_service.TextModelManager`, `.VisualModelManager`, `.FusedModelManager` with fake classes that deterministically return vectors of the right dimension.
- `oversize_image_path` (NEW) — file > 50 MB inside a tmp dir, used to trigger `validate_image_file_size`.

**This spec drives BOTH `tests/test_routes_unit.py` (RU-RT-NN) and `tests/test_routes_integration.py` (RI-L2-NN).**

---

| ID | Description | Steps | Test Data | Pre-condition | Expected Output |
|----|-------------|-------|-----------|---------------|-----------------|
| RU-RT-01 | GET /api/health returns healthy | Issue `client.get("/api/health")`. Assert status 200. Assert body JSON `status == "healthy"` and `service` field present. | None | `flask_client` ready | 200 + JSON `{"status":"healthy","service":"E-Commerce Product Retrieval System"}` |
| RU-RT-02 | GET /api/retrieval/models happy path | Monkeypatch `services.manager_service.get_available_models` to return `{"textual_models":[{"name":"ViT-B/32","dimension":512}],"visual_models":[],"defaults":{"textual":"ViT-B/32","visual":"ViT-B/32"}}`. GET `/api/retrieval/models`. | Stubbed `get_available_models` | `flask_client` ready | 200 + JSON `status=="success"` and `data` matches stub return |
| RU-RT-03 | GET /api/retrieval/models error path | Monkeypatch `get_available_models` to raise `RuntimeError("boom")`. GET `/api/retrieval/models`. | Patched function raising | `flask_client` ready | 500 + JSON `status=="error"`, message contains "boom" |
| RU-RT-04 | GET /api/retrieval/index-stats happy path | Monkeypatch `services.manager_service.get_all_index_stats` to return `{"ViT-B-32_512_embeddings":{"textual":3,"visual":2,"fused":0}}`. GET `/api/retrieval/index-stats`. | Stubbed function | `flask_client` ready | 200 + JSON `status=="success"`, `indices` equals the stub dict |
| RU-RT-05 | GET /api/retrieval/index-stats error path | Monkeypatch `get_all_index_stats` to raise `Exception("kapow")`. GET endpoint. | Stub raises | `flask_client` ready | 500 + JSON `status=="error"`, message contains "kapow" |
| RU-RT-06 | POST /api/retrieval/add-product happy path | With `stub_managers` active, POST JSON with id, name, textual_model_name="ViT-B/32", visual_model_name="ViT-B/32", `images=[]`. | `{"id":"p1","name":"Shoe","textual_model_name":"ViT-B/32","visual_model_name":"ViT-B/32"}` | `flask_client` + `stub_managers` | 201 + JSON status="success", details.product_id=="p1", textual_vector_id is int, visual_vector_ids==[], images_processed==0 |
| RU-RT-07 | POST /add-product missing required field returns 400 | POST with id+name only (no textual_model_name). | `{"id":"p2","name":"X"}` | `flask_client` | 400 + message contains "Missing required field: textual_model_name" |
| RU-RT-08 | POST /add-product missing JSON body | POST with empty body and no Content-Type. | empty body | `flask_client` | 400 OR 415 — assert non-2xx, JSON status=="error". Notes: Flask `get_json()` returns None when no JSON; `validate_required_fields` returns 400 "Request body must be valid JSON". |
| RU-RT-09 | POST /add-product image not found returns 400 | With `stub_managers` configured so `VisualModelManager.get_embedding` raises `FileNotFoundError("missing.jpg")`. POST id+name+models+images=["missing.jpg"]. | images=["C:/no/such/file.jpg"] | `stub_managers` raises FNF on get_embedding | 400 + message starts with "Image not found:" |
| RU-RT-10 | POST /add-product oversize image returns 400 | Use `oversize_image_path` fixture (>50MB). POST with images=[path]. `validate_image_file_size` raises ValueError. | images=[oversize path] | File exists, size > MAX_IMAGE_SIZE_BYTES | 400 + message contains "exceeds maximum limit of 50MB" |
| RU-RT-11 | POST /add-product duplicate skip path | Pre-seed real FAISSManager (via add) so product "p3" already in textual+visual. POST same id again with no images. | re-add p3 | textual+visual already_indexed both true | 200 + message "...already has embeddings...", details.skipped==True |
| RU-RT-12 | POST /add-product fused enabled with image | Provide fused_model_name="ViT-B/32" plus 1 image (`tmp_image_path`). | id="p4", images=[tmp_image_path], fused_model_name="ViT-B/32" | `stub_managers`, real FAISSManager | 201 + details has fused_vector_ids of length 1, fused_skipped==False |
| RU-RT-13 | DELETE /api/retrieval/delete-product/<id> happy path | Pre-add "p5" via real flow. DELETE `/api/retrieval/delete-product/p5`. | id "p5" exists | data added | 200 + JSON status="success", removed_counts non-empty, total_removed > 0 |
| RU-RT-14 | DELETE /delete-product unknown id returns 404 | DELETE `/api/retrieval/delete-product/ghost` with empty caches. | id "ghost" | clean caches | 404 + message contains "not found in any index" |
| RU-RT-15 | PUT /update-product happy path | Pre-add "p6". PUT `/api/retrieval/update-product/p6` with name+models. | name "newname", models ViT-B/32 | p6 exists, `stub_managers` | 200 + status="success", details.removed_counts present, textual_vector_id is int |
| RU-RT-16 | PUT /update-product missing required field | PUT with body `{"name":"x"}` (no textual_model_name). | partial body | n/a | 400 + "Missing required field: textual_model_name" |
| RU-RT-17 | PUT /update-product image not found | PUT with images=["nope.jpg"]; `VisualModelManager.get_embedding` raises FileNotFoundError. | images=["nope.jpg"] | stub raises | 400 + message starts "Image not found:" |
| RU-RT-18 | POST /search/text happy path | Pre-add 1 product via flow. POST `/search/text` with text + textual_model_name. | `{"text":"shoe","textual_model_name":"ViT-B/32"}` | `stub_managers`, p1 added | 200 + status="success", results is list, meta.total_results >= 0 |
| RU-RT-19 | POST /search/text missing text returns 400 | POST with only textual_model_name. | `{"textual_model_name":"ViT-B/32"}` | n/a | 400 + "Missing required field: text" |
| RU-RT-20 | POST /search/text invalid top_k=0 returns 400 | POST with text+model+`top_k=0`. | top_k=0 | n/a | 400 + message contains "top_k must be >= 1" |
| RU-RT-21 | POST /search/text top_k="abc" returns 400 | POST with top_k="abc". | top_k="abc" | n/a | 400 + message contains "top_k must be a valid integer" |
| RU-RT-22 | POST /search/text top_k missing uses DEFAULT_TOP_K | POST without top_k; spy on `faiss_manager.search_textual` (wrap real method) and verify it received `top_k = DEFAULT_TOP_K * 2 = 20`. | no top_k | `stub_managers`; spy installed | 200; spy observed top_k argument == 20 |
| RU-RT-23 | POST /search/text top_k clamped at MAX_TOP_K | POST with top_k=10000; spy on search_textual; verify arg == MAX_TOP_K * 2 = 200 (clamped to 100 first then *2). | top_k=10000 | spy installed | 200; spy observed top_k == 200; response result list length <= MAX_TOP_K (100) |
| RU-RT-24 | POST /search/image happy path | Pre-add product with 1 image. POST `/search/image` with `image=tmp_image_path` + visual_model_name. | image=tmp_image_path | data exists, `stub_managers` | 200 + results list, each entry has product_id/score/best_image_no |
| RU-RT-25 | POST /search/image missing image field returns 400 | POST `{"visual_model_name":"ViT-B/32"}`. | n/a | n/a | 400 + "Missing required field: image" |
| RU-RT-26 | POST /search/image oversize image returns 400 | POST with image=oversize_image_path. | oversize file | file exists | 400 + message contains "exceeds maximum limit of 50MB" |
| RU-RT-27 | POST /search/image invalid top_k=-1 returns 400 | POST with valid image+model and top_k=-1. | top_k=-1 | n/a | 400 + "top_k must be >= 1" |
| RU-RT-28 | POST /search/late happy path | Pre-add product with image. POST with text+textual_model+text_weight=0.5+image+visual_model. | full late-fusion body | data exists, `stub_managers` | 200 + results list, meta.text_weight==0.5, meta.image_weight==0.5 |
| RU-RT-29 | POST /search/late missing text_weight returns 400 | POST without text_weight. | partial body | n/a | 400 + "Missing required field: text_weight" |
| RU-RT-30 | POST /search/late text_weight out of range returns 400 | POST with text_weight=1.5. | text_weight=1.5 | n/a | 400 + "text_weight must be between 0 and 1" |
| RU-RT-31 | POST /search/early happy path | Pre-add product with fused index. POST with text+image+fused_model_name+text_weight. | fused body | fused index has data, `stub_managers` | 200 + status="success", meta.text_weight present |
| RU-RT-32 | POST /search/early missing fused_model_name returns 400 | POST with text+image only. | partial body | n/a | 400 + "Missing required field: fused_model_name" |
| RU-RT-33 | POST /search/early invalid top_k="0" returns 400 | POST with valid body + top_k=0. | top_k=0 | n/a | 400 + "top_k must be >= 1" |
| RU-RT-34 | POST /search/image-by-text non-multimodal model returns 400 | POST with fused_model_name="bge-large-en-v1.5" (a textual-only model). `validate_clip_model` should raise. | fused_model_name="bge-large-en-v1.5" | MODEL_REGISTRY contains BGE entry of type "textual" | 400 + message contains "multimodal" or "Cross-modal search requires" |
| RU-RT-35 | POST /search/image-by-text happy path | Pre-add product with fused index using ViT-B/32. POST with text + fused_model_name="ViT-B/32". | clip body | fused index populated, `stub_managers` | 200 + status="success", results list |
| RU-RT-36 | POST /search/image-by-text missing text returns 400 | POST with `{"fused_model_name":"ViT-B/32"}`. | partial body | n/a | 400 + "Missing required field: text" |
| RU-RT-37 | POST /search/image-by-text top_k clamp | POST with text+fused_model_name+top_k=999; spy on search_fused; verify top_k arg == 200. | top_k=999 | spy installed | 200; observed top_k == MAX_TOP_K*2 |
| RU-RT-38 | POST /search/text-by-image non-multimodal model returns 400 | POST with image+fused_model_name="bge-large-en-v1.5". | non-multimodal | MODEL_REGISTRY entry exists | 400 + "multimodal" in message |
| RU-RT-39 | POST /search/text-by-image happy path | Pre-add fused product. POST with image=tmp_image_path + fused_model_name="ViT-B/32". | clip body | fused index populated | 200 + status="success" |
| RU-RT-40 | POST /search/text-by-image missing image returns 400 | POST with `{"fused_model_name":"ViT-B/32"}`. | partial body | n/a | 400 + "Missing required field: image" |
| RU-RT-41 | POST /search/text-by-image oversize image returns 400 | POST with image=oversize_image_path + fused_model_name="ViT-B/32". | oversize file | file exists | 400 + "exceeds maximum limit of 50MB" |
| RU-RT-42 | POST /search/text empty index returns empty list | Clean caches; do NOT add any products. POST `/search/text` with text+model. | clean state | empty real FAISSManager | 200 + status="success", `results == []`, meta.total_results==0 |
| RU-RT-43 | POST /search/late combined_score equals weighted formula `text_weight*text_score + (1-text_weight)*image_score` rounded to 6 decimals | With `stub_managers` active, configure fake `TextModelManager.get_embedding` and `VisualModelManager.get_embedding`/`get_document_embedding` so the query and stored vectors are known (e.g., query text vector == stored textual vector for product "pw1" giving text_score==1.0; query image vector dotted with stored visual vector giving a known image_score, e.g., 0.5). Pre-add product "pw1" via real flow. POST `/search/late` with text+textual_model+image+visual_model+text_weight=0.7. Capture `body["results"][0]["text_score"]`, `["image_score"]`, `["combined_score"]`. Assert `combined_score == round(0.7*text_score + 0.3*image_score, 6)`. | text_weight=0.7; controllable stub vectors | clean caches; `stub_managers`; `flask_client`; `tmp_index_dir` | 200; combined_score equals weighted-sum formula to 6 decimals |
| RU-RT-44 | POST /search/late results sorted descending by combined_score | Configure stub vectors so two pre-indexed products P1 and P2 have known (text_score, image_score) pairs that, at text_weight=0.5, yield combined_score(P1) > combined_score(P2) (e.g., P1 (1.0,0.4) → 0.7; P2 (0.6,0.6) → 0.6). Pre-add P1 then P2 via real flow. POST `/search/late` with text_weight=0.5. Assert `body["results"][0]["product_id"]` is the higher-combined-score product (P1) and that the list `[r["combined_score"] for r in body["results"]]` is monotonically non-increasing. | two products with engineered scores | clean caches; `stub_managers` | 200; first result is P1; combined_score list non-increasing |
| RI-L2-01 | Add then text-search round-trip | POST `/api/retrieval/add-product` for id="rt1", name="Red Sneaker", textual_model_name="ViT-B/32", visual_model_name="ViT-B/32". Then POST `/api/retrieval/search/text` with text="Red Sneaker", textual_model_name="ViT-B/32", top_k=5. | full add + search bodies | clean caches, `stub_managers`, `tmp_index_dir` as DATA_BASE_PATH | Add → 201; Search → 200, results non-empty, top result product_id=="rt1" |
| RI-L2-02 | Add with image then image-search round-trip | POST add-product id="rt2" with images=[tmp_image_path] + ViT-B/32 models. Then POST `/search/image` with image=tmp_image_path. | clip body w/ image | tmp_image_path exists, `stub_managers` | Add → 201, visual_vector_ids length==1; Search → 200, top result product_id=="rt2" |
| RI-L2-03 | Delete after add removes from search | Add id="rt3" via flow. DELETE `/api/retrieval/delete-product/rt3`. POST `/search/text` with same text. | full bodies | rt3 added | DELETE → 200, total_removed > 0; subsequent search → 200, no result has product_id=="rt3" |
| RI-L2-04 | top_k clamp end-to-end on /search/text | Add 3 products via flow. POST `/search/text` with top_k=200 (above MAX_TOP_K). | top_k=200 | 3 products indexed | 200; len(results) <= 100 (MAX_TOP_K); meta.total_results <= 3 |
| RI-L2-05 | image-by-text cross-modal round-trip | Add product "rt5" with fused_model_name="ViT-B/32" and 1 image. POST `/search/image-by-text` with text="Red Sneaker", fused_model_name="ViT-B/32". | full body | fused index built, `stub_managers` (fused returns deterministic vec) | 200; results non-empty, top product_id=="rt5". Notes: deterministic fakes mean ordering is by hash similarity — assert rt5 appears in results, not necessarily exact rank if multiple products. |
| RI-L2-06 | text-by-image cross-modal round-trip | Add product "rt6" with fused index and image=tmp_image_path. POST `/search/text-by-image` with image=tmp_image_path, fused_model_name="ViT-B/32". | clip body | fused index has rt6 | 200; results contains entry with product_id=="rt6" |
| RI-L2-07 | index-stats reflects added products | Add 2 products via flow with ViT-B/32 (1 with image, 1 text-only). GET `/api/retrieval/index-stats`. | n/a | clean caches → 2 adds | 200; indices contains key whose value has textual >= 2 and visual >= 1 |
| RI-L2-08 | Adding same product twice — dedup at search | Add id="rt8" with one image. Add id="rt8" again with same payload. POST `/search/text`. | duplicate adds | n/a | 2nd add → 200 with skipped==True; search → 200 results contains rt8 exactly once |
| RI-L2-09 | Persistence smoke test (save → reload → search) | Add id="rt9" via flow (causes save to tmp_index_dir). Clear `_faiss_managers` cache only (force reload from disk). POST `/search/text`. | n/a | tmp_index_dir persists, lazy-load triggers from disk | 200; search results include rt9. Notes: relies on FAISSManager auto-load in `get_faiss_manager`. If component does not auto-load, mark SKIP — implementation detail of manager_service. |
| RI-L2-10 | Update endpoint replaces embeddings | Add id="rt10" with name="Old". PUT `/api/retrieval/update-product/rt10` with name="Brand New Hat" and same models. POST `/search/text` with text="Brand New Hat". | full bodies | rt10 added | PUT → 200, removed_counts non-zero; search → 200, top result product_id=="rt10" |
| RI-L2-11 | Late fusion end-to-end | Add id="rt11" with 1 image+ViT-B/32. POST `/search/late` with text+text_weight=0.5+image+models. | full late body | rt11 in textual+visual indices | 200; results list contains entry with product_id=="rt11"; meta.text_weight==0.5 and image_weight==0.5 |
| RI-L2-12 | Cross-modal rejected for non-multimodal model end-to-end | Add a product using textual_model_name="bge-large-en-v1.5". POST `/search/image-by-text` with fused_model_name="bge-large-en-v1.5". | non-multimodal | bge entry in MODEL_REGISTRY (type "textual") | 400; message contains "multimodal" |
| RI-L2-13 | End-to-end /search/late combined_score formula via real Flask request cycle | Through the real request cycle (no internal helper assertions): POST `/api/retrieval/add-product` for id="rl1" and id="rl2" with textual_model_name="ViT-B/32" and visual_model_name="ViT-B/32" and one image each (`tmp_image_path`). Then POST `/search/late` with text="query", textual_model_name="ViT-B/32", visual_model_name="ViT-B/32", image=tmp_image_path, text_weight=0.4. From the response take the top result; assert `combined_score == round(0.4*text_score + 0.6*image_score, 6)` to within 1e-6. | text_weight=0.4; two indexed products | clean caches; `stub_managers` (model-pool boundary only); `tmp_index_dir` | 200; for top result `abs(combined_score - round(0.4*text_score + 0.6*image_score, 6)) <= 1e-6` |

---

**File mapping:** Rows `RU-RT-01` through `RU-RT-44` go in `tests/test_routes_unit.py`. Rows `RI-L2-01` through `RI-L2-13` go in `tests/test_routes_integration.py`.

Wrote 54 test cases covering all listed feature bullets (12 endpoints × happy/error paths + 12 integration round-trips).

Gap-closure round (test plan §3.4): IDs RU-EM-84..90 / RU-RT-43..44 / RI-L2-13.
