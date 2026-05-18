"""
Locust performance scaffolding for the Retrieval service (Flask + FAISS).

Targets the running Retrieval Flask service. The endpoints used here are
defined in Retrieval/routes/search_routes.py:

    POST /api/retrieval/search/text   {"text", "textual_model_name", "top_k"}
    POST /api/retrieval/search/late   {"text","textual_model_name",
                                       "text_weight","image","visual_model_name",
                                       "top_k"}

Deliverable rows targeted by this file
--------------------------------------
  PERF-RT-001   text-search latency single-user
                  --users 1 -t 60s
  PERF-RT-002   late-fusion latency single-user
                  --users 1 -t 60s
  PERF-RT-003   sustained throughput >= 20 req/s
                  --users 4 --spawn-rate 4 -t 60s

PERF-RT-003 is the only row that genuinely needs concurrency; the pytest
suite skips it per the Scope Limitation note. The other two rows can also
be re-run via Locust as a cross-check against the pytest single-call
latencies.

Run commands
------------
Single-user latency (PERF-RT-001 / 002):

    cd Retrieval
    locust -f perf/locustfile.py --headless \\
        -u 1 -r 1 -t 60s \\
        --host http://localhost:5001 \\
        --csv perf/out/rt_single

Sustained-throughput run (PERF-RT-003 only):

    cd Retrieval
    locust -f perf/locustfile.py --headless \\
        -u 4 -r 4 -t 60s \\
        --host http://localhost:5001 \\
        --csv perf/out/rt_throughput

# Prometheus integration
# ----------------------
# This file does NOT modify production code; it is purely a load generator.
# The SLA assertions for Retrieval (P95 < 200 ms text, P95 < 500 ms late
# fusion) are read from a Prometheus histogram, NOT from locust's CSVs:
#
#   1. The Retrieval Flask app already (or, at M6, will) expose /metrics via
#      `prometheus_client.make_wsgi_app()` — see
#      perf/prometheus_metrics_reference.py for the reference wiring.
#   2. Prometheus is configured to scrape the Retrieval /metrics endpoint at
#      a 5-second interval (scrape_interval: 5s) while these locust runs are
#      in flight.
#   3. The histogram metric `flask_http_request_duration_seconds` (or the
#      reference name `retrieval_request_duration_seconds`) is consumed by
#      an external dashboarding tool (Grafana / Prometheus query API), which
#      computes histogram_quantile(0.95, ...) per endpoint label and asserts
#      against the SLA thresholds above.
#
# In short: Locust generates the load, Prometheus records latency, the
# dashboard checks the SLA. None of those three pieces lives in this file.

Notes
-----
- The /api/retrieval/search/late task uses a 1x1 synthetic JPEG byte
  literal as a stand-in for the query image. Production runs should swap
  in a real product image (or seed one on disk and write its absolute path
  into LATE_FUSION_QUERY_IMAGE_PATH below). The route validates file size
  and path existence, so a synthetic byte literal is NOT enough on its own
  unless the ImageBytesProvider helper below is updated to write the bytes
  to a temp file each iteration.
- The 200-word in-file corpus avoids any disk/jsonl dependency.
"""

import os
import random
import tempfile

from locust import HttpUser, task, between

# ---------------------------------------------------------------------------
# Optional Prometheus integration (gated by ImportError so locust still works
# even if prometheus_client is absent). The harness sets LOCUST_PROM_PORT
# from the run_all_perf.bat. Defaults to 9303 (Retrieval slot).
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram, start_http_server
    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    _PROM_AVAILABLE = False

_PROM_PORT = int(os.environ.get("LOCUST_PROM_PORT", "9303"))

if _PROM_AVAILABLE:
    REQ_LATENCY = Histogram(
        "locust_request_duration_seconds",
        "Locust request latency",
        ["endpoint", "method", "status"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    REQ_TOTAL = Counter(
        "locust_request_total",
        "Locust request count",
        ["endpoint", "method", "status"],
    )

    from locust import events as _events

    @_events.init.add_listener
    def _start_prom(environment, **kwargs):  # noqa: D401
        try:
            start_http_server(_PROM_PORT)
        except OSError:
            pass

    @_events.request.add_listener
    def _record_request(
        request_type,
        name,
        response_time,
        response_length,
        response,
        context,
        exception,
        start_time,
        url,
        **kwargs,
    ):
        status = "fail" if exception else "ok"
        REQ_LATENCY.labels(
            endpoint=name, method=request_type, status=status
        ).observe(response_time / 1000.0)
        REQ_TOTAL.labels(
            endpoint=name, method=request_type, status=status
        ).inc()


# ---------------------------------------------------------------------------
# 200-word in-file text corpus (single-token product-style queries)
# ---------------------------------------------------------------------------

TEXT_CORPUS = [
    "shoes", "sneakers", "boots", "sandals", "heels", "loafers", "slippers",
    "jacket", "coat", "blazer", "hoodie", "sweater", "cardigan", "vest",
    "shirt", "tshirt", "polo", "blouse", "tunic", "kimono", "kaftan",
    "pants", "trousers", "jeans", "leggings", "shorts", "chinos", "joggers",
    "skirt", "dress", "gown", "jumpsuit", "romper", "overalls", "kilt",
    "hat", "cap", "beanie", "scarf", "gloves", "mittens", "earmuffs",
    "watch", "bracelet", "necklace", "ring", "earrings", "anklet", "brooch",
    "bag", "backpack", "purse", "wallet", "tote", "clutch", "satchel",
    "phone", "tablet", "laptop", "monitor", "keyboard", "mouse", "headset",
    "speaker", "earbuds", "headphones", "microphone", "webcam", "router", "modem",
    "camera", "lens", "tripod", "drone", "gimbal", "flash", "stabilizer",
    "console", "controller", "joystick", "vrset", "gamepad", "racingwheel", "steeringwheel",
    "kettle", "toaster", "blender", "mixer", "juicer", "fryer", "oven",
    "fridge", "freezer", "dishwasher", "washer", "dryer", "vacuum", "iron",
    "fan", "heater", "humidifier", "purifier", "dehumidifier", "thermostat", "kindle",
    "book", "notebook", "pen", "pencil", "marker", "highlighter", "stapler",
    "lamp", "bulb", "candle", "incense", "diffuser", "rug", "curtain",
    "couch", "sofa", "armchair", "stool", "bench", "ottoman", "recliner",
    "table", "desk", "chair", "shelf", "cabinet", "wardrobe", "drawer",
    "bed", "mattress", "pillow", "blanket", "duvet", "sheet", "comforter",
    "towel", "robe", "soap", "shampoo", "conditioner", "lotion", "perfume",
    "razor", "trimmer", "clipper", "epilator", "tweezer", "mirror", "hairdryer",
    "ball", "bat", "racket", "glove", "cleats", "gloves", "skis",
    "snowboard", "skateboard", "scooter", "bicycle", "helmet", "kneepads", "elbowpads",
    "tent", "sleepingbag", "backpack", "compass", "binoculars", "thermos", "stove",
    "lantern", "flashlight", "powerbank", "charger", "cable", "adapter", "battery",
    "umbrella", "raincoat", "boots", "poncho", "windbreaker", "balaclava", "snowsuit",
    "stroller", "carseat", "highchair", "crib", "bassinet", "playmat", "babymonitor",
    "toy", "puzzle", "lego", "action_figure", "plush", "doll", "skateboard",
    "guitar", "ukulele", "piano", "keyboard", "drum", "violin", "cello",
    "yoga_mat", "dumbbell", "kettlebell", "treadmill", "elliptical", "rowing_machine", "bench_press",
]


# ---------------------------------------------------------------------------
# Synthetic 1x1 JPEG byte literal — acts as the late-fusion query image.
# The Retrieval route reads `data["image"]` as a filesystem path, so this
# helper writes the bytes to a temp file once per process and reuses it.
# Production runs should override LATE_FUSION_QUERY_IMAGE_PATH with a real
# product image path.
# ---------------------------------------------------------------------------

ONE_PIXEL_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000"
    "ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432"
    "ffc0000b080001000101011100"
    "ffc4001f0000010501010101010100000000000000000102030405060708090a0b"
    "ffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9fa"
    "ffda0008010100003f00d2cf20ffd9"
)


def _materialize_query_image() -> str:
    """Write the synthetic JPEG to a temp file and return its absolute path."""
    override = os.environ.get("LATE_FUSION_QUERY_IMAGE_PATH")
    if override and os.path.exists(override):
        return override
    fd, path = tempfile.mkstemp(prefix="locust_query_", suffix=".jpg")
    with os.fdopen(fd, "wb") as f:
        f.write(ONE_PIXEL_JPEG)
    return path


LATE_FUSION_QUERY_IMAGE_PATH = _materialize_query_image()


class RetrievalUser(HttpUser):
    """
    Simulated client of the Retrieval Flask service.

    Weight 4:1 (text vs late-fusion) reflects the documented production
    traffic mix; the throughput SLA in PERF-RT-003 is computed across the
    aggregate, but is dominated by the text path.
    """

    wait_time = between(0.0, 0.1)

    @task(4)
    def text_search(self):
        """POST /api/retrieval/search/text — PERF-RT-001."""
        word = random.choice(TEXT_CORPUS)
        self.client.post(
            "/api/retrieval/search/text",
            json={
                "text": word,
                "textual_model_name": "ViT-B/32",
                "top_k": 10,
            },
            name="POST /api/retrieval/search/text",
        )

    @task(1)
    def late_fusion_search(self):
        """POST /api/retrieval/search/late — PERF-RT-002."""
        word = random.choice(TEXT_CORPUS)
        self.client.post(
            "/api/retrieval/search/late",
            json={
                "text": word,
                "textual_model_name": "ViT-B/32",
                "text_weight": 0.5,
                "image": LATE_FUSION_QUERY_IMAGE_PATH,
                "visual_model_name": "ViT-B/32",
                "top_k": 10,
            },
            name="POST /api/retrieval/search/late",
        )
