# REFERENCE FILE — NOT WIRED. Production routes/__init__.py must opt in.
#
# This file demonstrates how the M6 owner can wire `prometheus_client` into
# the Retrieval Flask app so Locust runs (perf/locustfile.py) can be scraped
# and per-endpoint P95 latencies can be asserted against the deliverable's
# SLA targets:
#
#     PERF-RT-001  text-search     P95 < 200 ms
#     PERF-RT-002  late-fusion     P95 < 500 ms
#     PERF-RT-003  throughput      >= 20 req/s sustained
#
# Nothing in this module is imported by production. It exists ONLY as a
# copy-paste reference. The exact one-liner needed to actually mount the
# /metrics endpoint is documented as a code-block comment at the bottom of
# this file so it cannot accidentally be executed by an `import *`.
#
# Hard rule: this file MUST NOT be imported from any production module
# (`Retrieval/services/`, `Retrieval/routes/`, `Retrieval/models/`,
# `Retrieval/__init__.py`). The integration-test agent treats those paths
# as read-only.

import time

from flask import Flask, request, g
from prometheus_client import Counter, Histogram, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# Latency histogram. Buckets cover 5 ms .. 1 s with finer resolution under
# the SLA thresholds (200 ms text, 500 ms late fusion). Adjust as needed.
RETRIEVAL_REQUEST_DURATION_SECONDS = Histogram(
    "retrieval_request_duration_seconds",
    "Retrieval request duration in seconds, partitioned by endpoint and status.",
    labelnames=("endpoint", "status"),
    buckets=(
        0.005, 0.010, 0.025, 0.050, 0.075,
        0.100, 0.150, 0.200, 0.300, 0.400,
        0.500, 0.750, 1.000,
    ),
)

# Request counter (handy for QPS dashboards independent of the histogram).
RETRIEVAL_REQUEST_TOTAL = Counter(
    "retrieval_request_total",
    "Total number of retrieval requests, partitioned by endpoint and status.",
    labelnames=("endpoint", "status"),
)


# ---------------------------------------------------------------------------
# Flask hook pair (reference implementation)
# ---------------------------------------------------------------------------

def _start_timer() -> None:
    """before_request hook — stamp start time on flask.g."""
    g._prom_start_time = time.perf_counter()


def _record_metrics(response):
    """after_request hook — observe latency and bump counter."""
    start = getattr(g, "_prom_start_time", None)
    if start is not None:
        elapsed = time.perf_counter() - start
        endpoint = request.endpoint or "<unknown>"
        status = str(response.status_code)
        RETRIEVAL_REQUEST_DURATION_SECONDS.labels(
            endpoint=endpoint,
            status=status,
        ).observe(elapsed)
        RETRIEVAL_REQUEST_TOTAL.labels(
            endpoint=endpoint,
            status=status,
        ).inc()
    return response


def install_metrics(app: Flask) -> Flask:
    """
    Reference installer. The M6 owner would call this from wherever the
    Retrieval WSGI app is composed (typically `Retrieval/__init__.py` or
    `Retrieval/app.py`).

    NOTE: This function is here for documentation only. It is NOT invoked
    anywhere in production.
    """
    app.before_request(_start_timer)
    app.after_request(_record_metrics)

    # Mount the prometheus WSGI app at /metrics. DispatcherMiddleware lets
    # the Flask app and the metrics WSGI app coexist on the same port.
    metrics_app = make_wsgi_app()
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": metrics_app})
    return app


# ---------------------------------------------------------------------------
# One-liner the M6 owner adds to the production WSGI composer.
# ---------------------------------------------------------------------------
#
# In `Retrieval/__init__.py` (or `Retrieval/app.py`, wherever `app = Flask(__name__)`
# is created), add the following block AFTER all blueprints are registered.
# It is intentionally written as a code-block COMMENT so this reference file
# cannot accidentally execute it:
#
#     # >>> Prometheus metrics (M6 opt-in) <<<
#     # from prometheus_client import make_wsgi_app
#     # from werkzeug.middleware.dispatcher import DispatcherMiddleware
#     # from perf.prometheus_metrics_reference import (
#     #     _start_timer, _record_metrics,
#     # )
#     # app.before_request(_start_timer)
#     # app.after_request(_record_metrics)
#     # app.wsgi_app = DispatcherMiddleware(
#     #     app.wsgi_app, {"/metrics": make_wsgi_app()},
#     # )
#     # >>> end Prometheus metrics <<<
#
# Once that block is uncommented and shipped, Locust runs against the
# Retrieval service will produce histogram samples that Prometheus can
# scrape at /metrics, and the SLA dashboards described in
# perf/locustfile.py's "# Prometheus integration" block will populate.
