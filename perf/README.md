# Retrieval — Locust performance scaffolding

This directory contains:

- `locustfile.py` — Locust load-generator script for the Retrieval Flask
  service.
- `prometheus_metrics_reference.py` — REFERENCE wiring for
  `prometheus_client`. Not imported by production. The M6 owner copies the
  one-liner block at the bottom into `Retrieval/__init__.py` to mount
  `/metrics`.

The pytest suite at `Retrieval/tests/integration/test_performance.py`
covers single-call latency under mocks; this Locust script covers the
sustained-throughput row that pytest skips, and provides a real-traffic
cross-check for the latency rows.

## Deliverable rows covered

| Row         | Scenario                      | Locust args                   |
|-------------|-------------------------------|-------------------------------|
| PERF-RT-001 | text-search latency           | `-u 1 -r 1 -t 60s`            |
| PERF-RT-002 | late-fusion latency           | `-u 1 -r 1 -t 60s`            |
| PERF-RT-003 | sustained throughput >=20 r/s | `-u 4 -r 4 -t 60s`            |

PERF-RT-003 is the only row that genuinely needs concurrency.

## Run commands

Single-user latency (PERF-RT-001 / 002):

```
cd Retrieval
locust -f perf/locustfile.py --headless \
    -u 1 -r 1 -t 60s \
    --host http://localhost:5001 \
    --csv perf/out/rt_single
```

Sustained throughput (PERF-RT-003):

```
cd Retrieval
locust -f perf/locustfile.py --headless \
    -u 4 -r 4 -t 60s \
    --host http://localhost:5001 \
    --csv perf/out/rt_throughput
```

## Prerequisites

- Retrieval Flask app running on `http://localhost:5001`.
- FAISS indices loaded for `ViT-B/32` (both textual and visual).
- For late-fusion runs: a real product image on disk, exported via the
  `LATE_FUSION_QUERY_IMAGE_PATH` env var. If unset, the script writes the
  built-in 1x1 synthetic JPEG to a temp file and uses that — fine for
  smoke runs, but production-quality runs should use a real image.
- For SLA scraping: Prometheus configured to scrape the Retrieval
  `/metrics` endpoint at `scrape_interval: 5s`. This requires the M6 owner
  to have applied the wiring described in
  `prometheus_metrics_reference.py`.

## Output

Locust CSVs land under `perf/out/<run-name>_*.csv`. The SLA pass/fail
decision is NOT made from these CSVs — it is made from the Prometheus
histogram `retrieval_request_duration_seconds` (see the `# Prometheus
integration` block in `locustfile.py`). The CSVs are useful for
cross-checking and for offline p50/p95 computation when Prometheus is not
available.
