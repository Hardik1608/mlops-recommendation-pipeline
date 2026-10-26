"""Prometheus metrics exposed on ``/metrics``.

Label cardinality is bounded on purpose: routes use the template (``/recommendations/{user_id}``,
never the raw path) and variants are ``champion|challenger|shadow|none``.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
)

HTTP_REQUESTS = Counter(
    "recsys_http_requests_total",
    "HTTP requests by route, method, status code and A/B variant.",
    ["route", "method", "status", "variant"],
)
HTTP_LATENCY = Histogram(
    "recsys_http_request_duration_seconds",
    "End-to-end request latency inside the service.",
    ["route", "variant"],
    buckets=LATENCY_BUCKETS,
)
RECOMMENDATIONS = Counter(
    "recsys_recommendations_total",
    "Recommendation responses by variant and serving strategy.",
    ["variant", "strategy"],
)
CACHE_REQUESTS = Counter(
    "recsys_cache_requests_total",
    "Response-cache operations by result (hit|miss|error|bypass|set_error).",
    ["result"],
)
COLD_STARTS = Counter(
    "recsys_cold_start_total",
    "Requests for users unknown to the model, by fallback strategy (fold_in|popularity).",
    ["strategy"],
)
FEATURE_STORE_ERRORS = Counter(
    "recsys_feature_store_errors_total",
    "Online feature lookups that failed (request served without real-time context).",
)
MODEL_COMPUTE = Histogram(
    "recsys_model_compute_seconds",
    "Time spent in model scoring + top-k (excludes cache and feature lookups).",
    ["variant"],
    buckets=LATENCY_BUCKETS,
)
SHADOW_REQUESTS = Counter(
    "recsys_shadow_requests_total",
    "Requests also scored by the shadow model, by result (ok|error). Never served.",
    ["result"],
)
SHADOW_OVERLAP = Histogram(
    "recsys_shadow_overlap_ratio",
    "Share of the served top-k that the shadow model also recommends (0 = disjoint).",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
MODEL_INFO = Gauge(
    "recsys_model_info",
    "Loaded model metadata (value is always 1).",
    ["variant", "model_type", "version"],
)
MODEL_LOADED_AT = Gauge(
    "recsys_model_loaded_timestamp_seconds",
    "Unix time at which the variant's model was loaded.",
    ["variant"],
)
