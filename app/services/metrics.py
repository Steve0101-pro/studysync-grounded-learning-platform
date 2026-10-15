"""Low-cardinality Prometheus metrics for online evaluation and operations."""

import time
from contextlib import asynccontextmanager

from prometheus_client import Counter, Histogram, generate_latest

REQUESTS = Counter("document_intelligence_requests_total", "Requests by workflow and status", ["workflow", "status"])
FEEDBACK = Counter("document_intelligence_feedback_total", "Submitted feedback", ["rating", "variant"])
LATENCY = Histogram("document_intelligence_workflow_latency_seconds", "Workflow latency", ["workflow"])
HTTP_REQUESTS = Counter(
    "document_intelligence_http_requests_total",
    "HTTP requests by method, route template, and status code",
    ["method", "route", "status_code"],
)
HTTP_REQUEST_LATENCY = Histogram(
    "document_intelligence_http_request_duration_seconds",
    "End-to-end HTTP request duration, including streamed response delivery",
    ["method", "route"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, float("inf")),
)
DEPENDENCY_LATENCY = Histogram(
    "document_intelligence_dependency_duration_seconds",
    "Time spent waiting for downstream dependencies",
    ["dependency", "operation"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, float("inf")),
)
EVALUATIONS = Counter("document_intelligence_evaluations_total", "Online evaluations by status", ["status", "variant"])
JUDGE_FAILURES = Counter("document_intelligence_judge_failures_total", "LLM judge failures")


@asynccontextmanager
async def track_dependency(dependency: str, operation: str):
    """Observe a downstream operation without recording sensitive/high-cardinality labels."""
    started_at = time.perf_counter()
    try:
        yield
    finally:
        DEPENDENCY_LATENCY.labels(dependency, operation).observe(
            time.perf_counter() - started_at
        )


def metrics_payload() -> bytes:
    return generate_latest()