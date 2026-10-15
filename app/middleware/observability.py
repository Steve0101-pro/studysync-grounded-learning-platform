"""Low-overhead HTTP latency and request metrics."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.services.metrics import HTTP_REQUESTS, HTTP_REQUEST_LATENCY

ASGIMessage = dict[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]


class RequestMetricsMiddleware:
    """Measure request time with route templates instead of high-cardinality URLs."""

    def __init__(self, app):
        self.app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        if scope["type"] != "http" or scope.get("path") == "/monitoring/metrics":
            await self.app(scope, receive, send)
            return

        started_at = time.perf_counter()
        method = scope.get("method", "UNKNOWN")
        status_code = 500
        recorded = False

        async def measured_send(message: ASGIMessage) -> None:
            nonlocal status_code, recorded
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append(
                    (
                        b"x-process-time",
                        f"{time.perf_counter() - started_at:.6f}".encode("ascii"),
                    )
                )
                message = {**message, "headers": headers}
            await send(message)
            if (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                route = getattr(scope.get("route"), "path", "unmatched")
                HTTP_REQUESTS.labels(method, route, str(status_code)).inc()
                HTTP_REQUEST_LATENCY.labels(method, route).observe(
                    time.perf_counter() - started_at
                )
                recorded = True

        try:
            await self.app(scope, receive, measured_send)
        finally:
            if not recorded:
                route = getattr(scope.get("route"), "path", "unmatched")
                HTTP_REQUESTS.labels(method, route, str(status_code)).inc()
                HTTP_REQUEST_LATENCY.labels(method, route).observe(
                    time.perf_counter() - started_at
                )
