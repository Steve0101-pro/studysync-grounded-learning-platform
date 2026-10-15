from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app as main_app
from app.middleware.observability import RequestMetricsMiddleware
from app.services.metrics import HTTP_REQUESTS, HTTP_REQUEST_LATENCY


@pytest.mark.asyncio
async def test_request_metrics_use_route_template_and_time_stream_delivery():
    async def application(_scope, _receive, send):
        await send({"type": "http.response.start", "status": 201, "headers": []})
        await send({"type": "http.response.body", "body": b"part", "more_body": True})
        await send({"type": "http.response.body", "body": b"done", "more_body": False})

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/items/123",
        "route": SimpleNamespace(path="/items/{item_id}"),
    }
    sent = []

    async def send(message):
        sent.append(message)

    before_count = HTTP_REQUESTS.labels(
        "GET", "/items/{item_id}", "201"
    )._value.get()
    before_latency = HTTP_REQUEST_LATENCY.labels(
        "GET", "/items/{item_id}"
    )._sum.get()

    await RequestMetricsMiddleware(application)(scope, None, send)

    assert HTTP_REQUESTS.labels("GET", "/items/{item_id}", "201")._value.get() == before_count + 1
    assert HTTP_REQUEST_LATENCY.labels("GET", "/items/{item_id}")._sum.get() > before_latency
    headers = dict(sent[0]["headers"])
    assert float(headers[b"x-process-time"]) >= 0


@pytest.mark.asyncio
async def test_metrics_endpoint_is_excluded_from_request_metrics():
    async def application(_scope, _receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"metrics"})

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/monitoring/metrics",
        "route": SimpleNamespace(path="/monitoring/metrics"),
    }
    sent = []

    async def send(message):
        sent.append(message)

    before_count = HTTP_REQUESTS.labels(
        "GET", "/monitoring/metrics", "200"
    )._value.get()

    await RequestMetricsMiddleware(application)(scope, None, send)

    assert HTTP_REQUESTS.labels(
        "GET", "/monitoring/metrics", "200"
    )._value.get() == before_count
    assert b"x-process-time" not in dict(sent[0]["headers"])


@pytest.mark.asyncio
async def test_application_wires_request_metrics_middleware():
    async with AsyncClient(
        transport=ASGITransport(app=main_app),
        base_url="http://test",
    ) as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert float(response.headers["x-process-time"]) >= 0
    assert HTTP_REQUESTS.labels("GET", "/", "200")._value.get() >= 1
