import asyncio
import json
import time

import pytest

from app import main


@pytest.mark.asyncio
async def test_startup_dependency_timeout_degrades_without_blocking(caplog, monkeypatch):
    monkeypatch.setattr(main, "STARTUP_DEPENDENCY_TIMEOUT_SECONDS", 0.01)

    async def unavailable_dependency():
        await asyncio.sleep(1)
        return True

    healthy = await main._check_startup_dependency(
        "Redis",
        unavailable_dependency,
    )

    assert healthy is False
    assert "Redis startup health check timed out" in caplog.text


@pytest.mark.asyncio
async def test_startup_dependency_returns_health_result():
    async def healthy_dependency():
        return True

    assert await main._check_startup_dependency(
        "Redis",
        healthy_dependency,
    ) is True


@pytest.mark.asyncio
async def test_full_health_returns_degraded_when_dependencies_hang(monkeypatch):
    monkeypatch.setattr(main, "STARTUP_DEPENDENCY_TIMEOUT_SECONDS", 0.02)

    async def unavailable_dependency():
        await asyncio.sleep(1)
        return True

    def unavailable_graph():
        time.sleep(1)
        return {"configured": True, "status": "healthy"}

    monkeypatch.setattr(main, "check_database_connection", unavailable_dependency)
    monkeypatch.setattr(main, "check_redis_health", unavailable_dependency)
    monkeypatch.setattr(main, "neo4j_health", unavailable_graph)

    started_at = time.perf_counter()
    response = await main.full_health_check()
    elapsed = time.perf_counter() - started_at

    assert response.status_code == 503
    assert elapsed < 0.2
    assert json.loads(response.body)["database_status"] == "unhealthy"


@pytest.mark.asyncio
async def test_full_health_reports_healthy_dependencies(monkeypatch):
    async def healthy_dependency():
        return True

    monkeypatch.setattr(main, "check_database_connection", healthy_dependency)
    monkeypatch.setattr(main, "check_redis_health", healthy_dependency)
    monkeypatch.setattr(
        main,
        "neo4j_health",
        lambda: {"configured": True, "status": "healthy"},
    )

    result = await main.full_health_check()

    assert result["api_status"] == "healthy"
    assert result["database_status"] == "healthy"
    assert result["redis_status"] == "healthy"
