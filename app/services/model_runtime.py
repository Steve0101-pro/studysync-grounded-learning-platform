"""Shared request limits for external model providers."""

import httpx

from app.config import settings


def provider_timeout(*, read_timeout_seconds: float | None = None) -> httpx.Timeout:
    return httpx.Timeout(
        connect=settings.MODEL_CONNECT_TIMEOUT_SECONDS,
        read=(
            read_timeout_seconds
            if read_timeout_seconds is not None
            else settings.MODEL_READ_TIMEOUT_SECONDS
        ),
        write=settings.MODEL_WRITE_TIMEOUT_SECONDS,
        pool=settings.MODEL_POOL_TIMEOUT_SECONDS,
    )
