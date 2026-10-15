"""Shared request limits for external model providers."""

from openai import AsyncOpenAI
import httpx

from app.config import settings

_provider_clients: dict[tuple[str, str], AsyncOpenAI] = {}


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


def get_provider_client(api_key: str, base_url: str = "") -> AsyncOpenAI:
    """Reuse an async provider connection pool for the lifetime of the application."""
    key = (base_url, api_key)
    client = _provider_clients.get(key)
    if client is None:
        client = AsyncOpenAI(
            api_key=api_key,
            **({"base_url": base_url} if base_url else {}),
            timeout=provider_timeout(),
            max_retries=settings.MODEL_MAX_RETRIES,
        )
        _provider_clients[key] = client
    return client


async def close_provider_clients() -> None:
    clients = tuple(_provider_clients.values())
    _provider_clients.clear()
    for client in clients:
        await client.close()
