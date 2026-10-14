"""Shared native Google GenAI client configuration."""

from google import genai
from google.genai import types

from app.config import settings
from app.services.model_runtime import provider_timeout


def create_gemini_client(*, read_timeout_seconds: float | None = None) -> genai.Client:
    api_key = settings.GEMINI_API_KEY.get_secret_value()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for model requests")

    timeout = provider_timeout(read_timeout_seconds=read_timeout_seconds)
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            client_args={"timeout": timeout},
            async_client_args={"timeout": timeout},
            retry_options=types.HttpRetryOptions(
                attempts=max(1, settings.MODEL_MAX_RETRIES + 1),
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        ),
    )
