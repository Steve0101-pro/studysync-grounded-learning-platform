"""Native Gemini audio transcription for voice questions and lecture ingestion."""

import asyncio
import json
import logging
from typing import Any

import httpx
from google.genai import types
from google.genai.errors import APIError
from groq import APIConnectionError as GroqConnectionError
from groq import APIStatusError as GroqStatusError
from langsmith import traceable

from app.config import settings
from app.services.gemini_client import create_gemini_client

logger = logging.getLogger(__name__)

client = create_gemini_client(
    read_timeout_seconds=settings.TRANSCRIPTION_TIMEOUT_SECONDS
)
_TRANSCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "text": {"type": "string"},
                },
                "required": ["start", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}


def _transcription_models() -> list[str]:
    models: list[str] = []
    if settings.GROQ_API_KEY.get_secret_value():
        models.extend(
            [
                settings.GROQ_MODEL,
                settings.GROQ_STT_FALLBACK_MODEL,
            ]
        )
    for model in dict.fromkeys((settings.GEMINI_MODEL, settings.GEMINI_FALLBACK_MODEL)):
        if model:
            models.append(model)
    return [model for model in dict.fromkeys(models) if model]


async def _transcribe_segments(
    *,
    file_bytes: bytes,
    filename: str,
    content_type: str,
) -> list[dict[str, float | str]]:
    if not file_bytes:
        raise ValueError("Audio data is empty")
    if not content_type.startswith(("audio/", "video/")):
        raise ValueError("Transcription requires an audio or video content type")

    prompt = (
        "Transcribe the spoken words verbatim, preserving the original language. "
        "Return chronological speech segments with each segment's start time in "
        "seconds from the beginning of the media. Keep segment text concise and "
        "do not infer speech or add commentary. The uploaded filename is "
        f"{filename!r}. Return the required JSON object."
    )
    contents = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=file_bytes, mime_type=content_type),
    ]
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=_TRANSCRIPTION_SCHEMA,
        temperature=0,
        max_output_tokens=8_192,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )
    models = _transcription_models()
    if not models:
        raise RuntimeError("No transcription model is configured")

    failed_provider: str | None = None
    groq_models = {settings.GROQ_MODEL, settings.GROQ_STT_FALLBACK_MODEL}
    for index, model in enumerate(models):
        provider = "groq" if model in groq_models else "gemini"
        if provider == failed_provider:
            continue
        try:
            if provider == "groq":
                groq_transport = getattr(client, "audio", None)
                if groq_transport is None or not hasattr(groq_transport, "transcriptions"):
                    from openai import AsyncOpenAI

                    groq_transport = AsyncOpenAI(
                        api_key=settings.GROQ_API_KEY.get_secret_value(),
                        base_url="https://api.groq.com/openai/v1",
                    )

                response = await asyncio.wait_for(
                    groq_transport.transcriptions.create(
                        model=model,
                        file=(filename, file_bytes, content_type),
                    ),
                    timeout=settings.TRANSCRIPTION_TIMEOUT_SECONDS,
                )
                text = getattr(response, "text", "")
                if not text:
                    raise ValueError("Groq returned an empty transcription")
                return [{"start": 0.0, "text": text.strip()}]

            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                ),
                timeout=settings.TRANSCRIPTION_TIMEOUT_SECONDS,
            )
            if not response.text:
                raise ValueError("Gemini returned an empty transcription")
            payload = json.loads(response.text)
            segments = payload.get("segments") if isinstance(payload, dict) else None
            if not isinstance(segments, list):
                raise ValueError("Gemini returned invalid timestamped segments")
            normalized = [
                {"start": float(item["start"]), "text": item["text"].strip()}
                for item in segments
                if isinstance(item, dict)
                and isinstance(item.get("text"), str)
                and item["text"].strip()
                and isinstance(item.get("start"), (int, float))
                and item["start"] >= 0
            ]
            if not normalized:
                raise ValueError("Transcription contained no speech segments")
            return normalized
        except (APIError, httpx.HTTPError, GroqConnectionError, GroqStatusError, TimeoutError, ValueError) as exc:
            if index == len(models) - 1:
                raise
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in {400, 401, 403}:
                failed_provider = provider
            logger.warning(
                "Transcription model %s failed with %s; trying next configured provider",
                model,
                type(exc).__name__,
            )
    raise RuntimeError("No transcription model is configured")


@traceable(
    name="gemini_audio_transcription",
    run_type="llm",
    tags=["speech-to-text", "audio", "gemini"],
)
async def transcribe_audio(file_bytes: bytes, filename: str, content_type: str) -> str:
    """Transcribe audio bytes using Gemini's native audio input."""
    try:
        segments = await _transcribe_segments(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
        )
        return " ".join(str(segment["text"]) for segment in segments)
    except Exception as exc:
        logger.error("Gemini audio transcription failed with %s", type(exc).__name__)
        raise RuntimeError("Failed to transcribe audio data.") from exc


@traceable(
    name="gemini_lecture_video_transcription",
    run_type="llm",
    tags=["lecture-ingestion", "timestamps", "gemini"],
)
async def transcribe_audio_segments(
    file_bytes: bytes,
    filename: str,
    content_type: str,
) -> list[dict[str, float | str]]:
    """Transcribe media and preserve model-provided per-segment start times."""
    try:
        return await _transcribe_segments(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
        )
    except Exception as exc:
        logger.error("Gemini lecture transcription failed with %s", type(exc).__name__)
        raise RuntimeError("Failed to transcribe lecture audio.") from exc
