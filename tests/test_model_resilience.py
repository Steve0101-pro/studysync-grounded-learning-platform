from types import SimpleNamespace

import httpx
import pytest
from groq import APIConnectionError as GroqConnectionError
from groq import APIStatusError as GroqStatusError
from openai import APIConnectionError, APIStatusError
from pydantic import SecretStr

from app.config import settings
from app.services import gemini_vlm_service, model_runtime, stt, study_orchestration, vlm_service
from app.services.model_runtime import provider_timeout


def test_provider_timeout_uses_configured_bounds():
    timeout = provider_timeout()
    transcription_timeout = provider_timeout(
        read_timeout_seconds=settings.TRANSCRIPTION_TIMEOUT_SECONDS
    )

    assert timeout.connect == settings.MODEL_CONNECT_TIMEOUT_SECONDS
    assert timeout.read == settings.MODEL_READ_TIMEOUT_SECONDS
    assert timeout.write == settings.MODEL_WRITE_TIMEOUT_SECONDS
    assert timeout.pool == settings.MODEL_POOL_TIMEOUT_SECONDS
    assert transcription_timeout.read == settings.TRANSCRIPTION_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_provider_clients_are_reused_and_closed(monkeypatch):
    clients = []

    class _Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            clients.append(self)

        async def close(self):
            self.closed = True

    await model_runtime.close_provider_clients()
    monkeypatch.setattr(model_runtime, "AsyncOpenAI", _Client)

    first = model_runtime.get_provider_client("test-key", "https://provider.test/v1")
    second = model_runtime.get_provider_client("test-key", "https://provider.test/v1")

    assert first is second
    assert first.kwargs["timeout"].connect == settings.MODEL_CONNECT_TIMEOUT_SECONDS
    await model_runtime.close_provider_clients()
    assert first.closed is True


def test_tutor_provider_order_includes_nvidia_model_fallback(monkeypatch):
    monkeypatch.setattr(settings, "NVIDIA_API_KEY", SecretStr("nvidia-test"))
    monkeypatch.setattr(settings, "STUDY_TUTOR_NVIDIA_MODEL", "study-tutor-first")
    monkeypatch.setattr(settings, "NVIDIA_PRIMARY_MODEL", "reasoning-primary")
    monkeypatch.setattr(settings, "NVIDIA_FALLBACK_MODEL", "reasoning-fallback")
    monkeypatch.setattr(settings, "NVIDIA_BASE_URL", "https://nvidia.test/v1")
    monkeypatch.setattr(settings, "GROQ_API_KEY", SecretStr("groq-test"))
    monkeypatch.setattr(settings, "GROQ_FALLBACK_MODEL", "groq-chat-fallback")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", SecretStr(""))

    providers = study_orchestration._model_providers()

    assert [provider[0] for provider in providers] == [
        "NVIDIA",
        "NVIDIA",
        "NVIDIA",
        "Groq",
    ]
    assert [provider[3] for provider in providers] == [
        "study-tutor-first",
        "reasoning-primary",
        "reasoning-fallback",
        "groq-chat-fallback",
    ]


def test_tutor_fallback_includes_configured_gemini_models(monkeypatch):
    monkeypatch.setattr(settings, "NVIDIA_API_KEY", SecretStr(""))
    monkeypatch.setattr(settings, "GROQ_API_KEY", SecretStr(""))
    monkeypatch.setattr(settings, "OPENAI_API_KEY", SecretStr(""))
    monkeypatch.setattr(settings, "GEMINI_API_KEY", SecretStr("gemini-test"))
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-primary")
    monkeypatch.setattr(settings, "GEMINI_FALLBACK_MODEL", "gemini-fallback")

    assert study_orchestration._tutor_providers() == [
        ("Gemini", "", "", "gemini-primary"),
        ("Gemini", "", "", "gemini-fallback"),
    ]


@pytest.mark.asyncio
async def test_transcription_uses_fallback_after_connection_failure(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_MODEL", "whisper-primary")
    monkeypatch.setattr(settings, "GROQ_STT_FALLBACK_MODEL", "whisper-fallback")
    requested_models = []

    class _Transcriptions:
        async def create(self, **kwargs):
            requested_models.append(kwargs["model"])
            if kwargs["model"] == "whisper-primary":
                raise GroqConnectionError(
                    request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio")
                )
            return SimpleNamespace(text="recognized words")

    monkeypatch.setattr(
        stt,
        "client",
        SimpleNamespace(audio=SimpleNamespace(transcriptions=_Transcriptions())),
    )

    result = await stt.transcribe_audio(b"audio", "question.wav", "audio/wav")

    assert result == "recognized words"
    assert requested_models == ["whisper-primary", "whisper-fallback"]


@pytest.mark.asyncio
async def test_transcription_skips_broken_provider_and_uses_gemini_fallback(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", SecretStr("groq-test"))
    monkeypatch.setattr(settings, "GROQ_MODEL", "whisper-primary")
    monkeypatch.setattr(settings, "GROQ_STT_FALLBACK_MODEL", "whisper-fallback")
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-primary")
    monkeypatch.setattr(settings, "GEMINI_FALLBACK_MODEL", "gemini-fallback")
    requested_models = []
    gemini_requests = []

    class _Transcriptions:
        async def create(self, **kwargs):
            requested_models.append(kwargs["model"])
            response = httpx.Response(
                401,
                request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio"),
            )
            raise GroqStatusError("Invalid API key", response=response, body={})

    class _GeminiModels:
        async def generate_content(self, *, model, **_kwargs):
            gemini_requests.append(model)
            return SimpleNamespace(
                text='{"segments":[{"start":0,"text":"recognized by fallback"}]}'
            )

    monkeypatch.setattr(
        stt,
        "client",
        SimpleNamespace(
            audio=SimpleNamespace(transcriptions=_Transcriptions()),
            aio=SimpleNamespace(models=_GeminiModels()),
        ),
    )

    result = await stt.transcribe_audio(b"audio", "question.wav", "audio/wav")

    assert result == "recognized by fallback"
    assert requested_models == ["whisper-primary"]
    assert gemini_requests == ["gemini-primary"]


@pytest.mark.asyncio
async def test_transcription_falls_back_after_openai_sdk_authentication_error(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", SecretStr("groq-test"))
    monkeypatch.setattr(settings, "GROQ_MODEL", "whisper-primary")
    monkeypatch.setattr(settings, "GROQ_STT_FALLBACK_MODEL", "whisper-fallback")
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-primary")
    monkeypatch.setattr(settings, "GEMINI_FALLBACK_MODEL", "gemini-fallback")
    gemini_requests = []

    class _Transcriptions:
        async def create(self, **_kwargs):
            response = httpx.Response(
                401,
                request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio"),
            )
            raise APIStatusError("Invalid API key", response=response, body={})

    class _GeminiModels:
        async def generate_content(self, *, model, **_kwargs):
            gemini_requests.append(model)
            return SimpleNamespace(
                text='{"segments":[{"start":0,"text":"recognized by Gemini"}]}'
            )

    monkeypatch.setattr(
        stt,
        "client",
        SimpleNamespace(
            audio=SimpleNamespace(transcriptions=_Transcriptions()),
            aio=SimpleNamespace(models=_GeminiModels()),
        ),
    )

    result = await stt.transcribe_audio(b"audio", "question.wav", "audio/wav")

    assert result == "recognized by Gemini"
    assert gemini_requests == ["gemini-primary"]


def test_vision_uses_fallback_after_connection_failure(monkeypatch):
    monkeypatch.setattr(settings, "NVIDIA_VISION_MODEL", "vision-primary")
    monkeypatch.setattr(settings, "NVIDIA_VISION_FALLBACK_MODEL", "vision-fallback")
    requested_models = []

    class _Completions:
        def create(self, *, model, **_kwargs):
            requested_models.append(model)
            if model == "vision-primary":
                raise APIConnectionError(
                    request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1")
                )
            return SimpleNamespace(choices=["vision answer"])

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    result = vlm_service._request_vision_completion(client, [], temperature=0)

    assert result.choices == ["vision answer"]
    assert requested_models == ["vision-primary", "vision-fallback"]


@pytest.mark.asyncio
async def test_multimodal_uses_vision_fallback_for_missing_primary_model(monkeypatch):
    monkeypatch.setattr(settings, "NVIDIA_API_KEY", SecretStr("nvidia-test"))
    monkeypatch.setattr(settings, "NVIDIA_PRIMARY_MODEL", "omni-primary")
    monkeypatch.setattr(settings, "NVIDIA_VISION_MODEL", "vision-fallback")
    monkeypatch.setattr(gemini_vlm_service.portkeys_client, "gateway_enabled", False)
    requested_models = []

    class _Completions:
        async def create(self, *, model, **_kwargs):
            requested_models.append(model)
            if model == "omni-primary":
                response = httpx.Response(
                    404,
                    request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1"),
                )
                raise APIStatusError("Model not found", response=response, body={})
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="visual answer"))]
            )

    class _Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(
        gemini_vlm_service,
        "get_provider_client",
        lambda *_args: _Client(),
    )

    result = await gemini_vlm_service.generate_response(
        transcript="What is in the picture?",
        image_bytes=b"synthetic-image",
        image_content_type="image/png",
        conversation_context=[],
    )

    assert result == "visual answer"
    assert requested_models == ["omni-primary", "vision-fallback"]
