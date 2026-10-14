import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.documents import router as documents_router
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.api.multimodal_router import _has_valid_signature, _normalize_cache_text
from app.api.redis_service import create_cache_key
from app.rate_limit import setup_rate_limiting


def test_cache_key_changes_when_audio_content_changes():
    first = create_cache_key("user-1", "conversation-1", "question.wav", "a" * 64)
    second = create_cache_key("user-1", "conversation-1", "question.wav", "b" * 64)

    assert first != second
    assert first.startswith("multimodal:user-1:")


def test_audio_signature_must_match_declared_type():
    wav_header = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 32

    assert _has_valid_signature(wav_header, "audio/wav")
    assert not _has_valid_signature(b"not-audio", "audio/wav")


def test_image_signature_must_match_declared_type():
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

    assert _has_valid_signature(png_header, "image/png")
    assert not _has_valid_signature(b"not-an-image", "image/png")


def test_prompt_cache_text_ignores_case_and_whitespace():
    assert _normalize_cache_text("  Read   the TOTAL ") == "read the total"


@pytest.mark.asyncio
async def test_document_upload_rejects_content_that_does_not_match_image_type():
    app = FastAPI()
    setup_rate_limiting(app)
    app.include_router(documents_router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=uuid.uuid4())

    async def override_db():
        yield SimpleNamespace()

    app.dependency_overrides[get_db] = override_db
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/documents/",
                files={"file": ("spoofed.png", b"not a PNG", "image/png")},
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "Document content does not match its declared type"
    finally:
        app.dependency_overrides.clear()
