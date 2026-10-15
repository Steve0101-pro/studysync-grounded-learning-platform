import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.learning.tutor import router as tutor_router
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.rate_limit import setup_rate_limiting
from app.services import study_orchestration
from app.services.learning.citations import referenced_citations
from app.services.media_validation import (
    ALLOWED_TRANSCRIPTION_TYPES,
    has_valid_media_signature,
)


class _RetrievalGraph:
    def __init__(self, state):
        self.state = state
        self.queries = []

    async def ainvoke(self, state):
        self.queries.append(state["query"])
        return self.state


class _FakeResult:
    def __init__(self, values=()):
        self.values = list(values)

    def scalars(self):
        return self

    def all(self):
        return self.values


class _FakeDatabase:
    def __init__(self, saved, execute_results=()):
        self.saved = saved
        self.added = []
        self.execute_results = list(execute_results)

    async def execute(self, _statement):
        if self.execute_results:
            return self.execute_results.pop(0)
        return _FakeResult()

    def add(self, item):
        self.added.append(item)

    async def flush(self):
        for item in self.added:
            if item.__class__.__name__ == "Conversation" and item.id is None:
                item.id = uuid.uuid4()

    async def commit(self):
        self.saved.extend(self.added)
        self.added.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


def test_referenced_citations_only_includes_answer_markers_in_original_order():
    citations = [
        {"filename": "one.pdf"},
        {"filename": "two.pdf"},
        {"filename": "three.pdf"},
    ]

    assert referenced_citations("Claim [3]. Another claim [1].", citations) == [
        (1, citations[0]),
        (3, citations[2]),
    ]
    assert referenced_citations("No course citations.", citations) == []


def test_stream_citation_payload_uses_full_parent_context():
    citations = study_orchestration._citation_payload(
        [
            {
                "source_id": uuid.uuid4(),
                "filename": "course.pdf",
                "location_type": "page",
                "location_number": 4,
                "content": "retrieved child fragment",
                "parent_content": "Contact student@example.com for the full paragraph.",
            }
        ]
    )

    assert citations[0]["content"] == "Contact [EMAIL] for the full paragraph."


@pytest.mark.parametrize(
    ("mastery_estimate", "expected_instruction"),
    [
        (None, "hints-first"),
        (0.25, "hints-first"),
        (0.6, "moderate scaffolding"),
        (0.9, "strong recent practice"),
    ],
)
def test_adaptive_tutor_selects_scaffolding_for_practice_signal(
    mastery_estimate,
    expected_instruction,
):
    messages = study_orchestration._stream_messages(
        "Explain the topic.",
        [
            {
                "filename": "course.pdf",
                "location_type": "page",
                "location_number": 1,
                "content": "Evidence for the topic.",
            }
        ],
        [],
        None,
        "adaptive",
        mastery_estimate,
    )

    assert expected_instruction in messages[0]["content"]


@pytest.mark.asyncio
async def test_streaming_tutor_retrieves_with_followup_and_visual_context(monkeypatch):
    passage = {
        "source_id": str(uuid.uuid4()),
        "filename": "biology.pdf",
        "location_type": "page",
        "location_number": 4,
        "content": "Mitochondria produce ATP during respiration.",
    }
    graph = _RetrievalGraph(
        {"passages": [passage], "retrieval_backend": "bm25+cosine"}
    )
    monkeypatch.setattr(study_orchestration, "_study_retrieval_graph", graph)
    monkeypatch.setattr(study_orchestration, "_tutor_providers", lambda: [])

    events = [
        event
        async for event in study_orchestration.stream_study_tutor(
            "Why?",
            lambda _query: None,
            conversation_history=[
                {"role": "user", "content": "Explain mitochondria."},
                {"role": "assistant", "content": "They produce ATP [1]."},
            ],
            visual_context="The uploaded diagram labels the mitochondrion.",
        )
    ]

    assert graph.queries == [
        "Explain mitochondria.\nWhy?\nThe uploaded diagram labels the mitochondrion."
    ]
    assert events[0]["type"] == "context"
    assert events[0]["retrieval_backend"] == "bm25+cosine"
    assert events[-2]["type"] == "replace"
    assert "[1]" in events[-2]["text"]
    assert events[-1] == {
        "type": "complete",
        "generation_provider": "source_excerpt",
        "grounded": True,
        "tutoring_mode": "adaptive",
        "mastery_band": "cold_start",
    }
    assert "hints-first" in study_orchestration._stream_messages(
        "Why?",
        [passage],
        [],
        None,
        "adaptive",
        None,
    )[0]["content"]


@pytest.mark.asyncio
async def test_streaming_tutor_uses_fallback_provider_and_streams_tokens(monkeypatch):
    passage = {
        "source_id": str(uuid.uuid4()),
        "filename": "biology.pdf",
        "location_type": "page",
        "location_number": 4,
        "content": "Mitochondria produce ATP.",
    }
    monkeypatch.setattr(
        study_orchestration,
        "_study_retrieval_graph",
        _RetrievalGraph({"passages": [passage], "retrieval_backend": "bm25"}),
    )
    monkeypatch.setattr(
        study_orchestration,
        "_tutor_providers",
        lambda: [("Groq", "https://example.test/v1", "test-key", "test-model")],
    )
    received_messages = []

    class _Chunk:
        def __init__(self, content):
            self.choices = [SimpleNamespace(delta=SimpleNamespace(content=content))]

    class _Stream:
        def __aiter__(self):
            self._chunks = iter(
                [
                    SimpleNamespace(choices=[]),
                    _Chunk("ATP is "),
                    _Chunk("produced [1]."),
                ]
            )
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class _Completions:
        async def create(self, **kwargs):
            received_messages.extend(kwargs["messages"])
            assert kwargs["stream"] is True
            return _Stream()

    class _Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(
        study_orchestration,
        "get_provider_client",
        lambda *_args: _Client(),
    )
    events = [
        event
        async for event in study_orchestration.stream_study_tutor(
            "Explain ATP.",
            lambda _query: None,
            conversation_history=[{"role": "user", "content": "Earlier question"}],
        )
    ]

    assert [event["text"] for event in events if event["type"] == "token"] == [
        "ATP is ",
        "produced [1].",
    ]
    assert events[-1] == {
        "type": "complete",
        "generation_provider": "groq",
        "grounded": True,
        "tutoring_mode": "adaptive",
        "mastery_band": "cold_start",
    }
    assert any(message["content"] == "Earlier question" for message in received_messages)
    assert "hints-first" in received_messages[0]["content"]


@pytest.mark.asyncio
async def test_invalid_provider_stream_is_not_shown_before_gemini_fallback(monkeypatch):
    passage = {
        "source_id": str(uuid.uuid4()),
        "filename": "biology.pdf",
        "location_type": "page",
        "location_number": 4,
        "content": "Mitochondria produce ATP.",
    }
    monkeypatch.setattr(
        study_orchestration,
        "_study_retrieval_graph",
        _RetrievalGraph({"passages": [passage], "retrieval_backend": "bm25"}),
    )
    monkeypatch.setattr(
        study_orchestration,
        "_tutor_providers",
        lambda: [
            ("NVIDIA", "https://example.test/v1", "test-key", "nvidia-model"),
            ("Gemini", "", "", "gemini-primary"),
        ],
    )

    class _Chunk:
        def __init__(self, content):
            self.choices = [SimpleNamespace(delta=SimpleNamespace(content=content))]

    class _OpenAIStream:
        def __aiter__(self):
            self._chunks = iter([_Chunk("Uncited answer that must not appear.")])
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class _OpenAICompletions:
        async def create(self, **_kwargs):
            return _OpenAIStream()

    class _GeminiChunk:
        text = "Mitochondria produce ATP [1]; contact student@example.com."

    class _GeminiStream:
        def __aiter__(self):
            self.sent = False
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return _GeminiChunk()

    class _GeminiModels:
        async def generate_content_stream(self, **_kwargs):
            return _GeminiStream()

    monkeypatch.setattr(
        study_orchestration,
        "get_provider_client",
        lambda *_args: SimpleNamespace(
            chat=SimpleNamespace(
                completions=_OpenAICompletions(),
            )
        ),
    )
    monkeypatch.setattr(
        study_orchestration,
        "client",
        SimpleNamespace(aio=SimpleNamespace(models=_GeminiModels())),
    )

    events = [
        event
        async for event in study_orchestration.stream_study_tutor(
            "Explain ATP.",
            lambda _query: None,
        )
    ]
    displayed_text = "".join(
        event["text"] for event in events if event["type"] == "token"
    )

    assert "Uncited answer" not in displayed_text
    assert displayed_text == "Mitochondria produce ATP [1]; contact [EMAIL]."
    assert "student@example.com" not in displayed_text
    assert events[-1]["generation_provider"] == "gemini"


def test_shared_media_validation_checks_declared_content_type():
    assert has_valid_media_signature(b"\x89PNG\r\n\x1a\npayload", "image/png")
    assert not has_valid_media_signature(b"not-an-image", "image/png")
    assert not has_valid_media_signature(b"\x89PNG\r\n\x1a\npayload", "image/jpeg")
    assert "video/mp4" in ALLOWED_TRANSCRIPTION_TYPES
    assert "video/webm" in ALLOWED_TRANSCRIPTION_TYPES
    assert has_valid_media_signature(b"\x00\x00\x00\x18ftypmp42payload", "video/mp4")
    assert has_valid_media_signature(b"\x1a\x45\xdf\xa3payload", "video/webm")


@pytest.mark.asyncio
async def test_study_tutor_transcription_endpoint_returns_validated_transcript(monkeypatch):
    from app.api import study as study_api

    user_id = uuid.uuid4()

    async def fake_transcribe(audio_bytes, filename, content_type):
        assert audio_bytes == b"RIFF\x04\x00\x00\x00WAVEaudio"
        assert filename == "question.wav"
        assert content_type == "audio/wav"
        return "  Explain cellular respiration.  "

    monkeypatch.setattr(study_api, "transcribe_audio", fake_transcribe)
    app = FastAPI()
    setup_rate_limiting(app)
    app.include_router(tutor_router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/study/tutor/transcribe",
                files={
                    "audio": (
                        "question.wav",
                        b"RIFF\x04\x00\x00\x00WAVEaudio",
                        "audio/wav",
                    )
                },
            )

        assert response.status_code == 200
        assert response.json() == {"transcript": "Explain cellular respiration."}
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_study_tutor_transcription_endpoint_explains_provider_failure(monkeypatch):
    from app.api import study as study_api

    async def failed_transcription(*_args):
        raise RuntimeError("provider failure")

    monkeypatch.setattr(study_api, "transcribe_audio", failed_transcription)
    app = FastAPI()
    setup_rate_limiting(app)
    app.include_router(tutor_router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=uuid.uuid4())
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/study/tutor/transcribe",
                files={
                    "audio": (
                        "question.wav",
                        b"RIFF\x04\x00\x00\x00WAVEaudio",
                        "audio/wav",
                    )
                },
            )

        assert response.status_code == 503
        assert "Check the configured speech-provider credentials" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("history_topic", "has_relevant_practice"),
    [("mitochondria", True), ("calculus", False)],
)
@pytest.mark.asyncio
async def test_study_tutor_stream_endpoint_persists_both_turns(
    monkeypatch,
    history_topic,
    has_relevant_practice,
):
    from app.api import study as study_api

    user_id = uuid.uuid4()
    saved_messages = []
    request_db = _FakeDatabase(
        saved_messages,
        [
            _FakeResult(),
            _FakeResult(
                [(history_topic, True, datetime.now(timezone.utc))]
            ),
        ],
    )
    stream_arguments = {}
    passage = {
        "source_id": uuid.uuid4(),
        "filename": "biology.pdf",
        "location_type": "page",
        "location_number": 2,
        "content": "Mitochondria produce ATP.",
    }

    async def fake_retrieve(_db, _user_id, _query, limit):
        assert limit == 5
        return [passage], "bm25"

    async def fake_stream(query, retriever, **kwargs):
        stream_arguments.update(kwargs)
        passages, backend = await retriever(query)
        yield {
            "type": "context",
            "passages": [
                {
                    "source_id": str(passages[0]["source_id"]),
                    "filename": passages[0]["filename"],
                    "location_type": passages[0]["location_type"],
                    "location_number": passages[0]["location_number"],
                    "content": passages[0]["content"],
                }
            ],
            "retrieval_backend": backend,
        }
        yield {"type": "token", "text": "Mitochondria produce ATP [1]."}
        yield {
            "type": "complete",
            "generation_provider": "groq",
            "grounded": True,
            "tutoring_mode": "hints_first",
            "mastery_band": "cold_start",
        }

    monkeypatch.setattr(study_api, "_hybrid_retrieve_study_passages", fake_retrieve)
    monkeypatch.setattr(study_api, "stream_study_tutor", fake_stream)
    monkeypatch.setattr(study_api, "SessionLocal", lambda: _FakeDatabase(saved_messages))

    async def override_db():
        yield request_db

    app = FastAPI()
    setup_rate_limiting(app)
    app.include_router(tutor_router)
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/study/tutor/stream",
                data={
                    "query": "Explain mitochondria.",
                    "tutoring_mode": "hints_first",
                },
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "event: token" in response.text
        assert "Mitochondria produce ATP [1]." in response.text
        assert "event: done" in response.text
        assert '"tutoring_mode": "hints_first"' in response.text
        assert '"mastery_band": "cold_start"' in response.text
        user_messages = [
            message for message in saved_messages
            if message.__class__.__name__ == "Message" and message.role == "user"
        ]
        assistant_messages = [
            message for message in saved_messages
            if message.__class__.__name__ == "Message" and message.role == "assistant"
        ]
        assert len(user_messages) == len(assistant_messages) == 1
        assert user_messages[0].metadata_json["workflow"] == "study_tutor"
        assert assistant_messages[0].metadata_json["citations"][0]["filename"] == "biology.pdf"
        assert assistant_messages[0].content == "Mitochondria produce ATP [1]."
        assert (stream_arguments["mastery_estimate"] is not None) is has_relevant_practice
        assert stream_arguments["tutoring_mode"] == "hints_first"
    finally:
        app.dependency_overrides.clear()
