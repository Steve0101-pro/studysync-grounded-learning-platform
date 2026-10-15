import base64
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.study import router as study_router
from app.api.auth import router as auth_router
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models import StudyMockExam, StudyQuizQuestion
from app.rate_limit import setup_rate_limiting
from app.services import study_orchestration
from app.api import study as study_api
from app.services.study_materials import rank_content
from app.services.learning.mastery import estimate_mastery, recommended_review_days
from app.services.learning.analytics import practice_accuracy_chart, wilson_interval
from app.services.learning.retrieval import bm25_scores, fuse_bm25_and_cosine
from app.services.study_orchestration import _has_valid_citations
from app.api.learning.courses import router as courses_router
from app.api.learning import (
    assessments_router,
    dashboard_router,
    sources_router,
    tutor_router,
)

app = FastAPI()
setup_rate_limiting(app)
app.include_router(study_router)
app.include_router(courses_router)
app.include_router(auth_router)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def scalars(self):
        return self


class _StudyDatabase:
    def __init__(
        self,
        rows,
        source_count=0,
        mastery_events=None,
        used_passages=None,
        source=None,
    ):
        self.rows = rows
        self.question = None
        self.source_count = source_count
        self.mastery_events = mastery_events or []
        self.used_passages = used_passages or []
        self.source = source

    async def execute(self, statement):
        query = str(statement)
        if "answered_at" in query and "GROUP BY" not in query.upper():
            return _Result(self.mastery_events)
        if "study_quiz_questions" in query and "source_id" in query:
            return _Result(self.used_passages)
        return _Result(self.rows)

    def add(self, question):
        self.question = question
        self.used_passages.append(
            (
                question.source_id,
                question.location_type,
                question.location_number,
            )
        )

    async def commit(self):
        return None

    async def refresh(self, question):
        if question.id is None:
            question.id = uuid.uuid4()

    async def scalar(self, statement):
        if "study_sources" in str(statement):
            return self.source if self.source is not None else self.source_count
        return self.source_count or self.question


def _passage(source_id: uuid.UUID, filename: str, location_number: int, content: str):
    source = SimpleNamespace(id=source_id, filename=filename, user_id=uuid.uuid4())
    unit = SimpleNamespace(
        id=uuid.uuid4(),
        source_id=source_id,
        location_type="page",
        location_number=location_number,
        chunk_index=0,
        content=content,
    )
    return unit, source


@pytest.mark.asyncio
async def test_study_tutor_returns_only_cited_source_text_and_flags_gaps(monkeypatch):
    user_id = uuid.uuid4()
    rows = [
        _passage(uuid.uuid4(), "biology.pdf", 2, "Mitochondria produce ATP during respiration."),
        _passage(uuid.uuid4(), "biology.pdf", 3, "The cell membrane controls transport."),
    ]
    database = _StudyDatabase(rows)
    monkeypatch.setattr(
        study_api,
        "search_study_passages",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("Qdrant unavailable")),
    )
    monkeypatch.setattr(study_orchestration, "_model_providers", lambda: [])

    async def override_db():
        yield database

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            grounded = await client.get("/study/tutor", params={"q": "mitochondria ATP"})
            uncovered = await client.get("/study/tutor", params={"q": "quantum mechanics"})

        assert grounded.status_code == 200
        grounded_payload = grounded.json()
        assert grounded_payload["grounded"] is True
        assert grounded_payload["retrieval_backend"] == "bm25"
        assert grounded_payload["generation_provider"] == "source_excerpt"
        assert grounded_payload["citations"][0]["filename"] == "biology.pdf"
        assert grounded_payload["citations"][0]["location_number"] == 2
        assert grounded_payload["citations"][0]["content"] in grounded_payload["answer"]
        assert uncovered.status_code == 200
        assert uncovered.json()["grounded"] is False
        assert uncovered.json()["citations"] == []
        assert uncovered.json()["generation_provider"] == "none"
        assert "couldn't find" in uncovered.json()["answer"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_study_tutor_uses_fallback_model_after_primary_failure(monkeypatch):
    async def retrieve(_query):
        return (
            [
                {
                    "source_id": str(uuid.uuid4()),
                    "filename": "biology.pdf",
                    "location_type": "page",
                    "location_number": 3,
                    "content": "Mitochondria produce ATP.",
                }
            ],
            "qdrant",
        )

    providers = [
        ("NVIDIA", "https://example.test/v1", "primary-test-key", "primary-model"),
        ("Groq", "https://example.test/v1", "fallback-test-key", "fallback-model"),
    ]
    monkeypatch.setattr(study_orchestration, "_model_providers", lambda: providers)
    calls = []

    class _Completion:
        async def create(self, **_kwargs):
            calls.append(self.provider)
            if self.provider == "NVIDIA":
                raise RuntimeError("primary unavailable")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Mitochondria make ATP [1]."))]
            )

    class _Client:
        def __init__(self, *, api_key, **_kwargs):
            self.provider = "NVIDIA" if api_key == "primary-test-key" else "Groq"
            completion = _Completion()
            completion.provider = self.provider
            self.chat = SimpleNamespace(completions=completion)

    monkeypatch.setattr(study_orchestration, "AsyncOpenAI", _Client)
    result = await study_orchestration.run_study_tutor("mitochondria", retrieve)

    assert calls == ["NVIDIA", "Groq"]
    assert result["generation_provider"] == "groq"
    assert "ATP [1]" in result["answer"]


@pytest.mark.asyncio
async def test_study_quiz_hides_answer_records_response_and_blocks_second_attempt():
    topic = "mitochondria ATP"
    user_id = uuid.uuid4()
    rows = [
        _passage(uuid.uuid4(), "biology.pdf", 2, "Mitochondria produce ATP during respiration."),
        _passage(uuid.uuid4(), "biology.pdf", 5, "ATP is synthesized in the mitochondrial membrane."),
        _passage(uuid.uuid4(), "biology.pdf", 3, "The cell membrane controls transport."),
        _passage(uuid.uuid4(), "biology.pdf", 4, "Ribosomes assemble proteins."),
    ]
    rows.sort(key=lambda row: -rank_content(topic, row[0].content))
    database = _StudyDatabase(rows)

    async def override_db():
        yield database

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            created = await client.post("/study/quizzes", json={"topic": topic})
            assert created.status_code == 201
            payload = created.json()
            assert len(payload["options"]) == 3
            assert "answer_index" not in payload
            assert all("location_number" in option for option in payload["options"])

            correct_index = database.question.answer_index
            wrong_index = (correct_index + 1) % len(payload["options"])
            answered = await client.post(
                f"/study/quizzes/{payload['id']}/answer",
                json={"selected_index": wrong_index},
            )
            assert answered.status_code == 200
            assert answered.json()["correct"] is False
            assert answered.json()["answered_index"] == wrong_index
            assert answered.json()["explanation"] == payload["options"][correct_index]

            replay = await client.post(
                f"/study/quizzes/{payload['id']}/answer",
                json={"selected_index": correct_index},
            )
            assert replay.status_code == 409

            first_source = database.question.source_id
            second = await client.post("/study/quizzes", json={"topic": topic})
            assert second.status_code == 201
            assert database.question.source_id != first_source

            exhausted = await client.post("/study/quizzes", json={"topic": topic})
            assert exhausted.status_code == 409
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_study_quiz_source_and_location_scope_applies_to_answer_and_choices():
    topic = "mitochondria ATP"
    user_id = uuid.uuid4()
    selected_source_id = uuid.uuid4()
    other_source_id = uuid.uuid4()
    rows = [
        _passage(selected_source_id, "biology.pdf", 2, "Mitochondria produce ATP during respiration."),
        _passage(selected_source_id, "biology.pdf", 4, "ATP is synthesized in the mitochondrial membrane."),
        _passage(selected_source_id, "biology.pdf", 8, "ATP transfers energy within a mitochondrion."),
        _passage(other_source_id, "other.pdf", 3, "Mitochondria produce ATP in other course material."),
    ]
    rows.sort(key=lambda row: -rank_content(topic, row[0].content))
    database = _StudyDatabase(rows)

    async def override_db():
        yield database

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            created = await client.post(
                "/study/quizzes",
                json={
                    "topic": topic,
                    "source_id": str(selected_source_id),
                    "location_type": "page",
                    "location_start": 2,
                    "location_end": 4,
                },
            )
            invalid_range = await client.post(
                "/study/quizzes",
                json={
                    "topic": topic,
                    "source_id": str(selected_source_id),
                    "location_type": "page",
                    "location_start": 4,
                    "location_end": 2,
                },
            )

        assert created.status_code == 201
        assert database.question.source_id == selected_source_id
        assert database.question.location_number in {2, 4}
        assert all(
            option["source_id"] == str(selected_source_id)
            and option["location_type"] == "page"
            and 2 <= option["location_number"] <= 4
            for option in created.json()["options"]
        )
        assert invalid_range.status_code == 422
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_multiple_choice_quiz_hides_key_and_returns_source_citation(monkeypatch):
    user_id = uuid.uuid4()
    rows = [
        _passage(
            uuid.uuid4(),
            "biology.pdf",
            2,
            "Mitochondria produce ATP during cellular respiration.",
        )
    ]
    database = _StudyDatabase(rows)

    async def generate_verified_assessment(**_kwargs):
        return (
            {
                "question": "What do mitochondria produce during respiration?",
                "options": ["Glucose", "Oxygen", "ATP", "Protein"],
                "answer_index": 2,
                "explanation": "The passage states that respiration produces ATP.",
            },
            "test+verified",
        )

    monkeypatch.setattr(
        study_api,
        "generate_verified_assessment",
        generate_verified_assessment,
    )

    async def override_db():
        yield database

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            created = await client.post(
                "/study/quizzes",
                json={
                    "topic": "mitochondria ATP",
                    "assessment_type": "multiple_choice",
                    "difficulty": "easy",
                },
            )
            assert created.status_code == 201
            public_question = created.json()
            assert public_question["assessment_type"] == "multiple_choice"
            assert "answer_index" not in public_question
            assert len(public_question["options"]) == 4

            answered = await client.post(
                f"/study/quizzes/{public_question['id']}/answer",
                json={"selected_index": 2},
            )

        assert answered.status_code == 200
        assert answered.json()["correct"] is True
        assert answered.json()["explanation"] == (
            "The passage states that respiration produces ATP."
        )
        assert answered.json()["citation"]["filename"] == "biology.pdf"
        assert database.question.is_correct is True
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("assessment_type", "generated", "answer_payload"),
    [
        (
            "short_answer",
            {
                "question": "What does the inner membrane support?",
                "answer": "It supports ATP production.",
                "rubric": ["The membrane supports ATP production."],
                "explanation": "The passage links the inner membrane to ATP production.",
            },
            {"answer_text": "It is where ATP is produced."},
        ),
        (
            "numerical",
            {
                "question": "What distance is stated in the source?",
                "answer_value": 4.0,
                "unit": "m",
                "tolerance": 0.25,
                "explanation": "The passage gives the value as four metres.",
            },
            {"numerical_value": 4.1, "unit": "m"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_short_and_numerical_quizzes_are_graded_and_cited(
    assessment_type,
    generated,
    answer_payload,
    monkeypatch,
):
    user_id = uuid.uuid4()
    database = _StudyDatabase(
        [
            _passage(
                uuid.uuid4(),
                "lesson.pdf",
                7,
                "The inner membrane supports ATP production; the measured distance is four metres.",
            )
        ]
    )

    async def generate_verified_assessment(**_kwargs):
        return ({**generated, "question": generated["question"]}, "test+verified")

    async def grade_short_answer(**_kwargs):
        return True, "Your answer identifies the source's key point."

    monkeypatch.setattr(
        study_api,
        "generate_verified_assessment",
        generate_verified_assessment,
    )
    monkeypatch.setattr(study_api, "grade_short_answer", grade_short_answer)

    async def override_db():
        yield database

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            created = await client.post(
                "/study/quizzes",
                json={
                    "topic": "inner membrane distance",
                    "assessment_type": assessment_type,
                },
            )
            assert created.status_code == 201
            assert "answer" not in created.json()
            answered = await client.post(
                f"/study/quizzes/{created.json()['id']}/answer",
                json=answer_payload,
            )

        assert answered.status_code == 200
        assert answered.json()["correct"] is True
        assert answered.json()["citation"]["location_number"] == 7
        assert database.question.is_correct is True
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_timed_mock_exam_generates_mix_and_returns_final_cited_report(monkeypatch):
    user_id = uuid.uuid4()
    rows = [
        _passage(uuid.uuid4(), "lesson.pdf", 2, "Mitochondria produce ATP."),
        _passage(uuid.uuid4(), "lesson.pdf", 3, "The inner membrane supports ATP production."),
        _passage(uuid.uuid4(), "lesson.pdf", 4, "A measured distance is four metres."),
    ]

    class _MockExamDatabase(_StudyDatabase):
        def __init__(self, exam_rows):
            super().__init__(exam_rows)
            self.exam = None
            self.exam_questions = []

        async def execute(self, statement):
            query = str(statement).lower()
            if "study_quiz_questions.mock_exam_id" in query:
                return _Result(self.exam_questions)
            if "study_mock_exams" in query:
                return _Result([self.exam] if self.exam else [])
            return await super().execute(statement)

        async def scalar(self, statement):
            if "study_mock_exams" in str(statement):
                return self.exam
            return await super().scalar(statement)

        def add(self, item):
            if isinstance(item, StudyMockExam):
                self.exam = item
                return
            if isinstance(item, StudyQuizQuestion):
                self.exam_questions.append(item)
                return
            super().add(item)

        def add_all(self, items):
            for item in items:
                item.id = item.id or uuid.uuid4()
                self.exam_questions.append(item)

        async def flush(self):
            self.exam.id = self.exam.id or uuid.uuid4()
            self.exam.started_at = datetime.now(timezone.utc)

        async def refresh(self, item):
            if isinstance(item, StudyMockExam):
                return
            if item.id is None:
                item.id = uuid.uuid4()

    database = _MockExamDatabase(rows)

    async def generate_verified_assessment(*, assessment_type, **_kwargs):
        if assessment_type == "multiple_choice":
            return (
                {
                    "question": "Which molecule do mitochondria produce?",
                    "options": ["Water", "ATP", "Glucose", "Protein"],
                    "answer_index": 1,
                    "explanation": "The passage states mitochondria produce ATP.",
                },
                "test+verified",
            )
        if assessment_type == "short_answer":
            return (
                {
                    "question": "What does the inner membrane support?",
                    "answer": "It supports ATP production.",
                    "rubric": ["The membrane supports ATP production."],
                    "explanation": "The inner membrane supports ATP production.",
                },
                "test+verified",
            )
        return (
            {
                "question": "What distance is stated?",
                "answer_value": 4.0,
                "unit": "m",
                "tolerance": 0.1,
                "explanation": "The source states four metres.",
            },
            "test+verified",
        )

    async def grade_short_answer(**_kwargs):
        return True, "The response is supported."

    monkeypatch.setattr(
        study_api,
        "generate_verified_assessment",
        generate_verified_assessment,
    )
    monkeypatch.setattr(study_api, "grade_short_answer", grade_short_answer)

    async def override_db():
        yield database

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            created = await client.post(
                "/study/mock-exams",
                json={
                    "topic": "mitochondria ATP",
                    "question_count": 3,
                    "duration_minutes": 20,
                    "assessment_types": [
                        "multiple_choice",
                        "short_answer",
                        "numerical",
                    ],
                },
            )
            assert created.status_code == 201
            exam = created.json()
            assert len(exam["questions"]) == 3
            assert all("answer_index" not in item for item in exam["questions"])
            saved_exams = await client.get("/study/mock-exams")
            resumed = await client.get(f"/study/mock-exams/{exam['id']}")
            assert saved_exams.status_code == 200
            assert saved_exams.json()[0]["id"] == exam["id"]
            assert resumed.status_code == 200
            assert len(resumed.json()["questions"]) == 3
            assert all("assessment_data" not in item for item in resumed.json()["questions"])
            answers = [
                {
                    "question_id": exam["questions"][0]["id"],
                    "answer": {"selected_index": 1},
                },
                {
                    "question_id": exam["questions"][1]["id"],
                    "answer": {"answer_text": "The inner membrane supports ATP."},
                },
                {
                    "question_id": exam["questions"][2]["id"],
                    "answer": {"numerical_value": 4.0, "unit": "m"},
                },
            ]
            submitted = await client.post(
                f"/study/mock-exams/{exam['id']}/submit",
                json={"answers": answers},
            )
            restored_report = await client.get(f"/study/mock-exams/{exam['id']}")

        assert submitted.status_code == 200
        report = submitted.json()
        assert report["total_questions"] == 3
        assert report["answered_questions"] == 3
        assert report["correct_answers"] == 3
        assert report["score"] == 1.0
        assert all(item["citation"]["content"] for item in report["results"])
        assert database.exam.report_json["correct_answers"] == 3
        assert restored_report.status_code == 200
        assert restored_report.json()["correct_answers"] == 3
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_source_open_returns_short_lived_page_deep_link(monkeypatch):
    source = SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        filename="biology.pdf",
        content_type="application/pdf",
        storage_path="private/biology.pdf",
        page_count=12,
    )
    database = _StudyDatabase([], source=source)
    monkeypatch.setattr(
        study_api,
        "create_signed_url",
        lambda path, expiry: (
            f"https://storage.example.test/{path}?expires={expiry}"
        ),
    )

    async def override_db():
        yield database

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=source.user_id)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/study/sources/{source.id}/open",
                params={"location_type": "page", "location_number": 7},
            )

        assert response.status_code == 200
        assert response.json() == {
            "url": "https://storage.example.test/private/biology.pdf?expires=300#page=7",
            "expires_in_seconds": 300,
            "viewer_supported": True,
            "location_type": "page",
            "location_number": 7,
        }
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_powerpoint_slide_viewer_returns_owner_scoped_text_and_images(monkeypatch):
    source = SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        filename="course.pptx",
        content_type=study_api.PPTX_CONTENT_TYPE,
        storage_path="private/course.pptx",
        page_count=4,
    )
    database = _StudyDatabase([], source=source)
    monkeypatch.setattr(study_api, "download_file", lambda path: b"presentation")
    monkeypatch.setattr(
        study_api,
        "_pptx_slides",
        lambda file_bytes: [(3, "Gradient descent updates model parameters.", [b"raw-image"])],
    )
    monkeypatch.setattr(study_api, "_prepare_figure_image", lambda image: b"jpeg-preview")

    async def override_db():
        yield database

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=source.user_id)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(f"/study/sources/{source.id}/slides/3")

        assert response.status_code == 200
        payload = response.json()
        assert payload["filename"] == "course.pptx"
        assert payload["slide_number"] == 3
        assert payload["slide_count"] == 4
        assert payload["text"] == "Gradient descent updates model parameters."
        assert payload["images"] == [
            {
                "media_type": "image/jpeg",
                "data": base64.b64encode(b"jpeg-preview").decode("ascii"),
            }
        ]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_powerpoint_slide_viewer_hides_sources_from_other_users(monkeypatch):
    source = SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        filename="course.pptx",
        content_type=study_api.PPTX_CONTENT_TYPE,
        storage_path="private/course.pptx",
        page_count=4,
    )

    class _MissingSourceDatabase:
        async def scalar(self, _statement):
            return None

    async def override_db():
        yield _MissingSourceDatabase()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=uuid.uuid4())
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(f"/study/sources/{source.id}/slides/3")

        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_study_progress_reports_answer_accuracy_per_topic():
    user_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    mastery_events = [
        ("cell biology", True, now - timedelta(days=2)),
        ("cell biology", False, now - timedelta(days=1)),
        ("cell biology", True, now),
    ]
    database = _StudyDatabase(
        [
            ("cell biology", 4, 3, 2),
            ("genetics", 2, 0, 0),
        ],
        source_count=2,
        mastery_events=mastery_events,
    )

    async def override_db():
        yield database

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/study/progress")

        assert response.status_code == 200
        assert response.json() == {
            "total_sources": 2,
            "total_questions": 6,
            "answered_questions": 3,
            "correct_answers": 2,
            "practice_accuracy": pytest.approx(2 / 3),
            "topics": [
                {
                    "topic": "cell biology",
                    "answered": 3,
                    "correct": 2,
                    "accuracy": pytest.approx(2 / 3),
                    "mastery_estimate": pytest.approx(
                        estimate_mastery(
                            [(correct, answered_at) for _, correct, answered_at in mastery_events],
                            as_of=now,
                        )
                    ),
                    "recommended_review_in_days": recommended_review_days(
                        estimate_mastery(
                            [(correct, answered_at) for _, correct, answered_at in mastery_events],
                            as_of=now,
                        )
                    ),
                },
                {
                    "topic": "genetics",
                    "answered": 0,
                    "correct": 0,
                    "accuracy": 0.0,
                    "mastery_estimate": 0.2,
                    "recommended_review_in_days": None,
                },
            ],
        }
    finally:
        app.dependency_overrides.clear()


def test_wilson_accuracy_interval_exposes_uncertainty_for_small_samples():
    single_correct_interval = wilson_interval(1, 1)
    larger_sample_interval = wilson_interval(80, 100)

    assert single_correct_interval == pytest.approx((0.2065, 1.0), abs=0.001)
    assert larger_sample_interval is not None
    assert larger_sample_interval[0] < 0.8 < larger_sample_interval[1]
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError, match="0 <= correct <= answered"):
        wilson_interval(2, 1)


def test_practice_accuracy_chart_shows_attempted_topics_and_confidence_range():
    chart = practice_accuracy_chart(
        [
            {
                "topic": "one correct answer",
                "answered": 1,
                "correct": 1,
                "accuracy": 1.0,
            },
            {
                "topic": "not attempted",
                "answered": 0,
                "correct": 0,
                "accuracy": 0.0,
            },
        ]
    )

    assert chart is not None
    chart_spec = chart.to_dict()
    assert chart_spec["layer"][0]["encoding"]["x"]["field"] == "lower"
    assert chart_spec["layer"][0]["encoding"]["x2"]["field"] == "upper"
    assert chart_spec["layer"][1]["encoding"]["x"]["field"] == "accuracy"
    assert chart_spec["data"]["values"][0]["answered"] == 1
    assert chart_spec["data"]["values"][0]["lower"] == pytest.approx(0.2065, abs=0.001)
    assert practice_accuracy_chart(
        [{"topic": "not attempted", "answered": 0, "correct": 0, "accuracy": 0.0}]
    ) is None


def test_mastery_estimate_uses_response_and_forgets_over_time():
    now = datetime.now(timezone.utc)
    recent_correct = [(True, now)]
    recent_incorrect = [(False, now)]

    assert estimate_mastery(recent_correct, as_of=now) > estimate_mastery(
        recent_incorrect, as_of=now
    )
    assert estimate_mastery(recent_correct, as_of=now + timedelta(days=84)) < estimate_mastery(
        recent_correct, as_of=now
    )
    assert recommended_review_days(0.3) < recommended_review_days(0.8)


def test_bm25_and_cosine_hybrid_ranking_prefers_lexical_and_semantic_evidence():
    documents = [
        "Mitochondria generate cellular energy through respiration.",
        "The cell membrane regulates transport across the cell.",
        "Mitochondria mitochondria mitochondria produce energy.",
    ]
    lexical_scores = bm25_scores("mitochondria energy", documents)
    assert lexical_scores[2] > lexical_scores[0] > lexical_scores[1]

    combined = fuse_bm25_and_cosine(lexical_scores, [0.72, 0.98, 0.35])
    assert combined[0] > combined[1]
    assert combined[2] > combined[1]


def test_hybrid_retrieval_rejects_mismatched_score_lists():
    with pytest.raises(ValueError, match="same length"):
        fuse_bm25_and_cosine([1.0], [])


def test_tutor_requires_citations_on_each_response_line():
    assert _has_valid_citations("Respiration releases energy [1].\nATP is produced [2].", 2)
    assert not _has_valid_citations("Respiration releases energy [3].", 2)
    assert not _has_valid_citations("Respiration releases energy [1].\nATP is produced.", 2)
    assert _has_valid_citations(
        "Big data means varied sources [1].\n\n"
        "**Check-for-understanding:**\n"
        "Would millions of same-type records from one source qualify? Why?",
        1,
    )


@pytest.mark.asyncio
async def test_hybrid_retrieval_combines_bm25_and_qdrant_cosine(monkeypatch):
    user_id = uuid.uuid4()
    lexical_source_id = uuid.uuid4()
    semantic_source_id = uuid.uuid4()
    rows = [
        _passage(
            lexical_source_id,
            "biology.pdf",
            1,
            "Mitochondria produce ATP during cellular respiration.",
        ),
        _passage(
            semantic_source_id,
            "biology.pdf",
            2,
            "Energy conversion occurs in the organelle membrane.",
        ),
    ]
    database = _StudyDatabase(rows)
    monkeypatch.setattr(
        study_api,
        "search_study_passages",
        lambda *_args: [
            {
                "source_id": str(semantic_source_id),
                "filename": "biology.pdf",
                "location_type": "page",
                "location_number": 2,
                "chunk_index": 0,
                "content": "Energy conversion occurs in the organelle membrane.",
                "score": 0.91,
            }
        ],
    )

    passages, backend = await study_api._hybrid_retrieve_study_passages(
        database, user_id, "mitochondria ATP", 5
    )

    assert backend == "bm25+cosine"
    assert len(passages) == 2
    assert passages[0]["location_number"] == 1
    assert passages[0]["bm25_score"] > 0
    assert passages[1]["cosine_score"] == pytest.approx(0.91)


def test_learning_api_is_split_by_user_workflow():
    routers = {
        "sources": sources_router,
        "tutor": tutor_router,
        "assessments": assessments_router,
        "dashboard": dashboard_router,
    }
    expected_paths = {
        "sources": {
            "/study/sources",
            "/study/sources/{source_id}/open",
            "/study/sources/{source_id}/slides/{slide_number}",
        },
        "tutor": {
            "/study/search",
            "/study/tutor",
            "/study/tutor/transcribe",
            "/study/tutor/stream",
            "/study/conversations",
            "/study/conversations/{conversation_id}/messages",
        },
        "assessments": {
            "/study/quizzes",
            "/study/quizzes/{question_id}/answer",
            "/study/mock-exams",
            "/study/mock-exams/{exam_id}",
            "/study/mock-exams/{exam_id}/submit",
        },
        "dashboard": {"/study/progress"},
    }
    for name, router in routers.items():
        assert {route.path for route in router.routes} == expected_paths[name]


@pytest.mark.asyncio
async def test_login_reports_database_outage_as_service_unavailable():
    from sqlalchemy.exc import OperationalError

    class _UnavailableDatabase:
        async def execute(self, _statement):
            raise OperationalError("SELECT", {}, RuntimeError("database unavailable"))

    async def override_db():
        yield _UnavailableDatabase()

    app.dependency_overrides[get_db] = override_db
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/auth/login",
                json={"email": "student@example.com", "password": "StrongPass123"},
            )

        assert response.status_code == 503
        assert response.json()["detail"] == "Authentication service is temporarily unavailable"
    finally:
        app.dependency_overrides.clear()
