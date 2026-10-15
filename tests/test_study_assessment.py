import pytest
from pydantic import SecretStr

from app.config import settings
from app.services import study_assessment


def test_generated_multiple_choice_requires_four_distinct_choices_and_valid_key():
    valid = {
        "question": "Which molecule is produced in respiration?",
        "options": ["ATP", "Water", "Glucose", "Protein"],
        "answer_index": 0,
        "explanation": "The passage identifies ATP as the product.",
    }
    assert study_assessment._validate_generated_item(
        valid,
        assessment_type="multiple_choice",
    )["answer_index"] == 0

    invalid = {**valid, "answer_index": 4}
    with pytest.raises(ValueError, match="four unique options"):
        study_assessment._validate_generated_item(
            invalid,
            assessment_type="multiple_choice",
        )


def test_generated_short_answer_requires_a_rubric():
    valid = {
        "question": "What is the role of the inner membrane?",
        "answer": "It supports ATP production.",
        "rubric": ["The membrane supports ATP production."],
        "explanation": "This is stated in the source passage.",
    }
    validated = study_assessment._validate_generated_item(
        valid,
        assessment_type="short_answer",
    )
    assert validated["rubric"] == ["The membrane supports ATP production."]

    with pytest.raises(ValueError, match="non-empty grading rubric"):
        study_assessment._validate_generated_item(
            {**valid, "rubric": []},
            assessment_type="short_answer",
        )


def test_numerical_grading_requires_matching_unit_and_tolerance():
    args = {
        "answer_value": 2.04,
        "answer_unit": "cm",
        "expected_value": 2.0,
        "expected_unit": "cm",
        "tolerance": 0.05,
    }
    assert study_assessment.grade_numerical_answer(**args) is True
    assert study_assessment.grade_numerical_answer(**{**args, "answer_value": 2.06}) is False
    assert study_assessment.grade_numerical_answer(**{**args, "answer_unit": "mm"}) is False


@pytest.mark.asyncio
async def test_assessment_requires_distinct_generator_and_verifier_models(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", SecretStr(""))
    monkeypatch.setattr(
        study_assessment,
        "_model_providers",
        lambda: [
            ("provider", "https://models.example", "key", "same-model"),
            ("alias", "https://models.example", "key", "same-model"),
        ],
    )
    with pytest.raises(RuntimeError, match="two distinct configured models"):
        await study_assessment.generate_verified_assessment(
            topic="cell respiration",
            passage="Mitochondria produce ATP during respiration.",
            assessment_type="multiple_choice",
            difficulty="medium",
        )


@pytest.mark.asyncio
async def test_assessment_generation_requires_independent_positive_verification(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", SecretStr(""))
    monkeypatch.setattr(
        study_assessment,
        "_model_providers",
        lambda: [
            ("first", "https://first.example", "key", "model-a"),
            ("second", "https://second.example", "key", "model-b"),
        ],
    )
    calls = []

    async def request_json(provider, **_kwargs):
        calls.append(provider[3])
        if provider[3] == "model-a":
            return {
                "question": "Which molecule is produced during respiration?",
                "options": ["ATP", "Water", "Glucose", "Protein"],
                "answer_index": 0,
                "explanation": "The source says ATP is produced.",
            }
        return {"supported": True, "answer_correct": True, "reason": "Supported"}

    monkeypatch.setattr(study_assessment, "_request_json", request_json)
    item, verification = await study_assessment.generate_verified_assessment(
        topic="cell respiration",
        passage="Mitochondria produce ATP during respiration.",
        assessment_type="multiple_choice",
        difficulty="medium",
    )

    assert item["answer_index"] == 0
    assert calls == ["model-a", "model-b"]
    assert verification == "first+second_verified"


@pytest.mark.asyncio
async def test_rejected_candidate_is_regenerated_once_before_failing(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", SecretStr(""))
    monkeypatch.setattr(
        study_assessment,
        "_model_providers",
        lambda: [
            ("first", "https://first.example", "key", "model-a"),
            ("second", "https://second.example", "key", "model-b"),
        ],
    )
    generation_calls = []

    async def request_json(provider, *, system_prompt, user_payload):
        if "Independently verify" not in system_prompt:
            generation_calls.append(provider[3])
            return {
                "question": (
                    "Which unsupported claim is generated first?"
                    if provider[3] == "model-a"
                    else "Which molecule does the source say is produced?"
                ),
                "options": ["ATP", "Water", "Glucose", "Protein"],
                "answer_index": 0,
                "explanation": "The source identifies ATP.",
            }
        if "unsupported claim" in user_payload["candidate_item"]["question"]:
            return {
                "supported": False,
                "answer_correct": False,
                "reason": "The question is unsupported.",
            }
        return {"supported": True, "answer_correct": True, "reason": "Supported"}

    monkeypatch.setattr(study_assessment, "_request_json", request_json)
    item, verification = await study_assessment.generate_verified_assessment(
        topic="cell respiration",
        passage="Mitochondria produce ATP during respiration.",
        assessment_type="multiple_choice",
        difficulty="medium",
    )

    assert generation_calls == ["model-a", "model-b"]
    assert "does the source say" in item["question"]
    assert verification == "second+first_verified"


@pytest.mark.asyncio
async def test_gemini_is_available_as_assessment_generator_and_verifier(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", SecretStr("gemini-test"))
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-primary")
    monkeypatch.setattr(settings, "GEMINI_FALLBACK_MODEL", "gemini-fallback")
    monkeypatch.setattr(
        study_assessment,
        "_model_providers",
        lambda: [
            ("NVIDIA", "https://nvidia.example/v1", "key", "nvidia-model"),
        ],
    )
    calls = []

    async def request_json(provider, **_kwargs):
        calls.append(provider)
        if provider[0] == "NVIDIA":
            raise RuntimeError("NVIDIA model unavailable")
        if provider[3] == "gemini-primary":
            return {
                "question": "Which molecule is produced during respiration?",
                "options": ["ATP", "Water", "Glucose", "Protein"],
                "answer_index": 0,
                "explanation": "The passage says ATP is produced.",
            }
        return {"supported": True, "answer_correct": True, "reason": "Supported"}

    monkeypatch.setattr(study_assessment, "_request_json", request_json)
    item, verification = await study_assessment.generate_verified_assessment(
        topic="cell respiration",
        passage="Mitochondria produce ATP during respiration.",
        assessment_type="multiple_choice",
        difficulty="medium",
    )

    assert item["answer_index"] == 0
    assert [provider[3] for provider in calls] == [
        "nvidia-model",
        "gemini-primary",
        "nvidia-model",
        "gemini-fallback",
    ]
    assert verification == "gemini+gemini_verified"


@pytest.mark.asyncio
async def test_request_json_uses_gemini_native_json_mode(monkeypatch):
    captured = {}

    class _Models:
        async def generate_content(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"text": '{"supported": true}'})()

    monkeypatch.setattr(
        study_assessment,
        "gemini_client",
        type(
            "GeminiClient",
            (),
            {
                "aio": type(
                    "AsyncClient",
                    (),
                    {"models": _Models()},
                )(),
            },
        )(),
    )

    result = await study_assessment._request_json(
        ("Gemini", "", "", "gemini-primary"),
        system_prompt="Return JSON.",
        user_payload={"topic": "biology"},
    )

    assert result == {"supported": True}
    assert captured["model"] == "gemini-primary"
    assert captured["config"].response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_assessment_rejects_negative_verifier_result(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", SecretStr(""))
    monkeypatch.setattr(
        study_assessment,
        "_model_providers",
        lambda: [
            ("first", "https://first.example", "key", "model-a"),
            ("second", "https://second.example", "key", "model-b"),
        ],
    )

    async def request_json(provider, **_kwargs):
        if provider[3] == "model-a":
            return {
                "question": "Which molecule is produced during respiration?",
                "options": ["ATP", "Water", "Glucose", "Protein"],
                "answer_index": 0,
                "explanation": "The source says ATP is produced.",
            }
        return {"supported": False, "answer_correct": False, "reason": "Unsupported"}

    monkeypatch.setattr(study_assessment, "_request_json", request_json)
    with pytest.raises(RuntimeError, match="verification rejected"):
        await study_assessment.generate_verified_assessment(
            topic="cell respiration",
            passage="Mitochondria produce ATP during respiration.",
            assessment_type="multiple_choice",
            difficulty="medium",
        )
