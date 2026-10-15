import pytest

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
async def test_assessment_rejects_negative_verifier_result(monkeypatch):
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
