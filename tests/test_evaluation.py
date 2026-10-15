import json
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

from evals.benchmark import run_benchmark
from evals.prepare_pdf_ragas import _write_responses
from evals.run_ragas import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    NVIDIA_BASE_URL,
    _nvidia_embeddings_request,
    _ragas_rows,
    _validate_dataset,
    _validate_input_files,
)
from evals.simulate_learners import CONDITIONS, simulate


def test_benchmark_compares_three_baselines_and_reports_lexical_metrics():
    case = {
        "id": "cell-001",
        "question": "What produces ATP?",
        "reference_answer": "Mitochondria produce ATP.",
        "reference_facts": ["Mitochondria produce ATP"],
        "gold_context_ids": ["source:2"],
        "expected_abstention": False,
    }
    responses = {
        "cell-001": {
            "no_rag": {"answer": "", "abstained": True},
            "naive_rag": {
                "answer": "Mitochondria produce ATP.",
                "retrieved_context_ids": ["noise"],
                "retrieved_contexts": ["Ribosomes assemble proteins."],
                "citations": ["noise"],
            },
            "full_pipeline": {
                "answer": "Mitochondria produce ATP.",
                "retrieved_context_ids": ["source:2"],
                "retrieved_contexts": ["Mitochondria produce ATP."],
                "citations": ["source:2"],
            },
        }
    }

    report = run_benchmark([case], responses)

    assert report["case_count"] == 1
    assert report["metric_type"] == "deterministic_lexical_proxies_not_ragas"
    assert set(report["baselines"]) == {"no_rag", "naive_rag", "full_pipeline"}
    assert report["baselines"]["full_pipeline"]["context_recall"] == 1.0
    assert report["baselines"]["naive_rag"]["context_recall"] == 0.0
    assert report["baselines"]["full_pipeline"]["supported_claim_rate_lexical"] == 1.0


def test_simulation_runs_three_profiles_conditions_and_five_sessions():
    profiles = {
        "novice": [0.1, 0.2, 0.15, 0.1],
        "intermediate": [0.5, 0.6, 0.45, 0.55],
        "advanced": [0.8, 0.9, 0.75, 0.85],
    }

    report = simulate(profiles, sessions=5, seed=42)

    assert report["simulation_type"] == "synthetic_policy_comparison_not_learner_outcome_evidence"
    assert report["sessions_per_profile_condition"] == 5
    assert len(report["results"]) == 9
    assert set(report["summary"]) == set(CONDITIONS)
    assert all(
        0 <= row["question_repetition_rate"] <= 1
        and 0 <= row["weak_topic_targeting_rate"] <= 1
        and row["mastery_estimation_mae"] >= 0
        for row in report["results"]
    )
    assert all(
        row["question_repetition_rate"] <= 0.05
        for row in report["results"]
        if row["condition"] != "random"
    )


def test_trackd_simulation_covers_seven_behavior_profiles_and_all_conditions():
    profile_path = Path(__file__).resolve().parents[1] / "evals" / "simulated_profiles.json"
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))

    report = simulate(profiles, sessions=5, seed=42)

    assert set(profiles) == {
        "fast_learner",
        "slow_learner",
        "forgetful",
        "careless",
        "guesser",
        "misconception_prone",
        "cold_start_novice",
    }
    assert report["profile_count"] == 7
    assert len(report["results"]) == 7 * len(CONDITIONS)
    assert set(report["summary"]) == set(CONDITIONS)
    assert {
        row["profile"] for row in report["results"]
    } == set(profiles)
    assert all(
        row["sessions"] == 5
        and len(row["mastery_curve"]) == 5
        and all(
            0 <= point["mean_mastery"] <= 1
            for point in row["mastery_curve"]
        )
        and set(row["learner_parameters"])
        == {
            "initial_mastery",
            "learning_rate",
            "forgetting_rate",
            "slip_rate",
            "guess_rate",
            "misconception_rate",
        }
        for row in report["results"]
    )


def test_simulation_rejects_less_than_five_sessions():
    try:
        simulate({"novice": [0.1, 0.2, 0.15, 0.1]}, sessions=4)
    except ValueError as exc:
        assert "at least five sessions" in str(exc)
    else:
        raise AssertionError("The simulator accepted fewer than five sessions")


def test_ragas_gate_requires_100_cases_and_required_off_material_fraction():
    cases = [
        {
            "id": f"case-{index}",
            "question": "question",
            "ground_truth": "answer",
            "reference_contexts": ["source passage"],
            "expected_abstention": index < 20,
        }
        for index in range(100)
    ]
    responses = {
        case["id"]: {
            baseline: {
                "answer": "answer",
                "contexts": ["context"],
                "abstained": False,
            }
            for baseline in ("no_rag", "naive_rag", "full_pipeline")
        }
        for case in cases
    }

    _validate_dataset(cases, responses, 100)

    with pytest.raises(ValueError, match="at least 100"):
        _validate_dataset(cases[:99], responses, 100)
    cases[0]["expected_abstention"] = False
    with pytest.raises(ValueError, match="Out-of-material share"):
        _validate_dataset(cases, responses, 100)


def test_ragas_defaults_select_current_nvidia_models():
    assert DEFAULT_JUDGE_MODEL == "openai/gpt-oss-20b"
    assert DEFAULT_EMBEDDING_MODEL == "nvidia/nemotron-3-embed-1b"
    assert NVIDIA_BASE_URL == "https://integrate.api.nvidia.com/v1"


def test_nvidia_embeddings_use_requested_mode_and_restore_input_order():
    class FakeClient:
        def __init__(self):
            self.call = None

            class Embeddings:
                def __init__(inner_self):
                    inner_self.create = self.create

            self.embeddings = Embeddings()

        def create(self, **kwargs):
            self.call = kwargs
            return type(
                "EmbeddingResponse",
                (),
                {
                    "data": [
                        type("Embedding", (), {"index": 1, "embedding": [0.2]})(),
                        type("Embedding", (), {"index": 0, "embedding": [0.1]})(),
                    ]
                },
            )()

    client = FakeClient()

    vectors = _nvidia_embeddings_request(
        client,
        ["question", "passage"],
        model=DEFAULT_EMBEDDING_MODEL,
        input_type="passage",
    )

    assert vectors == [[0.1], [0.2]]
    assert client.call["model"] == DEFAULT_EMBEDDING_MODEL
    assert client.call["input"] == ["question", "passage"]
    assert client.call["extra_body"] == {
        "input_type": "passage",
        "modality": "text",
    }


def test_nvidia_embedding_helper_rejects_invalid_mode_and_skips_empty_batch():
    class FakeClient:
        def __init__(self):
            self.called = False
            self.embeddings = self

        def create(self, **_kwargs):
            assert not _kwargs
            self.called = True
            raise AssertionError("Empty batches must not call the service")

    client = FakeClient()
    assert _nvidia_embeddings_request(
        client,
        [],
        model=DEFAULT_EMBEDDING_MODEL,
        input_type="query",
    ) == []
    assert not client.called

    with pytest.raises(ValueError, match="input_type"):
        _nvidia_embeddings_request(
            client,
            ["question"],
            model=DEFAULT_EMBEDDING_MODEL,
            input_type="invalid",
        )


def test_ragas_dataset_uses_ragas_0215_single_turn_column_names():
    cases = [
        {
            "id": "case-1",
            "question": "What is ATP?",
            "ground_truth": "ATP is an energy carrier.",
            "reference_contexts": ["ATP is an energy carrier."],
        }
    ]
    responses = {
        "case-1": {
            "full_pipeline": {
                "answer": "ATP is an energy carrier.",
                "contexts": ["ATP is used for cellular energy."],
                "abstained": False,
            }
        }
    }

    assert _ragas_rows(cases, responses, "full_pipeline") == [
        {
            "user_input": "What is ATP?",
            "response": "ATP is an energy carrier.",
            "retrieved_contexts": ["ATP is used for cellular energy."],
            "reference": "ATP is an energy carrier.",
            "reference_contexts": ["ATP is an energy carrier."],
        }
    ]


def test_ragas_datasets_allow_empty_ood_reference_context_and_require_abstention():
    cases = [
        {
            "id": f"course-{index}",
            "question": "What is ATP?",
            "ground_truth": "Course answer.",
            "reference_contexts": ["Source excerpt."],
            "expected_abstention": False,
        }
        for index in range(1, 5)
    ]
    cases.append(
        {
            "id": "ood-1",
            "question": "What is the capital of France?",
            "ground_truth": "The course source does not establish this.",
            "reference_contexts": [],
            "expected_abstention": True,
        }
    )
    records = {
        case["id"]: {
            baseline: {
                "answer": "The source does not establish this.",
                "contexts": [],
                "abstained": True,
            }
            for baseline in ("no_rag", "naive_rag", "full_pipeline")
        }
        for case in cases
    }

    _validate_dataset(cases, records, 5)
    assert len(
        [
            case for case in cases
            if not case["expected_abstention"]
        ]
    ) == 4
    assert _ragas_rows(cases[:1], records, "full_pipeline")[0]["retrieved_contexts"] == [
        "[No context was retrieved by this baseline.]"
    ]


def test_pdf_baseline_generation_retries_missing_and_resumes_without_gold_leakage(tmp_path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Linear regression models relationships in data.")
    document.save(pdf_path)
    document.close()

    case = {
        "id": "ood-001",
        "question": "What is the capital city of France?",
        "ground_truth": "The course source does not establish this answer.",
        "reference_contexts": [],
        "gold_context_ids": ["gold-only-context"],
        "expected_abstention": True,
    }

    class FakeCompletions:
        def __init__(self):
            self.calls = 0
            self.tasks_seen = []

        def create(self, **kwargs):
            self.calls += 1
            tasks = json.loads(kwargs["messages"][-1]["content"])["tasks"]
            self.tasks_seen.extend(tasks)
            selected_tasks = tasks[:2] if self.calls == 1 else tasks
            response_items = [
                {
                    "case_id": task["case_id"],
                    "baseline": task["baseline"],
                    "answer": "Not established by the course.",
                    "abstained": True,
                }
                for task in selected_tasks
            ]
            content = json.dumps({"responses": response_items})
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content),
                    )
                ]
            )

    completions = FakeCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    responses_path = tmp_path / "responses.json"

    _write_responses(client, [case], pdf_path, responses_path)

    records = json.loads(responses_path.read_text(encoding="utf-8"))
    assert set(records[case["id"]]) == {"no_rag", "naive_rag", "full_pipeline"}
    assert completions.calls == 2
    assert all("expected_abstention" not in task for task in completions.tasks_seen)
    assert all(not task["passages"] for task in completions.tasks_seen)
    assert all(
        record["retrieved_context_ids"] == []
        for record in records[case["id"]].values()
    )

    _write_responses(client, [case], pdf_path, responses_path)
    assert completions.calls == 2


def test_ragas_missing_inputs_explain_templates_are_not_real_datasets(tmp_path):
    evals_dir = tmp_path / "evals"
    evals_dir.mkdir()
    (evals_dir / "golden_rag.example.json").write_text("[]", encoding="utf-8")
    (evals_dir / "responses.benchmark.example.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as error:
        _validate_input_files(
            evals_dir / "golden_rag.json",
            evals_dir / "responses.json",
        )

    message = str(error.value)
    assert "golden_rag.example.json is only a schema example" in message
    assert "at least 100 reviewed source-grounded cases" in message
    assert "responses.benchmark.example.json is only a schema example" in message
    assert "real responses" in message
