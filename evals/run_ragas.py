"""Run external-judge RAGAS metrics on a reviewed, source-linked gold set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
BASELINES = ("no_rag", "naive_rag", "full_pipeline")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-20b"
DEFAULT_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"


def _nvidia_embeddings_request(
    client: Any,
    texts: list[str],
    *,
    model: str,
    input_type: str,
) -> list[list[float]]:
    if input_type not in {"query", "passage"}:
        raise ValueError("NVIDIA embedding input_type must be 'query' or 'passage'")
    if not texts:
        return []

    response = client.embeddings.create(
        model=model,
        input=texts,
        extra_body={"input_type": input_type, "modality": "text"},
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    if len(ordered) != len(texts):
        raise RuntimeError(
            f"NVIDIA embeddings returned {len(ordered)} vectors for {len(texts)} inputs"
        )
    return [item.embedding for item in ordered]


def _build_nvidia_embeddings(api_key: str, model: str, base_url: str):
    try:
        from langchain_core.embeddings import Embeddings
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "NVIDIA RAGAS embeddings require langchain-core and the OpenAI-compatible client."
        ) from exc

    class NvidiaEmbeddings(Embeddings):
        def __init__(self) -> None:
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            self._model = model

        def embed_query(self, text: str) -> list[float]:
            return _nvidia_embeddings_request(
                self._client,
                [text],
                model=self._model,
                input_type="query",
            )[0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return _nvidia_embeddings_request(
                self._client,
                texts,
                model=self._model,
                input_type="passage",
            )

    return NvidiaEmbeddings()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_dataset(cases: list[dict[str, Any]], records: dict[str, Any], minimum: int) -> None:
    if len(cases) < minimum:
        raise ValueError(
            f"Gold set contains {len(cases)} cases; at least {minimum} reviewed cases are required"
        )
    ids = [str(case.get("id", "")) for case in cases]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Every gold case must have a unique non-empty id")
    missing = [
        case_id for case_id, case in zip(ids, cases, strict=True)
        if not case.get("question") or not case.get("ground_truth")
    ]
    if missing:
        raise ValueError(
            f"Cases require question and ground_truth; invalid IDs: {missing[:5]}"
        )
    in_scope = [case for case in cases if not case.get("expected_abstention")]
    out_of_scope = [case for case in cases if case.get("expected_abstention")]
    if not in_scope:
        raise ValueError("The gold set needs in-scope cases for RAGAS scoring")
    missing_contexts = [
        str(case["id"]) for case in in_scope
        if not case.get("reference_contexts")
    ]
    if missing_contexts:
        raise ValueError(
            "In-scope cases require reference_contexts; invalid IDs: "
            f"{missing_contexts[:5]}"
        )
    abstention_rate = len(out_of_scope) / len(cases)
    if not 0.20 <= abstention_rate <= 0.25:
        raise ValueError(
            f"Out-of-material share is {abstention_rate:.1%}; Track D requires 20 to 25 percent"
        )
    for baseline in BASELINES:
        absent = [case_id for case_id in ids if baseline not in records.get(case_id, {})]
        if absent:
            raise ValueError(f"Baseline {baseline} is missing responses for {len(absent)} cases")
        for case_id in ids:
            response = records[case_id][baseline]
            if not isinstance(response.get("answer"), str) or not isinstance(
                response.get("contexts"), list
            ) or not isinstance(response.get("abstained"), bool):
                raise ValueError(
                    f"Baseline {baseline} case {case_id} requires answer, contexts, "
                    "and boolean abstained"
                )


def _validate_input_files(cases_path: Path, responses_path: Path) -> None:
    missing = [
        path for path in (cases_path, responses_path)
        if not path.is_file()
    ]
    if not missing:
        return

    details = [f"Required RAGAS input file not found: {path}" for path in missing]
    if not cases_path.is_file():
        template = cases_path.with_name("golden_rag.example.json")
        if template.is_file():
            details.append(
                f"{template} is only a schema example; create {cases_path} with at least "
                "100 reviewed source-grounded cases and 20 to 25 percent expected abstentions."
            )
        available_gold = cases_path.with_name("golden.json")
        if available_gold.is_file():
            details.append(
                f"{available_gold} exists under a different filename; only use it with "
                "--cases if it meets the RAGAS schema and minimum case requirements."
            )
    if not responses_path.is_file():
        template = responses_path.with_name("responses.benchmark.example.json")
        if template.is_file():
            details.append(
                f"{template} is only a schema example; {responses_path} must contain "
                "real responses for no_rag, naive_rag, and full_pipeline for every gold case."
            )
    raise FileNotFoundError("\n".join(details))


def _mean_score(value: Any) -> float | None:
    score = float(value)
    return score if math.isfinite(score) else None


def _ragas_rows(
    cases: list[dict[str, Any]],
    records: dict[str, Any],
    baseline: str,
) -> list[dict[str, Any]]:
    return [
        {
            "user_input": case["question"],
            "response": records[str(case["id"])][baseline]["answer"],
            "retrieved_contexts": (
                records[str(case["id"])][baseline]["contexts"]
                or ["[No context was retrieved by this baseline.]"]
            ),
            "reference": case["ground_truth"],
            "reference_contexts": case["reference_contexts"],
        }
        for case in cases
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run RAGAS with NVIDIA-hosted models; evaluation questions, answers, "
            "and contexts leave this machine."
        )
    )
    parser.add_argument("responses", type=Path, help="Three-baseline response JSON")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("golden_rag.json"))
    parser.add_argument("--minimum-cases", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report; contains scores and data hashes, not case text",
    )
    parser.add_argument(
        "--corpus-sha256",
        default=os.getenv("RAGAS_CORPUS_SHA256"),
        help="Optional source-corpus checksum (defaults to RAGAS_CORPUS_SHA256)",
    )
    parser.add_argument(
        "--allow-unreviewed-candidates",
        action="store_true",
        help="Run an exploratory score on machine-generated data clearly marked unreviewed",
    )
    parser.add_argument(
        "--allow-external-judge",
        action="store_true",
        help="Approve sending evaluation questions, answers, and contexts to NVIDIA's hosted APIs",
    )
    args = parser.parse_args()
    if not args.allow_external_judge:
        parser.error(
            "Pass --allow-external-judge only after approving transfer of evaluation "
            "questions, answers, and contexts to NVIDIA's hosted APIs"
        )

    try:
        _validate_input_files(args.cases, args.responses)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        parser.error("NVIDIA_API_KEY is required for the NVIDIA-hosted judge and embeddings")

    judge_model = os.getenv("RAGAS_MODEL", DEFAULT_JUDGE_MODEL)
    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    base_url = os.getenv("RAGAS_BASE_URL", NVIDIA_BASE_URL).rstrip("/")

    try:
        cases: list[dict[str, Any]] = json.loads(args.cases.read_text(encoding="utf-8"))
        records: dict[str, Any] = json.loads(args.responses.read_text(encoding="utf-8"))
        metadata_path = args.cases.with_name("golden_rag.metadata.json")
        dataset_metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        if (
            dataset_metadata.get("human_reviewed") is False
            and not args.allow_unreviewed_candidates
        ):
            raise ValueError(
                f"{args.cases} is marked machine-generated and unreviewed. "
                "Review the cases or pass --allow-unreviewed-candidates for exploratory "
                "scores that must not be reported as benchmark compliance."
            )
        _validate_dataset(cases, records, args.minimum_cases)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        parser.error(
            "RAGAS dependencies are missing; install them with "
            "'.\\.venv-eval\\Scripts\\python.exe -m pip install -r requirements-eval.txt'. "
            f"Missing module: {exc.name}"
        )

    judge = ChatOpenAI(
        model=judge_model,
        temperature=1,
        api_key=api_key,
        base_url=base_url,
    )
    embeddings = _build_nvidia_embeddings(api_key, embedding_model, base_url)
    report: dict[str, Any] = {
        "evaluation_status": "in_progress",
        "metric_type": "ragas_external_llm_judge",
        "provider": "NVIDIA API Catalog",
        "case_count": len(cases),
        "dataset_status": dataset_metadata.get(
            "dataset_status",
            "review_status_unrecorded",
        ),
        "human_reviewed": dataset_metadata.get("human_reviewed"),
        "ragas_scored_case_count": sum(
            not bool(case.get("expected_abstention")) for case in cases
        ),
        "expected_abstention_fraction": sum(
            bool(case.get("expected_abstention")) for case in cases
        ) / len(cases),
        "judge_model": judge_model,
        "embedding_model": embedding_model,
        "api_base_url": base_url,
        "run_config": {
            "timeout_seconds": 120,
            "max_retries": 2,
            "max_workers": 8,
        },
        "cases_sha256": _sha256_file(args.cases),
        "responses_sha256": _sha256_file(args.responses),
        "corpus_sha256": args.corpus_sha256,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baselines": {},
    }
    in_scope_cases = [case for case in cases if not case.get("expected_abstention")]
    out_of_scope_cases = [case for case in cases if case.get("expected_abstention")]
    metric_set = [faithfulness, answer_relevancy, context_precision, context_recall]
    from ragas.run_config import RunConfig

    run_config = RunConfig(
        timeout=120,
        max_retries=2,
        max_wait=5,
        max_workers=8,
        log_tenacity=True,
    )

    for baseline in BASELINES:
        dataset = Dataset.from_list(_ragas_rows(in_scope_cases, records, baseline))
        scored = evaluate(
            dataset,
            metrics=metric_set,
            llm=judge,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=False,
        ).to_pandas()
        valid_counts = {
            metric: int(scored[metric].count())
            for metric in METRICS
        }
        failed_counts = {
            metric: len(in_scope_cases) - valid_counts[metric]
            for metric in METRICS
        }
        report["baselines"][baseline] = {
            metric: _mean_score(scored[metric].mean())
            for metric in METRICS
        }
        report["baselines"][baseline]["metric_valid_case_counts"] = valid_counts
        report["baselines"][baseline]["metric_failed_case_counts"] = failed_counts
        report["baselines"][baseline]["evaluation_status"] = (
            "complete" if not any(failed_counts.values()) else "partial_metric_failures"
        )
        if out_of_scope_cases:
            correct_abstentions = sum(
                bool(records[str(case["id"])][baseline]["abstained"])
                for case in out_of_scope_cases
            )
            report["baselines"][baseline]["out_of_scope_abstention_accuracy"] = (
                correct_abstentions / len(out_of_scope_cases)
            )
        report["evaluation_status"] = (
            "partial_metric_failures"
            if any(
                baseline_data["evaluation_status"] != "complete"
                for baseline_data in report["baselines"].values()
            )
            else "in_progress"
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    report["evaluation_status"] = (
        "partial_metric_failures"
        if any(
            baseline_data["evaluation_status"] != "complete"
            for baseline_data in report["baselines"].values()
        )
        else "complete"
    )
    report_json = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{report_json}\n", encoding="utf-8")
    print(report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
