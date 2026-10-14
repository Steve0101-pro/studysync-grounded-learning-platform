"""Deterministic benchmark metrics for source-grounded study responses."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_BASELINES = ("no_rag", "naive_rag", "full_pipeline")


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_PATTERN.findall(text) if len(token) > 1}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _case_metrics(case: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, float]:
    retrieved_ids = set(response.get("retrieved_context_ids", []))
    gold_ids = set(case.get("gold_context_ids", []))
    citations = set(response.get("citations", []))
    answer = str(response.get("answer", ""))

    context_precision = (
        len(retrieved_ids & gold_ids) / len(retrieved_ids) if retrieved_ids else 0.0
    )
    context_recall = len(retrieved_ids & gold_ids) / len(gold_ids) if gold_ids else 0.0

    expected_answer_tokens = _tokens(str(case.get("reference_answer", "")))
    answer_tokens = _tokens(answer)
    overlap = len(expected_answer_tokens & answer_tokens)
    answer_precision = overlap / len(answer_tokens) if answer_tokens else 0.0
    answer_recall = overlap / len(expected_answer_tokens) if expected_answer_tokens else 0.0
    answer_relevance = (
        2 * answer_precision * answer_recall / (answer_precision + answer_recall)
        if answer_precision + answer_recall
        else 0.0
    )

    reference_facts = case.get("reference_facts", [])
    stated_facts = [
        fact for fact in reference_facts if _tokens(fact) and _tokens(fact) <= answer_tokens
    ]
    retrieved_text = " ".join(str(item) for item in response.get("retrieved_contexts", []))
    retrieved_tokens = _tokens(retrieved_text)
    supported_facts = [
        fact for fact in stated_facts if _tokens(fact) <= retrieved_tokens
    ]
    supported_claim_rate = len(supported_facts) / len(stated_facts) if stated_facts else 1.0
    factual_coverage = (
        len(stated_facts) / len(reference_facts) if reference_facts else 1.0
    )

    citation_precision = (
        len(citations & retrieved_ids) / len(citations) if citations else 0.0
    )
    expected_abstention = bool(case.get("expected_abstention", False))
    abstention_accuracy = float(bool(response.get("abstained", False)) == expected_abstention)

    return {
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_relevance_lexical_f1": answer_relevance,
        "supported_claim_rate_lexical": supported_claim_rate,
        "factual_coverage_lexical": factual_coverage,
        "citation_precision": citation_precision,
        "abstention_accuracy": abstention_accuracy,
    }


def run_benchmark(
    cases: Sequence[Mapping[str, Any]],
    response_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Score the three required baselines without representing proxies as RAGAS."""
    case_ids = [str(case["id"]) for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Benchmark case IDs must be unique")
    missing_fields = [
        case_id for case_id, case in zip(case_ids, cases, strict=True)
        if "question" not in case or "reference_answer" not in case
    ]
    if missing_fields:
        raise ValueError(f"Benchmark cases are missing required fields: {missing_fields[:5]}")

    grouped: dict[str, dict[str, list[float]]] = {
        baseline: defaultdict(list) for baseline in _BASELINES
    }
    per_case: list[dict[str, Any]] = []
    for case, case_id in zip(cases, case_ids, strict=True):
        baseline_results: dict[str, dict[str, float]] = {}
        record = response_records.get(case_id, {})
        for baseline in _BASELINES:
            metrics = _case_metrics(case, record.get(baseline, {}))
            baseline_results[baseline] = metrics
            for name, value in metrics.items():
                grouped[baseline][name].append(value)
        per_case.append({"id": case_id, "baselines": baseline_results})

    return {
        "metric_type": "deterministic_lexical_proxies_not_ragas",
        "case_count": len(cases),
        "baselines": {
            baseline: {
                metric: _mean(values) for metric, values in sorted(metrics.items())
            }
            for baseline, metrics in grouped.items()
        },
        "results": per_case,
    }
