"""Generate locally stored RAGAS candidate cases and NVIDIA model responses.

Generated cases are candidates for human review, not a reviewed gold set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import fitz
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evals.run_ragas import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    NVIDIA_BASE_URL,
)

DEFAULT_PDF = PROJECT_ROOT / "An-Introduction-to-Machine-Learning.pdf"
DEFAULT_CASES = PROJECT_ROOT / "evals" / "golden_rag.json"
DEFAULT_RESPONSES = PROJECT_ROOT / "evals" / "responses.json"
OOD_QUESTIONS = [
    "What is the capital city of France?",
    "At what temperature does pure water boil at sea level in Celsius?",
    "Who painted the artwork Guernica?",
    "What is the chemical symbol for gold?",
    "Which ocean lies between Africa and Australia?",
    "Who wrote the play Hamlet?",
    "What is the largest planet in the Solar System?",
    "How many sides does a hexagon have?",
    "What is the main ingredient in traditional hummus?",
    "In which year did the Berlin Wall fall?",
    "What is the currency used in Japan?",
    "Which gas do plants absorb during photosynthesis?",
    "What is the square root of 144?",
    "Who composed the Four Seasons?",
    "Which continent contains the Sahara Desert?",
    "What is the freezing point of water in Celsius?",
    "How many chambers does a human heart have?",
    "What is the official language of Brazil?",
    "Which metal is liquid at ordinary room temperature?",
    "What is the distance formula for speed, distance, and time?",
]
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunks(text: str, size: int = 1000, overlap: int = 120) -> list[str]:
    normalized = " ".join(text.split())
    result: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if len(chunk) >= 160:
            result.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return result


def _tokens(text: str) -> set[str]:
    return {word.casefold() for word in _TOKEN_RE.findall(text) if len(word) > 2}


def _retrieve(question: str, passages: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    query_terms = _tokens(question)
    ranked = []
    for passage in passages:
        passage_terms = _tokens(passage["content"])
        if not query_terms:
            continue
        overlap = query_terms.intersection(passage_terms)
        score = len(overlap) / len(query_terms)
        if score:
            ranked.append((score, passage))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [passage for _, passage in ranked[:limit]]


def _response_text(response: Any) -> str:
    message = response.choices[0].message
    content = message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            "NVIDIA GPT-OSS returned no answer content; increase max_tokens or check provider output."
        )
    return content.strip()


def _parse_json_object(content: str) -> dict[str, Any]:
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("NVIDIA model response did not contain a JSON object")
    parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("NVIDIA model JSON response must be an object")
    return parsed


def _case_candidates(
    client: OpenAI,
    pages: list[tuple[int, str]],
    corpus_sha: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    cases: list[dict[str, Any]] = []
    passages: list[dict[str, str]] = []
    for page_number, page_text in pages:
        for chunk_number, content in enumerate(_chunks(page_text)):
            passages.append(
                {
                    "id": f"{corpus_sha[:16]}:page:{page_number}:chunk:{chunk_number}",
                    "page": page_number,
                    "content": content,
                }
            )

        page_passages = [
            passage for passage in passages if passage["page"] == page_number
        ]
        source_material = "\n".join(
            f"[{passage['id']}] {passage['content']}"
            for passage in page_passages
        )
        completion = client.chat.completions.create(
            model=DEFAULT_JUDGE_MODEL,
            temperature=1,
            max_tokens=6000,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create exactly 12 distinct, answerable educational question "
                        "candidates using only the supplied course excerpts. Each answer "
                        "must be supported by one supplied passage ID. Return a JSON object "
                        "with a 'cases' array; each item has question, answer, passage_id. "
                        "Use clear standalone questions. Do not invent facts or copy long "
                        "passages into answers."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Course PDF page {page_number}. Produce exactly 12 questions.\n"
                        f"Passages:\n{source_material}"
                    ),
                },
            ],
        )
        parsed = _parse_json_object(_response_text(completion))
        generated = parsed.get("cases")
        if not isinstance(generated, list):
            raise ValueError(f"Page {page_number}: model response has no cases list")
        by_id = {passage["id"]: passage for passage in page_passages}
        for item in generated:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            passage_id = str(item.get("passage_id", "")).strip()
            passage = by_id.get(passage_id)
            if not question or not answer or passage is None:
                continue
            cases.append(
                {
                    "id": f"ml-p{page_number:02d}-{len(cases) + 1:03d}",
                    "question": question,
                    "reference_answer": answer,
                    "reference_facts": [answer],
                    "ground_truth": answer,
                    "gold_context_ids": [passage_id],
                    "reference_contexts": [passage["content"]],
                    "expected_abstention": False,
                    "source_page": page_number,
                    "candidate_status": "machine_generated_unreviewed",
                }
            )
        accepted_count = sum(case["source_page"] == page_number for case in cases)
        print(
            f"Received {len(generated)} candidates and accepted {accepted_count} "
            f"source-linked candidates from page {page_number}."
        )
    return cases, passages


def _write_candidates(client: OpenAI, pdf_path: Path, cases_path: Path) -> list[dict[str, Any]]:
    corpus_sha = _sha256(pdf_path)
    with fitz.open(pdf_path) as pdf:
        pages = [
            (number, " ".join(page.get_text().split()))
            for number, page in enumerate(pdf, start=1)
            if len(" ".join(page.get_text().split())) >= 1000
        ]
    cases, _passages = _case_candidates(client, pages, corpus_sha)
    unique: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for case in cases:
        normalized = " ".join(case["question"].casefold().split())
        if normalized not in seen_questions:
            seen_questions.add(normalized)
            unique.append(case)
    if len(unique) < 80:
        raise RuntimeError(
            f"Generated only {len(unique)} unique in-scope cases; at least 80 are needed. "
            "Review source extraction and retry rather than filling with duplicates."
        )
    unique = unique[:80]

    for index, question in enumerate(OOD_QUESTIONS, start=1):
        unique.append(
            {
                "id": f"ood-{index:03d}",
                "question": question,
                "reference_answer": "The supplied course source does not establish this answer.",
                "reference_facts": [],
                "ground_truth": "The supplied course source does not establish this answer.",
                "gold_context_ids": [],
                "reference_contexts": [],
                "expected_abstention": True,
                "candidate_status": "human_review_required_out_of_scope",
            }
        )

    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(
        json.dumps(unique, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "dataset_status": "machine_generated_candidate_not_human_reviewed",
        "human_reviewed": False,
        "source_file": pdf_path.name,
        "source_sha256": corpus_sha,
        "candidate_generation_model": DEFAULT_JUDGE_MODEL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(unique),
        "in_scope_candidates": 80,
        "out_of_scope_candidates": len(OOD_QUESTIONS),
        "external_processing": "PDF page excerpts were sent to NVIDIA GPT-OSS-20B to draft candidates.",
    }
    metadata_path = cases_path.with_name("golden_rag.metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(unique)} unreviewed candidate cases to {cases_path}.")
    return unique


def _write_responses(
    client: OpenAI,
    cases: list[dict[str, Any]],
    pdf_path: Path,
    responses_path: Path,
) -> None:
    corpus_sha = _sha256(pdf_path)
    with fitz.open(pdf_path) as pdf:
        passages = [
            {
                "id": f"{corpus_sha[:16]}:page:{page_number}:chunk:{chunk_number}",
                "page": page_number,
                "content": content,
            }
            for page_number, page in enumerate(pdf, start=1)
            for chunk_number, content in enumerate(_chunks(" ".join(page.get_text().split())))
        ]

    if responses_path.is_file():
        try:
            previous_records = json.loads(responses_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Existing response file is not valid JSON: {responses_path}") from exc
        if not isinstance(previous_records, dict):
            raise ValueError(f"Existing response file must contain an object: {responses_path}")
    else:
        previous_records = {}
    records: dict[str, Any] = {
        str(case["id"]): {}
        for case in cases
    }
    for batch_start in range(0, len(cases), 5):
        batch = cases[batch_start : batch_start + 5]
        tasks: list[dict[str, Any]] = []
        task_context: dict[tuple[str, str], list[dict[str, str]]] = {}
        retrieval_methods: dict[tuple[str, str], str] = {}
        for case in batch:
            case_id = str(case["id"])
            prior_baselines = previous_records.get(case_id, {})
            if not isinstance(prior_baselines, dict):
                prior_baselines = {}
            for baseline, top_k in (
                ("no_rag", 0),
                ("naive_rag", 1),
                ("full_pipeline", 5),
            ):
                retrieved = (
                    []
                    if baseline == "no_rag"
                    else _retrieve(case["question"], passages, top_k)
                )
                ids = [item["id"] for item in retrieved]
                key = (case_id, baseline)
                task_context[key] = retrieved
                retrieval_methods[key] = (
                    "none" if baseline == "no_rag"
                    else f"local_lexical_top_{top_k}"
                )
                prior_response = prior_baselines.get(baseline)
                if (
                    isinstance(prior_response, dict)
                    and isinstance(prior_response.get("answer"), str)
                    and prior_response["answer"].strip()
                    and isinstance(prior_response.get("abstained"), bool)
                    and prior_response.get("retrieved_context_ids") == ids
                    and prior_response.get("contexts")
                    == [item["content"] for item in retrieved]
                    and prior_response.get("retrieval_method") == retrieval_methods[key]
                    and prior_response.get("generation_model") == DEFAULT_JUDGE_MODEL
                ):
                    records[case_id][baseline] = prior_response
                    continue
                tasks.append(
                    {
                        "case_id": case_id,
                        "baseline": baseline,
                        "question": case["question"],
                        "passages": [
                            {
                                "id": item["id"],
                                "content": item["content"],
                            }
                            for item in retrieved
                        ],
                    }
                )

        generated_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for attempt in range(3):
            pending_tasks = [
                task for task in tasks
                if (task["case_id"], task["baseline"]) not in generated_by_key
            ]
            if not pending_tasks:
                break
            completion = client.chat.completions.create(
                model=DEFAULT_JUDGE_MODEL,
                temperature=1,
                max_tokens=7000,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Produce one response for every supplied task. Return a JSON "
                            "object with a 'responses' array. Each item must include case_id, "
                            "baseline, a non-empty answer, and boolean abstained. For no_rag, "
                            "answer from general knowledge but clearly state it is not "
                            "grounded in the course. For naive_rag and full_pipeline, answer "
                            "only from the provided passages, cite factual claims with the "
                            "exact passage ID in square brackets, and abstain if passages do "
                            "not support the answer. Do not use outside knowledge for either "
                            "retrieval baseline."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"tasks": pending_tasks},
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
            generated = _parse_json_object(_response_text(completion)).get("responses")
            if not isinstance(generated, list):
                raise ValueError("NVIDIA response generation returned no responses list")
            for item in generated:
                if not isinstance(item, dict):
                    continue
                key = (str(item.get("case_id", "")), str(item.get("baseline", "")))
                answer = item.get("answer")
                if (
                    key in task_context
                    and isinstance(answer, str)
                    and answer.strip()
                    and isinstance(item.get("abstained"), bool)
                ):
                    generated_by_key[key] = item

        missing_keys = []
        for task in tasks:
            key = (task["case_id"], task["baseline"])
            response = generated_by_key.get(key)
            if response is None:
                missing_keys.append(key)
                continue
            retrieved = task_context[key]
            ids = [item["id"] for item in retrieved]
            answer = response["answer"].strip()
            records[key[0]][key[1]] = {
                "answer": answer,
                "contexts": [item["content"] for item in retrieved],
                "retrieved_context_ids": ids,
                "retrieved_contexts": [item["content"] for item in retrieved],
                "citations": [passage_id for passage_id in ids if passage_id in answer],
                "abstained": response["abstained"],
                "generation_model": DEFAULT_JUDGE_MODEL,
                "retrieval_method": retrieval_methods[key],
            }
        responses_path.parent.mkdir(parents=True, exist_ok=True)
        responses_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if missing_keys:
            raise RuntimeError(
                f"NVIDIA response generation omitted {len(missing_keys)} baseline "
                f"response(s) after retries; first missing key: {missing_keys[0]}. "
                f"Successful responses were saved to {responses_path}; rerun to resume."
            )
        print(f"Prepared baseline responses for cases {batch_start + 1}-{batch_start + len(batch)}.")

    responses_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote model-generated baseline responses to {responses_path}; "
        "no_rag uses no contexts, while the other baselines use local lexical retrieval."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Draft local PDF evaluation candidates and NVIDIA GPT-OSS baseline responses. "
            "The output is not human-reviewed gold data."
        )
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--skip-candidate-generation", action="store_true")
    parser.add_argument("--skip-response-generation", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        parser.error("NVIDIA_API_KEY is required; set it in the local .env or process environment")
    if not args.pdf.is_file():
        parser.error(f"PDF source not found: {args.pdf}")
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("RAGAS_BASE_URL", NVIDIA_BASE_URL),
        timeout=90,
        max_retries=3,
    )
    if args.skip_candidate_generation:
        if not args.cases.is_file():
            parser.error(f"Candidate dataset not found: {args.cases}")
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
    else:
        cases = _write_candidates(client, args.pdf, args.cases)
    if not args.skip_response_generation:
        _write_responses(client, cases, args.pdf, args.responses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
