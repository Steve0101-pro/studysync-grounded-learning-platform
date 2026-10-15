# Evaluation Workflows

## Current distinction

- `run_benchmark.py` compares no-RAG, naive-RAG, and full-pipeline output with
  deterministic lexical proxies. These proxies are not RAGAS metrics.
- `run_ragas.py` runs RAGAS faithfulness, answer relevancy, context precision,
  and context recall. Its defaults use NVIDIA-hosted
  `openai/gpt-oss-20b` as judge and NVIDIA
  `nvidia/nemotron-3-embed-1b` for embeddings. It is opt-in because evaluation
  questions, answers, and retrieved/reference contexts are sent to NVIDIA's
  hosted APIs.
- `simulate_learners.py` compares random, spaced-repetition, and adaptive
  policies over five or more sessions for fast, slow, forgetful, careless,
  guesser, misconception-prone, and cold-start synthetic learner profiles.
  It emits per-session mastery curves, but must not be reported as evidence
  of real learner outcomes.

The example JSON files are schema examples only. They are deliberately not
counted as gold cases. The required reviewed dataset must contain at least 100
unique questions, include 20–25% out-of-material questions, and identify
source-grounded reference answers and context locations. Do not place private
student information in evaluation records.

## Build and review a gold set

Copy `golden_rag.example.json` and add one row for each reviewed case:

- `id`, `question`, `ground_truth`, `reference_answer`
- `reference_facts` for deterministic lexical coverage checks
- `gold_context_ids` and `reference_contexts` with source IDs and exact
  page/slide/timestamp chunks
- `expected_abstention` set for questions outside the course material

Keep a separate response record for every case and each of `no_rag`,
`naive_rag`, and `full_pipeline`. `responses.benchmark.example.json` shows the
expected shape. Record retrieval IDs, retrieved context text, citations, and an
explicit abstention decision; do not fill missing model outputs with
success-shaped defaults.

## Deterministic metrics and policy simulation

```powershell
.\.venv\Scripts\python.exe evals\run_benchmark.py evals\responses.json
.\.venv\Scripts\python.exe evals\simulate_learners.py --sessions 5
```

The benchmark defaults to requiring 100 cases. Use `--minimum-cases 1` only
for testing the file shape; never use it to claim benchmark compliance.
The deterministic metrics identify themselves as lexical proxies and include
context precision/recall, citation precision, expected-fact coverage, and
abstention accuracy. They cannot determine semantic entailment.

The simulator emits normalized gain, exact-question repetition rate, weak-topic
targeting rate, mastery-estimation MAE, and per-session mastery curves by
profile and condition. Slip, guess, forgetting, learning, and misconception
rates are explicit profile assumptions, not fitted learner parameters. Its
seed, profile definitions, number of sessions, and synthetic-only evidence
label must be retained with any report.

## RAGAS run with NVIDIA API Catalog

Use a dedicated evaluation environment and install only the optional pinned
evaluation dependencies:

```powershell
.\.venv\Scripts\python.exe -m venv .venv-eval
.\.venv-eval\Scripts\python.exe -m pip install -r requirements-eval.txt
.\.venv-eval\Scripts\python.exe -c "import ragas, datasets, pyarrow"
$env:NVIDIA_API_KEY = "..."
.\.venv-eval\Scripts\python.exe evals\prepare_pdf_ragas.py
.\.venv-eval\Scripts\python.exe evals\run_ragas.py evals\responses.json --allow-external-judge --output evals\ragas-report.json
```

`prepare_pdf_ragas.py` locally extracts PDF passages, sends page excerpts to
NVIDIA GPT-OSS-20B to draft 80 in-scope question candidates, adds 20 proposed
out-of-scope questions, then creates model-generated exploratory response
records for no-RAG and local lexical-retrieval baselines. Run preparation in the
application environment (it uses the app's PDF and NVIDIA client dependencies):

```powershell
.\.venv\Scripts\python.exe evals\prepare_pdf_ragas.py
```

It writes `evals\golden_rag.json`,
`evals\golden_rag.metadata.json`, and `evals\responses.json`. The candidate
metadata explicitly records `human_reviewed: false`; these are not a reviewed
gold set. Review and correct the questions, answers, passage links, and
out-of-scope labels before treating the evaluation as requirement evidence.
The generated response records are not runs of the deployed tutor. `no_rag`
uses a model response without retrieved context; `naive_rag` and
`full_pipeline` use local lexical top-1 and top-5 passage retrieval,
respectively. They are not production hybrid/Qdrant retrieval baselines, and
the report must not be described as a production-system comparison. Baseline
records are checkpointed after each five-case batch; rerunning the preparation
command validates and reuses compatible saved responses, then retries missing
ones without replacing completed records.

The default RAGAS runner refuses datasets explicitly marked unreviewed. To get
an exploratory result before review, opt in explicitly:

```powershell
.\.venv-eval\Scripts\python.exe evals\run_ragas.py evals\responses.json --cases evals\golden_rag.json --allow-external-judge --allow-unreviewed-candidates --output evals\ragas-report.json
```

Exploratory candidate scores must not be presented as reviewed benchmark
compliance. The checked-in `.example.json` files are schema examples only.
RAGAS scores semantic response/retrieval metrics on in-scope items, and reports
out-of-scope abstention accuracy separately. The runner uses bounded concurrency
and retries; if a metric task still fails, its missing case count and an
incomplete `evaluation_status` are recorded rather than silently treating the
failure as a score.

The app and evaluation environments must remain separate: the tested RAGAS
version pins a `langchain-core` range incompatible with this project's current
LangGraph runtime. The default minimum is 100 cases and the runner verifies the
out-of-material share is between 20% and 25%. NVIDIA's current API Catalog trial
endpoints require an NVIDIA API key and are subject to NVIDIA's trial terms,
quotas, and availability; do not assume unlimited or permanently free use.

The embedding adapter sends query text with `input_type=query` and reference or
retrieved passages with `input_type=passage`, as required by NVIDIA's
Nemotron-3-Embed-1B API. RAGAS data is mapped to the RAGAS 0.2.15 single-turn
columns (`user_input`, `response`, `retrieved_contexts`, `reference`, and
`reference_contexts`). The API base defaults to
`https://integrate.api.nvidia.com/v1`. Optional overrides are `RAGAS_MODEL`,
`RAGAS_BASE_URL`, and `RAGAS_EMBEDDING_MODEL`; use only compatible NVIDIA
endpoints and models. The evaluation report records model IDs, API base, UTC
time, case/response SHA-256 hashes, and aggregate metrics, but not the API key
or source text. Supply the source-corpus SHA-256 with `--corpus-sha256` or
`RAGAS_CORPUS_SHA256` to include it in the report. Explicit approval
(`--allow-external-judge`) is mandatory.

This process transfers evaluation questions, ground truths, answers, and
contexts to NVIDIA. Confirm that corpus licensing, organizational policy, and
any student-data consent permit that transfer before running it. Do not include
private student data in evaluation records.

Model and API references: [NVIDIA GPT-OSS 20B API Catalog](https://build.nvidia.com/openai/gpt-oss-20b),
[NVIDIA Nemotron-3-Embed-1B API Catalog](https://build.nvidia.com/nvidia/nemotron-3-embed-1b),
and [NVIDIA NeMo Retriever embedding API guidance](https://docs.nvidia.com/nim/nemo-retriever/embedding/latest/use-the-api-openai.html).

## Synthetic learner simulation

The supplied profile file is an executable synthetic setup only. Replace the
profiles and transition assumptions with documented, consented learner
parameters before interpreting these numbers as product efficacy. It is not a
substitute for a randomized or otherwise controlled learner study.
