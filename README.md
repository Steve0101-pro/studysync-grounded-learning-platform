# StudySync

StudySync is a source-grounded study companion. It turns course PDFs,
PowerPoint decks, and lecture videos into searchable material, answers learner
questions with citations to that material, offers source-based practice, and
summarizes practice activity.

The project has two runnable applications:

- **Streamlit** provides the learner workspace.
- **FastAPI** provides authentication, ingestion, retrieval, tutor, practice,
  and progress APIs.

StudySync is an evolving prototype. Some capabilities described below are
implemented while larger-scale ingestion, validated learning outcomes, and
production performance remain to be established. The app labels practice
signals as revision aids rather than validated mastery measurements.

## Contents

- [How it works](#how-it-works)
- [Learner workflow](#learner-workflow)
- [Technology](#technology)
- [Requirements](#requirements)
- [Local setup](#local-setup)
- [Configuration](#configuration)
- [Run StudySync](#run-studysync)
- [Use the app](#use-the-app)
- [API overview](#api-overview)
- [Evaluation](#evaluation)
- [Tests and checks](#tests-and-checks)
- [Known limits](#known-limits)
- [Troubleshooting](#troubleshooting)
- [Repository map](#repository-map)

## How it works

```text
PDF / PowerPoint / lecture video
              │
              ▼
    Extract text and locations
    PDF pages · slide numbers · video timestamps
              │
              ▼
    Build parent and child passages
              │
       ┌──────┴──────┐
       ▼             ▼
   PostgreSQL     Qdrant vectors
       │             │
       └──────┬──────┘
              ▼
   BM25 + vector retrieval
              │
              ▼
    Grounded tutor and citations
              │
       ┌──────┴──────┐
       ▼             ▼
   Practice       Progress view
       └──────┬──────┘
              ▼
       Evaluation tools
```

The API stores user-owned sources and passages in PostgreSQL. Qdrant provides
semantic retrieval; a lexical search path remains available if vector search
is unavailable. Tutor requests retrieve course evidence first. The model is
instructed to answer from that evidence and cite it. Citation markers are
validated before a generated response is accepted. If evidence is missing,
or configured models cannot return a valid cited answer, the tutor reports the
gap or shows a source excerpt instead of presenting an uncited answer as
grounded.

## Learner workflow

1. **Create an account or sign in.** Protected API endpoints use the signed-in
   learner's identity to scope sources, conversations, quizzes, and progress.
2. **Add course materials.** Upload a PDF, PowerPoint (`.pptx`), MP4, or WebM
   file from the Study workspace. Ingestion extracts text and preserves page,
   slide, or transcript timestamp locations.
3. **Ask a question.** The tutor searches uploaded material and streams an
   answer. Each answer's citations identify the supporting source location.
   Conversation history supports follow-up questions.
4. **Use optional media in context.** A learner can record a question for
   transcription, attach an image or figure, or ask about an uploaded lecture
   video. These inputs feed the course tutor rather than replacing course
   retrieval with an unrelated response.
5. **Practice.** Generate a question from a source passage, submit an answer,
   and review the supporting passage and feedback.
6. **Review progress.** The dashboard shows source totals and practice
   outcomes. Accuracy is based on answered practice questions, not uploaded
   resources. Small samples are shown with a confidence interval; topic
   estimates are explicitly not calibrated mastery scores.
7. **Inspect evaluation.** The RAGAS Evaluation page presents the saved
   evaluation report and its status. It is clearly marked exploratory when
   cases are machine-generated or model responses are not actual deployed
   tutor runs.

## Technology

- **Python**, **FastAPI**, and **Pydantic** for the API and validation.
- **Streamlit** for the learner and evaluation interfaces.
- **PostgreSQL**, **SQLAlchemy**, and **Alembic** for persistent data and
  schema migrations.
- **Qdrant** and **FastEmbed** for vector indexing and semantic search.
- **BM25-style lexical retrieval** for keyword matching and retrieval fallback.
- **LangGraph** for the study-tutor flow.
- **NVIDIA API Catalog** for configured tutor and vision models.
- **Groq Whisper** for speech transcription; a configured Gemini model can
  provide transcription fallback.
- **Supabase Storage** for configured document storage.
- **Redis** for optional cache and related runtime services.
- **Neo4j** for optional concept and source-term indexing.
- **RAGAS** in a separate optional evaluation environment.

Optional services may be omitted in development when their relevant workflow
supports a local or graceful fallback. The database and model credentials
needed for the workflows you intend to use must still be configured.

## Requirements

- Windows 10/11, macOS, or Linux.
- Python 3.11 or later (Python 3.12 is recommended).
- PostgreSQL, local or managed.
- API credentials for at least one supported tutor provider. NVIDIA is the
  first configured tutor provider; Groq and OpenAI are optional fallbacks.
- Additional credentials only for the features you use: Supabase Storage,
  Groq/Gemini transcription, Qdrant Cloud, Redis, and Neo4j.
- Tesseract OCR is optional for scanned documents. Set `TESSERACT_CMD` if it
  is installed outside `PATH`.

## Local setup

Open PowerShell in the repository root.

### 1. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, use the environment's executable
directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Edit `.env` with local development credentials and connection strings. Keep
`.env` private: do not commit it, paste its contents into issue trackers, or
include API keys in logs.

### 3. Prepare the database

Create an empty PostgreSQL database, set `DATABASE_URL` in `.env`, and apply
the schema:

```powershell
.\.venv\Scripts\alembic.exe upgrade head
```

For development, the API connects to the database specified by
`DATABASE_URL`. Use the same database for migrations and runtime.

## Configuration

The full variable list and safe placeholders are in
[`.env.example`](./.env.example). Key settings:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string. |
| `JWT_SECRET_KEY` | Secret used to sign authentication tokens. |
| `NVIDIA_API_KEY` | Enables the configured NVIDIA tutor and vision models. |
| `NVIDIA_BASE_URL` | NVIDIA API Catalog OpenAI-compatible endpoint. |
| `STUDY_TUTOR_NVIDIA_MODEL` | First NVIDIA model attempted for grounded tutoring. The tested default is `nvidia/nemotron-3-super-120b-a12b`. |
| `NVIDIA_PRIMARY_MODEL` | Additional NVIDIA reasoning model used by tutor and multimodal workflows. |
| `NVIDIA_FALLBACK_MODEL` | Additional NVIDIA reasoning fallback. |
| `GROQ_API_KEY` | Enables Groq transcription and optional chat fallback. |
| `GROQ_FALLBACK_MODEL` | Groq chat model used for tutor fallback. |
| `GEMINI_API_KEY` | Enables Gemini-based fallback features where configured. |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | Enable Supabase-backed storage. |
| `QDRANT_URL`, `QDRANT_API_KEY` | Configure Qdrant Cloud; otherwise the app may use local Qdrant storage. |
| `REDIS_URL` | Optional Redis connection. Caching can be disabled if Redis is unavailable. |
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` | Optional graph indexing. |
| `CORS_ORIGINS` | Comma-separated browser origins allowed by the API. |
| `MODEL_*_TIMEOUT_SECONDS` | Bound external model connection and response time. |
| `MODEL_MAX_RETRIES` | Retry count for transient model errors. |

Model settings can be overridden with environment variables. Restart the API
after changing `.env`; the running process does not reload configuration
automatically.

## Run StudySync

Start the API in one PowerShell window:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Start the interface in a second PowerShell window:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

- StudySync: <http://127.0.0.1:8501/>
- API documentation: <http://127.0.0.1:8000/docs>
- API health: <http://127.0.0.1:8000/health/full>

If the API is not on `http://localhost:8000`, set `API_URL` in the UI process
environment before starting Streamlit.

## Use the app

### Add a course source

Sign in, open the Study workspace, and upload a PDF, PowerPoint, or lecture
video. The application extracts available text, builds searchable passages,
and retains their source locations. For scanned pages, configure Tesseract
OCR. Image understanding depends on a working configured vision provider.

### Ask the tutor

Ask a question about material in the library. The tutor retrieves matching
passages and uses citations such as `[1]` to link claims to evidence. Select
Adaptive, Hints first, or Direct explanation from the tutor controls.

Adaptive behavior uses recent practice history as a rough scaffolding signal.
For a cold-start learner, it begins with hints-first behavior; this is not a
validated diagnostic or mastery measure. If a provider fails or returns
invalid citations, the app safely falls back to the retrieved source excerpt.
That fallback means model generation did not produce an acceptable cited
answer; it is not an indication that the source passage is missing.

Audio recordings are transcribed before tutor retrieval. Image input provides
visual context for retrieval and the grounded tutor response. Uploaded video
supports timestamped transcript retrieval; video keyframe indexing is not
currently implemented.

### Practice and dashboard

Practice questions are currently source-passage selection questions. The
dashboard calculates observed quiz accuracy from correct answers divided by
answered questions. Uploading a source does not change this value. Sparse
samples include a Wilson confidence interval to make uncertainty visible.

## API overview

The interactive OpenAPI page at `/docs` is authoritative for request and
response schemas. Main route groups include:

| Route | Purpose |
|---|---|
| `/auth/*` | Registration, login, and account recovery. |
| `/study/courses` | List the signed-in learner's course library. |
| `/study/sources` | Upload and list course sources. |
| `/study/search` | Search indexed source passages. |
| `/study/tutor` | Request a non-streaming source-grounded response. |
| `/study/tutor/stream` | Stream a tutor turn and its citation metadata. |
| `/study/tutor/transcribe` | Transcribe a learner's recorded question. |
| `/study/conversations` | List saved study conversations. |
| `/study/quizzes` | Create source-based practice questions. |
| `/study/progress` | Read source and practice progress. |
| `/health/full` | Check API and configured service health. |

Protected routes require the bearer token returned by authentication. The
Streamlit application adds it automatically after sign-in.

## Evaluation

### RAGAS report in the UI

Open **RAGAS evaluation** in the workspace navigation to inspect
`evals/ragas-report.json`. The view compares faithfulness, answer relevancy,
context precision, and context recall across available response sets. It also
shows valid and failed metric counts, model identifiers, and the run status.

The current report is exploratory. Its cases are machine-generated and not
human-reviewed, and response sets labeled “full pipeline” are not verified
live runs of the deployed tutor. The displayed scores must not be described as
human-reviewed benchmark results or proof of compliance.

### Run the evaluator

External RAGAS dependencies are optional and intentionally isolated from the
application environment:

```powershell
.\.venv-eval\Scripts\python.exe evals\run_ragas.py --help
```

See [`evals/README.md`](./evals/README.md) for the complete setup, dataset
schema, consent gate, baseline-generation process, and evaluation command.
The external judge sends evaluation questions, answers, references, and
contexts to the configured provider. Confirm authorization and remove private
learner data before running it.

The repository also includes `evals/run_benchmark.py` for lexical proxy
comparisons; these are not RAGAS scores. Synthetic learner simulation is for
exploring study-policy behavior and is not evidence of real learner outcomes.

## Tests and checks

Run all automated tests from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

GitHub Actions runs this test suite on pushes to `main` and pull requests.
The workflow needs no cloud or model credentials.

Run a focused test file:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_study_tutor_stream.py -q
```

Apply migrations after updating the code:

```powershell
.\.venv\Scripts\alembic.exe upgrade head
```

## Known limits

- Course-source ingestion currently processes uploads synchronously and is
  limited to 25 MB. Resumable background jobs and large-course throughput have
  not been demonstrated.
- Video transcripts are indexed with timestamps, but keyframe extraction and
  indexing are not implemented.
- Assessment currently focuses on source-passage selection; a full exam
  blueprint, numerical grading, short-answer rubric grading, and cited
  post-exam reports remain future work.
- Topic practice signals are heuristic and are not calibrated measures of
  student competence.
- The saved RAGAS candidate dataset has not been human-reviewed. The reported
  baselines are exploratory and do not represent verified production runs.
- Accessibility, mobile behavior, production load, and latency targets require
  additional measurement before making claims about them.

## Troubleshooting

### The tutor says the model is unavailable

The tutor displays a source excerpt when every configured model fails or
returns an answer with invalid citations. Check the API process logs for
provider and error type, then verify:

1. `NVIDIA_API_KEY` is valid and has access to the selected model.
2. `NVIDIA_PRIMARY_MODEL` and `NVIDIA_FALLBACK_MODEL` are enabled in the
   provider account.
3. `GROQ_API_KEY` is valid if relying on Groq as an additional fallback.
4. The API was restarted after editing `.env`.
5. The provider is not returning quota, availability, or rate-limit errors.

Never print or share the secret key when debugging. A valid key cannot prevent
temporary provider overload; configure a second working model/provider.

### Retrieval returns no relevant passage

Confirm the source appears in **My library**, wait for its upload request to
finish, and search for a distinctive phrase from the material. Verify that
PostgreSQL is reachable and inspect API logs for Qdrant errors. Lexical
retrieval can continue if vector search is unavailable, but it may return
different results.

### Redis, Neo4j, or Qdrant is unavailable

These services are optional for some local workflows. Check `/health/full`
and confirm the matching connection settings. Redis cache unavailability
does not itself indicate that PostgreSQL or tutor providers are unavailable.
Local Qdrant storage is used when cloud Qdrant is not configured.

## Repository map

```text
app/
  api/                 FastAPI routers and workflow endpoints
  dependencies/        Authentication and request dependencies
  models/              Database models
  schemas/             Request and response validation
  services/            Ingestion, retrieval, tutoring, AI providers
alembic/               Database migrations
evals/                 RAGAS, smoke tests, and benchmark tooling
tests/                 Automated test suite
streamlit_app.py       StudySync learner interface
requirements.txt       Application dependencies
.env.example           Safe configuration template
docker-compose.yml     Local container orchestration
```

## License and data

Review the repository's license and the licenses of course materials before
redistributing or sending content to external model providers. Keep API
credentials and private learner data out of source control and evaluation
fixtures.
