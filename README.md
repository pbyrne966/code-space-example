# ConvFinQA RAG

Retrieval-augmented question answering for ConvFinQA-style financial records.

The application ingests cleaned ConvFinQA records, chunks each financial report
into text and table evidence, stores embeddings in Postgres with pgvector, and
answers record-scoped financial questions through an Ollama/Qwen model. Answers
are validated as structured JSON and arithmetic is executed deterministically
from model-selected table values.

## Architecture

- `src/chunking_service`: converts records into retrieval chunks and structured
  table-value candidates.
- `src/db_service`: owns Postgres schemas, chunk storage, chat history, and the
  decoupled answer cache.
- `src/rag_service.py`: retrieves context, builds prompts, validates model JSON,
  and executes calculation programs.
- `src/model_service`: wraps the model backend.
- `src/main.py`: CLI chat flow, cache replay, retrieval, requery, and history
  recording.

Chat history and answer caching are intentionally separate. Cache hits are read
from `answer_cache`, validated, and then replayed as fresh chat exchanges so the
conversation history remains an ordered record of what the user actually saw.

## Setup

Create a local environment and install dependencies:

```bash
uv sync
```

Start Postgres/pgvector and Ollama:

```bash
docker compose up -d postgres ollama
```

Create database tables:

```bash
uv run python scripts/setup_db.py
```

Ingest records:

```bash
uv run python scripts/ingest_convfinqa.py
```

Open a chat session for a record:

```bash
uv run main chat <record_id>
```

## Configuration

Runtime settings come from environment variables or `.env`:

- `MODEL_CONFIG_PATH`
- `RAW_DATA_PATH`
- `INGESTION_FILE_PATHS`, optional comma-separated list of ConvFinQA-shaped files to ingest
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_CONNECTION_URL`, optional SQLAlchemy URL override
- `CACHING_ON`, enables answer cache reuse

Model settings live in `configs/model_config.toml`.

## Quality Gates

```bash
uv run pytest tests/unit_tests
uv run mypy src
uv run ruff check src tests scripts
```

The heavier BDD/integration suites are opt-in because they require external
Postgres/Ollama services:

- `RUN_POSTGRES_BEHAVIOUR=1`
- `RUN_FULL_STACK_RAG_BDD=1`
- `RUN_MODEL_INTEGRATION=1`
