FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev --no-editable

COPY src ./src
COPY tests ./tests
COPY configs ./configs
COPY scripts ./scripts
COPY data ./data

ENV PYTHONPATH=/app

CMD ["uv", "run", "python", "scripts/ingest_convfinqa.py"]
