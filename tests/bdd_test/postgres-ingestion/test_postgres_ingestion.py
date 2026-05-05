import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from pytest_bdd import given, scenario, then, when
from sqlalchemy import create_engine

from scripts import setup_db
from src.chunking_service.data_loader import ProcessLayer
from src.config.settings import get_settings
from src.db_service.data_types import RetrievedChunkRecord
from src.db_service.postgres_controllers import PostgresChunkStore
from src.db_service.schemas import MAX_EMBEDDING_DIMENSION
from tests.unit_tests.mock_ollama_client import MockOllamaClient


def _postgres_url() -> str | None:
    return os.getenv("POSTGRES_BEHAVIOUR_URL") or os.getenv(
        "POSTGRES_CONNECTION_URL"
    )


@scenario(
    "postgres_ingestion.feature",
    "ProcessLayer ingests a sample file into Postgres",
)
def test_process_layer_ingests_sample_file_into_postgres() -> None:
    """Run the Postgres sample ingestion scenario."""


@given("a Postgres behaviour database is configured", target_fixture="database_url")
def database_url(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure the test process to use the behaviour Postgres database."""
    if os.getenv("RUN_POSTGRES_BEHAVIOUR") != "1" or not _postgres_url():
        pytest.skip(
            "Set RUN_POSTGRES_BEHAVIOUR=1 and POSTGRES_BEHAVIOUR_URL to run"
        )

    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("POSTGRES_CONNECTION_URL", _postgres_url() or "")
    monkeypatch.setenv("MODEL_CONFIG_PATH", str(repo_root / "configs/model_config.toml"))
    monkeypatch.setenv("RAW_DATA_PATH", str(repo_root / "data/convfinqa_dataset.json"))
    get_settings.cache_clear()
    setup_db.main()
    return _postgres_url() or ""


@given(
    "a raw ConvFinQA file built from one sample record",
    target_fixture="sample_context",
)
def sample_context(tmp_path: Path) -> dict[str, object]:
    """Write a ProcessLayer-compatible raw input file from a sample record."""
    sample_path = Path("data/samples/convfinqa_train_sample.json")
    sample_payload = json.loads(sample_path.read_text())
    record = sample_payload["records"][0]
    raw_payload = {
        "train": [record],
        "dev": [],
        "test": [],
    }
    raw_path = tmp_path / "one_record_convfinqa.json"
    raw_path.write_text(json.dumps(raw_payload))

    return {
        "raw_path": raw_path,
        "record_id": record["id"],
    }


@when("I run the process layer ingestion", target_fixture="ingestion_result")
def ingestion_result(database_url: str, sample_context: dict[str, object]) -> dict[str, object]:
    """Run ProcessLayer against a real Postgres-backed chunk store."""
    model_client = MockOllamaClient(model_name=f"behaviour-model-{uuid4()}")
    store = PostgresChunkStore(
        engine=create_engine(database_url),
        embedding_fn=model_client.embed,
        embedding_model=model_client.get_config().model_name,
        top_k=3,
    )
    process_layer = ProcessLayer(
        db_service=store,
        raw_file_src=sample_context["raw_path"],  # type: ignore[arg-type]
        model_client=model_client,
    )

    chunks = process_layer.process()
    return {
        "chunks": chunks,
        "model_client": model_client,
        "record_id": sample_context["record_id"],
        "store": store,
    }


@then("chunks should be persisted for the sample record")
def chunks_should_be_persisted(ingestion_result: dict[str, object]) -> None:
    """Assert the chunks written through ProcessLayer can be read back."""
    store = ingestion_result["store"]
    record_id = ingestion_result["record_id"]
    chunks = ingestion_result["chunks"]

    assert isinstance(store, PostgresChunkStore)
    assert isinstance(record_id, str)
    assert isinstance(chunks, list)
    assert chunks

    stored_chunks = store.get_chunks_for_record(record_id)
    assert stored_chunks == chunks


@then("vector retrieval should return persisted chunks")
def vector_retrieval_should_return_chunks(ingestion_result: dict[str, object]) -> None:
    """Assert pgvector retrieval returns Pydantic retrieval records."""
    store = ingestion_result["store"]
    chunks = ingestion_result["chunks"]

    assert isinstance(store, PostgresChunkStore)
    assert isinstance(chunks, list)

    results = store.retrieve(chunks[0].text)
    persisted_chunk_ids = {chunk.chunk_id for chunk in chunks}

    assert results
    assert all(isinstance(result, RetrievedChunkRecord) for result in results)
    assert results[0].chunk.chunk_id in persisted_chunk_ids
    assert len(results[0].chunk.text) > 0
    assert store.embedding_fn is not None
    assert len(store.embedding_fn(chunks[0].text)) == MAX_EMBEDDING_DIMENSION
