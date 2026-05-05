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
from src.db_service.data_types import (
    ChatHistoryPair,
    ChatSessionRecord,
    RetrievedChunkRecord,
)
from src.db_service.postgres_controllers import PostgresChatService, PostgresChunkStore
from src.db_service.schemas import MAX_EMBEDDING_DIMENSION, SourceRecordTable
from tests.unit_tests.mock_ollama_client import MockOllamaClient

BDD_ENV_PATH = Path("tests/bdd_test/postgres_behaviour.env")
BDD_MODEL_CONFIG_PATH = Path("configs/model_config.bdd.toml")
BDD_SAMPLE_DATA_PATH = Path("data/samples/convfinqa_dev_sample.json")


def _load_env_file(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not path.exists():
        return

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        if key not in os.environ:
            monkeypatch.setenv(key, value)


def _postgres_url() -> str | None:
    return os.getenv("POSTGRES_BEHAVIOUR_URL") or os.getenv("POSTGRES_CONNECTION_URL")


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[3] / candidate


@scenario(
    "postgres_ingestion.feature",
    "ProcessLayer ingests a sample file into Postgres",
)
def test_process_layer_ingests_sample_file_into_postgres() -> None:
    """Run the Postgres sample ingestion scenario."""


@scenario(
    "postgres_ingestion.feature",
    "Chat session records messages after sample ingestion",
)
def test_chat_session_records_messages_after_sample_ingestion() -> None:
    """Run the Postgres chat session behaviour scenario."""


@given("a Postgres behaviour database is configured", target_fixture="database_url")
def database_url(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure the test process to use the behaviour Postgres database."""
    env_path = Path(os.getenv("BDD_ENV_PATH", BDD_ENV_PATH))
    env_path = _repo_path(env_path)
    _load_env_file(env_path, monkeypatch)

    if os.getenv("RUN_POSTGRES_BEHAVIOUR") != "1" or not _postgres_url():
        pytest.skip("Set RUN_POSTGRES_BEHAVIOUR=1 and POSTGRES_BEHAVIOUR_URL to run")

    monkeypatch.setenv("POSTGRES_CONNECTION_URL", _postgres_url() or "")
    monkeypatch.setenv(
        "MODEL_CONFIG_PATH",
        str(_repo_path(os.getenv("MODEL_CONFIG_PATH", str(BDD_MODEL_CONFIG_PATH)))),
    )
    monkeypatch.setenv(
        "RAW_DATA_PATH",
        str(_repo_path(os.getenv("RAW_DATA_PATH", str(BDD_SAMPLE_DATA_PATH)))),
    )
    get_settings.cache_clear()
    setup_db.main()
    return _postgres_url() or ""


@given(
    "a raw ConvFinQA file built from one sample record",
    target_fixture="sample_context",
)
def sample_context(tmp_path: Path) -> dict[str, object]:
    """Write a ProcessLayer-compatible raw input file from a sample record."""
    sample_path = _repo_path(os.getenv("BDD_SAMPLE_DATA_PATH", BDD_SAMPLE_DATA_PATH))
    sample_payload = json.loads(sample_path.read_text())
    split = sample_payload["split"]
    record = sample_payload["records"][0]
    raw_payload = {
        "train": [],
        "dev": [],
        "test": [],
    }
    raw_payload[split] = [record]
    raw_path = tmp_path / "one_record_convfinqa.json"
    raw_path.write_text(json.dumps(raw_payload))

    return {
        "raw_path": raw_path,
        "record_id": record["id"],
        "source_file": str(sample_path),
        "split": split,
    }


@when("I run the process layer ingestion", target_fixture="ingestion_result")
@given("the sample record has been ingested", target_fixture="ingestion_result")
def ingestion_result(
    database_url: str, sample_context: dict[str, object]
) -> dict[str, object]:
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


@when("I start a chat session for the sample record", target_fixture="chat_context")
def chat_context(
    database_url: str, ingestion_result: dict[str, object]
) -> dict[str, object]:
    """Create or resume a chat session for the ingested record."""
    record_id = ingestion_result["record_id"]
    assert isinstance(record_id, str)

    chat_service = PostgresChatService(create_engine(database_url))
    chat_session = chat_service.start_or_resume_session(record_id)

    return {
        "chat_service": chat_service,
        "chat_session": chat_session,
        "record_id": record_id,
    }


@when("I append a user question and assistant answer")
def append_chat_messages(chat_context: dict[str, object]) -> None:
    """Append one full user/assistant turn to the chat session."""
    chat_service = chat_context["chat_service"]
    chat_session = chat_context["chat_session"]

    assert isinstance(chat_service, PostgresChatService)
    assert isinstance(chat_session, ChatSessionRecord)

    user_message = chat_service.record_user_message(
        chat_session.session_id,
        "What was the important sample fact?",
    )
    chat_service.record_assistant_message(
        chat_session.session_id,
        '{"answer":"The sample was ingested.","citations":[]}',
        user_message,
    )


@then("chunks should be persisted for the sample record")
def chunks_should_be_persisted(
    ingestion_result: dict[str, object], sample_context: dict[str, object]
) -> None:
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

    with store.session_factory() as session:
        source_record = session.get(SourceRecordTable, record_id)
    assert source_record is not None
    assert source_record.record_id == record_id
    assert source_record.source_file == str(sample_context["raw_path"])
    assert source_record.split == sample_context["split"]


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


@then("the chat session should track both messages")
def chat_session_should_track_messages(chat_context: dict[str, object]) -> None:
    """Assert session metadata advances after appending messages."""
    chat_service = chat_context["chat_service"]
    chat_session = chat_context["chat_session"]
    record_id = chat_context["record_id"]

    assert isinstance(chat_service, PostgresChatService)
    assert isinstance(chat_session, ChatSessionRecord)
    assert isinstance(record_id, str)

    updated_session = chat_service.get_session(record_id)
    assert isinstance(updated_session, ChatSessionRecord)
    assert updated_session.session_id == chat_session.session_id
    assert updated_session.record_id == record_id
    assert updated_session.message_count >= chat_session.message_count + 2
    assert updated_session.last_message_index >= chat_session.last_message_index + 2


@then("chat history should contain the user answer pair")
def chat_history_should_contain_pair(chat_context: dict[str, object]) -> None:
    """Assert history returns Pydantic user/assistant message pairs."""
    chat_service = chat_context["chat_service"]
    chat_session = chat_context["chat_session"]

    assert isinstance(chat_service, PostgresChatService)
    assert isinstance(chat_session, ChatSessionRecord)

    history = chat_service.show_history(chat_session.session_id, limit=2)
    assert len(history) == 1
    assert isinstance(history[0], ChatHistoryPair)
    assert history[0].user_question.role == "user"
    assert history[0].user_question.content == "What was the important sample fact?"
    assert history[0].assistant.role == "assistant"
    assert history[0].assistant.content == (
        '{"answer":"The sample was ingested.","citations":[]}'
    )
