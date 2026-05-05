import os
from pathlib import Path

import pytest
from pytest_bdd import given, scenario, then, when
from sqlalchemy import func, select

from scripts import setup_db
from src.chunking_service.data_loader import ProcessLayer
from src.config.settings import get_settings
from src.db_service.schemas import SourceRecordTable
from src.rag_service import RagAnswer
from src.runtime import AppState, build_context

FULL_STACK_ENV_PATH = Path("tests/bdd_test/full_stack_rag.env")


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[3] / candidate


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


@scenario(
    "full_stack_rag.feature",
    "Application answers for a random ingested record",
)
def test_application_answers_for_random_ingested_record() -> None:
    """Run the full-stack RAG answering scenario."""


@given("the configured full application stack is loaded", target_fixture="app_context")
def app_context(monkeypatch: pytest.MonkeyPatch) -> AppState:
    """Build the real runtime context from configured environment values."""
    env_path = Path(os.getenv("FULL_STACK_RAG_ENV_PATH", FULL_STACK_ENV_PATH))
    _load_env_file(_repo_path(env_path), monkeypatch)

    if os.getenv("RUN_FULL_STACK_RAG_BDD") != "1":
        pytest.skip("Set RUN_FULL_STACK_RAG_BDD=1 to run full-stack RAG BDD")

    get_settings.cache_clear()
    setup_db.main()
    get_settings.cache_clear()
    return build_context()


@given("the configured ConvFinQA data has been ingested")
def configured_data_ingested(app_context: AppState) -> None:
    """Ensure configured raw data is present in the configured Postgres database."""
    settings = app_context.settings
    client = app_context.client
    retriever = app_context.retriever

    if settings is None or client is None or retriever is None:
        raise AssertionError("Full application context was not built")

    if retriever.has_data():
        return

    ProcessLayer(
        db_service=retriever,
        raw_file_src=settings.raw_data_path,
        model_client=client,
    ).process()


@when("I ask a question for a random ingested record", target_fixture="rag_result")
def rag_result(app_context: AppState) -> dict[str, object]:
    """Pick one ingested record at random and ask through the real answer service."""
    answer_service = app_context.answer_service
    db_engine = app_context.db_engine
    retriever = app_context.retriever

    if answer_service is None or db_engine is None or retriever is None:
        raise AssertionError("Full application context was not built")

    with retriever.session_factory() as session:
        record_id = session.execute(
            select(SourceRecordTable.record_id).order_by(func.random()).limit(1)
        ).scalar_one()

    question = "What financial information is available for this record?"
    answer = answer_service.answer(question, record_id)
    return {
        "answer": answer,
        "question": question,
        "record_id": record_id,
    }


@then("the application should return a RAG answer")
def application_should_return_rag_answer(rag_result: dict[str, object]) -> None:
    """Assert the real RAG path returned a parsed answer for the selected record."""
    answer = rag_result["answer"]
    record_id = rag_result["record_id"]

    assert isinstance(record_id, str)
    assert isinstance(answer, RagAnswer)
    assert answer.answer.strip()
    assert isinstance(answer.citations, list)
