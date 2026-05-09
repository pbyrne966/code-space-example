"""Main Typer app for ConvFinQA."""

import typer
from pydantic import ValidationError
from rich import print as rich_print

from src.config import Settings
from src.data_types import CachedAnswerRecord, ChatHistoryPair, ChatSessionRecord
from src.db_service.postgres_controllers import PostgresChatService
from src.logger import get_logger
from src.rag_service import RagAnswer, RAGService
from src.runtime import build_context

logger = get_logger("typer_logger")

app = typer.Typer(
    name="main",
    help="CLI for ConvFinQA chat workflows",
    add_completion=True,
)


def validate_cached_answer(
    cached: CachedAnswerRecord,
    record_id: str,
    chat_service: PostgresChatService,
) -> RagAnswer | None:
    try:
        response = RagAnswer.model_validate_json(cached.content)
        return response
    except ValidationError:
        logger.exception("Cached answer failed validation for record_id=%s", record_id)
        rich_print("[red]Cached answer was not valid. Please try again.[/red]")
        if cached.cache_id is None:
            logger.warning("Could not invalidate cached answer without cache_id")
            return None

        try:
            invalidated = chat_service.invalidate_cached_answer(
                cached.cache_id,
                cached.hashed_content,
            )
        except Exception:
            logger.exception(
                "Failed to invalidate cached answer for cache_id=%s",
                cached.cache_id,
            )
        else:
            if not invalidated:
                logger.warning(
                    "No cached answer invalidated for cache_id=%s",
                    cached.cache_id,
                )
        return None


def record_cached_answer(
    chat_service: PostgresChatService,
    message: str,
    session: ChatSessionRecord,
    cached: CachedAnswerRecord,
    record_id: str,
) -> RagAnswer | None:
    response = validate_cached_answer(cached, record_id, chat_service)
    if response is None:
        return None

    user_chat_exchange = chat_service.record_user_message(session.session_id, message)
    chat_service.record_assistant_message(
        session.session_id,
        cached.content,
        user_chat_exchange,
    )
    return response


def retrieve_fn(
    message: str,
    session_history: list[ChatHistoryPair],
    answer_service: RAGService,
    record_id: str,
    is_requery: bool = False,
) -> RagAnswer | None:
    try:
        response = answer_service.answer(
            message,
            record_id,
            session_history=session_history,
            is_requery=is_requery,
        )
    except ValidationError:
        logger.exception("Model answer failed validation for record_id=%s", record_id)
        rich_print(
            "[red]The model returned an invalid answer format. Please try again.[/red]"
        )
        return None
    except Exception:
        logger.exception("Retrieval failed for record_id=%s", record_id)
        rich_print(
            "[red]Could not retrieve an answer for that question. "
            "Please try again.[/red]"
        )
        return None

    return response


def process_question(
    message: str,
    record_id: str,
    session: ChatSessionRecord,
    settings: Settings,
    chat_service: PostgresChatService,
    answer_service: RAGService,
) -> RagAnswer | None:
    if settings.caching and (cached := chat_service.get_cached(message, record_id)):
        response = record_cached_answer(
            chat_service,
            message,
            session,
            cached,
            record_id,
        )
        if response is not None:
            return response

    user_chat_exchange = chat_service.record_user_message(session.session_id, message)
    session_history = chat_service.show_history(session.session_id, limit=10)

    response = retrieve_fn(message, session_history, answer_service, record_id)

    if response is not None and response.requery is not None:
        requery = response.requery
        response = retrieve_fn(
            requery,
            session_history,
            answer_service,
            record_id,
            is_requery=True,
        )
        if response is not None:
            response = response.model_copy(update={"requery": requery})

    if response is None:
        return None

    answer_content = response.model_dump_json()
    chat_service.record_assistant_message(
        session.session_id, answer_content, user_chat_exchange
    )
    if settings.caching:
        chat_service.cache_answer(message, record_id, answer_content)
    return response


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
) -> None:
    """Open an interactive RAG chat session."""
    context = build_context()
    chat_service = context.chat_service
    answer_service = context.answer_service
    settings = context.settings
    session = chat_service.start_or_resume_session(record_id)

    rich_print(f"[dim]chat session: {session.session_id} for record: {record_id}[/dim]")

    while True:
        message = input(">>> ")
        if message.strip().lower() in {"exit", "quit"}:
            break

        response: RagAnswer | None = process_question(
            message,
            record_id,
            session,
            settings,
            chat_service,
            answer_service,
        )

        if response is None:
            rich_print("[red]Chat Error could not process response.[/red]")
        else:
            rich_print(f"[blue][bold]assistant:[/bold] {response.answer}[/blue]")
            if response.turn_program:
                rich_print(f"[dim]turn program: {response.turn_program}[/dim]")
            if response.citations:
                rich_print(f"[dim]citations: {', '.join(response.citations)}[/dim]")


if __name__ == "__main__":
    app()
