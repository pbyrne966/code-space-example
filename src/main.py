"""Main Typer app for ConvFinQA."""

import typer
from pydantic import ValidationError
from rich import print as rich_print

from src.data_types import ChatMessageRecord, ChatSessionRecord
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
    cached: ChatMessageRecord,
    record_id: str,
    chat_service: PostgresChatService,
) -> RagAnswer | None:
    try:
        response = RagAnswer.model_validate_json(cached.content)
        return response
    except ValidationError:
        logger.exception("Cached answer failed validation for record_id=%s", record_id)
        rich_print("[red]Cached answer was not valid. Please try again.[/red]")
        if cached.message_id is None:
            logger.warning("Could not invalidate cached answer without message_id")
            return None

        try:
            invalidated = chat_service.soft_delete(
                cached.message_id,
                cached.hashed_content,
            )
        except Exception:
            logger.exception(
                "Failed to invalidate cached answer for message_id=%s",
                cached.message_id,
            )
        else:
            if not invalidated:
                logger.warning(
                    "No cached answer invalidated for message_id=%s",
                    cached.message_id,
                )
        return None


def record_cached_answer(
    chat_service: PostgresChatService,
    message: str,
    session: ChatSessionRecord,
    cached: ChatMessageRecord,
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


def retirieve_fn(
    chat_service: PostgresChatService,
    message: str,
    session: ChatSessionRecord,
    answer_service: RAGService,
    record_id: str,
) -> RagAnswer | None:
    session_history = chat_service.show_history(session.session_id, limit=10)
    user_chat_exchange = chat_service.record_user_message(session.session_id, message)
    try:
        response = answer_service.answer(message, record_id, session_history)
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

    chat_service.record_assistant_message(
        session.session_id, response.model_dump_json(), user_chat_exchange
    )
    return response


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
) -> None:
    """Open an interactive RAG chat session."""
    context = build_context()
    retriever = context.retriever
    chat_service = context.chat_service
    answer_service = context.answer_service
    settings = context.settings

    if settings is None:
        rich_print("[red]Settings could not be built[/red]")
        raise typer.Exit(code=1)

    if retriever is None:
        rich_print("[red]Retreiever could not be built[/red]")
        raise typer.Exit(code=1)

    if chat_service is None:
        rich_print("[red]Could not build chat service[/red]")
        raise typer.Exit(code=1)

    if answer_service is None:
        rich_print("[red]Could not build answer service[/red]")
        raise typer.Exit(code=1)

    if not retriever.has_data():
        rich_print(
            "[yellow]Services are up, but no ingested data was found. Run the ingestion script first.[/yellow]"
        )
        raise typer.Exit(code=2)

    session = chat_service.start_or_resume_session(record_id)
    rich_print(f"[dim]chat session: {session.session_id} for record: {record_id}[/dim]")

    while True:
        message = input(">>> ")

        response: RagAnswer | None = None

        if message.strip().lower() in {"exit", "quit"}:
            break

        if settings.caching and (cached := chat_service.get_cached(message, record_id)):
            response = record_cached_answer(
                chat_service,
                message,
                session,
                cached,
                record_id,
            )

        if response is None:
            response = retirieve_fn(
                chat_service, message, session, answer_service, record_id
            )

        if response is None:
            continue

        rich_print(f"[blue][bold]assistant:[/bold] {response.answer}[/blue]")
        if response.turn_program:
            rich_print(f"[dim]turn program: {response.turn_program}[/dim]")
        if response.citations:
            rich_print(f"[dim]citations: {', '.join(response.citations)}[/dim]")


if __name__ == "__main__":
    app()
