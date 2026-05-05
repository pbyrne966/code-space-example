"""Main Typer app for ConvFinQA."""

import typer
from rich import print as rich_print

from src.logger import get_logger
from src.runtime import build_context

logger = get_logger("typer_logger")

app = typer.Typer(
    name="main",
    help="CLI for ConvFinQA chat workflows",
    add_completion=True,
)


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
) -> None:
    """Open an interactive RAG chat session."""
    context = build_context()
    retriever = context.retriever
    chat_service = context.chat_service
    answer_service = context.answer_service

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

        if message.strip().lower() in {"exit", "quit"}:
            break

        chat_service.record_user_message(session.session_id, message)
        response = answer_service.answer(message)
        rich_print(f"[blue][bold]assistant:[/bold] {response.answer}[/blue]")
        if response.citations:
            rich_print(f"[dim]citations: {', '.join(response.citations)}[/dim]")

        chat_service.record_assistant_message(
            session.session_id,
            response.model_dump_json(),
        )


if __name__ == "__main__":
    app()
