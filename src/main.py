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
    answer_service = context.answer_service

    if retriever is None:
        raise ValueError("Retreiever could not be built")

    if answer_service is None:
        raise ValueError("Could not build answer service")

    if not retriever.has_data():
        rich_print(
            "[yellow]Services are up, but no ingested data was found. Run the ingestion script first.[/yellow]"
        )
        raise typer.Exit(code=2)

    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break

        response = answer_service.answer(message)
        rich_print(f"[blue][bold]assistant:[/bold] {response.answer}[/blue]")
        if response.citations:
            rich_print(f"[dim]citations: {', '.join(response.citations)}[/dim]")


def show_history(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
):
    ...


if __name__ == "__main__":
    app()
