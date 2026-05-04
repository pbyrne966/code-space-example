"""
Main typer app for ConvFinQA
"""

import typer
from rich import print as rich_print
from sqlalchemy import create_engine

from chunking_service.data_loader import ProcessLayer
from config.settings import get_settings
from model_service.models import BaseModelFactory
from src.db_service.postgres_chunk_store import PostgresChunkStore
from src.db_service.schemas import PgVectorRetriever

app_settings = get_settings()
qwen_model_client = BaseModelFactory.create(
    app_settings.model_type, app_settings.model_config_path
)
db_engine = create_engine(app_settings.database_url)
db_service = PostgresChunkStore(db_engine)

ProcessLayer(
    db_service=db_service,
    raw_file_src=app_settings.raw_data_path,
    model_client=qwen_model_client,
)

retrevial_service = PgVectorRetriever(
    db_engine,
    embedding_fn=qwen_model_client.embed,
    embedding_model=qwen_model_client.get_config().model_name,
)

app = typer.Typer(
    name="main",
    help="Boilerplate app for ConvFinQA",
    add_completion=True,
    no_args_is_help=True,
)


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
) -> None:
    """Ask questions about a specific record"""
    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break

        # TODO: YOUR CODE HERE
        response = "RESPONSE"
        rich_print(f"[blue][bold]assistant:[/bold] {response}[/blue]")


if __name__ == "__main__":
    app()
