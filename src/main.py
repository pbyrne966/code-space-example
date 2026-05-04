"""
Main typer app for ConvFinQA
"""

import stat

import typer
from rich import print as rich_print
from sqlalchemy import create_engine

from src.chunking_service.data_loader import ProcessLayer
from src.config.settings import get_settings, Settings
from src.model_service.models import BaseModelFactory, ModelClient
from src.db_service.postgres_chunk_store import PostgresChunkStore
from src.db_service.schemas import PgVectorRetriever
from sqlalchemy.engine import Engine
from src.logger import get_logger
from dataclasses import dataclass
from typing import Optional


@dataclass
class AppState:
    app_settings: Optional[Settings] = None
    qwen_model_client: Optional[ModelClient] = None
    db_engine: Optional[Engine] = None
    retrieval_service: Optional[PgVectorRetriever] = None


state = AppState()

logger = get_logger("typer_logger")


def ensure_app_ready(state: AppState):
    attrs = [
        state.app_settings,
        state.qwen_model_client,
        state.db_engine,
        state.retrieval_service,
    ]
    return all((value is not None for value in attrs))


app = typer.Typer(
    name="main",
    help="Boilerplate app for ConvFinQA",
    add_completion=True,
)


@app.command()
def startup() -> str:
    """Start the application without requiring a subcommand."""
    logger.info("Starting Up App")
    app_settings = get_settings()
    logger.info("Initilising Qwen")
    model_client = BaseModelFactory.create(
        app_settings.model_type, app_settings.model_config_path
    )
    logger.info("Initilising Db Engine")
    db_engine = create_engine(app_settings.database_url)
    db_service = PostgresChunkStore(db_engine)

    logger.info("Processing Data Has Begun")
    ProcessLayer(
        db_service=db_service,
        raw_file_src=app_settings.raw_data_path,
        model_client=model_client,
    ).process()
    logger.info("Processing Data Has Finished")

    logger.info("Retreival Service Ready")
    retrevial_service = PgVectorRetriever(
        db_engine,
        embedding_fn=model_client.embed,
        embedding_model=model_client.get_config().model_name,
    )

    state.app_settings = app_settings
    state.qwen_model_client = model_client
    state.db_engine = db_engine
    state.retrieval_service = retrevial_service

    return "Processed"


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
) -> None:
    """Ask questions about a specific record"""
    if not ensure_app_ready(state):
        logger.info("App is not ready please run start up")
        return

    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break

        # TODO: YOUR CODE HERE
        response = "RESPONSE"
        rich_print(f"[blue][bold]assistant:[/bold] {response}[/blue]")


if __name__ == "__main__":
    app()
