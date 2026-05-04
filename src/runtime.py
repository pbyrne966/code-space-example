"""Shared runtime builders for ConvFinQA CLI entrypoints."""

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.config.settings import Settings, get_settings
from src.db_service.schemas import PgVectorRetriever
from src.logger import get_logger
from src.model_service.models import BaseModelFactory, ModelClient
from src.rag_service import RAGService

logger = get_logger("typer_logger")


@dataclass
class AppState:
    settings: Optional[Settings] = None
    client: Optional[ModelClient] = None
    db_engine: Optional[Engine] = None
    retriever: Optional[PgVectorRetriever] = None
    answer_service: Optional[RAGService] = None


def load_config() -> Settings:
    """Load and validate runtime settings from the environment."""
    logger.info("Loading application settings")
    return get_settings()


def build_client(settings: Settings) -> ModelClient:
    """Create the configured client."""
    logger.info("Initialising client")
    return BaseModelFactory.create(settings.model_type, settings.model_config_path)


def build_db_engine(settings: Settings) -> Engine:
    """Create the SQLAlchemy engine for the configured database."""
    logger.info("Initialising database engine")
    return create_engine(settings.database_url)


def build_retriever(db_engine: Engine, client: ModelClient) -> PgVectorRetriever:
    """Create the vector retriever used by chat."""
    logger.info("Initialising retriever")
    return PgVectorRetriever(
        db_engine,
        embedding_fn=client.embed,
        embedding_model=client.get_config().model_name,
    )


def build_answer_service(
    client: ModelClient, retriever: PgVectorRetriever
) -> RAGService:
    """Create the answer service from the client and retriever."""
    logger.info("Initialising answer service")
    return RAGService(model_client=client, retriever=retriever)


def build_context() -> AppState:
    """Build the runtime connections shared by CLI commands."""
    settings = load_config()
    client = build_client(settings)
    db_engine = build_db_engine(settings)
    retriever = build_retriever(db_engine, client)
    answer_service = build_answer_service(client, retriever)

    return AppState(
        settings=settings,
        client=client,
        db_engine=db_engine,
        retriever=retriever,
        answer_service=answer_service,
    )
