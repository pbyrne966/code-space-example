"""Shared runtime builders for ConvFinQA CLI entrypoints."""

from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.config.settings import Settings, get_settings
from src.db_service import PostgresChatService, PostgresChunkStore
from src.logger import get_logger
from src.model_service.models import BaseModelFactory, ModelClient
from src.rag_service import RAGService

logger = get_logger("typer_logger")


@dataclass
class AppState:
    settings: Settings | None = None
    client: ModelClient | None = None
    db_engine: Engine | None = None
    retriever: PostgresChunkStore | None = None
    chat_service: PostgresChatService | None = None
    answer_service: RAGService | None = None


@dataclass
class ValidatedAppState:
    settings: Settings
    client: ModelClient
    db_engine: Engine
    retriever: PostgresChunkStore
    chat_service: PostgresChatService
    answer_service: RAGService


@dataclass
class IngestionAppState:
    settings: Settings
    client: ModelClient
    db_engine: Engine


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


def build_retriever(db_engine: Engine, client: ModelClient) -> PostgresChunkStore:
    """Create the vector retriever used by chat."""
    logger.info("Initialising retriever")
    return PostgresChunkStore(
        db_engine,
        embedding_fn=client.embed,
        embedding_model=client.get_config().model_name,
        top_k=4,
    )


def build_chat_service(db_engine: Engine) -> PostgresChatService:
    """Create the chat persistence service."""
    logger.info("Initialising chat store")
    store = PostgresChatService(db_engine)
    store.setup()
    return store


def build_answer_service(
    client: ModelClient, retriever: PostgresChunkStore
) -> RAGService:
    """Create the answer service from the client and retriever."""
    logger.info("Initialising answer service")
    return RAGService(model_client=client, retriever=retriever)


def validate_app_state(
    settings: Settings,
    retriever: PostgresChunkStore,
    chat_service: PostgresChatService,
    answer_service: RAGService,
    db_engine: Engine,
    model_client: ModelClient,
) -> ValidatedAppState:
    if settings is None:
        raise ValueError("Settings could not be built")

    if retriever is None:
        raise ValueError("Retreiever could not be built")

    if chat_service is None:
        raise ValueError("Could not build chat service")

    if answer_service is None:
        raise ValueError("Could not build answer service")

    if not retriever.has_data():
        raise ValueError(
            "Services are up, but no ingested data was found. Run the ingestion script first."
        )
    return ValidatedAppState(
        settings,
        model_client,
        db_engine,
        retriever,
        chat_service,
        answer_service,
    )


def build_context() -> ValidatedAppState:
    """Build the runtime connections shared by CLI commands."""
    settings = load_config()
    client = build_client(settings)
    db_engine = build_db_engine(settings)
    retriever = build_retriever(db_engine, client)
    chat_service = build_chat_service(db_engine)
    answer_service = build_answer_service(client, retriever)

    return validate_app_state(
        settings, retriever, chat_service, answer_service, db_engine, client
    )


def build_ingestion_context() -> IngestionAppState:
    """Build only the services required before source data has been ingested."""
    settings = load_config()
    client = build_client(settings)
    db_engine = build_db_engine(settings)
    return IngestionAppState(settings=settings, client=client, db_engine=db_engine)
