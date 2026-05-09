"""Standalone ingestion script for ConvFinQA data."""

from __future__ import annotations

from src.chunking_service.data_loader import ProcessLayer
from src.db_service.postgres_controllers import PostgresChunkStore
from src.logger import get_logger
from src.runtime import build_ingestion_context

logger = get_logger("ingest_script")


def main() -> None:
    context = build_ingestion_context()

    logger.info("Starting ingestion")
    db_engine = context.db_engine
    if db_engine is None:
        raise ValueError("Could not build db engine")

    db_service = PostgresChunkStore(db_engine)
    settings = context.settings
    if settings is None:
        raise ValueError("No settings could be initialized")

    client = context.client
    if client is None:
        raise ValueError("Could not initialize client")

    total_chunks = 0
    for raw_file_src in settings.ingestion_paths:
        chunks = ProcessLayer(
            db_service=db_service,
            raw_file_src=raw_file_src,
            model_client=client,
        ).process()
        total_chunks += len(chunks)
        logger.info("Ingested %s chunks from %s", len(chunks), raw_file_src)

    logger.info("Ingestion finished")

    if not db_service.has_data():
        raise RuntimeError("Ingestion finished but no chunks were detected")

    logger.info("ingestion complete: %s chunks", total_chunks)


if __name__ == "__main__":
    main()
