"""Create Postgres extensions and application tables."""

from sqlalchemy import create_engine, text

from src.config.settings import get_settings
from src.db_service import Base
from src.logger import get_logger

logger = get_logger("db_setup")


def main() -> None:
    """Initialise the configured Postgres database schema."""
    settings = get_settings()
    engine = create_engine(settings.database_url)

    if engine.dialect.name != "postgresql":
        raise ValueError("Database setup requires a PostgreSQL engine")

    logger.info("Creating Postgres extensions")
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    logger.info("Creating database tables")
    Base.metadata.create_all(engine)
    logger.info("Database setup complete")


if __name__ == "__main__":
    main()
