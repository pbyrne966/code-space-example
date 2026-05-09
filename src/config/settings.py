"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path
from typing import cast

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Given a hard coded model name here
    model_type: str = "QWEN"
    model_config_path: Path = Field(validation_alias="MODEL_CONFIG_PATH")

    raw_data_path: Path = Field(validation_alias="RAW_DATA_PATH")
    ingestion_file_paths: list[Path] | None = Field(
        default=None,
        validation_alias="INGESTION_FILE_PATHS",
    )

    postgres_host: str = Field(validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(validation_alias="POSTGRES_DB")
    postgres_user: str = Field(validation_alias="POSTGRES_USER")
    caching: bool = Field(validation_alias="CACHING_ON", default=False)
    postgres_password: str = Field(
        validation_alias="POSTGRES_PASSWORD",
    )
    postgres_connection_url: str | None = Field(
        default=None,
        validation_alias="POSTGRES_CONNECTION_URL",
    )

    @field_validator("model_config_path")
    @classmethod
    def resolve_path(cls, v: Path) -> Path:
        v = v.expanduser().resolve()
        if not v.exists():
            raise ValueError(f"Model config path does not exist: {v}")
        return v

    @field_validator("ingestion_file_paths", mode="before")
    @classmethod
    def parse_ingestion_file_paths(
        cls, v: str | list[str] | list[Path] | None
    ) -> list[Path] | list[str] | None:
        if v is None or isinstance(v, list):
            return v
        return [Path(path.strip()) for path in v.split(",") if path.strip()]

    @property
    def ingestion_paths(self) -> list[Path]:
        """Return configured ingestion files, falling back to RAW_DATA_PATH."""
        return self.ingestion_file_paths or [self.raw_data_path]

    @property
    def database_url(self) -> str:
        """Return the SQLAlchemy-compatible Postgres connection URL."""
        if self.postgres_connection_url:
            return self.postgres_connection_url

        return (
            "postgresql+psycopg://"
            f"{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings() # type: ignore
