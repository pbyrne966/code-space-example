"""Mock application settings for tests."""

from pathlib import Path


class MockSettings:
    """Lightweight settings object with the fields tests need."""

    repo_root = Path(__file__).resolve().parents[2]

    def __init__(
        self,
        model_config_path: Path | str | None = None,
        raw_data_path: Path | str | None = None,
        postgres_host: str = "localhost",
        postgres_port: int = 5432,
        postgres_db: str = "test_db",
        postgres_user: str = "test_user",
        postgres_password: str = "test_password",
        postgres_connection_url: str | None = None,
    ) -> None:
        self.model_type = "QWEN"
        self.model_config_path = Path(
            model_config_path or self.repo_root / "configs" / "model_config.toml"
        )
        self.raw_data_path = Path(
            raw_data_path or self.repo_root / "data" / "convfinqa_dataset.json"
        )
        self.ingestion_file_paths = [self.raw_data_path]
        self.postgres_host = postgres_host
        self.postgres_port = postgres_port
        self.postgres_db = postgres_db
        self.postgres_user = postgres_user
        self.postgres_password = postgres_password
        self.postgres_connection_url = postgres_connection_url

    @property
    def database_url(self) -> str:
        """Return the Postgres URL or the injected override."""
        if self.postgres_connection_url:
            return self.postgres_connection_url

        return (
            "postgresql+psycopg://"
            f"{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_db}"
        )

    @property
    def ingestion_paths(self) -> list[Path]:
        """Return files used by ingestion helpers."""
        return self.ingestion_file_paths
