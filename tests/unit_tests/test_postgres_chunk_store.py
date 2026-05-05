import unittest

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from src.db_service.postgres_controllers import PostgresChunkStore
from src.db_service.schemas import ChunkEmbeddingTable, MAX_EMBEDDING_DIMENSION


class FakeDialect:
    name = "not-postgresql"


class FakeEngine:
    dialect = FakeDialect()


class PostgresChunkStoreTest(unittest.TestCase):
    def test_embedding_column_uses_pgvector(self) -> None:
        ddl = str(
            CreateTable(ChunkEmbeddingTable.__table__).compile(
                dialect=postgresql.dialect()
            )
        )

        self.assertIn(f"embedding VECTOR({MAX_EMBEDDING_DIMENSION}) NOT NULL", ddl)

    def test_setup_rejects_non_postgresql_engines(self) -> None:
        store = PostgresChunkStore(engine=FakeEngine())  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "requires a PostgreSQL engine"):
            store.setup()


if __name__ == "__main__":
    unittest.main()
