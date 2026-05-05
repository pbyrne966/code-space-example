import unittest

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql import select

from src.chunking_service.period_extraction import PeriodData
from src.db_service.postgres_controllers import PostgresChunkStore
from src.db_service.schemas import (
    ChunkEmbeddingTable,
    MAX_EMBEDDING_DIMENSION,
    RetrievalChunkTable,
)


class FakeDialect:
    name = "not-postgresql"


class FakeEngine:
    dialect = FakeDialect()


class PostgresChunkStoreTest(unittest.TestCase):
    def _compile_period_filter(self, period_data: PeriodData) -> str:
        store = PostgresChunkStore(engine=FakeEngine())  # type: ignore[arg-type]
        statement = store._apply_period_filters(
            select(RetrievalChunkTable),
            period_data,
        )
        return str(statement.compile(dialect=postgresql.dialect()))

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

    def test_period_filter_uses_postgres_jsonb_containment(self) -> None:
        statement = select(RetrievalChunkTable).where(
            PostgresChunkStore._jsonb_array_contains(
                RetrievalChunkTable.dates,
                ["2024-03-31"],
            )
        )

        sql = str(statement.compile(dialect=postgresql.dialect()))

        self.assertIn("retrieval_chunks.dates @>", sql)

    def test_period_filter_combines_year_and_month(self) -> None:
        sql = self._compile_period_filter(
            PeriodData(
                years=["2024"],
                months=[3],
            )
        )

        self.assertIn("retrieval_chunks.years @>", sql)
        self.assertIn("retrieval_chunks.months @>", sql)
        self.assertNotIn("retrieval_chunks.quarters @>", sql)
        self.assertNotIn("retrieval_chunks.dates @>", sql)

    def test_period_filter_combines_quarter_and_month(self) -> None:
        sql = self._compile_period_filter(
            PeriodData(
                quarters=[1],
                months=[3],
            )
        )

        self.assertIn("retrieval_chunks.quarters @>", sql)
        self.assertIn("retrieval_chunks.months @>", sql)
        self.assertNotIn("retrieval_chunks.years @>", sql)
        self.assertNotIn("retrieval_chunks.dates @>", sql)

    def test_period_filter_combines_year_quarter_and_month(self) -> None:
        sql = self._compile_period_filter(
            PeriodData(
                years=["2024"],
                quarters=[1],
                months=[3],
            )
        )

        self.assertIn("retrieval_chunks.years @>", sql)
        self.assertIn("retrieval_chunks.quarters @>", sql)
        self.assertIn("retrieval_chunks.months @>", sql)
        self.assertNotIn("retrieval_chunks.dates @>", sql)

    def test_exact_date_filter_takes_precedence(self) -> None:
        sql = self._compile_period_filter(
            PeriodData(
                years=["2024"],
                months=[3],
                days=[31],
                dates=["2024-03-31"],
            )
        )

        self.assertIn("retrieval_chunks.dates @>", sql)
        self.assertNotIn("retrieval_chunks.years @>", sql)
        self.assertNotIn("retrieval_chunks.months @>", sql)
        self.assertNotIn("retrieval_chunks.days @>", sql)


if __name__ == "__main__":
    unittest.main()
