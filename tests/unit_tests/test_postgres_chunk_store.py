import json
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.chunking_service.chunking import chunk_record
from src.chunking_service.data_types import ConvFinQARecord
from src.db_service.postgres_chunk_store import PostgresChunkStore
from src.db_service.schemas import ChunkEmbeddingTable
from tests.unit_tests.mock_ollama_client import MockOllamaClient

SAMPLE_DATA_DIR = Path("data/samples")


class PostgresChunkStoreSqlAlchemyTest(unittest.TestCase):
    def test_add_chunks_and_fetch_chunks_for_record_with_sqlalchemy_engine(
        self,
    ) -> None:
        sample_file = SAMPLE_DATA_DIR / "convfinqa_train_sample.json"
        sample_payload = json.loads(sample_file.read_text())
        record = ConvFinQARecord(**sample_payload["records"][0])
        chunks = chunk_record(
            record=record,
            split=sample_payload["split"],
            record_index=0,
            source_file=sample_file,
        )
        model_client = MockOllamaClient(model_name="mock-qwen")

        engine = create_engine("sqlite+pysqlite:///:memory:")
        store = PostgresChunkStore(engine=engine)

        store.setup()
        store.add_chunks(
            chunks,
            embedding_fn=model_client.embed,
            embedding_model=model_client.get_config().model_name,
        )

        stored_chunks = store.get_chunks_for_record(record.id)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        with session_factory() as session:
            embedding_rows = list(session.scalars(select(ChunkEmbeddingTable)))

        self.assertEqual(len(stored_chunks), len(chunks))
        self.assertEqual(len(embedding_rows), len(chunks))
        self.assertEqual(
            [chunk.chunk_id for chunk in stored_chunks],
            [chunk.chunk_id for chunk in chunks],
        )
        self.assertEqual(
            [chunk.chunk_index for chunk in stored_chunks],
            list(range(len(chunks))),
        )
        self.assertEqual(stored_chunks[0].record_id, record.id)
        self.assertEqual(stored_chunks[0].text, chunks[0].text)
        self.assertEqual(stored_chunks[0].matched_metrics, chunks[0].matched_metrics)
        self.assertEqual(stored_chunks[0].years, chunks[0].years)
        self.assertEqual(stored_chunks[0].months, chunks[0].months)
        self.assertEqual(stored_chunks[0].quarters, chunks[0].quarters)
        self.assertEqual(stored_chunks[0].days, chunks[0].days)
        self.assertEqual(stored_chunks[0].dates, chunks[0].dates)
        self.assertEqual(stored_chunks[0].period_labels, chunks[0].period_labels)
        self.assertEqual(embedding_rows[0].embedding_model, "mock-qwen")
        self.assertEqual(embedding_rows[0].embedding_dimension, 3)
        self.assertEqual(
            embedding_rows[0].embedding,
            [float(len(chunks[0].text)), float(len(chunks[0].text) % 10), 1.0],
        )


if __name__ == "__main__":
    unittest.main()
