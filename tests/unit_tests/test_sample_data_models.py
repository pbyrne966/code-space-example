import json
import unittest
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from src.chunking_service.chunking import chunk_record
from src.chunking_service.data_types import (
    ConvFinQARecord,
    EmbeddedChunk,
    RetrievalChunk,
)
from src.db_service.data_types import (
    ChatHistoryPair,
    ChatMessageRecord,
    ChatSessionRecord,
    RetrievedChunkRecord,
)
from src.db_service.mappers import (
    retrieval_chunk_to_embedding_table,
    retrieval_chunk_to_table,
)
from src.db_service.schemas import (
    ChatExchange,
    ChatSession,
    ChunkEmbeddingTable,
    MAX_EMBEDDING_DIMENSION,
    RetrievalChunkTable,
)

SAMPLE_DATA_DIR = Path("data/samples")


class SampleDataModelValidationTest(unittest.TestCase):
    def test_all_sample_records_validate_as_convfinqa_records(self) -> None:
        sample_files = sorted(SAMPLE_DATA_DIR.glob("convfinqa_*_sample.json"))
        self.assertTrue(sample_files, f"No sample files found in {SAMPLE_DATA_DIR}")

        for sample_file in sample_files:
            with self.subTest(sample_file=str(sample_file)):
                sample_payload = json.loads(sample_file.read_text())
                records = sample_payload["records"]

                self.assertEqual(sample_payload["sample_count"], len(records))
                for record in records:
                    self.assertIn("id", record)
                    record_id = record["id"]
                    try:
                        ConvFinQARecord(**record)
                    except ValidationError as error:
                        self.fail(
                            f"{sample_file} record {record_id} failed validation: {error}"
                        )

    def test_sample_records_convert_to_retrieval_chunk_table_rows(self) -> None:
        sample_files = sorted(SAMPLE_DATA_DIR.glob("convfinqa_*_sample.json"))
        self.assertTrue(sample_files, f"No sample files found in {SAMPLE_DATA_DIR}")

        for sample_file in sample_files:
            with self.subTest(sample_file=str(sample_file)):
                sample_payload = json.loads(sample_file.read_text())
                records = sample_payload["records"]

                for record_index, raw_record in enumerate(records):
                    record = ConvFinQARecord(**raw_record)
                    chunks = chunk_record(
                        record=record,
                        split=sample_payload["split"],
                        record_index=record_index,
                        source_file=sample_file,
                    )

                    self.assertTrue(chunks)

                    for chunk in chunks:
                        table_row = retrieval_chunk_to_table(chunk)

                        self.assertIsInstance(table_row, RetrievalChunkTable)
                        self.assertEqual(table_row.chunk_id, chunk.chunk_id)
                        self.assertEqual(table_row.record_id, record.id)
                        self.assertEqual(table_row.record_index, record_index)
                        self.assertEqual(table_row.chunk_index, chunk.chunk_index)
                        self.assertEqual(table_row.split, sample_payload["split"])
                        self.assertEqual(table_row.chunk_type, chunk.chunk_type.value)
                        self.assertEqual(table_row.text, chunk.text)
                        self.assertEqual(table_row.metric, chunk.metric)
                        self.assertEqual(
                            table_row.matched_metrics, chunk.matched_metrics
                        )
                        self.assertEqual(table_row.table_column, chunk.table_column)
                        self.assertEqual(table_row.turn_index, chunk.turn_index)
                        self.assertEqual(table_row.qa_split, chunk.qa_split)
                        self.assertEqual(table_row.years, chunk.years)
                        self.assertIsInstance(table_row.to_pydantic(), RetrievalChunk)
                        self.assertEqual(table_row.to_pydantic(), chunk)

    def test_sample_records_convert_to_chunk_embedding_table_rows(self) -> None:
        sample_files = sorted(SAMPLE_DATA_DIR.glob("convfinqa_*_sample.json"))
        self.assertTrue(sample_files, f"No sample files found in {SAMPLE_DATA_DIR}")

        def embedding_fn(text: str) -> list[float]:
            return [float(len(text))] * MAX_EMBEDDING_DIMENSION

        for sample_file in sample_files:
            with self.subTest(sample_file=str(sample_file)):
                sample_payload = json.loads(sample_file.read_text())
                records = sample_payload["records"]

                for record_index, raw_record in enumerate(records):
                    record = ConvFinQARecord(**raw_record)
                    chunks = chunk_record(
                        record=record,
                        split=sample_payload["split"],
                        record_index=record_index,
                        source_file=sample_file,
                    )

                    self.assertTrue(chunks)

                    for chunk in chunks:
                        embedding_row = retrieval_chunk_to_embedding_table(
                            chunk=chunk,
                            embedding_fn=embedding_fn,
                            embedding_model="mock-model",
                        )

                        self.assertIsInstance(embedding_row, ChunkEmbeddingTable)
                        self.assertEqual(embedding_row.chunk_id, chunk.chunk_id)
                        self.assertEqual(embedding_row.embedding_model, "mock-model")
                        self.assertEqual(
                            embedding_row.embedding_dimension,
                            MAX_EMBEDDING_DIMENSION,
                        )
                        self.assertEqual(
                            embedding_row.embedding, embedding_fn(chunk.text)
                        )
                        embedding_row.chunk = retrieval_chunk_to_table(chunk)
                        pydantic_row = embedding_row.to_pydantic()
                        self.assertIsInstance(pydantic_row, EmbeddedChunk)
                        self.assertEqual(pydantic_row.chunk, chunk)
                        self.assertEqual(pydantic_row.embedding_model, "mock-model")
                        retrieval_result = RetrievedChunkRecord(
                            chunk=embedding_row.chunk.to_pydantic(),
                            distance=0.1,
                        )
                        self.assertIsInstance(retrieval_result, RetrievedChunkRecord)
                        self.assertEqual(retrieval_result.chunk, chunk)

    def test_chat_rows_serialize_to_pydantic_records(self) -> None:
        now = datetime(2026, 5, 5, 12, 0, tzinfo=UTC)
        message = ChatExchange(
            message_id=7,
            session_id="session-1",
            role="user",
            content="What changed?",
            created_at=now,
            updated_at=now,
        )
        chat_session = ChatSession(
            session_id="session-1",
            record_id="record-1",
            title="Demo",
            message_count=1,
            last_message_index=0,
            created_at=now,
            last_message_at=now,
            updated_at=now,
            messages=[message],
        )

        message_record = message.to_pydantic()
        session_record = chat_session.to_pydantic()

        self.assertIsInstance(message_record, ChatMessageRecord)
        self.assertEqual(message_record.message_id, 7)
        self.assertEqual(message_record.role, "user")
        self.assertIsInstance(session_record, ChatSessionRecord)
        self.assertEqual(session_record.session_id, "session-1")
        self.assertEqual(session_record.messages, [message_record])
        history_pair = ChatHistoryPair(
            user_question=message_record,
            assistant=ChatMessageRecord(
                message_id=8,
                session_id="session-1",
                role="assistant",
                content="A lot.",
                created_at=now,
                updated_at=now,
            ),
        )
        self.assertIsInstance(history_pair, ChatHistoryPair)
        self.assertEqual(history_pair.user_question, message_record)

    def test_empty_embedding_vector_raises_value_error(self) -> None:
        sample_file = sorted(SAMPLE_DATA_DIR.glob("convfinqa_*_sample.json"))[0]
        sample_payload = json.loads(sample_file.read_text())
        record = ConvFinQARecord(**sample_payload["records"][0])
        chunk = chunk_record(
            record=record,
            split=sample_payload["split"],
            record_index=0,
            source_file=sample_file,
        )[0]

        def empty_embedding_fn(text: str) -> list[float]:
            return []

        with self.assertRaises(ValueError):
            retrieval_chunk_to_embedding_table(
                chunk=chunk,
                embedding_fn=empty_embedding_fn,
                embedding_model="mock-model",
            )


if __name__ == "__main__":
    unittest.main()
