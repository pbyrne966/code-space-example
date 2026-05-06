import json
import unittest
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from src.chunking_service.chunking import chunk_record
from src.data_types import (
    ChatHistoryPair,
    ChatMessageRecord,
    ChatSessionRecord,
    ChunkType,
    ConvFinQARecord,
    EmbeddedChunk,
    RetrievedChunkRecord,
    RetrievalChunk,
)
from src.db_service.mappers import (
    retrieval_chunk_to_embedding_table,
    retrieval_chunk_to_table,
    source_record_from_chunk,
)
from src.db_service.schemas import (
    MAX_EMBEDDING_DIMENSION,
    ChatExchange,
    ChatSession,
    ChunkEmbeddingTable,
    RetrievalChunkTable,
    SourceRecordTable,
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
                        self.assertEqual(table_row.source_file, chunk.source_file)
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
                        self.assertEqual(
                            table_row.table_values,
                            [value.model_dump() for value in chunk.table_values],
                        )
                        self.assertEqual(table_row.years, chunk.years)
                        self.assertIsInstance(table_row.to_pydantic(), RetrievalChunk)
                        self.assertEqual(table_row.to_pydantic(), chunk)

                    source_record = source_record_from_chunk(chunks[0])
                    self.assertIsInstance(source_record, SourceRecordTable)
                    self.assertEqual(source_record.record_id, record.id)
                    self.assertEqual(source_record.source_file, str(sample_file))
                    self.assertEqual(source_record.record_index, record_index)
                    self.assertEqual(source_record.split, sample_payload["split"])
                    self.assertEqual(source_record.to_pydantic().record_id, record.id)

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
                hashed_content="assistant-hash",
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

    def test_dialogue_questions_are_not_chunked_for_retrieval(self) -> None:
        sample_file = sorted(SAMPLE_DATA_DIR.glob("convfinqa_*_sample.json"))[0]
        sample_payload = json.loads(sample_file.read_text())
        record = ConvFinQARecord(**sample_payload["records"][0])

        chunks = chunk_record(
            record=record,
            split=sample_payload["split"],
            record_index=0,
            source_file=sample_file,
        )
        chunk_text = "\n".join(chunk.text for chunk in chunks)

        for question in record.dialogue.conv_questions:
            self.assertNotIn(question, chunk_text)

    def test_table_chunks_preserve_structured_values(self) -> None:
        sample_file = sorted(SAMPLE_DATA_DIR.glob("convfinqa_*_sample.json"))[0]
        sample_payload = json.loads(sample_file.read_text())
        record = ConvFinQARecord(**sample_payload["records"][0])

        chunks = chunk_record(
            record=record,
            split=sample_payload["split"],
            record_index=0,
            source_file=sample_file,
        )
        table_row_chunk = next(
            chunk for chunk in chunks if chunk.chunk_type == ChunkType.TABLE_ROW
        )
        table_metric_chunk = next(
            chunk for chunk in chunks if chunk.chunk_type == ChunkType.TABLE_METRIC
        )

        self.assertTrue(table_row_chunk.table_values)
        self.assertTrue(table_metric_chunk.table_values)
        self.assertTrue(
            all(
                table_value.table_column == table_row_chunk.table_column
                for table_value in table_row_chunk.table_values
            )
        )
        self.assertTrue(
            all(
                table_value.metric == table_metric_chunk.metric
                for table_value in table_metric_chunk.table_values
            )
        )
        self.assertTrue(
            all(
                table_value.numeric_value == float(table_value.value)
                for table_value in table_row_chunk.table_values
                if isinstance(table_value.value, int | float)
            )
        )


if __name__ == "__main__":
    unittest.main()
