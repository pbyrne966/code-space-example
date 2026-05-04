import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.chunking_service.chunking import chunk_record
from src.chunking_service.data_types import ConvFinQARecord
from src.db_service.mappers import (
    retrieval_chunk_to_embedding_table,
    retrieval_chunk_to_table,
)
from src.db_service.schemas import ChunkEmbeddingTable, RetrievalChunkTable

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

    def test_sample_records_convert_to_chunk_embedding_table_rows(self) -> None:
        sample_files = sorted(SAMPLE_DATA_DIR.glob("convfinqa_*_sample.json"))
        self.assertTrue(sample_files, f"No sample files found in {SAMPLE_DATA_DIR}")

        def embedding_fn(text: str) -> list[float]:
            return [float(len(text)), 1.0, 0.5]

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
                        self.assertEqual(embedding_row.embedding_dimension, 3)
                        self.assertEqual(
                            embedding_row.embedding, embedding_fn(chunk.text)
                        )

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
