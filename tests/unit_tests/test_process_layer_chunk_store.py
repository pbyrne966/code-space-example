import json
import tempfile
import unittest
from pathlib import Path

from src.chunking_service.data_loader import ProcessLayer
from tests.unit_tests.mock_chunk_store import MockChunkStore
from tests.unit_tests.mock_ollama_client import MockOllamaClient

SAMPLE_DATA_DIR = Path("data/samples")


class ProcessLayerChunkStoreTest(unittest.TestCase):
    def test_process_layer_adds_chunks_to_chunk_store(self) -> None:
        sample_file = SAMPLE_DATA_DIR / "convfinqa_train_sample.json"
        sample_payload = json.loads(sample_file.read_text())
        raw_payload = {
            "train": sample_payload["records"],
            "dev": [],
            "test": [],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_file = Path(tmp_dir) / "convfinqa_sample.json"
            raw_file.write_text(json.dumps(raw_payload))

            chunk_store = MockChunkStore()
            model_client = MockOllamaClient(model_name="mock-qwen")
            process_layer = ProcessLayer(
                db_service=chunk_store,
                raw_file_src=raw_file,
                model_client=model_client,
            )

            processed_chunks = process_layer.process()

        self.assertTrue(chunk_store.setup_called)
        self.assertEqual(chunk_store.add_chunks_called, 1)
        self.assertTrue(processed_chunks)
        self.assertEqual(len(chunk_store.chunks), len(processed_chunks))
        self.assertEqual(
            {chunk.chunk_id for chunk in chunk_store.chunks},
            {chunk.chunk_id for chunk in processed_chunks},
        )
        self.assertEqual(chunk_store.embedding_model, "mock-qwen")
        self.assertEqual(
            model_client.embedded_texts,
            [chunk.text for chunk in processed_chunks],
        )
        self.assertEqual(
            set(chunk_store.embeddings_by_chunk_id),
            {chunk.chunk_id for chunk in processed_chunks},
        )

        first_record_id = sample_payload["records"][0]["id"]
        stored_chunks = chunk_store.get_chunks_for_record(first_record_id)

        self.assertTrue(stored_chunks)
        self.assertTrue(
            all(chunk.record_id == first_record_id for chunk in stored_chunks)
        )
        self.assertEqual(
            [chunk.chunk_index for chunk in stored_chunks],
            list(range(len(stored_chunks))),
        )
        self.assertEqual(stored_chunks[0].split, "train")
        self.assertIn(first_record_id, stored_chunks[0].text)
        self.assertEqual(
            chunk_store.embeddings_by_chunk_id[stored_chunks[0].chunk_id],
            [float(len(stored_chunks[0].text)), float(len(stored_chunks[0].text) % 10), 1.0],
        )


if __name__ == "__main__":
    unittest.main()
