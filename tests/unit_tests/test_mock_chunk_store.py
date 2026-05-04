import unittest

from src.chunking_service.data_types import ChunkType, RetrievalChunk
from tests.unit_tests.mock_chunk_store import MockChunkStore


class MockChunkStoreTest(unittest.TestCase):
    def test_mock_chunk_store_tracks_chunks_embeddings_and_setup(self) -> None:
        """Verify the mock chunk store keeps chunks and embeddings in sync."""
        store = MockChunkStore()

        chunk_a = RetrievalChunk(
            chunk_id="chunk-a",
            record_id="record-1",
            record_index=0,
            chunk_index=0,
            split="train",
            chunk_type=ChunkType.PRE_TEXT,
            text="alpha",
            has_type2_question=False,
            has_duplicate_columns=False,
            has_non_numeric_values=False,
            num_dialogue_turns=1,
        )
        chunk_b = RetrievalChunk(
            chunk_id="chunk-b",
            record_id="record-1",
            record_index=0,
            chunk_index=1,
            split="train",
            chunk_type=ChunkType.POST_TEXT,
            text="beta",
            has_type2_question=False,
            has_duplicate_columns=False,
            has_non_numeric_values=False,
            num_dialogue_turns=1,
        )

        def embedding_fn(text: str) -> list[float]:
            return [float(len(text))]

        store.setup()
        store.add_chunks(
            [chunk_b, chunk_a],
            embedding_fn=embedding_fn,
            embedding_model="mock-model",
        )

        self.assertTrue(store.setup_called)
        self.assertEqual(store.add_chunks_called, 1)
        self.assertEqual(store.embedding_model, "mock-model")
        self.assertEqual(
            [chunk.chunk_id for chunk in store.chunks], ["chunk-b", "chunk-a"]
        )
        self.assertEqual(
            [chunk.chunk_id for chunk in store.get_chunks_for_record("record-1")],
            ["chunk-a", "chunk-b"],
        )
        self.assertEqual(store.embeddings_by_chunk_id["chunk-a"], [5.0])
        self.assertEqual(store.embeddings_by_chunk_id["chunk-b"], [4.0])


if __name__ == "__main__":
    unittest.main()
