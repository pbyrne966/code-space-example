import unittest

from src.chunking_service.period_extraction import PeriodData
from src.data_types import ChunkType, RetrievalChunk
from src.db_service.data_types import RetrievedChunkRecord
from src.rag_service import RagAnswer, RAGService
from tests.unit_tests.mock_ollama_client import MockOllamaClient


class FakeRetriever:
    """Return a fixed retrieval result for RAG tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, PeriodData | None]] = []

    def retrieve(
        self, query: str, record_id: str, period_data: PeriodData | None = None
    ):
        """Return one deterministic retrieval row."""
        if record_id != "record-1":
            raise AssertionError(f"Unexpected record_id: {record_id}")
        self.calls.append((query, record_id, period_data))
        chunk = RetrievalChunk(
            chunk_id="chunk-1",
            record_id="record-1",
            record_index=0,
            chunk_index=0,
            split="train",
            chunk_type=ChunkType.TABLE_METRIC,
            metric="revenue",
            years=["2024"],
            period_labels=["FY2024"],
            text="Revenue was 100.",
            has_type2_question=False,
            has_duplicate_columns=False,
            has_non_numeric_values=False,
            num_dialogue_turns=1,
        )
        return [RetrievedChunkRecord(chunk=chunk, distance=0.12)]


class RagServiceTest(unittest.TestCase):
    def test_build_prompt_requests_json_only(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output='{"answer":"ok","citations":[]}',
        )
        service = RAGService(
            model_client=model_client,
            retriever=FakeRetriever(),
        )

        prompt = service.build_prompt("What is revenue?", ["[Source 1] example"])

        self.assertIn("Output a single JSON object only.", prompt)
        self.assertIn(
            '{"answer":"...","citations":["chunk_id_1","chunk_id_2"]}', prompt
        )
        self.assertIn(
            'If the answer is not in the context, set answer to "I don\'t know".',
            prompt,
        )

    def test_answer_validates_json_and_calls_model(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output='{"answer":"Revenue was 100.","citations":["chunk-1"]}',
        )
        retriever = FakeRetriever()
        service = RAGService(
            model_client=model_client,
            retriever=retriever,
        )

        result = service.answer("What was revenue on March 31, 2024?", "record-1")

        self.assertIsInstance(result, RagAnswer)
        self.assertEqual(result.answer, "Revenue was 100.")
        self.assertEqual(result.citations, ["chunk-1"])
        self.assertEqual(len(model_client.prompts), 1)
        self.assertEqual(retriever.calls[0][2].dates, ["2024-03-31"])

        repeat = service.answer("What is revenue?", "record-1")
        self.assertEqual(repeat.answer, "Revenue was 100.")
        self.assertEqual(len(model_client.prompts), 2)

    def test_answer_rejects_invalid_json(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output="not json",
        )
        service = RAGService(
            model_client=model_client,
            retriever=FakeRetriever(),
        )

        with self.assertRaises(ValueError):
            service.answer("What is revenue?", "record-1")


if __name__ == "__main__":
    unittest.main()
