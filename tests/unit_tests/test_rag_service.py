import unittest
from types import SimpleNamespace

from src.rag_service import RagAnswer, RAGQwenService, RetrieverClient
from tests.unit_tests.mock_ollama_client import MockOllamaClient


class FakeRetriever(RetrieverClient):
    """Return a fixed retrieval result for RAG tests."""

    def retrieve(self, query: str):
        """Return one deterministic retrieval row."""
        chunk = SimpleNamespace(
            chunk_id="chunk-1",
            record_id="record-1",
            metric="revenue",
            years=["2024"],
            period_labels=["FY2024"],
            text="Revenue was 100.",
        )
        return [SimpleNamespace(RetrievalChunkTable=chunk, distance=0.12)]


class RagServiceTest(unittest.TestCase):
    def test_build_prompt_requests_json_only(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output='{"answer":"ok","citations":[]}',
        )
        service = RAGQwenService(
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

    def test_answer_validates_json_and_uses_cache(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output='{"answer":"Revenue was 100.","citations":["chunk-1"]}',
        )
        service = RAGQwenService(
            model_client=model_client,
            retriever=FakeRetriever(),
        )

        result = service.answer("What is revenue?")

        self.assertIsInstance(result, RagAnswer)
        self.assertEqual(result.answer, "Revenue was 100.")
        self.assertEqual(result.citations, ["chunk-1"])
        self.assertEqual(len(model_client.prompts), 1)

        cached = service.answer("What is revenue?")
        self.assertEqual(cached.answer, "Revenue was 100.")
        self.assertEqual(len(model_client.prompts), 1)

    def test_answer_rejects_invalid_json(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output="not json",
        )
        service = RAGQwenService(
            model_client=model_client,
            retriever=FakeRetriever(),
        )

        with self.assertRaises(ValueError):
            service.answer("What is revenue?")


if __name__ == "__main__":
    unittest.main()
