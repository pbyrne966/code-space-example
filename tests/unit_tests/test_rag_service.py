import unittest

from src.aggregator_service.query_intent import TableValueCandidate
from src.chunking_service.period_extraction import PeriodData
from src.data_types import ChunkType, RetrievalChunk, RetrievedChunkRecord
from src.rag_service import RagAnswer, RAGService
from tests.unit_tests.mock_ollama_client import MockOllamaClient

LOOKUP_ANSWER = '{"answer":"100","citations":["chunk-1"],"calculation_program":null}'


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
            table_values=[
                {
                    "metric": "revenue",
                    "table_column": "FY2024",
                    "value": 100.0,
                    "numeric_value": 100.0,
                },
                {
                    "metric": "revenue",
                    "table_column": "FY2023",
                    "value": 50.0,
                    "numeric_value": 50.0,
                },
            ],
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
            chat_output='{"answer":"ok","citations":[],"calculation_program":null}',
        )
        service = RAGService(
            model_client=model_client,
            retriever=FakeRetriever(),
        )

        prompt = service.build_prompt(
            "What is revenue?",
            ["[Source 1] example"],
            [
                TableValueCandidate(
                    value_id="chunk-1:value:0",
                    chunk_id="chunk-1",
                    metric="revenue",
                    table_column="FY2024",
                    numeric_value=100.0,
                )
            ],
        )

        self.assertIn("Output a single JSON object only.", prompt)
        self.assertIn('"calculation_program": null', prompt)
        self.assertIn(
            "Return exactly these keys: answer, citations, calculation_program.",
            prompt,
        )
        self.assertIn("answer must be only the value, not a sentence.", prompt)
        self.assertIn('"answer": "100"', prompt)
        self.assertIn(
            'If the answer is not in the context, set answer to "I don\'t know".',
            prompt,
        )
        self.assertIn(
            "calculation_program must always be null.",
            prompt,
        )
        self.assertNotIn("Calculation program schema:", prompt)
        self.assertNotIn("Available table values for calculation:", prompt)

    def test_answer_validates_json_and_calls_model(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output=LOOKUP_ANSWER,
        )
        retriever = FakeRetriever()
        service = RAGService(
            model_client=model_client,
            retriever=retriever,
        )

        result = service.answer("What was revenue on March 31, 2024?", "record-1")

        self.assertIsInstance(result, RagAnswer)
        self.assertEqual(result.answer, "100")
        self.assertEqual(result.citations, ["chunk-1"])
        self.assertIsNone(result.turn_program)
        self.assertEqual(len(model_client.prompts), 1)
        self.assertEqual(
            model_client.response_formats[0]["properties"]["calculation_program"],
            {"type": "null"},
        )
        self.assertEqual(retriever.calls[0][2].dates, ["2024-03-31"])

        repeat = service.answer("What is revenue?", "record-1")
        self.assertEqual(repeat.answer, "100")
        self.assertEqual(len(model_client.prompts), 2)

    def test_answer_keeps_scalar_answer_and_records_turn_program(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output=(
                '{"answer":"2","citations":["chunk-1"],'
                '"calculation_program":{"steps":[{"operation":"divide",'
                '"operands":[{"kind":"table_value","value_id":"chunk-1:value:0"},'
                '{"kind":"literal","literal":50.0}]}]}}'
            ),
        )
        service = RAGService(
            model_client=model_client,
            retriever=FakeRetriever(),
        )

        result = service.answer("How many times larger was revenue?", "record-1")

        self.assertEqual(result.answer, "2")
        self.assertEqual(result.turn_program, "divide(100, 50)")

    def test_build_table_value_candidates_assigns_stable_ids(self) -> None:
        service = RAGService(
            model_client=MockOllamaClient(model_name="test-model"),
            retriever=FakeRetriever(),
        )
        results = service.retriever.retrieve("What is revenue?", "record-1")

        candidates = service.build_table_value_candidates(results)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].value_id, "chunk-1:value:0")
        self.assertEqual(candidates[0].chunk_id, "chunk-1")
        self.assertEqual(candidates[0].metric, "revenue")
        self.assertEqual(candidates[0].table_column, "FY2024")
        self.assertEqual(candidates[0].numeric_value, 100.0)
        self.assertEqual(candidates[1].value_id, "chunk-1:value:1")
        self.assertEqual(candidates[1].numeric_value, 50.0)

    def test_response_format_forces_null_calculation_program(self) -> None:
        service = RAGService(
            model_client=MockOllamaClient(model_name="test-model"),
            retriever=FakeRetriever(),
        )
        candidates = service.build_table_value_candidates(
            service.retriever.retrieve("What is revenue?", "record-1")
        )

        response_format = service.build_response_format(candidates)

        self.assertEqual(
            response_format["properties"]["calculation_program"],
            {"type": "null"},
        )

    def test_answer_keeps_calculated_scalar_answer(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output=(
                '{"answer":"50","citations":["chunk-1"],'
                '"calculation_program":{"steps":[{"operation":"subtract",'
                '"operands":[{"kind":"table_value","value_id":"chunk-1:value:0"},'
                '{"kind":"table_value","value_id":"chunk-1:value:1"}]}]}}'
            ),
        )
        service = RAGService(
            model_client=model_client,
            retriever=FakeRetriever(),
        )

        result = service.answer("How much did revenue increase?", "record-1")

        self.assertEqual(result.answer, "50")
        self.assertEqual(result.turn_program, "subtract(100, 50)")

    def test_answer_requires_explicit_calculation_program_key(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output='{"answer":"100","citations":["chunk-1"]}',
        )
        service = RAGService(
            model_client=model_client,
            retriever=FakeRetriever(),
        )

        with self.assertRaises(ValueError):
            service.answer("What is revenue?", "record-1")

    def test_answer_raises_when_calculation_program_fails(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output=(
                '{"answer":"50","citations":["chunk-1"],'
                '"calculation_program":{"steps":[{"operation":"subtract",'
                '"operands":[{"kind":"table_value","value_id":"missing"},'
                '{"kind":"table_value","value_id":"chunk-1:value:1"}]}]}}'
            ),
        )
        service = RAGService(
            model_client=model_client,
            retriever=FakeRetriever(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Calculation program failed: Unknown table value_id: missing",
        ):
            service.answer("How much did revenue increase?", "record-1")

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
