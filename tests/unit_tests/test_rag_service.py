import unittest

from src.aggregator_service.query_intent import TableValueCandidate
from src.chunking_service.period_extraction import PeriodData
from src.data_types import (
    ChatHistoryPair,
    ChatMessageRecord,
    ChunkType,
    RetrievalChunk,
    RetrievedChunkRecord,
)
from src.rag_service import RagAnswer, RAGService
from tests.unit_tests.mock_ollama_client import MockOllamaClient

LOOKUP_ANSWER = (
    '{"answer":"100","citations":["chunk-1"],"calculation_program":null,"requery":null}'
)


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


class ThreeResultRetriever(FakeRetriever):
    def retrieve(
        self, query: str, record_id: str, period_data: PeriodData | None = None
    ):
        base_result = super().retrieve(query, record_id, period_data)[0]
        results = []
        for index in range(3):
            chunk = base_result.chunk.model_copy(
                update={
                    "chunk_id": f"chunk-{index + 1}",
                    "text": f"Revenue source {index + 1}.",
                }
            )
            results.append(RetrievedChunkRecord(chunk=chunk, distance=0.1 + index))
        return results


class RagServiceTest(unittest.TestCase):
    def test_build_prompt_requests_json_only(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output=(
                '{"answer":"ok","citations":[],"calculation_program":null,'
                '"requery":null}'
            ),
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

        self.assertIn("Output a single JSON object only", prompt)
        self.assertIn('"calculation_program": null', prompt)
        self.assertIn('"requery": null', prompt)
        self.assertIn(
            "return exactly these keys: answer, citations, calculation_program, requery.",
            prompt,
        )
        self.assertIn("No requery has been performed for this user question.", prompt)
        self.assertIn(
            "`requery` must be null unless a better retrieval pass is required before answering.",
            prompt,
        )
        self.assertIn("answer must be only the value, not a sentence.", prompt)
        self.assertIn(
            'If the answer is not in the context, set answer to "I don\'t know".',
            prompt,
        )
        self.assertIn(
            "Set calculation_program to null only when the answer is a direct lookup or text answer.",
            prompt,
        )
        self.assertIn(
            "When arithmetic is needed, return a non-null calculation_program",
            prompt,
        )
        self.assertIn(
            "For percent answers, include a final percentage step",
            prompt,
        )
        self.assertIn("Calculation program schema:", prompt)
        self.assertIn("Conversation history:", prompt)
        self.assertIn("No prior turns.", prompt)
        self.assertIn("Available table values for calculation:", prompt)
        self.assertIn('"value_id": "chunk-1:value:0"', prompt)
        self.assertIn('"answer": "14.1%"', prompt)
        self.assertIn('"operation": "percentage"', prompt)
        self.assertIn('"step_index": 1', prompt)
        self.assertNotIn('"answer": "The percentage change is 14.1%."', prompt)
        self.assertNotIn("Match both the requested metric phrase", prompt)

    def test_build_prompt_includes_session_history_guidance(self) -> None:
        service = RAGService(
            model_client=MockOllamaClient(model_name="test-model"),
            retriever=FakeRetriever(),
        )

        prompt = service.build_prompt(
            "What about in 2008?",
            ["[Source 1] current 2008 table row"],
            session_history=(
                "[Prior turn 1]\n"
                "User question: what is the net cash from operating activities in 2009?\n"
                "Assistant answer: 206588\n"
                "Prior retrieved context:\n"
                "[Source 1] net cash from operating activities values"
            ),
        )

        self.assertIn(
            'Use prior turns only to resolve conversational follow-ups such as "what about in 2008?".',
            prompt,
        )
        self.assertIn(
            "User question: what is the net cash from operating activities in 2009?",
            prompt,
        )
        self.assertIn("Assistant answer: 206588", prompt)
        self.assertIn("Retrieved context:\n[Source 1] current 2008 table row", prompt)

    def test_build_prompt_hides_requery_policy_after_requery(self) -> None:
        service = RAGService(
            model_client=MockOllamaClient(model_name="test-model"),
            retriever=FakeRetriever(),
        )

        prompt = service.build_prompt(
            "What about in 2008?",
            ["[Source 1] current 2008 table row"],
            is_requery=True,
        )

        self.assertIn(
            "A requery has already been performed for this user question.",
            prompt,
        )
        self.assertIn("Return `requery: null`.", prompt)
        self.assertNotIn(
            "`requery` must be null unless a better retrieval pass is required before answering.",
            prompt,
        )

    def test_parse_chat_history_formats_prior_answers_and_context(self) -> None:
        service = RAGService(
            model_client=MockOllamaClient(model_name="test-model"),
            retriever=FakeRetriever(),
        )
        history = [
            ChatHistoryPair(
                user_question=ChatMessageRecord(
                    message_id=1,
                    session_id="session-1",
                    role="user",
                    content="what is the net cash from operating activities in 2009?",
                    hashed_content="hash-1",
                ),
                assistant=ChatMessageRecord(
                    message_id=2,
                    session_id="session-1",
                    role="assistant",
                    content=RagAnswer(
                        answer="206588",
                        citations=["chunk-1"],
                        context_blocks=[
                            "[Source 1]\n"
                            "metric: net cash from operating activities\n"
                            "table_values: 2009=206588, 2008=181001"
                        ],
                    ).model_dump_json(),
                    hashed_content="hash-2",
                ),
            )
        ]

        parsed = service.parse_chat_history(history)

        self.assertIsNotNone(parsed)
        self.assertIn(
            "User question: what is the net cash from operating activities in 2009?",
            parsed,
        )
        self.assertIn("Assistant answer: 206588", parsed)
        self.assertIn('Assistant citations: ["chunk-1"]', parsed)
        self.assertIn("2008=181001", parsed)

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

        prior_turn = ChatHistoryPair(
            user_question=ChatMessageRecord(
                message_id=1,
                session_id="session-1",
                role="user",
                content="What is revenue in FY2024?",
                hashed_content="hash-1",
            ),
            assistant=ChatMessageRecord(
                message_id=2,
                session_id="session-1",
                role="assistant",
                content=RagAnswer(
                    answer="100",
                    citations=["chunk-1"],
                    context_blocks=["Revenue was 100."],
                ).model_dump_json(),
                hashed_content="hash-2",
            ),
        )

        result = service.answer(
            "What is the requested figure?", "record-1", [prior_turn]
        )

        self.assertIsInstance(result, RagAnswer)
        self.assertEqual(result.answer, "100")
        self.assertEqual(result.citations, ["chunk-1"])
        self.assertIsNone(result.turn_program)
        self.assertEqual(len(model_client.prompts), 1)
        self.assertEqual(
            model_client.response_formats[0],
            "json",
        )
        self.assertEqual(retriever.calls[0][2].dates, [])
        self.assertIn("Conversation history:", model_client.prompts[0])
        self.assertIn(
            "User question: What is revenue in FY2024?", model_client.prompts[0]
        )

        repeat = service.answer("What is revenue?", "record-1")
        self.assertEqual(repeat.answer, "100")
        self.assertEqual(len(model_client.prompts), 2)

    def test_answer_sends_retrieved_results_to_model(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output=LOOKUP_ANSWER,
        )
        service = RAGService(
            model_client=model_client,
            retriever=ThreeResultRetriever(),
        )

        service.answer("What is the requested figure?", "record-1")

        prompt = model_client.prompts[0]
        self.assertIn("Revenue source 1.", prompt)
        self.assertIn("Revenue source 2.", prompt)
        self.assertIn("Revenue source 3.", prompt)

    def test_answer_keeps_scalar_answer_and_records_turn_program(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output=(
                '{"answer":"2","citations":["chunk-1"],'
                '"calculation_program":{"steps":[{"operation":"divide",'
                '"operands":[{"kind":"table_value","value_id":"chunk-1:value:0"},'
                '{"kind":"literal","literal":50.0}]}]},"requery":null}'
            ),
        )
        service = RAGService(
            model_client=model_client,
            retriever=FakeRetriever(),
        )

        result = service.answer("How many times larger was revenue?", "record-1")

        self.assertEqual(result.answer, "2")
        self.assertIsNotNone(result.calculation_program)
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

    def test_answer_keeps_calculated_scalar_answer(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output=(
                '{"answer":"50","citations":["chunk-1"],'
                '"calculation_program":{"steps":[{"operation":"subtract",'
                '"operands":[{"kind":"table_value","value_id":"chunk-1:value:0"},'
                '{"kind":"table_value","value_id":"chunk-1:value:1"}]}]},'
                '"requery":null}'
            ),
        )
        service = RAGService(
            model_client=model_client,
            retriever=FakeRetriever(),
        )

        result = service.answer("How much did revenue increase?", "record-1")

        self.assertEqual(result.answer, "50")
        self.assertIsNotNone(result.calculation_program)
        self.assertEqual(result.turn_program, "subtract(100, 50)")

    def test_answer_formats_computed_percentage_answer(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output=(
                '{"answer":"100%","citations":["chunk-1"],'
                '"calculation_program":{"steps":['
                '{"operation":"subtract",'
                '"operands":[{"kind":"table_value","value_id":"chunk-1:value:0"},'
                '{"kind":"table_value","value_id":"chunk-1:value:1"}]},'
                '{"operation":"divide",'
                '"operands":[{"kind":"step_result","step_index":0},'
                '{"kind":"table_value","value_id":"chunk-1:value:1"}]},'
                '{"operation":"percentage",'
                '"operands":[{"kind":"step_result","step_index":1}]}]},'
                '"requery":null}'
            ),
        )
        service = RAGService(
            model_client=model_client,
            retriever=FakeRetriever(),
        )

        result = service.answer("What was the percentage increase?", "record-1")

        self.assertEqual(result.answer, "100%")
        self.assertEqual(
            result.turn_program,
            "subtract(100, 50), divide(#0, 50), percentage(#1)",
        )

    def test_answer_uses_computed_answer_when_model_answer_differs(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output=(
                '{"answer":"3","citations":["chunk-1"],'
                '"calculation_program":{"steps":[{"operation":"divide",'
                '"operands":[{"kind":"table_value","value_id":"chunk-1:value:0"},'
                '{"kind":"literal","literal":50.0}]}]},"requery":null}'
            ),
        )
        service = RAGService(
            model_client=model_client,
            retriever=FakeRetriever(),
        )

        result = service.answer("How many times larger was revenue?", "record-1")

        self.assertEqual(result.answer, "2")
        self.assertEqual(result.turn_program, "divide(100, 50)")

    def test_answer_defaults_missing_calculation_program_to_lookup(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output='{"answer":"100","citations":["chunk-1"],"requery":null}',
        )
        service = RAGService(
            model_client=model_client,
            retriever=FakeRetriever(),
        )

        result = service.answer("What is revenue?", "record-1")

        self.assertEqual(result.answer, "100")
        self.assertIsNone(result.calculation_program)

    def test_answer_raises_when_calculation_program_fails(self) -> None:
        model_client = MockOllamaClient(
            model_name="test-model",
            chat_output=(
                '{"answer":"50","citations":["chunk-1"],'
                '"calculation_program":{"steps":[{"operation":"subtract",'
                '"operands":[{"kind":"table_value","value_id":"missing"},'
                '{"kind":"table_value","value_id":"chunk-1:value:1"}]}]},'
                '"requery":null}'
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
