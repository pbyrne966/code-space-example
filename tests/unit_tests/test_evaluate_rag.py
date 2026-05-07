from pathlib import Path

from scripts.evaluate_rag import (
    EvaluationExample,
    ExampleResult,
    build_citation_ids,
    evaluate_answer,
    evaluate_one,
    summarize_results,
)
from src.chunking_service.chunking import chunk_record
from src.data_types import ConvFinQARecord
from src.rag_service import RagAnswer


def sample_record() -> ConvFinQARecord:
    """Build a small record with enough table structure for citation tests."""
    return ConvFinQARecord(
        id="Example/File.pdf-1",
        doc={
            "pre_text": "Company context before the table.",
            "post_text": "Company context after the table.",
            "table": {
                "2024": {
                    "revenue": 100.0,
                    "operating income": 25.0,
                },
                "2023": {
                    "revenue": 80.0,
                    "operating income": 20.0,
                },
            },
        },
        dialogue={
            "conv_questions": [
                "What was revenue in 2024?",
                "What was the change in revenue?",
            ],
            "conv_answers": ["100.0", "20.0"],
            "turn_program": ["100.0", "subtract(100.0, 80.0)"],
            "executed_answers": [100.0, 20.0],
            "qa_split": [False, False],
        },
        features={
            "num_dialogue_turns": 2,
            "has_type2_question": False,
            "has_duplicate_columns": False,
            "has_non_numeric_values": False,
        },
    )


def test_build_citation_ids_returns_record_chunk_ids() -> None:
    """Citation IDs are the chunk IDs generated for the source record."""
    record = sample_record()
    source_file = Path("raw_eval.json")

    citation_ids = build_citation_ids(
        record=record,
        split="dev",
        record_index=0,
        source_file=source_file,
    )
    chunks = chunk_record(
        record=record,
        split="dev",
        record_index=0,
        source_file=source_file,
    )

    assert citation_ids == [chunk.chunk_id for chunk in chunks]


def test_evaluate_answer_accepts_percentage_values() -> None:
    """Percentage answers compare as numeric values after stripping percent signs."""
    example = EvaluationExample(
        record_id="record-1",
        turn_index=0,
        question="What was the percentage change?",
        gold_answer="142.4%",
        question_index="record_idx:0:question_idx:0",
    )
    predicted = RagAnswer(answer="142.4%", citations=[])

    assert evaluate_answer(predicted, example, numeric_tolerance=0.0)


def test_summarize_results_reports_failed_examples() -> None:
    """Summary keeps the computed failed count and problematic record IDs."""
    example = EvaluationExample(
        record_id="record-1",
        turn_index=0,
        question="What was revenue?",
        gold_answer="100.0",
        expected_citation_ids=["chunk-1"],
        question_index="record_idx:0:question_idx:0",
    )
    result = ExampleResult(
        example=example,
        latency_seconds=1.0,
        answer_correct=False,
        citation_valid=True,
    )

    summary = summarize_results([result], [sample_record()])

    assert summary.failed_examples == 1
    assert summary.problematic_record_ids == ["record-1"]


def test_evaluate_one_records_error_when_prediction_is_missing() -> None:
    """Missing predictions are handled in normal Python, even with asserts disabled."""
    example = EvaluationExample(
        record_id="record-1",
        turn_index=0,
        question="What was revenue?",
        gold_answer="100.0",
        expected_citation_ids=["chunk-1"],
        question_index="record_idx:0:question_idx:0",
    )

    class FakeChatService:
        def get_or_create_session(self, record_id: str):
            return "session"

    result = evaluate_one(
        retrieval_fn=lambda question, record_id, session: None,
        chat_service=FakeChatService(),  # type: ignore[arg-type]
        example=example,
    )

    assert result.error == "Could not predict"
