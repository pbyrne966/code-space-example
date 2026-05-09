import json
from pathlib import Path

from scripts.evaluate_rag import (
    ContextWindowPerformanceResult,
    EvaluationExample,
    EvaluationReport,
    EvaluationSummary,
    ExampleResult,
    UnrelatedQuistionResult,
    build_citation_ids,
    build_examples,
    build_supporting_citation_ids,
    context_window_performace,
    default_metrics_output_path,
    evaluate_answer,
    evaluate_citations,
    evaluate_one,
    pull_random_record_ids,
    summarize_results,
)
from scripts.evaluate_rag_metrics import build_display_metrics, display_metrics_json
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


def test_build_examples_uses_turn_specific_supporting_citation_ids() -> None:
    """Expected citations are scoped to values used by the current turn."""
    record = sample_record()
    source_file = Path("raw_eval.json")

    examples = build_examples([record], split="dev", source_file=source_file)
    first_turn_expected = build_supporting_citation_ids(
        record=record,
        split="dev",
        record_index=0,
        source_file=source_file,
        gold_answer="100.0",
        gold_program="100.0",
    )
    all_record_citations = build_citation_ids(
        record=record,
        split="dev",
        record_index=0,
        source_file=source_file,
    )

    assert examples[0].expected_citation_ids == first_turn_expected
    assert set(examples[0].expected_citation_ids) < set(all_record_citations)


def test_evaluate_citations_rejects_same_record_irrelevant_chunk() -> None:
    """A citation from the same record is invalid unless it supports the turn."""
    record = sample_record()
    source_file = Path("raw_eval.json")
    chunks = chunk_record(
        record=record,
        split="dev",
        record_index=0,
        source_file=source_file,
    )
    expected_citations = build_supporting_citation_ids(
        record=record,
        split="dev",
        record_index=0,
        source_file=source_file,
        gold_answer="100.0",
        gold_program="100.0",
    )
    irrelevant_chunk = next(
        chunk.chunk_id for chunk in chunks if chunk.chunk_id not in expected_citations
    )
    example = EvaluationExample(
        record_id=record.id,
        turn_index=0,
        question="What was revenue in 2024?",
        gold_answer="100.0",
        expected_citation_ids=expected_citations,
        question_index="record_idx:0:question_idx:0",
    )

    assert not evaluate_citations(
        RagAnswer(answer="100.0", citations=[irrelevant_chunk]),
        example,
    )


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


def test_evaluate_answer_handles_text_values() -> None:
    """Text answers compare as normalized strings instead of evaluation errors."""
    example = EvaluationExample(
        record_id="record-1",
        turn_index=0,
        question="Who is the auditor?",
        gold_answer="I don't know",
        question_index="record_idx:0:question_idx:0",
    )
    predicted = RagAnswer(answer="i don't know", citations=[])

    assert evaluate_answer(predicted, example, numeric_tolerance=0.0)


def test_pull_random_record_ids_clamps_and_uses_seed() -> None:
    """Oversized samples do not crash and seeded samples are reproducible."""
    records = [
        sample_record().model_copy(update={"id": f"record-{idx}"})
        for idx in range(3)
    ]

    first_sample = pull_random_record_ids(records, sample_size=10, seed=7)
    second_sample = pull_random_record_ids(records, sample_size=10, seed=7)

    assert len(first_sample) == len(records)
    assert [record.id for record in first_sample] == [
        record.id for record in second_sample
    ]


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

    assert summary.total_examples == 1
    assert summary.answered_examples == 1
    assert summary.failed_examples == 1
    assert summary.problematic_record_ids == ["record-1"]


def test_summarize_results_uses_true_average_latency() -> None:
    """Average latency keeps fractional seconds."""
    example = EvaluationExample(
        record_id="record-1",
        turn_index=0,
        question="What was revenue?",
        gold_answer="100.0",
        expected_citation_ids=["chunk-1"],
        question_index="record_idx:0:question_idx:0",
    )
    results = [
        ExampleResult(
            example=example,
            latency_seconds=0.25,
            answer_correct=True,
            citation_valid=True,
        ),
        ExampleResult(
            example=example,
            latency_seconds=0.75,
            answer_correct=True,
            citation_valid=True,
        ),
    ]

    summary = summarize_results(results, [sample_record()])

    assert summary.average_latency_seconds == 0.5


def test_display_metrics_json_returns_raw_counts_and_rates() -> None:
    """Display metrics expose auditable counts and percentages as JSON."""
    example = EvaluationExample(
        record_id="record-1",
        turn_index=0,
        question="What was revenue?",
        gold_answer="100.0",
        expected_citation_ids=["chunk-1"],
        question_index="record_idx:0:question_idx:0",
    )
    results = [
        ExampleResult(
            example=example,
            latency_seconds=0.25,
            answer_correct=True,
            citation_valid=True,
            tokens_used=100,
        ),
        ExampleResult(
            example=example,
            latency_seconds=0.75,
            answer_correct=True,
            citation_valid=False,
            tokens_used=200,
        ),
        ExampleResult(
            example=example,
            latency_seconds=1.25,
            error="Could not predict",
            tokens_used=0,
        ),
    ]
    summary = EvaluationSummary(
        total_examples=3,
        answered_examples=2,
        failed_examples=2,
        problematic_record_ids=["record-1"],
        prompts_above_avg_latency=[],
        question_complexity={},
        complexity_per_quistion={},
        answer_accuracy=2 / 3,
        citation_validity=1 / 3,
    )
    report = EvaluationReport(
        summary=summary,
        results=results,
        context_window_performance={
            "record-1:question-1": ContextWindowPerformanceResult(
                tokens_used=50,
                quistion_correct=True,
                latency_seconds=0.5,
            )
        },
        unrelated_quistions=[
            UnrelatedQuistionResult(quistion="Who directed Inception?", as_expected=True),
            UnrelatedQuistionResult(
                quistion="Which studio produced Spirited Away?",
                as_expected=False,
                answer="Studio Ghibli",
            ),
        ],
    )

    metrics = build_display_metrics(report)
    metrics_json = json.loads(display_metrics_json(report))

    assert metrics_json == metrics
    assert metrics["core"]["total_examples"] == 3
    assert metrics["core"]["answered_examples"] == 2
    assert metrics["answer_quality"]["answer_accuracy_percent"] == 66.67
    assert metrics["citation_quality"]["citation_validity_percent"] == 33.33
    assert metrics["rag_success"]["fully_correct"] == 1
    assert metrics["failure_breakdown"]["invalid_citation_only"] == 1
    assert metrics["failure_breakdown"]["runtime_or_model_errors"] == 1
    assert metrics["latency"]["p95_seconds"] == 1.25
    assert metrics["tokens"]["total_tokens"] == 300
    assert metrics["context_window"]["accuracy_percent"] == 100.0
    assert metrics["unrelated_questions"]["abstention_accuracy_percent"] == 50.0


def test_default_metrics_output_path_uses_report_stem() -> None:
    """Metrics output defaults beside the report with a metrics suffix."""
    output = Path("data/evaluation/evaluation_results.json")

    assert default_metrics_output_path(output) == Path(
        "data/evaluation/evaluation_results_metrics.json"
    )


def test_context_window_keys_are_stable_question_ids(tmp_path: Path) -> None:
    """Context-window keys do not include mutable predicted answers."""
    payload = {"dev": [sample_record().model_dump()]}
    dataset_path = tmp_path / "context_window.json"
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    output = context_window_performace(
        context_window_data_path=dataset_path,
        split="dev",
        retrieval_fn=lambda question, record_id: RagAnswer(
            answer="answer text that should not be in the key",
            citations=[],
        ),
    )

    assert all("answer text" not in key for key in output)
    assert "Example/File.pdf-1:record_idx:0:question_idx:0" in output


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

    result = evaluate_one(
        retrieval_fn=lambda question, record_id: None,
        example=example,
    )

    assert result.error == "Could not predict"
