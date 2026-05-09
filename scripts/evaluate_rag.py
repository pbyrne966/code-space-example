from __future__ import annotations

import json
import random
import re
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, cast

import typer
from pydantic import BaseModel, Field

from scripts.evaluate_rag_metrics import display_metrics_json
from src.chunking_service.chunking import chunk_record
from src.data_types import ConvFinQARecord, RetrievalChunk, SplitName
from src.logger import get_logger
from src.rag_service import RagAnswer
from src.runtime import build_context

logger = get_logger("rag_evaluation")
RetrievalFn = Callable[[str, str], RagAnswer | None]
NUMERIC_REFERENCE_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")


class EvaluationExample(BaseModel):
    """One question/answer item to evaluate."""

    record_id: str
    turn_index: int
    question: str
    gold_answer: str | float | int
    gold_program: str | None = None
    split: str | None = None
    expected_citation_ids: list[str] = Field(default_factory=list)
    question_index: str


class ExampleResult(BaseModel):
    """Evaluation result for one question."""

    example: EvaluationExample
    predicted_answer: RagAnswer | None = None
    latency_seconds: float | None = None
    retrieval_recall_at_k: bool | None = None
    answer_correct: bool | None = None
    citation_valid: bool | None = None
    error: str | None = None
    tokens_used: int = 0


class QuestionComplexity(BaseModel):
    tokens_used: int
    question_index: str
    expected_citation_ids: list[str]


class ContextWindowPerformanceResult(BaseModel):
    tokens_used: int
    quistion_correct: bool | None = None
    latency_seconds: float | None = None


class UnrelatedQuistionResult(BaseModel):
    quistion: str
    as_expected: bool
    answer: str | None = None


class Qustions(BaseModel):
    quistions: list[str]


# TODO: Imeplement this brother
# Prescion & Recall -> The Rock Curve and F1 Score


def _normalize_expected_fallback(answer: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", answer.lower())


def _normalize_text_answer(answer: str | float | int) -> str:
    return re.sub(r"\s+", " ", str(answer).strip().lower())


def context_window_performace(
    context_window_data_path: Path,
    split: str,
    retrieval_fn: RetrievalFn,
) -> dict[str, ContextWindowPerformanceResult]:
    context_window_records = load_records(context_window_data_path, split)
    quistions = build_examples(
        context_window_records,
        cast(SplitName, split),
        context_window_data_path,
    )
    context_window_performance: dict[str, ContextWindowPerformanceResult] = {}
    for quistion in quistions:
        evaluated = evaluate_one(retrieval_fn, quistion)
        step_id = f"{quistion.record_id}:{quistion.question_index}"
        context_window_performance[step_id] = ContextWindowPerformanceResult(
            tokens_used=evaluated.tokens_used,
            quistion_correct=evaluated.answer_correct,
            latency_seconds=evaluated.latency_seconds,
        )

    return context_window_performance


def unrelated_quistions(
    record_id: str,
    unrelated_questions_path: Path,
    retrieval_fn: RetrievalFn,
    compare_to: str = "I dont know",
) -> list[UnrelatedQuistionResult]:
    unrelated_quistons = Qustions(**json.loads(unrelated_questions_path.read_text()))
    unrelated_qustions_output: list[UnrelatedQuistionResult] = []

    for question in unrelated_quistons.quistions:
        predicted: RagAnswer | None = retrieval_fn(question, record_id)
        is_expected = predicted is None
        if predicted is not None:
            is_expected = _normalize_expected_fallback(
                predicted.answer
            ) == _normalize_expected_fallback(compare_to)

        unrelated_qustions_output.append(
            UnrelatedQuistionResult(
                quistion=question,
                as_expected=is_expected,
                answer=predicted.answer if predicted is not None else None,
            )
        )

    return unrelated_qustions_output


class EvaluationSummary(BaseModel):
    """Aggregate metrics for an evaluation run."""

    total_examples: int = 0
    answered_examples: int = 0
    failed_examples: int = 0
    answer_accuracy: float | None = None
    citation_validity: float | None = None
    average_latency_seconds: float | None = None
    problematic_record_ids: list[str] = Field(default_factory=list)
    question_complexity: dict[str, list[QuestionComplexity]] = Field(
        default_factory=dict
    )
    prompts_above_avg_latency: list[str]
    complexity_per_quistion: dict[str, list[QuestionComplexity]]


class EvaluationReport(BaseModel):
    summary: EvaluationSummary
    results: list[ExampleResult] = Field(default_factory=list)
    context_window_performance: dict[str, ContextWindowPerformanceResult] = Field(
        default_factory=dict
    )
    unrelated_quistions: list[UnrelatedQuistionResult] = Field(default_factory=list)


def build_citation_ids(
    record: ConvFinQARecord,
    split: SplitName,
    record_index: int,
    source_file: Path,
) -> list[str]:
    """Build every valid citation chunk ID for a source record."""
    chunks = chunk_record(
        record=record,
        split=split,
        record_index=record_index,
        source_file=source_file,
    )
    return [chunk.chunk_id for chunk in chunks]


def _numeric_references(*values: str | float | int | None) -> set[float]:
    references: set[float] = set()
    for value in values:
        if value is None:
            continue
        for match in NUMERIC_REFERENCE_RE.findall(str(value)):
            references.add(float(match.replace(",", "").rstrip("%")))
    return references


def _chunk_contains_numeric_reference(
    chunk: RetrievalChunk,
    numeric_references: set[float],
) -> bool:
    return any(
        table_value.numeric_value in numeric_references
        for table_value in chunk.table_values
        if table_value.numeric_value is not None
    )


def build_supporting_citation_ids(
    record: ConvFinQARecord,
    split: SplitName,
    record_index: int,
    source_file: Path,
    gold_answer: str | float | int,
    gold_program: str | None,
) -> list[str]:
    """Build citation chunk IDs that contain values used by this specific turn."""
    chunks = chunk_record(
        record=record,
        split=split,
        record_index=record_index,
        source_file=source_file,
    )
    numeric_references = _numeric_references(gold_program, gold_answer)
    if not numeric_references:
        return []

    return [
        chunk.chunk_id
        for chunk in chunks
        if _chunk_contains_numeric_reference(chunk, numeric_references)
    ]


def _parse_answer_number(answer: str | float | int) -> float:
    return float(str(answer).strip().replace(",", "").rstrip("%"))


def load_records(dataset_path: Path, split: str) -> list[ConvFinQARecord]:
    """Load ConvFinQA records for the requested split."""
    raw_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if split not in raw_payload:
        raise ValueError(f"Unknown split '{split}' in {dataset_path}")

    raw_records = raw_payload[split]
    if not isinstance(raw_records, list):
        raise ValueError(f"Expected split '{split}' to contain a list of records")

    return [ConvFinQARecord(**record) for record in raw_records]


def pull_random_record_ids(
    records: list[ConvFinQARecord], sample_size: int = 10, seed: int | None = 42
) -> list[ConvFinQARecord]:
    """Return a random sample of source records for evaluation."""
    sample_size = min(sample_size, len(records))
    rng = random.Random(seed)
    random_sample = rng.sample(records, k=sample_size)
    return random_sample


def build_examples(
    records: list[ConvFinQARecord],
    split: SplitName,
    source_file: Path,
) -> list[EvaluationExample]:
    """Flatten ConvFinQA dialogue turns into evaluation examples."""
    examples: list[EvaluationExample] = []

    for idx, record in enumerate(records):
        for turn_index, question in enumerate(record.dialogue.conv_questions):
            gold_program = (
                record.dialogue.turn_program[turn_index]
                if turn_index < len(record.dialogue.turn_program)
                else None
            )
            example = EvaluationExample(
                record_id=record.id,
                turn_index=turn_index,
                question=question,
                gold_answer=record.dialogue.conv_answers[turn_index],
                gold_program=gold_program,
                split=split,
                expected_citation_ids=build_supporting_citation_ids(
                    record=record,
                    split=split,
                    record_index=idx,
                    source_file=source_file,
                    gold_answer=record.dialogue.conv_answers[turn_index],
                    gold_program=gold_program,
                ),
                question_index=f"record_idx:{idx}:question_idx:{turn_index}",
            )
            examples.append(example)

    return examples


def evaluate_answer(
    predicted: RagAnswer,
    example: EvaluationExample,
    numeric_tolerance: float,
) -> bool:
    """Return whether the predicted answer matches the gold answer."""
    try:
        predicted_number = _parse_answer_number(predicted.answer)
        gold_number = _parse_answer_number(example.gold_answer)
    except ValueError:
        return _normalize_text_answer(predicted.answer) == _normalize_text_answer(
            example.gold_answer
        )

    matched = str(predicted_number) == str(gold_number)
    if matched:
        return matched
    elif numeric_tolerance == 0.0:
        return False

    return abs(predicted_number - gold_number) <= numeric_tolerance


def evaluate_citations(
    predicted: RagAnswer,
    example: EvaluationExample,
) -> bool:
    """Return whether citations are valid and support the answer."""
    expected_citations = set(example.expected_citation_ids)
    if not predicted.citations or not expected_citations:
        return False
    return any(citation in expected_citations for citation in predicted.citations)


def evaluate_one(
    retrieval_fn: RetrievalFn,
    example: EvaluationExample,
    numeric_tolerance: float = 0.0,
) -> ExampleResult:
    """Run the complete RAG evaluation path for one example."""
    start = perf_counter()
    try:
        predicted: RagAnswer | None = retrieval_fn(example.question, example.record_id)
        latency_seconds = perf_counter() - start

        if predicted is None:
            raise ValueError("Could not predict")

        return ExampleResult(
            example=example,
            predicted_answer=predicted,
            latency_seconds=latency_seconds,
            answer_correct=evaluate_answer(
                predicted,
                example,
                numeric_tolerance,
            ),
            citation_valid=evaluate_citations(predicted, example),
            tokens_used=predicted.tokens_used,
        )

    except Exception as exc:
        logger.exception(
            "Evaluation failed for record_id=%s turn_index=%s",
            example.record_id,
            example.turn_index,
        )
        return ExampleResult(
            example=example,
            latency_seconds=perf_counter() - start,
            error=str(exc),
        )


def summarize_results(
    results: list[ExampleResult], records: list[Any]
) -> EvaluationSummary:
    """Aggregate per-example results into headline metrics."""
    # Look at question timing to understand how complexity influences response time.

    failed_examples = 0
    problematic_record_id: OrderedDict[str, None] = OrderedDict()
    average_latency_array = [
        latency.latency_seconds
        for latency in results
        if latency.latency_seconds is not None
    ]
    average_latency = (
        sum(average_latency_array) / len(average_latency_array)
        if average_latency_array
        else None
    )
    amount_of_tokens_per_question = defaultdict(list)
    prompts_above_latency = []

    valid_citations = 0
    valid_answers = 0

    for result in results:
        valid_answers += result.answer_correct or 0
        valid_citations += result.citation_valid or 0

        amount_of_tokens_per_question[result.example.record_id].append(
            QuestionComplexity(
                **{
                    "tokens_used": result.tokens_used,
                    "question_index": result.example.question_index,
                    "expected_citation_ids": result.example.expected_citation_ids,
                }
            )
        )

        if (
            result.error is not None
            or not result.answer_correct
            or not result.citation_valid
        ):
            failed_examples += 1
            problematic_record_id[result.example.record_id] = None

        if average_latency is not None and (result.latency_seconds or 0.0) > average_latency:
            prompts_above_latency.append(
                f"{result.example.record_id}:{result.example.question_index}"
            )

    answer_accuracy = valid_answers / len(results) if results else None
    citation_validity = valid_citations / len(results) if results else None

    evaluation_summary = EvaluationSummary(
        total_examples=len(results),
        answered_examples=sum(1 for result in results if result.error is None),
        failed_examples=failed_examples,
        problematic_record_ids=[*problematic_record_id.keys()],
        question_complexity=cast(
            dict[str, list[QuestionComplexity]], dict(amount_of_tokens_per_question)
        ),
        prompts_above_avg_latency=prompts_above_latency,
        average_latency_seconds=average_latency,
        complexity_per_quistion=amount_of_tokens_per_question,
        citation_validity=citation_validity,
        answer_accuracy=answer_accuracy,
    )

    return evaluation_summary


def default_metrics_output_path(output: Path) -> Path:
    """Return the default display-metrics path for an evaluation report path."""
    return output.with_name(f"{output.stem}_metrics{output.suffix}")


def main(
    split: Annotated[str, typer.Option(help="Dataset split to evaluate.")] = "dev",
    limit: Annotated[
        int | None,
        typer.Option(help="Maximum number of source records to sample."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(help="Random seed for source-record sampling."),
    ] = 42,
    numeric_tolerance: Annotated[
        float,
        typer.Option(help="Absolute tolerance for numeric answer matching."),
    ] = 1e-3,
    output: Annotated[
        Path,
        typer.Option(help="Path for the JSON evaluation report."),
    ] = Path("evaluation_results.json"),
    metrics_output: Annotated[
        Path | None,
        typer.Option(help="Path for the display metrics JSON report."),
    ] = None,
    context_window_data_path: Path | None = None,
    unrelated_questions_path: Path | None = None,
) -> None:
    """Run RAG evaluation for a ConvFinQA split."""
    context = build_context()
    settings = context.settings
    records = load_records(settings.raw_data_path, split)  # type: ignore
    example_records = pull_random_record_ids(records, limit or 10, seed)
    examples = build_examples(
        example_records,
        cast(SplitName, split),
        settings.raw_data_path,
    )

    def retrieve_partial(message: str, record_id: str) -> RagAnswer | None:
        response = context.answer_service.answer(
            message,
            record_id,
            session_history=[],
        )
        if response.requery is None:
            return response

        requery = response.requery
        response = context.answer_service.answer(
            requery,
            record_id,
            session_history=[],
            is_requery=True,
        )
        return response.model_copy(update={"requery": requery})

    results = [
        evaluate_one(
            retrieval_fn=retrieve_partial,
            example=example,
            numeric_tolerance=numeric_tolerance,
        )
        for example in examples
    ]
    result_summary = summarize_results(results, example_records)

    context_window_output = {}
    if context_window_data_path is not None:
        context_window_output = context_window_performace(
            context_window_data_path=context_window_data_path,
            split=split,
            retrieval_fn=retrieve_partial,
        )

    unrelated_output = []
    if unrelated_questions_path is not None:
        unrelated_output = unrelated_quistions(
            record_id=example_records[0].id,
            unrelated_questions_path=unrelated_questions_path,
            retrieval_fn=retrieve_partial,
        )

    report = EvaluationReport(
        summary=result_summary,
        results=results,
        context_window_performance=context_window_output,
        unrelated_quistions=unrelated_output,
    )

    output.write_text(report.model_dump_json(indent=4), encoding="utf-8")
    metrics_output = metrics_output or default_metrics_output_path(output)
    metrics_output.write_text(display_metrics_json(report), encoding="utf-8")


if __name__ == "__main__":
    typer.run(main)
