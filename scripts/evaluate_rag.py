from __future__ import annotations

import json
import random
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, cast

import typer
from pydantic import BaseModel, Field

from src.chunking_service.chunking import chunk_record
from src.data_types import ChatSessionRecord, ConvFinQARecord, SplitName
from src.db_service.postgres_controllers import PostgresChatService
from src.logger import get_logger
from src.main import process_question
from src.rag_service import RagAnswer
from src.runtime import build_context
import re

logger = get_logger("rag_evaluation")


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


def context_window_performace(
    context_window_data_path: Path,
    split: str,
    chat_service: PostgresChatService,
    retrieval_fn: Callable[[str, str, ChatSessionRecord], RagAnswer | None],
) -> dict[str, ContextWindowPerformanceResult]:
    context_window_records = load_records(context_window_data_path, split)
    quistions = build_examples(
        context_window_records,
        cast(SplitName, split),
        context_window_data_path,
    )
    context_window_performance: dict[str, ContextWindowPerformanceResult] = {}
    for step_idx, quistion in enumerate(quistions):
        evaluated = evaluate_one(retrieval_fn, chat_service, quistion)
        step_id = f"{step_idx}:{evaluated.predicted_answer.answer if evaluated.predicted_answer is not None else 'null'}"
        context_window_performance[step_id] = ContextWindowPerformanceResult(
            tokens_used=evaluated.tokens_used,
            quistion_correct=evaluated.answer_correct,
            latency_seconds=evaluated.latency_seconds,
        )

    return context_window_performance


def unrelated_quistions(
    record_id: str,
    chat_service: PostgresChatService,
    unrelated_questions_path: Path,
    retrieval_fn: Callable[[str, str, ChatSessionRecord], RagAnswer | None],
    compare_to: str = "I dont know",
) -> list[UnrelatedQuistionResult]:
    unrelated_quistons = Qustions(**json.loads(unrelated_questions_path.read_text()))
    unrelated_qustions_output: list[UnrelatedQuistionResult] = []
    chat_session = chat_service.get_or_create_session(record_id=record_id)

    for question in unrelated_quistons.quistions:
        predicted: RagAnswer | None = retrieval_fn(question, record_id, chat_session)
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


def _parse_answer_number(answer: str | float | int) -> float:
    return float(str(answer).strip().rstrip("%"))


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
    records: list[ConvFinQARecord], sample_size: int = 10
) -> list[ConvFinQARecord]:
    """Return a random sample of source records for evaluation."""
    random_sample = random.sample([d for d in records], k=sample_size)
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
                expected_citation_ids=build_citation_ids(
                    record=record,
                    split=split,
                    record_index=idx,
                    source_file=source_file,
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
    matched = str(_parse_answer_number(predicted.answer)) == str(
        _parse_answer_number(example.gold_answer)
    )
    if matched:
        return matched
    elif numeric_tolerance == 0.0:
        return False

    float_answer = _parse_answer_number(predicted.answer)
    gold_answer = _parse_answer_number(example.gold_answer)
    return abs(float_answer - gold_answer) <= numeric_tolerance


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
    retrieval_fn: Callable[[str, str, ChatSessionRecord], RagAnswer | None],
    chat_service: PostgresChatService,
    example: EvaluationExample,
    numeric_tolerance: float = 0.0,
) -> ExampleResult:
    """Run the complete RAG evaluation path for one example."""
    start = perf_counter()
    try:
        chat_session = chat_service.get_or_create_session(record_id=example.record_id)
        predicted: RagAnswer | None = retrieval_fn(
            example.question, example.record_id, chat_session
        )
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
    average_latency = sum(average_latency_array) // (len(average_latency_array) or 1)
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

        if (result.latency_seconds or 0.0) > average_latency:
            prompts_above_latency.append(
                f"{result.example.record_id}:{result.example.question_index}"
            )

    answer_accuracy = valid_answers / len(results) if results else None
    citation_validity = valid_citations / len(results) if results else None

    evaluation_summary = EvaluationSummary(
        total_examples=len(records),
        answered_examples=len(results),
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


def main(
    split: Annotated[str, typer.Option(help="Dataset split to evaluate.")] = "dev",
    limit: Annotated[
        int | None,
        typer.Option(help="Maximum number of examples to evaluate."),
    ] = None,
    numeric_tolerance: Annotated[
        float,
        typer.Option(help="Absolute tolerance for numeric answer matching."),
    ] = 1e-3,
    output: Annotated[
        Path,
        typer.Option(help="Path for the JSON evaluation report."),
    ] = Path("evaluation_results.json"),
    context_window_data_path: Path | None = None,
    unrelated_questions_path: Path | None = None,
) -> None:
    """Run RAG evaluation for a ConvFinQA split."""
    context = build_context()
    settings = context.settings
    records = load_records(settings.raw_data_path, split)  # type: ignore
    example_records = pull_random_record_ids(records, limit or 10)
    examples = build_examples(
        example_records,
        cast(SplitName, split),
        settings.raw_data_path,
    )

    def retrieve_partial(
        message: str, record_id: str, session: ChatSessionRecord
    ) -> RagAnswer | None:
        return process_question(
            message,
            record_id,
            session=session,
            settings=settings,
            chat_service=context.chat_service,
            answer_service=context.answer_service,
        )

    results = [
        evaluate_one(
            retrieval_fn=retrieve_partial,
            chat_service=context.chat_service,
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
            chat_service=context.chat_service,
            retrieval_fn=retrieve_partial,
        )

    unrelated_output = []
    if unrelated_questions_path is not None:
        unrelated_output = unrelated_quistions(
            record_id=example_records[0].id,
            chat_service=context.chat_service,
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


if __name__ == "__main__":
    typer.run(main)
