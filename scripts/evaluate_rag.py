from __future__ import annotations

import json
import random
from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer
from pydantic import BaseModel, Field

from src.data_types import ConvFinQARecord
from src.db_service.postgres_controllers import PostgresChatService
from src.logger import get_logger
from src.rag_service import RagAnswer
from src.runtime import build_context
from src.main import process_quistion, validate_app_state
from typing import Callable, List
from functools import partial

logger = get_logger("rag_evaluation")


class EvaluationExample(BaseModel):
    """One question/answer item to evaluate."""

    record_id: str
    turn_index: int
    question: str
    gold_answer: str | float | int
    gold_program: str | None = None
    split: str | None = None


class ExampleResult(BaseModel):
    """Evaluation result for one question."""

    example: EvaluationExample
    predicted_answer: RagAnswer | None = None
    latency_seconds: float | None = None
    retrieval_recall_at_k: bool | None = None
    answer_correct: bool | None = None
    citation_valid: bool | None = None
    error: str | None = None


class EvaluationSummary(BaseModel):
    """Aggregate metrics for an evaluation run."""

    total_examples: int = 0
    answered_examples: int = 0
    failed_examples: int = 0
    retrieval_recall_at_k: float | None = None
    answer_accuracy: float | None = None
    citation_validity: float | None = None
    average_latency_seconds: float | None = None
    problematic_record_ids: List[str] = []


class EvaluationReport(BaseModel):
    """Serializable evaluation payload."""

    summary: EvaluationSummary
    results: list[ExampleResult] = Field(default_factory=list)


def build_citation_ids(): ...


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
    random_sample = random.sample([d for d in records], k=sample_size)
    return random_sample


def build_examples(
    records: list[ConvFinQARecord],
) -> list[EvaluationExample]:
    """Flatten ConvFinQA dialogue turns into evaluation examples."""
    examples: list[EvaluationExample] = []

    for record in records:
        for turn_index, question in enumerate(record.dialogue.conv_questions):
            gold_program = (
                record.dialogue.turn_program[turn_index]
                if turn_index < len(record.dialogue.turn_program)
                else None
            )
            examples.append(
                EvaluationExample(
                    record_id=record.id,
                    turn_index=turn_index,
                    question=question,
                    gold_answer=record.dialogue.conv_answers[turn_index],
                    gold_program=gold_program,
                )
            )

    return examples


def evaluate_retrieval(
    example: EvaluationExample,
    top_k: int,
) -> bool:

    raise NotImplementedError("TODO: calculate retrieval recall at k")


def evaluate_answer(
    predicted: RagAnswer,
    example: EvaluationExample,
    numeric_tolerance: float,
) -> bool:
    """Return whether the predicted answer matches the gold answer."""
    matched = str(float(predicted.answer)) == str(float(example.gold_answer))
    if matched:
        return matched
    elif numeric_tolerance == 0.0:
        return False

    float_answer = float(predicted.answer)
    gold_answer = float(example.gold_answer)
    answers = [float_answer, gold_answer]
    diff = abs(max(answers) - min(answers))
    return diff > numeric_tolerance


def evaluate_citations(
    predicted: RagAnswer,
    example: EvaluationExample,
) -> bool:
    """Return whether citations are valid and support the answer."""
    # Have a build citation id function
    # This might have to come from parsing the query and grabing the source chunk
    example_citations = [example]
    return sorted(predicted.citations) == sorted(example_citations)


def evaluate_one(
    retieval_fn: Callable[[str, str, PostgresChatService], RagAnswer],
    chat_service: PostgresChatService,
    example: EvaluationExample,
    top_k: int,
    numeric_tolerance: float,
) -> ExampleResult:
    """Run the complete RAG evaluation path for one example."""

    start = perf_counter()
    try:
        chat_session = chat_service.get_or_create_session(record_id=example.record_id)
        predicted: RagAnswer = retieval_fn(
            message=example.message, record_id=example.record_id, session=chat_session
        )
        latency_seconds = perf_counter() - start

        return ExampleResult(
            example=example,
            predicted_answer=predicted,
            latency_seconds=latency_seconds,
            retrieval_recall_at_k=evaluate_retrieval(example, top_k),
            answer_correct=evaluate_answer(
                predicted,
                example,
                numeric_tolerance,
            ),
            citation_valid=evaluate_citations(predicted, example),
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


def summarize_results(results: list[ExampleResult], record_ids: List[str]) -> EvaluationSummary:
    """Aggregate per-example results into headline metrics."""
    # Look at the quistion and the timing -> see how quistion complexity influces the timing of the response
    
    failed_examples = 0
    problematic_record_id = set()


    for result in results:
        if result.error is not None or not result.answer_correct or not result.citation_valid:
            failed_examples += 1
            # Might to reference further here but this is fine for now
            problematic_record_id.add(result.record_id)

    evaluation_summary = EvaluationSummary(
        total_examples=len(record_ids),
        answered_examples=len(results),
        failed_examples=0
    )

def main(
    split: Annotated[str, typer.Option(help="Dataset split to evaluate.")] = "dev",
    limit: Annotated[
        int | None,
        typer.Option(help="Maximum number of examples to evaluate."),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option(help="Retrieval depth used for recall@k."),
    ] = 6,
    numeric_tolerance: Annotated[
        float,
        typer.Option(help="Absolute tolerance for numeric answer matching."),
    ] = 1e-3,
    output: Annotated[
        Path,
        typer.Option(help="Path for the JSON evaluation report."),
    ] = Path("evaluation_results.json"),
) -> None:
    """Run RAG evaluation for a ConvFinQA split."""
    context = build_context()
    validate_app_state(context)

    settings = context.settings
    records = load_records(settings.raw_data_path, split)
    example_records = pull_random_record_ids(records, limit or 10)
    examples = build_examples(example_records)

    retreival_fn = partial(
        process_quistion,
        settings=settings,
        chat_service=context.chat_service,
        answer_service=settings.answer_service,
    )

    results = [
        evaluate_one(
            retrieval_fn=retreival_fn,
            example=example,
            top_k=top_k,
            numeric_tolerance=numeric_tolerance,
        )
        for example in examples
    ]

    report = EvaluationReport(
        summary=summarize_results(results),
        results=results,
    )

    # typer.echo(report.summary.model_dump_json(indent=2))


if __name__ == "__main__":
    typer.run(main)
