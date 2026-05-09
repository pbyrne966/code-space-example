from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.evaluate_rag import EvaluationReport


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percent(rate: float | None) -> float | None:
    return round(rate * 100, 2) if rate is not None else None


def _average(values: list[float | int]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float | int], percentile: float) -> float | int | None:
    if not values:
        return None

    sorted_values = sorted(values)
    index = math.ceil((percentile / 100) * len(sorted_values)) - 1
    return sorted_values[max(index, 0)]


def build_display_metrics(report: EvaluationReport) -> dict[str, Any]:
    """Build display-ready raw counts and rates from a processed evaluation report."""
    results = report.results
    total_examples = len(results)
    errored_examples = sum(1 for result in results if result.error is not None)
    answered_examples = total_examples - errored_examples
    correct_answers = sum(result.answer_correct is True for result in results)
    incorrect_answers = sum(result.answer_correct is False for result in results)
    valid_citations = sum(result.citation_valid is True for result in results)
    invalid_citations = sum(result.citation_valid is False for result in results)
    fully_correct = sum(
        result.answer_correct is True and result.citation_valid is True
        for result in results
    )
    wrong_answer_only = sum(
        result.answer_correct is False and result.citation_valid is True
        for result in results
    )
    invalid_citation_only = sum(
        result.answer_correct is True and result.citation_valid is False
        for result in results
    )
    wrong_answer_and_citation = sum(
        result.answer_correct is False and result.citation_valid is False
        for result in results
    )
    latency_values = [
        result.latency_seconds
        for result in results
        if result.latency_seconds is not None
    ]
    token_values = [result.tokens_used for result in results]

    context_results = list(report.context_window_performance.values())
    context_total = len(context_results)
    context_correct = sum(
        result.quistion_correct is True for result in context_results
    )
    context_latency_values = [
        result.latency_seconds
        for result in context_results
        if result.latency_seconds is not None
    ]
    context_token_values = [result.tokens_used for result in context_results]

    unrelated_total = len(report.unrelated_quistions)
    unrelated_expected = sum(result.as_expected for result in report.unrelated_quistions)
    unrelated_unexpected = unrelated_total - unrelated_expected

    answer_accuracy = _rate(correct_answers, total_examples)
    citation_validity = _rate(valid_citations, total_examples)
    full_success_rate = _rate(fully_correct, total_examples)
    context_accuracy = _rate(context_correct, context_total)
    abstention_accuracy = _rate(unrelated_expected, unrelated_total)

    return {
        "core": {
            "total_examples": total_examples,
            "answered_examples": answered_examples,
            "errored_examples": errored_examples,
            "failed_examples": report.summary.failed_examples,
            "problematic_record_count": len(report.summary.problematic_record_ids),
            "problematic_record_ids": report.summary.problematic_record_ids,
        },
        "answer_quality": {
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
            "answer_accuracy": answer_accuracy,
            "answer_accuracy_percent": _percent(answer_accuracy),
        },
        "citation_quality": {
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "citation_validity": citation_validity,
            "citation_validity_percent": _percent(citation_validity),
        },
        "rag_success": {
            "fully_correct": fully_correct,
            "full_success_rate": full_success_rate,
            "full_success_percent": _percent(full_success_rate),
        },
        "failure_breakdown": {
            "wrong_answer_only": wrong_answer_only,
            "invalid_citation_only": invalid_citation_only,
            "wrong_answer_and_invalid_citation": wrong_answer_and_citation,
            "runtime_or_model_errors": errored_examples,
        },
        "latency": {
            "average_seconds": _average(latency_values),
            "min_seconds": min(latency_values) if latency_values else None,
            "max_seconds": max(latency_values) if latency_values else None,
            "p50_seconds": _percentile(latency_values, 50),
            "p95_seconds": _percentile(latency_values, 95),
        },
        "tokens": {
            "total_tokens": sum(token_values),
            "average_tokens": _average(token_values),
            "min_tokens": min(token_values) if token_values else None,
            "max_tokens": max(token_values) if token_values else None,
            "p50_tokens": _percentile(token_values, 50),
            "p95_tokens": _percentile(token_values, 95),
        },
        "context_window": {
            "total_questions": context_total,
            "correct_questions": context_correct,
            "incorrect_questions": context_total - context_correct,
            "accuracy": context_accuracy,
            "accuracy_percent": _percent(context_accuracy),
            "average_latency_seconds": _average(context_latency_values),
            "average_tokens": _average(context_token_values),
        },
        "unrelated_questions": {
            "total_questions": unrelated_total,
            "expected_abstentions": unrelated_expected,
            "unexpected_answers": unrelated_unexpected,
            "abstention_accuracy": abstention_accuracy,
            "abstention_accuracy_percent": _percent(abstention_accuracy),
        },
    }


def display_metrics_json(report: EvaluationReport, indent: int = 4) -> str:
    """Return display-ready evaluation metrics as formatted JSON."""
    return json.dumps(build_display_metrics(report), indent=indent)
