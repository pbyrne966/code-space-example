import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Sequence

from src.data_types import (
    ChunkType,
    ConvFinQARecord,
    RetrievalChunk,
    SplitName,
    TableValue,
)

from .period_extraction import extract_period_data

NormalizedMetric = tuple[str, str]


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
NON_WORD_RE = re.compile(r"[^a-z0-9%]+")
STABLE_KEY_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class TextWindow:
    text: str
    start_sentence: int
    end_sentence: int


def _stable_key(raw_text: str) -> str:
    return STABLE_KEY_RE.sub("_", raw_text.lower()).strip("_") or "UNKNOWN"


def _build_chunk_id(*parts: str) -> str:
    return ":".join(_stable_key(part) for part in parts)


def _format_table_row(record_id: str, table_column: str, values: dict[str, Any]) -> str:
    value_text = " ".join(f"{metric}: {value}." for metric, value in values.items())
    return f"Record {record_id}. Table column {table_column}. {value_text}"


def _format_table_metric(
    record_id: str, metric: str, column_values: dict[str, Any]
) -> str:
    value_text = " ".join(
        f"{table_column}: {value}." for table_column, value in column_values.items()
    )
    return f"Record {record_id}. Table metric {metric}. {value_text}"


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _table_values_for_column(
    table_column: str,
    values: dict[str, Any],
) -> Sequence[TableValue]:
    return [
        TableValue(
            metric=metric,
            table_column=table_column,
            value=value,
            numeric_value=_numeric_value(value),
        )
        for metric, value in values.items()
    ]


def _table_values_for_metric(
    metric: str,
    column_values: dict[str, Any],
) -> Sequence[TableValue]:
    return [
        TableValue(
            metric=metric,
            table_column=table_column,
            value=value,
            numeric_value=_numeric_value(value),
        )
        for table_column, value in column_values.items()
    ]

def _normalize_metric_text(text: str) -> str:
    return NON_WORD_RE.sub(" ", text.lower()).strip()


def _extract_table_metrics(table: dict[str, dict[str, Any]]) -> list[NormalizedMetric]:
    metrics: list[NormalizedMetric] = []
    seen_metrics: set[str] = set()
    for values in table.values():
        for metric in values:
            normalized_metric = _normalize_metric_text(metric)
            if normalized_metric and normalized_metric not in seen_metrics:
                seen_metrics.add(normalized_metric)
                metrics.append((metric, normalized_metric))
    return metrics


def _find_metrics_in_text(text: str, metrics: list[NormalizedMetric]) -> list[str]:
    normalized_text = f" {_normalize_metric_text(text)} "
    matched_metrics: list[str] = []
    for metric, normalized_metric in metrics:
        if f" {normalized_metric} " in normalized_text:
            matched_metrics.append(metric)

    return matched_metrics


def create_common_fields(
    record: ConvFinQARecord, record_index: int, split: SplitName, source_file: Path
) -> dict[str, Any]:
    return {
        "record_id": record.id,
        "source_file": source_file.as_posix(),
        "record_index": record_index,
        "split": split,
        "has_type2_question": record.features.has_type2_question,
        "has_duplicate_columns": record.features.has_duplicate_columns,
        "has_non_numeric_values": record.features.has_non_numeric_values,
        "num_dialogue_turns": record.features.num_dialogue_turns,
    }


def _transpose_table_by_metric(
    table: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    metric_values: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    for table_column, values in table.items():
        for metric, value in values.items():
            metric_values[metric][table_column] = value
    return dict(metric_values)


def chunk_record(
    record: ConvFinQARecord, split: SplitName, record_index: int, source_file: Path
) -> list[RetrievalChunk]:
    chunk_records: list[RetrievalChunk] = []
    chunk_counter = count()
    common_fields = create_common_fields(record, record_index, split, source_file)
    table_metrics = _extract_table_metrics(record.doc.table)

    source_file_name = source_file.name

    def append_chunk(
        local_id: str,
        chunk_type: ChunkType,
        text: str,
        metric: str | None = None,
        matched_metrics: list[str] | None = None,
        table_column: str | None = None,
        table_values: list[dict[str, TableValue]] | None = None,
        years: list[str] | None = None,
        months: list[int] | None = None,
        quarters: list[int] | None = None,
        days: list[int] | None = None,
        dates: list[str] | None = None,
        period_labels: list[str] | None = None,
    ) -> None:
        chunk_index = next(chunk_counter)
        chunk_records.append(
            RetrievalChunk(
                chunk_id=_build_chunk_id(
                    source_file_name,
                    split,
                    record.id,
                    chunk_type.value,
                    local_id,
                ),
                chunk_index=chunk_index,
                chunk_type=chunk_type,
                text=text,
                metric=metric,
                matched_metrics=matched_metrics or [],
                table_column=table_column,
                table_values=table_values or [],  # type: ignore
                years=years or [],
                months=months or [],
                quarters=quarters or [],
                days=days or [],
                dates=dates or [],
                period_labels=period_labels or [],
                **common_fields,
            )
        )

    for column_index, (table_column, values) in enumerate(record.doc.table.items()):
        period_data = extract_period_data([table_column])
        append_chunk(
            local_id=f"column_index{column_index}_{_stable_key(table_column)}",
            chunk_type=ChunkType.TABLE_ROW,
            text=_format_table_row(record.id, table_column, values),
            table_column=table_column,
            table_values=_table_values_for_column(table_column, values),  # type: ignore
            years=period_data.years,
            months=period_data.months,
            quarters=period_data.quarters,
            days=period_data.days,
            dates=period_data.dates,
            period_labels=period_data.period_labels,
        )

    for metric_index, (metric, column_values) in enumerate(
        _transpose_table_by_metric(record.doc.table).items()
    ):
        column_keys = list(column_values.keys())
        period_data = extract_period_data(column_keys)
        append_chunk(
            local_id=f"table_metric_{metric_index}_{_stable_key(metric)}",
            chunk_type=ChunkType.TABLE_METRIC,
            text=_format_table_metric(record.id, metric, column_values),
            metric=metric,
            table_values=_table_values_for_metric(metric, column_values),  # type: ignore
            years=period_data.years,
            months=period_data.months,
            quarters=period_data.quarters,
            days=period_data.days,
            dates=period_data.dates,
            period_labels=period_data.period_labels,
        )

    return chunk_records
