"""Type conversion helpers between Pydantic models and ORM tables."""

from collections.abc import Callable

from src.data_types import RetrievalChunk, SourceRecordMetadata

from .schemas import (
    MAX_EMBEDDING_DIMENSION,
    ChunkEmbeddingTable,
    RetrievalChunkTable,
    SourceRecordTable,
)


def source_record_from_chunk(chunk: RetrievalChunk) -> SourceRecordTable:
    """Build source record metadata from a retrieval chunk."""
    return SourceRecordTable(
        record_id=chunk.record_id,
        source_file=chunk.source_file,
        record_index=chunk.record_index,
        split=chunk.split,
        has_type2_question=chunk.has_type2_question,
        has_duplicate_columns=chunk.has_duplicate_columns,
        has_non_numeric_values=chunk.has_non_numeric_values,
        num_dialogue_turns=chunk.num_dialogue_turns,
    )


def source_record_to_table(record: SourceRecordMetadata) -> SourceRecordTable:
    """Convert source record metadata into an ORM table object."""
    return SourceRecordTable(
        record_id=record.record_id,
        source_file=record.source_file,
        record_index=record.record_index,
        split=record.split,
        has_type2_question=record.has_type2_question,
        has_duplicate_columns=record.has_duplicate_columns,
        has_non_numeric_values=record.has_non_numeric_values,
        num_dialogue_turns=record.num_dialogue_turns,
    )


def retrieval_chunk_to_table(chunk: RetrievalChunk) -> RetrievalChunkTable:
    """Convert a retrieval chunk Pydantic model into an ORM table object."""
    return RetrievalChunkTable(
        chunk_id=chunk.chunk_id,
        record_id=chunk.record_id,
        source_file=chunk.source_file,
        record_index=chunk.record_index,
        chunk_index=chunk.chunk_index,
        split=chunk.split,
        chunk_type=chunk.chunk_type.value,
        text=chunk.text,
        metric=chunk.metric,
        matched_metrics=chunk.matched_metrics,
        table_column=chunk.table_column,
        years=chunk.years,
        months=chunk.months,
        quarters=chunk.quarters,
        days=chunk.days,
        dates=chunk.dates,
        period_labels=chunk.period_labels,
        has_type2_question=chunk.has_type2_question,
        has_duplicate_columns=chunk.has_duplicate_columns,
        has_non_numeric_values=chunk.has_non_numeric_values,
        num_dialogue_turns=chunk.num_dialogue_turns,
    )


def retrieval_chunk_from_table(row: RetrievalChunkTable) -> RetrievalChunk:
    """Convert an ORM retrieval chunk row back into a Pydantic model."""
    return row.to_pydantic()


def retrieval_chunk_to_embedding_table(
    chunk: RetrievalChunk,
    embedding_fn: Callable[[str], list[float]],
    embedding_model: str,
) -> ChunkEmbeddingTable:
    embedding = embedding_fn(chunk.text)
    if not embedding:
        raise ValueError("embedding_fn returned an empty embedding vector")
    if len(embedding) != MAX_EMBEDDING_DIMENSION:
        raise ValueError(
            "embedding_fn returned an embedding vector with "
            f"{len(embedding)} dimensions; expected {MAX_EMBEDDING_DIMENSION}"
        )

    return ChunkEmbeddingTable(
        chunk_id=chunk.chunk_id,
        embedding_model=embedding_model,
        embedding_dimension=len(embedding),
        embedding=embedding,
    )
