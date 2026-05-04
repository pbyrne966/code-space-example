"""Type conversion helpers between Pydantic models and ORM tables."""

from collections.abc import Callable

from src.chunking_service.data_types import ChunkType, RetrievalChunk

from .schemas import ChunkEmbeddingTable, RetrievalChunkTable


def retrieval_chunk_to_table(chunk: RetrievalChunk) -> RetrievalChunkTable:
    """Convert a retrieval chunk Pydantic model into an ORM table object."""
    return RetrievalChunkTable(
        chunk_id=chunk.chunk_id,
        record_id=chunk.record_id,
        record_index=chunk.record_index,
        chunk_index=chunk.chunk_index,
        split=chunk.split,
        chunk_type=chunk.chunk_type.value,
        text=chunk.text,
        metric=chunk.metric,
        matched_metrics=chunk.matched_metrics,
        table_column=chunk.table_column,
        turn_index=chunk.turn_index,
        qa_split=chunk.qa_split,
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
    return RetrievalChunk(
        chunk_id=row.chunk_id,
        record_id=row.record_id,
        record_index=row.record_index,
        chunk_index=row.chunk_index,
        split=row.split,  # type: ignore[arg-type]
        chunk_type=ChunkType(row.chunk_type),
        text=row.text,
        metric=row.metric,
        matched_metrics=row.matched_metrics,
        table_column=row.table_column,
        turn_index=row.turn_index,
        qa_split=row.qa_split,
        years=row.years,
        months=row.months,
        quarters=row.quarters,
        days=row.days,
        dates=row.dates,
        period_labels=row.period_labels,
        has_type2_question=row.has_type2_question,
        has_duplicate_columns=row.has_duplicate_columns,
        has_non_numeric_values=row.has_non_numeric_values,
        num_dialogue_turns=row.num_dialogue_turns,
    )


def retrieval_chunk_to_embedding_table(
    chunk: RetrievalChunk,
    embedding_fn: Callable[[str], list[float]],
    embedding_model: str,
) -> ChunkEmbeddingTable:
    embedding = embedding_fn(chunk.text)
    if not embedding:
        raise ValueError("embedding_fn returned an empty embedding vector")

    return ChunkEmbeddingTable(
        chunk_id=chunk.chunk_id,
        embedding_model=embedding_model,
        embedding_dimension=len(embedding),
        embedding=embedding,
    )
