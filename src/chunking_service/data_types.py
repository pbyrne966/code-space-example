"""Pydantic models for the ConvFinQA embedding pipeline."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

SplitName = Literal["train", "dev", "test"]


class Document(BaseModel):
    """Financial document context for a ConvFinQA record."""

    pre_text: str
    post_text: str
    table: dict[str, dict[str, Any]]


class Dialogue(BaseModel):
    """Conversational question/answer fields for a ConvFinQA record."""

    conv_questions: list[str]
    conv_answers: list[str]
    turn_program: list[str]
    executed_answers: list[float | int | str]
    qa_split: list[bool]


class Features(BaseModel):
    """Record-level flags derived during dataset cleaning."""

    num_dialogue_turns: int
    has_type2_question: bool
    has_duplicate_columns: bool
    has_non_numeric_values: bool


class ConvFinQARecord(BaseModel):
    """Raw cleaned ConvFinQA record."""

    id: str
    doc: Document
    dialogue: Dialogue
    features: Features


class ChunkType(Enum):
    """Supported retrieval chunk categories."""

    PRE_TEXT = "pre_text"
    POST_TEXT = "post_text"
    TABLE_ROW = "table_row"
    TABLE_METRIC = "table_metric"
    DIALOGUE_TURN = "dialogue_turn"


class RetrievalChunk(BaseModel):
    """Normalized evidence unit sent to lexical/vector retrieval indexes."""

    chunk_id: str
    record_id: str
    record_index: int
    chunk_index: int
    split: SplitName
    chunk_type: ChunkType
    text: str

    metric: str | None = None
    matched_metrics: list[str] = Field(default_factory=list)
    table_column: str | None = None
    turn_index: int | None = None
    qa_split: bool | None = None
    years: list[str] = Field(default_factory=list)
    months: list[int] = Field(default_factory=list)
    quarters: list[int] = Field(default_factory=list)
    days: list[int] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    period_labels: list[str] = Field(default_factory=list)

    has_type2_question: bool
    has_duplicate_columns: bool
    has_non_numeric_values: bool
    num_dialogue_turns: int


class EmbeddedChunk(BaseModel):
    """Retrieval chunk with an embedding vector and model provenance."""

    chunk: RetrievalChunk
    embedding: list[float]
    embedding_model: str


class ChunkRow(BaseModel):
    """Flat storage representation for JSONL/SQLite/vector-store adapters."""

    chunk_id: str
    record_id: str
    record_index: int
    chunk_index: int
    split: SplitName
    chunk_type: ChunkType
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawDictRecords(BaseModel):
    train: list[dict[str, Any]]
    dev: list[dict[str, Any]]
    test: list[dict[str, Any]]
