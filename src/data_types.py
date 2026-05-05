"""Shared Pydantic models for source data, chunking, and retrieval."""

from enum import Enum
from typing import Any, Literal
from datetime import datetime

from pydantic import BaseModel, Field

from src.data_types import RetrievalChunk

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
    source_file: str | None = None
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


class SourceRecordMetadata(BaseModel):
    """Metadata for one source ConvFinQA record."""

    record_id: str
    source_file: str | None = None
    record_index: int
    split: SplitName
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
    """Flat storage representation for JSONL/vector-store adapters."""

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



class RetrievedChunkRecord(BaseModel):
    """Serializable retrieval result row."""

    chunk: RetrievalChunk
    distance: float


class ChatMessageRecord(BaseModel):
    """Serializable chat message row."""

    message_id: int | None = None
    session_id: str
    role: str
    content: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ChatHistoryPair(BaseModel):
    """Serializable user/assistant message pair."""

    user_question: ChatMessageRecord
    assistant: ChatMessageRecord


class ChatSessionRecord(BaseModel):
    """Serializable chat session row."""

    session_id: str
    record_id: str
    title: str | None = None
    message_count: int = 0
    last_message_index: int = -1
    created_at: datetime | None = None
    last_message_at: datetime | None = None
    updated_at: datetime | None = None
    messages: list[ChatMessageRecord] = Field(default_factory=list)
