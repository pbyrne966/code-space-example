"""Pydantic read models for database service rows."""

from datetime import datetime

from pydantic import BaseModel, Field

from src.data_types import RetrievalChunk


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
