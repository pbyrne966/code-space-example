"""SQLAlchemy ORM schemas for chunk storage."""

from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

from src.data_types import ChunkType, EmbeddedChunk, RetrievalChunk
from src.db_service.data_types import ChatMessageRecord, ChatSessionRecord

JSON_TYPE = JSON().with_variant(JSONB, "postgresql")
MAX_EMBEDDING_DIMENSION = 896


class Base(DeclarativeBase):
    """Base class for database tables."""


class RetrievalChunkTable(Base):
    """Flattened retrieval chunk metadata and text."""

    __tablename__ = "retrieval_chunks"
    __table_args__ = (
        CheckConstraint("record_index >= 0", name="ck_retrieval_chunks_record_index"),
        CheckConstraint("chunk_index >= 0", name="ck_retrieval_chunks_chunk_index"),
        Index("ix_retrieval_chunks_record_chunk", "record_id", "chunk_index"),
        Index("ix_retrieval_chunks_split_record", "split", "record_id"),
    )

    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    record_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    record_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    split: Mapped[str] = mapped_column(String, nullable=False, index=True)
    chunk_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    metric: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    matched_metrics: Mapped[list[str]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=list,
    )
    table_column: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        index=True,
    )
    turn_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    qa_split: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    years: Mapped[list[str]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    months: Mapped[list[int]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    quarters: Mapped[list[int]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    days: Mapped[list[int]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    dates: Mapped[list[str]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    period_labels: Mapped[list[str]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=list,
    )

    has_type2_question: Mapped[bool] = mapped_column(Boolean, nullable=False)
    has_duplicate_columns: Mapped[bool] = mapped_column(Boolean, nullable=False)
    has_non_numeric_values: Mapped[bool] = mapped_column(Boolean, nullable=False)
    num_dialogue_turns: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    embeddings: Mapped[list["ChunkEmbeddingTable"]] = relationship(
        back_populates="chunk",
        cascade="all, delete-orphan",
    )

    def to_pydantic(self) -> RetrievalChunk:
        """Serialize the ORM row into the retrieval chunk read model."""
        return RetrievalChunk(
            chunk_id=self.chunk_id,
            record_id=self.record_id,
            record_index=self.record_index,
            chunk_index=self.chunk_index,
            split=self.split,  # type: ignore[arg-type]
            chunk_type=ChunkType(self.chunk_type),
            text=self.text,
            metric=self.metric,
            matched_metrics=self.matched_metrics,
            table_column=self.table_column,
            turn_index=self.turn_index,
            qa_split=self.qa_split,
            years=self.years,
            months=self.months,
            quarters=self.quarters,
            days=self.days,
            dates=self.dates,
            period_labels=self.period_labels,
            has_type2_question=self.has_type2_question,
            has_duplicate_columns=self.has_duplicate_columns,
            has_non_numeric_values=self.has_non_numeric_values,
            num_dialogue_turns=self.num_dialogue_turns,
        )


class ChunkEmbeddingTable(Base):
    """Embedding metadata linked to a retrieval chunk."""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (Index("ix_chunk_embeddings_model", "embedding_model"),)

    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_chunks.chunk_id", ondelete="CASCADE"),
        primary_key=True,
    )

    embedding_model: Mapped[str] = mapped_column(String, primary_key=True)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(MAX_EMBEDDING_DIMENSION),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    chunk: Mapped[RetrievalChunkTable] = relationship(back_populates="embeddings")

    def to_pydantic(self) -> EmbeddedChunk:
        """Serialize the ORM row into the embedded chunk read model."""
        return EmbeddedChunk(
            chunk=self.chunk.to_pydantic(),
            embedding=list(self.embedding),
            embedding_model=self.embedding_model,
        )


class ChatSession(Base):
    """Chat session metadata for conversation persistence."""

    __tablename__ = "chat_sessions"
    __table_args__ = (
        CheckConstraint("message_count >= 0", name="ck_chat_sessions_message_count"),
        CheckConstraint(
            "last_message_index >= -1",
            name="ck_chat_sessions_last_message_index",
        ),
        Index(
            "ix_chat_sessions_record_last_message_at",
            "record_id",
            "last_message_at",
        ),
        Index("ix_chat_sessions_last_message_at", "last_message_at"),
    )

    session_id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    record_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)

    message_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    last_message_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("-1"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    last_message_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    messages: Mapped[list["ChatExchange"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatExchange.message_id",
    )

    def to_pydantic(self) -> ChatSessionRecord:
        """Serialize the ORM row into the chat session read model."""
        return ChatSessionRecord(
            session_id=self.session_id,
            record_id=self.record_id,
            title=self.title,
            message_count=self.message_count,
            last_message_index=self.last_message_index,
            created_at=self.created_at,
            last_message_at=self.last_message_at,
            updated_at=self.updated_at,
            messages=[message.to_pydantic() for message in self.messages],
        )


class ChatExchange(Base):
    """Individual chat messages stored in append order."""

    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant', 'system', 'tool')",
            name="ck_chat_messages_role",
        ),
        Index("ix_chat_messages_session_created_at", "session_id", "created_at"),
    )

    message_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    hashed_content: Mapped[str] = mapped_column(String(64), nullable=False)
    linked_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_messages.message_id"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    linked_message: Mapped["ChatExchange | None"] = relationship(
        foreign_keys=[linked_message_id],
        remote_side=[message_id],
        back_populates="linked_messages",
    )

    linked_messages: Mapped[list["ChatExchange"]] = relationship(
        foreign_keys=[linked_message_id],
        back_populates="linked_message",
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")

    def to_pydantic(self) -> ChatMessageRecord:
        """Serialize the ORM row into the chat message read model."""
        return ChatMessageRecord(
            message_id=self.message_id,
            session_id=self.session_id,
            role=self.role,
            content=self.content,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
