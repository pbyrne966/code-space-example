"""SQLAlchemy ORM schemas for chunk storage."""

from collections.abc import Callable
from datetime import datetime

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
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


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


class ChunkEmbeddingTable(Base):
    """Embedding metadata linked to a retrieval chunk.

    The vector column itself can be added once the embedding model and dimension are
    fixed.
    """

    __tablename__ = "chunk_embeddings"
    __table_args__ = (Index("ix_chunk_embeddings_model", "embedding_model"),)

    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_chunks.chunk_id", ondelete="CASCADE"),
        primary_key=True,
    )

    embedding_model: Mapped[str] = mapped_column(String, primary_key=True)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(JSON_TYPE, nullable=False)

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


class PgVectorRetriever:
    def __init__(
        self,
        engine: Engine,
        embedding_fn: Callable[[str], list[float]],
        embedding_model,
        top_k: int = 6,
    ) -> None:
        self.engine = engine
        self.session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        self.embedding_fn = embedding_fn
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(self, query: str):
        query_embedding = self.embedding_fn(query)

        stmt = (
            select(
                RetrievalChunkTable,
                ChunkEmbeddingTable.embedding.cosine_distance(query_embedding).label(
                    "distance"
                ),
            )
            .join(
                RetrievalChunkTable,
                ChunkEmbeddingTable.chunk_id == ChunkEmbeddingTable.chunk_id,
            )
            .where(ChunkEmbeddingTable.embedding_model == self.embedding_model)
            .order_by("distance")
            .limit(self.top_k)
        )

        with self.session_factory() as session:
            return session.execute(stmt).all()


class ChatHistory(Base):
    __tablename__ = "chat_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
