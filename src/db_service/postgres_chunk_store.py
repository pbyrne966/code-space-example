"""Postgres-backed chunk store."""

from collections.abc import Callable, Iterable
from typing import Protocol

from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.chunking_service.data_types import RetrievalChunk

from .mappers import (
    retrieval_chunk_from_table,
    retrieval_chunk_to_embedding_table,
    retrieval_chunk_to_table,
)
from .schemas import Base, RetrievalChunkTable


class ChunkStore(Protocol):
    """Storage boundary for flattened retrieval chunks."""

    def setup(self) -> None:
        """Prepare the backing store for chunk writes and reads."""
        raise NotImplementedError

    def add_chunks(
        self,
        chunks: Iterable[RetrievalChunk],
        embedding_fn: Callable[[str], list[float]],
        embedding_model: str,
    ) -> None:
        """Persist retrieval chunks."""
        raise NotImplementedError

    def get_chunks_for_record(self, record_id: str) -> list[RetrievalChunk]:
        """Fetch chunks linked to one parent record."""
        raise NotImplementedError


class PostgresChunkStore(ChunkStore):
    """Postgres implementation for retrieval chunk storage."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def setup(self) -> None:
        if self.engine.dialect.name == "postgresql":
            with self.engine.begin() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(self.engine)

    def add_chunks(
        self,
        chunks: Iterable[RetrievalChunk],
        embedding_fn: Callable[[str], list[float]],
        embedding_model: str,
    ) -> None:
        """Persist retrieval chunks."""
        with self.session_factory() as session:
            for chunk in chunks:
                session.merge(retrieval_chunk_to_table(chunk))
                session.merge(
                    retrieval_chunk_to_embedding_table(
                        chunk, embedding_fn, embedding_model
                    )
                )
            session.commit()

    def get_chunks_for_record(self, record_id: str) -> list[RetrievalChunk]:
        """Fetch chunks linked to one parent record."""
        with self.session_factory() as session:
            rows = self._get_rows_for_record(session, record_id)
            return [retrieval_chunk_from_table(row) for row in rows]

    def _get_rows_for_record(
        self,
        session: Session,
        record_id: str,
    ) -> list[RetrievalChunkTable]:
        statement = (
            select(RetrievalChunkTable)
            .where(RetrievalChunkTable.record_id == record_id)
            .order_by(RetrievalChunkTable.chunk_index)
        )
        return list(session.scalars(statement))
