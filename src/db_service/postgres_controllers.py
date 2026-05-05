from collections.abc import Callable, Iterable
from typing import Protocol
from uuid import UUID

from sqlalchemy import func, select, text, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.chunking_service.data_types import RetrievalChunk

from .data_types import ChatHistoryPair, ChatSessionRecord, RetrievedChunkRecord
from .mappers import (
    retrieval_chunk_from_table,
    retrieval_chunk_to_embedding_table,
    retrieval_chunk_to_table,
)
from .schemas import (
    ChatExchange,
    ChatSession,
    ChunkEmbeddingTable,
    RetrievalChunkTable,
)

import hashlib

class PostgresControllerContract(Protocol):
    def has_data(self) -> bool: ...
    def setup(self): ...


class ChunkStore(PostgresControllerContract, Protocol):
    """Storage boundary for flattened retrieval chunks."""

    def add_chunks(
        self,
        chunks: Iterable[RetrievalChunk],
        embedding_fn: Callable[[str], list[float]],
        embedding_model: str,
    ) -> None: ...

    def get_chunks_for_record(self, record_id: str) -> list[RetrievalChunk]: ...


class PostgresChatService(PostgresControllerContract):
    def __init__(self, engine: Engine):
        self.engine = engine
        self.session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def setup(self):
        return

    def _cache_key(self, prompt: str):
        hashed_quistion = hashlib.sha256(prompt.encode("utf-8")).digest()
        with self.session_factory() as session:
            select(ChatExchange)

    def create_session(self, record_id: str) -> ChatSessionRecord:
        query = ChatSession(record_id=record_id)
        with self.session_factory() as session:
            session.add(query)
            session.commit()
            session.refresh(query)
            return query.to_pydantic()

    def get_session(self, record_id: str) -> ChatSessionRecord | None:
        query = select(ChatSession).where(ChatSession.record_id == record_id).limit(1)

        with self.session_factory() as session:
            row = session.execute(query).scalar_one_or_none()
            if row is None:
                return None
            return row.to_pydantic()

    def get_or_create_session(self, record_id: str) -> ChatSessionRecord:
        session = self.get_session(record_id)
        if session is None:
            session = self.create_session(record_id)
        return session

    def append_message(self, role: str, session_id: UUID | str, content: str) -> None:
        session_id_str = str(session_id)
        with self.session_factory() as session:
            session.add(
                ChatExchange(session_id=session_id_str, role=role, content=content)
            )

            session.execute(
                update(ChatSession)
                .where(ChatSession.session_id == session_id_str)
                .values(
                    message_count=ChatSession.message_count + 1,
                    last_message_index=ChatSession.last_message_index + 1,
                    last_message_at=func.now(),
                )
            )
            session.commit()

    def start_or_resume_session(self, record_id: str) -> ChatSessionRecord:
        return self.get_or_create_session(record_id)

    def record_user_message(self, session_id: UUID | str, content: str) -> None:
        self.append_message("user", session_id, content)

    def record_assistant_message(self, session_id: UUID | str, content: str) -> None:
        self.append_message("assistant", session_id, content)

    def show_history(
        self, session_id: UUID | str, limit: int = 10
    ) -> list[ChatHistoryPair]:
        # Query last N messages for both user and assistant types.
        lim_each = limit // 2
        session_id_str = str(session_id)
        user_messages = (
            select(ChatExchange)
            .where(
                (ChatExchange.session_id == session_id_str)
                & (ChatExchange.role == "user")
            )
            .order_by(ChatExchange.created_at.desc())
            .limit(lim_each)
        )

        assistant_answers = (
            select(ChatExchange)
            .where(
                (ChatExchange.session_id == session_id_str)
                & (ChatExchange.role == "assistant")
            )
            .order_by(ChatExchange.created_at.desc())
            .limit(lim_each)
        )

        with self.session_factory() as session:
            user = session.execute(user_messages).scalars().all()
            assistant = session.execute(assistant_answers).scalars().all()
            together = [
                ChatHistoryPair(
                    user_question=user_item.to_pydantic(),
                    assistant=assistant_item.to_pydantic(),
                )
                for user_item, assistant_item in zip(user, assistant)
            ]

            return together

    def has_data(self) -> bool:
        statement = select(func.count()).select_from(ChatExchange)
        with self.session_factory() as session:
            count = session.execute(statement).scalar_one()
            return count > 0


class PostgresChunkStore(ChunkStore):
    """Postgres implementation for retrieval chunk storage."""

    def __init__(
        self,
        engine: Engine,
        embedding_fn: Callable[[str], list[float]] | None = None,
        embedding_model: str | None = None,
        top_k: int = 6,
    ) -> None:
        self.engine = engine
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.embedding_fn = embedding_fn
        self.embedding_model = embedding_model
        self.top_k = top_k

    def setup(self) -> None:
        if self.engine.dialect.name != "postgresql":
            raise ValueError("PostgresChunkStore requires a PostgreSQL engine")
        with self.engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

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

    def retrieve(
        self,
        query: str,
        embedding_fn: Callable[[str], list[float]] | None = None,
        embedding_model: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunkRecord]:
        embedding_fn = embedding_fn or self.embedding_fn
        embedding_model = embedding_model or self.embedding_model
        top_k = top_k or self.top_k

        if embedding_fn is None or embedding_model is None:
            raise ValueError("retrieve requires an embedding function and model")

        query_embedding = embedding_fn(query)
        stmt = (
            select(
                RetrievalChunkTable,
                ChunkEmbeddingTable.embedding.cosine_distance(query_embedding).label(
                    "distance"
                ),
            )
            .join(
                RetrievalChunkTable,
                RetrievalChunkTable.chunk_id == ChunkEmbeddingTable.chunk_id,
            )
            .where(ChunkEmbeddingTable.embedding_model == embedding_model)
            .order_by("distance")
            .limit(top_k)
        )

        with self.session_factory() as session:
            return [
                RetrievedChunkRecord(
                    chunk=row.RetrievalChunkTable.to_pydantic(),
                    distance=row.distance,
                )
                for row in session.execute(stmt).all()
            ]

    def has_data(self) -> bool:
        """Return whether any retrieval chunks exist for the active database."""
        statement = select(func.count()).select_from(RetrievalChunkTable)

        with self.session_factory() as session:
            count = session.execute(statement).scalar_one()
            return count > 0
