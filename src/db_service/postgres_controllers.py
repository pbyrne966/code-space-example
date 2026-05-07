import hashlib
from collections.abc import Callable, Iterable
from typing import Protocol
from uuid import UUID

from sqlalchemy import func, literal, select, text, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.chunking_service.period_extraction import PeriodData
from src.data_types import (
    ChatHistoryPair,
    ChatMessageRecord,
    ChatSessionRecord,
    RetrievalChunk,
    RetrievedChunkRecord,
    SourceRecordMetadata,
)

from .mappers import (
    retrieval_chunk_from_table,
    retrieval_chunk_to_embedding_table,
    retrieval_chunk_to_table,
    source_record_from_chunk,
)
from .schemas import (
    ChatExchange,
    ChatSession,
    ChunkEmbeddingTable,
    RetrievalChunkTable,
    SourceRecordTable,
)


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

    def get_cached(self, prompt: str, record_id: str) -> ChatMessageRecord | None:
        hashed_question = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        with self.session_factory() as session:
            found = session.execute(
                select(ChatExchange)
                .join(ChatSession)
                .where(
                    (ChatSession.record_id == record_id)
                    & (ChatExchange.hashed_content == hashed_question)
                    & (ChatExchange.role == "user")
                    & (ChatExchange.invalid.is_(False))
                )
                .order_by(ChatExchange.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()

            if found is None:
                return None

            answer = session.execute(
                select(ChatExchange)
                .where(
                    (ChatExchange.linked_message_id == found.message_id)
                    & (ChatExchange.role == "assistant")
                    & (ChatExchange.invalid.is_(False))
                )
                .order_by(ChatExchange.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if answer is None:
                return None
            return answer.to_pydantic()

    def soft_delete(self, message_id: int, hashed_content: str) -> bool:
        statement = (
            select(ChatExchange)
            .where(
                (ChatExchange.hashed_content == hashed_content)
                & (ChatExchange.message_id == message_id)
                & (ChatExchange.role == "assistant")
            )
            .limit(1)
        )
        with self.session_factory() as session:
            message = session.execute(statement).scalar_one_or_none()
            if message is None:
                return False

            message.invalid = True
            session.commit()
            return True

    def create_session(self, record_id: str) -> ChatSessionRecord:
        query = ChatSession(record_id=record_id)
        with self.session_factory() as session:
            if session.get(SourceRecordTable, record_id) is None:
                raise ValueError(f"Source record does not exist: {record_id}")
            session.add(query)
            session.commit()
            session.refresh(query)
            return query.to_pydantic()

    def get_session(self, record_id: str) -> ChatSessionRecord | None:
        query = (
            select(ChatSession)
            .where(ChatSession.record_id == record_id)
            .order_by(ChatSession.last_message_at.desc(), ChatSession.created_at.desc())
            .limit(1)
        )

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

    def append_message(
        self,
        role: str,
        session_id: UUID | str,
        content: str,
        linked_message: ChatMessageRecord | None = None,
    ) -> ChatMessageRecord:
        session_id_str = str(session_id)
        with self.session_factory() as session:
            hashed_content = hashlib.sha256(content.encode("utf-8")).hexdigest()
            linked_msg_id = (
                linked_message.message_id if linked_message is not None else None
            )
            chat_exchange = ChatExchange(
                session_id=session_id_str,
                role=role,
                content=content,
                hashed_content=hashed_content,
                linked_message_id=linked_msg_id,
            )
            session.add(chat_exchange)
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
            session.refresh(chat_exchange)
            return chat_exchange.to_pydantic()

    def start_or_resume_session(self, record_id: str) -> ChatSessionRecord:
        return self.get_or_create_session(record_id)

    def record_user_message(
        self,
        session_id: UUID | str,
        content: str,
    ) -> ChatMessageRecord:
        return self.append_message("user", session_id, content)

    def record_assistant_message(
        self, session_id: UUID | str, content: str, linked_message: ChatMessageRecord
    ) -> None:
        self.append_message("assistant", session_id, content, linked_message)

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
                & (ChatExchange.invalid.is_(False))
            )
            .order_by(ChatExchange.created_at.desc())
            .limit(lim_each)
        )

        with self.session_factory() as session:
            users = session.execute(user_messages).scalars().all()
            user_ids = [user.message_id for user in users]
            assistants = (
                session.execute(
                    select(ChatExchange)
                    .where(
                        (ChatExchange.linked_message_id.in_(user_ids))
                        & (ChatExchange.role == "assistant")
                        & (ChatExchange.invalid.is_(False))
                    )
                    .order_by(ChatExchange.created_at.desc())
                )
                .scalars()
                .all()
            )
            assistant_by_user_id = {
                assistant.linked_message_id: assistant for assistant in assistants
            }

            return [
                ChatHistoryPair(
                    user_question=user.to_pydantic(),
                    assistant=assistant.to_pydantic(),
                )
                for user in users
                if (assistant := assistant_by_user_id.get(user.message_id)) is not None
            ]

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
            chunks = list(chunks)
            source_records = {
                chunk.record_id: source_record_from_chunk(chunk) for chunk in chunks
            }
            for source_record in source_records.values():
                session.merge(source_record)
            session.flush()
            for chunk in chunks:
                session.merge(retrieval_chunk_to_table(chunk))
                embedding_row = retrieval_chunk_to_embedding_table(
                    chunk, embedding_fn, embedding_model
                )
                self._validate_embedding_dimension(
                    session,
                    embedding_model,
                    embedding_row.embedding_dimension,
                )
                session.merge(embedding_row)
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
        record_id: str,
        period_data: PeriodData | None = None,
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
        if not query_embedding:
            raise ValueError("embedding_fn returned an empty embedding vector")
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
            .where(
                ChunkEmbeddingTable.embedding_model == embedding_model,
                RetrievalChunkTable.record_id == record_id,
            )
            .order_by("distance")
            .limit(top_k)
        )
        stmt = self._apply_period_filters(stmt, period_data)

        with self.session_factory() as session:
            self._validate_embedding_dimension(
                session,
                embedding_model,
                len(query_embedding),
            )
            return [
                RetrievedChunkRecord(
                    chunk=row.RetrievalChunkTable.to_pydantic(),
                    distance=row.distance,
                )
                for row in session.execute(stmt).all()
            ]

    def _validate_embedding_dimension(
        self,
        session: Session,
        embedding_model: str,
        embedding_dimension: int,
    ) -> None:
        dimensions = set(
            session.execute(
                select(ChunkEmbeddingTable.embedding_dimension)
                .where(ChunkEmbeddingTable.embedding_model == embedding_model)
                .distinct()
            ).scalars()
        )
        if not dimensions:
            return
        if dimensions == {embedding_dimension}:
            return

        stored = ", ".join(str(dimension) for dimension in sorted(dimensions))
        raise ValueError(
            "Embedding dimension mismatch for model "
            f"{embedding_model!r}: stored chunks use {stored} dimensions, "
            f"but the current embedding function returned {embedding_dimension}. "
            "Re-ingest chunks with the configured embedding model, or configure "
            "the embedding model that created the stored rows."
        )

    def _apply_period_filters(self, statement, period_data: PeriodData | None):
        if period_data is None:
            return statement

        filters = []
        if period_data.dates:
            filters.append(
                self._jsonb_array_contains(RetrievalChunkTable.dates, period_data.dates)
            )
        else:
            if period_data.years:
                filters.append(
                    self._jsonb_array_contains(
                        RetrievalChunkTable.years, period_data.years
                    )
                )
            if period_data.quarters:
                filters.append(
                    self._jsonb_array_contains(
                        RetrievalChunkTable.quarters, period_data.quarters
                    )
                )
            if period_data.months:
                filters.append(
                    self._jsonb_array_contains(
                        RetrievalChunkTable.months, period_data.months
                    )
                )
            if period_data.days:
                filters.append(
                    self._jsonb_array_contains(
                        RetrievalChunkTable.days, period_data.days
                    )
                )

        if filters:
            return statement.where(*filters)

        return statement

    @staticmethod
    def _jsonb_array_contains(column, values):
        return column.op("@>")(literal(values, type_=JSONB))

    def has_data(self) -> bool:
        """Return whether any retrieval chunks exist for the active database."""
        statement = select(func.count()).select_from(RetrievalChunkTable)

        with self.session_factory() as session:
            count = session.execute(statement).scalar_one()
            return count > 0

    def get_source_record(self, record_id: str) -> SourceRecordMetadata | None:
        """Fetch source record metadata by source record id."""
        with self.session_factory() as session:
            record = session.get(SourceRecordTable, record_id)
            if record is None:
                return None
            return record.to_pydantic()
