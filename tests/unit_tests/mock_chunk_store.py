"""In-memory chunk store for tests."""

from collections.abc import Callable, Iterable

from src.chunking_service.data_types import RetrievalChunk
from src.db_service.postgres_chunk_store import ChunkStore


class MockChunkStore(ChunkStore):
    """Test double for the chunk store contract."""

    def __init__(self) -> None:
        self.setup_called = False
        self._chunks_by_id: dict[str, RetrievalChunk] = {}
        self.add_chunks_called = 0
        self.embedding_model: str | None = None
        self.embeddings_by_chunk_id: dict[str, list[float]] = {}

    @property
    def chunks(self) -> list[RetrievalChunk]:
        """Return all stored chunks in insertion order."""
        return list(self._chunks_by_id.values())

    def setup(self) -> None:
        """Mark the backing store as prepared."""
        self.setup_called = True

    def add_chunks(
        self,
        chunks: Iterable[RetrievalChunk],
        embedding_fn: Callable[[str], list[float]],
        embedding_model: str,
    ) -> None:
        """Store chunks by id and capture the generated embeddings."""
        self.add_chunks_called += 1
        self.embedding_model = embedding_model
        for chunk in chunks:
            self._chunks_by_id[chunk.chunk_id] = chunk
            self.embeddings_by_chunk_id[chunk.chunk_id] = embedding_fn(chunk.text)

    def get_chunks_for_record(self, record_id: str) -> list[RetrievalChunk]:
        """Fetch chunks linked to one parent record."""
        chunks = [
            chunk
            for chunk in self._chunks_by_id.values()
            if chunk.record_id == record_id
        ]
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)
