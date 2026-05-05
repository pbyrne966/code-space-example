import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.db_service.postgres_controllers import ChunkStore
from src.model_service.models import ModelClient

from .chunking import chunk_record
from .data_types import (
    ConvFinQARecord,
    RawDictRecords,
    RetrievalChunk,
    SplitName,
)


class ProcessLayer:
    def __init__(
        self, db_service: ChunkStore, raw_file_src: Path, model_client: ModelClient
    ) -> None:
        self.raw_file_src = raw_file_src
        self.db_service = db_service
        self.model_client = model_client

    def open_data_from_path(self) -> RawDictRecords:
        try:
            if not self.raw_file_src.is_file():
                raise ValueError("You must have pass in a valid file")
            if not self.raw_file_src.exists():
                raise ValueError("Could not find specified file")

            with self.raw_file_src.open() as file:
                raw_records = json.load(file)
            return RawDictRecords(**raw_records)

        except Exception as e:
            str_e = str(e)
            raise ValueError(f"Could not open file due to: {str_e}") from e

    def _iter_split_records(
        self, raw_data: RawDictRecords
    ) -> Iterable[tuple[SplitName, int, dict[str, Any]]]:
        for split in ("train", "dev", "test"):
            records = getattr(raw_data, split)
            for record_index, line in enumerate(records):
                yield split, record_index, line

    def process(self) -> list[RetrievalChunk]:
        raw_data = self.open_data_from_path()
        chunks: list[RetrievalChunk] = []

        self.db_service.setup()
        for split, record_index, line in self._iter_split_records(raw_data):
            record = ConvFinQARecord(**line)
            chunks.extend(
                chunk_record(
                    record=record,
                    split=split,
                    record_index=record_index,
                    source_file=self.raw_file_src,
                )
            )
        model_config = self.model_client.get_config()
        embed_fn = self.model_client.embed
        self.db_service.add_chunks(
            chunks, embedding_fn=embed_fn, embedding_model=model_config.model_name
        )
        return chunks
