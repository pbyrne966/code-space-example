"""Create and ingest a small ConvFinQA subset for faster evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from src.chunking_service.data_loader import ProcessLayer
from src.data_types import RawDictRecords
from src.db_service.postgres_controllers import PostgresChunkStore
from src.logger import get_logger
from src.runtime import IngestionAppState, build_ingestion_context

logger = get_logger("subset_ingestion")

app = typer.Typer(
    name="ingest-convfinqa-subset",
    help="Ingest a small ConvFinQA subset for snappy evaluation.",
    add_completion=False,
)

DEFAULT_OUTPUT_PATH = Path("data/evaluation/convfinqa_subset.json")


def load_dataset(source_path: Path) -> RawDictRecords:
    """Load and validate the ConvFinQA dataset file."""
    raw_payload = json.loads(source_path.read_text(encoding="utf-8"))
    return RawDictRecords(**raw_payload)


def build_subset_payload(
    dataset: RawDictRecords,
    splits: list[str],
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    """Take the first N records from each requested split."""
    subset = {"train": [], "dev": [], "test": []}

    for split in splits:
        if split not in subset:
            raise ValueError(f"Unsupported split: {split}")

        records = getattr(dataset, split)
        subset[split] = records[:limit]

    return subset


def write_subset(
    source_path: Path,
    output_path: Path,
    splits: list[str],
    limit: int,
) -> Path:
    """Write a ConvFinQA-shaped subset file for ProcessLayer."""
    dataset = load_dataset(source_path)
    subset_payload = build_subset_payload(dataset, splits, limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(subset_payload, indent=2),
        encoding="utf-8",
    )
    return output_path


def ingest_subset(context: IngestionAppState, subset_path: Path) -> int:
    """Run ProcessLayer over the subset file and return chunk count."""
    if context.db_engine is None:
        raise ValueError("Database engine was not initialised")
    if context.client is None:
        raise ValueError("Model client was not initialised")

    db_service = PostgresChunkStore(context.db_engine)
    chunks = ProcessLayer(
        db_service=db_service,
        raw_file_src=subset_path,
        model_client=context.client,
    ).process()

    return len(chunks)


@app.command()
def main(
    source: Annotated[
        Path | None,
        typer.Option(help="Source ConvFinQA dataset. Defaults to RAW_DATA_PATH."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(help="Where to write the temporary subset dataset."),
    ] = DEFAULT_OUTPUT_PATH,
    limit: Annotated[
        int,
        typer.Option(help="Number of records to keep from each selected split."),
    ] = 300,
    splits: Annotated[
        list[str] | None,
        typer.Option(help="Dataset split(s) to include."),
    ] = None,
) -> None:
    """Write and ingest a small dataset subset."""
    source_path = source or Path("data/convfinqa_dataset.json")
    selected_splits = splits or ["dev"]
    subset_path = write_subset(
        source_path=source_path,
        output_path=output,
        splits=selected_splits,
        limit=limit,
    )

    context = build_ingestion_context()
    chunk_count = ingest_subset(context, subset_path)

    typer.echo(
        f"Ingested {chunk_count} chunks from {subset_path} "
        f"using splits={','.join(selected_splits)} limit={limit}"
    )


if __name__ == "__main__":
    app()
