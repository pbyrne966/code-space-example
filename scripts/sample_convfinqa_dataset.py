"""Create deterministic sample files from the ConvFinQA dataset."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Annotated, Any

import typer

DEFAULT_SPLITS = ("train", "dev", "test")


def _load_dataset(source_file: Path) -> dict[str, list[dict[str, Any]]]:
    raw_data = json.loads(source_file.read_text(encoding="utf-8"))

    if not isinstance(raw_data, dict):
        raise ValueError("Expected the dataset root to be a JSON object")

    dataset: dict[str, list[dict[str, Any]]] = {}
    for split, records in raw_data.items():
        if not isinstance(records, list):
            raise ValueError(f"Expected split '{split}' to contain a list of records")
        dataset[split] = records

    return dataset


def _sample_records(
    records: list[dict[str, Any]], sample_size: int, rng: Random
) -> list[dict[str, Any]]:
    if len(records) <= sample_size:
        return list(records)

    sample_indexes = sorted(rng.sample(range(len(records)), sample_size))
    return [records[index] for index in sample_indexes]


def write_samples(
    source_file: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> list[Path]:
    """Write deterministic sample files for the requested dataset splits."""
    dataset = _load_dataset(source_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    for split in splits:
        records = dataset.get(split, [])
        if not records:
            continue

        rng = Random(f"{seed}:{split}")
        sampled_records = _sample_records(records, sample_size, rng)
        output_payload = {
            "source_file": str(source_file),
            "split": split,
            "seed": seed,
            "source_count": len(records),
            "sample_count": len(sampled_records),
            "records": sampled_records,
        }

        output_file = Path(output_dir / f"{source_file.name}_{split}_sample.json")
        output_file.write_text(json.dumps(output_payload, indent=4), encoding="utf-8")
        written_files.append(output_file)

    return written_files


def main(
    source: Annotated[
        Path,
        typer.Option(help="Path to the full ConvFinQA dataset JSON file."),
    ] = Path("data/convfinqa_dataset.json"),
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where sample JSON files will be written."),
    ] = Path("data/samples"),
    sample_size: Annotated[
        int,
        typer.Option(help="Number of records to sample from each split."),
    ] = 5,
    seed: Annotated[
        int,
        typer.Option(help="Seed used to create deterministic samples."),
    ] = 96,
    splits: Annotated[
        list[str] | None,
        typer.Option(help="Dataset splits to sample."),
    ] = None,
) -> None:
    """Create deterministic ConvFinQA sample JSON files."""
    selected_splits = tuple(splits or DEFAULT_SPLITS)
    written_files = write_samples(
        source_file=source,
        output_dir=output_dir,
        sample_size=sample_size,
        seed=seed,
        splits=selected_splits,
    )
    for written_file in written_files:
        typer.echo(written_file)


if __name__ == "__main__":
    typer.run(main)
