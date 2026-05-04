"""Create deterministic sample files from the ConvFinQA dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic ConvFinQA sample JSON files."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/convfinqa_dataset.json"),
        help="Path to the full ConvFinQA dataset JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/samples"),
        help="Directory where sample JSON files will be written.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of records to sample from each split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=96,
        help="Seed used to create deterministic samples.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_samples(
        source_file=args.source,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        splits=tuple(args.splits),
    )


if __name__ == "__main__":
    main()
