"""
Dataset loading utilities.

Reads processed JSONL paired datasets from data/processed/ and returns them
in formats suitable for feature extraction (src/features/extraction.py) and
evaluation pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from src.data.prompt_templates import DatasetType, PromptPair

PROCESSED_DATA_DIR = Path("data/processed")


def load_pairs(
    dataset_type: DatasetType | str,
    data_dir: str | Path | None = None,
) -> list[PromptPair]:
    """
    Load all prompt pairs for a given dataset type.

    Args:
        dataset_type: One of the DatasetType enum values or its string value.
        data_dir: Override the default data directory (data/processed).

    Returns:
        List of PromptPair instances.
    """
    if isinstance(dataset_type, str):
        dataset_type = DatasetType(dataset_type)
    base_dir = Path(data_dir) if data_dir else PROCESSED_DATA_DIR
    path = base_dir / dataset_type.value / "pairs.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run dataset_builder.DatasetBuilder.build_all() first."
        )
    return list(_iter_jsonl(path))


def load_all_pairs(
    data_dir: str | Path | None = None,
) -> dict[DatasetType, list[PromptPair]]:
    """Load all five paired datasets."""
    return {dt: load_pairs(dt, data_dir) for dt in DatasetType}


def _iter_jsonl(path: Path) -> Iterator[PromptPair]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            yield PromptPair(
                dataset_type=DatasetType(record["dataset_type"]),
                prompt_id=record["prompt_id"],
                control=record["control"],
                treatment=record["treatment"],
                correct_answer=record["correct_answer"],
                metadata=record.get("metadata", {}),
            )
