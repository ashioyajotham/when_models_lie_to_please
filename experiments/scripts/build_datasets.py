"""
Dataset construction script.

Builds all five paired prompt datasets and writes them to data/processed/.

Usage:
    python experiments/scripts/build_datasets.py \
        [--output-dir data/processed] \
        [--min-pairs 500] \
        [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.data.dataset_builder import DatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_datasets")


def parse_args():
    parser = argparse.ArgumentParser(description="Build paired prompt datasets")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--min-pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    builder = DatasetBuilder(
        output_dir=args.output_dir,
        min_pairs=args.min_pairs,
        seed=args.seed,
    )
    paths = builder.build_all()
    for dataset_type, path in paths.items():
        logger.info("Built %s → %s", dataset_type.value, path)


if __name__ == "__main__":
    main()
