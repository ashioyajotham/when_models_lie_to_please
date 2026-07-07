"""
Local import script.
Unpacks a Colab results zip file directly into the project structure.

Usage:
    python experiments/scripts/import_colab_results.py --zip-path /path/to/colab_results_*.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Unpack Colab results zip into local workspace")
    parser.add_argument(
        "--zip-path",
        required=True,
        help="Path to the downloaded colab_results_*.zip file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    zip_path = Path(args.zip_path)

    if not zip_path.exists():
        print(f"Error: Zip file not found at '{zip_path}'", file=sys.stderr)
        sys.exit(1)

    # Project root is two levels up from this script (when_models_lie_to_please/)
    project_root = Path(__file__).parents[2].resolve()
    print(f"Extracting results into project root: {project_root}...")

    extracted_count = 0
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            # Skip directories
            if member.is_dir():
                continue

            # Check if paths are relative and clean
            dest_path = project_root / member.filename
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            with zip_ref.open(member) as source, open(dest_path, "wb") as target:
                target.write(source.read())

            extracted_count += 1
            print(f"  Extracted: {member.filename}")

    print(f"\nSuccessfully imported {extracted_count} files from '{zip_path.name}'!")


if __name__ == "__main__":
    main()
