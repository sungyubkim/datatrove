#!/usr/bin/env python3
"""
Calculate Metadata for Hub Upload

Calculates num_bytes, download_size, and dataset_size for YAML metadata.
"""

import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq


def calculate_metadata(parquet_file: str):
    """Calculate metadata for a parquet file.

    Args:
        parquet_file: Path to parquet file

    Returns:
        Dict with metadata
    """
    print(f"Calculating metadata for: {parquet_file}")

    # Read parquet file
    table = pq.read_table(parquet_file)
    num_examples = len(table)

    # Calculate uncompressed size (dataset_size)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression=None)
    dataset_size = len(buf.getvalue())

    # Get compressed size (download_size)
    download_size = Path(parquet_file).stat().st_size

    # Calculate num_bytes (same as dataset_size for single file)
    num_bytes = dataset_size

    print(f"  num_examples: {num_examples:,}")
    print(f"  num_bytes: {num_bytes:,}")
    print(f"  download_size: {download_size:,}")
    print(f"  dataset_size: {dataset_size:,}")
    print(f"  compression ratio: {dataset_size / download_size:.2f}x")

    return {
        'num_examples': num_examples,
        'num_bytes': num_bytes,
        'download_size': download_size,
        'dataset_size': dataset_size,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Calculate metadata for Hub upload'
    )

    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help='Parquet file to analyze'
    )

    args = parser.parse_args()

    # Validate input
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        return

    # Calculate metadata
    metadata = calculate_metadata(args.file)

    print("\n" + "=" * 70)
    print("YAML Metadata:")
    print("=" * 70)
    print(f"  num_bytes: {metadata['num_bytes']}")
    print(f"  num_examples: {metadata['num_examples']}")
    print(f"download_size: {metadata['download_size']}")
    print(f"dataset_size: {metadata['dataset_size']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
