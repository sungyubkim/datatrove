#!/usr/bin/env python3
"""
Add original_dataset field to extra_info for math-verl-unified.

This script modifies a VERL parquet file to add the 'original_dataset'
field to the extra_info struct, preparing it for upload to the unified
math-verl dataset repository.

Usage:
    python scripts/conversion/add_original_dataset_field.py \
        --input output/mathx-5m-cleaned-deduped/train.parquet \
        --output output/hub-upload/math-verl-unified/data/mathx-5m-verl.parquet \
        --dataset-name "MathX-5M"
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def add_original_dataset_field(
    input_file: Path,
    output_file: Path,
    dataset_name: str,
    batch_size: int = 10000,
) -> None:
    """Add original_dataset field to extra_info struct.

    Args:
        input_file: Input parquet file path
        output_file: Output parquet file path
        dataset_name: Name of the dataset (for original_dataset field)
        batch_size: Batch size for streaming writes
    """
    print(f"\n{'='*70}")
    print(f"Adding original_dataset Field to extra_info")
    print(f"{'='*70}\n")

    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Dataset Name: {dataset_name}")
    print(f"Batch Size: {batch_size:,}\n")

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Define new schema with original_dataset field
    new_schema = pa.schema([
        ('data_source', pa.string()),
        ('prompt', pa.list_(pa.struct([
            ('role', pa.string()),
            ('content', pa.string()),
        ]))),
        ('ability', pa.string()),
        ('reward_model', pa.struct([
            ('style', pa.string()),
            ('ground_truth', pa.string()),
        ])),
        ('extra_info', pa.struct([
            ('index', pa.int64()),
            ('split', pa.string()),
            ('original_dataset', pa.string()),  # NEW FIELD
        ])),
    ])

    # Read input file
    print(f"Reading input file...")
    input_table = pq.read_table(input_file)
    total_rows = len(input_table)
    print(f"✓ Loaded {total_rows:,} rows\n")

    # Convert to Python list for processing
    print(f"Processing rows and adding original_dataset field...")
    rows = input_table.to_pylist()

    # Add original_dataset field to each row
    for row in rows:
        row['extra_info']['original_dataset'] = dataset_name

    # Write output in batches
    print(f"Writing output file with new schema...\n")
    writer = None

    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            batch_table = pa.Table.from_pylist(batch, schema=new_schema)

            if writer is None:
                writer = pq.ParquetWriter(output_file, schema=new_schema, compression='snappy')

            writer.write_table(batch_table)

            processed = min(i + batch_size, total_rows)
            print(f"  Written: {processed:,} / {total_rows:,} rows ({100*processed/total_rows:.1f}%)", end='\r')

    finally:
        if writer is not None:
            writer.close()

    print(f"\n\n✅ Successfully added original_dataset field!")

    # Verify output
    print(f"\n{'='*70}")
    print(f"Verification")
    print(f"{'='*70}\n")

    output_table = pq.read_table(output_file)
    sample = output_table.to_pylist()[0]

    print(f"✓ Output rows: {len(output_table):,}")
    print(f"✓ Output schema: {list(output_table.schema.names)}")
    print(f"✓ extra_info fields: {list(sample['extra_info'].keys())}")
    print(f"✓ original_dataset value: {sample['extra_info']['original_dataset']}")

    # File size
    file_size = output_file.stat().st_size
    print(f"✓ Output file size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Add original_dataset field to extra_info for unified dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input parquet file path'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output parquet file path'
    )

    parser.add_argument(
        '--dataset-name',
        type=str,
        required=True,
        help='Dataset name for original_dataset field (e.g., "MathX-5M")'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Batch size for writing (default: 10000)'
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Validate input
    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        return 1

    # Run conversion
    add_original_dataset_field(
        input_file=input_path,
        output_file=output_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
    )

    print(f"{'='*70}")
    print(f"✅ Conversion complete!")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    exit(main())
