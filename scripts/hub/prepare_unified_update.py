#!/usr/bin/env python3
"""
Prepare DAPO split for unified dataset upload.

This script:
1. Reads the cleaned/deduped dataset
2. Adds extra_info.original_dataset field (required for unified schema)
3. Ensures data_source matches convention
4. Generates parquet file for unified dataset
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


def prepare_unified_dapo_split(
    input_path: str,
    output_dir: str,
    original_dataset_name: str = "dapo-math-17k-verl",
) -> dict:
    """
    Prepare DAPO split for unified dataset.

    Args:
        input_path: Path to cleaned/deduped parquet file
        output_dir: Output directory for upload files
        original_dataset_name: Value for original_dataset field

    Returns:
        Dictionary with metadata
    """
    print(f"Reading dataset from: {input_path}")
    table = pq.read_table(input_path)
    df = table.to_pandas()

    print(f"Total samples: {len(df)}")
    print(f"Current schema: {table.schema}")

    # Check current extra_info structure
    sample_extra = df['extra_info'].iloc[0]
    print(f"\nCurrent extra_info fields: {list(sample_extra.keys())}")

    # Add original_dataset field if not present
    if 'original_dataset' not in sample_extra:
        print(f"\n✓ Adding 'original_dataset' field to extra_info...")

        def add_original_dataset(info):
            if isinstance(info, dict):
                info['original_dataset'] = original_dataset_name
            return info

        df['extra_info'] = df['extra_info'].apply(add_original_dataset)

        # Verify
        updated_extra = df['extra_info'].iloc[0]
        print(f"✓ Updated extra_info fields: {list(updated_extra.keys())}")
    else:
        print(f"\n✓ 'original_dataset' field already exists")

    # Ensure data_source is correct
    current_sources = df['data_source'].unique()
    print(f"\nCurrent data_source values: {current_sources}")

    if 'dapo-math-17k' in current_sources:
        print("✓ Fixing data_source: 'dapo-math-17k' → 'DAPO-Math-17K'")
        df['data_source'] = df['data_source'].str.replace('dapo-math-17k', 'DAPO-Math-17K', regex=False)

    print(f"✓ Final data_source: {df['data_source'].unique()}")

    # Convert back to Arrow table
    table = pa.Table.from_pandas(df, preserve_index=False)
    print(f"\nFinal schema:\n{table.schema}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_dir = output_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Write parquet file (use hyphenated name for unified)
    output_file = data_dir / "dapo-math-17k-verl.parquet"
    print(f"\nWriting to: {output_file}")
    pq.write_table(table, output_file, compression='snappy')

    # Calculate metadata
    file_size = output_file.stat().st_size

    # Calculate dataset size (uncompressed)
    import io
    buf = io.BytesIO()
    pq.write_table(table, buf, compression=None)
    uncompressed_size = len(buf.getvalue())

    metadata = {
        'num_examples': len(table),
        'download_size': file_size,
        'dataset_size': uncompressed_size,
        'output_file': str(output_file),
        'schema': str(table.schema),
        'original_dataset_field': original_dataset_name,
    }

    # Save metadata
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Unified DAPO split prepared successfully!")
    print(f"   Samples: {metadata['num_examples']}")
    print(f"   Download size: {metadata['download_size']:,} bytes ({metadata['download_size'] / 1024 / 1024:.2f} MB)")
    print(f"   Dataset size: {metadata['dataset_size']:,} bytes ({metadata['dataset_size'] / 1024 / 1024:.2f} MB)")
    print(f"   Output: {output_file}")
    print(f"   Metadata: {metadata_file}")

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare DAPO split for unified dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="output/dapo-math-cleaned-deduped/train.parquet",
        help="Input parquet file (cleaned/deduped)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/hub-upload/unified",
        help="Output directory for upload files"
    )
    parser.add_argument(
        "--original-dataset",
        type=str,
        default="dapo-math-17k-verl",
        help="Value for original_dataset field"
    )

    args = parser.parse_args()

    metadata = prepare_unified_dapo_split(
        input_path=args.input,
        output_dir=args.output,
        original_dataset_name=args.original_dataset,
    )

    print("\n" + "="*60)
    print("Next steps:")
    print("1. Review the generated files")
    print("2. Update unified dataset README")
    print("3. Upload to HuggingFace Hub")
    print("="*60)
