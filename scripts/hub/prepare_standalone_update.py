#!/usr/bin/env python3
"""
Prepare standalone dataset for HuggingFace Hub upload.

This script:
1. Reads the cleaned/deduped dataset
2. Ensures schema matches standalone format (no original_dataset field)
3. Generates parquet file for upload
4. Calculates file sizes for README metadata
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def prepare_standalone_dataset(
    input_path: str,
    output_dir: str,
) -> dict:
    """
    Prepare standalone dataset for upload.

    Args:
        input_path: Path to cleaned/deduped parquet file
        output_dir: Output directory for upload files

    Returns:
        Dictionary with metadata (file sizes, sample count, etc.)
    """
    print(f"Reading dataset from: {input_path}")
    table = pq.read_table(input_path)

    print(f"Total samples: {len(table)}")
    print(f"Schema: {table.schema}")

    # Verify schema matches standalone format
    # Expected schema for standalone:
    # - data_source: string
    # - prompt: list<struct<role: string, content: string>>
    # - ability: string
    # - reward_model: struct<style: string, ground_truth: string>
    # - extra_info: struct<split: string, index: int64>

    # Check if extra_info has original_dataset field (from unified dataset)
    extra_info_type = table.schema.field('extra_info').type

    if isinstance(extra_info_type, pa.StructType):
        field_names = [f.name for f in extra_info_type]
        print(f"extra_info fields: {field_names}")

        if 'original_dataset' in field_names:
            print("WARNING: Found 'original_dataset' field in extra_info")
            print("This field should be removed for standalone schema")
            print("Removing 'original_dataset' field...")

            # Convert to pandas, remove field, convert back
            import pandas as pd
            df = table.to_pandas()

            # Modify extra_info to remove original_dataset
            def clean_extra_info(info):
                if isinstance(info, dict):
                    return {k: v for k, v in info.items() if k != 'original_dataset'}
                return info

            df['extra_info'] = df['extra_info'].apply(clean_extra_info)

            # Convert back to Arrow table
            table = pa.Table.from_pandas(df, preserve_index=False)
            print(f"New schema: {table.schema}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_dir = output_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Write parquet file
    output_file = data_dir / "train-00000.parquet"
    print(f"Writing to: {output_file}")
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
    }

    # Save metadata
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Standalone dataset prepared successfully!")
    print(f"   Samples: {metadata['num_examples']}")
    print(f"   Download size: {metadata['download_size']:,} bytes ({metadata['download_size'] / 1024 / 1024:.2f} MB)")
    print(f"   Dataset size: {metadata['dataset_size']:,} bytes ({metadata['dataset_size'] / 1024 / 1024:.2f} MB)")
    print(f"   Output: {output_file}")
    print(f"   Metadata: {metadata_file}")

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare standalone dataset for HF Hub upload")
    parser.add_argument(
        "--input",
        type=str,
        default="output/dapo-math-cleaned-deduped/train.parquet",
        help="Input parquet file (cleaned/deduped)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/hub-upload/standalone",
        help="Output directory for upload files"
    )

    args = parser.parse_args()

    metadata = prepare_standalone_dataset(
        input_path=args.input,
        output_dir=args.output,
    )

    print("\n" + "="*60)
    print("Next steps:")
    print("1. Review the generated files")
    print("2. Create updated README.md")
    print("3. Upload to HuggingFace Hub")
    print("="*60)
