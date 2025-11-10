#!/usr/bin/env python3
"""
Add 'original_dataset' field to extra_info for unified dataset compatibility.

This script modifies the v2 cleaned dataset to be compatible with the unified
math-verl dataset structure by adding the 'original_dataset' field to extra_info.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def add_original_dataset_field(
    input_path: str,
    output_path: str,
    original_dataset: str = "deepscaler-preview-verl"
):
    """
    Add original_dataset field to extra_info in the dataset.

    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        original_dataset: Value for the original_dataset field
    """
    print(f"Loading dataset from: {input_path}")
    table = pq.read_table(input_path)
    df = table.to_pandas()

    print(f"Original dataset shape: {df.shape}")
    print(f"Original extra_info fields: {list(df['extra_info'].iloc[0].keys())}")

    # Add original_dataset field to each extra_info dict
    def add_field(extra_info):
        if isinstance(extra_info, dict):
            extra_info_copy = extra_info.copy()
            extra_info_copy['original_dataset'] = original_dataset
            return extra_info_copy
        return extra_info

    print(f"\nAdding 'original_dataset' = '{original_dataset}' to all samples...")
    df['extra_info'] = df['extra_info'].apply(add_field)

    # Verify the change
    sample_extra_info = df['extra_info'].iloc[0]
    print(f"Updated extra_info fields: {list(sample_extra_info.keys())}")
    print(f"Sample extra_info: {sample_extra_info}")

    # Convert back to Arrow table and save
    print(f"\nSaving to: {output_path}")
    output_table = pa.Table.from_pandas(df, preserve_index=False)

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pq.write_table(output_table, output_path)

    print(f"✓ Successfully saved updated dataset")
    print(f"  Output shape: {df.shape}")
    print(f"  Output size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add original_dataset field to extra_info")
    parser.add_argument(
        "--input",
        default="output/deepscaler-cleaned-v2/train.parquet",
        help="Input parquet file path"
    )
    parser.add_argument(
        "--output",
        default="output/hub-upload/deepscaler-unified/train.parquet",
        help="Output parquet file path"
    )
    parser.add_argument(
        "--original-dataset",
        default="deepscaler-preview-verl",
        help="Value for original_dataset field"
    )

    args = parser.parse_args()

    add_original_dataset_field(
        input_path=args.input,
        output_path=args.output,
        original_dataset=args.original_dataset
    )
