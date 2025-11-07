#!/usr/bin/env python3
"""
Create dataset splits from combined deduplicated dataset.

This script reads the combined Phase 2 dataset and splits it by original_dataset
field, creating separate parquet files for each dataset that can be uploaded
as HuggingFace dataset splits.
"""

import argparse
import glob
import os
from collections import defaultdict
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def create_splits(
    combined_dir: str,
    output_dir: str,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Split combined dataset by original_dataset field.

    Args:
        combined_dir: Directory containing combined Phase 2 parquet files
        output_dir: Output directory for split datasets
        verbose: Print detailed progress

    Returns:
        Dictionary mapping dataset names to row counts
    """
    print(f"\n{'=' * 70}")
    print("Creating Dataset Splits")
    print(f"{'=' * 70}\n")

    # Find all combined files
    combined_files = sorted(glob.glob(os.path.join(combined_dir, '*.parquet')))
    if not combined_files:
        raise ValueError(f"No parquet files found in {combined_dir}")

    print(f"Found {len(combined_files)} combined files")
    print()

    # Track rows by dataset
    dataset_rows: Dict[str, List[Dict]] = defaultdict(list)
    dataset_counts: Dict[str, int] = defaultdict(int)

    # Read all combined files and group by original_dataset
    print("Reading and grouping rows by original_dataset...")
    for file_path in tqdm(combined_files):
        table = pq.read_table(file_path)
        df = table.to_pandas()

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            dataset_name = row_dict['extra_info'].get('original_dataset', 'unknown')
            dataset_rows[dataset_name].append(row_dict)
            dataset_counts[dataset_name] += 1

    print()
    print("Dataset distribution:")
    total_rows = 0
    for dataset_name in sorted(dataset_counts.keys()):
        count = dataset_counts[dataset_name]
        total_rows += count
        print(f"  {dataset_name}: {count:,} rows")
    print(f"  TOTAL: {total_rows:,} rows")
    print()

    # Write each dataset split
    print("Writing dataset splits...")
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name in tqdm(sorted(dataset_rows.keys())):
        rows = dataset_rows[dataset_name]

        # Create output file path (using split name as filename)
        # This allows HuggingFace to recognize different splits
        output_path = os.path.join(output_dir, f"{dataset_name}.parquet")

        if verbose:
            print(f"  Writing {dataset_name}: {len(rows):,} rows → {output_path}")

        # Convert to pandas and then to parquet
        df_split = pd.DataFrame(rows)
        table_split = pa.Table.from_pandas(df_split)
        pq.write_table(table_split, output_path)

    print()
    print(f"{'=' * 70}")
    print("Split Creation Complete")
    print(f"{'=' * 70}")
    print(f"Output directory: {output_dir}")
    print(f"Total splits created: {len(dataset_rows)}")
    print(f"Total rows: {total_rows:,}")
    print(f"{'=' * 70}\n")

    return dict(dataset_counts)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create dataset splits from combined deduplicated dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--combined-dir',
        type=str,
        default='_deduplicated/phase2-inter/combined/data',
        help='Directory containing combined parquet files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='_deduplicated/phase2-inter/splits',
        help='Output directory for split datasets'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )

    args = parser.parse_args()

    try:
        stats = create_splits(
            combined_dir=args.combined_dir,
            output_dir=args.output_dir,
            verbose=args.verbose
        )

        print("✅ Dataset splits created successfully!\n")

    except Exception as e:
        print(f"\n❌ Error creating splits: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
