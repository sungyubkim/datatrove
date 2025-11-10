#!/usr/bin/env python3
"""
Download all splits from sungyub/math-verl-unified Hub dataset.

This script downloads all 9 splits from the Hub and saves them as parquet files
in a structure matching the Hub layout (data/ folder with individual parquet files).
"""

import argparse
import os
import sys
from typing import Dict, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

# Standard VERL schema for validation
STANDARD_SCHEMA = pa.schema([
    ('data_source', pa.string()),
    ('prompt', pa.list_(pa.struct([
        ('role', pa.string()),
        ('content', pa.string())
    ]))),
    ('ability', pa.string()),
    ('reward_model', pa.struct([
        ('ground_truth', pa.string()),
        ('style', pa.string())
    ])),
    ('extra_info', pa.struct([
        ('index', pa.int64()),
        ('original_dataset', pa.string()),
        ('split', pa.string())
    ]))
])

# Split names matching Hub filenames (without .parquet extension)
SPLIT_NAMES = [
    'dapo-math-17k-verl',
    'deepscaler-preview-verl',
    'orz-math-72k-verl',
    'deepmath_103k_verl',
    'skywork_or1_math_verl',
    'openr1-math-verl',
    'big-math-rl-verl',
    'eurus-2-math-verl',
    'mathx-5m-verl'
]


def download_split(
    repo_id: str,
    split_name: str,
    output_dir: str,
    batch_size: int = 10000,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Download a single split from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        split_name: Name of the split (matches Hub filename without .parquet)
        output_dir: Output directory (should end with /data)
        batch_size: Rows per batch for processing
        verbose: Print detailed progress

    Returns:
        Statistics dictionary
    """
    print(f"\n{'─' * 70}")
    print(f"Downloading: {split_name}")
    print(f"{'─' * 70}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Output file path (matches Hub filename exactly)
    output_file = os.path.join(output_dir, f"{split_name}.parquet")

    # Load dataset in streaming mode for memory efficiency
    try:
        # HuggingFace uses underscore for split names
        split_name_hf = split_name.replace('-', '_')
        dataset = load_dataset(
            repo_id,
            split=split_name_hf,
            streaming=True
        )
    except Exception as e:
        print(f"  ❌ Error loading split: {e}")
        return {
            'split': split_name,
            'status': 'error',
            'error': str(e)
        }

    # Collect all rows
    rows = []
    total_rows = 0

    print(f"  Downloading rows...")
    for row in tqdm(dataset, desc=f"  Progress", disable=not verbose):
        rows.append(row)
        total_rows += 1

        # Write in batches to manage memory
        if len(rows) >= batch_size:
            df = pd.DataFrame(rows)
            # Validate and write
            if not os.path.exists(output_file):
                # First batch - create file with schema
                table = pa.Table.from_pandas(df, schema=STANDARD_SCHEMA)
                pq.write_table(table, output_file)
            else:
                # Append to existing file
                table = pa.Table.from_pandas(df, schema=STANDARD_SCHEMA)
                writer = pq.ParquetWriter(output_file, STANDARD_SCHEMA)
                writer.write_table(table)
                writer.close()

            rows = []
            if verbose:
                print(f"  Wrote {total_rows:,} rows...")

    # Write remaining rows
    if rows:
        df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df, schema=STANDARD_SCHEMA)

        if not os.path.exists(output_file):
            pq.write_table(table, output_file)
        else:
            # Append
            writer = pq.ParquetWriter(output_file, STANDARD_SCHEMA)
            writer.write_table(table)
            writer.close()

    # Verify output
    table = pq.read_table(output_file)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

    print(f"  ✅ Downloaded: {total_rows:,} rows ({file_size_mb:.1f} MB)")
    print(f"  Saved to: {output_file}")

    return {
        'split': split_name,
        'status': 'success',
        'rows': total_rows,
        'file_size_mb': file_size_mb,
        'output_file': output_file
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download math-verl-unified dataset from HuggingFace Hub',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all splits to default location
  python download_math_verl_unified.py

  # Download to custom directory
  python download_math_verl_unified.py --output-dir /path/to/output

  # Download specific splits only
  python download_math_verl_unified.py --splits dapo-math-17k-verl deepscaler-preview-verl

  # Verbose mode
  python download_math_verl_unified.py --verbose
        """
    )

    parser.add_argument(
        '--repo-id',
        type=str,
        default='sungyub/math-verl-unified',
        help='HuggingFace repository ID (default: sungyub/math-verl-unified)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/downloads/math-verl-unified',
        help='Output directory (default: output/downloads/math-verl-unified)'
    )

    parser.add_argument(
        '--splits',
        nargs='+',
        default=None,
        help='Specific splits to download (default: all)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Rows per batch (default: 10000)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )

    args = parser.parse_args()

    # Determine which splits to download
    splits_to_download = args.splits if args.splits else SPLIT_NAMES

    # Validate split names
    for split in splits_to_download:
        if split not in SPLIT_NAMES:
            print(f"❌ Error: Unknown split '{split}'")
            print(f"Available splits: {', '.join(SPLIT_NAMES)}")
            sys.exit(1)

    # Create data directory
    data_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    print(f"{'=' * 70}")
    print("DOWNLOADING MATH-VERL-UNIFIED DATASET")
    print(f"{'=' * 70}")
    print(f"Repository: {args.repo_id}")
    print(f"Output dir: {args.output_dir}")
    print(f"Splits: {len(splits_to_download)}")
    print()

    # Download each split
    results = []
    for split_name in splits_to_download:
        result = download_split(
            repo_id=args.repo_id,
            split_name=split_name,
            output_dir=data_dir,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
        results.append(result)

    # Summary
    print(f"\n{'=' * 70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 70}")

    success_count = sum(1 for r in results if r['status'] == 'success')
    total_rows = sum(r.get('rows', 0) for r in results)
    total_size_mb = sum(r.get('file_size_mb', 0) for r in results)

    print(f"Successful downloads: {success_count}/{len(results)}")
    print(f"Total rows:           {total_rows:,}")
    print(f"Total size:           {total_size_mb:.1f} MB")
    print(f"Output directory:     {data_dir}")
    print()

    # List failed downloads
    failed = [r for r in results if r['status'] != 'success']
    if failed:
        print("Failed downloads:")
        for r in failed:
            print(f"  - {r['split']}: {r.get('error', 'Unknown error')}")
        print()

    print("✅ Download complete!")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
