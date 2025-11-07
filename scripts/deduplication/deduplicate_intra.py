#!/usr/bin/env python3
"""
Phase 1: Intra-Dataset Deduplication

This script removes duplicates within each dataset independently.
Processes parquet files using streaming for memory efficiency.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import glob

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deduplication.utils import (
    compute_hash,
    extract_problem_text,
    validate_verl_row,
    save_stats,
    format_number,
    format_percentage,
    DuplicationStats
)


def deduplicate_dataset(
    dataset_name: str,
    input_dir: str,
    output_dir: str,
    batch_size: int = 10000,
    dry_run: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Deduplicate a single dataset.

    Args:
        dataset_name: Name of dataset directory
        input_dir: Root directory containing datasets
        output_dir: Output directory for deduplicated data
        batch_size: Number of rows to process per batch
        dry_run: If True, only count duplicates without writing
        verbose: Print detailed progress

    Returns:
        Statistics dictionary
    """
    start_time = time.time()

    # Setup paths
    input_path = os.path.join(input_dir, dataset_name)
    output_path = os.path.join(output_dir, dataset_name)

    if not os.path.exists(input_path):
        raise ValueError(f"Dataset not found: {input_path}")

    # Find all parquet files
    data_dir = os.path.join(input_path, 'data')
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")

    parquet_files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    if not parquet_files:
        raise ValueError(f"No parquet files found in: {data_dir}")

    print(f"\n{'=' * 70}")
    print(f"Processing: {dataset_name}")
    print(f"{'=' * 70}")
    print(f"Input files: {len(parquet_files)}")

    # Initialize statistics
    stats = DuplicationStats(dataset_name)
    hash_set = set()
    hash_to_text = {}  # For collision detection

    # Create output directory
    if not dry_run:
        os.makedirs(os.path.join(output_path, 'data'), exist_ok=True)

    # Prepare writer
    writer = None
    schema = None
    output_file_idx = 0
    rows_in_current_file = 0
    ROWS_PER_OUTPUT_FILE = 500000  # Split large outputs

    # Process each input file
    for file_idx, input_file in enumerate(parquet_files):
        if verbose:
            print(f"\nProcessing file {file_idx + 1}/{len(parquet_files)}: {os.path.basename(input_file)}")

        # Read file
        table = pq.read_table(input_file)
        total_rows_in_file = table.num_rows

        if schema is None:
            schema = table.schema

        # Process in batches
        for batch_start in tqdm(
            range(0, total_rows_in_file, batch_size),
            desc=f"  Batches",
            disable=not verbose
        ):
            batch_end = min(batch_start + batch_size, total_rows_in_file)
            batch = table.slice(batch_start, batch_end - batch_start)
            df_batch = batch.to_pandas()

            unique_rows = []

            for idx, row in df_batch.iterrows():
                # Validate format
                row_dict = row.to_dict()
                is_valid, error_msg = validate_verl_row(row_dict)

                if not is_valid:
                    print(f"⚠️  Warning: Invalid row at index {idx}: {error_msg}")
                    continue

                # Extract and hash problem text
                try:
                    problem_text = extract_problem_text(row_dict)
                    problem_hash = compute_hash(problem_text)
                except Exception as e:
                    print(f"⚠️  Warning: Error processing row {idx}: {e}")
                    continue

                # Check for duplicates
                is_duplicate = problem_hash in hash_set

                stats.add_row(problem_hash, problem_text, is_duplicate)

                if not is_duplicate:
                    hash_set.add(problem_hash)
                    hash_to_text[problem_hash] = problem_text
                    unique_rows.append(row_dict)

            # Write unique rows
            if not dry_run and unique_rows:
                # Convert to pyarrow table
                df_unique = pd.DataFrame(unique_rows)
                table_unique = pa.Table.from_pandas(df_unique, schema=schema)

                # Initialize or write to writer
                if writer is None or rows_in_current_file >= ROWS_PER_OUTPUT_FILE:
                    # Close previous writer
                    if writer is not None:
                        writer.close()

                    # Start new file
                    output_filename = f"train-{output_file_idx:05d}.parquet"
                    output_filepath = os.path.join(output_path, 'data', output_filename)

                    writer = pq.ParquetWriter(output_filepath, schema)
                    rows_in_current_file = 0
                    output_file_idx += 1

                    if verbose:
                        print(f"  Writing to: {output_filename}")

                writer.write_table(table_unique)
                rows_in_current_file += len(unique_rows)

    # Close final writer
    if writer is not None:
        writer.close()

    # Calculate statistics
    duration = time.time() - start_time

    stats_dict = {
        'dataset': dataset_name,
        'phase': 'intra',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input': {
            'files': len(parquet_files),
            'total_rows': stats.total_rows
        },
        'output': {
            'files': output_file_idx if not dry_run else 0,
            'unique_rows': stats.unique_rows
        },
        'deduplication': {
            'duplicates_found': stats.duplicate_rows,
            'duplicate_rate': stats.get_duplicate_rate(),
            'unique_rate': 1 - stats.get_duplicate_rate()
        },
        'processing': {
            'duration_seconds': duration,
            'rows_per_second': stats.total_rows / duration if duration > 0 else 0
        },
        'top_duplicates': stats.get_top_duplicates(10)
    }

    # Save statistics
    if not dry_run:
        stats_file = os.path.join(output_path, 'stats.json')
        save_stats(stats_dict, stats_file)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Summary: {dataset_name}")
    print(f"{'=' * 70}")
    print(f"Total rows:      {format_number(stats.total_rows)}")
    print(f"Unique rows:     {format_number(stats.unique_rows)}")
    print(f"Duplicates:      {format_number(stats.duplicate_rows)} ({format_percentage(stats.duplicate_rows, stats.total_rows)})")
    print(f"Processing time: {duration:.1f}s ({format_number(int(stats.total_rows / duration if duration > 0 else 0))} rows/sec)")
    if not dry_run:
        print(f"Output location: {output_path}")
    print(f"{'=' * 70}\n")

    return stats_dict


def get_dataset_list(input_dir: str) -> List[str]:
    """
    Get list of all dataset directories in input directory.

    Args:
        input_dir: Root directory containing datasets

    Returns:
        List of dataset directory names
    """
    datasets = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            # Check if it has a data/ subdirectory with parquet files
            data_dir = os.path.join(item_path, 'data')
            if os.path.isdir(data_dir):
                parquet_files = glob.glob(os.path.join(data_dir, '*.parquet'))
                if parquet_files:
                    datasets.append(item)
    return sorted(datasets)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 1: Intra-dataset deduplication',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single dataset
  python deduplicate_intra.py --dataset docqa-rl-verl

  # Process all datasets in directory
  python deduplicate_intra.py --all

  # Dry run (count only)
  python deduplicate_intra.py --dataset docqa-rl-verl --dry-run

  # Custom paths
  python deduplicate_intra.py --dataset toolrl-4k-verl --input-dir ~/data/qa --output-dir ~/data/qa_dedup
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name to process (e.g., docqa-rl-verl)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all available datasets in input directory'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='.',
        help='Root directory containing datasets (default: current directory)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='_deduplicated/phase1-intra',
        help='Output directory (default: _deduplicated/phase1-intra)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Rows per batch (default: 10000)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Count duplicates without writing output'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.dataset and not args.all:
        parser.error("Must specify either --dataset or --all")

    if args.dataset and args.all:
        parser.error("Cannot specify both --dataset and --all")

    # Get datasets to process
    if args.all:
        datasets = get_dataset_list(args.input_dir)

        if not datasets:
            print("No datasets found!")
            sys.exit(1)

        print(f"\nFound {len(datasets)} datasets to process:")
        for ds in datasets:
            print(f"  - {ds}")
    else:
        datasets = [args.dataset]

    # Process each dataset
    all_stats = []

    for dataset in datasets:
        try:
            stats = deduplicate_dataset(
                dataset_name=dataset,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                verbose=args.verbose
            )
            all_stats.append(stats)

        except Exception as e:
            print(f"\n❌ Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print overall summary
    if len(all_stats) > 1:
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)

        total_input = sum(s['input']['total_rows'] for s in all_stats)
        total_output = sum(s['output']['unique_rows'] for s in all_stats)
        total_duplicates = sum(s['deduplication']['duplicates_found'] for s in all_stats)
        total_time = sum(s['processing']['duration_seconds'] for s in all_stats)

        print(f"Datasets processed: {len(all_stats)}")
        print(f"Total input rows:   {format_number(total_input)}")
        print(f"Total output rows:  {format_number(total_output)}")
        print(f"Total duplicates:   {format_number(total_duplicates)} ({format_percentage(total_duplicates, total_input)})")
        print(f"Total time:         {total_time / 60:.1f} minutes")
        print("=" * 70)

        # Save combined stats
        if not args.dry_run:
            combined_stats_path = os.path.join(args.output_dir, 'phase1_summary.json')
            combined_stats = {
                'phase': 'intra',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'datasets_processed': len(all_stats),
                'total_input_rows': total_input,
                'total_output_rows': total_output,
                'total_duplicates': total_duplicates,
                'duplicate_rate': total_duplicates / total_input if total_input > 0 else 0,
                'total_duration_seconds': total_time,
                'individual_stats': all_stats
            }
            save_stats(combined_stats, combined_stats_path)

    print("\n✅ Phase 1 (Intra-dataset deduplication) completed!\n")


if __name__ == '__main__':
    main()
