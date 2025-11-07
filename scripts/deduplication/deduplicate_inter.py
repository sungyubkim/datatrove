#!/usr/bin/env python3
"""
Phase 2: Inter-Dataset Deduplication

This script removes duplicates across different datasets, using a priority order
to preserve higher-priority datasets.
"""

import argparse
import os
import sys
import time
import glob
from typing import Dict, Any, List, Set
from collections import defaultdict

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
)


def deduplicate_inter_dataset(
    input_dir: str,
    output_dir: str,
    priority_order: List[str],
    combine_output: bool = True,
    batch_size: int = 10000,
    verbose: bool = False,
    use_current_datasets: bool = False
) -> Dict[str, Any]:
    """
    Remove duplicates across datasets using priority order.

    Args:
        input_dir: Directory containing Phase 1 deduplicated datasets
        output_dir: Output directory for Phase 2 results
        priority_order: List of dataset names in priority order (highest first)
        combine_output: If True, create single combined dataset
        batch_size: Rows per batch for processing
        verbose: Print detailed progress
        use_current_datasets: If True, read from current directory ./{dataset}/data/

    Returns:
        Statistics dictionary
    """
    start_time = time.time()

    print(f"\n{'=' * 70}")
    print("Phase 2: Inter-Dataset Deduplication")
    print(f"{'=' * 70}\n")

    # Global hash set (across all datasets)
    global_hash_set: Set[str] = set()

    # Track which dataset each hash comes from
    hash_to_dataset: Dict[str, str] = {}

    # Statistics
    stats_by_dataset = {}
    cross_dataset_duplicates = defaultdict(lambda: defaultdict(int))

    # Output setup
    combined_output_path = os.path.join(output_dir, 'combined', 'data')
    if combine_output:
        os.makedirs(combined_output_path, exist_ok=True)

    combined_writer = None
    combined_schema = None
    combined_file_idx = 0
    combined_rows_in_file = 0
    ROWS_PER_FILE = 500000

    print("Priority order:")
    for i, ds in enumerate(priority_order, 1):
        print(f"  {i}. {ds}")
    print()

    # Process each dataset in priority order
    for priority_idx, dataset_name in enumerate(priority_order):
        # Determine input path based on mode
        if use_current_datasets:
            # Read from current directory: ./{dataset_name}/data/
            dataset_input_path = os.path.join('.', dataset_name, 'data')
        else:
            # Read from Phase 1 output: {input_dir}/{dataset_name}/data/
            dataset_input_path = os.path.join(input_dir, dataset_name, 'data')

        # Check if dataset exists
        if not os.path.exists(dataset_input_path):
            print(f"⚠️  Warning: Dataset not found, skipping: {dataset_name}")
            continue

        # Find parquet files
        parquet_files = sorted(glob.glob(os.path.join(dataset_input_path, '*.parquet')))
        if not parquet_files:
            print(f"⚠️  Warning: No parquet files in {dataset_name}, skipping")
            continue

        print(f"\n{'─' * 70}")
        print(f"[{priority_idx + 1}/{len(priority_order)}] Processing: {dataset_name}")
        print(f"{'─' * 70}")

        # Dataset statistics
        total_rows = 0
        kept_rows = 0
        duplicate_rows = 0

        # Process files
        for file_idx, input_file in enumerate(parquet_files):
            if verbose:
                print(f"  File {file_idx + 1}/{len(parquet_files)}: {os.path.basename(input_file)}")

            table = pq.read_table(input_file)
            total_rows_in_file = table.num_rows

            if combined_schema is None:
                combined_schema = table.schema

            # Process in batches
            for batch_start in tqdm(
                range(0, total_rows_in_file, batch_size),
                desc=f"  Processing",
                disable=not verbose
            ):
                batch_end = min(batch_start + batch_size, total_rows_in_file)
                batch = table.slice(batch_start, batch_end - batch_start)
                df_batch = batch.to_pandas()

                unique_rows = []

                for idx, row in df_batch.iterrows():
                    row_dict = row.to_dict()
                    total_rows += 1

                    # Validate
                    is_valid, error_msg = validate_verl_row(row_dict)
                    if not is_valid:
                        if verbose:
                            print(f"  ⚠️  Invalid row: {error_msg}")
                        continue

                    # Extract and hash
                    try:
                        problem_text = extract_problem_text(row_dict)
                        problem_hash = compute_hash(problem_text)
                    except Exception as e:
                        if verbose:
                            print(f"  ⚠️  Error: {e}")
                        continue

                    # Check against global hash set
                    if problem_hash in global_hash_set:
                        # Duplicate with higher-priority dataset
                        duplicate_rows += 1
                        source_dataset = hash_to_dataset.get(problem_hash, 'unknown')
                        cross_dataset_duplicates[dataset_name][source_dataset] += 1
                    else:
                        # New unique problem
                        kept_rows += 1
                        global_hash_set.add(problem_hash)
                        hash_to_dataset[problem_hash] = dataset_name

                        # Normalize extra_info to consistent schema
                        original_extra_info = row_dict.get('extra_info', {})
                        row_dict['extra_info'] = {
                            'index': original_extra_info.get('index', 0),
                            'split': original_extra_info.get('split', 'train'),
                            'original_dataset': dataset_name
                        }

                        unique_rows.append(row_dict)

                # Write to combined output
                if combine_output and unique_rows:
                    df_unique = pd.DataFrame(unique_rows)
                    # Don't enforce schema to allow new fields like original_dataset
                    table_unique = pa.Table.from_pandas(df_unique)

                    # Create or rotate writer
                    if combined_writer is None or combined_rows_in_file >= ROWS_PER_FILE:
                        if combined_writer is not None:
                            combined_writer.close()

                        output_filename = f"train-{combined_file_idx:05d}.parquet"
                        output_filepath = os.path.join(combined_output_path, output_filename)
                        # Use schema from the table being written (includes original_dataset)
                        combined_writer = pq.ParquetWriter(output_filepath, table_unique.schema)
                        combined_rows_in_file = 0
                        combined_file_idx += 1

                        if verbose:
                            print(f"  Writing to combined: {output_filename}")

                    combined_writer.write_table(table_unique)
                    combined_rows_in_file += len(unique_rows)

        # Dataset summary
        stats_by_dataset[dataset_name] = {
            'input_rows': total_rows,
            'kept_rows': kept_rows,
            'removed_duplicates': duplicate_rows,
            'duplicate_rate': duplicate_rows / total_rows if total_rows > 0 else 0,
            'duplicate_sources': dict(cross_dataset_duplicates[dataset_name])
        }

        print(f"  Input:      {format_number(total_rows)}")
        print(f"  Kept:       {format_number(kept_rows)}")
        print(f"  Duplicates: {format_number(duplicate_rows)} ({format_percentage(duplicate_rows, total_rows)})")

    # Close combined writer
    if combined_writer is not None:
        combined_writer.close()

    # Overall statistics
    duration = time.time() - start_time

    total_input = sum(s['input_rows'] for s in stats_by_dataset.values())
    total_kept = sum(s['kept_rows'] for s in stats_by_dataset.values())
    total_removed = sum(s['removed_duplicates'] for s in stats_by_dataset.values())

    overall_stats = {
        'phase': 'inter',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'priority_order': priority_order,
        'total_input_rows': total_input,
        'total_output_rows': total_kept,
        'total_duplicates_removed': total_removed,
        'duplicate_rate': total_removed / total_input if total_input > 0 else 0,
        'by_dataset': stats_by_dataset,
        'processing': {
            'duration_seconds': duration,
            'rows_per_second': total_input / duration if duration > 0 else 0
        }
    }

    # Save statistics
    stats_dir = os.path.join(output_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)

    stats_file = os.path.join(stats_dir, 'phase2_stats.json')
    save_stats(overall_stats, stats_file)

    # Print final summary
    print(f"\n{'=' * 70}")
    print("PHASE 2 SUMMARY")
    print(f"{'=' * 70}")
    print(f"Datasets processed:  {len(stats_by_dataset)}")
    print(f"Total input rows:    {format_number(total_input)}")
    print(f"Total output rows:   {format_number(total_kept)}")
    print(f"Duplicates removed:  {format_number(total_removed)} ({format_percentage(total_removed, total_input)})")
    print(f"Processing time:     {duration / 60:.1f} minutes")
    if combine_output:
        print(f"Combined output:     {os.path.join(output_dir, 'combined')}")
        print(f"Output files:        {combined_file_idx}")
    print(f"{'=' * 70}\n")

    return overall_stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 2: Inter-dataset deduplication',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with priority file
  python deduplicate_inter.py --priority-file qa_priority.txt

  # Custom input/output directories
  python deduplicate_inter.py --priority-file qa_priority.txt --input-dir _deduplicated/phase1-intra --output-dir _deduplicated/phase2-inter

  # Use current directory datasets (not Phase 1 output)
  python deduplicate_inter.py --priority-file if_priority.txt --use-current-datasets
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='_deduplicated/phase1-intra',
        help='Input directory with Phase 1 results (default: _deduplicated/phase1-intra)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='_deduplicated/phase2-inter',
        help='Output directory (default: _deduplicated/phase2-inter)'
    )

    parser.add_argument(
        '--priority-file',
        type=str,
        required=True,
        help='File containing dataset priority order (one per line, smallest first)'
    )

    parser.add_argument(
        '--no-combine',
        action='store_true',
        help='Do not create combined output (keep separate datasets)'
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

    parser.add_argument(
        '--use-current-datasets',
        action='store_true',
        help='Read datasets from current directory (./{dataset}/data/) instead of input-dir'
    )

    args = parser.parse_args()

    # Load priority order from file
    with open(args.priority_file, 'r') as f:
        priority_order = [line.strip() for line in f if line.strip()]

    print(f"Loaded priority order from: {args.priority_file}")
    print(f"Datasets: {len(priority_order)}")

    # Run deduplication
    try:
        stats = deduplicate_inter_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            priority_order=priority_order,
            combine_output=not args.no_combine,
            batch_size=args.batch_size,
            verbose=args.verbose,
            use_current_datasets=args.use_current_datasets
        )

        print("✅ Phase 2 (Inter-dataset deduplication) completed!\n")

    except Exception as e:
        print(f"\n❌ Error during Phase 2: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
