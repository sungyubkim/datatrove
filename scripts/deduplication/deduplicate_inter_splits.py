#!/usr/bin/env python3
"""
Phase 2: Inter-Dataset Deduplication for math-verl-unified Hub Structure

This script removes duplicates across different dataset splits while maintaining
the Hub's flat folder structure (data/*.parquet files).
"""

import argparse
import os
import sys
import time
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
    normalize_row_schema,
)


# Standard VERL schema for consistent output
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
        ('index', pa.int64())
    ]))
])


def deduplicate_inter_splits(
    input_data_dir: str,
    output_data_dir: str,
    priority_order: List[str],
    batch_size: int = 10000,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Remove duplicates across dataset splits using priority order.

    This version is designed for HuggingFace Hub's flat folder structure,
    where all splits are in a single data/ folder as individual parquet files.

    Args:
        input_data_dir: Input data directory (e.g., downloads/math-verl-unified/data/)
        output_data_dir: Output data directory (e.g., deduplicated-inter/data/)
        priority_order: List of split names in priority order (highest first)
        batch_size: Rows per batch for processing
        verbose: Print detailed progress

    Returns:
        Statistics dictionary
    """
    start_time = time.time()

    print(f"\n{'=' * 70}")
    print("Phase 2: Inter-Dataset Deduplication (Hub Structure)")
    print(f"{'=' * 70}\n")

    # Global hash set (across all splits)
    global_hash_set: Set[str] = set()

    # Track which split each hash comes from
    hash_to_split: Dict[str, str] = {}

    # Statistics
    stats_by_split = {}
    cross_split_duplicates = defaultdict(lambda: defaultdict(int))

    # Per-split writers
    split_writers: Dict[str, pq.ParquetWriter] = {}

    # Create output directory
    os.makedirs(output_data_dir, exist_ok=True)

    print("Priority order (highest to lowest):")
    for i, split_name in enumerate(priority_order, 1):
        print(f"  {i}. {split_name}")
    print()

    # Process each split in priority order
    for priority_idx, split_name in enumerate(priority_order):
        # Input file path
        input_file = os.path.join(input_data_dir, f"{split_name}.parquet")

        # Check if split exists
        if not os.path.exists(input_file):
            print(f"⚠️  Warning: Split not found, skipping: {split_name}")
            print(f"    Expected at: {input_file}")
            continue

        print(f"\n{'─' * 70}")
        print(f"[{priority_idx + 1}/{len(priority_order)}] Processing: {split_name}")
        print(f"{'─' * 70}")

        # Split statistics
        total_rows = 0
        kept_rows = 0
        duplicate_rows = 0

        # Output file path
        output_file = os.path.join(output_data_dir, f"{split_name}.parquet")

        # Read input file
        table = pq.read_table(input_file)
        total_rows_in_file = table.num_rows

        print(f"  Input file: {os.path.basename(input_file)}")
        print(f"  Total rows: {format_number(total_rows_in_file)}")

        # Create writer for this split
        split_writers[split_name] = pq.ParquetWriter(output_file, STANDARD_SCHEMA)

        # Process in batches
        for batch_start in tqdm(
            range(0, total_rows_in_file, batch_size),
            desc=f"  Deduplicating",
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
                    # Duplicate with higher-priority split
                    duplicate_rows += 1
                    source_split = hash_to_split.get(problem_hash, 'unknown')
                    cross_split_duplicates[split_name][source_split] += 1
                else:
                    # New unique problem
                    kept_rows += 1
                    global_hash_set.add(problem_hash)
                    hash_to_split[problem_hash] = split_name

                    # Normalize row to standard schema
                    normalized_row = normalize_row_schema(row_dict)
                    unique_rows.append(normalized_row)

            # Write unique rows for this batch
            if unique_rows:
                df_unique = pd.DataFrame(unique_rows)
                # Enforce standard schema for consistency
                table_unique = pa.Table.from_pandas(df_unique, schema=STANDARD_SCHEMA)
                split_writers[split_name].write_table(table_unique)

                if verbose:
                    print(f"  Wrote {len(unique_rows)} unique rows")

        # Close writer for this split
        split_writers[split_name].close()

        # Split summary
        stats_by_split[split_name] = {
            'input_rows': total_rows,
            'kept_rows': kept_rows,
            'removed_duplicates': duplicate_rows,
            'duplicate_rate': duplicate_rows / total_rows if total_rows > 0 else 0,
            'duplicate_sources': dict(cross_split_duplicates[split_name])
        }

        print(f"  Input:      {format_number(total_rows)}")
        print(f"  Kept:       {format_number(kept_rows)}")
        print(f"  Duplicates: {format_number(duplicate_rows)} ({format_percentage(duplicate_rows, total_rows)})")
        print(f"  Output:     {output_file}")

    # Overall statistics
    duration = time.time() - start_time

    total_input = sum(s['input_rows'] for s in stats_by_split.values())
    total_kept = sum(s['kept_rows'] for s in stats_by_split.values())
    total_removed = sum(s['removed_duplicates'] for s in stats_by_split.values())

    overall_stats = {
        'phase': 'inter-splits',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'priority_order': priority_order,
        'total_input_rows': total_input,
        'total_output_rows': total_kept,
        'total_duplicates_removed': total_removed,
        'duplicate_rate': total_removed / total_input if total_input > 0 else 0,
        'by_split': stats_by_split,
        'processing': {
            'duration_seconds': duration,
            'rows_per_second': total_input / duration if duration > 0 else 0
        }
    }

    # Save statistics
    stats_dir = os.path.join(os.path.dirname(output_data_dir), 'stats')
    os.makedirs(stats_dir, exist_ok=True)

    stats_file = os.path.join(stats_dir, 'phase2_inter_splits_stats.json')
    save_stats(overall_stats, stats_file)

    # Save duplicate matrix separately
    duplicate_matrix = {
        split: dict(dups) for split, dups in cross_split_duplicates.items()
    }
    matrix_file = os.path.join(stats_dir, 'duplicate_matrix.json')
    save_stats(duplicate_matrix, matrix_file)

    # Print final summary
    print(f"\n{'=' * 70}")
    print("PHASE 2 SUMMARY")
    print(f"{'=' * 70}")
    print(f"Splits processed:    {len(stats_by_split)}")
    print(f"Total input rows:    {format_number(total_input)}")
    print(f"Total output rows:   {format_number(total_kept)}")
    print(f"Duplicates removed:  {format_number(total_removed)} ({format_percentage(total_removed, total_input)})")
    print(f"Processing time:     {duration / 60:.1f} minutes")
    print(f"Output directory:    {output_data_dir}")
    print(f"Statistics:          {stats_dir}")
    print(f"{'=' * 70}\n")

    return overall_stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 2: Inter-split deduplication for Hub structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with priority file
  python deduplicate_inter_splits.py --priority-file math_verl_priority.txt

  # Custom input/output directories
  python deduplicate_inter_splits.py \\
    --input-dir output/downloads/math-verl-unified/data \\
    --output-dir output/deduplicated-inter/data \\
    --priority-file math_verl_priority.txt

  # Verbose mode
  python deduplicate_inter_splits.py --priority-file math_verl_priority.txt --verbose
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='output/downloads/math-verl-unified/data',
        help='Input data directory containing split parquet files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/deduplicated-inter/data',
        help='Output data directory for deduplicated parquet files'
    )

    parser.add_argument(
        '--priority-file',
        type=str,
        required=True,
        help='File containing split priority order (one per line, highest priority first)'
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

    # Load priority order from file
    if not os.path.exists(args.priority_file):
        print(f"❌ Error: Priority file not found: {args.priority_file}")
        sys.exit(1)

    with open(args.priority_file, 'r') as f:
        priority_order = [line.strip() for line in f if line.strip()]

    if not priority_order:
        print(f"❌ Error: Priority file is empty: {args.priority_file}")
        sys.exit(1)

    print(f"Loaded priority order from: {args.priority_file}")
    print(f"Splits: {len(priority_order)}")

    # Run deduplication
    try:
        stats = deduplicate_inter_splits(
            input_data_dir=args.input_dir,
            output_data_dir=args.output_dir,
            priority_order=priority_order,
            batch_size=args.batch_size,
            verbose=args.verbose
        )

        print("✅ Phase 2 (Inter-split deduplication) completed!\n")

    except Exception as e:
        print(f"\n❌ Error during Phase 2: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
