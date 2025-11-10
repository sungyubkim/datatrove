#!/usr/bin/env python3
"""
Dataset Comparison Tool

Compares two VERL format datasets to analyze differences in:
1. Sample counts
2. Content changes (cleaning effects)
3. Specific example diffs

Usage:
    python scripts/validation/compare_datasets.py \
        --original output/dapo-math-original/train.parquet \
        --processed output/dapo-math-cleaned-deduped/train.parquet \
        --output output/comparison/
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of text."""
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()


def extract_problem_text(row: dict) -> str:
    """Extract problem text from VERL format row."""
    if 'prompt' not in row:
        return ""

    prompt = row['prompt']
    if not isinstance(prompt, list) or len(prompt) == 0:
        return ""

    first_message = prompt[0]
    if isinstance(first_message, dict) and 'content' in first_message:
        return first_message['content']

    return ""


def compare_datasets(
    original_file: str,
    processed_file: str,
    output_dir: Path,
    num_samples: int = 50,
) -> dict:
    """Compare two datasets and generate analysis.

    Args:
        original_file: Path to original dataset parquet
        processed_file: Path to processed dataset parquet
        output_dir: Output directory for comparison results
        num_samples: Number of random samples to show in detail

    Returns:
        Statistics dictionary
    """
    print(f"\n{'='*70}")
    print(f"Dataset Comparison Analysis")
    print(f"{'='*70}")
    print(f"Original:  {original_file}")
    print(f"Processed: {processed_file}")
    print(f"{'='*70}\n")

    # Load datasets
    print("Step 1: Loading datasets...")
    try:
        original_ds = load_dataset("parquet", data_files=original_file, split="train")
        processed_ds = load_dataset("parquet", data_files=processed_file, split="train")
        print(f"✓ Original:  {len(original_ds):,} samples")
        print(f"✓ Processed: {len(processed_ds):,} samples")
    except Exception as e:
        print(f"✗ Failed to load datasets: {e}")
        return {}

    # Build hash maps
    print("\nStep 2: Building hash maps for comparison...")
    original_hashes = {}
    processed_hashes = {}

    for idx, row in enumerate(original_ds):
        problem_text = extract_problem_text(row)
        if problem_text:
            h = compute_hash(problem_text)
            original_hashes[h] = {
                'index': idx,
                'problem': problem_text,
                'ground_truth': row.get('reward_model', {}).get('ground_truth', 'N/A') if isinstance(row.get('reward_model'), dict) else 'N/A',
                'data': row
            }

    for idx, row in enumerate(processed_ds):
        problem_text = extract_problem_text(row)
        if problem_text:
            h = compute_hash(problem_text)
            processed_hashes[h] = {
                'index': idx,
                'problem': problem_text,
                'ground_truth': row.get('reward_model', {}).get('ground_truth', 'N/A') if isinstance(row.get('reward_model'), dict) else 'N/A',
                'data': row
            }

    print(f"✓ Original unique problems:  {len(original_hashes):,}")
    print(f"✓ Processed unique problems: {len(processed_hashes):,}")

    # Find differences
    print("\nStep 3: Analyzing differences...")

    only_in_original = set(original_hashes.keys()) - set(processed_hashes.keys())
    only_in_processed = set(processed_hashes.keys()) - set(original_hashes.keys())
    in_both = set(original_hashes.keys()) & set(processed_hashes.keys())

    print(f"  Common problems:     {len(in_both):,}")
    print(f"  Only in original:    {len(only_in_original):,}")
    print(f"  Only in processed:   {len(only_in_processed):,}")

    # Analyze text changes for common problems
    print("\nStep 4: Analyzing text changes...")
    text_changes = []
    identical_count = 0

    for h in in_both:
        orig = original_hashes[h]
        proc = processed_hashes[h]

        if orig['problem'] == proc['problem']:
            identical_count += 1
        else:
            text_changes.append({
                'hash': h,
                'original_problem': orig['problem'],
                'processed_problem': proc['problem'],
                'original_gt': orig['ground_truth'],
                'processed_gt': proc['ground_truth'],
            })

    print(f"  Identical text:      {identical_count:,}")
    print(f"  Modified text:       {len(text_changes):,}")

    # Sample selection
    print(f"\nStep 5: Selecting random samples for detailed comparison...")

    # Select samples
    sample_only_original = random.sample(list(only_in_original), min(10, len(only_in_original))) if only_in_original else []
    sample_only_processed = random.sample(list(only_in_processed), min(10, len(only_in_processed))) if only_in_processed else []
    sample_text_changes = random.sample(text_changes, min(num_samples, len(text_changes))) if text_changes else []

    print(f"✓ Selected {len(sample_text_changes)} text change examples")

    # Statistics
    stats = {
        'original_count': len(original_ds),
        'processed_count': len(processed_ds),
        'original_unique': len(original_hashes),
        'processed_unique': len(processed_hashes),
        'common_problems': len(in_both),
        'only_in_original': len(only_in_original),
        'only_in_processed': len(only_in_processed),
        'identical_text': identical_count,
        'modified_text': len(text_changes),
        'sample_only_original': sample_only_original,
        'sample_only_processed': sample_only_processed,
        'sample_text_changes': sample_text_changes,
    }

    return stats, original_hashes, processed_hashes


def generate_report(
    original_file: str,
    processed_file: str,
    stats: dict,
    original_hashes: dict,
    processed_hashes: dict,
) -> str:
    """Generate comparison report.

    Args:
        original_file: Original dataset path
        processed_file: Processed dataset path
        stats: Statistics dictionary
        original_hashes: Original dataset hash map
        processed_hashes: Processed dataset hash map

    Returns:
        Report text
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
{'='*70}
Dataset Comparison Report
{'='*70}
Original Dataset:  {original_file}
Processed Dataset: {processed_file}
Timestamp:         {timestamp}

{'='*70}
Summary Statistics
{'='*70}
Dataset Size:
  Original:        {stats['original_count']:,} samples
  Processed:       {stats['processed_count']:,} samples
  Difference:      {stats['processed_count'] - stats['original_count']:+,} samples

Unique Problems (by hash):
  Original:        {stats['original_unique']:,} unique problems
  Processed:       {stats['processed_unique']:,} unique problems

Set Comparison:
  Common problems: {stats['common_problems']:,}
  Only in original:{stats['only_in_original']:,}
  Only in processed:{stats['only_in_processed']:,}

Text Changes (for common problems):
  Identical:       {stats['identical_text']:,} ({100*stats['identical_text']/stats['common_problems']:.1f}%)
  Modified:        {stats['modified_text']:,} ({100*stats['modified_text']/stats['common_problems']:.1f}%)

{'='*70}
Interpretation
{'='*70}
"""

    if stats['processed_count'] > stats['original_count']:
        diff = stats['processed_count'] - stats['original_count']
        report += f"The processed dataset has {diff} MORE samples than the original.\n"
        report += f"This suggests that the source dataset had {diff} additional samples\n"
        report += f"that were not present in the Hub version.\n\n"
    elif stats['processed_count'] < stats['original_count']:
        diff = stats['original_count'] - stats['processed_count']
        report += f"The processed dataset has {diff} FEWER samples than the original.\n"
        report += f"This is due to deduplication and filtering during processing.\n\n"
    else:
        report += f"The datasets have the SAME number of samples.\n\n"

    if stats['only_in_original'] > 0:
        report += f"\nProblems only in original ({stats['only_in_original']} samples):\n"
        report += f"These were removed during deduplication or filtering.\n"

    if stats['only_in_processed'] > 0:
        report += f"\nProblems only in processed ({stats['only_in_processed']} samples):\n"
        report += f"These are NEW samples not present in the Hub version.\n"

    # Sample examples - Only in Original
    if stats['sample_only_original']:
        report += f"\n{'='*70}\n"
        report += f"Sample: Problems Only in Original Dataset ({len(stats['sample_only_original'])} shown)\n"
        report += f"{'='*70}\n"

        for i, h in enumerate(stats['sample_only_original'], 1):
            orig = original_hashes[h]
            report += f"\n{'-'*70}\n"
            report += f"Example {i}\n"
            report += f"{'-'*70}\n"
            report += f"Problem:\n{orig['problem'][:2000]}\n\n"
            report += f"Ground Truth: {orig['ground_truth'][:500]}\n"

    # Sample examples - Only in Processed
    if stats['sample_only_processed']:
        report += f"\n{'='*70}\n"
        report += f"Sample: Problems Only in Processed Dataset ({len(stats['sample_only_processed'])} shown)\n"
        report += f"{'='*70}\n"

        for i, h in enumerate(stats['sample_only_processed'], 1):
            proc = processed_hashes[h]
            report += f"\n{'-'*70}\n"
            report += f"Example {i}\n"
            report += f"{'-'*70}\n"
            report += f"Problem:\n{proc['problem'][:2000]}\n\n"
            report += f"Ground Truth: {proc['ground_truth'][:500]}\n"

    # Sample examples - Text changes
    if stats['sample_text_changes']:
        report += f"\n{'='*70}\n"
        report += f"Sample: Text Changes from Cleaning ({len(stats['sample_text_changes'])} shown)\n"
        report += f"{'='*70}\n"

        for i, change in enumerate(stats['sample_text_changes'][:20], 1):
            report += f"\n{'-'*70}\n"
            report += f"Example {i}\n"
            report += f"{'-'*70}\n"
            report += f"ORIGINAL:\n{change['original_problem'][:2000]}\n\n"
            report += f"PROCESSED:\n{change['processed_problem'][:2000]}\n\n"

            # Show what changed
            if change['original_problem'] != change['processed_problem']:
                # Simple diff - show first difference
                orig_lines = change['original_problem'].split('\n')
                proc_lines = change['processed_problem'].split('\n')
                report += f"Changes detected:\n"
                if len(orig_lines) != len(proc_lines):
                    report += f"  - Line count changed: {len(orig_lines)} → {len(proc_lines)}\n"
                if change['original_problem'].startswith(tuple('0123456789')) and not change['processed_problem'].startswith(tuple('0123456789')):
                    report += f"  - Problem number removed from start\n"

    report += f"\n{'='*70}\n"

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare original and processed datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--original',
        type=str,
        required=True,
        help='Path to original dataset parquet file'
    )

    parser.add_argument(
        '--processed',
        type=str,
        required=True,
        help='Path to processed dataset parquet file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./output/comparison/',
        help='Output directory for comparison results'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=50,
        help='Number of random samples to include in report'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison
    stats, original_hashes, processed_hashes = compare_datasets(
        original_file=args.original,
        processed_file=args.processed,
        output_dir=output_dir,
        num_samples=args.num_samples,
    )

    if not stats:
        print("\n✗ Comparison failed")
        return

    # Generate report
    report = generate_report(
        original_file=args.original,
        processed_file=args.processed,
        stats=stats,
        original_hashes=original_hashes,
        processed_hashes=processed_hashes,
    )

    # Print report
    print(report)

    # Save report
    report_path = output_dir / "comparison_report.txt"
    report_path.write_text(report)
    print(f"✓ Report saved to: {report_path}")

    # Save stats as JSON (without large sample data)
    stats_for_json = {k: v for k, v in stats.items() if not k.startswith('sample_')}
    stats_for_json['num_text_change_samples'] = len(stats['sample_text_changes'])

    stats_path = output_dir / "comparison_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "original_file": args.original,
            "processed_file": args.processed,
            "timestamp": datetime.now().isoformat(),
            "stats": stats_for_json,
        }, f, indent=2)
    print(f"✓ Stats saved to: {stats_path}")

    print(f"\n{'='*70}")
    print(f"✅ Comparison completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
