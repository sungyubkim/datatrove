#!/usr/bin/env python3
"""
Code Instruction Standardization: Format Standardization + Deduplication

This script processes code datasets (local parquet files OR HuggingFace Hub) through two stages:
1. Standardization: Apply InstructionStandardizer with specified preset
2. Deduplication: Remove exact duplicates based on problem text hash

Input: Local parquet file OR HuggingFace Hub dataset identifier (VERL format)
Output: Single standardized and deduplicated parquet file

Usage:
    # Local file with Python preset
    python scripts/processing/standardize_instructions.py \
        --input output/kodcode-original/train.parquet \
        --output output/kodcode-standardized/train.parquet \
        --preset python-code

    # HuggingFace Hub dataset with Verilog preset
    python scripts/processing/standardize_instructions.py \
        --input sungyub/codev-r1-verl \
        --output output/codev-standardized/train.parquet \
        --preset verilog-hdl

    # Competitive programming preset
    python scripts/processing/standardize_instructions.py \
        --input sungyub/eurus-2-code-verl \
        --output output/eurus-standardized/train.parquet \
        --preset competitive
"""

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset

from datatrove.data import Document
from datatrove.pipeline.formatters import InstructionStandardizer


def is_local_file(input_path: str) -> bool:
    """Check if input is a local file path or HuggingFace Hub dataset.

    Args:
        input_path: Input path or dataset identifier

    Returns:
        True if local file, False if HuggingFace Hub dataset identifier
    """
    # Check if path exists on filesystem
    if Path(input_path).exists():
        return True

    # Check if it's a Hub dataset identifier (format: "username/dataset-name")
    # Hub identifiers don't start with ./ or ../ and have exactly one /
    if "/" in input_path:
        parts = input_path.split("/")
        # Exactly 2 parts and not a relative path -> likely Hub dataset
        if len(parts) == 2 and not input_path.startswith("./") and not input_path.startswith("../"):
            return False

    # Default to treating as local file (will fail with clear error if doesn't exist)
    return True


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of text.

    Args:
        text: Text to hash

    Returns:
        Hex digest of hash
    """
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()


def extract_problem_text(row_dict: dict) -> str:
    """Extract problem text from VERL format row.

    Args:
        row_dict: Row dictionary with VERL schema

    Returns:
        Problem text string
    """
    if 'prompt' not in row_dict:
        return ""

    prompt = row_dict['prompt']
    if not isinstance(prompt, list) or len(prompt) == 0:
        return ""

    first_message = prompt[0]
    if isinstance(first_message, dict) and 'content' in first_message:
        return first_message['content']

    return ""


def extract_problem_section(text: str) -> str:
    """Extract only the Problem section from standardized instruction format.

    This ensures deduplication is based solely on problem content,
    not on format changes (e.g., adding Implementation Requirements section).

    Args:
        text: Full instruction text with sections

    Returns:
        Problem section content only, or full text if no Problem section found
    """
    if not text:
        return ""

    # Try to extract Problem section using regex
    # Match: ## Problem\n ... until next ## or end of string
    match = re.search(r'## Problem\n(.*?)(?=\n##|$)', text, re.DOTALL)

    if match:
        return match.group(1).strip()

    # If no Problem section found, return full text (backward compatibility)
    return text.strip()


def process_dataset(
    input_file: str,
    output_file: str,
    preset_name: str = "python-code",
    batch_size: int = 1000,
    sample_rate: int = 1000,
) -> dict:
    """Process dataset through instruction standardization and deduplication.

    Args:
        input_file: Path to input parquet file or Hub dataset ID
        output_file: Path to output parquet file
        preset_name: Standardization preset name
        batch_size: Batch size for processing
        sample_rate: Collect sample every N documents

    Returns:
        Statistics dictionary with samples
    """
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"Code Instruction Standardization: Format + Deduplication")
    print(f"{'='*70}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Preset: {preset_name}")
    print(f"{'='*70}\n")

    # Load input dataset
    print("Step 1: Loading input dataset...")
    try:
        is_local = is_local_file(input_file)

        if is_local:
            print(f"  Loading local parquet file: {input_file}")
            dataset = load_dataset("parquet", data_files=input_file, split="train", streaming=True)
        else:
            print(f"  Loading from HuggingFace Hub: {input_file}")
            dataset = load_dataset(input_file, split="train", streaming=True)

        print(f"✓ Dataset loaded in streaming mode")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return {}

    # Initialize standardizer
    print(f"\nStep 2: Initializing InstructionStandardizer (preset: {preset_name})...")
    try:
        standardizer = InstructionStandardizer.from_preset(preset_name)
        print(f"✓ Standardizer initialized")
    except ValueError as e:
        print(f"✗ Invalid preset: {e}")
        return {}

    # Statistics tracking
    stats = {
        'total_input': 0,
        'after_standardization': 0,
        'filtered_by_standardization': 0,
        'after_deduplication': 0,
        'duplicates_removed': 0,
        'standardization_stats': defaultdict(int),
        'duplicate_hashes': defaultdict(int),
        'sample_examples': [],  # Collect sample examples
    }

    # Process: Standardization
    print(f"\nStep 3: Applying standardization and deduplication with PyArrow streaming...")
    print(f"  This may take a while for {input_file}...")

    # PyArrow streaming setup
    hash_set = set()
    hash_to_text = {}
    batch_data = []
    BATCH_SIZE = 10000
    writer = None

    # Get VERL schema from first example
    schema = None

    # Create document generator
    def document_generator(dataset):
        """Generate Documents from dataset."""
        for idx, example in enumerate(dataset):
            doc = Document(
                id=f"doc-{idx}",
                text="",  # VERL format uses metadata
                metadata=example,
            )
            yield doc

    # Apply standardizer
    doc_generator = document_generator(dataset)
    standardized_generator = standardizer.run(doc_generator, rank=0, world_size=1)

    # Collect and deduplicate with streaming writes
    doc_count = 0
    unique_count = 0
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for doc in standardized_generator:
            doc_count += 1
            stats['total_input'] = doc_count
            stats['after_standardization'] = doc_count

            # Extract problem text and hash
            if doc.metadata and 'prompt' in doc.metadata:
                problem_text = extract_problem_text(doc.metadata)
                if problem_text:
                    # Extract ONLY Problem section for consistent hashing across format changes
                    problem_section = extract_problem_section(problem_text)
                    problem_hash = compute_hash(problem_section)

                    # Check for duplicate
                    if problem_hash in hash_set:
                        stats['duplicate_hashes'][problem_hash] += 1
                        continue  # Skip duplicate

                    # Add to unique set
                    hash_set.add(problem_hash)
                    hash_to_text[problem_hash] = problem_section[:100]  # Store snippet

                    # Keep this example
                    batch_data.append(doc.metadata)
                    unique_count += 1

                    # Collect samples for report (every sample_rate documents)
                    if doc_count % sample_rate == 0 and len(stats['sample_examples']) < 50:
                        ground_truth = ""
                        if 'reward_model' in doc.metadata and isinstance(doc.metadata['reward_model'], dict):
                            ground_truth = doc.metadata['reward_model'].get('ground_truth', 'N/A')

                        stats['sample_examples'].append({
                            'index': doc_count - 1,
                            'problem': problem_text,  # Full problem text
                            'ground_truth': ground_truth if ground_truth else 'N/A',  # Full ground truth
                        })

                    # Write batch when full
                    if len(batch_data) >= BATCH_SIZE:
                        batch_table = pa.Table.from_pylist(batch_data, schema=schema)

                        # Infer schema from first batch
                        if schema is None:
                            schema = batch_table.schema

                        if writer is None:
                            # First batch: create writer
                            writer = pq.ParquetWriter(output_file, schema=schema, compression='snappy')

                        writer.write_table(batch_table)

                        # Progress indicator
                        dupe_count = doc_count - unique_count
                        print(f"  Processed {doc_count:,} documents... (unique: {unique_count:,}, duplicates: {dupe_count:,})", end="\r", flush=True)

                        # Clear batch
                        batch_data = []

            # Progress indicator (for skipped docs)
            if doc_count % 1000 == 0 and len(batch_data) < BATCH_SIZE:
                dupe_count = doc_count - unique_count
                print(f"  Processed {doc_count:,} documents... (unique: {unique_count:,}, duplicates: {dupe_count:,})", end="\r", flush=True)

        # Write remaining batch
        if batch_data:
            batch_table = pa.Table.from_pylist(batch_data, schema=schema)

            if schema is None:
                schema = batch_table.schema

            if writer is None:
                # Only one batch: use simple write
                pq.write_table(batch_table, output_file, compression='snappy')
            else:
                writer.write_table(batch_table)

    finally:
        # Close writer
        if writer is not None:
            writer.close()

    # Final counts
    stats['after_deduplication'] = unique_count
    stats['duplicates_removed'] = stats['after_standardization'] - stats['after_deduplication']

    print(f"\n✓ Processing complete")
    print(f"  Total input:             {stats['total_input']:,}")
    print(f"  After standardization:   {stats['after_standardization']:,}")
    print(f"  After deduplication:     {stats['after_deduplication']:,}")
    print(f"  Duplicates removed:      {stats['duplicates_removed']:,}")

    # Extract standardizer statistics
    if hasattr(standardizer, 'stats') and hasattr(standardizer.stats, 'stats'):
        standardizer_stats = standardizer.stats.stats

        for stat_name in ['standardized', 'unchanged', 'domain_python',
                          'domain_verilog', 'domain_competitive', 'domain_unknown',
                          'format_changed', 'skipped_no_prompt', 'skipped_invalid_prompt',
                          'skipped_invalid_message', 'skipped_empty_content', 'error']:
            if stat_name in standardizer_stats:
                stats['standardization_stats'][stat_name] = int(standardizer_stats[stat_name].total)

    print(f"\n✓ Saved to: {output_file}")

    # Calculate duration
    duration = time.time() - start_time
    stats['duration_seconds'] = duration
    stats['rows_per_second'] = stats['total_input'] / duration if duration > 0 else 0

    return stats


def generate_report(
    input_file: str,
    output_file: str,
    preset_name: str,
    stats: dict,
) -> str:
    """Generate processing report.

    Args:
        input_file: Input file path or Hub dataset ID
        output_file: Output file path
        preset_name: Standardization preset used
        stats: Statistics dictionary with samples

    Returns:
        Report text
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
{'='*70}
Code Instruction Standardization Report
{'='*70}
Input:     {input_file}
Output:    {output_file}
Preset:    {preset_name}
Timestamp: {timestamp}

{'='*70}
Processing Summary
{'='*70}
Total input samples:              {stats['total_input']:,}
After standardization:            {stats['after_standardization']:,}
  Filtered by standardization:    {stats['filtered_by_standardization']:,}
After deduplication:              {stats['after_deduplication']:,}
  Duplicates removed:             {stats['duplicates_removed']:,}

Final output samples:             {stats['after_deduplication']:,}
Reduction from input:             {stats['total_input'] - stats['after_deduplication']:,} ({100*(stats['total_input'] - stats['after_deduplication'])/stats['total_input']:.1f}%)

{'='*70}
Standardization Statistics
{'='*70}
"""

    standardization_stats = stats.get('standardization_stats', {})
    if standardization_stats:
        report += f"Standardized samples:        {standardization_stats.get('standardized', 0):,}\n"
        report += f"Unchanged samples:           {standardization_stats.get('unchanged', 0):,}\n\n"
        report += f"Domain Detection:\n"
        report += f"{'─'*70}\n"
        report += f"Python:                      {standardization_stats.get('domain_python', 0):,} samples\n"
        report += f"Verilog:                     {standardization_stats.get('domain_verilog', 0):,} samples\n"
        report += f"Competitive:                 {standardization_stats.get('domain_competitive', 0):,} samples\n"
        report += f"Unknown:                     {standardization_stats.get('domain_unknown', 0):,} samples\n\n"
        report += f"Format Changes:\n"
        report += f"{'─'*70}\n"
        report += f"Format changed:              {standardization_stats.get('format_changed', 0):,} samples\n"
        report += f"Skipped (no prompt):         {standardization_stats.get('skipped_no_prompt', 0):,} samples\n"
        report += f"Skipped (invalid prompt):    {standardization_stats.get('skipped_invalid_prompt', 0):,} samples\n"
        report += f"Skipped (invalid message):   {standardization_stats.get('skipped_invalid_message', 0):,} samples\n"
        report += f"Skipped (empty content):     {standardization_stats.get('skipped_empty_content', 0):,} samples\n"
        report += f"Errors:                      {standardization_stats.get('error', 0):,} samples\n"
    else:
        report += "No standardization statistics available\n"

    report += f"\n{'='*70}\n"
    report += f"Deduplication Statistics\n"
    report += f"{'='*70}\n"
    report += f"Unique samples retained:     {stats['after_deduplication']:,}\n"
    report += f"Duplicate samples removed:   {stats['duplicates_removed']:,}\n"
    report += f"Duplicate rate:              {100*stats['duplicates_removed']/stats['after_standardization']:.2f}%\n"

    report += f"\n{'='*70}\n"
    report += f"Performance\n"
    report += f"{'='*70}\n"
    report += f"Processing time:             {stats['duration_seconds']:.1f} seconds ({stats['duration_seconds']/60:.1f} minutes)\n"
    report += f"Processing speed:            {stats['rows_per_second']:.0f} rows/second\n"

    # Add sample examples
    sample_examples = stats.get('sample_examples', [])
    if sample_examples:
        report += f"\n{'='*70}\n"
        report += f"Sample Documents ({len(sample_examples)} collected)\n"
        report += f"{'='*70}\n"

        for i, example in enumerate(sample_examples[:20], 1):  # Show first 20
            report += f"\n{'-'*70}\n"
            report += f"Sample {i} (#{example['index']})\n"
            report += f"{'-'*70}\n"
            report += f"{example['problem']}\n\n"
            report += f"Ground Truth: {example['ground_truth']}\n"

    report += f"\n{'='*70}\n"

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Standardize code instruction formats: standardization + deduplication',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process local parquet file with python-code preset
  python scripts/processing/standardize_instructions.py \\
      --input output/kodcode-original/train.parquet \\
      --output output/kodcode-standardized/train.parquet \\
      --preset python-code

  # Process HuggingFace Hub dataset with verilog-hdl preset
  python scripts/processing/standardize_instructions.py \\
      --input sungyub/codev-r1-verl \\
      --output output/codev-standardized/train.parquet \\
      --preset verilog-hdl

  # Use competitive preset
  python scripts/processing/standardize_instructions.py \\
      --input sungyub/eurus-2-code-verl \\
      --output output/eurus-standardized/train.parquet \\
      --preset competitive
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input parquet file path OR HuggingFace Hub dataset identifier (e.g., sungyub/kodcode-v1-verl)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output parquet file path'
    )

    parser.add_argument(
        '--preset',
        type=str,
        default='python-code',
        choices=['python-code', 'verilog-hdl', 'competitive', 'auto'],
        help='Standardization preset name (default: python-code)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for processing (default: 1000)'
    )

    parser.add_argument(
        '--sample-rate',
        type=int,
        default=1000,
        help='Collect sample every N documents (default: 1000)'
    )

    parser.add_argument(
        '--report-file',
        type=str,
        default=None,
        help='Save report to file (default: <output_dir>/standardization_report.txt)'
    )

    args = parser.parse_args()

    # Validate input
    is_local = is_local_file(args.input)

    if is_local:
        # Only check file existence for local files
        if not Path(args.input).exists():
            print(f"Error: Local file not found: {args.input}")
            return
        print(f"Input type: Local parquet file")
    else:
        print(f"Input type: HuggingFace Hub dataset")
        print(f"Dataset identifier: {args.input}")

    # Process dataset
    stats = process_dataset(
        input_file=args.input,
        output_file=args.output,
        preset_name=args.preset,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
    )

    if not stats:
        print("\n✗ Processing failed")
        return

    # Generate report
    report = generate_report(
        input_file=args.input,
        output_file=args.output,
        preset_name=args.preset,
        stats=stats,
    )

    # Print report
    print(report)

    # Save report
    if args.report_file:
        report_path = Path(args.report_file)
    else:
        output_dir = Path(args.output).parent
        report_path = output_dir / "standardization_report.txt"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"✓ Report saved to: {report_path}")

    # Save stats as JSON
    stats_path = Path(args.output).parent / "standardization_stats.json"
    with open(stats_path, "w") as f:
        # Convert defaultdict to regular dict for JSON
        stats_copy = dict(stats)
        stats_copy['standardization_stats'] = dict(stats['standardization_stats'])
        stats_copy['duplicate_hashes'] = {}  # Too large to save

        json.dump({
            "input": args.input,
            "output": args.output,
            "preset": args.preset,
            "timestamp": datetime.now().isoformat(),
            "stats": stats_copy,
        }, f, indent=2)
    print(f"✓ Stats saved to: {stats_path}")

    print(f"\n{'='*70}")
    print(f"✅ Processing completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
