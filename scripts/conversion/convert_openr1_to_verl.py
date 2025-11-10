#!/usr/bin/env python3
"""
Convert OpenR1-Math-220k to VERL format (WITHOUT correctness filtering).

This script converts the original OpenR1-Math-220k dataset to VERL schema format
while preserving ALL 220k samples (no correctness filtering applied).

Differences from Hub version:
- Hub version: Filters based on correctness_math_verify (220k → 52k)
- This version: Keeps all samples for cleaning/deduplication comparison

Usage:
    python scripts/conversion/convert_openr1_to_verl.py \
        --output output/openr1-raw-verl/train.parquet \
        --batch-size 10000
"""

import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


def get_verl_schema() -> pa.Schema:
    """Define VERL schema for OpenR1-Math dataset.

    Returns:
        PyArrow schema with VERL-compliant fields
    """
    return pa.schema([
        # Data source identifier
        ('data_source', pa.string()),

        # Prompt (list of message dictionaries)
        ('prompt', pa.list_(pa.struct([
            ('role', pa.string()),
            ('content', pa.string()),
        ]))),

        # Ability category
        ('ability', pa.string()),

        # Reward model configuration
        ('reward_model', pa.struct([
            ('style', pa.string()),
            ('ground_truth', pa.string()),
        ])),

        # Additional metadata
        ('extra_info', pa.struct([
            ('split', pa.string()),
            ('index', pa.int64()),
            ('source', pa.string()),
            ('problem_type', pa.string()),
            ('question_type', pa.string()),
        ])),
    ])


def convert_to_verl_row(example: dict, index: int) -> dict:
    """Convert a single example to VERL format.

    Args:
        example: Original OpenR1-Math-220k row
        index: Global index for this sample

    Returns:
        VERL-formatted dictionary
    """
    # Extract problem text and answer
    problem = example.get('problem', '').strip()
    answer = example.get('answer', '').strip()

    # Create VERL row
    return {
        'data_source': 'OpenR1-Math-220k',
        'prompt': [
            {'role': 'user', 'content': problem}
        ],
        'ability': 'math',
        'reward_model': {
            'style': 'rule',
            'ground_truth': answer,
        },
        'extra_info': {
            'split': example.get('split', 'default'),
            'index': index,
            'source': example.get('source', ''),
            'problem_type': example.get('problem_type', ''),
            'question_type': example.get('question_type', ''),
        },
    }


def convert_dataset(
    source_dataset: str = "open-r1/OpenR1-Math-220k",
    output_file: str = "output/openr1-raw-verl/train.parquet",
    batch_size: int = 10000,
    split: str = "all",
) -> dict:
    """Convert OpenR1-Math-220k to VERL format without filtering.

    Args:
        source_dataset: HuggingFace dataset identifier
        output_file: Path to output parquet file
        batch_size: Number of samples to process at once
        split: Dataset split to use ('default', 'all', 'extended')

    Returns:
        Statistics dictionary
    """
    print(f"\n{'='*70}")
    print(f"OpenR1-Math to VERL Conversion (NO filtering)")
    print(f"{'='*70}")
    print(f"Source: {source_dataset}")
    print(f"Split: {split}")
    print(f"Output: {output_file}")
    print(f"Batch size: {batch_size:,}")
    print(f"{'='*70}\n")

    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset in streaming mode
    print(f"Loading dataset in streaming mode (config='{split}')...")
    dataset = load_dataset(source_dataset, split, split='train', streaming=True)

    # Get schema
    schema = get_verl_schema()

    # Process in batches and write
    print(f"\nProcessing samples (batch size: {batch_size:,})...\n")

    total_samples = 0
    start_time = time.time()

    # Use ParquetWriter for streaming writes
    writer = None

    try:
        # Collect batch
        batch_data = []

        for idx, example in enumerate(dataset):
            # Convert to VERL format
            verl_row = convert_to_verl_row(example, idx)
            batch_data.append(verl_row)

            # Process batch when full
            if len(batch_data) >= batch_size:
                # Write batch to parquet
                batch_table = pa.Table.from_pylist(batch_data, schema=schema)

                if writer is None:
                    # First batch: create writer
                    writer = pq.ParquetWriter(output_file, schema=schema, compression='snappy')

                writer.write_table(batch_table)

                total_samples += len(batch_data)
                elapsed = time.time() - start_time
                samples_per_sec = total_samples / elapsed if elapsed > 0 else 0

                print(f"  Processed {total_samples:,} samples... ({samples_per_sec:.0f} samples/sec)", end='\r')

                # Clear batch
                batch_data = []

        # Write remaining batch
        if batch_data:
            batch_table = pa.Table.from_pylist(batch_data, schema=schema)

            if writer is None:
                # Only one batch: use simple write
                pq.write_table(batch_table, output_file, compression='snappy')
            else:
                writer.write_table(batch_table)

            total_samples += len(batch_data)

    finally:
        # Close writer
        if writer is not None:
            writer.close()

    # Final statistics
    elapsed = time.time() - start_time
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0

    print(f"\n\n✅ Conversion complete!")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Processing time: {elapsed:.1f} seconds")
    print(f"  Speed: {samples_per_sec:.0f} samples/second")

    # Calculate file size
    file_size = output_path.stat().st_size
    print(f"  Output file size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  Output location: {output_file}")

    # Verify with datasets library
    print("\n" + "="*70)
    print("Verification")
    print("="*70)

    try:
        from datasets import load_dataset as ld

        verify_ds = ld('parquet', data_files=output_file, split='train')
        print(f"✓ Dataset loadable with datasets library")
        print(f"✓ Sample count: {len(verify_ds):,}")
        print(f"✓ Features: {list(verify_ds.features.keys())}")

        # Show first sample
        sample = verify_ds[0]
        print(f"\n✓ First sample:")
        print(f"  data_source: {sample['data_source']}")
        print(f"  ability: {sample['ability']}")
        print(f"  prompt messages: {len(sample['prompt'])}")
        print(f"  problem preview: {sample['prompt'][0]['content'][:100]}...")
        print(f"  ground_truth preview: {sample['reward_model']['ground_truth'][:100]}...")

    except Exception as e:
        print(f"❌ Verification failed: {e}")

    print("\n" + "="*70 + "\n")

    return {
        'total_samples': total_samples,
        'processing_time': elapsed,
        'samples_per_second': samples_per_sec,
        'output_file': output_file,
        'file_size_bytes': file_size,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert OpenR1-Math-220k to VERL format (no filtering)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python scripts/conversion/convert_openr1_to_verl.py

  # Custom output location
  python scripts/conversion/convert_openr1_to_verl.py \
      --output data/openr1-raw.parquet

  # Larger batch size for faster processing
  python scripts/conversion/convert_openr1_to_verl.py \
      --batch-size 20000
        """
    )

    parser.add_argument(
        '--source',
        type=str,
        default='open-r1/OpenR1-Math-220k',
        help='Source dataset identifier (default: open-r1/OpenR1-Math-220k)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output/openr1-raw-verl/train.parquet',
        help='Output parquet file path (default: output/openr1-raw-verl/train.parquet)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Batch size for processing (default: 10000)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='all',
        choices=['default', 'all', 'extended'],
        help='Dataset split to use (default: all for 220k samples)'
    )

    args = parser.parse_args()

    # Run conversion
    stats = convert_dataset(
        source_dataset=args.source,
        output_file=args.output,
        batch_size=args.batch_size,
        split=args.split,
    )

    # Save stats
    stats_file = Path(args.output).parent / 'conversion_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Statistics saved to: {stats_file}\n")


if __name__ == "__main__":
    main()
