#!/usr/bin/env python3
"""
Prepare Standalone Dataset for Hub Upload

This script prepares the cleaned big-math-rl-verl dataset for upload to HuggingFace Hub
as a standalone dataset, preserving the full 5-field extra_info schema.

Input: ./output/big-math-rl-verl-cleaned/train.parquet
Output: ./output/hub-upload/big-math-rl-verl/data/train-00000.parquet

Schema: Original 5-field extra_info
- split, index, source, domain, solve_rate

Usage:
    python scripts/hub/prepare_standalone_data.py
"""

import os
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset


def prepare_standalone_dataset(
    input_file: str = "./output/big-math-rl-verl-cleaned/train.parquet",
    output_dir: str = "./output/hub-upload/big-math-rl-verl",
):
    """Prepare standalone dataset for Hub upload.

    Args:
        input_file: Path to cleaned parquet file
        output_dir: Output directory for Hub upload
    """
    print(f"\n{'='*70}")
    print(f"Preparing Standalone Dataset for Hub Upload")
    print(f"{'='*70}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Check input exists
    if not Path(input_file).exists():
        print(f"✗ Input file not found: {input_file}")
        return False

    # Load dataset
    print("Step 1: Loading cleaned dataset...")
    try:
        dataset = load_dataset("parquet", data_files=input_file, split="train")
        print(f"✓ Loaded {len(dataset):,} samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

    # Verify schema
    print("\nStep 2: Verifying schema...")
    required_fields = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    for field in required_fields:
        if field not in dataset.features:
            print(f"✗ Missing required field: {field}")
            return False

    # Check extra_info structure
    extra_info_fields = list(dataset.features["extra_info"].keys())
    expected_fields = ["split", "index", "source", "domain", "solve_rate"]

    print(f"  extra_info fields: {extra_info_fields}")
    if set(extra_info_fields) != set(expected_fields):
        print(f"✗ extra_info structure mismatch!")
        print(f"  Expected: {expected_fields}")
        print(f"  Found: {extra_info_fields}")
        return False

    print(f"✓ Schema verified (5-field extra_info)")

    # Create output directory
    print("\nStep 3: Creating output directory...")
    output_path = Path(output_dir) / "data"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created {output_path}")

    # Write to parquet
    print("\nStep 4: Writing to parquet...")
    output_file = output_path / "train-00000.parquet"

    try:
        dataset.to_parquet(output_file)
        print(f"✓ Saved to {output_file}")
    except Exception as e:
        print(f"✗ Failed to write parquet: {e}")
        return False

    # Verify output
    print("\nStep 5: Verifying output...")
    try:
        table = pq.read_table(output_file)
        num_rows = len(table)
        file_size = output_file.stat().st_size

        print(f"✓ Verification successful")
        print(f"  Rows: {num_rows:,}")
        print(f"  File size: {file_size / (1024**2):.2f} MB")
        print(f"  Schema: {len(table.schema)} columns")
    except Exception as e:
        print(f"✗ Failed to verify output: {e}")
        return False

    print(f"\n{'='*70}")
    print(f"✅ Standalone dataset preparation completed successfully!")
    print(f"{'='*70}\n")

    return True


def main():
    """Main entry point."""
    success = prepare_standalone_dataset()

    if success:
        print("\n📝 Next steps:")
        print("  1. Update README: python scripts/hub/update_standalone_readme.py")
        print("  2. Validate: python scripts/hub/validate_before_upload.py --dataset-dir output/hub-upload/big-math-rl-verl")
        print("  3. Dry run: python scripts/upload/upload_to_hub.py --dataset big-math-rl-verl --dry-run")
    else:
        print("\n✗ Preparation failed. Please check errors above.")
        exit(1)


if __name__ == "__main__":
    main()
