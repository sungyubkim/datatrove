#!/usr/bin/env python3
"""
Prepare Unified Dataset with Minimal Schema

This script transforms the cleaned big-math-rl-verl dataset to conform to the
minimal 3-field extra_info schema used in math-verl-unified dataset.

Transformation:
- REMOVE: source, domain, solve_rate from extra_info
- ADD: original_dataset = "big-math-rl-verl" to extra_info
- KEEP: index, split

Input: ./output/big-math-rl-verl-cleaned/train.parquet
Output: ./output/hub-upload/math-verl-unified/data/big-math-rl-verl.parquet

Schema: Minimal 3-field extra_info
- index, original_dataset, split

Usage:
    python scripts/hub/prepare_unified_minimal_schema.py
"""

from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset


def transform_to_minimal_schema(example):
    """Transform example to minimal schema.

    Removes: source, domain, solve_rate
    Adds: original_dataset
    Keeps: index, split
    """
    # Extract existing fields
    index = example["extra_info"]["index"]
    split = example["extra_info"]["split"]

    # Create new minimal extra_info
    example["extra_info"] = {
        "index": index,
        "original_dataset": "big-math-rl-verl",
        "split": split,
    }

    return example


def prepare_unified_dataset(
    input_file: str = "./output/big-math-rl-verl-cleaned/train.parquet",
    output_dir: str = "./output/hub-upload/math-verl-unified",
):
    """Prepare unified dataset with minimal schema.

    Args:
        input_file: Path to cleaned parquet file
        output_dir: Output directory for Hub upload
    """
    print(f"\n{'='*70}")
    print(f"Preparing Unified Dataset with Minimal Schema")
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
        print(f"  Original extra_info fields: {list(dataset.features['extra_info'].keys())}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

    # Transform schema
    print("\nStep 2: Transforming to minimal schema...")
    print("  Removing: source, domain, solve_rate")
    print("  Adding: original_dataset = 'big-math-rl-verl'")
    print("  Keeping: index, split")

    try:
        transformed_dataset = dataset.map(
            transform_to_minimal_schema,
            desc="Transforming schema",
        )
        print(f"✓ Transformed {len(transformed_dataset):,} samples")
        print(f"  New extra_info fields: {list(transformed_dataset.features['extra_info'].keys())}")
    except Exception as e:
        print(f"✗ Failed to transform dataset: {e}")
        return False

    # Verify transformation
    print("\nStep 3: Verifying transformation...")
    sample = transformed_dataset[0]
    extra_info = sample["extra_info"]

    expected_fields = {"index", "original_dataset", "split"}
    actual_fields = set(extra_info.keys())

    if actual_fields != expected_fields:
        print(f"✗ Schema transformation failed!")
        print(f"  Expected fields: {expected_fields}")
        print(f"  Actual fields: {actual_fields}")
        return False

    if extra_info["original_dataset"] != "big-math-rl-verl":
        print(f"✗ original_dataset value incorrect: {extra_info['original_dataset']}")
        return False

    print(f"✓ Schema verification successful")
    print(f"  Fields: {actual_fields}")
    print(f"  original_dataset: {extra_info['original_dataset']}")

    # Create output directory
    print("\nStep 4: Creating output directory...")
    output_path = Path(output_dir) / "data"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created {output_path}")

    # Write to parquet
    print("\nStep 5: Writing to parquet...")
    output_file = output_path / "big-math-rl-verl.parquet"

    try:
        transformed_dataset.to_parquet(output_file)
        print(f"✓ Saved to {output_file}")
    except Exception as e:
        print(f"✗ Failed to write parquet: {e}")
        return False

    # Verify output
    print("\nStep 6: Verifying output...")
    try:
        table = pq.read_table(output_file)
        num_rows = len(table)
        file_size = output_file.stat().st_size

        print(f"✓ Verification successful")
        print(f"  Rows: {num_rows:,}")
        print(f"  File size: {file_size / (1024**2):.2f} MB")
        print(f"  Schema: {len(table.schema)} columns")

        # Verify a few random samples
        print("\n  Sample verification:")
        import random
        for i in random.sample(range(len(transformed_dataset)), min(3, len(transformed_dataset))):
            sample_extra_info = transformed_dataset[i]["extra_info"]
            print(f"    Sample {i}: {sample_extra_info}")

    except Exception as e:
        print(f"✗ Failed to verify output: {e}")
        return False

    print(f"\n{'='*70}")
    print(f"✅ Unified dataset preparation completed successfully!")
    print(f"{'='*70}")
    print(f"\n📊 Schema Comparison:")
    print(f"  Original: {{split, index, source, domain, solve_rate}} (5 fields)")
    print(f"  Minimal:  {{index, original_dataset, split}} (3 fields)")
    print(f"  Data loss: source, domain, solve_rate removed")
    print(f"  Data gain: original_dataset added")
    print(f"\n")

    return True


def main():
    """Main entry point."""
    success = prepare_unified_dataset()

    if success:
        print("\n📝 Next steps:")
        print("  1. Update README: python scripts/hub/update_unified_readme.py")
        print("  2. Validate: python scripts/hub/validate_before_upload.py --dataset-dir output/hub-upload/math-verl-unified")
        print("  3. Dry run: python scripts/upload/upload_to_hub.py --dataset math-verl-unified --dry-run")
    else:
        print("\n✗ Preparation failed. Please check errors above.")
        exit(1)


if __name__ == "__main__":
    main()
