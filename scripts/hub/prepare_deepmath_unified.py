#!/usr/bin/env python3
"""
Prepare DeepMath-103K for Unified Dataset Upload

This script prepares the cleaned deepmath-103k-verl dataset for addition to
the unified math-verl-unified dataset as a new split.

Input: ./output/deepmath-103k-verl-cleaned/train.parquet
Output: ./output/hub-upload/math-verl-unified/data/deepmath_103k_verl.parquet

Split name: deepmath_103k_verl (9th split in unified dataset)

Usage:
    python scripts/hub/prepare_deepmath_unified.py
"""

import shutil
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset


def prepare_unified_dataset(
    input_file: str = "./output/deepmath-103k-verl-cleaned/train.parquet",
    output_dir: str = "./output/hub-upload/math-verl-unified",
    split_name: str = "deepmath_103k_verl",
):
    """Prepare dataset for unified repo upload.

    Args:
        input_file: Path to cleaned parquet file
        output_dir: Output directory for unified hub upload
        split_name: Name of the split in unified dataset
    """
    print(f"\n{'='*70}")
    print(f"Preparing DeepMath-103K for Unified Dataset")
    print(f"{'='*70}")
    print(f"Input:      {input_file}")
    print(f"Output:     {output_dir}")
    print(f"Split name: {split_name}")
    print(f"Target:     sungyub/math-verl-unified")
    print(f"{'='*70}\n")

    # Check input exists
    if not Path(input_file).exists():
        print(f"✗ Input file not found: {input_file}")
        print(f"\nPlease ensure the cleaned dataset exists:")
        print(f"  python scripts/processing/process_local_dataset.py \\")
        print(f"      --input output/deepmath-103k-verl/train.parquet \\")
        print(f"      --output output/deepmath-103k-verl-cleaned/train.parquet \\")
        print(f"      --preset orz-math")
        return False

    # Load and verify dataset
    print("Step 1: Loading and verifying dataset...")
    try:
        dataset = load_dataset("parquet", data_files=input_file, split="train")
        print(f"✓ Loaded {len(dataset):,} samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

    # Verify schema
    print("\nStep 2: Verifying schema for unified compatibility...")
    required_fields = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    for field in required_fields:
        if field not in dataset.features:
            print(f"✗ Missing required field: {field}")
            return False

    # Check extra_info has minimal 3-field format
    extra_info_fields = set(dataset.features["extra_info"].keys())
    minimal_fields = {"index", "original_dataset", "split"}

    if minimal_fields.issubset(extra_info_fields):
        print(f"✓ Schema verified (minimal 3-field extra_info)")
        print(f"  Fields: {sorted(extra_info_fields)}")
    else:
        print(f"✗ extra_info missing required fields for unified dataset")
        print(f"  Required: {minimal_fields}")
        print(f"  Found: {extra_info_fields}")
        return False

    # Verify data_source and original_dataset fields
    sample = dataset[0]
    if sample["data_source"] != "deepmath-103k":
        print(f"✗ Unexpected data_source: {sample['data_source']}")
        return False

    if sample["extra_info"]["original_dataset"] != "deepmath-103k":
        print(f"✗ Unexpected original_dataset: {sample['extra_info']['original_dataset']}")
        print(f"  Note: This field is REQUIRED for unified dataset")
        return False

    print(f"✓ data_source verified: {sample['data_source']}")
    print(f"✓ original_dataset verified: {sample['extra_info']['original_dataset']}")

    # Create output directory
    print("\nStep 3: Creating output directory...")
    output_path = Path(output_dir) / "data"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created {output_path}")

    # Copy to unified directory with correct name
    print("\nStep 4: Copying to unified directory...")
    output_file = output_path / f"{split_name}.parquet"

    try:
        shutil.copy2(input_file, output_file)
        print(f"✓ Copied to {output_file}")
    except Exception as e:
        print(f"✗ Failed to copy file: {e}")
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
        print(f"  Split name: {split_name}")

        # Display sample
        print(f"\nSample data (first row):")
        ds_check = load_dataset("parquet", data_files=str(output_file), split="train")
        first = ds_check[0]
        print(f"  data_source: {first['data_source']}")
        print(f"  problem: {first['prompt'][0]['content'][:100]}...")
        print(f"  ground_truth: {first['reward_model']['ground_truth']}")
        print(f"  original_dataset: {first['extra_info']['original_dataset']}")

    except Exception as e:
        print(f"✗ Failed to verify output: {e}")
        return False

    print(f"\n{'='*70}")
    print(f"✅ Unified dataset preparation completed successfully!")
    print(f"{'='*70}")
    print(f"\nDataset ready to be added as 9th split to sungyub/math-verl-unified")
    print(f"\nNext steps:")
    print(f"1. Update unified README: python scripts/hub/update_unified_with_deepmath.py")
    print(f"2. Validate: python scripts/hub/validate_deepmath_upload.py --unified")
    print(f"3. Upload: python scripts/upload/upload_deepmath_to_hub.py --unified")
    print(f"{'='*70}\n")

    return True


def main():
    """Main entry point."""
    success = prepare_unified_dataset()

    if success:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
