#!/usr/bin/env python3
"""
Prepare DeepMath-103K Standalone Dataset for Hub Upload

This script prepares the cleaned deepmath-103k-verl dataset for upload to HuggingFace Hub
as a standalone dataset, with minimal 3-field extra_info schema.

Input: ./output/deepmath-103k-verl-cleaned/train.parquet
Output: ./output/hub-upload/deepmath-103k-verl/data/train-00000.parquet

Schema: Minimal 3-field extra_info
- index, original_dataset, split

Usage:
    python scripts/hub/prepare_deepmath_standalone.py
"""

import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset


# macOS system files to exclude
MACOS_SYSTEM_FILES = [
    ".DS_Store",
    "._.DS_Store",
    "._*",
    ".Spotlight-V100",
    ".Trashes",
    ".fseventsd",
]


def clean_macos_files(directory: Path):
    """Remove macOS system files from directory.

    Args:
        directory: Directory to clean
    """
    removed = []
    for pattern in MACOS_SYSTEM_FILES:
        if "*" in pattern:
            # Handle wildcard patterns
            for file in directory.rglob(pattern.replace("*", "")):
                if file.is_file():
                    file.unlink()
                    removed.append(file.name)
        else:
            # Handle exact matches
            for file in directory.rglob(pattern):
                if file.is_file():
                    file.unlink()
                    removed.append(file.name)

    if removed:
        print(f"  Removed {len(removed)} macOS system files: {', '.join(set(removed))}")
    else:
        print(f"  No macOS system files found")


def prepare_standalone_dataset(
    input_file: str = "./output/deepmath-103k-verl-cleaned/train.parquet",
    output_dir: str = "./output/hub-upload/deepmath-103k-verl",
):
    """Prepare standalone dataset for Hub upload.

    Args:
        input_file: Path to cleaned parquet file
        output_dir: Output directory for Hub upload
    """
    print(f"\n{'='*70}")
    print(f"Preparing DeepMath-103K Standalone Dataset for Hub Upload")
    print(f"{'='*70}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_dir}")
    print(f"Target: sungyub/deepmath-103k-verl")
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

    # Check extra_info structure (minimal 3-field format)
    extra_info_fields = list(dataset.features["extra_info"].keys())
    expected_fields = ["index", "original_dataset", "split"]

    print(f"  extra_info fields: {extra_info_fields}")
    if set(extra_info_fields) != set(expected_fields):
        print(f"⚠️  Warning: extra_info structure differs from minimal format")
        print(f"  Expected: {expected_fields}")
        print(f"  Found: {extra_info_fields}")
        print(f"  Continuing anyway...")
    else:
        print(f"✓ Schema verified (minimal 3-field extra_info)")

    # Verify data_source
    sample = dataset[0]
    if sample["data_source"] != "deepmath-103k":
        print(f"✗ Unexpected data_source: {sample['data_source']}")
        return False
    print(f"✓ data_source verified: deepmath-103k")

    # Create output directory
    print("\nStep 3: Creating output directory...")
    output_path = Path(output_dir) / "data"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created {output_path}")

    # Clean macOS system files
    print("\nStep 4: Cleaning macOS system files...")
    clean_macos_files(Path(output_dir))

    # Write to parquet
    print("\nStep 5: Writing to parquet...")
    output_file = output_path / "train-00000.parquet"

    try:
        dataset.to_parquet(output_file)
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

        # Display sample
        print(f"\nSample data (first row):")
        ds_check = load_dataset("parquet", data_files=str(output_file), split="train")
        first = ds_check[0]
        print(f"  data_source: {first['data_source']}")
        print(f"  prompt: {first['prompt'][0]['content'][:100]}...")
        print(f"  ground_truth: {first['reward_model']['ground_truth']}")
        print(f"  extra_info: {first['extra_info']}")

    except Exception as e:
        print(f"✗ Failed to verify output: {e}")
        return False

    print(f"\n{'='*70}")
    print(f"✅ Standalone dataset preparation completed successfully!")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"1. Generate README: python scripts/hub/generate_deepmath_readme.py")
    print(f"2. Validate: python scripts/hub/validate_deepmath_upload.py")
    print(f"3. Upload: python scripts/upload/upload_deepmath_to_hub.py --individual")
    print(f"{'='*70}\n")

    return True


def main():
    """Main entry point."""
    success = prepare_standalone_dataset()

    if success:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
