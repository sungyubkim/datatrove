#!/usr/bin/env python3
"""
Prepare Skywork-OR1-Math-VERL for Unified Dataset Upload

This script prepares the cleaned Skywork dataset for the unified Hub repository.
It REMOVES model_difficulty and ADDS original_dataset and split fields.

Schema Transformation:
  BEFORE: extra_info = {index, model_difficulty}
  AFTER:  extra_info = {index, original_dataset, split}

Input: Cleaned dataset with model_difficulty
Output: Unified-compatible dataset (3-field extra_info)
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset


def normalize_extra_info(row: dict) -> dict:
    """Normalize extra_info to unified schema (3 fields only).

    Args:
        row: Dataset row

    Returns:
        Row with normalized extra_info
    """
    # Get current extra_info
    extra_info = row.get('extra_info', {})

    # Create new extra_info with only 3 standard fields
    new_extra_info = {
        'index': extra_info.get('index', 0),
        'original_dataset': 'skywork-or1-math-verl',  # Fixed value
        'split': 'train'  # Fixed value
    }

    # Update row
    row['extra_info'] = new_extra_info
    return row


def prepare_unified(input_file: str, output_dir: str, original_dataset: str = "skywork-or1-math-verl"):
    """Prepare unified dataset (removes model_difficulty, adds original_dataset/split).

    Args:
        input_file: Path to cleaned parquet file
        output_dir: Output directory for Hub upload
        original_dataset: Original dataset name (default: skywork-or1-math-verl)
    """
    print("=" * 80)
    print("Preparing Skywork-OR1-Math-VERL for Unified Dataset")
    print("=" * 80)
    print(f"Input:  {input_file}")
    print(f"Output: {output_dir}")
    print(f"Schema: 3-field extra_info (index, original_dataset, split)")
    print()

    # Load cleaned dataset
    print("Step 1: Loading cleaned dataset...")
    df = pd.read_parquet(input_file)
    print(f"✓ Loaded {len(df):,} samples")

    # Verify input schema
    print("\nStep 2: Verifying input schema...")
    sample = df.iloc[0]

    required_fields = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
    for field in required_fields:
        assert field in sample, f"Missing field: {field}"
    print(f"✓ All required fields present")

    # Check current extra_info
    print(f"  Current extra_info fields: {list(sample['extra_info'].keys())}")
    if 'model_difficulty' in sample['extra_info']:
        print(f"  ⚠️  model_difficulty will be REMOVED for unified schema")

    # Transform schema
    print("\nStep 3: Transforming schema...")
    print(f"  Removing: model_difficulty")
    print(f"  Adding: original_dataset = '{original_dataset}'")
    print(f"  Adding: split = 'train'")

    # Apply normalization
    df_normalized = df.copy()
    for idx in range(len(df_normalized)):
        row_dict = df_normalized.iloc[idx].to_dict()
        normalized_row = normalize_extra_info(row_dict)
        df_normalized.at[idx, 'extra_info'] = normalized_row['extra_info']

    print(f"✓ Schema transformation complete")

    # Verify transformed schema
    verify_sample = df_normalized.iloc[0]
    assert 'model_difficulty' not in verify_sample['extra_info'], "model_difficulty should be removed!"
    assert 'original_dataset' in verify_sample['extra_info'], "original_dataset must be added!"
    assert 'split' in verify_sample['extra_info'], "split must be added!"
    assert len(verify_sample['extra_info']) == 3, f"Must have exactly 3 fields, got {len(verify_sample['extra_info'])}"

    print(f"✓ Transformed extra_info fields: {list(verify_sample['extra_info'].keys())}")
    print(f"✓ Verification passed: exactly 3 fields ✨")

    # Create output directory structure
    print("\nStep 4: Creating output directory...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {output_path}")

    # Save dataset
    print("\nStep 5: Saving dataset...")
    output_file = output_path / f"skywork_or1_math_verl.parquet"

    # Use datasets library to ensure proper format
    dataset = Dataset.from_pandas(df_normalized)
    dataset.to_parquet(str(output_file))

    print(f"✓ Saved: {output_file}")
    print(f"  Samples: {len(dataset):,}")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Final verification
    print("\nStep 6: Final verification...")
    verify_df = pd.read_parquet(output_file)
    verify_sample = verify_df.iloc[0]

    # Check schema
    assert 'model_difficulty' not in verify_sample['extra_info'], "Verification failed: model_difficulty still present!"
    assert verify_sample['extra_info']['original_dataset'] == original_dataset, "original_dataset mismatch!"
    assert verify_sample['extra_info']['split'] == 'train', "split must be 'train'!"
    assert len(verify_sample['extra_info']) == 3, f"Must have exactly 3 fields, got {len(verify_sample['extra_info'])}"

    print(f"✓ Final verification passed")
    print(f"  extra_info: {verify_sample['extra_info']}")

    # Print statistics
    print("\n" + "=" * 80)
    print("Unified Dataset Preparation Complete!")
    print("=" * 80)
    print(f"✅ Schema: TRANSFORMED to 3-field standard")
    print(f"✅ Removed: model_difficulty")
    print(f"✅ Added: original_dataset = '{original_dataset}'")
    print(f"✅ Added: split = 'train'")
    print(f"✅ Samples: {len(verify_df):,}")
    print(f"✅ Ready for upload to: sungyub/math-verl-unified")
    print("=" * 80)
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare Skywork dataset for unified Hub upload (3-field extra_info)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/hub/prepare_unified_skywork.py \\
      --input output/skywork-or1-cleaned-maxclean/train.parquet \\
      --output output/hub-upload/math-verl-unified \\
      --original-dataset "skywork-or1-math-verl"
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input cleaned parquet file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for Hub upload'
    )

    parser.add_argument(
        '--original-dataset',
        type=str,
        default='skywork-or1-math-verl',
        help='Original dataset name (default: skywork-or1-math-verl)'
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    # Prepare dataset
    prepare_unified(
        input_file=args.input,
        output_dir=args.output,
        original_dataset=args.original_dataset,
    )


if __name__ == "__main__":
    main()
