#!/usr/bin/env python3
"""
Prepare Skywork-OR1-Math-VERL Standalone Dataset for Hub Upload

This script prepares the cleaned Skywork dataset for the standalone Hub repository.
It PRESERVES the model_difficulty field for curriculum learning applications.

Input: Cleaned dataset with model_difficulty
Output: Hub-ready dataset (same schema)
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset


def prepare_standalone(input_file: str, output_dir: str):
    """Prepare standalone dataset (preserves model_difficulty).

    Args:
        input_file: Path to cleaned parquet file
        output_dir: Output directory for Hub upload
    """
    print("=" * 80)
    print("Preparing Skywork-OR1-Math-VERL Standalone Dataset")
    print("=" * 80)
    print(f"Input:  {input_file}")
    print(f"Output: {output_dir}")
    print()

    # Load cleaned dataset
    print("Step 1: Loading cleaned dataset...")
    df = pd.read_parquet(input_file)
    print(f"✓ Loaded {len(df):,} samples")

    # Verify schema
    print("\nStep 2: Verifying schema...")
    sample = df.iloc[0]

    required_fields = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
    for field in required_fields:
        assert field in sample, f"Missing field: {field}"
    print(f"✓ All required fields present: {required_fields}")

    # Verify model_difficulty exists
    assert 'model_difficulty' in sample['extra_info'], "model_difficulty must exist!"
    assert 'index' in sample['extra_info'], "index must exist!"
    print(f"✓ extra_info fields: {list(sample['extra_info'].keys())}")
    print(f"✓ model_difficulty preserved for curriculum learning ✨")

    # Verify model_difficulty structure
    model_diff = sample['extra_info']['model_difficulty']
    expected_models = [
        'DeepSeek-R1-Distill-Qwen-1.5B',
        'DeepSeek-R1-Distill-Qwen-32B',
        'DeepSeek-R1-Distill-Qwen-7B'
    ]
    for model in expected_models:
        assert model in model_diff, f"Missing model in model_difficulty: {model}"
    print(f"✓ All 3 difficulty models present")

    # Create output directory structure
    print("\nStep 3: Creating output directory...")
    output_path = Path(output_dir)
    data_dir = output_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {data_dir}")

    # Save dataset (NO schema changes - simple copy)
    print("\nStep 4: Saving dataset...")
    output_file = data_dir / "train-00000.parquet"

    # Use datasets library to ensure proper format
    dataset = Dataset.from_pandas(df)
    dataset.to_parquet(str(output_file))

    print(f"✓ Saved: {output_file}")
    print(f"  Samples: {len(dataset):,}")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Verify output
    print("\nStep 5: Verifying output...")
    verify_df = pd.read_parquet(output_file)
    verify_sample = verify_df.iloc[0]

    assert 'model_difficulty' in verify_sample['extra_info'], "Verification failed: model_difficulty missing!"
    assert 'index' in verify_sample['extra_info'], "Verification failed: index missing!"
    assert 'original_dataset' not in verify_sample['extra_info'], "Should NOT have original_dataset in standalone!"
    assert 'split' not in verify_sample['extra_info'], "Should NOT have split in standalone!"

    print(f"✓ Verification passed")
    print(f"  extra_info fields: {list(verify_sample['extra_info'].keys())}")
    print(f"  model_difficulty: {verify_sample['extra_info']['model_difficulty']}")

    # Print statistics
    print("\n" + "=" * 80)
    print("Standalone Dataset Preparation Complete!")
    print("=" * 80)
    print(f"✅ Schema: PRESERVED (with model_difficulty)")
    print(f"✅ Samples: {len(verify_df):,}")
    print(f"✅ Ready for upload to: sungyub/skywork-or1-math-verl")
    print("=" * 80)
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare Skywork standalone dataset for Hub upload (preserves model_difficulty)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/hub/prepare_standalone_skywork.py \\
      --input output/skywork-or1-cleaned-maxclean/train.parquet \\
      --output output/hub-upload/skywork-or1-math-verl
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

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    # Prepare dataset
    prepare_standalone(
        input_file=args.input,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
