#!/usr/bin/env python3
"""
Validate Skywork Dataset Uploads

Validates both standalone and unified datasets before Hub upload.
Checks schema, data quality, and ensures no macOS artifacts.
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def validate_standalone(data_dir: str) -> bool:
    """Validate standalone dataset (should have model_difficulty).

    Args:
        data_dir: Directory containing data/train-*.parquet

    Returns:
        True if validation passed
    """
    print("=" * 70)
    print("Validating Standalone Dataset")
    print("=" * 70)

    parquet_files = list(Path(data_dir).glob("data/train-*.parquet"))
    if not parquet_files:
        print(f"✗ No parquet files found in {data_dir}/data/")
        return False

    print(f"Found {len(parquet_files)} parquet file(s)")

    # Load dataset
    print("\n📥 Loading dataset...")
    df = pd.read_parquet(parquet_files[0])
    print(f"✓ Loaded {len(df):,} samples")

    # Schema validation
    print("\n📋 Schema Validation:")
    required_fields = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
    for field in required_fields:
        assert field in df.columns, f"Missing field: {field}"
    print(f"✓ All required fields present: {required_fields}")

    # Extra info validation (STANDALONE)
    sample = df.iloc[0]
    extra_info = sample['extra_info']

    print("\n🔍 Extra Info Validation (Standalone):")
    assert 'index' in extra_info, "Missing 'index' in extra_info"
    print("  ✓ index: present")

    assert 'model_difficulty' in extra_info, "STANDALONE MUST have model_difficulty!"
    print("  ✓ model_difficulty: present ✨")

    # Check model_difficulty structure
    model_diff = extra_info['model_difficulty']
    expected_models = [
        'DeepSeek-R1-Distill-Qwen-1.5B',
        'DeepSeek-R1-Distill-Qwen-32B',
        'DeepSeek-R1-Distill-Qwen-7B'
    ]
    for model in expected_models:
        assert model in model_diff, f"Missing model: {model}"
    print(f"  ✓ All 3 difficulty models present")

    # Should NOT have unified fields
    assert 'original_dataset' not in extra_info, "Should NOT have original_dataset in standalone!"
    assert 'split' not in extra_info, "Should NOT have split in standalone!"
    print("  ✓ No unified fields (correct for standalone)")

    # Data quality
    print("\n✅ Data Quality:")
    print(f"  Samples: {len(df):,}")
    print(f"  Unique data_sources: {df['data_source'].nunique()}")

    # Check macOS artifacts
    print("\n🔍 File System Check:")
    parent_dir = Path(data_dir)
    artifacts = []
    for pattern in ['.DS_Store', '._*', '.Spotlight-V100', '.Trashes']:
        artifacts.extend(list(parent_dir.rglob(pattern)))

    if artifacts:
        print(f"  ✗ Found macOS artifacts:")
        for artifact in artifacts:
            print(f"    - {artifact}")
        return False
    print("  ✓ No macOS artifacts")

    print("\n" + "=" * 70)
    print("✅ STANDALONE VALIDATION PASSED")
    print("=" * 70)
    return True


def validate_unified(data_dir: str) -> bool:
    """Validate unified dataset (should have 3-field extra_info).

    Args:
        data_dir: Directory containing skywork_or1_math_verl.parquet

    Returns:
        True if validation passed
    """
    print("\n" + "=" * 70)
    print("Validating Unified Dataset")
    print("=" * 70)

    parquet_file = Path(data_dir) / "skywork_or1_math_verl.parquet"
    if not parquet_file.exists():
        print(f"✗ File not found: {parquet_file}")
        return False

    # Load dataset
    print("\n📥 Loading dataset...")
    df = pd.read_parquet(parquet_file)
    print(f"✓ Loaded {len(df):,} samples")

    # Schema validation
    print("\n📋 Schema Validation:")
    required_fields = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
    for field in required_fields:
        assert field in df.columns, f"Missing field: {field}"
    print(f"✓ All required fields present: {required_fields}")

    # Extra info validation (UNIFIED)
    sample = df.iloc[0]
    extra_info = sample['extra_info']

    print("\n🔍 Extra Info Validation (Unified - 3 fields):")
    assert 'index' in extra_info, "Missing 'index' in extra_info"
    print("  ✓ index: present")

    assert 'original_dataset' in extra_info, "UNIFIED MUST have original_dataset!"
    print(f"  ✓ original_dataset: {extra_info['original_dataset']}")
    assert extra_info['original_dataset'] == 'skywork-or1-math-verl', "Wrong original_dataset value!"

    assert 'split' in extra_info, "UNIFIED MUST have split!"
    print(f"  ✓ split: {extra_info['split']}")
    assert extra_info['split'] == 'train', "Wrong split value!"

    # Should NOT have model_difficulty
    assert 'model_difficulty' not in extra_info, "UNIFIED should NOT have model_difficulty!"
    print("  ✓ model_difficulty: correctly removed ✨")

    # Must be exactly 3 fields
    assert len(extra_info) == 3, f"Must have exactly 3 fields, got {len(extra_info)}"
    print(f"  ✓ Exactly 3 fields (standard schema)")

    # Data quality
    print("\n✅ Data Quality:")
    print(f"  Samples: {len(df):,}")
    print(f"  Unique data_sources: {df['data_source'].nunique()}")

    # Verify all samples have correct fields
    print("\n🔍 Checking all samples...")
    issues = 0
    for idx in range(min(1000, len(df))):  # Check first 1000
        ei = df.iloc[idx]['extra_info']
        if 'model_difficulty' in ei:
            print(f"  ✗ Sample {idx} still has model_difficulty!")
            issues += 1
        if ei.get('original_dataset') != 'skywork-or1-math-verl':
            print(f"  ✗ Sample {idx} has wrong original_dataset!")
            issues += 1

    if issues > 0:
        print(f"  ✗ Found {issues} issues in first 1000 samples")
        return False
    print(f"  ✓ First 1000 samples verified")

    # Check macOS artifacts
    print("\n🔍 File System Check:")
    parent_dir = Path(data_dir)
    artifacts = []
    for pattern in ['.DS_Store', '._*', '.Spotlight-V100', '.Trashes']:
        artifacts.extend(list(parent_dir.rglob(pattern)))

    if artifacts:
        print(f"  ✗ Found macOS artifacts:")
        for artifact in artifacts:
            print(f"    - {artifact}")
        return False
    print("  ✓ No macOS artifacts")

    print("\n" + "=" * 70)
    print("✅ UNIFIED VALIDATION PASSED")
    print("=" * 70)
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate Skywork datasets before Hub upload'
    )

    parser.add_argument(
        '--standalone-dir',
        type=str,
        default='output/hub-upload/skywork-or1-math-verl',
        help='Standalone dataset directory'
    )

    parser.add_argument(
        '--unified-dir',
        type=str,
        default='output/hub-upload/math-verl-unified',
        help='Unified dataset directory'
    )

    parser.add_argument(
        '--skip-standalone',
        action='store_true',
        help='Skip standalone validation'
    )

    parser.add_argument(
        '--skip-unified',
        action='store_true',
        help='Skip unified validation'
    )

    args = parser.parse_args()

    all_passed = True

    # Validate standalone
    if not args.skip_standalone:
        if not validate_standalone(args.standalone_dir):
            all_passed = False

    # Validate unified
    if not args.skip_unified:
        if not validate_unified(args.unified_dir):
            all_passed = False

    # Final summary
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("=" * 70)
        print("✅ Standalone: model_difficulty preserved")
        print("✅ Unified: 3-field standard schema")
        print("✅ Ready for Hub upload")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("=" * 70)
        print("Please fix issues before uploading")

    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
