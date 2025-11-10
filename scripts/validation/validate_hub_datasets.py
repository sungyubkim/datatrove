#!/usr/bin/env python3
"""
Validate standalone and unified datasets before Hub upload.

This script performs comprehensive validation including:
- Schema structure verification
- Field type checking
- Dataset loading tests
- Sample quality inspection
"""

import sys
from pathlib import Path
from datasets import load_dataset
import pyarrow.parquet as pq
import pandas as pd


def validate_schema(df, dataset_name, expected_extra_info_fields):
    """Validate dataset schema"""
    print(f"\n{'='*70}")
    print(f"Schema Validation: {dataset_name}")
    print(f"{'='*70}")

    # Check required fields
    required_fields = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    for field in required_fields:
        if field in df.columns:
            print(f"✓ Field '{field}' exists")
        else:
            print(f"✗ Missing field '{field}'")
            return False

    # Check field types
    print(f"\nField types:")
    print(f"  data_source: {df['data_source'].dtype}")
    print(f"  ability: {df['ability'].dtype}")

    # Check nested structures
    sample = df.iloc[0]

    # Check prompt structure (pandas may store lists as objects)
    try:
        prompt_item = sample['prompt'][0]
        if isinstance(prompt_item, dict) and 'role' in prompt_item and 'content' in prompt_item:
            print(f"✓ prompt structure valid (list of dicts with role/content)")
        else:
            print(f"✗ prompt structure invalid: {type(prompt_item)}")
            return False
    except (TypeError, IndexError, KeyError) as e:
        print(f"✗ prompt structure error: {e}")
        return False

    # Check reward_model structure
    if isinstance(sample['reward_model'], dict):
        if 'style' in sample['reward_model'] and 'ground_truth' in sample['reward_model']:
            print(f"✓ reward_model structure valid (dict with style/ground_truth)")
        else:
            print(f"✗ reward_model missing required fields")
            return False
    else:
        print(f"✗ reward_model is not a dict")
        return False

    # Check extra_info structure
    if isinstance(sample['extra_info'], dict):
        extra_info_keys = set(sample['extra_info'].keys())
        expected_keys = set(expected_extra_info_fields)

        print(f"\nextra_info validation:")
        print(f"  Expected fields: {expected_extra_info_fields}")
        print(f"  Actual fields: {list(extra_info_keys)}")

        if extra_info_keys == expected_keys:
            print(f"✓ extra_info structure valid")
        else:
            missing = expected_keys - extra_info_keys
            extra = extra_info_keys - expected_keys
            if missing:
                print(f"✗ Missing fields: {missing}")
            if extra:
                print(f"✗ Extra fields: {extra}")
            return False
    else:
        print(f"✗ extra_info is not a dict")
        return False

    return True


def inspect_samples(df, dataset_name, num_samples=5):
    """Inspect sample data quality"""
    print(f"\n{'='*70}")
    print(f"Sample Inspection: {dataset_name}")
    print(f"{'='*70}")

    for i in range(min(num_samples, len(df))):
        sample = df.iloc[i]
        print(f"\n--- Sample {i+1} ---")
        print(f"data_source: {sample['data_source']}")
        print(f"ability: {sample['ability']}")
        print(f"prompt[0]['role']: {sample['prompt'][0]['role']}")
        print(f"prompt[0]['content'][:100]: {sample['prompt'][0]['content'][:100]}...")
        print(f"reward_model['style']: {sample['reward_model']['style']}")
        print(f"reward_model['ground_truth']: {sample['reward_model']['ground_truth']}")
        print(f"extra_info: {sample['extra_info']}")


def test_dataset_loading(file_path, dataset_name):
    """Test loading dataset with datasets library"""
    print(f"\n{'='*70}")
    print(f"Dataset Loading Test: {dataset_name}")
    print(f"{'='*70}")

    try:
        # Load with datasets library
        dataset = load_dataset("parquet", data_files=str(file_path), split="train")
        print(f"✓ Successfully loaded with datasets library")
        print(f"  Number of rows: {len(dataset):,}")
        print(f"  Features: {list(dataset.features.keys())}")

        # Test iteration
        first_sample = dataset[0]
        print(f"✓ Successfully accessed first sample")

        return True
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False


def main():
    print("=" * 70)
    print("Dataset Validation for Hub Upload")
    print("=" * 70)

    # Paths
    standalone_path = Path("output/deepscaler-cleaned-v2/train.parquet")
    unified_path = Path("output/hub-upload/deepscaler-unified/train.parquet")

    results = {}

    # Validate standalone dataset
    if standalone_path.exists():
        print(f"\n📄 Validating standalone dataset: {standalone_path}")
        table = pq.read_table(standalone_path)
        df_standalone = table.to_pandas()

        print(f"\nDataset shape: {df_standalone.shape}")
        print(f"Columns: {list(df_standalone.columns)}")

        # Expected schema for standalone
        expected_standalone_fields = ["index", "split"]

        results['standalone_schema'] = validate_schema(
            df_standalone,
            "Standalone (sungyub/deepscaler-preview-verl)",
            expected_standalone_fields
        )

        inspect_samples(df_standalone, "Standalone", num_samples=3)

        results['standalone_loading'] = test_dataset_loading(
            standalone_path,
            "Standalone"
        )
    else:
        print(f"✗ Standalone dataset not found: {standalone_path}")
        results['standalone_schema'] = False
        results['standalone_loading'] = False

    # Validate unified dataset
    if unified_path.exists():
        print(f"\n📄 Validating unified dataset: {unified_path}")
        table = pq.read_table(unified_path)
        df_unified = table.to_pandas()

        print(f"\nDataset shape: {df_unified.shape}")
        print(f"Columns: {list(df_unified.columns)}")

        # Expected schema for unified (with original_dataset)
        expected_unified_fields = ["index", "split", "original_dataset"]

        results['unified_schema'] = validate_schema(
            df_unified,
            "Unified (sungyub/math-verl-unified - deepscaler split)",
            expected_unified_fields
        )

        inspect_samples(df_unified, "Unified", num_samples=3)

        results['unified_loading'] = test_dataset_loading(
            unified_path,
            "Unified"
        )
    else:
        print(f"✗ Unified dataset not found: {unified_path}")
        results['unified_schema'] = False
        results['unified_loading'] = False

    # Summary
    print(f"\n{'='*70}")
    print("Validation Summary")
    print(f"{'='*70}")

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n✅ All validations passed! Datasets are ready for Hub upload.")
        return 0
    else:
        print(f"\n❌ Some validations failed. Please fix issues before uploading.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
