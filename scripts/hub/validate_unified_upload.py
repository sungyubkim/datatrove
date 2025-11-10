#!/usr/bin/env python3
"""
Validate DAPO split for unified dataset upload.

Checks:
1. Schema has original_dataset field (required for unified)
2. Sample count is correct (17,186)
3. Data quality
4. No macOS artifacts
5. Datasets library loading
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset


def validate_unified_schema(table, expected_num_examples: int = 17186):
    """Validate schema for unified dataset."""
    print("=" * 60)
    print("UNIFIED SCHEMA VALIDATION")
    print("=" * 60)

    # Check sample count
    actual_count = len(table)
    print(f"✓ Sample count: {actual_count}")
    if actual_count != expected_num_examples:
        print(f"❌ ERROR: Expected {expected_num_examples} samples, got {actual_count}")
        return False

    # Check required fields
    required_fields = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
    schema_fields = [field.name for field in table.schema]
    print(f"✓ Schema fields: {schema_fields}")

    for field in required_fields:
        if field not in schema_fields:
            print(f"❌ ERROR: Missing required field: {field}")
            return False

    # Check extra_info structure (MUST have original_dataset for unified)
    extra_info_type = table.schema.field('extra_info').type
    if hasattr(extra_info_type, '__iter__'):
        extra_info_fields = [f.name for f in extra_info_type]
        print(f"✓ extra_info fields: {extra_info_fields}")

        if 'original_dataset' not in extra_info_fields:
            print(f"❌ ERROR: Missing 'original_dataset' in extra_info (required for unified)")
            return False

        expected_extra_fields = ['split', 'index', 'original_dataset']
        for field in expected_extra_fields:
            if field not in extra_info_fields:
                print(f"❌ WARNING: Missing recommended field in extra_info: {field}")

    print("✓ Unified schema validation passed\n")
    return True


def validate_unified_data(table):
    """Validate data quality for unified dataset."""
    print("=" * 60)
    print("UNIFIED DATA VALIDATION")
    print("=" * 60)

    df = table.to_pandas()

    # Check data_source
    unique_sources = df['data_source'].unique()
    print(f"✓ Unique data sources: {unique_sources}")
    if len(unique_sources) != 1 or unique_sources[0] != "DAPO-Math-17K":
        print(f"❌ ERROR: Expected 'DAPO-Math-17K', got {unique_sources}")
        return False

    # Check ability
    unique_abilities = df['ability'].unique()
    print(f"✓ Unique abilities: {unique_abilities}")
    if len(unique_abilities) != 1 or unique_abilities[0] != "math":
        print(f"❌ ERROR: Expected 'math', got {unique_abilities}")
        return False

    # Check original_dataset field
    original_datasets = df['extra_info'].apply(
        lambda x: x.get('original_dataset') if isinstance(x, dict) else None
    )
    unique_orig = original_datasets.unique()
    print(f"✓ Unique original_dataset values: {unique_orig}")

    if len(unique_orig) != 1 or unique_orig[0] != "dapo-math-17k-verl":
        print(f"❌ ERROR: Expected 'dapo-math-17k-verl', got {unique_orig}")
        return False

    # Count samples with original_dataset field
    has_orig = original_datasets.notna().sum()
    print(f"✓ Samples with original_dataset: {has_orig}/{len(df)}")

    if has_orig != len(df):
        print(f"❌ ERROR: Not all samples have original_dataset field")
        return False

    # Check prompt structure
    import numpy as np
    sample_prompt = df['prompt'].iloc[0]
    is_valid = (isinstance(sample_prompt, (list, np.ndarray)) and len(sample_prompt) > 0)
    if not is_valid:
        print(f"❌ ERROR: Invalid prompt structure")
        return False
    print(f"✓ Prompt structure: list/array with {len(sample_prompt)} messages")

    # Check reward_model
    sample_reward = df['reward_model'].iloc[0]
    if not isinstance(sample_reward, dict):
        print(f"❌ ERROR: Invalid reward_model structure")
        return False
    if 'style' not in sample_reward or 'ground_truth' not in sample_reward:
        print(f"❌ ERROR: reward_model missing required fields")
        return False
    print(f"✓ Reward model structure: {list(sample_reward.keys())}")

    # Check null values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"❌ WARNING: Found null values:")
        print(null_counts[null_counts > 0])
    else:
        print("✓ No null values found")

    print("✓ Unified data validation passed\n")
    return True


def validate_file_system(upload_dir: Path):
    """Check for macOS artifacts."""
    print("=" * 60)
    print("FILE SYSTEM VALIDATION")
    print("=" * 60)

    macos_artifacts = ['.DS_Store', '._*']
    found_artifacts = []
    for pattern in macos_artifacts:
        matches = list(upload_dir.rglob(pattern))
        if matches:
            found_artifacts.extend(matches)

    if found_artifacts:
        print(f"❌ ERROR: Found macOS system files:")
        for artifact in found_artifacts:
            print(f"   - {artifact}")
        return False
    else:
        print("✓ No macOS system files found")

    # List files
    all_files = list(upload_dir.rglob('*'))
    upload_files = [f for f in all_files if f.is_file()]

    print(f"✓ Files to upload:")
    for f in upload_files:
        rel_path = f.relative_to(upload_dir)
        file_size = f.stat().st_size
        print(f"   - {rel_path} ({file_size:,} bytes)")

    print("✓ File system validation passed\n")
    return True


def validate_loading(parquet_path: Path):
    """Test loading with datasets library."""
    print("=" * 60)
    print("DATASETS LIBRARY LOADING TEST")
    print("=" * 60)

    try:
        dataset = load_dataset('parquet', data_files=str(parquet_path), split='train')

        print(f"✓ Successfully loaded dataset")
        print(f"✓ Dataset length: {len(dataset)}")
        print(f"✓ Dataset features: {list(dataset.features.keys())}")

        sample = dataset[0]
        print(f"✓ Sample access works")
        print(f"   - data_source: {sample['data_source']}")
        print(f"   - ability: {sample['ability']}")
        print(f"   - prompt messages: {len(sample['prompt'])}")
        print(f"   - has ground_truth: {'ground_truth' in sample['reward_model']}")
        print(f"   - original_dataset: {sample['extra_info']['original_dataset']}")

        count = 0
        for example in dataset.select(range(min(10, len(dataset)))):
            count += 1
        print(f"✓ Iteration works (tested {count} samples)")

        print("✓ Datasets library loading test passed\n")
        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(upload_dir: str):
    """Run all validation checks."""
    upload_path = Path(upload_dir)

    if not upload_path.exists():
        print(f"❌ ERROR: Upload directory not found: {upload_dir}")
        return False

    parquet_files = list(upload_path.rglob('*.parquet'))
    if len(parquet_files) == 0:
        print(f"❌ ERROR: No parquet files found")
        return False

    parquet_path = parquet_files[0]
    print(f"Validating: {parquet_path}\n")

    table = pq.read_table(parquet_path)

    checks = [
        ("Unified Schema", lambda: validate_unified_schema(table)),
        ("Unified Data", lambda: validate_unified_data(table)),
        ("File System", lambda: validate_file_system(upload_path)),
        ("Datasets Loading", lambda: validate_loading(parquet_path)),
    ]

    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ ERROR in {check_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((check_name, False))

    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for check_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {check_name}")
        if not result:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All validation checks passed!")
        print("✅ Ready to upload to math-verl-unified\n")
        return True
    else:
        print("\n❌ Some validation checks failed")
        print("⚠️  Please fix issues before uploading\n")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate unified dataset upload")
    parser.add_argument(
        "--upload-dir",
        type=str,
        default="output/hub-upload/unified",
        help="Directory containing files to upload"
    )

    args = parser.parse_args()

    success = main(args.upload_dir)
    sys.exit(0 if success else 1)
