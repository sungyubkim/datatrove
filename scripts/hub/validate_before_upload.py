#!/usr/bin/env python3
"""
Validate dataset before HuggingFace Hub upload.

This script performs comprehensive validation:
1. Schema integrity check
2. Sample count verification
3. Data quality checks
4. File system checks (no macOS artifacts)
5. Test loading with datasets library
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset


def validate_schema(table, expected_num_examples: int = 17186):
    """Validate schema matches expected standalone format."""
    print("=" * 60)
    print("SCHEMA VALIDATION")
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

    # Check extra_info structure (should NOT have original_dataset for standalone)
    extra_info_type = table.schema.field('extra_info').type
    if hasattr(extra_info_type, '__iter__'):
        extra_info_fields = [f.name for f in extra_info_type]
        print(f"✓ extra_info fields: {extra_info_fields}")

        if 'original_dataset' in extra_info_fields:
            print(f"❌ ERROR: Found 'original_dataset' in extra_info (should not be in standalone)")
            return False

        expected_extra_fields = ['split', 'index']
        for field in expected_extra_fields:
            if field not in extra_info_fields:
                print(f"❌ ERROR: Missing field in extra_info: {field}")
                return False

    print("✓ Schema validation passed\n")
    return True


def validate_data_quality(table):
    """Validate data quality and content."""
    print("=" * 60)
    print("DATA QUALITY VALIDATION")
    print("=" * 60)

    # Convert to pandas for easier analysis
    df = table.to_pandas()

    # Check data_source consistency
    unique_sources = df['data_source'].unique()
    print(f"✓ Unique data sources: {unique_sources}")
    if len(unique_sources) != 1 or unique_sources[0] != "DAPO-Math-17K":
        print(f"❌ ERROR: Expected single data_source 'DAPO-Math-17K', got {unique_sources}")
        return False

    # Check ability consistency
    unique_abilities = df['ability'].unique()
    print(f"✓ Unique abilities: {unique_abilities}")
    if len(unique_abilities) != 1 or unique_abilities[0] != "math":
        print(f"❌ ERROR: Expected single ability 'math', got {unique_abilities}")
        return False

    # Check prompt structure (can be list or numpy array from pandas)
    import numpy as np
    sample_prompt = df['prompt'].iloc[0]
    is_valid = (isinstance(sample_prompt, (list, np.ndarray)) and len(sample_prompt) > 0)
    if not is_valid:
        print(f"❌ ERROR: Invalid prompt structure (type: {type(sample_prompt)})")
        return False
    print(f"✓ Prompt structure: list/array with {len(sample_prompt)} messages")

    # Check reward_model structure
    sample_reward = df['reward_model'].iloc[0]
    if not isinstance(sample_reward, dict):
        print(f"❌ ERROR: Invalid reward_model structure")
        return False
    if 'style' not in sample_reward or 'ground_truth' not in sample_reward:
        print(f"❌ ERROR: reward_model missing required fields")
        return False
    print(f"✓ Reward model structure: {list(sample_reward.keys())}")

    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"❌ WARNING: Found null values:")
        print(null_counts[null_counts > 0])
    else:
        print("✓ No null values found")

    # Check index range in extra_info
    indices = df['extra_info'].apply(lambda x: x.get('index') if isinstance(x, dict) else None)
    min_idx = indices.min()
    max_idx = indices.max()
    print(f"✓ Index range: {min_idx} to {max_idx}")
    if min_idx != 0 or max_idx != len(df) - 1:
        print(f"❌ WARNING: Index range not continuous (expected 0 to {len(df)-1})")

    print("✓ Data quality validation passed\n")
    return True


def validate_file_system(upload_dir: Path):
    """Check for macOS system files and other artifacts."""
    print("=" * 60)
    print("FILE SYSTEM VALIDATION")
    print("=" * 60)

    # Check for macOS artifacts
    macos_artifacts = [
        '.DS_Store',
        '._*',
        '.Spotlight-V100',
        '.Trashes',
        '.fseventsd',
    ]

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

    # List all files that will be uploaded
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
        # Test loading parquet directly
        dataset = load_dataset(
            'parquet',
            data_files=str(parquet_path),
            split='train'
        )

        print(f"✓ Successfully loaded dataset")
        print(f"✓ Dataset length: {len(dataset)}")
        print(f"✓ Dataset features: {list(dataset.features.keys())}")

        # Test accessing a few samples
        sample = dataset[0]
        print(f"✓ Sample access works")
        print(f"   - data_source: {sample['data_source']}")
        print(f"   - ability: {sample['ability']}")
        print(f"   - prompt messages: {len(sample['prompt'])}")
        print(f"   - has ground_truth: {'ground_truth' in sample['reward_model']}")

        # Test iteration
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

    # Find parquet file
    parquet_files = list(upload_path.rglob('*.parquet'))
    if len(parquet_files) == 0:
        print(f"❌ ERROR: No parquet files found in {upload_dir}")
        return False
    elif len(parquet_files) > 1:
        print(f"❌ WARNING: Multiple parquet files found:")
        for f in parquet_files:
            print(f"   - {f}")

    parquet_path = parquet_files[0]
    print(f"Validating: {parquet_path}\n")

    # Load parquet file
    table = pq.read_table(parquet_path)

    # Run validation checks
    checks = [
        ("Schema", lambda: validate_schema(table)),
        ("Data Quality", lambda: validate_data_quality(table)),
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

    # Print summary
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
        print("✅ Ready to upload to HuggingFace Hub\n")
        return True
    else:
        print("\n❌ Some validation checks failed")
        print("⚠️  Please fix issues before uploading\n")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate dataset before Hub upload")
    parser.add_argument(
        "--upload-dir",
        type=str,
        default="output/hub-upload/standalone",
        help="Directory containing files to upload"
    )

    args = parser.parse_args()

    success = main(args.upload_dir)
    sys.exit(0 if success else 1)
