#!/usr/bin/env python3
"""
Validate DeepMath-103K dataset before HuggingFace Hub upload.

This script performs comprehensive validation for both standalone and unified uploads:
1. Schema integrity check
2. Sample count verification
3. Data quality checks
4. File system checks (no macOS artifacts)
5. Test loading with datasets library
6. README validation

Usage:
    # Validate standalone repo
    python scripts/hub/validate_deepmath_upload.py --individual

    # Validate unified repo addition
    python scripts/hub/validate_deepmath_upload.py --unified

    # Validate both
    python scripts/hub/validate_deepmath_upload.py --both
"""

import argparse
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from datasets import load_dataset


EXPECTED_SAMPLES = 101844


def validate_schema(table, mode: str = "individual"):
    """Validate schema matches expected format.

    Args:
        table: PyArrow table
        mode: "individual" or "unified"
    """
    print("=" * 70)
    print("SCHEMA VALIDATION")
    print("=" * 70)

    # Check sample count
    actual_count = len(table)
    print(f"✓ Sample count: {actual_count:,}")
    if actual_count != EXPECTED_SAMPLES:
        print(f"❌ ERROR: Expected {EXPECTED_SAMPLES:,} samples, got {actual_count:,}")
        return False

    # Check required fields
    required_fields = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
    schema_fields = [field.name for field in table.schema]

    print(f"✓ Schema fields: {schema_fields}")

    for field in required_fields:
        if field not in schema_fields:
            print(f"❌ ERROR: Missing required field: {field}")
            return False

    # Check extra_info structure
    extra_info_type = table.schema.field('extra_info').type
    if hasattr(extra_info_type, '__iter__'):
        extra_info_fields = [f.name for f in extra_info_type]
        print(f"✓ extra_info fields: {extra_info_fields}")

        # Both modes should have minimal 3-field format
        required_extra_fields = ['index', 'original_dataset', 'split']
        for field in required_extra_fields:
            if field not in extra_info_fields:
                print(f"❌ ERROR: Missing field in extra_info: {field}")
                print(f"   Required: {required_extra_fields}")
                return False

        # Check no extra fields beyond the 3 required
        if set(extra_info_fields) != set(required_extra_fields):
            print(f"⚠️  WARNING: extra_info has unexpected fields")
            print(f"   Expected: {required_extra_fields}")
            print(f"   Found: {extra_info_fields}")

    print("✓ Schema validation passed\n")
    return True


def validate_data_quality(table):
    """Validate data quality and content."""
    print("=" * 70)
    print("DATA QUALITY VALIDATION")
    print("=" * 70)

    # Convert to pandas for easier analysis
    df = table.to_pandas()

    # Check data_source consistency
    unique_sources = df['data_source'].unique()
    print(f"✓ Unique data sources: {unique_sources}")
    if len(unique_sources) != 1 or unique_sources[0] != "deepmath-103k":
        print(f"❌ ERROR: Expected single data_source 'deepmath-103k', got {unique_sources}")
        return False

    # Check ability consistency
    unique_abilities = df['ability'].unique()
    print(f"✓ Unique abilities: {unique_abilities}")
    if len(unique_abilities) != 1 or unique_abilities[0] != "math":
        print(f"❌ ERROR: Expected single ability 'math', got {unique_abilities}")
        return False

    # Check prompt structure
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

    # Check original_dataset field
    original_datasets = df['extra_info'].apply(
        lambda x: x.get('original_dataset') if isinstance(x, dict) else None
    ).unique()
    print(f"✓ Unique original_dataset values: {original_datasets}")
    if len(original_datasets) != 1 or original_datasets[0] != "deepmath-103k":
        print(f"❌ ERROR: Expected original_dataset='deepmath-103k', got {original_datasets}")
        return False

    print("✓ Data quality validation passed\n")
    return True


def validate_file_system(upload_dir: Path):
    """Check for macOS system files and other artifacts."""
    print("=" * 70)
    print("FILE SYSTEM VALIDATION")
    print("=" * 70)

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
        if '*' in pattern:
            # Handle wildcard patterns
            matches = [f for f in upload_dir.rglob('*') if f.name.startswith(pattern.replace('*', ''))]
        else:
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
        print(f"   - {rel_path} ({file_size / (1024**2):.2f} MB)")

    print("✓ File system validation passed\n")
    return True


def validate_loading(parquet_path: Path):
    """Test loading with datasets library."""
    print("=" * 70)
    print("DATASETS LIBRARY LOADING TEST")
    print("=" * 70)

    try:
        # Test loading parquet directly
        dataset = load_dataset(
            'parquet',
            data_files=str(parquet_path),
            split='train'
        )

        print(f"✓ Successfully loaded dataset")
        print(f"✓ Dataset length: {len(dataset):,}")
        print(f"✓ Dataset features: {list(dataset.features.keys())}")

        # Test accessing a few samples
        sample = dataset[0]
        print(f"✓ Sample access works")
        print(f"   - data_source: {sample['data_source']}")
        print(f"   - ability: {sample['ability']}")
        print(f"   - prompt messages: {len(sample['prompt'])}")
        print(f"   - has ground_truth: {'ground_truth' in sample['reward_model']}")
        print(f"   - original_dataset: {sample['extra_info']['original_dataset']}")

        # Test iteration
        count = 0
        for example in dataset.select(range(min(10, len(dataset)))):
            count += 1
        print(f"✓ Iteration works (tested {count} samples)")

        print("✓ Loading test passed\n")
        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset: {e}")
        return False


def validate_readme(readme_path: Path, mode: str = "individual"):
    """Validate README.md file.

    Args:
        readme_path: Path to README.md
        mode: "individual" or "unified"
    """
    print("=" * 70)
    print("README VALIDATION")
    print("=" * 70)

    if not readme_path.exists():
        print(f"❌ ERROR: README.md not found at {readme_path}")
        return False

    with open(readme_path, 'r', encoding='utf-8') as f:
        readme = f.read()

    print(f"✓ README.md found ({len(readme):,} characters)")

    # Check for YAML frontmatter
    if not readme.startswith('---'):
        print(f"❌ ERROR: README missing YAML frontmatter")
        return False

    # Extract YAML
    try:
        yaml_match = re.match(r'^---\n(.*?)\n---', readme, re.DOTALL)
        if not yaml_match:
            print(f"❌ ERROR: Could not extract YAML frontmatter")
            return False

        yaml_content = yaml_match.group(1)
        yaml_data = yaml.safe_load(yaml_content)
        print(f"✓ YAML frontmatter parsed successfully")

        # Validate YAML fields
        required_yaml_fields = ['license', 'task_categories', 'tags', 'size_categories',
                                'language', 'pretty_name', 'dataset_info', 'configs']
        for field in required_yaml_fields:
            if field not in yaml_data:
                print(f"❌ ERROR: Missing YAML field: {field}")
                return False

        print(f"✓ All required YAML fields present")

        # Check license
        if mode == "individual":
            if yaml_data.get('license') != 'mit':
                print(f"⚠️  WARNING: Expected license 'mit', got '{yaml_data.get('license')}'")

        # Check dataset_info
        if 'dataset_info' in yaml_data:
            dataset_info = yaml_data['dataset_info']
            if 'splits' in dataset_info:
                splits = dataset_info['splits']
                if len(splits) > 0:
                    train_split = splits[0]
                    if train_split.get('num_examples') != EXPECTED_SAMPLES:
                        print(f"⚠️  WARNING: num_examples mismatch")
                        print(f"   YAML: {train_split.get('num_examples'):,}")
                        print(f"   Expected: {EXPECTED_SAMPLES:,}")

    except yaml.YAMLError as e:
        print(f"❌ ERROR: Failed to parse YAML: {e}")
        return False

    # Check for required sections
    required_sections = ['Dataset Summary', 'Schema', 'Usage']
    for section in required_sections:
        if section not in readme:
            print(f"⚠️  WARNING: Missing section: {section}")

    # Check for citation
    if 'Citation' not in readme and 'citation' not in readme.lower():
        print(f"⚠️  WARNING: No citation section found")

    print("✓ README validation passed\n")
    return True


def validate_individual():
    """Validate standalone repo upload."""
    print("\n" + "=" * 70)
    print("VALIDATING INDIVIDUAL REPO: sungyub/deepmath-103k-verl")
    print("=" * 70 + "\n")

    upload_dir = Path("./output/hub-upload/deepmath-103k-verl")
    parquet_file = upload_dir / "data" / "train-00000.parquet"
    readme_file = upload_dir / "README.md"

    if not upload_dir.exists():
        print(f"❌ ERROR: Upload directory not found: {upload_dir}")
        print(f"\nPlease run preparation scripts first:")
        print(f"  python scripts/hub/prepare_deepmath_standalone.py")
        print(f"  python scripts/hub/generate_deepmath_readme.py")
        return False

    if not parquet_file.exists():
        print(f"❌ ERROR: Parquet file not found: {parquet_file}")
        return False

    # Run validations
    table = pq.read_table(parquet_file)

    all_passed = True
    all_passed &= validate_schema(table, mode="individual")
    all_passed &= validate_data_quality(table)
    all_passed &= validate_file_system(upload_dir)
    all_passed &= validate_loading(parquet_file)
    all_passed &= validate_readme(readme_file, mode="individual")

    return all_passed


def validate_unified():
    """Validate unified repo addition."""
    print("\n" + "=" * 70)
    print("VALIDATING UNIFIED REPO: sungyub/math-verl-unified")
    print("=" * 70 + "\n")

    upload_dir = Path("./output/hub-upload/math-verl-unified")
    parquet_file = upload_dir / "data" / "deepmath_103k_verl.parquet"
    readme_file = upload_dir / "README.md"

    if not upload_dir.exists():
        print(f"❌ ERROR: Upload directory not found: {upload_dir}")
        print(f"\nPlease run preparation scripts first:")
        print(f"  python scripts/hub/prepare_deepmath_unified.py")
        print(f"  python scripts/hub/update_unified_with_deepmath.py")
        return False

    if not parquet_file.exists():
        print(f"❌ ERROR: Parquet file not found: {parquet_file}")
        return False

    # Run validations
    table = pq.read_table(parquet_file)

    all_passed = True
    all_passed &= validate_schema(table, mode="unified")
    all_passed &= validate_data_quality(table)
    all_passed &= validate_file_system(upload_dir)
    all_passed &= validate_loading(parquet_file)
    all_passed &= validate_readme(readme_file, mode="unified")

    return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate DeepMath-103K dataset before Hub upload'
    )
    parser.add_argument(
        '--individual',
        action='store_true',
        help='Validate standalone repo (sungyub/deepmath-103k-verl)'
    )
    parser.add_argument(
        '--unified',
        action='store_true',
        help='Validate unified repo addition (sungyub/math-verl-unified)'
    )
    parser.add_argument(
        '--both',
        action='store_true',
        help='Validate both individual and unified'
    )

    args = parser.parse_args()

    # Default to both if no args
    if not (args.individual or args.unified or args.both):
        args.both = True

    all_passed = True

    if args.both or args.individual:
        passed = validate_individual()
        all_passed &= passed
        if passed:
            print(f"\n{'='*70}")
            print(f"✅ INDIVIDUAL REPO VALIDATION PASSED")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print(f"❌ INDIVIDUAL REPO VALIDATION FAILED")
            print(f"{'='*70}\n")

    if args.both or args.unified:
        passed = validate_unified()
        all_passed &= passed
        if passed:
            print(f"\n{'='*70}")
            print(f"✅ UNIFIED REPO VALIDATION PASSED")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print(f"❌ UNIFIED REPO VALIDATION FAILED")
            print(f"{'='*70}\n")

    if all_passed:
        print(f"\n🎉 ALL VALIDATIONS PASSED!")
        print(f"\nReady to upload:")
        if args.both or args.individual:
            print(f"  python scripts/upload/upload_deepmath_to_hub.py --individual")
        if args.both or args.unified:
            print(f"  python scripts/upload/upload_deepmath_to_hub.py --unified")
        print()
        sys.exit(0)
    else:
        print(f"\n❌ VALIDATION FAILED - Please fix errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
