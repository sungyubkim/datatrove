#!/usr/bin/env python3
"""
Consolidated dataset validation script for HuggingFace Hub uploads.

This script consolidates validation logic for multiple datasets, replacing:
- validate_deepmath_upload.py
- validate_skywork_upload.py
- validate_unified_upload.py
- validate_before_upload.py

Usage:
    # Validate DeepMath standalone
    python scripts/hub/validate_upload.py --dataset deepmath --mode standalone

    # Validate Skywork standalone
    python scripts/hub/validate_upload.py --dataset skywork --mode standalone

    # Validate unified dataset with DAPO
    python scripts/hub/validate_upload.py --dataset dapo --mode unified

    # Validate unified dataset with DeepMath
    python scripts/hub/validate_upload.py --dataset deepmath --mode unified

    # Custom directory
    python scripts/hub/validate_upload.py --dataset deepmath --mode standalone \
        --upload-dir custom/path
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pyarrow.parquet as pq
import yaml
from datasets import load_dataset


# Dataset-specific configuration
DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "deepmath": {
        "standalone": {
            "expected_samples": 101844,
            "data_source": "deepmath-103k",
            "ability": "math",
            "default_upload_dir": "output/hub-upload/deepmath-103k-verl",
            "parquet_pattern": "data/train-*.parquet",
            "repo_name": "sungyub/deepmath-103k-verl",
            "extra_info_fields": ["index"],  # standalone: no original_dataset
        },
        "unified": {
            "expected_samples": 101844,
            "data_source": "deepmath-103k",
            "ability": "math",
            "original_dataset": "deepmath-103k",
            "default_upload_dir": "output/hub-upload/math-verl-unified",
            "parquet_pattern": "data/deepmath_103k_verl.parquet",
            "repo_name": "sungyub/math-verl-unified",
            "extra_info_fields": ["index", "original_dataset", "split"],  # unified: 3-field
        },
    },
    "skywork": {
        "standalone": {
            "expected_samples": None,  # Variable count
            "data_source": "skywork-or1-math-verl",
            "ability": "math",
            "default_upload_dir": "output/hub-upload/skywork-or1-math-verl",
            "parquet_pattern": "data/train-*.parquet",
            "repo_name": "sungyub/skywork-or1-math-verl",
            "extra_info_fields": ["index", "model_difficulty"],  # standalone: has model_difficulty
            "has_model_difficulty": True,
        },
        "unified": {
            "expected_samples": None,  # Variable count
            "data_source": "skywork-or1-math-verl",
            "ability": "math",
            "original_dataset": "skywork-or1-math-verl",
            "default_upload_dir": "output/hub-upload/math-verl-unified",
            "parquet_pattern": "skywork_or1_math_verl.parquet",
            "repo_name": "sungyub/math-verl-unified",
            "extra_info_fields": ["index", "original_dataset", "split"],  # unified: 3-field
            "no_model_difficulty": True,  # Should NOT have model_difficulty in unified
        },
    },
    "dapo": {
        "standalone": {
            "expected_samples": 17186,
            "data_source": "DAPO-Math-17K",
            "ability": "math",
            "default_upload_dir": "output/hub-upload/standalone",
            "parquet_pattern": "*.parquet",
            "extra_info_fields": ["split", "index"],  # standalone: no original_dataset
        },
        "unified": {
            "expected_samples": 17186,
            "data_source": "DAPO-Math-17K",
            "ability": "math",
            "original_dataset": "dapo-math-17k-verl",
            "default_upload_dir": "output/hub-upload/unified",
            "parquet_pattern": "*.parquet",
            "repo_name": "sungyub/math-verl-unified",
            "extra_info_fields": ["index", "original_dataset", "split"],  # unified: 3-field
        },
    },
}


def validate_schema(table, config: Dict[str, Any], mode: str) -> bool:
    """Validate schema matches expected format.

    Args:
        table: PyArrow table
        config: Dataset-specific configuration
        mode: "standalone" or "unified"

    Returns:
        True if validation passed
    """
    print("=" * 70)
    print("SCHEMA VALIDATION")
    print("=" * 70)

    # Check sample count (if expected)
    actual_count = len(table)
    print(f"✓ Sample count: {actual_count:,}")
    expected_samples = config.get("expected_samples")
    if expected_samples is not None and actual_count != expected_samples:
        print(f"❌ ERROR: Expected {expected_samples:,} samples, got {actual_count:,}")
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

        expected_extra_fields = config.get("extra_info_fields", [])

        # Check for required fields
        for field in expected_extra_fields:
            if field not in extra_info_fields:
                print(f"❌ ERROR: Missing field in extra_info: {field}")
                print(f"   Required: {expected_extra_fields}")
                return False

        # Mode-specific checks
        if mode == "unified":
            if "original_dataset" not in extra_info_fields:
                print(f"❌ ERROR: Missing 'original_dataset' in extra_info (required for unified)")
                return False
        elif mode == "standalone":
            if "original_dataset" in extra_info_fields:
                print(f"⚠️  WARNING: Found 'original_dataset' in standalone dataset")

        # Check for unwanted fields (skywork-specific)
        if config.get("no_model_difficulty") and "model_difficulty" in extra_info_fields:
            print(f"❌ ERROR: Unified dataset should NOT have model_difficulty")
            return False

        if config.get("has_model_difficulty") and "model_difficulty" not in extra_info_fields:
            print(f"❌ ERROR: Standalone Skywork MUST have model_difficulty")
            return False

    print("✓ Schema validation passed\n")
    return True


def validate_data_quality(table, config: Dict[str, Any]) -> bool:
    """Validate data quality and content.

    Args:
        table: PyArrow table
        config: Dataset-specific configuration

    Returns:
        True if validation passed
    """
    print("=" * 70)
    print("DATA QUALITY VALIDATION")
    print("=" * 70)

    # Convert to pandas for easier analysis
    df = table.to_pandas()

    # Check data_source consistency
    expected_source = config["data_source"]
    unique_sources = df['data_source'].unique()
    print(f"✓ Unique data sources: {unique_sources}")
    if len(unique_sources) != 1 or unique_sources[0] != expected_source:
        print(f"❌ ERROR: Expected single data_source '{expected_source}', got {unique_sources}")
        return False

    # Check ability consistency
    expected_ability = config["ability"]
    unique_abilities = df['ability'].unique()
    print(f"✓ Unique abilities: {unique_abilities}")
    if len(unique_abilities) != 1 or unique_abilities[0] != expected_ability:
        print(f"❌ ERROR: Expected single ability '{expected_ability}', got {unique_abilities}")
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
        print(f"⚠️  WARNING: Found null values:")
        print(null_counts[null_counts > 0])
    else:
        print("✓ No null values found")

    # Check index range in extra_info
    indices = df['extra_info'].apply(lambda x: x.get('index') if isinstance(x, dict) else None)
    min_idx = indices.min()
    max_idx = indices.max()
    print(f"✓ Index range: {min_idx} to {max_idx}")

    # Check original_dataset field (for unified mode)
    if "original_dataset" in config:
        original_datasets = df['extra_info'].apply(
            lambda x: x.get('original_dataset') if isinstance(x, dict) else None
        ).unique()
        print(f"✓ Unique original_dataset values: {original_datasets}")
        expected_orig = config["original_dataset"]
        if len(original_datasets) != 1 or original_datasets[0] != expected_orig:
            print(f"❌ ERROR: Expected original_dataset='{expected_orig}', got {original_datasets}")
            return False

    # Check model_difficulty structure (skywork standalone)
    if config.get("has_model_difficulty"):
        sample_extra = df.iloc[0]['extra_info']
        if 'model_difficulty' in sample_extra:
            model_diff = sample_extra['model_difficulty']
            expected_models = [
                'DeepSeek-R1-Distill-Qwen-1.5B',
                'DeepSeek-R1-Distill-Qwen-32B',
                'DeepSeek-R1-Distill-Qwen-7B'
            ]
            for model in expected_models:
                if model not in model_diff:
                    print(f"❌ ERROR: Missing model in model_difficulty: {model}")
                    return False
            print(f"✓ All 3 difficulty models present in model_difficulty")

    print("✓ Data quality validation passed\n")
    return True


def validate_file_system(upload_dir: Path) -> bool:
    """Check for macOS system files and other artifacts.

    Args:
        upload_dir: Directory to validate

    Returns:
        True if validation passed
    """
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


def validate_loading(parquet_path: Path, config: Dict[str, Any]) -> bool:
    """Test loading with datasets library.

    Args:
        parquet_path: Path to parquet file
        config: Dataset-specific configuration

    Returns:
        True if validation passed
    """
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

        # Show original_dataset if present
        if 'original_dataset' in sample['extra_info']:
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
        import traceback
        traceback.print_exc()
        return False


def validate_readme(readme_path: Path, config: Dict[str, Any]) -> bool:
    """Validate README.md file (optional).

    Args:
        readme_path: Path to README.md
        config: Dataset-specific configuration

    Returns:
        True if validation passed
    """
    print("=" * 70)
    print("README VALIDATION")
    print("=" * 70)

    if not readme_path.exists():
        print(f"⚠️  WARNING: README.md not found at {readme_path}")
        print("   (Skipping README validation)")
        return True

    with open(readme_path, 'r', encoding='utf-8') as f:
        readme = f.read()

    print(f"✓ README.md found ({len(readme):,} characters)")

    # Check for YAML frontmatter
    if not readme.startswith('---'):
        print(f"⚠️  WARNING: README missing YAML frontmatter")
        return True

    # Extract YAML
    try:
        yaml_match = re.match(r'^---\n(.*?)\n---', readme, re.DOTALL)
        if not yaml_match:
            print(f"⚠️  WARNING: Could not extract YAML frontmatter")
            return True

        yaml_content = yaml_match.group(1)
        yaml_data = yaml.safe_load(yaml_content)
        print(f"✓ YAML frontmatter parsed successfully")

        # Basic YAML validation
        basic_fields = ['license', 'task_categories', 'tags']
        for field in basic_fields:
            if field not in yaml_data:
                print(f"⚠️  WARNING: Missing YAML field: {field}")

        print(f"✓ README appears valid")

    except yaml.YAMLError as e:
        print(f"⚠️  WARNING: Failed to parse YAML: {e}")

    print("✓ README validation passed\n")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Consolidated dataset validation for Hub upload',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate DeepMath standalone
  python scripts/hub/validate_upload.py --dataset deepmath --mode standalone

  # Validate Skywork standalone
  python scripts/hub/validate_upload.py --dataset skywork --mode standalone

  # Validate DAPO unified
  python scripts/hub/validate_upload.py --dataset dapo --mode unified

  # Custom directory
  python scripts/hub/validate_upload.py --dataset deepmath --mode standalone \\
      --upload-dir custom/path
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASET_CONFIG.keys()),
        help='Dataset to validate'
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['standalone', 'unified'],
        help='Validation mode (standalone or unified)'
    )

    parser.add_argument(
        '--upload-dir',
        type=str,
        default=None,
        help='Upload directory (default: from config)'
    )

    parser.add_argument(
        '--skip-readme',
        action='store_true',
        help='Skip README validation'
    )

    args = parser.parse_args()

    # Get configuration
    if args.dataset not in DATASET_CONFIG:
        print(f"❌ ERROR: Unknown dataset: {args.dataset}")
        print(f"   Available: {list(DATASET_CONFIG.keys())}")
        sys.exit(1)

    if args.mode not in DATASET_CONFIG[args.dataset]:
        print(f"❌ ERROR: Mode '{args.mode}' not available for dataset '{args.dataset}'")
        sys.exit(1)

    config = DATASET_CONFIG[args.dataset][args.mode]

    # Determine upload directory
    upload_dir = args.upload_dir or config.get("default_upload_dir")
    if not upload_dir:
        print(f"❌ ERROR: No upload directory specified")
        sys.exit(1)

    upload_path = Path(upload_dir)

    # Print header
    print("\n" + "=" * 70)
    dataset_name = config.get("repo_name", f"{args.dataset} ({args.mode})")
    print(f"VALIDATING: {dataset_name}")
    print("=" * 70 + "\n")

    if not upload_path.exists():
        print(f"❌ ERROR: Upload directory not found: {upload_dir}")
        sys.exit(1)

    # Find parquet file
    parquet_pattern = config.get("parquet_pattern", "*.parquet")
    parquet_files = list(upload_path.glob(parquet_pattern))
    if not parquet_files:
        parquet_files = list(upload_path.rglob("*.parquet"))

    if len(parquet_files) == 0:
        print(f"❌ ERROR: No parquet files found in {upload_dir}")
        sys.exit(1)

    parquet_path = parquet_files[0]
    print(f"Found parquet file: {parquet_path}\n")

    # Load parquet table
    table = pq.read_table(parquet_path)

    # Run validation checks
    checks = [
        ("Schema", lambda: validate_schema(table, config, args.mode)),
        ("Data Quality", lambda: validate_data_quality(table, config)),
        ("File System", lambda: validate_file_system(upload_path)),
        ("Datasets Loading", lambda: validate_loading(parquet_path, config)),
    ]

    # Add README check if not skipped
    if not args.skip_readme:
        readme_path = upload_path / "README.md"
        checks.append(("README", lambda: validate_readme(readme_path, config)))

    # Execute checks
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
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for check_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {check_name}")
        if not result:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print(f"✅ Ready to upload: {dataset_name}\n")
        sys.exit(0)
    else:
        print("\n❌ SOME VALIDATIONS FAILED")
        print("⚠️  Please fix issues before uploading\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
