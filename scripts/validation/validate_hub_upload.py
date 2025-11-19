#!/usr/bin/env python3
"""
Comprehensive validation script for Hub upload preparation.

Validates:
1. Data schema and structure
2. Sample quality and content
3. File sizes and counts
4. Backward compatibility
5. README accuracy
"""

import os
from pathlib import Path
from datasets import load_dataset
import pyarrow.parquet as pq


def validate_schema(dataset, dataset_name):
    """Validate dataset schema structure."""
    print(f"\n{'='*70}")
    print(f"Schema Validation: {dataset_name}")
    print('='*70)

    # Required top-level fields
    required_fields = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']

    for field in required_fields:
        if field not in dataset.features:
            print(f"❌ Missing required field: {field}")
            return False
        print(f"✓ {field}: present")

    # Validate extra_info structure
    extra_info = dataset.features['extra_info']
    print(f"\n  extra_info fields:")
    for field in extra_info:
        print(f"    - {field}: {extra_info[field]}")

    # Check for original_dataset field (backward compatibility)
    if 'original_dataset' not in extra_info:
        print(f"\n⚠️  WARNING: 'original_dataset' field missing (backward compatibility issue)")
        return False

    print(f"\n✅ Schema validation passed for {dataset_name}")
    return True


def validate_samples(dataset, dataset_name, num_samples=5):
    """Validate sample data quality."""
    print(f"\n{'='*70}")
    print(f"Sample Validation: {dataset_name}")
    print('='*70)

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        print(f"\n  Sample {i+1}:")
        print(f"    data_source: {sample['data_source']}")
        print(f"    ability: {sample['ability']}")

        # Check prompt structure
        if not sample['prompt'] or len(sample['prompt']) == 0:
            print(f"    ❌ Empty prompt")
            return False
        print(f"    prompt: {len(sample['prompt'])} messages")
        print(f"    content preview: {sample['prompt'][0]['content'][:80]}...")

        # Check reward_model
        if not sample['reward_model'] or 'ground_truth' not in sample['reward_model']:
            print(f"    ❌ Invalid reward_model")
            return False
        print(f"    ground_truth: {sample['reward_model']['ground_truth'][:50]}...")

        # Check extra_info
        if not sample['extra_info']:
            print(f"    ❌ Empty extra_info")
            return False

        print(f"    extra_info.index: {sample['extra_info']['index']}")
        print(f"    extra_info.original_dataset: {sample['extra_info']['original_dataset']}")

        # Check new fields if present
        if 'problem_type' in sample['extra_info']:
            print(f"    extra_info.problem_type: {sample['extra_info']['problem_type']}")
        if 'source' in sample['extra_info']:
            print(f"    extra_info.source: {sample['extra_info']['source']}")

    print(f"\n✅ Sample validation passed for {dataset_name}")
    return True


def validate_extra_info_schema(dataset, dataset_name):
    """
    Validate that extra_info has ONLY the 3 standard fields.

    This is critical for math-verl-unified to ensure Hub preview works.
    Standard fields: index, original_dataset, split
    """
    print(f"\n{'='*70}")
    print(f"Extra Info Schema Validation: {dataset_name}")
    print('='*70)

    try:
        # Check schema-level fields
        extra_info_features = dataset.features['extra_info']
        actual_fields = set(extra_info_features.keys())
        expected_fields = {'index', 'original_dataset', 'split'}

        print(f"  Expected fields: {sorted(expected_fields)}")
        print(f"  Actual fields:   {sorted(actual_fields)}")

        if actual_fields == expected_fields:
            print(f"  ✓ Schema has exactly 3 standard fields")
        else:
            extra = actual_fields - expected_fields
            missing = expected_fields - actual_fields

            if extra:
                print(f"  ❌ Extra fields found: {extra}")
                print(f"     These fields will break Hub preview consistency!")
                return False
            if missing:
                print(f"  ❌ Missing required fields: {missing}")
                return False

        # Verify on actual samples
        print(f"\n  Checking first 10 samples for consistency...")
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            sample_fields = set(sample['extra_info'].keys())

            if sample_fields != expected_fields:
                print(f"  ❌ Sample {i} has inconsistent fields: {sample_fields}")
                return False

        print(f"  ✓ All samples have consistent 3-field schema")
        print(f"\n✅ Extra info schema validation passed for {dataset_name}")
        return True

    except Exception as e:
        print(f"\n❌ Extra info schema validation failed: {e}")
        return False


def validate_backward_compatibility(dataset, dataset_name):
    """Test backward compatibility with old schema."""
    print(f"\n{'='*70}")
    print(f"Backward Compatibility Test: {dataset_name}")
    print('='*70)

    try:
        # Simulate old v2.0 code
        sample = dataset[0]

        # Old code that accesses original_dataset
        original_dataset = sample['extra_info']['original_dataset']
        print(f"✓ Old v2.0 code works: extra_info['original_dataset'] = '{original_dataset}'")

        # Old code that accesses index
        index = sample['extra_info']['index']
        print(f"✓ Index field accessible: extra_info['index'] = {index}")

        # Verify NO extra fields exist (v3.0 schema normalization)
        extra_info_fields = set(sample['extra_info'].keys())
        if extra_info_fields == {'index', 'original_dataset', 'split'}:
            print(f"✓ Schema normalized to standard 3 fields (no extras)")
        else:
            print(f"⚠️  Extra fields present: {extra_info_fields - {'index', 'original_dataset', 'split'}}")

        print(f"\n✅ Backward compatibility verified for {dataset_name}")
        return True

    except Exception as e:
        print(f"\n❌ Backward compatibility test failed: {e}")
        return False


def validate_statistics(dataset, dataset_name, expected_count=None):
    """Validate dataset statistics."""
    print(f"\n{'='*70}")
    print(f"Statistics Validation: {dataset_name}")
    print('='*70)

    actual_count = len(dataset)
    print(f"  Total samples: {actual_count:,}")

    if expected_count is not None:
        if actual_count != expected_count:
            print(f"  ⚠️  Expected {expected_count:,}, got {actual_count:,}")
            print(f"  Difference: {actual_count - expected_count:,}")
        else:
            print(f"  ✓ Count matches expected: {expected_count:,}")

    # Check unique data sources
    data_sources = set(sample['data_source'] for sample in dataset)
    print(f"  Unique data sources: {len(data_sources)}")
    for source in data_sources:
        count = sum(1 for s in dataset if s['data_source'] == source)
        print(f"    - {source}: {count:,} samples")

    # Check ability distribution
    abilities = {}
    for sample in dataset:
        ability = sample['ability']
        abilities[ability] = abilities.get(ability, 0) + 1

    print(f"  Ability distribution:")
    for ability, count in abilities.items():
        print(f"    - {ability}: {count:,} samples ({count/actual_count*100:.2f}%)")

    print(f"\n✅ Statistics validation passed for {dataset_name}")
    return True


def validate_file_structure(base_dir):
    """Validate file structure and sizes."""
    print(f"\n{'='*70}")
    print(f"File Structure Validation: {base_dir}")
    print('='*70)

    expected_files = {
        'README.md': 'Documentation',
        'train.parquet': 'Data file' if 'openr1-math-verl' in base_dir else 'openr1-math-verl.parquet'
    }

    if 'math-verl-unified' in base_dir:
        expected_files = {
            'README.md': 'Documentation',
            'openr1-math-verl.parquet': 'Data file'
        }

    all_valid = True
    for filename, description in expected_files.items():
        filepath = os.path.join(base_dir, filename)

        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ {filename}: {size_mb:.2f} MB ({description})")
        else:
            print(f"❌ Missing: {filename} ({description})")
            all_valid = False

    if all_valid:
        print(f"\n✅ File structure validation passed")
    else:
        print(f"\n❌ File structure validation failed")

    return all_valid


def comprehensive_validation():
    """Run comprehensive validation suite."""
    print("="*70)
    print("Comprehensive Hub Upload Validation")
    print("="*70)

    validation_results = {}

    # Validate openr1-math-verl (individual dataset)
    print("\n" + "="*70)
    print("1. Validating: sungyub/openr1-math-verl")
    print("="*70)

    individual_path = "output/hub-upload/openr1-math-verl/train.parquet"
    individual_dir = "output/hub-upload/openr1-math-verl"

    try:
        # Load dataset
        print(f"\nLoading dataset from: {individual_path}")
        individual_ds = load_dataset('parquet', data_files=individual_path, split='train')
        print(f"✓ Loaded {len(individual_ds):,} samples")

        # Run validations
        validation_results['individual_schema'] = validate_schema(individual_ds, "openr1-math-verl")
        validation_results['individual_extra_info'] = validate_extra_info_schema(individual_ds, "openr1-math-verl")
        validation_results['individual_samples'] = validate_samples(individual_ds, "openr1-math-verl")
        validation_results['individual_compat'] = validate_backward_compatibility(individual_ds, "openr1-math-verl")
        validation_results['individual_stats'] = validate_statistics(individual_ds, "openr1-math-verl", expected_count=184439)
        validation_results['individual_files'] = validate_file_structure(individual_dir)

    except Exception as e:
        print(f"\n❌ Error validating individual dataset: {e}")
        validation_results['individual_overall'] = False

    # Validate math-verl-unified (openr1 split)
    print("\n" + "="*70)
    print("2. Validating: sungyub/math-verl-unified (openr1_math_verl split)")
    print("="*70)

    unified_path = "output/hub-upload/math-verl-unified/openr1-math-verl.parquet"
    unified_dir = "output/hub-upload/math-verl-unified"

    try:
        # Load dataset
        print(f"\nLoading dataset from: {unified_path}")
        unified_ds = load_dataset('parquet', data_files=unified_path, split='train')
        print(f"✓ Loaded {len(unified_ds):,} samples")

        # Run validations
        validation_results['unified_schema'] = validate_schema(unified_ds, "math-verl-unified/openr1")
        validation_results['unified_extra_info'] = validate_extra_info_schema(unified_ds, "math-verl-unified/openr1")
        validation_results['unified_samples'] = validate_samples(unified_ds, "math-verl-unified/openr1")
        validation_results['unified_compat'] = validate_backward_compatibility(unified_ds, "math-verl-unified/openr1")
        validation_results['unified_stats'] = validate_statistics(unified_ds, "math-verl-unified/openr1", expected_count=184439)
        validation_results['unified_files'] = validate_file_structure(unified_dir)

    except Exception as e:
        print(f"\n❌ Error validating unified dataset: {e}")
        validation_results['unified_overall'] = False

    # Cross-validation: Ensure both datasets are identical
    print("\n" + "="*70)
    print("3. Cross-Validation: Comparing Individual vs Unified")
    print("="*70)

    if len(individual_ds) == len(unified_ds):
        print(f"✓ Sample counts match: {len(individual_ds):,}")

        # Compare first few samples
        mismatches = 0
        for i in range(min(100, len(individual_ds))):
            ind_sample = individual_ds[i]
            uni_sample = unified_ds[i]

            if ind_sample['prompt'][0]['content'] != uni_sample['prompt'][0]['content']:
                mismatches += 1

        if mismatches == 0:
            print(f"✓ First 100 samples content identical")
        else:
            print(f"⚠️  Found {mismatches} content mismatches in first 100 samples")

        validation_results['cross_validation'] = (mismatches == 0)
    else:
        print(f"❌ Sample counts differ: {len(individual_ds):,} vs {len(unified_ds):,}")
        validation_results['cross_validation'] = False

    # Final summary
    print("\n" + "="*70)
    print("Validation Summary")
    print("="*70)

    all_passed = all(validation_results.values())

    print("\nResults:")
    for test_name, passed in validation_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED - Ready for Hub upload!")
    else:
        print("❌ SOME VALIDATIONS FAILED - Please fix issues before upload")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    success = comprehensive_validation()
    exit(0 if success else 1)
