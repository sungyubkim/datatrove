#!/usr/bin/env python3
"""
Prepare OpenR1-Math cleaned dataset for Hugging Face Hub upload.

This script:
1. Reads the maximum-cleaned OpenR1-Math dataset
2. Adds 'original_dataset' field to extra_info for backward compatibility
3. Validates schema structure
4. Outputs two versions:
   - Individual dataset version (for sungyub/openr1-math-verl)
   - Unified dataset version (for sungyub/math-verl-unified)
"""

import os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

def add_original_dataset_field(dataset, original_dataset_name="OpenR1-Math-220k"):
    """
    Add 'original_dataset' field to extra_info structure for backward compatibility.

    Args:
        dataset: HuggingFace Dataset object
        original_dataset_name: Name to use for original_dataset field

    Returns:
        Modified dataset with updated extra_info structure
    """
    print(f"Adding 'original_dataset' field to extra_info...")
    print(f"  Value: {original_dataset_name}")

    def add_field(example):
        """Add original_dataset field to each example's extra_info."""
        if 'extra_info' in example and example['extra_info'] is not None:
            # Create new extra_info dict with original_dataset field
            extra_info = dict(example['extra_info'])
            extra_info['original_dataset'] = original_dataset_name
            example['extra_info'] = extra_info
        return example

    # Apply transformation
    dataset = dataset.map(add_field, desc="Adding original_dataset field")

    return dataset

def normalize_extra_info_schema(dataset):
    """
    Normalize extra_info to match math-verl-unified standard schema.

    Keeps only 3 standard fields:
    - index: int
    - original_dataset: str
    - split: str

    Removes extra fields: problem_type, question_type, source

    Args:
        dataset: HuggingFace Dataset object

    Returns:
        Modified dataset with normalized extra_info structure
    """
    from datasets import Dataset, Features, Value, Sequence

    print(f"\nNormalizing extra_info schema to standard 3-field structure...")

    # First, flatten the extra_info fields to top-level columns
    def flatten_extra_info(example):
        extra_info = example.get('extra_info', {})
        return {
            'data_source': example['data_source'],
            'prompt': example['prompt'],
            'ability': example['ability'],
            'reward_model': example['reward_model'],
            'extra_info_index': extra_info.get('index', 0),
            'extra_info_original_dataset': extra_info.get('original_dataset', 'OpenR1-Math-220k'),
            'extra_info_split': extra_info.get('split', 'train'),
        }

    dataset = dataset.map(flatten_extra_info, desc="Flattening extra_info", remove_columns=dataset.column_names)

    # Now restructure with only the 3 standard extra_info fields
    def restructure_extra_info(example):
        return {
            'data_source': example['data_source'],
            'prompt': example['prompt'],
            'ability': example['ability'],
            'reward_model': example['reward_model'],
            'extra_info': {
                'index': example['extra_info_index'],
                'original_dataset': example['extra_info_original_dataset'],
                'split': example['extra_info_split'],
            }
        }

    dataset = dataset.map(restructure_extra_info, desc="Restructuring extra_info", remove_columns=['extra_info_index', 'extra_info_original_dataset', 'extra_info_split'])

    # Verify normalization
    sample = dataset[0]
    actual_fields = set(sample['extra_info'].keys())
    expected_fields = {'index', 'original_dataset', 'split'}

    if actual_fields == expected_fields:
        print(f"  ✓ Normalized to {len(actual_fields)} fields: {sorted(actual_fields)}")
    else:
        extra = actual_fields - expected_fields
        missing = expected_fields - actual_fields
        if extra:
            print(f"  ⚠️  Extra fields still present: {extra}")
        if missing:
            print(f"  ⚠️  Missing fields: {missing}")

    return dataset

def validate_schema(dataset):
    """
    Validate that the dataset has the expected schema.

    Args:
        dataset: HuggingFace Dataset object

    Returns:
        bool: True if schema is valid, raises error otherwise
    """
    print("\nValidating schema...")

    # Check required top-level fields
    required_fields = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
    for field in required_fields:
        if field not in dataset.features:
            raise ValueError(f"Missing required field: {field}")
        print(f"  ✓ {field}: {dataset.features[field]}")

    # Check extra_info structure
    extra_info_features = dataset.features['extra_info']
    required_extra_info_fields = ['index', 'original_dataset']

    print("\n  extra_info structure:")
    for field in extra_info_features:
        print(f"    - {field}: {extra_info_features[field]}")

    for field in required_extra_info_fields:
        if field not in extra_info_features:
            raise ValueError(f"Missing required extra_info field: {field}")

    # Validate sample data
    print("\n  Sample validation:")
    sample = dataset[0]
    print(f"    data_source: {sample['data_source']}")
    print(f"    ability: {sample['ability']}")
    print(f"    extra_info.original_dataset: {sample['extra_info']['original_dataset']}")
    print(f"    extra_info.index: {sample['extra_info']['index']}")
    if 'problem_type' in sample['extra_info']:
        print(f"    extra_info.problem_type: {sample['extra_info']['problem_type']}")
    if 'source' in sample['extra_info']:
        print(f"    extra_info.source: {sample['extra_info']['source']}")

    print("\n  ✓ Schema validation passed!")
    return True

def save_for_hub_upload(
    input_path: str,
    output_dir_individual: str,
    output_dir_unified: str,
    original_dataset_name: str = "OpenR1-Math-220k"
):
    """
    Prepare dataset for Hub upload with schema compatibility.

    Args:
        input_path: Path to the cleaned dataset parquet file
        output_dir_individual: Output directory for individual dataset
        output_dir_unified: Output directory for unified dataset
        original_dataset_name: Name for the original_dataset field
    """
    print("="*70)
    print("Preparing OpenR1-Math Dataset for Hugging Face Hub Upload")
    print("="*70)

    # Load dataset
    print(f"\nLoading dataset from: {input_path}")
    dataset = load_dataset('parquet', data_files=input_path, split='train')
    print(f"  Total samples: {len(dataset):,}")

    # Add original_dataset field
    dataset = add_original_dataset_field(dataset, original_dataset_name)

    # Normalize extra_info schema to match math-verl-unified standard
    dataset = normalize_extra_info_schema(dataset)

    # Validate schema
    validate_schema(dataset)

    # Create output directories
    os.makedirs(output_dir_individual, exist_ok=True)
    os.makedirs(output_dir_unified, exist_ok=True)

    # Save for individual dataset (sungyub/openr1-math-verl)
    individual_output = os.path.join(output_dir_individual, "train.parquet")
    print(f"\nSaving for individual dataset...")
    print(f"  Output: {individual_output}")
    dataset.to_parquet(individual_output)
    file_size_mb = os.path.getsize(individual_output) / (1024 * 1024)
    print(f"  ✓ Saved: {file_size_mb:.2f} MB")

    # Save for unified dataset (sungyub/math-verl-unified)
    unified_output = os.path.join(output_dir_unified, "openr1-math-verl.parquet")
    print(f"\nSaving for unified dataset...")
    print(f"  Output: {unified_output}")
    dataset.to_parquet(unified_output)
    file_size_mb = os.path.getsize(unified_output) / (1024 * 1024)
    print(f"  ✓ Saved: {file_size_mb:.2f} MB")

    print("\n" + "="*70)
    print("Dataset Preparation Summary")
    print("="*70)
    print(f"Total samples:        {len(dataset):,}")
    print(f"Schema validated:     ✓")
    print(f"Individual dataset:   {individual_output}")
    print(f"Unified dataset:      {unified_output}")
    print(f"original_dataset:     {original_dataset_name}")
    print("="*70)
    print("✅ Preparation complete!")
    print("="*70)

    return dataset

def verify_compatibility(dataset):
    """
    Verify backward compatibility with old schema.

    Simulates old user code that accesses extra_info.original_dataset
    """
    print("\n" + "="*70)
    print("Testing Backward Compatibility")
    print("="*70)

    try:
        # Simulate old user code
        sample = dataset[0]
        original_dataset = sample['extra_info']['original_dataset']
        print(f"✓ Old code compatibility: extra_info['original_dataset'] = '{original_dataset}'")

        # Test new fields
        if 'problem_type' in sample['extra_info']:
            print(f"✓ New field available: extra_info['problem_type'] = '{sample['extra_info']['problem_type']}'")
        if 'source' in sample['extra_info']:
            print(f"✓ New field available: extra_info['source'] = '{sample['extra_info']['source']}'")

        print("\n✅ Backward compatibility verified!")
        return True

    except Exception as e:
        print(f"\n❌ Compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    # Configuration
    input_path = "output/openr1-cleaned-maxclean/train.parquet"
    output_dir_individual = "output/hub-upload/openr1-math-verl"
    output_dir_unified = "output/hub-upload/math-verl-unified"
    original_dataset_name = "OpenR1-Math-220k"

    # Prepare datasets
    dataset = save_for_hub_upload(
        input_path=input_path,
        output_dir_individual=output_dir_individual,
        output_dir_unified=output_dir_unified,
        original_dataset_name=original_dataset_name
    )

    # Verify compatibility
    verify_compatibility(dataset)
