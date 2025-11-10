#!/usr/bin/env python3
"""
Verify uploaded datasets on Hugging Face Hub.

This script loads datasets from Hub and performs quick verification to ensure
uploads were successful and data is accessible.
"""

import sys
from datasets import load_dataset


def verify_standalone():
    """Verify standalone dataset on Hub"""
    print("=" * 70)
    print("Verifying: sungyub/deepscaler-preview-verl")
    print("=" * 70)

    try:
        # Load in streaming mode for quick check
        dataset = load_dataset("sungyub/deepscaler-preview-verl", split="train", streaming=True)
        print("✓ Dataset loaded successfully in streaming mode")

        # Get first sample
        first_sample = next(iter(dataset))
        print(f"✓ First sample accessed successfully")

        # Verify structure
        print(f"\nSample structure:")
        print(f"  data_source: {first_sample['data_source']}")
        print(f"  ability: {first_sample['ability']}")
        print(f"  prompt[0]['role']: {first_sample['prompt'][0]['role']}")
        print(f"  prompt[0]['content'][:80]: {first_sample['prompt'][0]['content'][:80]}...")
        print(f"  reward_model['style']: {first_sample['reward_model']['style']}")
        print(f"  reward_model['ground_truth']: {first_sample['reward_model']['ground_truth']}")
        print(f"  extra_info: {first_sample['extra_info']}")

        # Verify extra_info has correct fields (standalone)
        extra_info_keys = set(first_sample['extra_info'].keys())
        expected_keys = {'index', 'split'}

        if extra_info_keys == expected_keys:
            print(f"✓ extra_info has correct fields: {expected_keys}")
        else:
            print(f"✗ extra_info fields mismatch!")
            print(f"  Expected: {expected_keys}")
            print(f"  Got: {extra_info_keys}")
            return False

        return True

    except Exception as e:
        print(f"✗ Error loading standalone dataset: {e}")
        return False


def verify_unified():
    """Verify unified dataset on Hub"""
    print("\n" + "=" * 70)
    print("Verifying: sungyub/math-verl-unified (deepscaler_preview_verl split)")
    print("=" * 70)

    try:
        # Load deepscaler split in streaming mode
        dataset = load_dataset("sungyub/math-verl-unified", split="deepscaler_preview_verl", streaming=True)
        print("✓ Dataset split loaded successfully in streaming mode")

        # Get first sample
        first_sample = next(iter(dataset))
        print(f"✓ First sample accessed successfully")

        # Verify structure
        print(f"\nSample structure:")
        print(f"  data_source: {first_sample['data_source']}")
        print(f"  ability: {first_sample['ability']}")
        print(f"  prompt[0]['role']: {first_sample['prompt'][0]['role']}")
        print(f"  prompt[0]['content'][:80]: {first_sample['prompt'][0]['content'][:80]}...")
        print(f"  reward_model['style']: {first_sample['reward_model']['style']}")
        print(f"  reward_model['ground_truth']: {first_sample['reward_model']['ground_truth']}")
        print(f"  extra_info: {first_sample['extra_info']}")

        # Verify extra_info has correct fields (unified with original_dataset)
        extra_info_keys = set(first_sample['extra_info'].keys())
        expected_keys = {'index', 'split', 'original_dataset'}

        if extra_info_keys == expected_keys:
            print(f"✓ extra_info has correct fields: {expected_keys}")

            # Verify original_dataset value
            if first_sample['extra_info']['original_dataset'] == 'deepscaler-preview-verl':
                print(f"✓ original_dataset field has correct value")
            else:
                print(f"✗ original_dataset value incorrect: {first_sample['extra_info']['original_dataset']}")
                return False
        else:
            print(f"✗ extra_info fields mismatch!")
            print(f"  Expected: {expected_keys}")
            print(f"  Got: {extra_info_keys}")
            return False

        return True

    except Exception as e:
        print(f"✗ Error loading unified dataset: {e}")
        return False


def main():
    print("=" * 70)
    print("Hub Upload Verification")
    print("=" * 70)
    print("\nThis script verifies that uploaded datasets are accessible on Hub")
    print("and have the correct structure.\n")

    results = {}

    # Verify standalone
    results['standalone'] = verify_standalone()

    # Verify unified
    results['unified'] = verify_unified()

    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)

    all_passed = True
    for dataset_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {dataset_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n✅ All verifications passed! Datasets are live and working on Hub.")
        print(f"\n📊 Dataset URLs:")
        print(f"  Standalone: https://huggingface.co/datasets/sungyub/deepscaler-preview-verl")
        print(f"  Unified: https://huggingface.co/datasets/sungyub/math-verl-unified")
        return 0
    else:
        print(f"\n❌ Some verifications failed. Please check the datasets on Hub.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
