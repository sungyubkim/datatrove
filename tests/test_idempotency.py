#!/usr/bin/env python3
"""
Test idempotency and schema preservation of MathDatasetCleaner.

This script verifies that:
1. Input/output schema is preserved
2. Cleaning is idempotent (applying multiple times produces same result)
"""

import json
from datasets import load_dataset
from datatrove.data import Document
from datatrove.pipeline.formatters import MathDatasetCleaner


def test_schema_preservation():
    """Test that all VERL schema fields are preserved."""
    print("=" * 80)
    print("TEST 1: Schema Preservation")
    print("=" * 80)

    # Load a sample
    dataset = load_dataset("sungyub/openr1-math-verl", split="train", streaming=True)
    sample = next(iter(dataset))

    # Check original schema
    original_keys = set(sample.keys())
    print(f"\nOriginal schema fields: {sorted(original_keys)}")

    # Convert to Document
    doc = Document(
        id="test-1",
        text="",
        metadata=sample
    )

    # Apply cleaning
    cleaner = MathDatasetCleaner.from_preset("openr1-math")
    cleaned_docs = list(cleaner.run([doc], rank=0, world_size=1))
    cleaned_doc = cleaned_docs[0]

    # Check cleaned schema
    cleaned_keys = set(cleaned_doc.metadata.keys())
    print(f"Cleaned schema fields: {sorted(cleaned_keys)}")

    # Verify schema preservation
    if original_keys == cleaned_keys:
        print("\n✓ PASS: All schema fields preserved")
        schema_preserved = True
    else:
        print("\n✗ FAIL: Schema mismatch")
        print(f"  Missing fields: {original_keys - cleaned_keys}")
        print(f"  Added fields: {cleaned_keys - original_keys}")
        schema_preserved = False

    # Check nested structure preservation
    print("\n--- Nested Structure Check ---")

    checks = {
        "prompt": isinstance(cleaned_doc.metadata.get("prompt"), list),
        "prompt[0]": isinstance(cleaned_doc.metadata.get("prompt", [{}])[0], dict),
        "prompt[0].role": "role" in cleaned_doc.metadata.get("prompt", [{}])[0],
        "prompt[0].content": "content" in cleaned_doc.metadata.get("prompt", [{}])[0],
        "reward_model": isinstance(cleaned_doc.metadata.get("reward_model"), dict),
        "reward_model.style": "style" in cleaned_doc.metadata.get("reward_model", {}),
        "reward_model.ground_truth": "ground_truth" in cleaned_doc.metadata.get("reward_model", {}),
        "extra_info": isinstance(cleaned_doc.metadata.get("extra_info"), dict),
    }

    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}: {result}")

    all_nested_ok = all(checks.values())
    if all_nested_ok:
        print("\n✓ PASS: Nested structure preserved")
    else:
        print("\n✗ FAIL: Nested structure compromised")

    return schema_preserved and all_nested_ok


def test_idempotency():
    """Test that cleaning is idempotent (applying twice gives same result)."""
    print("\n" + "=" * 80)
    print("TEST 2: Idempotency")
    print("=" * 80)

    # Load samples
    dataset = load_dataset("sungyub/openr1-math-verl", split="train", streaming=True)
    samples = [next(iter(dataset)) for _ in range(10)]

    cleaner = MathDatasetCleaner.from_preset("openr1-math")

    idempotent_results = []

    for idx, sample in enumerate(samples):
        # First cleaning
        doc1 = Document(id=f"test-{idx}", text="", metadata=sample)
        cleaned1 = list(cleaner.run([doc1], rank=0, world_size=1))[0]

        # Second cleaning (on already cleaned data)
        doc2 = Document(id=f"test-{idx}-2", text="", metadata=cleaned1.metadata)
        cleaned2 = list(cleaner.run([doc2], rank=0, world_size=1))[0]

        # Compare results
        content1 = cleaned1.metadata["prompt"][0]["content"]
        content2 = cleaned2.metadata["prompt"][0]["content"]

        is_idempotent = content1 == content2

        if not is_idempotent:
            print(f"\n✗ Sample {idx} NOT idempotent:")
            print(f"  After 1st cleaning: {content1[:100]}...")
            print(f"  After 2nd cleaning: {content2[:100]}...")
        else:
            print(f"✓ Sample {idx}: idempotent")

        idempotent_results.append(is_idempotent)

        # Also check all other fields
        for key in cleaned1.metadata.keys():
            if key == "prompt":
                continue  # Already checked content
            if cleaned1.metadata[key] != cleaned2.metadata[key]:
                print(f"  ✗ Field '{key}' changed on second pass!")
                idempotent_results[-1] = False

    # Summary
    passed = sum(idempotent_results)
    total = len(idempotent_results)

    print(f"\n{'='*80}")
    print(f"Idempotency Test: {passed}/{total} passed")
    print(f"{'='*80}")

    if passed == total:
        print("✓ PASS: Cleaning is idempotent")
        return True
    else:
        print(f"✗ FAIL: {total - passed} samples not idempotent")
        return False


def test_multiple_passes():
    """Test applying cleaning 5 times to verify stability."""
    print("\n" + "=" * 80)
    print("TEST 3: Multiple Pass Stability")
    print("=" * 80)

    # Load sample
    dataset = load_dataset("sungyub/openr1-math-verl", split="train", streaming=True)
    sample = next(iter(dataset))

    cleaner = MathDatasetCleaner.from_preset("openr1-math")

    # Apply cleaning 5 times
    current_metadata = sample
    contents = []

    for pass_num in range(1, 6):
        doc = Document(id=f"test-pass-{pass_num}", text="", metadata=current_metadata)
        cleaned = list(cleaner.run([doc], rank=0, world_size=1))[0]
        current_metadata = cleaned.metadata

        content = cleaned.metadata["prompt"][0]["content"]
        contents.append(content)
        print(f"Pass {pass_num}: {len(content)} chars, hash={hash(content)}")

    # Check all passes produced same result
    all_same = all(c == contents[0] for c in contents)

    if all_same:
        print("\n✓ PASS: All 5 passes produced identical results")
        return True
    else:
        print("\n✗ FAIL: Results differ across passes")
        for i, content in enumerate(contents, 1):
            print(f"  Pass {i}: {content[:100]}...")
        return False


def test_field_immutability():
    """Test that non-prompt fields are never modified."""
    print("\n" + "=" * 80)
    print("TEST 4: Field Immutability")
    print("=" * 80)

    # Load samples
    dataset = load_dataset("sungyub/openr1-math-verl", split="train", streaming=True)
    samples = [next(iter(dataset)) for _ in range(10)]

    cleaner = MathDatasetCleaner.from_preset("openr1-math")

    all_immutable = True

    immutable_fields = [
        "data_source",
        "ability",
        "reward_model",
        "extra_info",
    ]

    for idx, sample in enumerate(samples):
        doc = Document(id=f"test-{idx}", text="", metadata=sample)
        cleaned = list(cleaner.run([doc], rank=0, world_size=1))[0]

        for field in immutable_fields:
            original_value = sample.get(field)
            cleaned_value = cleaned.metadata.get(field)

            if original_value != cleaned_value:
                print(f"✗ Sample {idx}: Field '{field}' was modified!")
                print(f"  Original: {original_value}")
                print(f"  Cleaned:  {cleaned_value}")
                all_immutable = False
            else:
                print(f"✓ Sample {idx}: Field '{field}' unchanged")

    if all_immutable:
        print("\n✓ PASS: All immutable fields preserved")
        return True
    else:
        print("\n✗ FAIL: Some fields were improperly modified")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MATH DATASET CLEANER - IDEMPOTENCY & SCHEMA VALIDATION")
    print("=" * 80)

    results = {
        "schema_preservation": test_schema_preservation(),
        "idempotency": test_idempotency(),
        "multiple_passes": test_multiple_passes(),
        "field_immutability": test_field_immutability(),
    }

    # Final report
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nThe MathDatasetCleaner is:")
        print("  • Schema-preserving: All VERL fields maintained")
        print("  • Idempotent: Safe to apply multiple times")
        print("  • Stable: Converges after first pass")
        print("  • Conservative: Only modifies prompt content")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nReview failures above and fix issues before production use.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
