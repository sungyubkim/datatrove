#!/usr/bin/env python3
"""
Integration test for IF Eval scorer with datatrove dispatcher.
"""

import sys
sys.path.insert(0, "/Users/sungyubkim/repos/open-instruct")

from datatrove.utils.reward_score import compute_score

def test_dispatcher_routing():
    """Test that dispatcher correctly routes to ifeval module."""
    print("=" * 80)
    print("TEST: Dispatcher Routing")
    print("=" * 80)

    # Test data
    gt = "[{'instruction_id': ['last_word:last_word_answer'], 'kwargs': [{'last_word': 'brief'}]}]"
    output_pass = "This is a test response that ends with brief"
    output_fail = "This is a test response that ends with other"

    # Test 1: allenai/IF_multi_constraints_upto5
    print("\n1. Testing data_source='allenai/IF_multi_constraints_upto5'")
    result_pass = compute_score(
        data_source="allenai/IF_multi_constraints_upto5",
        solution_str=output_pass,
        ground_truth=gt
    )
    result_fail = compute_score(
        data_source="allenai/IF_multi_constraints_upto5",
        solution_str=output_fail,
        ground_truth=gt
    )

    print(f"   Pass case: {result_pass}")
    print(f"   Fail case: {result_fail}")

    assert result_pass['score'] > result_fail['score'], "Pass score should be higher than fail score"
    assert result_pass['score'] == 1.0, "Pass score should be 1.0"
    assert result_fail['score'] == 0.0, "Fail score should be 0.0"
    print("   ✅ PASSED")

    # Test 2: ifeval
    print("\n2. Testing data_source='ifeval'")
    result_pass = compute_score(
        data_source="ifeval",
        solution_str=output_pass,
        ground_truth=gt
    )
    result_fail = compute_score(
        data_source="ifeval",
        solution_str=output_fail,
        ground_truth=gt
    )

    print(f"   Pass case: {result_pass}")
    print(f"   Fail case: {result_fail}")

    assert result_pass['score'] > result_fail['score']
    print("   ✅ PASSED")

    # Test 3: sungyub/ifbench-verl
    print("\n3. Testing data_source='sungyub/ifbench-verl'")
    result_pass = compute_score(
        data_source="sungyub/ifbench-verl",
        solution_str=output_pass,
        ground_truth=gt
    )

    print(f"   Pass case: {result_pass}")
    assert result_pass['score'] == 1.0
    print("   ✅ PASSED")

    print("\n" + "=" * 80)
    print("✅ All dispatcher routing tests PASSED")
    print("=" * 80)


def test_with_real_dataset():
    """Test with samples from actual IF dataset."""
    print("\n" + "=" * 80)
    print("TEST: Real Dataset Samples")
    print("=" * 80)

    from datasets import load_dataset

    # Load test dataset
    print("\nLoading test dataset...")
    ds = load_dataset("allenai/IF_multi_constraints_upto5", split='train', streaming=True)

    # Get 5 samples
    samples = list(ds.take(5))

    print(f"\nTesting {len(samples)} samples...")
    for i, sample in enumerate(samples):
        print(f"\n  Sample {i+1}:")
        print(f"    Ground truth (first 100 chars): {sample['ground_truth'][:100]}...")

        # Test with a simple response
        response = "This is a brief test response."

        try:
            result = compute_score(
                data_source="allenai/IF_multi_constraints_upto5",
                solution_str=response,
                ground_truth=sample['ground_truth']
            )
            print(f"    Score: {result['score']:.2f}")
            print(f"    ✓ Computation successful")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            raise

    print("\n" + "=" * 80)
    print("✅ Real dataset test PASSED")
    print("=" * 80)


def test_empty_and_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 80)
    print("TEST: Edge Cases")
    print("=" * 80)

    gt = "[{'instruction_id': ['last_word:last_word_answer'], 'kwargs': [{'last_word': 'brief'}]}]"

    # Test 1: Empty output
    print("\n1. Empty model output")
    result = compute_score(
        data_source="ifeval",
        solution_str="",
        ground_truth=gt
    )
    print(f"   Result: {result}")
    assert result['score'] == 0.0, "Empty output should score 0.0"
    print("   ✅ PASSED")

    # Test 2: Very long output
    print("\n2. Very long model output")
    long_output = "word " * 1000 + "brief"
    result = compute_score(
        data_source="ifeval",
        solution_str=long_output,
        ground_truth=gt
    )
    print(f"   Result: {result}")
    assert result['score'] == 1.0, "Long output ending with 'brief' should score 1.0"
    print("   ✅ PASSED")

    # Test 3: Multiple constraints
    print("\n3. Multiple constraints")
    gt_multi = "[{'instruction_id': ['detectable_format:sentence_hyphens', 'last_word:last_word_answer'], 'kwargs': [None, {'last_word': 'brief'}]}]"
    output_multi_pass = "This is the first sentence.-Here is another sentence ending with brief."
    output_multi_partial = "First-sentence with hyphens in wrong places ending with brief."  # Hyphens don't separate sentences properly

    result_pass = compute_score(
        data_source="ifeval",
        solution_str=output_multi_pass,
        ground_truth=gt_multi
    )
    result_partial = compute_score(
        data_source="ifeval",
        solution_str=output_multi_partial,
        ground_truth=gt_multi
    )

    print(f"   Full pass: {result_pass}")
    print(f"   Partial pass: {result_partial}")

    assert result_pass['score'] == 1.0, "Full pass should score 1.0"
    assert 0.0 < result_partial['score'] < 1.0, "Partial pass should score between 0 and 1"
    print("   ✅ PASSED")

    print("\n" + "=" * 80)
    print("✅ All edge case tests PASSED")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("IF EVAL INTEGRATION TESTS")
    print("=" * 80)

    try:
        # Run tests
        test_dispatcher_routing()
        test_with_real_dataset()
        test_empty_and_edge_cases()

        print("\n" + "=" * 80)
        print("🎉 ALL INTEGRATION TESTS PASSED 🎉")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
