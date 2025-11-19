#!/usr/bin/env python3
"""
Test that compute_score works for all data_source values.

This script verifies that the router correctly dispatches to scorers
for each supported data_source type.
"""

import sys
from datatrove.utils.reward_score import compute_score


def test_math_scoring():
    """Test math domain scoring."""
    print("Testing Math domain...")

    # Test with a simple math problem
    data_sources = ["Big-Math-RL-Verified", "DAPO-Math-17K", "train-math-deepscaler"]

    for ds in data_sources:
        try:
            # Simple test case
            result = compute_score(
                data_source=ds,
                solution_str="The answer is \\boxed{42}",
                ground_truth="42",
                format_type="auto"
            )
            print(f"  ✓ {ds}: {result}")
        except Exception as e:
            print(f"  ✗ {ds}: {type(e).__name__}: {e}")

    print()


def test_toolrl_scoring():
    """Test tool learning scoring."""
    print("Testing Tool Learning domain...")

    try:
        # Test rlla_gpt
        result = compute_score(
            data_source="rlla_gpt",
            solution_str="<think>I need to search</think>\n<tool_call>\n{\"name\": \"search\"}\n</tool_call>",
            ground_truth="<think>...</think>\n<tool_call>\n{\"name\": \"search\"}\n</tool_call>",
            format_type="auto"
        )
        print(f"  ✓ rlla_gpt: {result}")
    except Exception as e:
        print(f"  ✗ rlla_gpt: {type(e).__name__}: {e}")

    print()


def test_code_scoring_dry_run():
    """Test code execution scoring (without actual sandbox)."""
    print("Testing Code domain (dry run - no sandbox)...")

    data_sources = ["code-contests-plus", "kodcode-leetcode", "oss", "rstar-coder"]

    for ds in data_sources:
        try:
            # This should raise ValueError about missing sandbox
            result = compute_score(
                data_source=ds,
                solution_str="print(42)",
                ground_truth={"inputs": [""], "outputs": ["42"]},
            )
            print(f"  ✗ {ds}: Expected ValueError but got {result}")
        except ValueError as e:
            if "sandbox_fusion_url" in str(e):
                print(f"  ✓ {ds}: Correctly requires sandbox_fusion_url")
            else:
                print(f"  ✗ {ds}: Unexpected ValueError: {e}")
        except Exception as e:
            print(f"  ✗ {ds}: {type(e).__name__}: {e}")

    print()


def main():
    print("=" * 80)
    print("Testing Reward Scoring System")
    print("=" * 80)
    print()

    test_math_scoring()
    test_toolrl_scoring()
    test_code_scoring_dry_run()

    print("=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
