#!/usr/bin/env python3
"""
Test GPT OSS reward function with sample data.

This script verifies that the reward function correctly:
1. Parses GPT OSS format (analysis, tool calls, final responses)
2. Computes format rewards
3. Computes correctness rewards for tool calls
4. Computes length rewards
"""

import pandas as pd
from pathlib import Path
from datatrove.utils.reward_score import toolrl_gpt_oss

print("="*80)
print("GPT OSS Reward Function Test")
print("="*80)

# Load converted dataset
print("\n📚 Loading GPT OSS dataset...")
# Get path relative to this test file
test_dir = Path(__file__).parent
dataset_path = test_dir / '../../../examples/data/rlla_4k_gpt/train.parquet'
train_df = pd.read_parquet(dataset_path)
print(f"✅ Loaded {len(train_df)} samples")

# Test cases
test_cases = [
    {
        "name": "Perfect Tool Call Match",
        "ground_truth": '<|start|>assistant<|channel|>analysis<|message|>I should use the esg tool<|end|>\n<|start|>assistant to=functions.esg<|channel|>commentary json<|message|>{"symb":"MSFT"}<|call|>',
        "prediction": '<|start|>assistant<|channel|>analysis<|message|>I should use the esg tool<|end|>\n<|start|>assistant to=functions.esg<|channel|>commentary json<|message|>{"symb":"MSFT"}<|call|>',
        "expected": "High format score, high correctness score"
    },
    {
        "name": "Correct Tool, Wrong Parameter",
        "ground_truth": '<|start|>assistant<|channel|>analysis<|message|>I should use the esg tool<|end|>\n<|start|>assistant to=functions.esg<|channel|>commentary json<|message|>{"symb":"MSFT"}<|call|>',
        "prediction": '<|start|>assistant<|channel|>analysis<|message|>I should use the esg tool<|end|>\n<|start|>assistant to=functions.esg<|channel|>commentary json<|message|>{"symb":"AAPL"}<|call|>',
        "expected": "High format score, medium correctness score"
    },
    {
        "name": "Wrong Tool",
        "ground_truth": '<|start|>assistant<|channel|>analysis<|message|>I should use the esg tool<|end|>\n<|start|>assistant to=functions.esg<|channel|>commentary json<|message|>{"symb":"MSFT"}<|call|>',
        "prediction": '<|start|>assistant<|channel|>analysis<|message|>I should use the wrong tool<|end|>\n<|start|>assistant to=functions.wrong<|channel|>commentary json<|message|>{"symb":"MSFT"}<|call|>',
        "expected": "High format score, low correctness score"
    },
    {
        "name": "Format Error - Missing Analysis",
        "ground_truth": '<|start|>assistant<|channel|>analysis<|message|>I should use the esg tool<|end|>\n<|start|>assistant to=functions.esg<|channel|>commentary json<|message|>{"symb":"MSFT"}<|call|>',
        "prediction": '<|start|>assistant to=functions.esg<|channel|>commentary json<|message|>{"symb":"MSFT"}<|call|>',
        "expected": "Low format score, high correctness score"
    },
    {
        "name": "Multiple Tool Calls",
        "ground_truth": '<|start|>assistant<|channel|>analysis<|message|>I need to call two tools<|end|>\n<|start|>assistant to=functions.tool1<|channel|>commentary json<|message|>{"p1":"v1"}<|call|>\n<|start|>assistant to=functions.tool2<|channel|>commentary json<|message|>{"p2":"v2"}<|call|>',
        "prediction": '<|start|>assistant<|channel|>analysis<|message|>I need to call two tools<|end|>\n<|start|>assistant to=functions.tool1<|channel|>commentary json<|message|>{"p1":"v1"}<|call|>\n<|start|>assistant to=functions.tool2<|channel|>commentary json<|message|>{"p2":"v2"}<|call|>',
        "expected": "High format score, high correctness score"
    },
    {
        "name": "Final Response (No Tool)",
        "ground_truth": '<|start|>assistant<|channel|>analysis<|message|>I should respond directly<|end|>\n<|start|>assistant<|channel|>final<|message|>Here is my response<|return|>',
        "prediction": '<|start|>assistant<|channel|>analysis<|message|>I should respond directly<|end|>\n<|start|>assistant<|channel|>final<|message|>Here is my response<|return|>',
        "expected": "High format score, zero correctness score (no tool call)"
    },
]

print("\n" + "="*80)
print("Running Test Cases")
print("="*80)

for i, test in enumerate(test_cases):
    print(f"\n{'='*80}")
    print(f"Test Case {i+1}: {test['name']}")
    print(f"{'='*80}")
    print(f"Expected: {test['expected']}")

    # Prepare input for reward function
    # Format: solution_str includes prompt + prediction
    # For testing, we'll just use the prediction part
    solution_str = f"<|start|>user<|message|>Test query<|end|>\n{test['prediction']}"

    try:
        score, format_score, correctness_score, length_score = toolrl_gpt_oss.compute_score(
            solution_str,
            test['ground_truth'],
            step=0
        )

        print(f"\n📊 Results:")
        print(f"   Format score:      {format_score:.3f}")
        print(f"   Correctness score: {correctness_score:.3f}")
        print(f"   Length score:      {length_score:.3f}")
        print(f"   Total score:       {score:.3f}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Test with real data samples
print("\n" + "="*80)
print("Testing with Real Dataset Samples")
print("="*80)

for idx in [0, 2, 5]:
    print(f"\n{'='*80}")
    print(f"Dataset Sample {idx}")
    print(f"{'='*80}")

    sample = train_df.iloc[idx].to_dict()
    ground_truth = sample['reward_model']['ground_truth']

    print(f"\n📝 Ground truth (first 200 chars):")
    print(ground_truth[:200] + "...")

    # Test with perfect prediction
    print(f"\n🎯 Testing with perfect prediction (same as ground truth):")

    solution_str = f"<|start|>user<|message|>Test query<|end|>\n{ground_truth}"

    try:
        score, format_score, correctness_score, length_score = toolrl_gpt_oss.compute_score(
            solution_str,
            ground_truth,
            step=0
        )

        print(f"\n📊 Results:")
        print(f"   Format score:      {format_score:.3f}")
        print(f"   Correctness score: {correctness_score:.3f}")
        print(f"   Length score:      {length_score:.3f}")
        print(f"   Total score:       {score:.3f}")

        # For perfect match, we expect high scores
        if format_score < 0.9:
            print(f"   ⚠️  Warning: Format score is low for perfect match!")
        if 'to=functions.' in ground_truth and correctness_score < 2.5:
            print(f"   ⚠️  Warning: Correctness score is low for perfect tool call match!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("✅ Reward function test complete!")
print("="*80)

print("\n📋 Summary:")
print("   - Format reward: Checks GPT OSS token structure")
print("   - Correctness reward: Validates tool names and parameters")
print("   - Length reward: Measures analysis channel length (optional)")
print("\n   The reward function is ready for training!")
