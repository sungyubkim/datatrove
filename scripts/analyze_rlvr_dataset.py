"""Analyze RLVR-IFeval dataset and verify compatibility with IFEval mapping.

This script:
1. Loads the RLVR-IFeval dataset
2. Extracts all unique function names
3. Verifies all functions are mapped to IFEval instructions
4. Reports statistics about the dataset
"""

import json
from collections import Counter

from datasets import load_dataset

from datatrove.utils.reward_score.ifeval.rlvr_mapping import RLVR_TO_IFEVAL_MAP


def analyze_rlvr_dataset(split="train", max_examples=None):
    """Analyze RLVR dataset and print statistics.

    Args:
        split: Dataset split to analyze (default: "train")
        max_examples: Maximum number of examples to analyze (default: None = all)
    """
    print(f"Loading RLVR-IFeval dataset (split: {split})...")

    # Load dataset
    if max_examples:
        dataset = load_dataset("allenai/RLVR-IFeval", split=f"{split}[:{max_examples}]", trust_remote_code=True)
    else:
        dataset = load_dataset("allenai/RLVR-IFeval", split=split, trust_remote_code=True)

    print(f"Loaded {len(dataset)} examples")
    print()

    # Extract function names
    func_names = []
    constraint_types = []
    unmapped_functions = set()

    for example in dataset:
        gt = json.loads(example["ground_truth"])
        func_name = gt["func_name"]
        func_names.append(func_name)

        # Check if mapped
        if func_name not in RLVR_TO_IFEVAL_MAP:
            unmapped_functions.add(func_name)

        # Extract constraint type if available
        if "constraint_type" in example:
            constraint_types.append(example["constraint_type"])

    # Print statistics
    unique_funcs = set(func_names)
    print(f"Unique function names: {len(unique_funcs)}")
    print(f"Functions in mapping: {len(RLVR_TO_IFEVAL_MAP)}")
    print()

    # Check coverage
    print("=" * 80)
    print("MAPPING COVERAGE")
    print("=" * 80)

    mapped_funcs = unique_funcs & set(RLVR_TO_IFEVAL_MAP.keys())
    print(f"Mapped functions: {len(mapped_funcs)}/{len(unique_funcs)}")

    if unmapped_functions:
        print()
        print("⚠️  UNMAPPED FUNCTIONS:")
        for func in sorted(unmapped_functions):
            print(f"  - {func}")
        print()
    else:
        print("✅ All functions are mapped!")
        print()

    # Function frequency
    print("=" * 80)
    print("FUNCTION FREQUENCY (Top 10)")
    print("=" * 80)
    func_counter = Counter(func_names)
    for func, count in func_counter.most_common(10):
        ifeval_id = RLVR_TO_IFEVAL_MAP.get(func, "UNMAPPED")
        pct = 100 * count / len(func_names)
        print(f"{func:40} → {ifeval_id:50} ({count:5} / {pct:5.2f}%)")
    print()

    # Constraint types
    if constraint_types:
        print("=" * 80)
        print("CONSTRAINT TYPE FREQUENCY (Top 10)")
        print("=" * 80)
        ct_counter = Counter(constraint_types)
        for ct, count in ct_counter.most_common(10):
            pct = 100 * count / len(constraint_types)
            print(f"{ct:50} ({count:5} / {pct:5.2f}%)")
        print()

    # Show mapping
    print("=" * 80)
    print("COMPLETE MAPPING")
    print("=" * 80)
    print(f"{'RLVR Function':40} → {'IFEval Instruction ID':50}")
    print("-" * 90)
    for func in sorted(unique_funcs):
        ifeval_id = RLVR_TO_IFEVAL_MAP.get(func, "❌ UNMAPPED")
        status = "✅" if func in RLVR_TO_IFEVAL_MAP else "❌"
        print(f"{status} {func:38} → {ifeval_id:50}")
    print()

    # Sample examples
    print("=" * 80)
    print("SAMPLE EXAMPLES")
    print("=" * 80)
    for i in range(min(3, len(dataset))):
        example = dataset[i]
        gt = json.loads(example["ground_truth"])
        print(f"\nExample {i+1}:")
        print(f"  Function: {gt['func_name']}")
        if isinstance(example['messages'], list) and len(example['messages']) > 0:
            print(f"  Message: {example['messages'][0]['content'][:100]}...")
        print(f"  Ground Truth (first 200 chars): {json.dumps(gt)[:200]}...")
    print()

    return {
        "total_examples": len(dataset),
        "unique_functions": len(unique_funcs),
        "mapped_functions": len(mapped_funcs),
        "unmapped_functions": list(unmapped_functions),
        "function_distribution": dict(func_counter),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze RLVR-IFeval dataset")
    parser.add_argument("--split", default="train", help="Dataset split to analyze (default: train)")
    parser.add_argument(
        "--max-examples", type=int, default=None, help="Maximum number of examples to analyze (default: all)"
    )

    args = parser.parse_args()

    stats = analyze_rlvr_dataset(split=args.split, max_examples=args.max_examples)

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total examples analyzed: {stats['total_examples']}")
    print(f"Unique functions found: {stats['unique_functions']}")
    print(f"Mapped functions: {stats['mapped_functions']}/{stats['unique_functions']}")
    if stats['unmapped_functions']:
        print(f"⚠️  Unmapped functions: {stats['unmapped_functions']}")
        exit(1)
    else:
        print("✅ All functions successfully mapped!")
        exit(0)
