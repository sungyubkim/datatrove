#!/usr/bin/env python3
"""
Check HuggingFace Hub datasets for data_source coverage.

This script checks all datasets under a HuggingFace username and identifies
which data_source values are not covered by the reward scoring system.
"""

import sys
from collections import defaultdict
from typing import Dict, List, Set

from datasets import load_dataset
from huggingface_hub import HfApi


def get_user_datasets(username: str) -> List[str]:
    """Get all datasets for a given HuggingFace username."""
    api = HfApi()
    datasets = list(api.list_datasets(author=username))
    return [ds.id for ds in datasets]


def get_data_source_values(dataset_name: str, max_samples: int = 100) -> Set[str]:
    """
    Get unique data_source values from a dataset without downloading it.

    Uses streaming to avoid downloading the entire dataset.
    """
    data_sources = set()

    try:
        # Try to load with streaming
        print(f"  Checking {dataset_name}...", end=" ")
        ds = load_dataset(dataset_name, split="train", streaming=True)

        # Sample first N items to get data_source values
        count = 0
        for sample in ds:
            if "data_source" in sample:
                data_sources.add(sample["data_source"])
            count += 1
            if count >= max_samples:
                break

        print(f"✓ Found {len(data_sources)} unique data_source values (from {count} samples)")

    except Exception as e:
        print(f"✗ Error: {e}")

    return data_sources


def get_supported_data_sources() -> Set[str]:
    """
    Get all data_source values currently supported by the reward scoring system.

    This is based on the routing logic in __init__.py
    """
    supported = {
        # Math domain
        "openai/gsm8k", "lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval",
        "HuggingFaceH4/MATH-500", "numina_aops_forum", "numina_synthetic_math",
        "numina_amc_aime", "numina_synthetic_amc", "numina_cn_k12",
        "numina_olympiads", "math_dapo", "math", "math_dapo_reasoning",
        # Additional math datasets
        "Big-Math-RL-Verified", "DAPO-Math-17K", "DeepScaleR-Preview",
        "MathX-5M", "OpenR1-Math-220k", "orz-math-72k",
        "train-math-deepscaler", "train-math-numinamath1.5_amc_aime",
        "train-math-numinamath1.5_aops_forum", "train-math-numinamath1.5_cn_contest",
        "train-math-numinamath1.5_olympiads", "train-math-numinamath1.5_olympiads_ref",
        "train-math-still3",

        # Tool learning
        "rlla", "toolrl", "tool_learning", "toolace", "hammer", "xlam",
        "sungyub/toolrl-verl", "rlla_gpt",

        # Code execution
        "codecontests", "apps", "codeforces", "taco",
        # Additional code execution datasets
        "code-contests-plus", "kodcode-leetcode", "oss", "rstar-coder",
        "train-code-leetcode-Easy", "train-code-leetcode-Medium",
        "train-code-leetcode-Hard", "test-code-leetcode-Medium",
        "train-code-taco-easy", "train-code-taco-medium",
        "train-code-taco-hard", "train-code-taco-medium_hard",
        "train-code-taco-very_hard", "train-code-taco-unknown_difficulty",

        # Instruction following
        "allenai/IF_multi_constraints_upto5", "ifeval", "sungyub/ifbench-verl",

        # CodeV
        "codev", "sungyub/codev-r1-verl",

        # Table reasoning - boxed
        "hitab", "multihier", "finqa",

        # Table QA - JSON list
        "WTQ", "HiTab",

        # Table fact verification
        "TabFact",

        # Free-form table QA
        "FeTaQA",

        # Logic reasoning
        "ordering_puzzle", "zebra_puzzle", "graph_logical",
        "arcagi1", "arcagi2", "barc",
    }

    # Pattern-based matches (substring matching)
    pattern_matches = {
        "long_toc_choices",  # any dataset with this substring
        "docmath",           # any dataset with this substring
        "multihoprag",       # any dataset with this substring
        "musique",           # any dataset with this substring
        "puzzle",            # any dataset with this substring
        "arcagi",            # any dataset with this substring
        "barc",              # any dataset with this substring (duplicate but ok)
    }

    return supported, pattern_matches


def is_supported(data_source: str, supported: Set[str], patterns: Set[str]) -> bool:
    """Check if a data_source is supported by exact match or pattern."""
    # Exact match
    if data_source in supported:
        return True

    # Pattern match
    for pattern in patterns:
        if pattern in data_source:
            return True

    return False


def main():
    username = "sungyub"

    print(f"🔍 Checking datasets for user: {username}\n")

    # Get all datasets
    print("📦 Fetching dataset list from HuggingFace Hub...")
    try:
        datasets = get_user_datasets(username)
        print(f"Found {len(datasets)} datasets:\n")
        for ds in datasets:
            print(f"  - {ds}")
        print()
    except Exception as e:
        print(f"❌ Error fetching datasets: {e}")
        sys.exit(1)

    # Get data_source values from each dataset
    print("🔎 Analyzing data_source values in each dataset...\n")
    dataset_data_sources: Dict[str, Set[str]] = {}

    for dataset_name in datasets:
        data_sources = get_data_source_values(dataset_name, max_samples=100)
        if data_sources:
            dataset_data_sources[dataset_name] = data_sources

    print()

    # Get supported data sources
    supported, patterns = get_supported_data_sources()

    # Analyze coverage
    print("📊 Coverage Analysis:\n")
    print("=" * 80)

    all_data_sources = set()
    for ds_name, ds_values in dataset_data_sources.items():
        all_data_sources.update(ds_values)

    # Categorize
    covered = set()
    uncovered = set()

    for data_source in sorted(all_data_sources):
        if is_supported(data_source, supported, patterns):
            covered.add(data_source)
        else:
            uncovered.add(data_source)

    # Print results
    print(f"\n✅ COVERED data_source values ({len(covered)}):")
    for ds in sorted(covered):
        # Find which datasets use this
        using_datasets = [name for name, values in dataset_data_sources.items() if ds in values]
        print(f"  ✓ {ds}")
        for dataset_name in using_datasets:
            print(f"      (used in {dataset_name})")

    print(f"\n❌ UNCOVERED data_source values ({len(uncovered)}):")
    if uncovered:
        for ds in sorted(uncovered):
            # Find which datasets use this
            using_datasets = [name for name, values in dataset_data_sources.items() if ds in values]
            print(f"  ✗ {ds}")
            for dataset_name in using_datasets:
                print(f"      (used in {dataset_name})")
    else:
        print("  None! All data_source values are covered! 🎉")

    print("\n" + "=" * 80)
    print(f"\nSummary:")
    print(f"  Total unique data_source values: {len(all_data_sources)}")
    print(f"  Covered: {len(covered)} ({len(covered) / len(all_data_sources) * 100:.1f}%)")
    print(f"  Uncovered: {len(uncovered)} ({len(uncovered) / len(all_data_sources) * 100:.1f}%)")

    if uncovered:
        print(f"\n⚠️  Action needed: Add support for {len(uncovered)} uncovered data_source values")
        sys.exit(1)
    else:
        print("\n✅ All data_source values are supported!")
        sys.exit(0)


if __name__ == "__main__":
    main()
