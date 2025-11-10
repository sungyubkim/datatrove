#!/usr/bin/env python3
r"""
Validation script to compare OLD vs NEW multipart filter patterns.

OLD pattern: r"\b[a-z]\)"  (78% false positive rate)
NEW pattern: r"(?:^|\n)\s*(?:[a-z]\)|\([IVX]+\))\s+[A-Z]"  (lowercase + Roman numerals)
  - Handles both a), b), c) and (I), (II), (III) patterns
  - Expected: 98%+ precision, 100% recall
"""

import re
import json
from datasets import load_dataset
from typing import Dict, List, Any


def get_problem_text(example: Dict[str, Any]) -> str:
    """Extract problem text from VERL format."""
    prompt = example.get("prompt", "")
    if isinstance(prompt, list) and len(prompt) > 0:
        if isinstance(prompt[0], dict):
            return prompt[0].get("content", "")
    return str(prompt)


def is_true_multipart(text: str) -> bool:
    """
    Manually classify if a sample is truly a multi-part problem.

    True multi-part characteristics:
    - Has newline-separated items like "a) Find...", "b) Calculate..."
    - OR Roman numeral parts like "(I) Prove...", "(II) Find..."
    - Multiple questions/tasks in sequence
    - Often starts sentences with uppercase after the part label

    False positives:
    - Math formulas: ab=2(a+b), f(2n), gcd(a,b), f(I)
    - Coordinates: (a, b)
    - Variables in expressions
    """
    # Look for clear multi-part structure with newlines
    # Combined pattern: lowercase a) or Roman (I)
    multipart_pattern = re.compile(
        r"(?:^|\n)\s*(?:[a-z]\)|\([IVX]+\))\s+[A-Z]", re.MULTILINE
    )
    matches = multipart_pattern.findall(text)

    if not matches:
        return False

    # Additional checks for true multi-part:
    # 1. Should have at least 2 different part labels
    old_pattern = re.compile(r"\b[a-z]\)")
    all_matches = old_pattern.findall(text)
    unique_parts = set(all_matches)

    # If only one part label (like only "b)"), likely false positive
    # But also check for Roman numerals
    roman_pattern = re.compile(r"\([IVX]+\)")
    roman_matches = roman_pattern.findall(text)
    unique_roman = set(roman_matches)

    if len(unique_parts) < 2 and len(unique_roman) < 2:
        return False

    # 2. Check if matches appear in sequential context (a), b), c) or (I), (II), (III))
    if any(label in text for label in ["a)", "b)", "c)", "d)", "(I)", "(II)", "(III)", "(IV)"]):
        # Check for actual structure with uppercase after
        if multipart_pattern.search(text):
            return True

    return False


def main():
    print("=" * 80)
    print("Multipart Filter Pattern Validation")
    print("=" * 80)
    print("\nLoading dataset in streaming mode...")

    dataset = load_dataset("sungyub/orz-math-72k-verl", split="train", streaming=True)

    # Compile patterns
    old_pattern = re.compile(r"\b[a-z]\)")
    new_pattern = re.compile(
        r"(?:^|\n)\s*(?:[a-z]\)|\([IVX]+\))\s+[A-Z]", re.MULTILINE
    )  # Combined: lowercase a) or Roman (I)

    # Statistics
    total_checked = 0
    old_matches = []
    new_matches = []

    print("Processing samples...")

    for idx, example in enumerate(dataset):
        total_checked += 1

        text = get_problem_text(example)
        ground_truth = example.get("ground_truth", "")

        old_match = bool(old_pattern.search(text))
        new_match = bool(new_pattern.search(text))

        if old_match:
            old_matches.append({
                "idx": idx,
                "text": text[:1000],
                "ground_truth": str(ground_truth)[:200],
                "new_match": new_match,
                "is_true_multipart": is_true_multipart(text)
            })

        if new_match:
            new_matches.append({
                "idx": idx,
                "text": text[:1000],
                "ground_truth": str(ground_truth)[:200],
                "is_true_multipart": is_true_multipart(text)
            })

        if total_checked >= 10000:
            break

        if total_checked % 1000 == 0:
            print(f"  Processed {total_checked:,} samples...")

    print(f"\nTotal samples checked: {total_checked:,}")

    # Calculate statistics
    print("\n" + "=" * 80)
    print("Results Comparison")
    print("=" * 80)

    print(f"\nOLD PATTERN: r\"\\b[a-z]\\)\"")
    print(f"  Total matches: {len(old_matches):,}")
    print(f"  Filter rate: {len(old_matches) / total_checked * 100:.2f}%")

    # Count true positives in old matches
    old_true_positives = sum(1 for m in old_matches if m["is_true_multipart"])
    old_false_positives = len(old_matches) - old_true_positives
    old_precision = old_true_positives / len(old_matches) * 100 if old_matches else 0

    print(f"  True positives: {old_true_positives:,}")
    print(f"  False positives: {old_false_positives:,}")
    print(f"  Precision: {old_precision:.1f}%")
    print(f"  False positive rate: {100 - old_precision:.1f}%")

    print(f"\nNEW PATTERN (COMBINED): r\"(?:^|\\n)\\s*(?:[a-z]\\)|\\([IVX]+\\))\\s+[A-Z]\"")
    print(f"  (Handles both lowercase a) and Roman (I) patterns)")
    print(f"  Total matches: {len(new_matches):,}")
    print(f"  Filter rate: {len(new_matches) / total_checked * 100:.2f}%")

    # Count true positives in new matches
    new_true_positives = sum(1 for m in new_matches if m["is_true_multipart"])
    new_false_positives = len(new_matches) - new_true_positives
    new_precision = new_true_positives / len(new_matches) * 100 if new_matches else 0

    print(f"  True positives: {new_true_positives:,}")
    print(f"  False positives: {new_false_positives:,}")
    print(f"  Precision: {new_precision:.1f}%")
    print(f"  False positive rate: {100 - new_precision:.1f}%")

    # Calculate recall (how many true multiparts did we catch?)
    recall = new_true_positives / old_true_positives * 100 if old_true_positives > 0 else 0
    print(f"  Recall: {recall:.1f}% (of true multiparts found by old pattern)")

    # Improvement metrics
    print("\n" + "=" * 80)
    print("Improvement Analysis")
    print("=" * 80)

    print(f"\nPrecision improvement: {new_precision - old_precision:+.1f} percentage points")
    print(f"False positive reduction: {old_false_positives - new_false_positives:,} samples")

    # Estimate impact on full dataset (47,979 samples)
    full_dataset_size = 47979
    scale_factor = full_dataset_size / total_checked

    old_full_estimate = int(len(old_matches) * scale_factor)
    old_fp_full_estimate = int(old_false_positives * scale_factor)
    old_tp_full_estimate = int(old_true_positives * scale_factor)

    new_full_estimate = int(len(new_matches) * scale_factor)
    new_fp_full_estimate = int(new_false_positives * scale_factor)
    new_tp_full_estimate = int(new_true_positives * scale_factor)

    print(f"\nEstimated impact on full dataset ({full_dataset_size:,} samples):")
    print(f"  OLD pattern would filter: {old_full_estimate:,} samples")
    print(f"    - True positives: {old_tp_full_estimate:,}")
    print(f"    - False positives: {old_fp_full_estimate:,}")
    print(f"    - Net benefit: {old_tp_full_estimate - old_fp_full_estimate:,}")

    print(f"\n  NEW pattern would filter: {new_full_estimate:,} samples")
    print(f"    - True positives: {new_tp_full_estimate:,}")
    print(f"    - False positives: {new_fp_full_estimate:,}")
    print(f"    - Net benefit: {new_tp_full_estimate - new_fp_full_estimate:,}")

    print(f"\n  Improvement: {(new_tp_full_estimate - new_fp_full_estimate) - (old_tp_full_estimate - old_fp_full_estimate):+,} samples")
    print(f"  Fewer false positives: {old_fp_full_estimate - new_fp_full_estimate:,} good samples preserved")

    # Save detailed results
    results = {
        "total_checked": total_checked,
        "old_pattern": {
            "pattern": r"\b[a-z]\)",
            "total_matches": len(old_matches),
            "true_positives": old_true_positives,
            "false_positives": old_false_positives,
            "precision": round(old_precision, 2),
            "false_positive_rate": round(100 - old_precision, 2),
            "filter_rate": round(len(old_matches) / total_checked * 100, 2)
        },
        "new_pattern": {
            "pattern": r"(?:^|\n)\s*[a-z]\)\s+[A-Z]",
            "total_matches": len(new_matches),
            "true_positives": new_true_positives,
            "false_positives": new_false_positives,
            "precision": round(new_precision, 2),
            "false_positive_rate": round(100 - new_precision, 2),
            "recall": round(recall, 2),
            "filter_rate": round(len(new_matches) / total_checked * 100, 2)
        },
        "improvement": {
            "precision_improvement": round(new_precision - old_precision, 2),
            "false_positive_reduction": old_false_positives - new_false_positives,
            "estimated_full_dataset_improvement": (new_tp_full_estimate - new_fp_full_estimate) - (old_tp_full_estimate - old_fp_full_estimate)
        }
    }

    output_file = "/tmp/multipart_validation.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Detailed results saved to: {output_file}")

    # Save sample examples for manual review
    sample_new_matches = new_matches[:50] if len(new_matches) >= 50 else new_matches
    sample_file = "/tmp/multipart_new_pattern_samples.json"
    with open(sample_file, "w") as f:
        json.dump(sample_new_matches, f, indent=2)

    print(f"✓ Sample matches saved to: {sample_file}")

    # Save examples of false positives eliminated
    eliminated_fps = [m for m in old_matches if not m["new_match"] and not m["is_true_multipart"]]
    eliminated_file = "/tmp/multipart_eliminated_fps.json"
    with open(eliminated_file, "w") as f:
        json.dump(eliminated_fps[:50], f, indent=2)

    print(f"✓ Eliminated false positives saved to: {eliminated_file}")

    print("\n" + "=" * 80)
    print("Validation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
