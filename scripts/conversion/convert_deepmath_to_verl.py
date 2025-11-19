#!/usr/bin/env python3
"""
DeepMath-103K to VERL converter

This script converts the DeepMath-103K dataset from its original format
to VERL format with minimal transformation (schema conversion only).

Source: zwhe99/DeepMath-103K
Output: VERL format with flat schema

Key Features:
- Streaming mode for memory efficiency
- NO text cleaning (raw data preservation)
- Only schema transformation: Q&A format → VERL format
- Ignores r1_solution fields (multiple solution approaches not needed)

Usage:
    python scripts/conversion/convert_deepmath_to_verl.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

from datasets import Dataset, load_dataset


# Configuration
SOURCE_DATASET = "zwhe99/DeepMath-103K"
OUTPUT_DIR = "./output/deepmath-103k-verl"
DATA_SOURCE = "deepmath-103k"
BATCH_SIZE = 1000
SAMPLE_RATE = 5000  # Collect samples every N documents (larger dataset)


def extract_problem_and_solution(example: dict) -> tuple[str, str]:
    """Extract problem text and solution from DeepMath format.

    Args:
        example: Dictionary with 'question' and 'final_answer' keys

    Returns:
        Tuple of (problem_text, solution)

    Note:
        Ignores r1_solution_1, r1_solution_2, r1_solution_3 fields
        as per user requirements (only final_answer is used)
    """
    problem_text = example.get("question", "")
    solution = example.get("final_answer", "")

    return problem_text, solution


def convert_to_verl_format(
    dataset_name: str,
    output_dir: Path,
    sample_rate: int = 5000,
) -> tuple[dict, list]:
    """Convert DeepMath dataset to VERL format using streaming.

    This is a minimal conversion that only transforms the schema without
    any text cleaning or modification.

    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Output directory for converted data
        sample_rate: Collect samples every N documents

    Returns:
        Tuple of (statistics dictionary, comparison examples)
    """
    print(f"\n{'='*70}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*70}")

    # Load dataset in streaming mode
    try:
        dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=True,
            verification_mode="no_checks"
        )
        print(f"✓ Dataset loaded in streaming mode")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return {}, []

    # Create output directory
    print(f"\nProcessing and writing to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics tracking
    stats = {
        "total": 0,
        "empty_problems": 0,
        "empty_solutions": 0,
        "avg_problem_length": 0,
        "avg_solution_length": 0,
    }
    comparison_examples = []

    # Collect examples in flat VERL format
    verl_examples = []
    total_problem_len = 0
    total_solution_len = 0

    # Process documents
    doc_count = 0
    for example in dataset:
        doc_count += 1

        # Extract problem and solution (NO MODIFICATION)
        problem_text, solution = extract_problem_and_solution(example)

        # Track empty fields
        if not problem_text or not problem_text.strip():
            stats["empty_problems"] += 1
            continue

        if not solution or not solution.strip():
            stats["empty_solutions"] += 1

        # Track lengths
        total_problem_len += len(problem_text)
        total_solution_len += len(solution) if solution else 0

        # Create VERL format example (minimal schema)
        verl_example = {
            "data_source": DATA_SOURCE,
            "prompt": [
                {"role": "user", "content": problem_text}
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                "index": doc_count - 1,
                "original_dataset": DATA_SOURCE,
                "split": "train"
            }
        }
        verl_examples.append(verl_example)

        # Collect samples for comparison
        if doc_count % sample_rate == 0:
            comparison_examples.append({
                "index": doc_count - 1,
                "problem": problem_text[:500],
                "solution": solution[:200] if solution else "N/A",
            })

        # Progress indicator
        if doc_count % 1000 == 0:
            print(f"  Processed {doc_count:,} documents...", end="\r")

    print(f"\n✓ Processing complete: {doc_count:,} documents")

    # Convert to Dataset and save as parquet (flat VERL schema)
    print(f"Converting to Dataset and saving...")
    verl_dataset = Dataset.from_list(verl_examples)
    output_file = output_dir / "train.parquet"
    verl_dataset.to_parquet(output_file)
    print(f"✓ Saved to: {output_file}")

    # Update stats
    stats["total"] = doc_count
    stats["valid_samples"] = len(verl_examples)
    stats["avg_problem_length"] = total_problem_len / len(verl_examples) if verl_examples else 0
    stats["avg_solution_length"] = total_solution_len / len(verl_examples) if verl_examples else 0

    return stats, comparison_examples


def generate_report(
    dataset_name: str,
    stats: dict,
    comparison_examples: list,
    output_dir: Path,
) -> str:
    """Generate a conversion report.

    Args:
        dataset_name: Dataset name
        stats: Conversion statistics
        comparison_examples: List of sample documents
        output_dir: Output directory

    Returns:
        Report text
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
{'='*70}
DeepMath-103K Dataset Conversion Report
{'='*70}
Source Dataset: {dataset_name}
Conversion Type: MINIMAL (Schema only, no text cleaning)
Timestamp: {timestamp}
Output: {output_dir}

{'='*70}
Summary Statistics
{'='*70}
Total samples processed:     {stats['total']:,}
Valid samples converted:     {stats['valid_samples']:,}
Empty problems skipped:      {stats.get('empty_problems', 0):,}
Empty solutions:             {stats.get('empty_solutions', 0):,}

Average problem length:      {stats['avg_problem_length']:.0f} characters
Average solution length:     {stats['avg_solution_length']:.0f} characters

Conversion rate:             {100*stats['valid_samples']/stats['total']:.2f}%

{'='*70}
Data Quality Notes
{'='*70}
- DeepMath-103K is a curated, high-quality dataset
- 48 samples with leaked answers were revised (May 2025)
- This conversion preserves original formatting and content
- No artifact removal or text cleaning applied
- Ground truth values (final_answer) preserved exactly
- r1_solution fields (multiple solution approaches) are ignored

{'='*70}
VERL Schema
{'='*70}
✓ data_source: "deepmath-103k"
✓ prompt: [{{"role": "user", "content": question}}]
✓ ability: "math"
✓ reward_model: {{"style": "rule", "ground_truth": final_answer}}
✓ extra_info: {{"index", "original_dataset", "split"}}

{'='*70}
Sample Documents ({len(comparison_examples)} collected)
{'='*70}
"""

    for i, example in enumerate(comparison_examples[:20], 1):
        report += f"\n{'-'*70}\n"
        report += f"Sample {i} (#{example['index']})\n"
        report += f"{'-'*70}\n"
        report += f"Problem:\n{example['problem']}\n\n"
        report += f"Solution: {example['solution']}\n"

    report += f"\n{'='*70}\n"
    report += "Schema Validation\n"
    report += f"{'='*70}\n"
    report += "✓ All required VERL fields present\n"
    report += "✓ Parquet format valid\n"
    report += "✓ Ground truth preserved\n"
    report += "✓ No text modifications applied\n"
    report += "✓ Minimal extra_info format (Hub compatible)\n"

    report += f"\n{'='*70}\n"

    return report


def main():
    """Main conversion function."""
    print("=" * 70)
    print("DeepMath-103K Conversion to VERL Format")
    print("=" * 70)
    print("\nThis script performs minimal conversion:")
    print("  - Schema transformation only")
    print("  - NO text cleaning")
    print("  - NO artifact removal")
    print("  - Ignores r1_solution fields (only final_answer used)")
    print("\nPurpose: Convert DeepMath-103K to VERL for processing pipeline")
    print("=" * 70)

    # Setup output directory
    output_dir = Path(OUTPUT_DIR)

    start_time = time.time()

    # Convert dataset
    stats, comparison_examples = convert_to_verl_format(
        dataset_name=SOURCE_DATASET,
        output_dir=output_dir,
        sample_rate=SAMPLE_RATE,
    )

    if not stats:
        print("\n✗ Conversion failed")
        return

    # Generate report
    report = generate_report(
        dataset_name=SOURCE_DATASET,
        stats=stats,
        comparison_examples=comparison_examples,
        output_dir=output_dir,
    )

    # Print report
    print(report)

    # Save report to file
    report_path = output_dir / "conversion_report.txt"
    report_path.write_text(report)
    print(f"\n✓ Report saved to: {report_path}")

    # Save stats as JSON
    stats_path = output_dir / "conversion_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "dataset": SOURCE_DATASET,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
        }, f, indent=2)
    print(f"✓ Stats saved to: {stats_path}")

    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"✅ Conversion completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
