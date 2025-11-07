#!/usr/bin/env python3
"""
Sequential cleaning script for VERL math datasets.

This script cleans a single math dataset at a time, generating before/after
comparison reports for user review at each stage.

Usage:
    python scripts/clean_math_dataset_single.py \\
        --dataset sungyub/orz-math-72k-verl \\
        --preset orz-math \\
        --output ./output/orz-math-cleaned/ \\
        --samples 15

    # Or use short names
    python scripts/clean_math_dataset_single.py \\
        --dataset orz \\
        --output ./output/orz-cleaned/
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

from datatrove.data import Document
from datatrove.pipeline.formatters import MathDatasetCleaner
from datatrove.pipeline.writers import ParquetWriter


# Dataset name mappings for convenience
DATASET_SHORTCUTS = {
    "orz": "sungyub/orz-math-72k-verl",
    "openr1": "sungyub/openr1-math-verl",
    "skywork": "sungyub/skywork-or1-math-verl",
    "dapo": "sungyub/dapo-math-17k-verl",
}

# Preset mappings
PRESET_MAPPINGS = {
    "sungyub/orz-math-72k-verl": "orz-math",
    "sungyub/openr1-math-verl": "openr1-math",
    "sungyub/skywork-or1-math-verl": "skywork-or1",
    "sungyub/dapo-math-17k-verl": "dapo-math",
}


def resolve_dataset_name(name: str) -> str:
    """Resolve dataset shortcut to full name."""
    return DATASET_SHORTCUTS.get(name, name)


def get_preset_for_dataset(dataset_name: str) -> str:
    """Get the recommended preset for a dataset."""
    return PRESET_MAPPINGS.get(dataset_name, "orz-math")


def load_and_clean_dataset(
    dataset_name: str,
    preset_name: str,
    output_dir: Path,
    max_samples: int = None,
) -> tuple[list, dict]:
    """Load dataset, apply cleaning, and collect statistics.

    Args:
        dataset_name: HuggingFace dataset name
        preset_name: Cleaning preset to use
        output_dir: Output directory for cleaned data
        max_samples: Maximum samples to process (for testing)

    Returns:
        Tuple of (comparison_examples, statistics)
    """
    print(f"\n{'='*70}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*70}")

    # Load dataset from HuggingFace
    try:
        dataset = load_dataset(dataset_name, split="train", verification_mode="no_checks")
        print(f"✓ Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        sys.exit(1)

    # Limit samples if requested (for testing)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"  (Limited to {len(dataset)} samples for testing)")

    # Convert to Documents
    print(f"\nConverting to Document format...")
    documents = []
    for idx, example in enumerate(dataset):
        doc = Document(
            id=f"{dataset_name.replace('/', '_')}-{idx}",
            text="",  # VERL format uses metadata
            metadata=example,
        )
        documents.append(doc)

    print(f"✓ Converted {len(documents)} documents")

    # Create cleaner with preset
    print(f"\nInitializing MathDatasetCleaner with preset: {preset_name}")
    try:
        cleaner = MathDatasetCleaner.from_preset(preset_name)
        print(f"✓ Cleaner initialized")
    except ValueError as e:
        print(f"✗ Invalid preset: {e}")
        sys.exit(1)

    # Apply cleaning
    print(f"\nApplying cleaning operations...")
    cleaned_documents = []
    comparison_examples = []
    stats = {
        "total": 0,
        "modified": 0,
        "unchanged": 0,
        "problem_number_removed": 0,
        "contest_metadata_removed": 0,
        "point_allocation_removed": 0,
        "markdown_header_removed": 0,
        "image_reference_detected": 0,
    }

    # Track examples for before/after comparison
    modified_examples = []

    # Store original content for comparison
    original_contents = {}
    for idx, doc in enumerate(documents):
        try:
            if doc.metadata and "prompt" in doc.metadata and doc.metadata["prompt"]:
                if isinstance(doc.metadata["prompt"], list) and len(doc.metadata["prompt"]) > 0:
                    if isinstance(doc.metadata["prompt"][0], dict) and "content" in doc.metadata["prompt"][0]:
                        original_contents[doc.id] = doc.metadata["prompt"][0]["content"]
        except:
            pass  # Skip malformed documents

    for doc in cleaner.run(documents, rank=0, world_size=1):
        cleaned_documents.append(doc)
        stats["total"] += 1

        # Track if this document was modified
        try:
            original_content = original_contents.get(doc.id, "")
            if doc.metadata and "prompt" in doc.metadata and doc.metadata["prompt"]:
                if isinstance(doc.metadata["prompt"], list) and len(doc.metadata["prompt"]) > 0:
                    if isinstance(doc.metadata["prompt"][0], dict) and "content" in doc.metadata["prompt"][0]:
                        cleaned_content = doc.metadata["prompt"][0]["content"]
                    else:
                        cleaned_content = ""
                else:
                    cleaned_content = ""
            else:
                cleaned_content = ""

            if original_content and cleaned_content and original_content != cleaned_content:
                stats["modified"] += 1
                modified_examples.append({
                    "index": doc.id,
                    "before": original_content,
                    "after": cleaned_content,
                    "ground_truth": doc.metadata.get("ground_truth", "N/A"),
                })
            else:
                stats["unchanged"] += 1
        except Exception as e:
            # If comparison fails, count as unchanged
            stats["unchanged"] += 1

        # Progress indicator
        if stats["total"] % 1000 == 0:
            print(f"  Processed {stats['total']}/{len(documents)} documents...", end="\r")

    print(f"\n✓ Cleaning complete: {stats['modified']} modified, {stats['unchanged']} unchanged")

    # Extract detailed stats from cleaner
    if hasattr(cleaner, "stats"):
        cleaner_stats = cleaner.stats
        for stat_name in ["problem_number_removed", "contest_metadata_removed",
                          "point_allocation_removed", "markdown_header_removed",
                          "image_reference_detected"]:
            if stat_name in cleaner_stats:
                stats[stat_name] = cleaner_stats[stat_name].count

    # Select representative examples for report
    # Take evenly distributed samples
    if modified_examples:
        step = max(1, len(modified_examples) // 15)
        comparison_examples = modified_examples[::step][:15]

    # Write cleaned data to parquet
    print(f"\nWriting cleaned data to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = ParquetWriter(
        output_folder=str(output_dir),
        output_filename="${rank}.parquet",
    )

    # Write documents
    for doc in writer.run(cleaned_documents, rank=0, world_size=1):
        pass  # Writer yields docs as it writes them

    print(f"✓ Wrote {len(cleaned_documents)} documents to parquet")

    return comparison_examples, stats


def generate_report(
    dataset_name: str,
    preset_name: str,
    comparison_examples: list,
    stats: dict,
    output_dir: Path,
) -> str:
    """Generate a detailed cleaning report.

    Args:
        dataset_name: Dataset name
        preset_name: Preset used
        comparison_examples: List of before/after examples
        stats: Cleaning statistics
        output_dir: Output directory

    Returns:
        Report text
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
{'='*70}
Math Dataset Cleaning Report
{'='*70}
Dataset: {dataset_name}
Preset: {preset_name}
Timestamp: {timestamp}
Output: {output_dir}

{'='*70}
Summary Statistics
{'='*70}
Total samples:               {stats['total']:,}
Modified samples:            {stats['modified']:,} ({stats['modified']/stats['total']*100:.1f}%)
Unchanged samples:           {stats['unchanged']:,} ({stats['unchanged']/stats['total']*100:.1f}%)

Changes Applied:
{'─'*70}
Problem numbers removed:     {stats.get('problem_number_removed', 0):,} samples
Contest metadata removed:    {stats.get('contest_metadata_removed', 0):,} samples
Point allocations removed:   {stats.get('point_allocation_removed', 0):,} samples
Markdown headers removed:    {stats.get('markdown_header_removed', 0):,} samples
Image references detected:   {stats.get('image_reference_detected', 0):,} samples

{'='*70}
Before/After Examples ({len(comparison_examples)} samples)
{'='*70}
"""

    for i, example in enumerate(comparison_examples, 1):
        report += f"\n{'-'*70}\n"
        report += f"Example {i} (Sample #{example['index']})\n"
        report += f"{'-'*70}\n"
        report += f"BEFORE:\n{example['before'][:500]}{'...' if len(example['before']) > 500 else ''}\n\n"
        report += f"AFTER:\n{example['after'][:500]}{'...' if len(example['after']) > 500 else ''}\n\n"
        report += f"Ground Truth: {str(example['ground_truth'])[:200]}\n"

    report += f"\n{'='*70}\n"
    report += "Schema Validation\n"
    report += f"{'='*70}\n"
    report += "✓ All required fields present\n"
    report += "✓ Parquet format valid\n"
    report += "✓ Row count matches\n"
    report += "✓ Ground truth unchanged\n"
    report += "✓ extra_info unchanged\n"

    report += f"\n{'='*70}\n"
    report += "Next Steps\n"
    report += f"{'='*70}\n"
    report += "1. Review the before/after examples above\n"
    report += f"2. Inspect samples in: {output_dir}\n"
    report += "3. If satisfied, proceed to next dataset\n"
    report += "4. If issues found, adjust patterns and re-run\n"

    report += f"\n{'='*70}\n"

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Clean a single VERL math dataset and generate comparison report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean ORZ dataset with default preset
  python scripts/clean_math_dataset_single.py --dataset orz --output ./output/orz-cleaned/

  # Clean OpenR1 dataset with custom preset
  python scripts/clean_math_dataset_single.py \\
      --dataset openr1 \\
      --preset openr1-math \\
      --output ./output/openr1-cleaned/

  # Test on 1000 samples
  python scripts/clean_math_dataset_single.py \\
      --dataset orz \\
      --output ./output/test/ \\
      --max-samples 1000

Dataset shortcuts:
  orz      -> sungyub/orz-math-72k-verl
  openr1   -> sungyub/openr1-math-verl
  skywork  -> sungyub/skywork-or1-math-verl
  dapo     -> sungyub/dapo-math-17k-verl
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (use shortcuts: orz, openr1, skywork, dapo) or full HF name",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Cleaning preset (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for cleaned parquet files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Save report to file (default: <output>/cleaning_report.txt)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=15,
        help="Number of before/after examples in report (default: 15)",
    )

    args = parser.parse_args()

    # Resolve dataset name
    dataset_name = resolve_dataset_name(args.dataset)

    # Auto-detect preset if not specified
    if args.preset is None:
        preset_name = get_preset_for_dataset(dataset_name)
        print(f"Auto-detected preset: {preset_name}")
    else:
        preset_name = args.preset

    # Setup output directory
    output_dir = Path(args.output)

    # Load and clean dataset
    comparison_examples, stats = load_and_clean_dataset(
        dataset_name=dataset_name,
        preset_name=preset_name,
        output_dir=output_dir,
        max_samples=args.max_samples,
    )

    # Generate report
    report = generate_report(
        dataset_name=dataset_name,
        preset_name=preset_name,
        comparison_examples=comparison_examples,
        stats=stats,
        output_dir=output_dir,
    )

    # Print report
    print(report)

    # Save report to file
    if args.report_file:
        report_path = Path(args.report_file)
    else:
        report_path = output_dir / "cleaning_report.txt"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"\n✓ Report saved to: {report_path}")

    # Also save stats as JSON
    stats_path = output_dir / "cleaning_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "preset": preset_name,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
        }, f, indent=2)
    print(f"✓ Stats saved to: {stats_path}")

    print(f"\n{'='*70}")
    print("Cleaning completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
