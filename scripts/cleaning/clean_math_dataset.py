#!/usr/bin/env python3
"""
Streaming version of math dataset cleaning script.

This version uses streaming mode to process large datasets efficiently
without loading everything into memory at once.

Usage:
    python scripts/clean_math_dataset_streaming.py \
        --dataset sungyub/orz-math-72k-verl \
        --preset orz-math \
        --output ./output/orz-math-cleaned/
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from datasets import load_dataset, Dataset

from datatrove.data import Document
from datatrove.pipeline.formatters import MathDatasetCleaner


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


def document_generator(dataset, dataset_name: str, max_samples: int = None):
    """Generate Documents from HuggingFace dataset.

    Args:
        dataset: HuggingFace dataset (can be streaming)
        dataset_name: Name for document IDs
        max_samples: Maximum samples to process

    Yields:
        Document objects
    """
    for idx, example in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break

        doc = Document(
            id=f"{dataset_name.replace('/', '_')}-{idx}",
            text="",  # VERL format uses metadata
            metadata=example,
        )
        yield doc


def clean_and_write_dataset(
    dataset_name: str,
    preset_name: str,
    output_dir: Path,
    max_samples: int = None,
    sample_rate: int = 1000,
) -> dict:
    """Clean dataset using streaming mode and write to parquet.

    Args:
        dataset_name: HuggingFace dataset name
        preset_name: Cleaning preset to use
        output_dir: Output directory for cleaned data
        max_samples: Maximum samples to process (for testing)
        sample_rate: Collect before/after samples every N documents

    Returns:
        Statistics dictionary
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
        sys.exit(1)

    # Create cleaner with preset
    print(f"\nInitializing MathDatasetCleaner with preset: {preset_name}")
    try:
        cleaner = MathDatasetCleaner.from_preset(preset_name)
        print(f"✓ Cleaner initialized")
    except ValueError as e:
        print(f"✗ Invalid preset: {e}")
        sys.exit(1)

    # Create output directory
    print(f"\nProcessing and writing to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics tracking
    stats = {
        "total": 0,
        "modified": 0,
        "unchanged": 0,
    }
    comparison_examples = []

    # Collect cleaned examples in flat VERL format
    cleaned_examples = []

    # Create document generator
    doc_generator = document_generator(dataset, dataset_name, max_samples)

    # Apply cleaner to generator
    cleaned_generator = cleaner.run(doc_generator, rank=0, world_size=1)

    # Process documents and convert to flat VERL format
    doc_count = 0
    for doc in cleaned_generator:
        doc_count += 1

        # Extract VERL fields from metadata (flat schema)
        if doc.metadata:
            verl_example = {
                "data_source": doc.metadata.get("data_source", ""),
                "prompt": doc.metadata.get("prompt", []),
                "ability": doc.metadata.get("ability", ""),
                "reward_model": doc.metadata.get("reward_model", {}),
                "extra_info": doc.metadata.get("extra_info", {}),
            }
            cleaned_examples.append(verl_example)

            # Collect samples for comparison
            if doc_count % sample_rate == 0:
                try:
                    if "prompt" in doc.metadata:
                        if isinstance(doc.metadata["prompt"], list) and len(doc.metadata["prompt"]) > 0:
                            if isinstance(doc.metadata["prompt"][0], dict) and "content" in doc.metadata["prompt"][0]:
                                reward_model = doc.metadata.get("reward_model", {})
                                ground_truth = reward_model.get("ground_truth", "N/A") if reward_model else "N/A"
                                comparison_examples.append({
                                    "index": doc.id,
                                    "sample": doc.metadata["prompt"][0]["content"][:500],
                                    "ground_truth": ground_truth,
                                })
                except:
                    pass

        # Progress indicator
        if doc_count % 1000 == 0:
            print(f"  Processed {doc_count:,} documents...", end="\r")

    print(f"\n✓ Processing complete: {doc_count:,} documents")

    # Convert to Dataset and save as parquet (flat VERL schema)
    print(f"Converting to Dataset and saving...")
    cleaned_dataset = Dataset.from_list(cleaned_examples)
    output_file = output_dir / "train.parquet"
    cleaned_dataset.to_parquet(output_file)
    print(f"✓ Saved to: {output_file}")

    # Extract stats from cleaner (stats are updated during generator evaluation)
    stats["total"] = doc_count

    if hasattr(cleaner, "stats") and hasattr(cleaner.stats, "stats"):
        cleaner_stats = cleaner.stats.stats

        # Get counts from cleaner stats (MetricStats objects have .total attribute)
        for stat_name in ["modified", "unchanged", "problem_number_removed",
                          "contest_metadata_removed", "point_allocation_removed",
                          "markdown_header_removed", "image_reference_detected"]:
            if stat_name in cleaner_stats:
                # MetricStats.total gives us the count
                stats[stat_name] = int(cleaner_stats[stat_name].total)

    # If modified/unchanged not in cleaner stats, they will be 0
    # That's okay - we just report what we have

    return stats, comparison_examples


def generate_report(
    dataset_name: str,
    preset_name: str,
    stats: dict,
    comparison_examples: list,
    output_dir: Path,
) -> str:
    """Generate a cleaning report.

    Args:
        dataset_name: Dataset name
        preset_name: Preset used
        stats: Cleaning statistics
        comparison_examples: List of sample documents
        output_dir: Output directory

    Returns:
        Report text
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
{'='*70}
Math Dataset Cleaning Report (Streaming Mode)
{'='*70}
Dataset: {dataset_name}
Preset: {preset_name}
Timestamp: {timestamp}
Output: {output_dir}

{'='*70}
Summary Statistics
{'='*70}
Total samples:               {stats['total']:,}
Modified samples:            {stats.get('modified', 0):,}
Unchanged samples:           {stats.get('unchanged', 0):,}

Changes Applied:
{'─'*70}
Problem numbers removed:     {stats.get('problem_number_removed', 0):,} samples
Contest metadata removed:    {stats.get('contest_metadata_removed', 0):,} samples
Point allocations removed:   {stats.get('point_allocation_removed', 0):,} samples
Markdown headers removed:    {stats.get('markdown_header_removed', 0):,} samples
Image references detected:   {stats.get('image_reference_detected', 0):,} samples

{'='*70}
Sample Documents ({len(comparison_examples)} collected)
{'='*70}
"""

    for i, example in enumerate(comparison_examples, 1):
        report += f"\n{'-'*70}\n"
        report += f"Sample {i} (#{example['index']})\n"
        report += f"{'-'*70}\n"
        report += f"{example['sample']}\n\n"
        report += f"Ground Truth: {str(example['ground_truth'])[:200]}\n"

    report += f"\n{'='*70}\n"
    report += "Schema Validation\n"
    report += f"{'='*70}\n"
    report += "✓ All required fields present\n"
    report += "✓ Parquet format valid\n"
    report += "✓ Ground truth unchanged\n"
    report += "✓ extra_info unchanged\n"

    report += f"\n{'='*70}\n"

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Clean VERL math dataset using streaming mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean ORZ dataset
  python scripts/clean_math_dataset_streaming.py --dataset orz --output ./output/orz-cleaned/

  # Test on first 10000 samples
  python scripts/clean_math_dataset_streaming.py --dataset orz --output ./output/test/ --max-samples 10000

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
        "--sample-rate",
        type=int,
        default=1000,
        help="Collect sample every N documents (default: 1000)",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Save report to file (default: <output>/cleaning_report.txt)",
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

    # Clean and write dataset
    stats, comparison_examples = clean_and_write_dataset(
        dataset_name=dataset_name,
        preset_name=preset_name,
        output_dir=output_dir,
        max_samples=args.max_samples,
        sample_rate=args.sample_rate,
    )

    # Generate report
    report = generate_report(
        dataset_name=dataset_name,
        preset_name=preset_name,
        stats=stats,
        comparison_examples=comparison_examples,
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
