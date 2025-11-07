"""Batch process RLVR-IFeval dataset to IFBench-VERL format.

This script transforms the entire RLVR-IFeval dataset from HuggingFace
into the IFBench-VERL format for use with datatrove's IFEval scorer.

Usage:
    # Process train split locally
    python scripts/process_rlvr_to_ifbench.py

    # Process with custom output path
    python scripts/process_rlvr_to_ifbench.py --output output/custom-path

    # Process specific split
    python scripts/process_rlvr_to_ifbench.py --split validation
"""

import argparse
import json
from pathlib import Path

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.writers import JsonlWriter


def process_rlvr_to_ifbench(
    output_dir: str = "output/ifbench-rlvr-verl",
    split: str = "train",
    logging_dir: str = ".datatrove_logs/rlvr_processing",
    tasks: int = 1,
):
    """Process RLVR dataset to IFBench format.

    Args:
        output_dir: Output directory for processed dataset
        split: Dataset split to process (e.g., "train", "validation")
        logging_dir: Directory for processing logs
        tasks: Number of parallel tasks
    """
    from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench
    from datasets import load_dataset

    # Load RLVR dataset
    print(f"Loading RLVR-IFeval dataset (split: {split})...")
    dataset = load_dataset("allenai/RLVR-IFeval", split=split)
    print(f"Loaded {len(dataset)} examples")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process dataset
    print(f"Processing dataset to IFBench-VERL format...")
    transformed_examples = []

    for idx, example in enumerate(dataset):
        try:
            # Construct RLVR example from dataset row
            rlvr_example = {
                "messages": example["messages"],
                "ground_truth": example["ground_truth"],
                "dataset": example.get("dataset", "ifeval"),
            }

            # Transform to IFBench format
            ifbench_example = transform_to_ifbench(rlvr_example, index=idx)
            transformed_examples.append(ifbench_example)

            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(dataset)} examples")

        except Exception as e:
            print(f"  Warning: Failed to transform example {idx}: {e}")
            continue

    print(f"Successfully transformed {len(transformed_examples)}/{len(dataset)} examples")

    # Save to JSONL format (one example per line)
    output_file = output_path / f"{split}.jsonl"
    print(f"Writing to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for example in transformed_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(transformed_examples)} examples to {output_file}")

    # Save stats
    stats_file = output_path / f"{split}_stats.json"
    stats = {
        "split": split,
        "total_examples": len(dataset),
        "transformed_examples": len(transformed_examples),
        "failed_examples": len(dataset) - len(transformed_examples),
    }

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Saved stats to {stats_file}")
    print("\nProcessing complete!")

    return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process RLVR-IFeval dataset to IFBench-VERL format"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/ifbench-rlvr-verl",
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--logging-dir",
        type=str,
        default=".datatrove_logs/rlvr_processing",
        help="Directory for processing logs",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=1,
        help="Number of parallel tasks (default: 1)",
    )

    args = parser.parse_args()

    process_rlvr_to_ifbench(
        output_dir=args.output,
        split=args.split,
        logging_dir=args.logging_dir,
        tasks=args.tasks,
    )


if __name__ == "__main__":
    main()
