#!/usr/bin/env python3
"""
Clean ORZ Math dataset while preserving flat VERL schema.

This script ensures the output maintains the exact schema structure:
- data_source: string
- prompt: list<struct<content, role>>
- ability: string
- reward_model: struct<ground_truth, style>
- extra_info: struct<index, split>

Usage:
    python scripts/clean_orz_flat_schema.py --output ./output/orz-flat-cleaned/
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

from datatrove.data import Document
from datatrove.pipeline.formatters import MathDatasetCleaner


def clean_and_deduplicate(
    dataset_name: str = "Open-Reasoner-Zero/orz_math_72k_collection_extended",
    preset_name: str = "orz-math",
    output_dir: Path = Path("./output/orz-final"),
):
    """
    Clean and deduplicate dataset while preserving flat VERL schema.

    Args:
        dataset_name: HuggingFace dataset to load
        preset_name: MathDatasetCleaner preset
        output_dir: Output directory
    """
    print(f"\n{'='*70}")
    print("ORZ Math Dataset - Clean & Deduplicate (Flat VERL Schema)")
    print(f"{'='*70}\n")

    # Load original dataset
    print(f"Loading original dataset: {dataset_name}")
    original_dataset = load_dataset(dataset_name, split="train")
    print(f"✓ Loaded {len(original_dataset):,} samples\n")

    # Convert to VERL format
    print("Converting to VERL format...")
    verl_samples = []
    for idx, example in enumerate(tqdm(original_dataset, desc="Converting")):
        problem_text = example['0']['value']
        ground_truth = example['1']['ground_truth']['value']

        verl_sample = {
            "data_source": "orz-math-72k",
            "prompt": [{"role": "user", "content": problem_text}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {"split": "train", "index": idx},
        }
        verl_samples.append(verl_sample)

    print(f"✓ Converted {len(verl_samples):,} samples to VERL format\n")

    # Now use verl_samples as the dataset
    dataset = verl_samples

    # Initialize cleaner
    print(f"Initializing cleaner with preset: {preset_name}")
    cleaner = MathDatasetCleaner.from_preset(preset_name)
    print(f"✓ Cleaner ready\n")

    # Process samples
    print("Step 1: Cleaning problem text...")
    print("="*70)

    cleaned_samples = []
    seen_hashes = set()
    duplicate_count = 0
    cleaning_stats = {
        "total": 0,
        "modified": 0,
        "duplicates": 0,
    }

    for idx, example in enumerate(tqdm(dataset, desc="Processing")):
        # Create Document for cleaning
        doc = Document(
            id=f"orz-{idx}",
            text="",
            metadata=example
        )

        # Apply cleaning
        cleaned_doc = None
        for cleaned in cleaner.run([doc], rank=0, world_size=1):
            cleaned_doc = cleaned
            break

        if cleaned_doc is None:
            continue

        # Extract cleaned problem text for deduplication
        problem_text = cleaned_doc.metadata["prompt"][0]["content"]
        problem_hash = hashlib.sha256(problem_text.encode('utf-8')).hexdigest()

        # Check for duplicates
        if problem_hash in seen_hashes:
            duplicate_count += 1
            cleaning_stats["duplicates"] += 1
            continue

        seen_hashes.add(problem_hash)

        # Extract metadata in flat VERL format (NO nested structure!)
        cleaned_sample = {
            "data_source": cleaned_doc.metadata["data_source"],
            "prompt": cleaned_doc.metadata["prompt"],
            "ability": cleaned_doc.metadata["ability"],
            "reward_model": cleaned_doc.metadata["reward_model"],
            "extra_info": cleaned_doc.metadata["extra_info"],
        }

        cleaned_samples.append(cleaned_sample)
        cleaning_stats["total"] += 1

    print(f"\n✓ Cleaning complete:")
    print(f"  Total processed: {len(dataset):,}")
    print(f"  Duplicates removed: {duplicate_count:,} ({duplicate_count/len(dataset)*100:.2f}%)")
    print(f"  Unique samples: {len(cleaned_samples):,}\n")

    # Get cleaning statistics from cleaner
    if hasattr(cleaner, "stats") and hasattr(cleaner.stats, "stats"):
        cleaner_stats = cleaner.stats.stats
        for stat_name in ["modified", "unchanged", "problem_number_removed",
                          "contest_metadata_removed", "point_allocation_removed",
                          "markdown_header_removed"]:
            if stat_name in cleaner_stats:
                cleaning_stats[stat_name] = int(cleaner_stats[stat_name].total)

    # Save as Parquet with flat schema
    print("Step 2: Saving to Parquet...")
    print("="*70)

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Convert to pandas DataFrame
    df = pd.DataFrame(cleaned_samples)

    # Define explicit PyArrow schema (flat VERL format)
    schema = pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([
            ("content", pa.string()),
            ("role", pa.string()),
        ]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([
            ("ground_truth", pa.string()),
            ("style", pa.string()),
        ])),
        ("extra_info", pa.struct([
            ("index", pa.int64()),
            ("split", pa.string()),
        ])),
    ])

    # Convert to PyArrow Table with explicit schema
    table = pa.Table.from_pandas(df, schema=schema)

    # Write to Parquet
    output_file = data_dir / "train-00000.parquet"
    pq.write_table(table, output_file)

    print(f"✓ Saved to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB\n")

    # Verify schema
    print("Step 3: Verifying schema...")
    print("="*70)

    verify_table = pq.read_table(output_file)
    print(f"✓ Schema verification:")
    print(f"  Columns: {verify_table.column_names}")
    print(f"  Samples: {verify_table.num_rows:,}")
    print()
    print("Schema details:")
    print(verify_table.schema)
    print()

    # Save statistics
    stats = {
        "dataset": dataset_name,
        "preset": preset_name,
        "original_samples": len(dataset),
        "final_samples": len(cleaned_samples),
        "duplicates_removed": duplicate_count,
        "duplicate_rate": duplicate_count / len(dataset),
        "cleaning_stats": cleaning_stats,
    }

    stats_file = output_dir / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Stats saved to: {stats_file}\n")

    print("="*70)
    print("✅ Processing complete!")
    print("="*70)
    print(f"Final dataset: {len(cleaned_samples):,} samples")
    print(f"Schema: Flat VERL (data_source, prompt, ability, reward_model, extra_info)")
    print(f"Output: {data_dir}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Clean and deduplicate ORZ Math dataset with flat VERL schema"
    )
    parser.add_argument(
        "--dataset",
        default="sungyub/orz-math-72k-verl",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--preset",
        default="orz-math",
        help="Cleaning preset"
    )
    parser.add_argument(
        "--output",
        default="./output/orz-final",
        help="Output directory"
    )

    args = parser.parse_args()

    clean_and_deduplicate(
        dataset_name=args.dataset,
        preset_name=args.preset,
        output_dir=Path(args.output)
    )


if __name__ == "__main__":
    main()
