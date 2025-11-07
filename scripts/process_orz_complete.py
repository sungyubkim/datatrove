#!/usr/bin/env python3
"""
Complete processing pipeline for ORZ Math 72K dataset:
1. Load original dataset
2. Convert to VERL format
3. Apply cleaning
4. Perform intra-dataset deduplication
5. Save final dataset

Usage:
    python scripts/process_orz_complete.py --output ./output/orz-math-final/
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from datatrove.data import Document
from datatrove.pipeline.formatters import MathDatasetCleaner
from datatrove.pipeline.writers import ParquetWriter


def convert_to_verl_format(original_dataset, dataset_name="orz-math-72k"):
    """Convert original ORZ format to VERL format.

    Args:
        original_dataset: HuggingFace dataset
        dataset_name: Name for data_source field

    Yields:
        Document objects in VERL format
    """
    print("\n" + "="*70)
    print("Step 1: Converting to VERL format")
    print("="*70)

    for idx, example in enumerate(tqdm(original_dataset, desc="Converting")):
        # Extract data from original format
        problem_text = example['0']['value']
        ground_truth = example['1']['ground_truth']['value']

        # Create VERL-formatted metadata
        verl_metadata = {
            'data_source': dataset_name,
            'prompt': [
                {
                    'role': 'user',
                    'content': problem_text
                }
            ],
            'ability': 'math',
            'reward_model': {
                'style': 'rule',
                'ground_truth': ground_truth
            },
            'extra_info': {
                'split': 'train',
                'index': idx
            }
        }

        # Create Document
        doc = Document(
            id=f"{dataset_name.replace('-', '_')}-{idx}",
            text="",  # VERL format uses metadata
            metadata=verl_metadata
        )

        yield doc


def clean_documents(doc_generator, preset_name="orz-math"):
    """Apply cleaning to documents.

    Args:
        doc_generator: Generator yielding Document objects
        preset_name: Cleaning preset to use

    Yields:
        Cleaned Document objects
    """
    print("\n" + "="*70)
    print(f"Step 2: Applying cleaning (preset: {preset_name})")
    print("="*70)

    # Create cleaner
    cleaner = MathDatasetCleaner.from_preset(preset_name)

    # Apply cleaning
    for doc in cleaner.run(doc_generator, rank=0, world_size=1):
        yield doc


def deduplicate_documents(doc_generator):
    """Perform intra-dataset deduplication using exact hash matching.

    Args:
        doc_generator: Generator yielding Document objects

    Yields:
        Deduplicated Document objects
    """
    print("\n" + "="*70)
    print("Step 3: Performing intra-dataset deduplication")
    print("="*70)

    seen_hashes = set()
    total_docs = 0
    duplicate_docs = 0

    for doc in tqdm(doc_generator, desc="Deduplicating"):
        total_docs += 1

        # Extract problem text for hashing
        if doc.metadata and 'prompt' in doc.metadata:
            problem_text = doc.metadata['prompt'][0]['content']

            # Create hash of problem text
            text_hash = hashlib.sha256(problem_text.encode('utf-8')).hexdigest()

            if text_hash in seen_hashes:
                duplicate_docs += 1
                continue  # Skip duplicate

            seen_hashes.add(text_hash)
            yield doc
        else:
            # If metadata is malformed, keep the document
            yield doc

    print(f"\n✓ Deduplication complete:")
    print(f"  Total documents: {total_docs:,}")
    print(f"  Duplicates removed: {duplicate_docs:,}")
    print(f"  Unique documents: {total_docs - duplicate_docs:,}")
    print(f"  Deduplication rate: {duplicate_docs/total_docs*100:.2f}%")


def process_orz_dataset(output_dir: Path):
    """Complete processing pipeline for ORZ Math dataset.

    Args:
        output_dir: Output directory for processed dataset
    """
    print("\n" + "="*70)
    print("ORZ Math 72K - Complete Processing Pipeline")
    print("="*70)
    print(f"Output directory: {output_dir}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load original dataset
    print("="*70)
    print("Loading original dataset")
    print("="*70)
    original_ds = load_dataset(
        'Open-Reasoner-Zero/orz_math_72k_collection_extended',
        split='train'
    )
    print(f"✓ Loaded {len(original_ds):,} examples")

    # Step 2: Convert to VERL format
    verl_docs = convert_to_verl_format(original_ds)

    # Step 3: Apply cleaning
    cleaned_docs = clean_documents(verl_docs)

    # Step 4: Deduplicate
    deduped_docs = deduplicate_documents(cleaned_docs)

    # Step 5: Write to parquet
    print("\n" + "="*70)
    print("Step 4: Writing to Parquet")
    print("="*70)

    writer = ParquetWriter(
        output_folder=str(output_dir),
        output_filename="${rank}.parquet"
    )

    doc_count = 0
    for doc in writer.run(deduped_docs, rank=0, world_size=1):
        doc_count += 1

    print(f"\n✓ Processing complete: {doc_count:,} documents written")

    # Save processing stats
    stats = {
        "source_dataset": "Open-Reasoner-Zero/orz_math_72k_collection_extended",
        "original_count": len(original_ds),
        "final_count": doc_count,
        "steps": [
            "VERL format conversion",
            "Cleaning (orz-math preset)",
            "Intra-dataset deduplication"
        ]
    }

    stats_path = output_dir / "processing_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Stats saved to: {stats_path}")
    print("\n" + "="*70)
    print("All steps completed successfully!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Complete processing pipeline for ORZ Math 72K dataset"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed dataset"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    process_orz_dataset(output_dir)


if __name__ == "__main__":
    main()
