#!/usr/bin/env python3
"""
Download VERL datasets from HuggingFace Hub.

This script downloads datasets and organizes them in the format expected
by the deduplication pipeline: ./{dataset-name}/data/*.parquet
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset
import pyarrow.parquet as pq


# Dataset configurations
QA_DATASETS = [
    'sungyub/docqa-rl-verl',
    'sungyub/guru-logic-verl',
    'sungyub/toolrl-4k-verl',
    'sungyub/guru-table-verl',
    'sungyub/table-r1-zero-verl',
]

IF_DATASETS = [
    'sungyub/ifeval-rlvr-verl',
    'sungyub/ifbench-verl',
]


def download_dataset(
    repo_id: str,
    output_dir: str,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Download a single dataset from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "sungyub/docqa-rl-verl")
        output_dir: Root directory for output
        verbose: Print detailed progress

    Returns:
        Dictionary with download statistics
    """
    # Extract dataset name from repo_id
    dataset_name = repo_id.split('/')[-1]

    print(f"\n{'=' * 70}")
    print(f"Downloading: {repo_id}")
    print(f"{'=' * 70}\n")

    # Create output directory
    data_dir = os.path.join(output_dir, dataset_name, 'data')
    os.makedirs(data_dir, exist_ok=True)

    try:
        # Load dataset from Hub
        print("Loading dataset from Hub...")
        dataset = load_dataset(repo_id)

        if verbose:
            print(f"Dataset loaded: {dataset}")
            print(f"Splits: {list(dataset.keys())}")

        # Combine all splits into a single dataset
        all_examples = []
        split_info = {}

        for split_name in dataset.keys():
            split_data = dataset[split_name]
            split_size = len(split_data)
            split_info[split_name] = split_size

            if verbose:
                print(f"  - {split_name}: {split_size:,} examples")

            # Add split information to extra_info if not already present
            for example in split_data:
                if 'extra_info' in example and isinstance(example['extra_info'], dict):
                    if 'split' not in example['extra_info']:
                        example['extra_info']['split'] = split_name
                all_examples.append(example)

        total_examples = len(all_examples)
        print(f"\nTotal examples: {total_examples:,}")

        # Convert to Parquet
        print(f"Saving to parquet...")
        from datasets import Dataset
        combined_dataset = Dataset.from_list(all_examples)

        # Determine if we need to split the file
        ROWS_PER_FILE = 500000
        if total_examples > ROWS_PER_FILE:
            # Split into multiple files
            num_files = (total_examples + ROWS_PER_FILE - 1) // ROWS_PER_FILE

            for file_idx in range(num_files):
                start_idx = file_idx * ROWS_PER_FILE
                end_idx = min(start_idx + ROWS_PER_FILE, total_examples)

                file_name = f"train-{file_idx:05d}-of-{num_files:05d}.parquet"
                file_path = os.path.join(data_dir, file_name)

                # Slice and save
                slice_dataset = combined_dataset.select(range(start_idx, end_idx))
                slice_dataset.to_parquet(file_path)

                if verbose:
                    print(f"  Saved: {file_name} ({end_idx - start_idx:,} rows)")
        else:
            # Single file
            file_name = "train-00000.parquet"
            file_path = os.path.join(data_dir, file_name)
            combined_dataset.to_parquet(file_path)

            if verbose:
                print(f"  Saved: {file_name} ({total_examples:,} rows)")

        print(f"\n✅ Download complete!")
        print(f"  Location: {data_dir}")
        print(f"  Total rows: {total_examples:,}")

        return {
            'dataset': dataset_name,
            'repo_id': repo_id,
            'total_rows': total_examples,
            'splits': split_info,
            'output_dir': data_dir
        }

    except Exception as e:
        print(f"\n❌ Error downloading {repo_id}: {e}")
        import traceback
        traceback.print_exc()
        raise


def download_collection(
    datasets: List[str],
    output_dir: str,
    verbose: bool = False
) -> List[Dict[str, int]]:
    """
    Download multiple datasets.

    Args:
        datasets: List of HuggingFace repository IDs
        output_dir: Root directory for output
        verbose: Print detailed progress

    Returns:
        List of download statistics dictionaries
    """
    results = []

    print(f"\n{'=' * 70}")
    print(f"Downloading {len(datasets)} datasets")
    print(f"{'=' * 70}\n")

    print("Datasets to download:")
    for i, repo_id in enumerate(datasets, 1):
        print(f"  {i}. {repo_id}")
    print()

    for repo_id in datasets:
        try:
            stats = download_dataset(repo_id, output_dir, verbose)
            results.append(stats)
        except Exception as e:
            print(f"⚠️  Skipping {repo_id} due to error: {e}")
            continue

    # Print summary
    print(f"\n{'=' * 70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 70}\n")

    total_rows = sum(s['total_rows'] for s in results)
    print(f"Datasets downloaded: {len(results)}/{len(datasets)}")
    print(f"Total examples: {total_rows:,}\n")

    for stats in results:
        print(f"  - {stats['dataset']}: {stats['total_rows']:,} rows")

    print(f"\n{'=' * 70}\n")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download VERL datasets from HuggingFace Hub',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all QA datasets
  python download_datasets.py --collection qa

  # Download all IF datasets
  python download_datasets.py --collection if

  # Download specific dataset
  python download_datasets.py --dataset sungyub/docqa-rl-verl

  # Download with custom output directory
  python download_datasets.py --collection qa --output-dir ~/data/qa
        """
    )

    parser.add_argument(
        '--collection',
        type=str,
        choices=['qa', 'if'],
        help='Collection to download (qa or if)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Single dataset to download (e.g., sungyub/docqa-rl-verl)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory (default: current directory)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.collection and not args.dataset:
        parser.error("Must specify either --collection or --dataset")

    if args.collection and args.dataset:
        parser.error("Cannot specify both --collection and --dataset")

    # Download datasets
    try:
        if args.collection:
            datasets = QA_DATASETS if args.collection == 'qa' else IF_DATASETS
            download_collection(datasets, args.output_dir, args.verbose)
        else:
            download_dataset(args.dataset, args.output_dir, args.verbose)

        print("\n✅ All downloads completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
