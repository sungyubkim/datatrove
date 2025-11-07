#!/usr/bin/env python3
"""
Upload cleaned and deduplicated ORZ Math dataset to HuggingFace Hub.

Usage:
    python scripts/upload_orz_to_hub.py --dataset-dir ./output/orz-math-final
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def upload_dataset_to_hub(
    dataset_dir: str,
    repo_id: str = "sungyub/orz-math-72k-verl",
    repo_type: str = "dataset",
    private: bool = False
):
    """
    Upload dataset files to HuggingFace Hub.

    Args:
        dataset_dir: Local directory containing the dataset
        repo_id: HuggingFace repository ID
        repo_type: Type of repository ('dataset' or 'model')
        private: Whether to create private repository
    """
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    # Initialize HF API
    api = HfApi()

    print(f"\n{'='*70}")
    print(f"Uploading to HuggingFace Hub")
    print(f"{'='*70}")
    print(f"Repository: {repo_id}")
    print(f"Local path: {dataset_path}")
    print(f"{'='*70}\n")

    # Create repository if it doesn't exist (or update if it does)
    try:
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True  # Don't error if repo already exists
        )
        print(f"✓ Repository ready: {repo_id}\n")
    except Exception as e:
        print(f"Note: Repository may already exist: {e}\n")

    # Upload files
    files_to_upload = [
        ("README.md", "README.md"),
        ("stats.json", "stats.json"),
    ]

    # Upload individual files
    for local_file, path_in_repo in files_to_upload:
        local_path = dataset_path / local_file
        if local_path.exists():
            print(f"Uploading {local_file}...")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            print(f"✓ Uploaded {local_file}\n")
        else:
            print(f"⚠️  Warning: {local_file} not found, skipping\n")

    # Upload entire data directory
    data_dir = dataset_path / "data"
    if data_dir.exists():
        print(f"Uploading data directory...")
        api.upload_folder(
            folder_path=str(data_dir),
            path_in_repo="data",
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✓ Uploaded data directory\n")
    else:
        print(f"⚠️  Warning: data directory not found\n")

    print(f"{'='*70}")
    print(f"✅ Upload complete!")
    print(f"{'='*70}")
    print(f"View at: https://huggingface.co/datasets/{repo_id}")
    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload cleaned and deduplicated ORZ Math dataset to HuggingFace Hub"
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./output/orz-math-final",
        help="Local directory containing the dataset (default: ./output/orz-math-final)"
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default="sungyub/orz-math-72k-verl",
        help="HuggingFace repository ID (default: sungyub/orz-math-72k-verl)"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository"
    )

    args = parser.parse_args()

    upload_dataset_to_hub(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        private=args.private
    )


if __name__ == "__main__":
    main()
