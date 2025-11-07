"""Upload RLVR-IFeval-VERL dataset to HuggingFace Hub.

This script uploads the transformed dataset to HuggingFace Hub in Parquet format
with comprehensive documentation.

Usage:
    # Upload to default repository
    python scripts/upload_to_hub.py --token $HF_TOKEN

    # Upload to custom repository
    python scripts/upload_to_hub.py \
        --repo-id your-username/your-dataset \
        --token $HF_TOKEN

    # Upload as private dataset
    python scripts/upload_to_hub.py \
        --repo-id sungyub/ifeval-rlvr-verl \
        --token $HF_TOKEN \
        --private
"""

import argparse
import os
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, create_repo


def upload_dataset(
    repo_id: str,
    data_dir: str = "output/ifbench-rlvr-verl",
    token: str = None,
    private: bool = False,
):
    """Upload dataset to HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "sungyub/ifeval-rlvr-verl")
        data_dir: Directory containing Parquet and README files
        token: HuggingFace authentication token
        private: Whether to create a private repository

    Returns:
        URL of the uploaded dataset
    """
    data_dir = Path(data_dir)
    parquet_path = data_dir / "train.parquet"
    readme_path = data_dir / "README.md"

    # Validate files exist
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    if not readme_path.exists():
        raise FileNotFoundError(f"README file not found: {readme_path}")

    print(f"Uploading dataset to: {repo_id}")
    print(f"  Parquet: {parquet_path}")
    print(f"  README: {readme_path}")
    print(f"  Private: {private}")

    # Create repository
    print("\n1. Creating repository...")
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        print(f"   ✓ Repository created/exists: {repo_url}")
    except Exception as e:
        print(f"   ✗ Failed to create repository: {e}")
        raise

    # Load dataset from Parquet
    print("\n2. Loading dataset from Parquet...")
    try:
        dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
        print(f"   ✓ Loaded {len(dataset)} examples")
        print(f"   ✓ Columns: {dataset.column_names}")
    except Exception as e:
        print(f"   ✗ Failed to load dataset: {e}")
        raise

    # Push dataset to Hub
    print("\n3. Pushing dataset to Hub...")
    try:
        dataset.push_to_hub(
            repo_id=repo_id,
            token=token,
            private=private,
            commit_message="Upload RLVR-IFeval-VERL dataset",
        )
        print("   ✓ Dataset pushed successfully")
    except Exception as e:
        print(f"   ✗ Failed to push dataset: {e}")
        raise

    # Upload README
    print("\n4. Uploading README...")
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Upload comprehensive README",
        )
        print("   ✓ README uploaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to upload README: {e}")
        raise

    # Verify upload
    print("\n5. Verifying upload...")
    try:
        # Try to load from Hub
        test_dataset = load_dataset(repo_id, split="train", streaming=True, token=token)
        first_example = next(iter(test_dataset))
        print(f"   ✓ Dataset verified on Hub")
        print(f"   ✓ First example keys: {list(first_example.keys())}")
    except Exception as e:
        print(f"   ⚠ Warning: Could not verify dataset (may need time to process): {e}")

    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"\n✓ Upload complete!")
    print(f"  Dataset URL: {dataset_url}")

    return dataset_url


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="sungyub/ifeval-rlvr-verl",
        help="HuggingFace repository ID (default: sungyub/ifeval-rlvr-verl)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output/ifbench-rlvr-verl",
        help="Directory containing Parquet and README files",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace authentication token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )

    args = parser.parse_args()

    # Get token from args or environment
    # If None, huggingface_hub will use cached token from huggingface-cli login
    token = args.token or os.environ.get("HF_TOKEN")

    upload_dataset(
        repo_id=args.repo_id,
        data_dir=args.data_dir,
        token=token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
