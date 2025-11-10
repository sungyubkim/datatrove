"""
Upload all README files to Hugging Face Hub.

This script uploads README files for:
- 9 individual math datasets
- 1 unified collection

Usage:
    python upload_readmes_to_hub.py --readme-dir output/readmes-unified
    python upload_readmes_to_hub.py --readme-dir output/readmes-unified --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from source_dataset_mapping import DATASET_IDS


class HubReadmeUploader:
    """Upload README files to Hugging Face Hub."""

    def __init__(self, readme_dir: Path, dry_run: bool = False):
        """Initialize uploader.

        Args:
            readme_dir: Directory containing dataset subdirectories with READMEs
            dry_run: If True, print commands without executing
        """
        self.readme_dir = Path(readme_dir)
        self.dry_run = dry_run
        self.results = []

    def upload_all(self):
        """Upload READMEs for all datasets."""
        # Upload individual datasets
        all_datasets = DATASET_IDS + ["math-verl-unified"]

        print(f"Uploading READMEs for {len(all_datasets)} datasets to Hugging Face Hub...")
        print("=" * 70)

        for dataset_id in all_datasets:
            try:
                self.upload_single_dataset(dataset_id)
                self.results.append({"dataset": dataset_id, "status": "success"})
            except Exception as e:
                print(f"✗ Error uploading {dataset_id}: {e}")
                self.results.append({"dataset": dataset_id, "status": "error", "error": str(e)})

        self.print_summary()

    def upload_single_dataset(self, dataset_id: str):
        """Upload README for a single dataset.

        Args:
            dataset_id: Dataset identifier
        """
        readme_path = self.readme_dir / dataset_id / "README.md"

        if not readme_path.exists():
            raise FileNotFoundError(f"README not found: {readme_path}")

        repo_id = f"sungyub/{dataset_id}"

        print(f"\n📤 Uploading: {dataset_id}")
        print("-" * 70)

        # Build huggingface-cli command
        cmd = [
            "huggingface-cli",
            "upload",
            repo_id,
            str(readme_path),
            "README.md",
            "--repo-type",
            "dataset",
            "--commit-message",
            f"Update README with unified format (v3.0)",
        ]

        if self.dry_run:
            print(f"  [DRY RUN] Would execute: {' '.join(cmd)}")
            print(f"  ✓ (Skipped in dry-run mode)")
        else:
            # Execute upload
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )

                print(f"  ✓ README uploaded successfully")
                print(f"    Repository: https://huggingface.co/datasets/{repo_id}")

                if result.stdout:
                    # Print relevant output (skip verbose logs)
                    for line in result.stdout.split('\n'):
                        if 'commit' in line.lower() or 'uploaded' in line.lower():
                            print(f"    {line.strip()}")

            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else str(e)
                raise RuntimeError(f"Upload failed: {error_msg}")

    def print_summary(self):
        """Print upload summary."""
        print("\n" + "=" * 70)
        print("UPLOAD SUMMARY")
        print("=" * 70)

        successful = [r for r in self.results if r["status"] == "success"]
        failed = [r for r in self.results if r["status"] == "error"]

        print(f"\n✓ Successful: {len(successful)}/{len(self.results)}")
        print(f"✗ Failed: {len(failed)}/{len(self.results)}")

        if failed:
            print("\nFailed uploads:")
            for result in failed:
                print(f"  - {result['dataset']}: {result['error']}")

        if self.dry_run:
            print("\n⚠ DRY RUN MODE - No files were actually uploaded")
        else:
            print(f"\n✓ All README files have been uploaded to Hugging Face Hub!")
            print("\nNext steps:")
            print("  1. Verify READMEs on Hugging Face website")
            print("  2. Check dataset cards render correctly")
            print("  3. Update any additional metadata if needed")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Upload README files to Hugging Face Hub")
    parser.add_argument(
        "--readme-dir",
        type=Path,
        required=True,
        help="Directory containing dataset subdirectories with READMEs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing (for testing)",
    )
    parser.add_argument(
        "--datasets",
        help="Comma-separated list of specific datasets to upload (default: all)",
    )

    args = parser.parse_args()

    if not args.readme_dir.exists():
        print(f"Error: README directory not found: {args.readme_dir}")
        sys.exit(1)

    # Create uploader
    uploader = HubReadmeUploader(readme_dir=args.readme_dir, dry_run=args.dry_run)

    if args.datasets:
        # Upload specific datasets
        dataset_list = [d.strip() for d in args.datasets.split(",")]
        print(f"Uploading {len(dataset_list)} specific datasets...")

        for dataset_id in dataset_list:
            try:
                uploader.upload_single_dataset(dataset_id)
                uploader.results.append({"dataset": dataset_id, "status": "success"})
            except Exception as e:
                print(f"✗ Error: {e}")
                uploader.results.append({"dataset": dataset_id, "status": "error", "error": str(e)})

        uploader.print_summary()
    else:
        # Upload all datasets
        uploader.upload_all()


if __name__ == "__main__":
    main()
