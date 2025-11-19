#!/usr/bin/env python3
"""
Upload DeepMath-103K datasets to Hugging Face Hub.

Requirements:
1. Must be logged in: huggingface-cli login
2. Must have write access to the datasets

Uploads:
- sungyub/deepmath-103k-verl (standalone dataset)
- sungyub/math-verl-unified (deepmath_103k_verl split)

Usage:
    # Dry run (test without uploading)
    python scripts/upload/upload_deepmath_to_hub.py --individual --dry-run
    python scripts/upload/upload_deepmath_to_hub.py --unified --dry-run

    # Actual upload
    python scripts/upload/upload_deepmath_to_hub.py --individual
    python scripts/upload/upload_deepmath_to_hub.py --unified
    python scripts/upload/upload_deepmath_to_hub.py --both
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def check_login():
    """Check if user is logged in to Hugging Face Hub."""
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Not logged in to Hugging Face Hub")
        print(f"   Please run: huggingface-cli login")
        return False


def upload_individual(dry_run=False):
    """
    Upload deepmath-103k-verl standalone dataset.

    Args:
        dry_run: If True, only show what would be uploaded without actually uploading
    """
    print("\n" + "=" * 70)
    print("Uploading: sungyub/deepmath-103k-verl (standalone)")
    print("=" * 70)

    repo_id = "sungyub/deepmath-103k-verl"
    local_dir = "output/hub-upload/deepmath-103k-verl"

    # Files to upload
    readme_path = os.path.join(local_dir, "README.md")
    data_path = os.path.join(local_dir, "data/train-00000.parquet")

    # Check files exist
    if not Path(readme_path).exists():
        print(f"❌ ERROR: README.md not found at {readme_path}")
        print(f"\nPlease run preparation scripts first:")
        print(f"  python scripts/hub/prepare_deepmath_standalone.py")
        print(f"  python scripts/hub/generate_deepmath_readme.py")
        return False

    if not Path(data_path).exists():
        print(f"❌ ERROR: Data file not found at {data_path}")
        print(f"\nPlease run preparation scripts first:")
        print(f"  python scripts/hub/prepare_deepmath_standalone.py")
        return False

    size_mb = os.path.getsize(data_path) / (1024 * 1024)

    if dry_run:
        print("\n[DRY RUN] Would upload the following files:")
        print(f"  README.md → README.md")
        print(f"  data/train-00000.parquet → data/train-00000.parquet ({size_mb:.2f} MB)")
        print(f"\n[DRY RUN] Commit message:")
        print(f"  'Initial upload: DeepMath-103K cleaned (101,844 samples)'")
        return True

    api = HfApi()

    try:
        # Create repo if it doesn't exist
        print("\n1. Creating repository if needed...")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                exist_ok=True,
                private=False,
            )
            print(f"   ✓ Repository ready: {repo_id}")
        except Exception as e:
            print(f"   ℹ Repository may already exist: {e}")

        # Upload README first
        print("\n2. Uploading README.md...")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add comprehensive README with DeepMath-103K metadata and cleaning statistics",
        )
        print("   ✓ README.md uploaded")

        # Upload data file
        print("\n3. Uploading data/train-00000.parquet...")
        print(f"   File size: {size_mb:.2f} MB")
        print(f"   This may take a few minutes...")

        api.upload_file(
            path_or_fileobj=data_path,
            path_in_repo="data/train-00000.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload DeepMath-103K cleaned dataset: 101,844 samples (1,147 duplicates removed, orz-math cleaning)",
        )
        print("   ✓ Data file uploaded")

        print(f"\n✅ Upload completed for {repo_id}")
        print(f"   View at: https://huggingface.co/datasets/{repo_id}")
        print(f"\nDataset ready to use:")
        print(f"  from datasets import load_dataset")
        print(f"  dataset = load_dataset('{repo_id}', split='train')")

        return True

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def upload_unified(dry_run=False):
    """
    Upload deepmath_103k_verl split to math-verl-unified.

    Args:
        dry_run: If True, only show what would be uploaded without actually uploading
    """
    print("\n" + "=" * 70)
    print("Uploading: sungyub/math-verl-unified (deepmath_103k_verl split)")
    print("=" * 70)

    repo_id = "sungyub/math-verl-unified"
    local_dir = "output/hub-upload/math-verl-unified"

    # Files to upload
    readme_path = os.path.join(local_dir, "README.md")
    data_path = os.path.join(local_dir, "data/deepmath_103k_verl.parquet")

    # Check files exist
    if not Path(readme_path).exists():
        print(f"❌ ERROR: README.md not found at {readme_path}")
        print(f"\nPlease run preparation scripts first:")
        print(f"  python scripts/hub/prepare_deepmath_unified.py")
        print(f"  python scripts/hub/update_unified_with_deepmath.py")
        return False

    if not Path(data_path).exists():
        print(f"❌ ERROR: Data file not found at {data_path}")
        print(f"\nPlease run preparation scripts first:")
        print(f"  python scripts/hub/prepare_deepmath_unified.py")
        return False

    size_mb = os.path.getsize(data_path) / (1024 * 1024)

    if dry_run:
        print("\n[DRY RUN] Would upload the following files:")
        print(f"  README.md → README.md")
        print(f"  data/deepmath_103k_verl.parquet → data/deepmath_103k_verl.parquet ({size_mb:.2f} MB)")
        print(f"\n[DRY RUN] Commit message:")
        print(f"  'Add deepmath_103k_verl split: 101,844 samples (9th dataset)'")
        return True

    api = HfApi()

    try:
        # Upload README first
        print("\n1. Uploading README.md...")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add deepmath_103k_verl split to unified dataset (101,844 samples)",
        )
        print("   ✓ README.md uploaded")

        # Upload data file
        print("\n2. Uploading data/deepmath_103k_verl.parquet...")
        print(f"   File size: {size_mb:.2f} MB")
        print(f"   This may take a few minutes...")

        api.upload_file(
            path_or_fileobj=data_path,
            path_in_repo="data/deepmath_103k_verl.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add deepmath_103k_verl split: 101,844 challenging math problems (cleaned & deduplicated)",
        )
        print("   ✓ Data file uploaded")

        print(f"\n✅ Upload completed for {repo_id}")
        print(f"   View at: https://huggingface.co/datasets/{repo_id}")
        print(f"\nNew split ready to use:")
        print(f"  from datasets import load_dataset")
        print(f"  dataset = load_dataset('{repo_id}', split='deepmath_103k_verl')")

        return True

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Upload DeepMath-103K datasets to Hugging Face Hub',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (test without uploading)
  python scripts/upload/upload_deepmath_to_hub.py --individual --dry-run
  python scripts/upload/upload_deepmath_to_hub.py --unified --dry-run

  # Upload to individual repo
  python scripts/upload/upload_deepmath_to_hub.py --individual

  # Upload to unified repo
  python scripts/upload/upload_deepmath_to_hub.py --unified

  # Upload to both
  python scripts/upload/upload_deepmath_to_hub.py --both

Note: Run validation before uploading:
  python scripts/hub/validate_deepmath_upload.py --both
        """
    )
    parser.add_argument(
        '--individual',
        action='store_true',
        help='Upload to standalone repo (sungyub/deepmath-103k-verl)'
    )
    parser.add_argument(
        '--unified',
        action='store_true',
        help='Upload to unified repo (sungyub/math-verl-unified)'
    )
    parser.add_argument(
        '--both',
        action='store_true',
        help='Upload to both repos'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be uploaded without actually uploading'
    )

    args = parser.parse_args()

    # Default to both if no args
    if not (args.individual or args.unified or args.both):
        print("Please specify --individual, --unified, or --both")
        parser.print_help()
        sys.exit(1)

    # Check login (skip for dry-run)
    if not args.dry_run:
        if not check_login():
            sys.exit(1)
    else:
        print("=" * 70)
        print("DRY RUN MODE - No actual uploads will be performed")
        print("=" * 70)

    all_passed = True

    if args.both or args.individual:
        passed = upload_individual(dry_run=args.dry_run)
        all_passed &= passed
        if not passed:
            print(f"\n❌ Individual repo upload failed")
            if not args.both:
                sys.exit(1)

    if args.both or args.unified:
        passed = upload_unified(dry_run=args.dry_run)
        all_passed &= passed
        if not passed:
            print(f"\n❌ Unified repo upload failed")
            sys.exit(1)

    if all_passed:
        if args.dry_run:
            print(f"\n✅ DRY RUN COMPLETED - All uploads would succeed")
            print(f"\nTo perform actual upload, remove --dry-run flag:")
            if args.both or args.individual:
                print(f"  python scripts/upload/upload_deepmath_to_hub.py --individual")
            if args.both or args.unified:
                print(f"  python scripts/upload/upload_deepmath_to_hub.py --unified")
        else:
            print(f"\n🎉 ALL UPLOADS COMPLETED SUCCESSFULLY!")
            print(f"\nDatasets available at:")
            if args.both or args.individual:
                print(f"  - https://huggingface.co/datasets/sungyub/deepmath-103k-verl")
            if args.both or args.unified:
                print(f"  - https://huggingface.co/datasets/sungyub/math-verl-unified")
        print()
        sys.exit(0)
    else:
        print(f"\n❌ UPLOAD FAILED - Please check errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
