#!/usr/bin/env python3
"""
Upload big-math-rl-verl datasets to Hugging Face Hub.

Requirements:
1. Must be logged in: huggingface-cli login
2. Must have write access to the datasets

Uploads:
- sungyub/big-math-rl-verl (v2.0 cleaned)
- sungyub/math-verl-unified (big_math_rl_verl split update)
"""

import os
import sys
from pathlib import Path

from datasets import load_dataset
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


def upload_big_math_rl_verl(dry_run=False):
    """
    Upload big-math-rl-verl dataset (v2.0 cleaned).

    Args:
        dry_run: If True, only show what would be uploaded without actually uploading
    """
    print("\n" + "=" * 70)
    print("Uploading: sungyub/big-math-rl-verl (v2.0 cleaned)")
    print("=" * 70)

    repo_id = "sungyub/big-math-rl-verl"
    local_dir = "output/hub-upload/big-math-rl-verl"

    # Files to upload
    readme_path = os.path.join(local_dir, "README.md")
    data_path = os.path.join(local_dir, "data/train-00000.parquet")

    if dry_run:
        print("\n[DRY RUN] Would upload the following files:")
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"  README.md → README.md")
        print(f"  data/train-00000.parquet → data/train-00000.parquet ({size_mb:.2f} MB)")
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
            commit_message="Update to v2.0: Maximum cleaning, 242,092 samples (2 duplicates removed, 30 filtered)",
        )
        print("   ✓ README.md uploaded")

        # Upload data file
        print("\n2. Uploading data/train-00000.parquet...")
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")
        print(f"   This may take a few minutes...")

        api.upload_file(
            path_or_fileobj=data_path,
            path_in_repo="data/train-00000.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload v2.0 data: 242,092 samples with orz-math maximum cleaning",
        )
        print("   ✓ Data file uploaded")

        print(f"\n✅ Upload completed for {repo_id}")
        print(f"   View at: https://huggingface.co/datasets/{repo_id}")

        return True

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def upload_math_verl_unified(dry_run=False):
    """
    Upload updated big_math_rl_verl split to math-verl-unified.

    Args:
        dry_run: If True, only show what would be uploaded without actually uploading
    """
    print("\n" + "=" * 70)
    print("Uploading: sungyub/math-verl-unified (big_math_rl_verl split)")
    print("=" * 70)

    repo_id = "sungyub/math-verl-unified"
    local_dir = "output/hub-upload/math-verl-unified"

    # Files to upload
    readme_path = os.path.join(local_dir, "README.md")
    data_path = os.path.join(local_dir, "data/big-math-rl-verl.parquet")

    if dry_run:
        print("\n[DRY RUN] Would upload the following files:")
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"  README.md → README.md")
        print(f"  data/big-math-rl-verl.parquet → data/big-math-rl-verl.parquet ({size_mb:.2f} MB)")
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
            commit_message="Update big_math_rl_verl split to v2.0 (223,839 → 242,092 samples)",
        )
        print("   ✓ README.md uploaded")

        # Upload data file
        print("\n2. Uploading data/big-math-rl-verl.parquet...")
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")
        print(f"   This may take a few minutes...")

        api.upload_file(
            path_or_fileobj=data_path,
            path_in_repo="data/big-math-rl-verl.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update big_math_rl_verl split to v2.0: 242,092 samples with minimal schema",
        )
        print("   ✓ Data file uploaded")

        print(f"\n✅ Upload completed for {repo_id}")
        print(f"   View at: https://huggingface.co/datasets/{repo_id}")

        return True

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_upload(repo_id, split_name, expected_count):
    """Verify uploaded dataset by loading it."""
    print(f"\nVerifying upload for {repo_id} (split: {split_name})...")

    try:
        # Load dataset from Hub
        dataset = load_dataset(repo_id, split=split_name)

        # Check sample count
        actual_count = len(dataset)

        if actual_count == expected_count:
            print(f"   ✓ Sample count verified: {actual_count:,}")
            return True
        else:
            print(
                f"   ⚠️  Sample count mismatch: expected {expected_count:,}, got {actual_count:,}"
            )
            return False

    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main upload workflow."""
    print("=" * 70)
    print("Hugging Face Hub Upload Script - Big Math RL VERL")
    print("=" * 70)

    # Parse command line arguments
    dry_run = "--dry-run" in sys.argv
    skip_verify = "--skip-verify" in sys.argv
    auto_yes = "--yes" in sys.argv or "-y" in sys.argv

    if dry_run:
        print("\n⚠️  DRY RUN MODE - No actual uploads will be performed")

    # Check login status
    if not check_login() and not dry_run:
        print("\n❌ Please log in first: huggingface-cli login")
        return False

    # Confirm with user
    if not dry_run:
        print("\n" + "=" * 70)
        print("⚠️  WARNING: This will update production datasets!")
        print("=" * 70)
        print("\nDatasets to be updated:")
        print("  1. sungyub/big-math-rl-verl (242,124 → 242,092 samples)")
        print("  2. sungyub/math-verl-unified (big_math_rl_verl split: 223,839 → 242,092)")
        print("\nThis action will:")
        print("  - Replace existing data files")
        print("  - Update README documentation")
        print("  - Create new commits in the repositories")

        if auto_yes:
            print("\n✓ Auto-confirmed with --yes flag")
        else:
            try:
                response = input("\nContinue? (yes/no): ").strip().lower()
                if response != "yes":
                    print("\n❌ Upload cancelled by user")
                    return False
            except (EOFError, KeyboardInterrupt):
                print("\n\n❌ Upload cancelled by user")
                return False

    # Upload standalone dataset
    print("\n" + "=" * 70)
    print("Step 1: Upload sungyub/big-math-rl-verl")
    print("=" * 70)

    success1 = upload_big_math_rl_verl(dry_run=dry_run)

    if not success1:
        print("\n❌ Failed to upload big-math-rl-verl, aborting")
        return False

    # Verify upload
    if not dry_run and not skip_verify:
        verify_upload("sungyub/big-math-rl-verl", "train", expected_count=242092)

    # Upload unified dataset
    print("\n" + "=" * 70)
    print("Step 2: Upload sungyub/math-verl-unified")
    print("=" * 70)

    success2 = upload_math_verl_unified(dry_run=dry_run)

    if not success2:
        print("\n❌ Failed to upload math-verl-unified")
        return False

    # Verify upload
    if not dry_run and not skip_verify:
        verify_upload(
            "sungyub/math-verl-unified", "big_math_rl_verl", expected_count=242092
        )

    # Final summary
    print("\n" + "=" * 70)
    print("Upload Summary")
    print("=" * 70)

    if dry_run:
        print("\n✅ DRY RUN COMPLETE - No files were actually uploaded")
        print("\nTo perform actual upload, run:")
        print("  python scripts/upload/upload_big_math_to_hub.py --yes")
    else:
        print("\n✅ ALL UPLOADS COMPLETED SUCCESSFULLY!")
        print("\nUpdated datasets:")
        print("  ✓ sungyub/big-math-rl-verl (v2.0)")
        print("    → https://huggingface.co/datasets/sungyub/big-math-rl-verl")
        print("  ✓ sungyub/math-verl-unified (big_math_rl_verl split)")
        print("    → https://huggingface.co/datasets/sungyub/math-verl-unified")

        print("\nNext steps:")
        print("  1. Check dataset previews on Hub")
        print("  2. Verify README rendering")
        print("  3. Test loading with: load_dataset('sungyub/big-math-rl-verl')")
        print("  4. Run post-upload verification: python scripts/hub/verify_hub_upload.py")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload big-math-rl-verl datasets to Hugging Face Hub"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--skip-verify", action="store_true", help="Skip post-upload verification"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Auto-confirm upload without prompting"
    )

    args = parser.parse_args()
    sys.argv = [sys.argv[0]]  # Clear argv for HfApi
    if args.dry_run:
        sys.argv.append("--dry-run")
    if args.skip_verify:
        sys.argv.append("--skip-verify")
    if args.yes:
        sys.argv.append("--yes")

    success = main()
    sys.exit(0 if success else 1)
