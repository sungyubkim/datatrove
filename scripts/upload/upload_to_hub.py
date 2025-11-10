#!/usr/bin/env python3
"""
Upload prepared datasets to Hugging Face Hub.

Requirements:
1. Must be logged in: huggingface-cli login
2. Must have write access to the datasets
3. All validation must pass before upload

Uploads:
- sungyub/openr1-math-verl (v3.0)
- sungyub/math-verl-unified (openr1_math_verl split update)
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, Repository, upload_file, upload_folder
from datasets import load_dataset


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


def upload_openr1_math_verl(dry_run=False):
    """
    Upload openr1-math-verl dataset (v3.0).

    Args:
        dry_run: If True, only show what would be uploaded without actually uploading
    """
    print("\n" + "="*70)
    print("Uploading: sungyub/openr1-math-verl (v3.0)")
    print("="*70)

    repo_id = "sungyub/openr1-math-verl"
    local_dir = "output/hub-upload/openr1-math-verl"

    files_to_upload = {
        "train.parquet": "data/train-00000.parquet",  # Rename for Hub
        "README.md": "README.md"
    }

    if dry_run:
        print("\n[DRY RUN] Would upload the following files:")
        for local_file, hub_path in files_to_upload.items():
            local_path = os.path.join(local_dir, local_file)
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"  {local_file} → {hub_path} ({size_mb:.2f} MB)")
        return True

    api = HfApi()

    try:
        # Upload README first
        print("\n1. Uploading README.md...")
        readme_path = os.path.join(local_dir, "README.md")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update to v3.0: Enhanced cleaning, full dataset coverage (184,439 samples)"
        )
        print("   ✓ README.md uploaded")

        # Upload data file
        print("\n2. Uploading train.parquet → data/train-00000.parquet...")
        data_path = os.path.join(local_dir, "train.parquet")
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")
        print(f"   This may take a few minutes...")

        api.upload_file(
            path_or_fileobj=data_path,
            path_in_repo="data/train-00000.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload v3.0 data: 184,439 samples with maximum cleaning"
        )
        print("   ✓ Data file uploaded")

        print("\n✅ Upload completed for sungyub/openr1-math-verl")
        print(f"   View at: https://huggingface.co/datasets/{repo_id}")

        return True

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return False


def upload_math_verl_unified(dry_run=False):
    """
    Upload updated openr1_math_verl split to math-verl-unified.

    Args:
        dry_run: If True, only show what would be uploaded without actually uploading
    """
    print("\n" + "="*70)
    print("Uploading: sungyub/math-verl-unified (openr1_math_verl split)")
    print("="*70)

    repo_id = "sungyub/math-verl-unified"
    local_dir = "output/hub-upload/math-verl-unified"

    files_to_upload = {
        "openr1-math-verl.parquet": "data/openr1-math-verl.parquet",
        "README.md": "README.md"
    }

    if dry_run:
        print("\n[DRY RUN] Would upload the following files:")
        for local_file, hub_path in files_to_upload.items():
            local_path = os.path.join(local_dir, local_file)
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"  {local_file} → {hub_path} ({size_mb:.2f} MB)")
        return True

    api = HfApi()

    try:
        # Upload README first
        print("\n1. Uploading README.md...")
        readme_path = os.path.join(local_dir, "README.md")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update openr1_math_verl split to v3.0 (18,713 → 184,439 samples)"
        )
        print("   ✓ README.md uploaded")

        # Upload data file
        print("\n2. Uploading openr1-math-verl.parquet → data/openr1-math-verl.parquet...")
        data_path = os.path.join(local_dir, "openr1-math-verl.parquet")
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")
        print(f"   This may take a few minutes...")

        api.upload_file(
            path_or_fileobj=data_path,
            path_in_repo="data/openr1-math-verl.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update openr1_math_verl split to v3.0: 184,439 samples with enhanced metadata"
        )
        print("   ✓ Data file uploaded")

        print("\n✅ Upload completed for sungyub/math-verl-unified")
        print(f"   View at: https://huggingface.co/datasets/{repo_id}")

        return True

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return False


def verify_upload(repo_id, expected_count):
    """Verify uploaded dataset by loading it."""
    print(f"\nVerifying upload for {repo_id}...")

    try:
        # Load dataset from Hub
        dataset = load_dataset(repo_id)

        # Check sample count
        actual_count = len(dataset['train']) if 'train' in dataset else len(dataset[list(dataset.keys())[0]])

        if actual_count == expected_count:
            print(f"   ✓ Sample count verified: {actual_count:,}")
            return True
        else:
            print(f"   ⚠️  Sample count mismatch: expected {expected_count:,}, got {actual_count:,}")
            return False

    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False


def main():
    """Main upload workflow."""
    print("="*70)
    print("Hugging Face Hub Upload Script")
    print("="*70)

    # Parse command line arguments
    dry_run = '--dry-run' in sys.argv
    skip_verify = '--skip-verify' in sys.argv
    auto_yes = '--yes' in sys.argv or '-y' in sys.argv

    if dry_run:
        print("\n⚠️  DRY RUN MODE - No actual uploads will be performed")

    # Check login status
    if not check_login() and not dry_run:
        print("\n❌ Please log in first: huggingface-cli login")
        return False

    # Confirm with user
    if not dry_run:
        print("\n" + "="*70)
        print("⚠️  WARNING: This will update production datasets!")
        print("="*70)
        print("\nDatasets to be updated:")
        print("  1. sungyub/openr1-math-verl (52,874 → 184,439 samples)")
        print("  2. sungyub/math-verl-unified (openr1 split: 18,713 → 184,439)")
        print("\nThis action will:")
        print("  - Replace existing data files")
        print("  - Update README documentation")
        print("  - Create new commits in the repositories")

        if auto_yes:
            print("\n✓ Auto-confirmed with --yes flag")
        else:
            try:
                response = input("\nContinue? (yes/no): ").strip().lower()
                if response != 'yes':
                    print("\n❌ Upload cancelled by user")
                    return False
            except EOFError:
                print("\n❌ Cannot get user input. Use --yes flag to auto-confirm.")
                return False

    # Upload individual dataset
    print("\n" + "="*70)
    print("Step 1: Upload sungyub/openr1-math-verl")
    print("="*70)

    success1 = upload_openr1_math_verl(dry_run=dry_run)

    if not success1:
        print("\n❌ Failed to upload openr1-math-verl, aborting")
        return False

    # Verify upload
    if not dry_run and not skip_verify:
        verify_upload("sungyub/openr1-math-verl", expected_count=184439)

    # Upload unified dataset
    print("\n" + "="*70)
    print("Step 2: Upload sungyub/math-verl-unified")
    print("="*70)

    success2 = upload_math_verl_unified(dry_run=dry_run)

    if not success2:
        print("\n❌ Failed to upload math-verl-unified")
        return False

    # Verify upload
    if not dry_run and not skip_verify:
        # Note: This loads the specific split
        try:
            dataset = load_dataset("sungyub/math-verl-unified", split="openr1_math_verl")
            if len(dataset) == 184439:
                print(f"   ✓ openr1_math_verl split verified: 184,439 samples")
            else:
                print(f"   ⚠️  Split count mismatch: {len(dataset):,}")
        except Exception as e:
            print(f"   ❌ Verification failed: {e}")

    # Final summary
    print("\n" + "="*70)
    print("Upload Summary")
    print("="*70)

    if dry_run:
        print("\n✅ DRY RUN COMPLETE - No files were actually uploaded")
        print("\nTo perform actual upload, run:")
        print("  python scripts/upload/upload_to_hub.py")
    else:
        print("\n✅ ALL UPLOADS COMPLETED SUCCESSFULLY!")
        print("\nUpdated datasets:")
        print("  ✓ sungyub/openr1-math-verl (v3.0)")
        print("    → https://huggingface.co/datasets/sungyub/openr1-math-verl")
        print("  ✓ sungyub/math-verl-unified (openr1_math_verl split)")
        print("    → https://huggingface.co/datasets/sungyub/math-verl-unified")

        print("\nNext steps:")
        print("  1. Check dataset previews on Hub")
        print("  2. Verify README rendering")
        print("  3. Test loading with: load_dataset('sungyub/openr1-math-verl')")
        print("  4. Announce the update to users")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload datasets to Hugging Face Hub")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be uploaded without uploading")
    parser.add_argument('--skip-verify', action='store_true', help="Skip post-upload verification")
    parser.add_argument('--yes', '-y', action='store_true', help="Auto-confirm upload without prompting")

    args = parser.parse_args()
    sys.argv = [sys.argv[0]]  # Clear argv for HfApi
    if args.dry_run:
        sys.argv.append('--dry-run')
    if args.skip_verify:
        sys.argv.append('--skip-verify')
    if args.yes:
        sys.argv.append('--yes')

    success = main()
    sys.exit(0 if success else 1)
