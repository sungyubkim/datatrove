#!/usr/bin/env python3
"""
Upload improved MathX-5M VERL dataset to Hugging Face Hub.

This script uploads the cleaned and deduplicated MathX-5M dataset to the
sungyub/mathx-5m-verl repository on Hugging Face Hub.

Features:
- Pre-upload validation (schema, file existence)
- macOS file exclusion (.DS_Store, ._*, etc.)
- Proper commit messaging
- Post-upload verification

Usage:
    python scripts/upload/upload_mathx_verl.py
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, login


def validate_upload_files(upload_dir: Path) -> bool:
    """Validate upload directory before uploading.

    Args:
        upload_dir: Path to upload directory

    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n{'='*70}")
    print("Pre-Upload Validation")
    print(f"{'='*70}\n")

    # Check README exists
    readme_path = upload_dir / "README.md"
    if not readme_path.exists():
        print(f"✗ README.md not found at {readme_path}")
        return False
    print(f"✓ README.md found ({readme_path.stat().st_size:,} bytes)")

    # Check parquet file exists
    parquet_path = upload_dir / "data" / "train.parquet"
    if not parquet_path.exists():
        print(f"✗ train.parquet not found at {parquet_path}")
        return False

    # Validate parquet file
    try:
        table = pq.read_table(parquet_path)
        num_rows = len(table)
        file_size = parquet_path.stat().st_size

        print(f"✓ train.parquet found:")
        print(f"  - Rows: {num_rows:,}")
        print(f"  - Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        print(f"  - Schema: {list(table.schema.names)}")

        # Check VERL schema
        expected_fields = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
        actual_fields = list(table.schema.names)

        if actual_fields != expected_fields:
            print(f"✗ Schema mismatch!")
            print(f"  Expected: {expected_fields}")
            print(f"  Got: {actual_fields}")
            return False

        print(f"✓ Schema matches VERL standard (5 fields)")

    except Exception as e:
        print(f"✗ Failed to read parquet file: {e}")
        return False

    # Check for macOS artifacts
    macos_files = list(upload_dir.rglob(".DS_Store")) + \
                  list(upload_dir.rglob("._*")) + \
                  list(upload_dir.rglob(".Spotlight-V100")) + \
                  list(upload_dir.rglob(".Trashes"))

    if macos_files:
        print(f"\n✗ Found {len(macos_files)} macOS artifact files:")
        for f in macos_files[:10]:  # Show first 10
            print(f"  - {f.relative_to(upload_dir)}")
        if len(macos_files) > 10:
            print(f"  ... and {len(macos_files) - 10} more")
        print("\nPlease remove these files before uploading.")
        return False

    print(f"✓ No macOS artifacts found")

    print(f"\n{'='*70}")
    print("✅ Validation passed! Ready to upload.")
    print(f"{'='*70}\n")

    return True


def upload_to_hub(upload_dir: Path, repo_id: str, commit_message: str) -> bool:
    """Upload directory to Hugging Face Hub.

    Args:
        upload_dir: Path to upload directory
        repo_id: Repository ID (e.g., "sungyub/mathx-5m-verl")
        commit_message: Commit message

    Returns:
        True if upload succeeds, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Uploading to {repo_id}")
    print(f"{'='*70}\n")

    try:
        api = HfApi()

        # Upload folder
        print(f"📤 Uploading from: {upload_dir}")
        print(f"📦 Target repo: {repo_id}")
        print(f"💬 Commit message: {commit_message}\n")

        api.upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
            ignore_patterns=[".DS_Store", "._*", ".Spotlight-V100", ".Trashes"],
        )

        print(f"\n✅ Upload successful!")
        print(f"🔗 View at: https://huggingface.co/datasets/{repo_id}")

        return True

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        return False


def verify_upload(repo_id: str) -> bool:
    """Verify upload by checking repository info.

    Args:
        repo_id: Repository ID

    Returns:
        True if verification passes, False otherwise
    """
    print(f"\n{'='*70}")
    print("Post-Upload Verification")
    print(f"{'='*70}\n")

    try:
        api = HfApi()

        # Get repo info
        repo_info = api.repo_info(repo_id, repo_type="dataset")

        print(f"✓ Repository accessible: {repo_id}")
        print(f"✓ Last modified: {repo_info.last_modified}")
        print(f"✓ SHA: {repo_info.sha}")

        # List files
        files = api.list_repo_files(repo_id, repo_type="dataset")
        print(f"\n✓ Files in repository ({len(files)} total):")
        for f in sorted(files):
            print(f"  - {f}")

        # Check for expected files
        expected_files = ["README.md", "data/train.parquet"]
        missing_files = [f for f in expected_files if f not in files]

        if missing_files:
            print(f"\n✗ Missing expected files: {missing_files}")
            return False

        print(f"\n✓ All expected files present")

        print(f"\n{'='*70}")
        print("✅ Verification passed!")
        print(f"{'='*70}\n")

        return True

    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        return False


def main():
    """Main upload function."""
    print("=" * 70)
    print("MathX-5M VERL Dataset Upload to Hugging Face Hub")
    print("=" * 70)

    # Configuration
    UPLOAD_DIR = Path("output/hub-upload/mathx-5m-verl-v2")
    REPO_ID = "sungyub/mathx-5m-verl"
    COMMIT_MESSAGE = (
        "Update: Massive quality improvement (v2.0)\n\n"
        "- Deduplication: 26.9M → 1.45M samples (94.6% reduction)\n"
        "- Cleaning: Applied orz-math preset (7 artifact patterns)\n"
        "- Modified: 3.28M samples cleaned\n"
        "- Processing: 1h 42min, PyArrow streaming, 4,378 samples/sec\n"
        "- Schema: Standardized VERL format\n"
        "- Quality: Removed problem numbers, contest metadata, trailing artifacts\n\n"
        "This is a major quality improvement over the previous version."
    )

    # Step 1: Validate files
    if not validate_upload_files(UPLOAD_DIR):
        print("\n❌ Validation failed. Please fix issues and try again.")
        sys.exit(1)

    # Step 2: Confirm upload
    print(f"\n⚠️  You are about to upload to: {REPO_ID}")
    print(f"📁 Source directory: {UPLOAD_DIR}")
    print(f"\nCommit message:")
    print("-" * 70)
    print(COMMIT_MESSAGE)
    print("-" * 70)

    response = input("\n🔔 Proceed with upload? (yes/no): ").strip().lower()
    if response != "yes":
        print("\n❌ Upload cancelled by user.")
        sys.exit(0)

    # Step 3: Upload
    if not upload_to_hub(UPLOAD_DIR, REPO_ID, COMMIT_MESSAGE):
        print("\n❌ Upload failed.")
        sys.exit(1)

    # Step 4: Verify
    if not verify_upload(REPO_ID):
        print("\n⚠️  Upload succeeded but verification failed.")
        print("Please check the repository manually:")
        print(f"https://huggingface.co/datasets/{REPO_ID}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("🎉 SUCCESS! Dataset uploaded and verified.")
    print(f"{'='*70}")
    print(f"\n🔗 View your dataset at:")
    print(f"   https://huggingface.co/datasets/{REPO_ID}")
    print(f"\n📊 Test loading with:")
    print(f'   from datasets import load_dataset')
    print(f'   dataset = load_dataset("{REPO_ID}")')
    print()


if __name__ == "__main__":
    main()
