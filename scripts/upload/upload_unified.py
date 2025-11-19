#!/usr/bin/env python3
"""
Upload MathX-5M to math-verl-unified repository.

This script uploads the cleaned and deduplicated MathX-5M dataset to the
sungyub/math-verl-unified repository as a new split.

Usage:
    python scripts/upload/upload_unified.py
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi


def main():
    """Main upload function."""
    print("=" * 70)
    print("Upload MathX-5M to math-verl-unified")
    print("=" * 70)

    # Configuration
    UPLOAD_DIR = Path("output/hub-upload/math-verl-unified")
    REPO_ID = "sungyub/math-verl-unified"
    COMMIT_MESSAGE = (
        "Add MathX-5M dataset (v2.0 - cleaned & deduplicated)\n\n"
        "- Samples: 1.45M (from 26.9M original, 94.6% deduplication)\n"
        "- Cleaning: orz-math preset (7 artifact patterns)\n"
        "- Modified: 3.28M samples cleaned\n"
        "- Schema: Added original_dataset field for provenance tracking\n"
        "- File: data/mathx-5m-verl.parquet\n\n"
        "This is a major quality improvement with comprehensive cleaning and deduplication."
    )

    # Verify files
    parquet_file = UPLOAD_DIR / "data" / "mathx-5m-verl.parquet"
    update_doc = UPLOAD_DIR / "MATHX_UPDATE.md"

    if not parquet_file.exists():
        print(f"✗ Parquet file not found: {parquet_file}")
        return 1

    if not update_doc.exists():
        print(f"✗ Update documentation not found: {update_doc}")
        return 1

    # Validate parquet
    try:
        table = pq.read_table(parquet_file)
        print(f"\n✓ Parquet file validated:")
        print(f"  - Rows: {len(table):,}")
        print(f"  - Size: {parquet_file.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"  - Schema: {list(table.schema.names)}")

        # Check original_dataset field
        sample = table.to_pylist()[0]
        if 'original_dataset' not in sample['extra_info']:
            print(f"✗ Missing original_dataset field in extra_info")
            return 1

        print(f"  - original_dataset: {sample['extra_info']['original_dataset']}")
        print()

    except Exception as e:
        print(f"✗ Failed to validate parquet: {e}")
        return 1

    # Confirm upload
    print(f"⚠️  You are about to upload to: {REPO_ID}")
    print(f"📁 Files to upload:")
    print(f"  - data/mathx-5m-verl.parquet ({parquet_file.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  - MATHX_UPDATE.md")
    print(f"\nCommit message:")
    print("-" * 70)
    print(COMMIT_MESSAGE)
    print("-" * 70)

    response = input("\n🔔 Proceed with upload? (yes/no): ").strip().lower()
    if response != "yes":
        print("\n❌ Upload cancelled by user.")
        return 0

    # Upload
    print(f"\n{'='*70}")
    print(f"Uploading to {REPO_ID}")
    print(f"{'='*70}\n")

    try:
        api = HfApi()

        api.upload_folder(
            folder_path=str(UPLOAD_DIR),
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=COMMIT_MESSAGE,
            ignore_patterns=[".DS_Store", "._*", ".Spotlight-V100", ".Trashes"],
        )

        print(f"\n✅ Upload successful!")
        print(f"🔗 View at: https://huggingface.co/datasets/{REPO_ID}")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        return 1

    # Verify
    print(f"\n{'='*70}")
    print("Post-Upload Verification")
    print(f"{'='*70}\n")

    try:
        repo_info = api.repo_info(REPO_ID, repo_type="dataset")
        print(f"✓ Repository accessible")
        print(f"✓ Last modified: {repo_info.last_modified}")

        files = api.list_repo_files(REPO_ID, repo_type="dataset")

        # Check for our files
        expected = ["data/mathx-5m-verl.parquet", "MATHX_UPDATE.md"]
        found = [f for f in expected if f in files]

        print(f"\n✓ Files uploaded:")
        for f in found:
            print(f"  - {f}")

        if len(found) != len(expected):
            print(f"\n⚠️  Warning: Not all expected files found")
            missing = [f for f in expected if f not in files]
            print(f"Missing: {missing}")

    except Exception as e:
        print(f"\n⚠️  Verification failed: {e}")

    print(f"\n{'='*70}")
    print("🎉 SUCCESS! MathX-5M added to math-verl-unified")
    print(f"{'='*70}")
    print(f"\n🔗 View at: https://huggingface.co/datasets/{REPO_ID}")
    print(f"\n📊 Load with:")
    print(f'   from datasets import load_dataset')
    print(f'   dataset = load_dataset("{REPO_ID}", data_files="data/mathx-5m-verl.parquet")')
    print()

    return 0


if __name__ == "__main__":
    exit(main())
