#!/usr/bin/env python3
"""
Clean up old files from HuggingFace Hub repositories.

This script removes outdated files from the mathx-5m-verl repositories
after uploading the new improved version.

Usage:
    python scripts/hub/cleanup_old_files.py
"""

from huggingface_hub import HfApi


def cleanup_mathx_verl():
    """Clean up old files from sungyub/mathx-5m-verl."""
    api = HfApi()
    repo_id = "sungyub/mathx-5m-verl"

    # Files to delete (old version)
    files_to_delete = [
        "data/train-00000.parquet",
        "data/train-00001.parquet",
        "data/train-00002.parquet",
        "convert_mathx5m_multifile.py",
        "convert_mathx5m_to_verl.py",
        "requirements.txt",
        "stats.json",
        "CLAUDE.md",
    ]

    print(f"\n{'='*70}")
    print(f"Cleaning up: {repo_id}")
    print(f"{'='*70}\n")

    for file_path in files_to_delete:
        try:
            print(f"  Deleting: {file_path}...", end=" ")
            api.delete_file(
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"chore: Remove old file {file_path}",
            )
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    print(f"\n✅ Cleanup complete for {repo_id}")


def cleanup_unified():
    """Clean up old files from sungyub/math-verl-unified."""
    api = HfApi()
    repo_id = "sungyub/math-verl-unified"

    # Old MathX-5M files (15 split files)
    old_mathx_files = [
        f"data/mathx-5m-verl-{i:05d}-of-00015.parquet"
        for i in range(15)
    ]

    # macOS artifacts and misplaced files
    other_files = [
        "data/._mathx-5m-verl.parquet",
        "openr1-math-verl.parquet",
        "skywork_or1_math_verl.parquet",
    ]

    files_to_delete = old_mathx_files + other_files

    print(f"\n{'='*70}")
    print(f"Cleaning up: {repo_id}")
    print(f"{'='*70}\n")

    for file_path in files_to_delete:
        try:
            print(f"  Deleting: {file_path}...", end=" ")
            api.delete_file(
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"chore: Remove old file {file_path}",
            )
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    print(f"\n✅ Cleanup complete for {repo_id}")


def main():
    """Main cleanup function."""
    print("=" * 70)
    print("HuggingFace Hub Cleanup - Old MathX-5M Files")
    print("=" * 70)

    # Confirm
    print("\nThis will delete old files from:")
    print("  1. sungyub/mathx-5m-verl (8 files)")
    print("  2. sungyub/math-verl-unified (18 files)")
    print("\nNew files will be kept:")
    print("  - data/train.parquet (mathx-5m-verl)")
    print("  - data/mathx-5m-verl.parquet (math-verl-unified)")

    response = input("\n🔔 Proceed with cleanup? (yes/no): ").strip().lower()
    if response != "yes":
        print("\n❌ Cleanup cancelled.")
        return

    # Clean up both repositories
    cleanup_mathx_verl()
    cleanup_unified()

    print(f"\n{'='*70}")
    print("🎉 All cleanup complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
