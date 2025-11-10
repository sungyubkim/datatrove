#!/usr/bin/env python3
"""
Clean up duplicate Skywork file from math-verl-unified repository.

This script removes the older, incomplete Skywork OR1 Math dataset file
while keeping the newer, complete version.

Files:
  - DELETE: data/skywork-or1-math-verl.parquet (30,603 samples, 5.70 MB)
  - KEEP:   data/skywork_or1_math_verl-00000.parquet (102,669 samples, 16.21 MB)

Usage:
    python scripts/hub/cleanup_skywork_duplicate.py
"""

from huggingface_hub import HfApi


def main():
    """Clean up duplicate skywork file."""
    print("=" * 70)
    print("HuggingFace Hub Cleanup - Duplicate Skywork File")
    print("=" * 70)

    api = HfApi()
    repo_id = "sungyub/math-verl-unified"

    # File to delete (older, smaller version)
    file_to_delete = "data/skywork-or1-math-verl.parquet"

    # File to keep (newer, larger version)
    file_to_keep = "data/skywork_or1_math_verl-00000.parquet"

    # Show comparison
    print("\n📊 File Comparison:")
    print(f"  DELETE: {file_to_delete}")
    print(f"    - Size: 5.70 MB")
    print(f"    - Samples: 30,603")
    print()
    print(f"  KEEP:   {file_to_keep}")
    print(f"    - Size: 16.21 MB")
    print(f"    - Samples: 102,669")
    print(f"    - 3.35x MORE samples than old file")

    # Auto-proceed (script designed for automated execution)
    print(f"\n⚠️  Deleting the OLDER Skywork file from: {repo_id}")

    # Delete old file
    print(f"\n{'='*70}")
    print(f"Deleting old Skywork file from {repo_id}")
    print(f"{'='*70}\n")

    try:
        print(f"  Deleting: {file_to_delete}...", end=" ")
        api.delete_file(
            path_in_repo=file_to_delete,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"chore: Remove duplicate Skywork file (keeping newer version with 102K samples)",
        )
        print("✓")

        print(f"\n✅ Cleanup complete!")
        print(f"\nRemaining file:")
        print(f"  ✓ {file_to_keep} (102,669 samples)")

    except Exception as e:
        print(f"✗ Failed: {e}")
        return

    print(f"\n{'='*70}")
    print("🎉 Skywork duplicate removed successfully!")
    print(f"{'='*70}")
    print(f"\n🔗 Verify at: https://huggingface.co/datasets/{repo_id}/tree/main/data")
    print()


if __name__ == "__main__":
    main()
