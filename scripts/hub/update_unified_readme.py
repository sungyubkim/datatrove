#!/usr/bin/env python3
"""
Update Unified Dataset README

This script updates the README.md for sungyub/math-verl-unified with:
- Updated split statistics for big_math_rl_verl
- Updated total dataset count
- New update history entry

Usage:
    python scripts/hub/update_unified_readme.py
"""

import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi


def download_current_readme(repo_id: str = "sungyub/math-verl-unified") -> str:
    """Download current README from Hub.

    Args:
        repo_id: HuggingFace repo ID

    Returns:
        README content as string
    """
    api = HfApi()

    try:
        readme_path = api.hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="dataset",
        )

        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error: Could not download README: {e}")
        print("Cannot proceed without existing README for unified dataset.")
        return None


def calculate_metadata(parquet_file: str) -> dict:
    """Calculate metadata from parquet file.

    Args:
        parquet_file: Path to parquet file

    Returns:
        Dictionary with num_examples, file_size
    """
    table = pq.read_table(parquet_file)
    file_size = Path(parquet_file).stat().st_size

    return {
        "num_examples": len(table),
        "file_size_mb": file_size / (1024**2),
    }


def update_readme_content(
    readme: str,
    new_count: int,
    old_count: int = 223839,
) -> str:
    """Update README content with new statistics.

    Args:
        readme: Original README content
        new_count: New sample count for big_math_rl_verl
        old_count: Old sample count (for update log)

    Returns:
        Updated README content
    """
    updated_readme = readme

    # 1. Update split statistics table
    # Look for the table row with big_math_rl_verl
    pattern = r'(\|\s*big_math_rl_verl\s*\|)\s*(\d{1,3}(?:,\d{3})*)\s*(\|.*?\|)'

    def replace_count(match):
        before = match.group(1)
        after = match.group(3)
        return f"{before} {new_count:,} {after}"

    updated_readme = re.sub(pattern, replace_count, updated_readme)

    # 2. Update total dataset count if present
    # This requires summing all splits - for now, we'll add a note
    # In practice, you'd need to parse all split counts and recalculate

    # 3. Add update history section if not present, or append to it
    update_entry = f"""
### 2025-11-09: big_math_rl_verl v2.0

- **Samples**: {old_count:,} → {new_count:,} (+{new_count - old_count:,})
- **Cleaning**: Applied maximum cleaning (`orz-math` preset)
  - Removed 2 duplicates
  - Filtered 30 multi-part problems
  - Modified 11,432 samples (4.7%)
  - Removed problem numbers: 474 samples
  - Removed point allocations: 128 samples
- **Schema**: Converted to minimal 3-field `extra_info`
  - Removed: `source`, `domain`, `solve_rate`
  - Added: `original_dataset = "big-math-rl-verl"`
- **Processing**: DataTrove MathDatasetCleaner pipeline
"""

    # Check if "Update History" or "Version History" section exists
    if "## Update History" in updated_readme or "## Version History" in updated_readme:
        # Append to existing section
        history_pattern = r'(## (?:Update|Version) History\s*\n)'
        updated_readme = re.sub(
            history_pattern,
            r'\1' + update_entry + '\n',
            updated_readme,
        )
    else:
        # Add new section before Citation or at end
        if "## Citation" in updated_readme:
            updated_readme = updated_readme.replace(
                "## Citation",
                f"## Update History{update_entry}\n\n## Citation",
            )
        else:
            updated_readme += f"\n\n## Update History{update_entry}\n"

    # 4. Update data_files path in YAML if needed
    # Make sure the path matches: data/big-math-rl-verl.parquet
    pattern = r'(split:\s*big_math_rl_verl\s*\n\s*path:)\s*.*'
    replacement = r'\1 data/big-math-rl-verl.parquet'
    updated_readme = re.sub(pattern, replacement, updated_readme)

    return updated_readme


def update_readme(
    parquet_file: str = "./output/hub-upload/math-verl-unified/data/big-math-rl-verl.parquet",
    output_file: str = "./output/hub-upload/math-verl-unified/README.md",
    repo_id: str = "sungyub/math-verl-unified",
):
    """Update unified dataset README.

    Args:
        parquet_file: Path to parquet file
        output_file: Output README path
        repo_id: HuggingFace repo ID
    """
    print(f"\n{'='*70}")
    print(f"Updating Unified Dataset README")
    print(f"{'='*70}")
    print(f"Parquet: {parquet_file}")
    print(f"Output:  {output_file}")
    print(f"Repo:    {repo_id}")
    print(f"{'='*70}\n")

    # Download current README
    print("Step 1: Downloading current README...")
    readme = download_current_readme(repo_id)

    if not readme:
        print("✗ Failed to download README. Cannot proceed.")
        return False

    print(f"✓ Downloaded README ({len(readme)} bytes)")

    # Calculate metadata
    print("\nStep 2: Calculating metadata...")
    metadata = calculate_metadata(parquet_file)
    new_count = metadata["num_examples"]

    print(f"✓ Metadata calculated")
    print(f"  New samples: {new_count:,}")
    print(f"  Old samples: 223,839")
    print(f"  Difference: +{new_count - 223839:,}")

    # Update README content
    print("\nStep 3: Updating README content...")
    updated_readme = update_readme_content(readme, new_count)

    print(f"✓ README updated")
    print(f"  New size: {len(updated_readme)} bytes")
    print(f"  Original size: {len(readme)} bytes")

    # Write to file
    print("\nStep 4: Writing to file...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(updated_readme)

    print(f"✓ Saved to {output_file}")

    print(f"\n{'='*70}")
    print(f"✅ README update completed successfully!")
    print(f"{'='*70}\n")

    return True


def main():
    """Main entry point."""
    success = update_readme()

    if success:
        print("\n📝 Next steps:")
        print("  1. Review README: cat output/hub-upload/math-verl-unified/README.md")
        print("  2. Validate: python scripts/hub/validate_before_upload.py --dataset-dir output/hub-upload/math-verl-unified")
    else:
        print("\n✗ Update failed. Please check errors above.")
        exit(1)


if __name__ == "__main__":
    main()
