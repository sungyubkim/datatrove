#!/usr/bin/env python3
"""
Update Unified Dataset README with DeepMath-103K

This script updates the README.md for sungyub/math-verl-unified by adding:
- New split: deepmath_103k_verl (9th dataset)
- Updated total dataset count
- New update history entry

Usage:
    python scripts/hub/update_unified_with_deepmath.py
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


def add_new_split_to_readme(
    readme: str,
    split_name: str,
    num_samples: int,
    description: str,
) -> str:
    """Add new split to README.

    Args:
        readme: Original README content
        split_name: Name of new split (e.g., "deepmath_103k_verl")
        num_samples: Number of samples in split
        description: Description of the split

    Returns:
        Updated README content
    """
    updated_readme = readme

    # 1. Add to splits table
    # Find the table and add new row before the total row or at the end
    # Typical table format:
    # | Split Name | Samples | Percentage | Description |
    # |------------|---------|------------|-------------|
    # | big_math_rl_verl | 242,092 | ... | ... |
    # | **Total** | ... | ... | ... |

    # Calculate percentage (we'll need to extract total first)
    # For now, add row and note that percentages need recalculation
    new_row = f"| {split_name} | {num_samples:,} | TBD | {description} |"

    # Find the table section and add before Total row
    # Look for "| **Total**" or similar
    total_pattern = r'(\|\s*\*\*Total\*\*.*\|)'

    if re.search(total_pattern, updated_readme):
        # Insert before Total row
        updated_readme = re.sub(
            total_pattern,
            f"{new_row}\n" + r'\1',
            updated_readme,
        )
    else:
        # If no Total row, add at end of table (after last data row)
        # This is a fallback - ideally the table should have a Total row
        print("Warning: No Total row found in splits table. Adding at end.")

    # 2. Add to YAML configs
    # Look for the configs section and add new split
    new_config = f"""  - config_name: {split_name}
    data_files:
      - split: {split_name}
        path: data/{split_name}.parquet"""

    # Find configs section in YAML
    configs_pattern = r'(configs:\n(?:  - config_name:.*\n(?:    .*\n)*)*)'

    if re.search(configs_pattern, updated_readme):
        # Append to configs list
        updated_readme = re.sub(
            configs_pattern,
            r'\1' + new_config + '\n',
            updated_readme,
        )
    else:
        print("Warning: Could not find configs section in YAML frontmatter.")

    # 3. Add update history entry
    update_entry = f"""
### 2025-11-09: Added deepmath_103k_verl

- **New Split**: `deepmath_103k_verl`
- **Samples**: {num_samples:,}
- **Source**: [zwhe99/DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K)
- **License**: MIT
- **Processing**:
  - Cleaning: `orz-math` preset (aggressive)
  - Duplicates removed: 1,147 (1.11%)
  - Modified samples: 4,366 (4.2%)
  - Problem numbers removed: 1,555 samples
  - Multi-part problems filtered: 31 samples
- **Schema**: Minimal 3-field `extra_info` (index, original_dataset, split)
- **Quality**: High-quality, verified, decontaminated math problems
- **Coverage**: Advanced mathematics (calculus, algebra, probability, number theory, topology)
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

    # 4. Add note about recalculating percentages
    note = """

**Note**: Dataset percentages need to be recalculated after adding deepmath_103k_verl.
Please update the table with new total and percentages.
"""

    # Add note after the splits table if not already present
    if "percentages need to be recalculated" not in updated_readme:
        # Find end of splits table and add note
        table_end_pattern = r'(\|\s*\*\*Total\*\*.*\|.*\n)'
        updated_readme = re.sub(
            table_end_pattern,
            r'\1' + note + '\n',
            updated_readme,
        )

    return updated_readme


def update_readme(
    parquet_file: str = "./output/hub-upload/math-verl-unified/data/deepmath_103k_verl.parquet",
    output_file: str = "./output/hub-upload/math-verl-unified/README.md",
    repo_id: str = "sungyub/math-verl-unified",
    split_name: str = "deepmath_103k_verl",
):
    """Update unified dataset README with new split.

    Args:
        parquet_file: Path to new split's parquet file
        output_file: Output README path
        repo_id: HuggingFace repo ID
        split_name: Name of new split
    """
    print(f"\n{'='*70}")
    print(f"Adding DeepMath-103K to Unified Dataset README")
    print(f"{'='*70}")
    print(f"Parquet:    {parquet_file}")
    print(f"Output:     {output_file}")
    print(f"Repo:       {repo_id}")
    print(f"Split name: {split_name}")
    print(f"{'='*70}\n")

    # Check parquet file exists
    if not Path(parquet_file).exists():
        print(f"✗ Parquet file not found: {parquet_file}")
        print(f"\nPlease run prepare_deepmath_unified.py first:")
        print(f"  python scripts/hub/prepare_deepmath_unified.py")
        return False

    # Download current README
    print("Step 1: Downloading current README...")
    readme = download_current_readme(repo_id)

    if not readme:
        print("✗ Failed to download README. Cannot proceed.")
        return False

    print(f"✓ Downloaded README ({len(readme):,} bytes)")

    # Calculate metadata
    print("\nStep 2: Calculating metadata...")
    metadata = calculate_metadata(parquet_file)
    num_samples = metadata["num_examples"]
    file_size = metadata["file_size_mb"]

    print(f"✓ Metadata calculated")
    print(f"  Split: {split_name}")
    print(f"  Samples: {num_samples:,}")
    print(f"  File size: {file_size:.2f} MB")

    # Update README content
    print("\nStep 3: Adding new split to README...")
    description = "DeepMath-103K: Challenging, verified math problems (cleaned)"
    updated_readme = add_new_split_to_readme(readme, split_name, num_samples, description)

    print(f"✓ README updated")
    print(f"  New size: {len(updated_readme):,} bytes")
    print(f"  Original size: {len(readme):,} bytes")
    print(f"  Difference: +{len(updated_readme) - len(readme):,} bytes")

    # Write to file
    print("\nStep 4: Writing to file...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(updated_readme)

    print(f"✓ Saved to {output_file}")

    print(f"\n{'='*70}")
    print(f"✅ README update completed successfully!")
    print(f"{'='*70}")
    print(f"\n⚠️  Important: Please manually review and update:")
    print(f"  1. Total dataset count in splits table")
    print(f"  2. Percentage values for all splits")
    print(f"  3. Overall dataset description if needed")
    print(f"\nNext steps:")
    print(f"1. Review: cat {output_file}")
    print(f"2. Validate: python scripts/hub/validate_deepmath_upload.py --unified")
    print(f"3. Upload: python scripts/upload/upload_deepmath_to_hub.py --unified")
    print(f"{'='*70}\n")

    return True


def main():
    """Main entry point."""
    success = update_readme()

    if success:
        exit(0)
    else:
        print("\n✗ Update failed. Please check errors above.")
        exit(1)


if __name__ == "__main__":
    main()
