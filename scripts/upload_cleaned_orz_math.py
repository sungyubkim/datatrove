#!/usr/bin/env python3
"""
Upload cleaned ORZ-Math dataset to HuggingFace Hub.

This script uploads the cleaned dataset to the EXISTING repository:
sungyub/orz-math-72k-verl

Uploads:
- Cleaned parquet file(s)
- Updated README with cleaning documentation
- Quality verification report
- Cleaning statistics
"""

from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
import sys

def upload_cleaned_dataset(
    repo_id: str = "sungyub/orz-math-72k-verl",
    data_dir: str = "output/orz-math-cleaned",
    commit_message: str = "Update with cleaned dataset v2.0 - 99.75% quality rate",
):
    """Upload cleaned dataset files to HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        data_dir: Local directory containing cleaned files
        commit_message: Git commit message
    """
    print(f"\n{'='*70}")
    print(f"Uploading cleaned dataset to HuggingFace Hub")
    print(f"{'='*70}")
    print(f"Repository: {repo_id}")
    print(f"Local directory: {data_dir}")
    print(f"\n")

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"✗ Error: Directory {data_dir} does not exist")
        sys.exit(1)

    # Initialize HF API
    api = HfApi()

    # Collect files to upload
    operations = []

    # 1. Upload parquet file(s) to data/ directory
    parquet_files = list(data_path.glob("*.parquet"))
    if not parquet_files:
        print(f"✗ Error: No parquet files found in {data_dir}")
        sys.exit(1)

    print(f"Parquet files to upload ({len(parquet_files)}):")
    for pf in parquet_files:
        # Upload to data/train-*.parquet to match YAML metadata
        remote_path = f"data/train-{pf.stem.split('_')[-1]}.parquet"
        print(f"  • {pf.name} → {remote_path}")
        operations.append(
            CommitOperationAdd(
                path_in_repo=remote_path,
                path_or_fileobj=str(pf),
            )
        )

    # 2. Upload README.md
    readme_path = data_path / "README.md"
    if readme_path.exists():
        print(f"\nREADME.md:")
        print(f"  • README.md → README.md")
        operations.append(
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=str(readme_path),
            )
        )
    else:
        print(f"\n⚠ Warning: README.md not found")

    # 3. Upload quality report
    quality_report = data_path / "DATASET_QUALITY_REPORT.md"
    if quality_report.exists():
        print(f"\nQuality Report:")
        print(f"  • DATASET_QUALITY_REPORT.md → DATASET_QUALITY_REPORT.md")
        operations.append(
            CommitOperationAdd(
                path_in_repo="DATASET_QUALITY_REPORT.md",
                path_or_fileobj=str(quality_report),
            )
        )

    # 4. Upload cleaning stats (JSON)
    stats_json = data_path / "cleaning_stats.json"
    if stats_json.exists():
        print(f"\nCleaning Statistics:")
        print(f"  • cleaning_stats.json → cleaning_stats.json")
        operations.append(
            CommitOperationAdd(
                path_in_repo="cleaning_stats.json",
                path_or_fileobj=str(stats_json),
            )
        )

    # 5. Upload cleaning report (text)
    report_txt = data_path / "cleaning_report.txt"
    if report_txt.exists():
        print(f"\nCleaning Report:")
        print(f"  • cleaning_report.txt → cleaning_report.txt")
        operations.append(
            CommitOperationAdd(
                path_in_repo="cleaning_report.txt",
                path_or_fileobj=str(report_txt),
            )
        )

    print(f"\n{'='*70}")
    print(f"Uploading {len(operations)} file(s) to {repo_id}...")
    print(f"{'='*70}\n")

    try:
        # Create commit with all operations
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message,
        )

        print(f"\n{'='*70}")
        print(f"✓ Upload successful!")
        print(f"{'='*70}")
        print(f"\nDataset URL: https://huggingface.co/datasets/{repo_id}")
        print(f"\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ Upload failed!")
        print(f"{'='*70}")
        print(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    upload_cleaned_dataset()
