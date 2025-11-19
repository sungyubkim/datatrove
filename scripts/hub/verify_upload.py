#!/usr/bin/env python3
"""
Consolidated upload verification script for local and Hub datasets.

This script consolidates verification logic, replacing:
- verify_hub_upload.py
- verify_local_files.py
- verify_hub_uploads.py (partially)

Usage:
    # Verify local files before upload
    python scripts/hub/verify_upload.py --mode local \
        --local-dir output/deduplicated-inter/data

    # Verify Hub upload (quick check)
    python scripts/hub/verify_upload.py --mode hub \
        --repo-id sungyub/math-verl-unified \
        --check-splits dapo_math_17k_verl,deepscaler_preview_verl

    # Verify Hub upload with row count validation
    python scripts/hub/verify_upload.py --mode hub \
        --repo-id sungyub/math-verl-unified \
        --check-splits dapo_math_17k_verl \
        --expected-counts dapo_math_17k_verl:17186
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import list_repo_files


def verify_local_files(local_dir: Path, check_macos_artifacts: bool = True) -> bool:
    """Verify local parquet files before upload.

    Args:
        local_dir: Directory containing parquet files
        check_macos_artifacts: Whether to check for macOS system files

    Returns:
        True if verification passed
    """
    print("=" * 70)
    print("LOCAL FILES VERIFICATION")
    print("=" * 70)

    if not local_dir.exists():
        print(f"❌ ERROR: Directory not found: {local_dir}")
        return False

    # Find parquet files
    files = sorted(local_dir.glob("*.parquet"))
    if not files:
        # Try recursive search
        files = sorted(local_dir.rglob("*.parquet"))

    if not files:
        print(f"❌ ERROR: No parquet files found in {local_dir}")
        return False

    print(f"\n✓ Found {len(files)} parquet file(s)\n")
    print(f"{'File':<50} {'Rows':>12} {'Size (MB)':>12}")
    print("-" * 74)

    total_rows = 0
    total_size_mb = 0
    file_stats = {}

    for f in files:
        try:
            table = pq.read_table(f)
            rows = len(table)
            size_mb = f.stat().st_size / (1024 * 1024)
            total_rows += rows
            total_size_mb += size_mb

            filename = f.name
            file_stats[filename] = {'rows': rows, 'size_mb': size_mb}

            # Truncate long filenames for display
            display_name = filename if len(filename) <= 50 else filename[:47] + "..."
            print(f"{display_name:<50} {rows:>12,} {size_mb:>12.1f}")
        except Exception as e:
            print(f"{f.name:<50} {'ERROR':>12} {str(e)[:12]:>12}")
            return False

    print("-" * 74)
    print(f"{'TOTAL':<50} {total_rows:>12,} {total_size_mb:>12.1f}")

    # Check for macOS system files
    if check_macos_artifacts:
        print(f"\n{'='*70}")
        print("MACOS ARTIFACTS CHECK")
        print("=" * 70)

        artifact_patterns = ['._*', '.DS_Store', '.Spotlight-V100', '.Trashes', '.fseventsd']
        found_artifacts = []

        for pattern in artifact_patterns:
            if '*' in pattern:
                matches = [f for f in local_dir.rglob('*') if f.name.startswith(pattern.replace('*', ''))]
            else:
                matches = list(local_dir.rglob(pattern))
            found_artifacts.extend(matches)

        if found_artifacts:
            print(f"❌ ERROR: Found {len(found_artifacts)} macOS system file(s):")
            for artifact in found_artifacts:
                rel_path = artifact.relative_to(local_dir)
                print(f"   - {rel_path}")
            print(f"\n⚠️  Remove these files before uploading to Hub")
            return False
        else:
            print("✓ No macOS system files found")

    # Show schema for first file
    print(f"\n{'='*70}")
    print("SCHEMA CHECK (first file)")
    print("=" * 70)

    first_file = files[0]
    table = pq.read_table(first_file)
    print(f"\nFile: {first_file.name}")
    print(f"Schema:")
    for field in table.schema:
        print(f"  - {field.name}: {field.type}")

    # Check required fields
    required_fields = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
    schema_fields = [field.name for field in table.schema]
    missing = [f for f in required_fields if f not in schema_fields]

    if missing:
        print(f"\n❌ ERROR: Missing required fields: {missing}")
        return False
    else:
        print(f"\n✓ All required fields present")

    print(f"\n{'='*70}")
    print("✅ LOCAL VERIFICATION PASSED")
    print("=" * 70)
    print(f"✓ {len(files)} file(s) verified")
    print(f"✓ {total_rows:,} total rows")
    print(f"✓ {total_size_mb:.1f} MB total size")
    print(f"✓ Schema valid")
    if check_macos_artifacts:
        print(f"✓ No macOS artifacts")
    print("=" * 70)

    return True


def verify_hub_upload(
    repo_id: str,
    check_splits: Optional[List[str]] = None,
    expected_counts: Optional[Dict[str, int]] = None,
    sample_check_only: bool = True
) -> bool:
    """Verify dataset uploaded to Hugging Face Hub.

    Args:
        repo_id: Hub repository ID (e.g., 'sungyub/math-verl-unified')
        check_splits: List of split names to verify
        expected_counts: Dict mapping split names to expected row counts
        sample_check_only: If True, only check a few samples

    Returns:
        True if verification passed
    """
    print("=" * 70)
    print(f"HUB UPLOAD VERIFICATION")
    print("=" * 70)
    print(f"Repository: {repo_id}\n")

    # 1. Check file list
    print("1. FILE LIST CHECK")
    print("-" * 70)

    try:
        files = list_repo_files(repo_id, repo_type='dataset')
        parquet_files = sorted([f for f in files if f.endswith('.parquet')])

        print(f"✓ Total parquet files: {len(parquet_files)}")

        if len(parquet_files) == 0:
            print(f"❌ ERROR: No parquet files found in repository")
            return False

        print("\nParquet files:")
        for f in parquet_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(parquet_files) > 10:
            print(f"  ... and {len(parquet_files) - 10} more")

    except Exception as e:
        print(f"❌ ERROR: Failed to list repository files: {e}")
        return False

    # 2. Check splits (if specified)
    if check_splits:
        print(f"\n2. SPLIT VERIFICATION")
        print("-" * 70)

        if sample_check_only:
            print("Running sample checks (not exhaustive)")

        print(f"\n{'Split':<35} {'Expected':>12} {'Actual':>12} {'Status'}")
        print("-" * 70)

        all_passed = True

        for split_name in check_splits:
            try:
                # Load dataset split
                ds = load_dataset(repo_id, split=split_name, streaming=sample_check_only)

                if sample_check_only:
                    # Just check that we can load it
                    first_sample = next(iter(ds))
                    status = "✅ Accessible"
                    actual_rows = "N/A"
                else:
                    # Count actual rows
                    actual_rows = len(ds)
                    expected_rows = expected_counts.get(split_name) if expected_counts else None

                    if expected_rows is not None:
                        if actual_rows == expected_rows:
                            status = "✅"
                        else:
                            status = f"❌ Mismatch"
                            all_passed = False
                    else:
                        status = "✓"

                expected_display = expected_counts.get(split_name, "N/A") if expected_counts else "N/A"
                if isinstance(expected_display, int):
                    expected_display = f"{expected_display:,}"
                if isinstance(actual_rows, int):
                    actual_rows = f"{actual_rows:,}"

                print(f"{split_name:<35} {expected_display:>12} {actual_rows:>12} {status}")

            except Exception as e:
                print(f"{split_name:<35} {'N/A':>12} {'ERROR':>12} ❌")
                print(f"  Error: {str(e)[:60]}")
                all_passed = False

        if not all_passed:
            return False

    # 3. Sample data structure check
    print(f"\n3. DATA STRUCTURE CHECK")
    print("-" * 70)

    try:
        # Load first split or default split
        if check_splits and len(check_splits) > 0:
            test_split = check_splits[0]
        else:
            test_split = 'train'  # Default

        ds = load_dataset(repo_id, split=test_split, streaming=True)
        first_sample = next(iter(ds))

        print(f"✓ Successfully loaded sample from split '{test_split}'")
        print(f"\nSample structure:")
        print(f"  - data_source: {first_sample.get('data_source', 'N/A')}")
        print(f"  - ability: {first_sample.get('ability', 'N/A')}")
        print(f"  - prompt: {type(first_sample.get('prompt', None)).__name__} "
              f"with {len(first_sample.get('prompt', [])) if isinstance(first_sample.get('prompt'), list) else 0} messages")
        print(f"  - reward_model: {type(first_sample.get('reward_model', None)).__name__}")
        print(f"  - extra_info: {first_sample.get('extra_info', {})}")

        # Check required fields
        required_fields = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
        missing = [f for f in required_fields if f not in first_sample]

        if missing:
            print(f"\n❌ ERROR: Missing required fields: {missing}")
            return False
        else:
            print(f"\n✓ All required fields present")

    except Exception as e:
        print(f"❌ ERROR: Failed to load sample data: {e}")
        return False

    # 4. Summary
    print(f"\n{'='*70}")
    print("✅ HUB VERIFICATION PASSED")
    print("=" * 70)
    print(f"✓ Repository accessible: {repo_id}")
    print(f"✓ {len(parquet_files)} parquet file(s) found")
    if check_splits:
        print(f"✓ {len(check_splits)} split(s) verified")
    print(f"✓ Data structure valid")
    print(f"\n🔗 Visit: https://huggingface.co/datasets/{repo_id}")
    print("=" * 70)

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Verify dataset files locally or on Hub',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify local files
  python scripts/hub/verify_upload.py --mode local \\
      --local-dir output/deduplicated-inter/data

  # Verify Hub upload (quick check)
  python scripts/hub/verify_upload.py --mode hub \\
      --repo-id sungyub/math-verl-unified \\
      --check-splits dapo_math_17k_verl,deepscaler_preview_verl

  # Verify with row count validation
  python scripts/hub/verify_upload.py --mode hub \\
      --repo-id sungyub/math-verl-unified \\
      --check-splits dapo_math_17k_verl \\
      --expected-counts dapo_math_17k_verl:17186 \\
      --full-count-check
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['local', 'hub'],
        help='Verification mode'
    )

    parser.add_argument(
        '--local-dir',
        type=str,
        help='Local directory with parquet files (required for local mode)'
    )

    parser.add_argument(
        '--repo-id',
        type=str,
        help='Hub repository ID (required for hub mode)'
    )

    parser.add_argument(
        '--check-splits',
        type=str,
        help='Comma-separated list of split names to verify (hub mode)'
    )

    parser.add_argument(
        '--expected-counts',
        type=str,
        help='Expected row counts as split:count pairs (e.g., "split1:1000,split2:2000")'
    )

    parser.add_argument(
        '--full-count-check',
        action='store_true',
        help='Perform full row count (slower, loads entire dataset)'
    )

    parser.add_argument(
        '--skip-macos-check',
        action='store_true',
        help='Skip macOS artifacts check (local mode)'
    )

    args = parser.parse_args()

    # Mode-specific validation
    if args.mode == 'local':
        if not args.local_dir:
            print("❌ ERROR: --local-dir required for local mode")
            sys.exit(1)

        local_path = Path(args.local_dir)
        success = verify_local_files(
            local_path,
            check_macos_artifacts=not args.skip_macos_check
        )

    elif args.mode == 'hub':
        if not args.repo_id:
            print("❌ ERROR: --repo-id required for hub mode")
            sys.exit(1)

        # Parse check_splits
        check_splits = None
        if args.check_splits:
            check_splits = [s.strip() for s in args.check_splits.split(',')]

        # Parse expected_counts
        expected_counts = None
        if args.expected_counts:
            expected_counts = {}
            for pair in args.expected_counts.split(','):
                split, count = pair.strip().split(':')
                expected_counts[split] = int(count)

        success = verify_hub_upload(
            args.repo_id,
            check_splits=check_splits,
            expected_counts=expected_counts,
            sample_check_only=not args.full_count_check
        )

    else:
        print(f"❌ ERROR: Unknown mode: {args.mode}")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
