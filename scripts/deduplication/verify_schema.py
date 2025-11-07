"""
Verify data schema consistency across unified dataset splits.

This script validates that all rows in the unified dataset conform to the
VERL schema and that the schema is consistent across all splits.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import pandas as pd
import pyarrow.parquet as pq
from collections import Counter, defaultdict

# Import validation from utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import validate_verl_row, format_number, format_percentage


def analyze_schema(split_path: Path, split_name: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze schema for a single split.

    Args:
        split_path: Path to parquet file
        split_name: Name of the split
        verbose: Print detailed information

    Returns:
        Dictionary with schema analysis results
    """
    df = pd.read_parquet(split_path)
    total_rows = len(df)

    # Schema statistics
    stats = {
        'split_name': split_name,
        'total_rows': total_rows,
        'valid_rows': 0,
        'invalid_rows': 0,
        'validation_errors': Counter(),
        'data_sources': set(),
        'abilities': set(),
        'prompt_roles': set(),
        'reward_styles': set(),
        'field_presence': defaultdict(int),
        'sample_errors': [],
    }

    for idx, row in df.iterrows():
        row_dict = row.to_dict()

        # Validate VERL schema
        is_valid, error_msg = validate_verl_row(row_dict)

        if is_valid:
            stats['valid_rows'] += 1

            # Collect schema info
            stats['data_sources'].add(row_dict.get('data_source', 'N/A'))
            stats['abilities'].add(row_dict.get('ability', 'N/A'))

            if 'prompt' in row_dict and len(row_dict['prompt']) > 0:
                stats['prompt_roles'].add(row_dict['prompt'][0].get('role', 'N/A'))

            if 'reward_model' in row_dict:
                stats['reward_styles'].add(row_dict['reward_model'].get('style', 'N/A'))

            # Track field presence
            for field in ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']:
                if field in row_dict:
                    stats['field_presence'][field] += 1

        else:
            stats['invalid_rows'] += 1
            stats['validation_errors'][error_msg] += 1

            # Store sample errors (max 5)
            if len(stats['sample_errors']) < 5:
                stats['sample_errors'].append({
                    'index': idx,
                    'error': error_msg,
                    'data_source': row_dict.get('data_source', 'N/A')
                })

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Split: {split_name}")
        print(f"{'=' * 70}")
        print(f"Total rows:    {format_number(total_rows)}")
        print(f"Valid rows:    {format_number(stats['valid_rows'])} ({format_percentage(stats['valid_rows'], total_rows)})")
        print(f"Invalid rows:  {format_number(stats['invalid_rows'])} ({format_percentage(stats['invalid_rows'], total_rows)})")

        if stats['invalid_rows'] > 0:
            print(f"\nValidation errors:")
            for error, count in stats['validation_errors'].most_common():
                print(f"  - {error}: {count} rows")

            print(f"\nSample errors:")
            for sample in stats['sample_errors']:
                print(f"  Row {sample['index']} ({sample['data_source']}): {sample['error']}")

        print(f"\nData sources: {len(stats['data_sources'])}")
        for ds in sorted(stats['data_sources']):
            print(f"  - {ds}")

        print(f"\nAbilities: {len(stats['abilities'])}")
        for ability in sorted(stats['abilities']):
            print(f"  - {ability}")

        print(f"\nPrompt roles: {len(stats['prompt_roles'])}")
        for role in sorted(stats['prompt_roles']):
            print(f"  - {role}")

        print(f"\nReward styles: {len(stats['reward_styles'])}")
        for style in sorted(stats['reward_styles']):
            print(f"  - {style}")

    return stats


def verify_consistency(all_stats: List[Dict[str, Any]], verbose: bool = False) -> bool:
    """
    Verify consistency across all splits.

    Args:
        all_stats: List of schema analysis results
        verbose: Print detailed information

    Returns:
        True if consistent, False otherwise
    """
    print(f"\n{'=' * 70}")
    print("Cross-Split Consistency Check")
    print(f"{'=' * 70}")

    # Collect all unique values
    all_data_sources = set()
    all_abilities = set()
    all_prompt_roles = set()
    all_reward_styles = set()

    total_valid = 0
    total_invalid = 0

    for stats in all_stats:
        all_data_sources.update(stats['data_sources'])
        all_abilities.update(stats['abilities'])
        all_prompt_roles.update(stats['prompt_roles'])
        all_reward_styles.update(stats['reward_styles'])
        total_valid += stats['valid_rows']
        total_invalid += stats['invalid_rows']

    # Check consistency
    inconsistencies = []

    # All splits should have same required fields
    first_fields = set(all_stats[0]['field_presence'].keys())
    for stats in all_stats[1:]:
        fields = set(stats['field_presence'].keys())
        if fields != first_fields:
            inconsistencies.append(
                f"Field mismatch between {all_stats[0]['split_name']} and {stats['split_name']}"
            )

    # Print summary
    print(f"\nTotal splits analyzed: {len(all_stats)}")
    print(f"Total rows:            {format_number(total_valid + total_invalid)}")
    print(f"Valid rows:            {format_number(total_valid)} ({format_percentage(total_valid, total_valid + total_invalid)})")
    print(f"Invalid rows:          {format_number(total_invalid)} ({format_percentage(total_invalid, total_valid + total_invalid)})")

    print(f"\nUnique data_sources:   {len(all_data_sources)}")
    for ds in sorted(all_data_sources):
        print(f"  - {ds}")

    print(f"\nUnique abilities:      {len(all_abilities)}")
    for ability in sorted(all_abilities):
        print(f"  - {ability}")

    print(f"\nUnique prompt roles:   {len(all_prompt_roles)}")
    for role in sorted(all_prompt_roles):
        print(f"  - {role}")

    print(f"\nUnique reward styles:  {len(all_reward_styles)}")
    for style in sorted(all_reward_styles):
        print(f"  - {style}")

    # Check for inconsistencies
    if inconsistencies:
        print(f"\n⚠️  Inconsistencies found:")
        for issue in inconsistencies:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n✅ All splits have consistent schema!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Verify data schema consistency for unified dataset'
    )
    parser.add_argument(
        '--splits-dir',
        type=str,
        required=True,
        help='Directory containing split parquet files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information for each split'
    )

    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)

    if not splits_dir.exists():
        print(f"❌ Error: Splits directory not found: {splits_dir}")
        sys.exit(1)

    # Find all parquet files
    parquet_files = sorted(splits_dir.glob('*.parquet'))

    if not parquet_files:
        print(f"❌ Error: No parquet files found in {splits_dir}")
        sys.exit(1)

    print(f"{'=' * 70}")
    print(f"Schema Verification for Unified Dataset")
    print(f"{'=' * 70}")
    print(f"Splits directory: {splits_dir}")
    print(f"Found {len(parquet_files)} split(s)")

    # Analyze each split
    all_stats = []
    for parquet_file in parquet_files:
        split_name = parquet_file.stem
        stats = analyze_schema(parquet_file, split_name, verbose=args.verbose)
        all_stats.append(stats)

    # Verify consistency
    is_consistent = verify_consistency(all_stats, verbose=args.verbose)

    # Final verdict
    print(f"\n{'=' * 70}")
    if is_consistent and all(s['invalid_rows'] == 0 for s in all_stats):
        print("✅ Schema verification PASSED")
        print(f"{'=' * 70}\n")
        sys.exit(0)
    else:
        print("❌ Schema verification FAILED")
        print(f"{'=' * 70}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
