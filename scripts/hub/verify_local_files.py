#!/usr/bin/env python3
"""
Verify local deduplicated files before Hub upload.
"""

import pyarrow.parquet as pq
import json
from pathlib import Path


def main():
    dedup_dir = Path('output/deduplicated-inter/data')

    print('=' * 70)
    print('LOCAL DEDUPLICATED FILES VERIFICATION')
    print('=' * 70)

    files = sorted(dedup_dir.glob('*.parquet'))
    total_rows = 0
    total_size_mb = 0

    print(f'\nFound {len(files)} parquet files:\n')
    print(f"{'File':<35} {'Rows':>12} {'Size (MB)':>12}")
    print('-' * 70)

    file_stats = {}
    for f in files:
        table = pq.read_table(f)
        rows = len(table)
        size_mb = f.stat().st_size / (1024 * 1024)
        total_rows += rows
        total_size_mb += size_mb

        filename = f.name
        file_stats[filename] = {'rows': rows, 'size_mb': size_mb}
        print(f'{filename:<35} {rows:>12,} {size_mb:>12.1f}')

    print('-' * 70)
    print(f"{'TOTAL':<35} {total_rows:>12,} {total_size_mb:>12.1f}")

    # Check for macOS system files
    system_files = list(dedup_dir.glob('._*')) + list(dedup_dir.glob('.DS_Store'))
    if system_files:
        print(f'\n⚠️  WARNING: Found {len(system_files)} macOS system files:')
        for f in system_files:
            print(f'  - {f.name}')
    else:
        print('\n✅ No macOS system files found')

    # Print schema check
    print('\n' + '=' * 70)
    print('SCHEMA VERIFICATION (first file)')
    print('=' * 70)
    first_file = files[0]
    table = pq.read_table(first_file)
    print(f'\nFile: {first_file.name}')
    print(f'Schema:\n{table.schema}')

    # Save stats for README
    stats_file = Path('output/dedup-stats-for-readme.json')
    with open(stats_file, 'w') as f:
        json.dump(file_stats, f, indent=2)
    print(f'\n✅ Stats saved to: {stats_file}')


if __name__ == '__main__':
    main()
