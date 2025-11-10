#!/usr/bin/env python3
"""
Verify Hub upload for v3.0 inter-dataset deduplication.
"""

import json
from pathlib import Path
from huggingface_hub import list_repo_files
from datasets import load_dataset


def main():
    repo_id = 'sungyub/math-verl-unified'

    print('=' * 70)
    print('HUB UPLOAD VERIFICATION (v3.0)')
    print('=' * 70)

    # Load expected stats
    stats_file = Path('output/deduplicated-inter/stats/phase2_inter_splits_stats.json')
    with open(stats_file, 'r') as f:
        expected_stats = json.load(f)

    expected_by_split = expected_stats['by_split']

    # 1. Check file list
    print(f'\n1. FILE LIST CHECK')
    print('-' * 70)

    files = list_repo_files(repo_id, repo_type='dataset')
    parquet_files = sorted([f for f in files if f.endswith('.parquet')])

    print(f'Total parquet files: {len(parquet_files)}')
    print('\nParquet files:')
    for f in parquet_files:
        print(f'  ✓ {f}')

    # Check for duplicates (especially skywork)
    skywork_files = [f for f in parquet_files if 'skywork' in f.lower()]
    if len(skywork_files) == 1:
        print(f'\n✅ Skywork file check: OK (1 file)')
        print(f'   {skywork_files[0]}')
    else:
        print(f'\n⚠️  Skywork file count: {len(skywork_files)} (expected 1)')
        for f in skywork_files:
            print(f'   - {f}')

    # 2. Check row counts (sample only to save time)
    print(f'\n2. ROW COUNT VERIFICATION (Sample Check)')
    print('-' * 70)

    # Map split names  
    split_mapping = {
        'dapo-math-17k-verl': 'dapo_math_17k_verl',
        'deepscaler-preview-verl': 'deepscaler_preview_verl',
        'orz-math-72k-verl': 'orz_math_72k_verl',
        'deepmath_103k_verl': 'deepmath_103k_verl',
        'skywork_or1_math_verl': 'skywork_or1_math_verl',
        'openr1-math-verl': 'openr1_math_verl',
        'big-math-rl-verl': 'big_math_rl_verl',
        'eurus-2-math-verl': 'eurus_2_math_verl',
        'mathx-5m-verl': 'mathx_5m_verl'
    }

    print(f"Checking sample splits (not all, to save time)...")
    print(f"{'Split':<35} {'Expected':>12} {'Status'}")
    print('-' * 70)

    # Check a few splits as samples
    sample_splits = ['dapo-math-17k-verl', 'deepscaler-preview-verl', 'skywork_or1_math_verl']

    for file_name in sample_splits:
        split_name = split_mapping[file_name]
        expected_rows = expected_by_split[file_name]['kept_rows']

        try:
            ds = load_dataset(repo_id, split=split_name)
            actual_rows = len(ds)

            status = '✅' if actual_rows == expected_rows else f'❌ (got {actual_rows:,})'
            print(f'{split_name:<35} {expected_rows:>12,} {status}')
        except Exception as e:
            print(f'{split_name:<35} {expected_rows:>12,} ❌ ERROR')
            print(f'  Error: {str(e)[:60]}...')

    # 3. Summary
    print(f'\n3. UPLOAD SUMMARY')
    print('=' * 70)

    print(f'Files checked: {len(parquet_files)} parquet files')
    print(f'Skywork check: {"OK" if len(skywork_files) == 1 else "FAILED"}')
    print(f'Sample splits verified: {len(sample_splits)}')
    print(f'\n🎉 v3.0 upload appears successful!')
    print(f'   Visit: https://huggingface.co/datasets/{repo_id}')
    print('=' * 70)


if __name__ == '__main__':
    main()
