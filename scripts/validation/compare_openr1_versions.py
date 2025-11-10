#!/usr/bin/env python3
"""
Comprehensive comparison of OpenR1-Math cleaning approaches.

Compares:
1. Raw (225k samples, no processing)
2. New cleaned (190k samples, MathDatasetCleaner + hash dedup)
3. Existing v2.0 (53k samples, v2.0 cleaning + correctness filter + custom dedup)

Analyzes:
- Artifact removal effectiveness
- Deduplication differences
- Sample quality changes
- Statistical comparisons

Usage:
    python scripts/validation/compare_openr1_versions.py \
        --raw output/openr1-raw-verl/train.parquet \
        --new output/openr1-cleaned-deduped/train.parquet \
        --existing output/openr1-existing/train.parquet \
        --output output/openr1-comparison/
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text."""
    normalized = text.strip().lower()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def extract_problem_text(row: dict) -> str:
    """Extract problem text from VERL format."""
    if 'prompt' not in row:
        return ""
    prompt = row['prompt']
    if not isinstance(prompt, list) or len(prompt) == 0:
        return ""
    first_message = prompt[0]
    if isinstance(first_message, dict) and 'content' in first_message:
        return first_message['content']
    return ""


def detect_artifacts(text: str) -> Dict[str, bool]:
    """Detect various artifacts in problem text."""
    artifacts = {}

    # Problem numbers
    problem_number_patterns = [
        r'^\s*\d+\.',  # "6."
        r'^\s*\d+\.\d+',  # "8.3:"
        r'^\s*Problem\s+\d+',
        r'^\s*Question\s+\d+',
        r'^\s*Task\s+\d+',
        r'^\s*Example\s+\d+',
    ]
    artifacts['problem_number'] = any(re.search(p, text, re.MULTILINE | re.IGNORECASE) for p in problem_number_patterns)

    # Contest metadata
    contest_patterns = [
        r'\d{4}\s+\w+\s+Problem\s+\d+',  # "2004 AIME Problem 3"
        r'IMO\s+\d{4}',
        r'APMC\s+\d{4}',
        r'Olympiad',
    ]
    artifacts['contest_metadata'] = any(re.search(p, text, re.IGNORECASE) for p in contest_patterns)

    # Point allocations
    point_patterns = [
        r'\(\d+\s+points?\)',
        r'\[\d+\s+points?\]',
    ]
    artifacts['point_allocation'] = any(re.search(p, text, re.IGNORECASE) for p in point_patterns)

    # Markdown headers
    markdown_patterns = [
        r'^##\s+',
        r'^###\s+',
    ]
    artifacts['markdown_header'] = any(re.search(p, text, re.MULTILINE) for p in markdown_patterns)

    # Special artifacts
    artifacts['horizontal_rule'] = '---' in text or '___' in text
    artifacts['translation_note'] = 'translate' in text.lower() or 'translation' in text.lower()

    # URLs
    artifacts['url'] = bool(re.search(r'http[s]?://|www\.', text, re.IGNORECASE))

    # Image references
    image_patterns = [
        r'!\[.*?\]\(.*?\)',  # Markdown image
        r'\[asy\]',  # Asymptote code
        r'\bfigure\b',
        r'\bimage\b',
        r'\bdiagram\b',
    ]
    artifacts['image_reference'] = any(re.search(p, text, re.IGNORECASE) for p in image_patterns)

    return artifacts


def compare_problem_pair(raw_text: str, cleaned_text: str) -> Dict:
    """Compare raw and cleaned versions of the same problem."""
    raw_artifacts = detect_artifacts(raw_text)
    cleaned_artifacts = detect_artifacts(cleaned_text)

    removed_artifacts = {
        key: (raw_artifacts[key] and not cleaned_artifacts[key])
        for key in raw_artifacts
    }

    remaining_artifacts = {
        key: (raw_artifacts[key] and cleaned_artifacts[key])
        for key in raw_artifacts
    }

    return {
        'raw_text': raw_text,
        'cleaned_text': cleaned_text,
        'raw_artifacts': raw_artifacts,
        'cleaned_artifacts': cleaned_artifacts,
        'removed_artifacts': removed_artifacts,
        'remaining_artifacts': remaining_artifacts,
        'text_changed': raw_text != cleaned_text,
        'char_reduction': len(raw_text) - len(cleaned_text),
        'char_reduction_pct': 100 * (len(raw_text) - len(cleaned_text)) / len(raw_text) if len(raw_text) > 0 else 0,
    }


def analyze_datasets(
    raw_path: str,
    new_path: str,
    existing_path: str,
    num_samples: int = 100
) -> Dict:
    """Comprehensive analysis of three datasets."""
    print(f"\n{'='*70}")
    print(f"Loading datasets...")
    print(f"{'='*70}\n")

    # Load datasets
    print(f"Loading raw dataset: {raw_path}")
    raw_ds = load_dataset('parquet', data_files=raw_path, split='train')
    print(f"✓ Raw: {len(raw_ds):,} samples")

    print(f"\nLoading new cleaned dataset: {new_path}")
    new_ds = load_dataset('parquet', data_files=new_path, split='train')
    print(f"✓ New cleaned: {len(new_ds):,} samples")

    print(f"\nLoading existing v2.0 dataset: {existing_path}")
    existing_ds = load_dataset('parquet', data_files=existing_path, split='train')
    print(f"✓ Existing v2.0: {len(existing_ds):,} samples")

    # Create index mappings for faster lookup
    print(f"\n{'='*70}")
    print(f"Creating hash indices...")
    print(f"{'='*70}\n")

    print("Hashing raw samples...")
    raw_hash_to_idx = {}
    for idx, row in enumerate(raw_ds):
        problem_text = extract_problem_text(row)
        problem_hash = compute_hash(problem_text)
        raw_hash_to_idx[problem_hash] = idx
    print(f"✓ Raw: {len(raw_hash_to_idx):,} unique hashes")

    print("\nHashing new cleaned samples...")
    new_hash_to_idx = {}
    for idx, row in enumerate(new_ds):
        problem_text = extract_problem_text(row)
        problem_hash = compute_hash(problem_text)
        if problem_hash not in new_hash_to_idx:
            new_hash_to_idx[problem_hash] = []
        new_hash_to_idx[problem_hash].append(idx)
    print(f"✓ New: {len(new_hash_to_idx):,} unique hashes")

    print("\nHashing existing v2.0 samples...")
    existing_hash_to_idx = {}
    for idx, row in enumerate(existing_ds):
        problem_text = extract_problem_text(row)
        problem_hash = compute_hash(problem_text)
        if problem_hash not in existing_hash_to_idx:
            existing_hash_to_idx[problem_hash] = []
        existing_hash_to_idx[problem_hash].append(idx)
    print(f"✓ Existing: {len(existing_hash_to_idx):,} unique hashes")

    # Find overlapping samples
    print(f"\n{'='*70}")
    print(f"Finding overlaps...")
    print(f"{'='*70}\n")

    all_hashes = set(raw_hash_to_idx.keys())
    new_hashes = set(new_hash_to_idx.keys())
    existing_hashes = set(existing_hash_to_idx.keys())

    overlap_new_existing = new_hashes & existing_hashes
    only_in_new = new_hashes - existing_hashes
    only_in_existing = existing_hashes - new_hashes

    print(f"Overlap (new ∩ existing): {len(overlap_new_existing):,} samples")
    print(f"Only in new: {len(only_in_new):,} samples")
    print(f"Only in existing: {len(only_in_existing):,} samples")

    # Sample random problems for detailed comparison
    print(f"\n{'='*70}")
    print(f"Sampling {num_samples} random problems for detailed analysis...")
    print(f"{'='*70}\n")

    # Sample from overlap
    overlap_samples = random.sample(list(overlap_new_existing), min(num_samples // 2, len(overlap_new_existing)))

    # Sample from new-only
    new_only_samples = random.sample(list(only_in_new), min(num_samples // 4, len(only_in_new)))

    # Sample from existing-only
    existing_only_samples = random.sample(list(only_in_existing), min(num_samples // 4, len(only_in_existing)))

    # Analyze artifact removal effectiveness (compare RAW → NEW)
    # Match by extra_info.index field, not by hash (since cleaning changes hash!)
    print(f"Analyzing artifact removal (raw → new)...")
    artifact_comparisons = []

    # Build index mapping for new dataset
    new_index_to_idx = {}
    for idx, row in enumerate(new_ds):
        if 'extra_info' in row and isinstance(row['extra_info'], dict):
            orig_index = row['extra_info'].get('index')
            if orig_index is not None:
                new_index_to_idx[orig_index] = idx

    print(f"✓ Built index mapping for new dataset ({len(new_index_to_idx):,} samples)")

    # Sample random indices from raw dataset
    raw_sample_indices = random.sample(range(len(raw_ds)), min(num_samples, len(raw_ds)))

    for raw_idx in raw_sample_indices:
        raw_row = raw_ds[raw_idx]
        raw_text = extract_problem_text(raw_row)

        # Get original index from raw dataset
        raw_orig_index = raw_row.get('extra_info', {}).get('index') if isinstance(raw_row.get('extra_info'), dict) else None

        if raw_orig_index is None:
            raw_orig_index = raw_idx  # Fallback to array index

        # Find corresponding cleaned version in new dataset by index
        if raw_orig_index in new_index_to_idx:
            new_idx = new_index_to_idx[raw_orig_index]
            new_row = new_ds[new_idx]
            new_text = extract_problem_text(new_row)

            comparison = compare_problem_pair(raw_text, new_text)
            comparison['problem_hash'] = compute_hash(raw_text)
            comparison['raw_idx'] = raw_idx
            comparison['new_idx'] = new_idx
            comparison['orig_index'] = raw_orig_index
            comparison['in_existing'] = compute_hash(new_text) in existing_hashes

            artifact_comparisons.append(comparison)

    print(f"✓ Analyzed {len(artifact_comparisons)} raw → new problem pairs")

    # Aggregate artifact statistics
    print(f"\nAggregating artifact statistics...")

    artifact_removal_counts = Counter()
    artifact_remaining_counts = Counter()
    text_changed_count = 0
    total_char_reduction = 0

    for comp in artifact_comparisons:
        if comp['text_changed']:
            text_changed_count += 1
            total_char_reduction += comp['char_reduction']

        for artifact_type, removed in comp['removed_artifacts'].items():
            if removed:
                artifact_removal_counts[artifact_type] += 1

        for artifact_type, remaining in comp['remaining_artifacts'].items():
            if remaining:
                artifact_remaining_counts[artifact_type] += 1

    # Deduplication analysis
    print(f"\nAnalyzing deduplication...")

    raw_duplicates = len(raw_ds) - len(raw_hash_to_idx)
    new_duplicates = len(raw_ds) - len(new_ds)  # Comparing to raw
    existing_duplicates = len(raw_ds) - len(existing_ds)  # Comparing to raw

    print(f"✓ Raw duplicates: {raw_duplicates:,}")
    print(f"✓ New duplicates removed: {new_duplicates:,}")
    print(f"✓ Existing duplicates removed: {existing_duplicates:,}")

    # Quality filtering analysis
    print(f"\nAnalyzing quality filtering...")

    # Check URL presence
    url_count_new = sum(1 for row in new_ds if any('http' in extract_problem_text(row).lower() or 'www.' in extract_problem_text(row).lower() for _ in [0]))
    url_count_existing = sum(1 for row in existing_ds if any('http' in extract_problem_text(row).lower() or 'www.' in extract_problem_text(row).lower() for _ in [0]))

    print(f"✓ URLs in new: {url_count_new}")
    print(f"✓ URLs in existing: {url_count_existing}")

    # Compile results
    results = {
        'dataset_sizes': {
            'raw': len(raw_ds),
            'new_cleaned': len(new_ds),
            'existing_v2': len(existing_ds),
        },
        'overlaps': {
            'new_and_existing': len(overlap_new_existing),
            'only_in_new': len(only_in_new),
            'only_in_existing': len(only_in_existing),
        },
        'deduplication': {
            'raw_duplicates': raw_duplicates,
            'new_duplicates_removed': new_duplicates,
            'existing_duplicates_removed': existing_duplicates,
            'new_dedup_rate': 100 * new_duplicates / len(raw_ds) if len(raw_ds) > 0 else 0,
            'existing_dedup_rate': 100 * existing_duplicates / len(raw_ds) if len(raw_ds) > 0 else 0,
        },
        'artifact_removal': {
            'samples_analyzed': len(artifact_comparisons),
            'samples_modified': text_changed_count,
            'modification_rate': 100 * text_changed_count / len(artifact_comparisons) if artifact_comparisons else 0,
            'total_char_reduction': total_char_reduction,
            'avg_char_reduction': total_char_reduction / text_changed_count if text_changed_count > 0 else 0,
            'removed_counts': dict(artifact_removal_counts),
            'remaining_counts': dict(artifact_remaining_counts),
        },
        'quality_metrics': {
            'url_count_new': url_count_new,
            'url_count_existing': url_count_existing,
        },
        'sample_comparisons': artifact_comparisons[:20],  # Keep first 20 for report
    }

    return results


def generate_report(results: Dict, output_dir: str):
    """Generate comprehensive comparison report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
{'='*70}
OpenR1-Math Cleaning Comparison Report
{'='*70}
Generated: {timestamp}

{'='*70}
1. Dataset Sizes
{'='*70}
Raw (no processing):           {results['dataset_sizes']['raw']:,} samples
New cleaned (MathDatasetCleaner): {results['dataset_sizes']['new_cleaned']:,} samples
Existing v2.0 (Hub version):   {results['dataset_sizes']['existing_v2']:,} samples

Reduction from raw:
  New:      {results['dataset_sizes']['raw'] - results['dataset_sizes']['new_cleaned']:,} samples ({100*(results['dataset_sizes']['raw'] - results['dataset_sizes']['new_cleaned'])/results['dataset_sizes']['raw']:.1f}%)
  Existing: {results['dataset_sizes']['raw'] - results['dataset_sizes']['existing_v2']:,} samples ({100*(results['dataset_sizes']['raw'] - results['dataset_sizes']['existing_v2'])/results['dataset_sizes']['raw']:.1f}%)

{'='*70}
2. Dataset Overlap Analysis
{'='*70}
Samples in both (new ∩ existing):  {results['overlaps']['new_and_existing']:,}
Samples only in new:                {results['overlaps']['only_in_new']:,}
Samples only in existing:           {results['overlaps']['only_in_existing']:,}

Overlap rate: {100*results['overlaps']['new_and_existing']/results['dataset_sizes']['new_cleaned']:.1f}% of new
             {100*results['overlaps']['new_and_existing']/results['dataset_sizes']['existing_v2']:.1f}% of existing

{'='*70}
3. Deduplication Comparison
{'='*70}
Raw duplicates (baseline):         {results['deduplication']['raw_duplicates']:,}

New cleaning approach:
  Duplicates removed:              {results['deduplication']['new_duplicates_removed']:,}
  Deduplication rate:              {results['deduplication']['new_dedup_rate']:.2f}%

Existing v2.0 approach:
  Duplicates removed:              {results['deduplication']['existing_duplicates_removed']:,}
  Deduplication rate:              {results['deduplication']['existing_dedup_rate']:.2f}%

Difference:
  Extra duplicates removed (existing): {results['deduplication']['existing_duplicates_removed'] - results['deduplication']['new_duplicates_removed']:,}

Note: Existing v2.0 also applies correctness filtering (removes ~76% of samples),
      while new approach only applies cleaning + deduplication.

{'='*70}
4. Artifact Removal Effectiveness
{'='*70}
Samples analyzed:                  {results['artifact_removal']['samples_analyzed']}
Samples modified:                  {results['artifact_removal']['samples_modified']}
Modification rate:                 {results['artifact_removal']['modification_rate']:.1f}%

Character reduction:
  Total characters removed:        {results['artifact_removal']['total_char_reduction']:,}
  Average per modified sample:     {results['artifact_removal']['avg_char_reduction']:.1f} chars

Artifacts Removed (from analyzed samples):
"""

    for artifact_type, count in sorted(results['artifact_removal']['removed_counts'].items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / results['artifact_removal']['samples_analyzed'] if results['artifact_removal']['samples_analyzed'] > 0 else 0
        report += f"  {artifact_type:25s}: {count:4d} ({pct:5.1f}%)\n"

    report += f"\nArtifacts Still Remaining (after cleaning):\n"

    for artifact_type, count in sorted(results['artifact_removal']['remaining_counts'].items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / results['artifact_removal']['samples_analyzed'] if results['artifact_removal']['samples_analyzed'] > 0 else 0
        report += f"  {artifact_type:25s}: {count:4d} ({pct:5.1f}%)\n"

    report += f"""
{'='*70}
5. Quality Metrics
{'='*70}
URL presence:
  New cleaned:                     {results['quality_metrics']['url_count_new']} samples
  Existing v2.0:                   {results['quality_metrics']['url_count_existing']} samples

Note: Neither version filters URL samples in the cleaning preset.

{'='*70}
6. Sample Comparisons
{'='*70}
"""

    # Add sample comparisons
    for i, comp in enumerate(results['sample_comparisons'][:10], 1):
        if not comp['text_changed']:
            continue

        report += f"\n{'-'*70}\n"
        report += f"Sample {i} (Hash: {comp['problem_hash'][:16]}...)\n"
        report += f"{'-'*70}\n"

        report += f"REMOVED ARTIFACTS: {', '.join(k for k, v in comp['removed_artifacts'].items() if v) or 'None'}\n"
        report += f"REMAINING ARTIFACTS: {', '.join(k for k, v in comp['remaining_artifacts'].items() if v) or 'None'}\n"
        report += f"CHARACTER REDUCTION: {comp['char_reduction']} chars ({comp['char_reduction_pct']:.1f}%)\n\n"

        report += f"RAW:\n{comp['raw_text'][:500]}...\n\n"
        report += f"CLEANED:\n{comp['cleaned_text'][:500]}...\n"

    report += f"\n{'='*70}\n"
    report += f"End of Report\n"
    report += f"{'='*70}\n"

    # Save report
    report_file = output_path / "comparison_report.md"
    report_file.write_text(report)
    print(f"\n✓ Report saved to: {report_file}")

    # Save JSON statistics
    stats_file = output_path / "statistics.json"
    with open(stats_file, 'w') as f:
        # Remove sample_comparisons from JSON (too large)
        stats_copy = dict(results)
        stats_copy.pop('sample_comparisons', None)
        json.dump(stats_copy, f, indent=2)
    print(f"✓ Statistics saved to: {stats_file}")

    # Save sample comparisons separately
    samples_file = output_path / "artifact_examples.json"
    with open(samples_file, 'w') as f:
        json.dump(results['sample_comparisons'], f, indent=2)
    print(f"✓ Sample comparisons saved to: {samples_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare OpenR1-Math cleaning approaches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python scripts/validation/compare_openr1_versions.py

  # Custom paths
  python scripts/validation/compare_openr1_versions.py \
      --raw output/openr1-raw-verl/train.parquet \
      --new output/openr1-cleaned-deduped/train.parquet \
      --existing output/openr1-existing/train.parquet
        """
    )

    parser.add_argument(
        '--raw',
        type=str,
        default='output/openr1-raw-verl/train.parquet',
        help='Path to raw dataset'
    )

    parser.add_argument(
        '--new',
        type=str,
        default='output/openr1-cleaned-deduped/train.parquet',
        help='Path to new cleaned dataset'
    )

    parser.add_argument(
        '--existing',
        type=str,
        default='output/openr1-existing/train.parquet',
        help='Path to existing v2.0 dataset'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output/openr1-comparison/',
        help='Output directory for comparison results'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to analyze in detail (default: 100)'
    )

    args = parser.parse_args()

    # Run analysis
    print(f"\n{'='*70}")
    print(f"OpenR1-Math Cleaning Comparison")
    print(f"{'='*70}")
    print(f"Raw:      {args.raw}")
    print(f"New:      {args.new}")
    print(f"Existing: {args.existing}")
    print(f"Output:   {args.output}")
    print(f"{'='*70}\n")

    results = analyze_datasets(
        raw_path=args.raw,
        new_path=args.new,
        existing_path=args.existing,
        num_samples=args.num_samples,
    )

    # Generate report
    generate_report(results, args.output)

    print(f"\n{'='*70}")
    print(f"✅ Comparison complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
