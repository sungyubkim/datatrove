#!/usr/bin/env python3
"""
Data Integrity Verification Tool

Verifies that problem text in parquet files is not truncated.
Checks specific examples that appear truncated in reports.

Usage:
    python scripts/validation/verify_data_integrity.py \
        --files output/dapo-math-verl-converted/train.parquet \
                output/dapo-math-cleaned-deduped/train.parquet \
                output/dapo-math-original/train.parquet
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from datasets import load_dataset


def extract_problem_text(row: dict) -> str:
    """Extract problem text from VERL format row."""
    if 'prompt' not in row:
        return ""

    prompt = row['prompt']
    if not isinstance(prompt, list) or len(prompt) == 0:
        return ""

    first_message = prompt[0]
    if isinstance(first_message, dict) and 'content' in first_message:
        return first_message['content']

    return ""


def verify_file_integrity(file_path: str) -> dict:
    """Verify data integrity for a single parquet file.

    Args:
        file_path: Path to parquet file

    Returns:
        Statistics dictionary
    """
    print(f"\n{'='*70}")
    print(f"Verifying: {file_path}")
    print(f"{'='*70}")

    # Load dataset
    try:
        dataset = load_dataset("parquet", data_files=file_path, split="train")
        print(f"✓ Loaded {len(dataset):,} samples")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return {}

    # Statistics
    stats = {
        'total_samples': len(dataset),
        'length_distribution': defaultdict(int),
        'samples_over_500': 0,
        'samples_over_1000': 0,
        'samples_over_2000': 0,
        'min_length': float('inf'),
        'max_length': 0,
        'total_length': 0,
        'suspicious_samples': [],
    }

    # Suspicious patterns that might indicate truncation
    suspicious_patterns = [
        'for(int i = 1',
        'Please provide t',
        '[/asy]The answer is in the',
        'students enr',
        'for(real i=ceil',
    ]

    # Check each sample
    for idx, row in enumerate(dataset):
        problem_text = extract_problem_text(row)
        length = len(problem_text)

        # Update statistics
        stats['length_distribution'][length // 100 * 100] += 1  # Bucket by 100s
        stats['total_length'] += length
        stats['min_length'] = min(stats['min_length'], length)
        stats['max_length'] = max(stats['max_length'], length)

        if length > 500:
            stats['samples_over_500'] += 1
        if length > 1000:
            stats['samples_over_1000'] += 1
        if length > 2000:
            stats['samples_over_2000'] += 1

        # Check for suspicious truncation patterns
        for pattern in suspicious_patterns:
            if problem_text.endswith(pattern):
                ground_truth = row.get('reward_model', {}).get('ground_truth', 'N/A') if isinstance(row.get('reward_model'), dict) else 'N/A'
                stats['suspicious_samples'].append({
                    'index': idx,
                    'length': length,
                    'ending': problem_text[-100:],
                    'full_text': problem_text,
                    'ground_truth': ground_truth,
                    'pattern': pattern,
                })
                break

    # Calculate average
    stats['avg_length'] = stats['total_length'] / stats['total_samples'] if stats['total_samples'] > 0 else 0

    return stats


def find_specific_examples(file_path: str, search_patterns: list) -> dict:
    """Find specific examples that contain certain patterns.

    Args:
        file_path: Path to parquet file
        search_patterns: List of text patterns to search for

    Returns:
        Dictionary of found examples
    """
    print(f"\n{'='*70}")
    print(f"Searching for specific examples in: {file_path}")
    print(f"{'='*70}")

    # Load dataset
    try:
        dataset = load_dataset("parquet", data_files=file_path, split="train")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return {}

    found_examples = {}

    for idx, row in enumerate(dataset):
        problem_text = extract_problem_text(row)
        ground_truth = row.get('reward_model', {}).get('ground_truth', 'N/A') if isinstance(row.get('reward_model'), dict) else 'N/A'

        for pattern in search_patterns:
            if pattern in problem_text:
                if pattern not in found_examples:
                    found_examples[pattern] = []

                found_examples[pattern].append({
                    'index': idx,
                    'length': len(problem_text),
                    'problem': problem_text,
                    'ground_truth': ground_truth,
                })

                # Only keep first 3 examples per pattern
                if len(found_examples[pattern]) >= 3:
                    break

    return found_examples


def generate_report(file_stats: dict, file_paths: list) -> str:
    """Generate integrity verification report.

    Args:
        file_stats: Dictionary mapping file paths to statistics
        file_paths: List of file paths checked

    Returns:
        Report text
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
{'='*70}
Data Integrity Verification Report
{'='*70}
Timestamp: {timestamp}
Files Checked: {len(file_paths)}

{'='*70}
Summary by File
{'='*70}
"""

    for file_path in file_paths:
        stats = file_stats.get(file_path, {})
        if not stats:
            continue

        file_name = Path(file_path).parent.name
        report += f"\n{'-'*70}\n"
        report += f"File: {file_name}\n"
        report += f"{'-'*70}\n"
        report += f"Total samples:           {stats['total_samples']:,}\n"
        report += f"Min length:              {stats['min_length']:,} characters\n"
        report += f"Max length:              {stats['max_length']:,} characters\n"
        report += f"Average length:          {stats['avg_length']:.0f} characters\n\n"
        report += f"Samples > 500 chars:     {stats['samples_over_500']:,} ({100*stats['samples_over_500']/stats['total_samples']:.1f}%)\n"
        report += f"Samples > 1000 chars:    {stats['samples_over_1000']:,} ({100*stats['samples_over_1000']/stats['total_samples']:.1f}%)\n"
        report += f"Samples > 2000 chars:    {stats['samples_over_2000']:,} ({100*stats['samples_over_2000']/stats['total_samples']:.1f}%)\n\n"

        if stats['suspicious_samples']:
            report += f"⚠️  SUSPICIOUS: {len(stats['suspicious_samples'])} samples with truncation patterns\n"
        else:
            report += f"✅ NO suspicious truncation patterns found\n"

    # Detailed suspicious samples
    for file_path in file_paths:
        stats = file_stats.get(file_path, {})
        if not stats or not stats['suspicious_samples']:
            continue

        file_name = Path(file_path).parent.name
        report += f"\n{'='*70}\n"
        report += f"Suspicious Samples in {file_name}\n"
        report += f"{'='*70}\n"

        for i, sample in enumerate(stats['suspicious_samples'], 1):
            report += f"\n{'-'*70}\n"
            report += f"Suspicious Sample {i} (Index #{sample['index']})\n"
            report += f"{'-'*70}\n"
            report += f"Length: {sample['length']} characters\n"
            report += f"Pattern detected: '{sample['pattern']}'\n"
            report += f"Ending (last 100 chars): ...{sample['ending']}\n\n"
            report += f"FULL TEXT:\n{sample['full_text']}\n\n"
            report += f"Ground Truth: {sample['ground_truth']}\n"

    # Verdict
    report += f"\n{'='*70}\n"
    report += f"VERDICT\n"
    report += f"{'='*70}\n"

    has_suspicious = any(file_stats.get(fp, {}).get('suspicious_samples', []) for fp in file_paths)

    if has_suspicious:
        report += f"⚠️  WARNING: Suspicious truncation patterns detected!\n"
        report += f"Some samples appear to end abruptly with incomplete text.\n"
        report += f"This suggests actual data truncation, not just display truncation.\n\n"
        report += f"RECOMMENDATION: Investigate source data and conversion process.\n"
    else:
        report += f"✅ ALL CLEAR: No suspicious truncation patterns detected.\n"
        report += f"All samples appear to have complete text.\n"
        report += f"The truncation seen in reports is display-only ([:500] slicing).\n\n"
        report += f"RECOMMENDATION: Update report scripts to show more text.\n"

    report += f"\n{'='*70}\n"

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Verify data integrity of parquet files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--files',
        type=str,
        nargs='+',
        required=True,
        help='Parquet files to verify'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./output/verification/',
        help='Output directory for verification report'
    )

    parser.add_argument(
        '--search-patterns',
        type=str,
        nargs='*',
        default=['Regular hexagon', 'orthocenter', 'rectangle $ABCD$', 'equation $y ='],
        help='Specific text patterns to search for'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify each file
    file_stats = {}
    for file_path in args.files:
        if not Path(file_path).exists():
            print(f"⚠️  File not found: {file_path}")
            continue

        stats = verify_file_integrity(file_path)
        file_stats[file_path] = stats

    # Search for specific examples
    print(f"\n{'='*70}")
    print(f"Searching for Specific Patterns")
    print(f"{'='*70}")

    all_found_examples = {}
    for file_path in args.files:
        if not Path(file_path).exists():
            continue

        found = find_specific_examples(file_path, args.search_patterns)
        if found:
            all_found_examples[file_path] = found

            for pattern, examples in found.items():
                print(f"\n✓ Found {len(examples)} examples with '{pattern[:30]}...'")
                for ex in examples:
                    print(f"  - Index {ex['index']}: {ex['length']} chars")

    # Generate report
    report = generate_report(file_stats, args.files)

    # Print report
    print(report)

    # Save report
    report_path = output_dir / "data_integrity_report.txt"
    report_path.write_text(report)
    print(f"✓ Report saved to: {report_path}")

    # Save detailed examples
    if all_found_examples:
        examples_path = output_dir / "detailed_examples.json"
        with open(examples_path, "w") as f:
            # Convert to serializable format
            serializable = {}
            for file_path, patterns in all_found_examples.items():
                file_name = Path(file_path).parent.name
                serializable[file_name] = {}
                for pattern, examples in patterns.items():
                    serializable[file_name][pattern] = [
                        {
                            'index': ex['index'],
                            'length': ex['length'],
                            'problem': ex['problem'][:1000],  # First 1000 chars
                            'ground_truth': ex['ground_truth'],
                        }
                        for ex in examples
                    ]

            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"✓ Detailed examples saved to: {examples_path}")

    # Save statistics
    stats_path = output_dir / "integrity_stats.json"
    with open(stats_path, "w") as f:
        serializable_stats = {}
        for file_path, stats in file_stats.items():
            file_name = Path(file_path).parent.name
            # Remove non-serializable data
            clean_stats = {k: v for k, v in stats.items() if k != 'suspicious_samples'}
            clean_stats['num_suspicious'] = len(stats.get('suspicious_samples', []))
            clean_stats['length_distribution'] = dict(stats.get('length_distribution', {}))
            serializable_stats[file_name] = clean_stats

        json.dump({
            "timestamp": datetime.now().isoformat(),
            "files_checked": [Path(fp).parent.name for fp in args.files],
            "stats": serializable_stats,
        }, f, indent=2)
    print(f"✓ Statistics saved to: {stats_path}")

    print(f"\n{'='*70}")
    print(f"✅ Verification completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
