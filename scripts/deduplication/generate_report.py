#!/usr/bin/env python3
"""
Generate Deduplication Report

Creates comprehensive reports from deduplication statistics.
"""

import argparse
import os
import sys
import json
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import (
    format_number,
    format_percentage,
    format_bytes,
    load_stats
)


def generate_markdown_report(
    phase1_stats_file: str,
    phase2_stats_file: str,
    output_file: str
) -> None:
    """
    Generate comprehensive Markdown report.

    Args:
        phase1_stats_file: Path to Phase 1 summary JSON
        phase2_stats_file: Path to Phase 2 stats JSON
        output_file: Path to save markdown report
    """
    # Load statistics
    phase1_stats = load_stats(phase1_stats_file) if os.path.exists(phase1_stats_file) else None
    phase2_stats = load_stats(phase2_stats_file) if os.path.exists(phase2_stats_file) else None

    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("# Dataset Deduplication Report\n\n")
        f.write(f"**Generated:** {phase2_stats.get('timestamp', 'N/A') if phase2_stats else 'N/A'}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")

        if phase1_stats and phase2_stats:
            original_rows = phase1_stats.get('total_input_rows', 0)
            final_rows = phase2_stats.get('total_output_rows', 0)
            total_removed = original_rows - final_rows
            reduction_rate = total_removed / original_rows if original_rows > 0 else 0

            f.write(f"- **Original dataset size:** {format_number(original_rows)} problems\n")
            f.write(f"- **Final dataset size:** {format_number(final_rows)} problems\n")
            f.write(f"- **Total removed:** {format_number(total_removed)} duplicates ({reduction_rate * 100:.1f}%)\n")
            f.write(f"- **Processing time:** {(phase1_stats.get('total_duration_seconds', 0) + phase2_stats.get('processing', {}).get('duration_seconds', 0)) / 60:.1f} minutes\n")
            f.write("\n")

        # Phase 1 Results
        f.write("---\n\n")
        f.write("## Phase 1: Intra-Dataset Deduplication\n\n")

        if phase1_stats:
            f.write("Removed duplicates within each dataset independently.\n\n")

            f.write("| Dataset | Input Rows | Output Rows | Duplicates | Rate |\n")
            f.write("|---------|------------|-------------|------------|------|\n")

            for ds_stats in phase1_stats.get('individual_stats', []):
                dataset_name = ds_stats.get('dataset', 'Unknown')
                input_rows = ds_stats.get('input', {}).get('total_rows', 0)
                output_rows = ds_stats.get('output', {}).get('unique_rows', 0)
                duplicates = ds_stats.get('deduplication', {}).get('duplicates_found', 0)
                dup_rate = ds_stats.get('deduplication', {}).get('duplicate_rate', 0)

                f.write(f"| {dataset_name} | {format_number(input_rows)} | {format_number(output_rows)} | {format_number(duplicates)} | {dup_rate * 100:.2f}% |\n")

            f.write("\n")

            # Phase 1 Summary
            total_input = phase1_stats.get('total_input_rows', 0)
            total_output = phase1_stats.get('total_output_rows', 0)
            total_dups = phase1_stats.get('total_duplicates', 0)

            f.write(f"**Phase 1 Summary:**\n")
            f.write(f"- Total input: {format_number(total_input)}\n")
            f.write(f"- Total output: {format_number(total_output)}\n")
            f.write(f"- Duplicates removed: {format_number(total_dups)} ({format_percentage(total_dups, total_input)})\n")
            f.write(f"- Processing time: {phase1_stats.get('total_duration_seconds', 0) / 60:.1f} minutes\n")
            f.write("\n")

        # Phase 2 Results
        f.write("---\n\n")
        f.write("## Phase 2: Inter-Dataset Deduplication\n\n")

        if phase2_stats:
            f.write("Removed duplicates across datasets using priority order.\n\n")

            # Priority order
            f.write("**Priority Order:**\n")
            for i, ds in enumerate(phase2_stats.get('priority_order', []), 1):
                f.write(f"{i}. {ds}\n")
            f.write("\n")

            # Results by dataset
            f.write("| Dataset | Input | Kept | Removed | Rate |\n")
            f.write("|---------|-------|------|---------|------|\n")

            by_dataset = phase2_stats.get('by_dataset', {})
            for dataset_name, ds_stats in by_dataset.items():
                input_rows = ds_stats.get('input_rows', 0)
                kept_rows = ds_stats.get('kept_rows', 0)
                removed = ds_stats.get('removed_duplicates', 0)
                dup_rate = ds_stats.get('duplicate_rate', 0)

                f.write(f"| {dataset_name} | {format_number(input_rows)} | {format_number(kept_rows)} | {format_number(removed)} | {dup_rate * 100:.2f}% |\n")

            f.write("\n")

            # Phase 2 Summary
            total_input = phase2_stats.get('total_input_rows', 0)
            total_output = phase2_stats.get('total_output_rows', 0)
            total_removed = phase2_stats.get('total_duplicates_removed', 0)

            f.write(f"**Phase 2 Summary:**\n")
            f.write(f"- Total input: {format_number(total_input)}\n")
            f.write(f"- Total output: {format_number(total_output)}\n")
            f.write(f"- Duplicates removed: {format_number(total_removed)} ({format_percentage(total_removed, total_input)})\n")
            f.write(f"- Processing time: {phase2_stats.get('processing', {}).get('duration_seconds', 0) / 60:.1f} minutes\n")
            f.write("\n")

            # Cross-dataset duplicates
            f.write("### Cross-Dataset Duplicate Sources\n\n")
            f.write("Shows which higher-priority datasets contributed duplicates:\n\n")

            for dataset_name, ds_stats in by_dataset.items():
                sources = ds_stats.get('duplicate_sources', {})
                if sources:
                    f.write(f"**{dataset_name}:**\n")
                    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"- {source}: {format_number(count)} duplicates\n")
                    f.write("\n")

        # Overall Summary
        f.write("---\n\n")
        f.write("## Overall Summary\n\n")

        if phase1_stats and phase2_stats:
            original_rows = phase1_stats.get('total_input_rows', 0)
            final_rows = phase2_stats.get('total_output_rows', 0)

            f.write(f"```\n")
            f.write(f"Original:  {format_number(original_rows)} problems\n")
            f.write(f"           ↓\n")
            f.write(f"Phase 1:   {format_number(phase1_stats.get('total_output_rows', 0))} problems ({format_percentage(phase1_stats.get('total_duplicates', 0), original_rows)} removed)\n")
            f.write(f"           ↓\n")
            f.write(f"Phase 2:   {format_number(final_rows)} problems ({format_percentage(phase2_stats.get('total_duplicates_removed', 0), phase1_stats.get('total_output_rows', 1))} removed)\n")
            f.write(f"           ↓\n")
            f.write(f"Final:     {format_number(final_rows)} unique problems\n")
            f.write(f"\n")
            f.write(f"Total reduction: {format_percentage(original_rows - final_rows, original_rows)}\n")
            f.write(f"```\n\n")

        # Validation Status
        f.write("---\n\n")
        f.write("## Validation\n\n")
        f.write("To validate the deduplicated datasets, run:\n\n")
        f.write("```bash\n")
        f.write("python scripts/validate.py --check-format --check-collisions --check-counts\n")
        f.write("```\n\n")

        # Usage
        f.write("---\n\n")
        f.write("## Using the Deduplicated Dataset\n\n")
        f.write("```python\n")
        f.write("from datasets import load_dataset\n\n")
        f.write("# Load from local directory\n")
        f.write('dataset = load_dataset("parquet", data_dir="_deduplicated/phase2-inter/combined/data")\n\n')
        f.write("# Or upload to HuggingFace Hub and load\n")
        f.write('# dataset = load_dataset("your-username/math-deduplicated")\n')
        f.write("```\n")

    print(f"✅ Report generated: {output_file}")


def generate_json_summary(
    phase1_stats_file: str,
    phase2_stats_file: str,
    output_file: str
) -> None:
    """
    Generate JSON summary combining both phases.

    Args:
        phase1_stats_file: Path to Phase 1 summary JSON
        phase2_stats_file: Path to Phase 2 stats JSON
        output_file: Path to save JSON summary
    """
    # Load statistics
    phase1_stats = load_stats(phase1_stats_file) if os.path.exists(phase1_stats_file) else {}
    phase2_stats = load_stats(phase2_stats_file) if os.path.exists(phase2_stats_file) else {}

    # Calculate overall metrics
    original_rows = phase1_stats.get('total_input_rows', 0)
    final_rows = phase2_stats.get('total_output_rows', 0)
    total_removed = original_rows - final_rows

    summary = {
        'summary': {
            'original_total_rows': original_rows,
            'final_total_rows': final_rows,
            'total_removed': total_removed,
            'reduction_rate': total_removed / original_rows if original_rows > 0 else 0
        },
        'phase1_summary': {
            'input_rows': phase1_stats.get('total_input_rows', 0),
            'output_rows': phase1_stats.get('total_output_rows', 0),
            'duplicates_removed': phase1_stats.get('total_duplicates', 0),
            'duplicate_rate': phase1_stats.get('duplicate_rate', 0),
            'duration_minutes': phase1_stats.get('total_duration_seconds', 0) / 60
        },
        'phase2_summary': {
            'input_rows': phase2_stats.get('total_input_rows', 0),
            'output_rows': phase2_stats.get('total_output_rows', 0),
            'duplicates_removed': phase2_stats.get('total_duplicates_removed', 0),
            'duplicate_rate': phase2_stats.get('duplicate_rate', 0),
            'duration_minutes': phase2_stats.get('processing', {}).get('duration_seconds', 0) / 60
        },
        'phase1_details': phase1_stats,
        'phase2_details': phase2_stats
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON summary generated: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate deduplication reports',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--phase1-stats',
        type=str,
        default='_deduplicated/phase1-intra/phase1_summary.json',
        help='Phase 1 summary JSON file'
    )

    parser.add_argument(
        '--phase2-stats',
        type=str,
        default='_deduplicated/phase2-inter/stats/phase2_stats.json',
        help='Phase 2 stats JSON file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='_deduplicated/reports',
        help='Output directory for reports'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['markdown', 'json', 'both'],
        default='both',
        help='Report format (default: both)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("GENERATING DEDUPLICATION REPORTS")
    print("=" * 70)

    # Generate reports
    if args.format in ['markdown', 'both']:
        markdown_file = os.path.join(args.output_dir, 'deduplication_report.md')
        generate_markdown_report(
            args.phase1_stats,
            args.phase2_stats,
            markdown_file
        )

    if args.format in ['json', 'both']:
        json_file = os.path.join(args.output_dir, 'deduplication_summary.json')
        generate_json_summary(
            args.phase1_stats,
            args.phase2_stats,
            json_file
        )

    print("\n✅ Report generation completed!\n")


if __name__ == '__main__':
    main()
