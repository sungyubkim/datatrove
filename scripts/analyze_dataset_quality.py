#!/usr/bin/env python3
"""
Quality Analysis Script for Cleaned Math Dataset
Analyzes the cleaned dataset for remaining artifacts and data integrity
"""

import pandas as pd
import re
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import random


class DatasetQualityAnalyzer:
    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        self.df = pd.read_parquet(parquet_path)
        self.total_samples = len(self.df)

        # Define artifact patterns to check
        self.artifact_patterns = {
            'markdown_headers': [
                r'^#{1,6}\s+(Problem|Task|Solution|Question|Exercise|Answer)',
                r'^#{1,6}\s+\d+\.',
            ],
            'problem_numbering': [
                r'^(Problem|Question|Exercise)\s+\d+[.:]\s*',
                r'^\d+\.\s*\d+[.:]\s*',  # e.g., "8.3:", "G1.4:"
                r'^[A-Z]\d+\.\d+[.:]\s*',  # e.g., "G1.4:"
                r'^Question\s+\d+,\s*',  # e.g., "Question 230,"
            ],
            'contest_metadata': [
                r'\b\d{4}\s+(AIME|AMC|USAMO|IMO|APMC)\b',
                r'\b(AIME|AMC|USAMO|IMO|APMC)\s+\d{4}\b',
                r'\bProblem\s+\d+\b.*\b(AIME|AMC|USAMO|IMO|APMC)\b',
            ],
            'point_allocations': [
                r'\(\d+\s+points?\)',
                r'\[\d+\s+points?\]',
            ],
        }

        self.issues = defaultdict(list)
        self.clean_samples = []
        self.problematic_samples = []

    def extract_text(self, row) -> Tuple[str, dict]:
        """Extract text and metadata from a row"""
        metadata = row['metadata']

        # Extract text from prompt field (which is a conversation array, stored as numpy array)
        prompt = metadata.get('prompt', [])
        text = ''

        if hasattr(prompt, '__len__') and len(prompt) > 0:
            # Handle both list and numpy array
            first_msg = prompt[0]
            if isinstance(first_msg, dict):
                text = first_msg.get('content', '')

        return text, metadata

    def check_artifacts(self, text: str, sample_id: str) -> Dict[str, List[str]]:
        """Check for remaining artifacts in text"""
        found_artifacts = defaultdict(list)

        # Split into lines for line-by-line analysis
        lines = text.split('\n')

        for artifact_type, patterns in self.artifact_patterns.items():
            for pattern in patterns:
                # Check first few lines more carefully for headers/numbering
                if artifact_type in ['markdown_headers', 'problem_numbering']:
                    for i, line in enumerate(lines[:5]):  # Check first 5 lines
                        if re.search(pattern, line.strip(), re.IGNORECASE):
                            found_artifacts[artifact_type].append({
                                'pattern': pattern,
                                'match': line.strip(),
                                'line_num': i
                            })
                else:
                    # Check entire text for other artifacts
                    matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        found_artifacts[artifact_type].append({
                            'pattern': pattern,
                            'match': match.group(0),
                            'context': text[max(0, match.start()-50):match.end()+50]
                        })

        return found_artifacts

    def verify_data_integrity(self, text: str, metadata: dict) -> Dict[str, bool]:
        """Verify data integrity checks"""
        checks = {
            'has_ground_truth': False,
            'has_prompt': False,
            'text_not_empty': False,
            'ground_truth_not_empty': False,
        }

        # Check for prompt/text
        prompt = metadata.get('prompt', [])
        checks['has_prompt'] = len(prompt) > 0
        checks['text_not_empty'] = len(text.strip()) > 0

        # Check for ground truth
        reward_model = metadata.get('reward_model', {})
        ground_truth = reward_model.get('ground_truth', '')
        checks['has_ground_truth'] = 'ground_truth' in reward_model
        checks['ground_truth_not_empty'] = len(str(ground_truth).strip()) > 0

        return checks

    def sample_dataset(self, n_samples: int = 2000) -> List[int]:
        """Sample indices from beginning, middle, and end of dataset"""
        indices = []

        # Beginning (first 20%)
        beginning_size = n_samples // 3
        beginning_end = min(beginning_size * 5, self.total_samples // 5)
        indices.extend(random.sample(range(0, beginning_end), beginning_size))

        # Middle (40%-60%)
        middle_size = n_samples // 3
        middle_start = self.total_samples * 2 // 5
        middle_end = self.total_samples * 3 // 5
        indices.extend(random.sample(range(middle_start, middle_end), middle_size))

        # End (last 20%)
        end_size = n_samples - beginning_size - middle_size
        end_start = max(self.total_samples * 4 // 5, self.total_samples - end_size * 5)
        indices.extend(random.sample(range(end_start, self.total_samples), end_size))

        return sorted(indices)

    def analyze(self, n_samples: int = 2000):
        """Run comprehensive analysis"""
        print(f"Analyzing {n_samples} samples from {self.total_samples} total samples...")
        print(f"Sampling from beginning, middle, and end of dataset...\n")

        sample_indices = self.sample_dataset(n_samples)

        issue_counts = defaultdict(int)
        integrity_failures = defaultdict(int)

        for idx in sample_indices:
            row = self.df.iloc[idx]
            sample_id = row['id']
            text, metadata = self.extract_text(row)

            # Check for artifacts
            artifacts = self.check_artifacts(text, sample_id)

            # Check data integrity
            integrity = self.verify_data_integrity(text, metadata)

            # Record results
            has_issues = len(artifacts) > 0
            has_integrity_issues = not all(integrity.values())

            if has_issues or has_integrity_issues:
                self.problematic_samples.append({
                    'id': sample_id,
                    'index': idx,
                    'text': text,
                    'metadata': metadata,
                    'artifacts': dict(artifacts),
                    'integrity': integrity
                })

                for artifact_type in artifacts:
                    issue_counts[artifact_type] += 1

                for check, passed in integrity.items():
                    if not passed:
                        integrity_failures[check] += 1
            else:
                self.clean_samples.append({
                    'id': sample_id,
                    'index': idx,
                    'text': text,
                    'metadata': metadata
                })

        return {
            'n_samples': n_samples,
            'n_clean': len(self.clean_samples),
            'n_problematic': len(self.problematic_samples),
            'issue_counts': dict(issue_counts),
            'integrity_failures': dict(integrity_failures)
        }

    def generate_report(self, results: dict):
        """Generate comprehensive quality report"""
        print("=" * 80)
        print("DATASET QUALITY ANALYSIS REPORT")
        print("=" * 80)
        print()

        # Summary Statistics
        print("SUMMARY STATISTICS")
        print("-" * 80)
        print(f"Total samples in dataset:  {self.total_samples:,}")
        print(f"Samples analyzed:          {results['n_samples']:,}")
        print(f"Clean samples:             {results['n_clean']:,}")
        print(f"Problematic samples:       {results['n_problematic']:,}")

        success_rate = (results['n_clean'] / results['n_samples']) * 100
        print(f"Success rate:              {success_rate:.2f}%")
        print()

        # Issue Breakdown
        if results['issue_counts']:
            print("ARTIFACT ISSUES BREAKDOWN")
            print("-" * 80)
            for issue_type, count in sorted(results['issue_counts'].items(), key=lambda x: -x[1]):
                percentage = (count / results['n_samples']) * 100
                print(f"  {issue_type:30} {count:5} ({percentage:5.2f}%)")
            print()

        # Integrity Issues
        if results['integrity_failures']:
            print("DATA INTEGRITY ISSUES")
            print("-" * 80)
            for check, count in sorted(results['integrity_failures'].items(), key=lambda x: -x[1]):
                percentage = (count / results['n_samples']) * 100
                print(f"  {check:30} {count:5} ({percentage:5.2f}%)")
            print()

        # Perfect Cleaning Examples
        print("EXAMPLES OF PERFECT CLEANING (5 samples)")
        print("-" * 80)
        for i, sample in enumerate(self.clean_samples[:5], 1):
            text_preview = sample['text'][:300].replace('\n', ' ')
            print(f"\n{i}. ID: {sample['id']} (index: {sample['index']})")
            print(f"   Preview: {text_preview}...")
            gt = sample['metadata'].get('reward_model', {}).get('ground_truth', '')
            if gt:
                gt_preview = str(gt)[:100].replace('\n', ' ')
                print(f"   Ground Truth: {gt_preview}...")
        print()

        # Problematic Examples
        if self.problematic_samples:
            print("EXAMPLES OF REMAINING ISSUES (up to 10 samples)")
            print("-" * 80)
            for i, sample in enumerate(self.problematic_samples[:10], 1):
                print(f"\n{i}. ID: {sample['id']} (index: {sample['index']})")

                if sample['artifacts']:
                    print(f"   Artifacts found:")
                    for artifact_type, matches in sample['artifacts'].items():
                        print(f"     - {artifact_type}:")
                        for match in matches[:2]:  # Show first 2 matches
                            if 'match' in match:
                                print(f"       * \"{match['match']}\"")

                if sample['integrity']:
                    failed_checks = [k for k, v in sample['integrity'].items() if not v]
                    if failed_checks:
                        print(f"   Failed integrity checks: {', '.join(failed_checks)}")

                text_preview = sample['text'][:200].replace('\n', ' ')
                print(f"   Text preview: {text_preview}...")
        print()

        # Overall Assessment
        print("OVERALL QUALITY ASSESSMENT")
        print("-" * 80)

        if success_rate >= 99.5:
            quality = "EXCELLENT"
            assessment = "The dataset is extremely clean with minimal artifacts."
        elif success_rate >= 97.0:
            quality = "GOOD"
            assessment = "The dataset is mostly clean with minor artifacts remaining."
        elif success_rate >= 90.0:
            quality = "FAIR"
            assessment = "The dataset has noticeable artifacts that should be addressed."
        else:
            quality = "POOR"
            assessment = "The dataset has significant artifacts requiring additional cleaning."

        print(f"Quality Rating: {quality}")
        print(f"Assessment: {assessment}")
        print()

        # Recommendations
        if results['n_problematic'] > 0:
            print("RECOMMENDATIONS FOR IMPROVEMENT")
            print("-" * 80)

            for issue_type, count in sorted(results['issue_counts'].items(), key=lambda x: -x[1]):
                if count > results['n_samples'] * 0.01:  # More than 1%
                    print(f"  - Address '{issue_type}' artifacts (found in {count} samples)")

            if results['integrity_failures']:
                print(f"  - Fix data integrity issues")

            print()

        print("=" * 80)
        print("END OF REPORT")
        print("=" * 80)


def main():
    parquet_path = './output/orz-math-cleaned/000_00000.parquet'

    analyzer = DatasetQualityAnalyzer(parquet_path)
    results = analyzer.analyze(n_samples=2000)
    analyzer.generate_report(results)


if __name__ == "__main__":
    main()
