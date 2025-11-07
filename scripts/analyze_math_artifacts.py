#!/usr/bin/env python3
"""
Analyze math dataset queries to identify text artifacts and patterns for cleaning.

This script samples from a math dataset and performs comprehensive pattern analysis
to discover artifacts that should be removed during data cleaning.
"""

import re
from collections import Counter, defaultdict
from datasets import load_dataset
import argparse
from typing import Dict, List, Tuple
import json


class MathArtifactAnalyzer:
    """Analyzes math dataset queries to identify cleaning patterns."""

    def __init__(self, max_samples: int = 5000):
        self.max_samples = max_samples
        self.queries = []
        self.stats = defaultdict(list)

    def load_samples(self, dataset_name: str):
        """Load samples from HuggingFace dataset."""
        print(f"Loading up to {self.max_samples} samples from {dataset_name}...")

        # Load with streaming to avoid memory issues
        dataset = load_dataset(dataset_name, split="train", streaming=True)

        for i, example in enumerate(dataset):
            if i >= self.max_samples:
                break

            # Extract query from VERL format
            prompt = example.get("prompt", [])
            if prompt and len(prompt) > 0:
                user_message = prompt[0]
                content = user_message.get("content", "")
                if content:
                    self.queries.append(content)

            if (i + 1) % 1000 == 0:
                print(f"  Loaded {i + 1} samples...")

        print(f"✓ Loaded {len(self.queries)} queries\n")

    def analyze_prefixes(self, max_length: int = 100) -> Dict[str, int]:
        """Analyze common prefixes in queries."""
        print("Analyzing prefixes...")
        prefixes = Counter()

        for query in self.queries:
            # Extract first line or first max_length characters
            first_line = query.split('\n')[0][:max_length]

            # Look for patterns like "Problem X.", "Question X:", etc.
            # Extract up to first punctuation or 50 chars
            match = re.match(r'^([^.!?\n]{1,50}[.!?:])\s', first_line)
            if match:
                prefix = match.group(1)
                prefixes[prefix] += 1

        # Filter to prefixes that appear at least 10 times
        common_prefixes = {k: v for k, v in prefixes.items() if v >= 10}

        print(f"  Found {len(common_prefixes)} common prefixes (appearing ≥10 times)")
        return dict(common_prefixes)

    def analyze_pattern_types(self) -> Dict[str, List[Tuple[str, int]]]:
        """Categorize queries by artifact types."""
        print("Analyzing artifact patterns...")

        patterns = {
            "numbered_problems": [],
            "contest_metadata": [],
            "point_allocations": [],
            "markdown_headers": [],
            "html_entities": [],
            "urls_links": [],
            "difficulty_ratings": [],
            "source_attribution": [],
            "bracket_patterns": [],
            "special_markers": [],
        }

        # Define regex patterns to search for
        pattern_regexes = {
            "numbered_problems": [
                r'^\s*(?:Problem|Question|Exercise|Task|Example)\s*[#\d]+[.:)]',
                r'^\s*\d+\.\s+',
                r'^\s*[A-Z]\d+\.',
                r'^\s*\(\d+\)',
            ],
            "contest_metadata": [
                r'\d{4}\s+(?:IMO|AMC|AIME|USAMO|Putnam|Olympiad)',
                r'(?:International|National|Regional)\s+\w+\s+(?:Olympiad|Competition)',
                r'\d+(?:st|nd|rd|th)\s+\w+\s+\d{4}',
            ],
            "point_allocations": [
                r'\(\s*\d+\s*points?\s*\)',
                r'\[\s*\d+\s*marks?\s*\]',
                r'\d+\s*pts',
            ],
            "markdown_headers": [
                r'^#{1,6}\s+\w+',
                r'^\*\*[^*]+\*\*\s*$',
            ],
            "html_entities": [
                r'&[a-z]+;',
                r'&#\d+;',
                r'&nbsp;',
            ],
            "urls_links": [
                r'https?://\S+',
                r'\[.*?\]\(.*?\)',
                r'www\.\S+',
            ],
            "difficulty_ratings": [
                r'(?:Difficulty|Level|Rating):\s*\S+',
                r'[★☆]{2,}',
                r'\*{1,5}\s*(?:difficulty|level)',
            ],
            "source_attribution": [
                r'(?:Source|From|Reference):\s*\S+',
                r'(?:Adapted|Taken)\s+from',
                r'\(courtesy of\s+\S+\)',
            ],
            "bracket_patterns": [
                r'\[asy\].*?\[/asy\]',
                r'\[img\].*?\[/img\]',
                r'\[figure\s+\d+\]',
            ],
            "special_markers": [
                r'^\s*[-=_]{3,}\s*$',
                r'^\s*\*{3,}\s*$',
                r'BEGIN\s+(?:PROBLEM|QUESTION)',
                r'END\s+(?:PROBLEM|QUESTION)',
            ],
        }

        # Search for each pattern type
        for category, regexes in pattern_regexes.items():
            matches = Counter()
            for query in self.queries:
                for regex in regexes:
                    found = re.findall(regex, query, re.MULTILINE | re.IGNORECASE)
                    for match in found:
                        # Truncate long matches
                        match_str = match[:100] if len(match) > 100 else match
                        matches[match_str] += 1

            # Keep top matches that appear at least 5 times
            top_matches = [(k, v) for k, v in matches.most_common(50) if v >= 5]
            patterns[category] = top_matches

            if top_matches:
                print(f"  {category}: Found {len(top_matches)} patterns")

        return patterns

    def analyze_special_characters(self) -> Dict[str, int]:
        """Analyze special character usage."""
        print("Analyzing special characters...")

        special_chars = Counter()

        for query in self.queries:
            # Count occurrences of various special characters
            for char in query:
                if ord(char) > 127 or char in '★☆●○■□◆◇':  # Non-ASCII or special symbols
                    special_chars[char] += 1

        # Filter to characters appearing at least 100 times
        common_special = {k: v for k, v in special_chars.items() if v >= 100}

        print(f"  Found {len(common_special)} common special characters")
        return dict(common_special)

    def find_repetitive_substrings(self, min_length: int = 20) -> Dict[str, int]:
        """Find substrings that appear frequently across queries."""
        print("Finding repetitive substrings...")

        substrings = Counter()

        for query in self.queries:
            # Extract first 100 and last 100 characters
            if len(query) > min_length:
                # First part
                first = query[:100].strip()
                if len(first) >= min_length:
                    substrings[first] += 1

                # Last part
                last = query[-100:].strip()
                if len(last) >= min_length:
                    substrings[last] += 1

        # Filter to substrings appearing at least 10 times
        common_substrings = {k: v for k, v in substrings.items() if v >= 10}

        print(f"  Found {len(common_substrings)} repetitive substrings")
        return dict(common_substrings)

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append("MATH DATASET ARTIFACT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"\nDataset: {len(self.queries)} queries analyzed\n")

        # Analyze all aspects
        prefixes = self.analyze_prefixes()
        patterns = self.analyze_pattern_types()
        special_chars = self.analyze_special_characters()
        repetitive = self.find_repetitive_substrings()

        # Report prefixes
        report.append("\n" + "=" * 80)
        report.append("COMMON PREFIXES (appearing ≥10 times)")
        report.append("=" * 80)
        if prefixes:
            for prefix, count in sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:30]:
                report.append(f"  [{count:4d}x] {prefix}")
        else:
            report.append("  No common prefixes found")

        # Report patterns by category
        report.append("\n" + "=" * 80)
        report.append("ARTIFACT PATTERNS BY CATEGORY")
        report.append("=" * 80)

        for category, matches in patterns.items():
            if matches:
                report.append(f"\n{category.upper().replace('_', ' ')}:")
                for pattern, count in matches[:10]:  # Show top 10
                    # Clean up pattern for display
                    display_pattern = pattern.replace('\n', '\\n')[:80]
                    report.append(f"  [{count:4d}x] {display_pattern}")

        # Report special characters
        report.append("\n" + "=" * 80)
        report.append("SPECIAL CHARACTERS (appearing ≥100 times)")
        report.append("=" * 80)
        if special_chars:
            for char, count in sorted(special_chars.items(), key=lambda x: x[1], reverse=True)[:30]:
                char_display = char if char.isprintable() else f"U+{ord(char):04X}"
                report.append(f"  [{count:5d}x] {char_display} (ord={ord(char)})")
        else:
            report.append("  No unusual special characters found")

        # Report repetitive substrings
        report.append("\n" + "=" * 80)
        report.append("REPETITIVE SUBSTRINGS (appearing ≥10 times)")
        report.append("=" * 80)
        if repetitive:
            for substring, count in sorted(repetitive.items(), key=lambda x: x[1], reverse=True)[:20]:
                display_sub = substring.replace('\n', '\\n')[:80]
                report.append(f"  [{count:4d}x] {display_sub}")
        else:
            report.append("  No repetitive substrings found")

        # Summary statistics
        report.append("\n" + "=" * 80)
        report.append("SUMMARY STATISTICS")
        report.append("=" * 80)

        total_with_artifacts = 0
        for category, matches in patterns.items():
            if matches:
                # Count how many queries have this artifact type
                count = sum(m[1] for m in matches)
                report.append(f"  {category.replace('_', ' ').title()}: ~{count} occurrences")
                total_with_artifacts += min(count, len(self.queries))

        # Calculate percentage (rough estimate)
        artifact_percentage = (total_with_artifacts / len(self.queries) * 100) if self.queries else 0
        report.append(f"\nEstimated queries with artifacts: ~{artifact_percentage:.1f}%")

        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATION: Review patterns above to identify new cleaning rules")
        report.append("=" * 80)

        return "\n".join(report)

    def save_detailed_analysis(self, output_path: str):
        """Save detailed analysis as JSON for programmatic access."""
        print(f"\nSaving detailed analysis to {output_path}...")

        analysis = {
            "dataset_info": {
                "total_queries": len(self.queries),
                "sample_queries": self.queries[:10],  # First 10 as examples
            },
            "prefixes": self.analyze_prefixes(),
            "patterns": self.analyze_pattern_types(),
            "special_characters": self.analyze_special_characters(),
            "repetitive_substrings": self.find_repetitive_substrings(),
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved detailed analysis")


def main():
    parser = argparse.ArgumentParser(description="Analyze math dataset for text artifacts")
    parser.add_argument(
        "--dataset",
        type=str,
        default="sungyub/openr1-math-verl",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum number of samples to analyze"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="./artifact_analysis_report.txt",
        help="Output path for text report"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="./artifact_analysis_data.json",
        help="Output path for JSON analysis data"
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = MathArtifactAnalyzer(max_samples=args.max_samples)
    analyzer.load_samples(args.dataset)

    # Generate and save report
    report = analyzer.generate_report()
    print("\n" + report)

    with open(args.output_report, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✓ Report saved to {args.output_report}")

    # Save detailed JSON analysis
    analyzer.save_detailed_analysis(args.output_json)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
