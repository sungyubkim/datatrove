"""
README Generator for Math VERL Datasets.

This script generates standardized README files for individual and unified math datasets
using Jinja2 templates, source dataset mapping, and preprocessing examples.

Usage:
    python generate_readme.py --dataset mathx-5m-verl --output /path/to/README.md
    python generate_readme.py --dataset math-verl-unified --output /path/to/README.md
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, Template

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_examples import PREPROCESSING_EXAMPLES, get_examples
from source_dataset_mapping import (
    DATASET_IDS,
    LICENSE_NOTES,
    SOURCE_DATASET_MAPPING,
    get_source_info,
)


class ReadmeGenerator:
    """Generate README files from templates and dataset information."""

    def __init__(self, template_path: Optional[str] = None):
        """Initialize the README generator.

        Args:
            template_path: Path to Jinja2 template file. If None, uses default template.
        """
        if template_path is None:
            # Use default template from repository
            template_path = Path(__file__).parent.parent.parent / "templates" / "math_dataset_readme_template.md"

        self.template_path = Path(template_path)

        # Set up Jinja2 environment
        template_dir = self.template_path.parent
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))

        # Add custom filters
        self.env.filters["format_number"] = self._format_number
        self.env.filters["replace"] = str.replace

        # Load template
        self.template = self.env.get_template(self.template_path.name)

    @staticmethod
    def _format_number(value: int) -> str:
        """Format number with comma separators.

        Args:
            value: Number to format

        Returns:
            Formatted string (e.g., 1000 -> "1,000")
        """
        return f"{value:,}"

    def generate_readme(
        self,
        dataset_id: str,
        dataset_stats: Dict[str, Any],
        cleaning_stats: Optional[Dict[str, Any]] = None,
        dedup_stats: Optional[Dict[str, Any]] = None,
        version_history: Optional[str] = None,
    ) -> str:
        """Generate README content for a dataset.

        Args:
            dataset_id: Dataset identifier (e.g., 'mathx-5m-verl')
            dataset_stats: Dataset statistics (samples, size, etc.)
            cleaning_stats: Optional cleaning statistics
            dedup_stats: Optional deduplication statistics
            version_history: Optional version history markdown

        Returns:
            Generated README markdown content
        """
        # Get source information
        source_info = get_source_info(dataset_id)

        # Prepare template variables
        template_vars = self._prepare_template_variables(
            dataset_id=dataset_id,
            source_info=source_info,
            dataset_stats=dataset_stats,
            cleaning_stats=cleaning_stats,
            dedup_stats=dedup_stats,
            version_history=version_history,
        )

        # Render template
        readme_content = self.template.render(**template_vars)

        return readme_content

    def _prepare_template_variables(
        self,
        dataset_id: str,
        source_info: Dict[str, Any],
        dataset_stats: Dict[str, Any],
        cleaning_stats: Optional[Dict[str, Any]],
        dedup_stats: Optional[Dict[str, Any]],
        version_history: Optional[str],
    ) -> Dict[str, Any]:
        """Prepare all template variables for rendering.

        Args:
            dataset_id: Dataset identifier
            source_info: Source dataset information
            dataset_stats: Dataset statistics
            cleaning_stats: Cleaning statistics
            dedup_stats: Deduplication statistics
            version_history: Version history

        Returns:
            Dictionary of template variables
        """
        # Basic dataset info
        vars = {
            "dataset_name": source_info["dataset_name"],
            "dataset_id": dataset_id,
            "license": dataset_stats.get("license", source_info["source_license"]),
            "size_category": self._get_size_category(dataset_stats["num_examples"]),
            "num_bytes": dataset_stats.get("num_bytes", 0),
            "num_examples": dataset_stats["num_examples"],
            "download_size": dataset_stats.get("download_size", dataset_stats.get("num_bytes", 0)),
            "dataset_size": dataset_stats.get("dataset_size", dataset_stats.get("num_bytes", 0)),
            "dataset_size_human": self._format_bytes(dataset_stats.get("dataset_size", dataset_stats.get("num_bytes", 0))),
            "hub_dataset_path": f"sungyub/{dataset_id}",
        }

        # Dataset summary and key features
        vars["dataset_summary"] = dataset_stats.get(
            "summary",
            f"This dataset contains {self._format_number(dataset_stats['num_examples'])} "
            f"mathematical reasoning problems in VERL format, processed from {source_info['source_repo_name']}.",
        )
        vars["key_features"] = dataset_stats.get(
            "key_features",
            f"- **{self._format_number(dataset_stats['num_examples'])} high-quality math problems**\n"
            f"- Converted to VERL format for reward modeling\n"
            f"- Verified ground truth answers\n"
            f"- Ready for reinforcement learning training",
        )

        # Source information
        vars.update(
            {
                "source_repo_name": source_info["source_repo_name"],
                "source_repo_url": source_info["source_repo_url"],
                "source_license": source_info["source_license"],
                "source_paper_title": source_info["source_paper_title"],
                "source_paper_url": source_info["source_paper_url"],
                "source_authors": source_info["source_authors"],
                "source_description": source_info["source_description"],
                "source_citation": source_info["source_citation"],
            }
        )

        # Preprocessing info
        cleaning_preset = source_info["cleaning_preset"]
        if cleaning_preset and cleaning_stats:
            vars["cleaning_preset"] = cleaning_preset
            vars["cleaning_patterns"] = self._format_cleaning_patterns(cleaning_preset)
            vars["original_samples"] = cleaning_stats.get("original_samples", 0)
            vars["cleaned_samples"] = cleaning_stats.get("cleaned_samples", dataset_stats["num_examples"])
            vars["removed_samples"] = cleaning_stats.get("removed_samples", 0)
            vars["removal_rate"] = cleaning_stats.get("removal_rate", 0)
            vars["artifacts_removed"] = cleaning_stats.get("artifacts_removed", 0)
        else:
            vars["cleaning_preset"] = None

        # Deduplication stats
        if dedup_stats:
            vars["dedup_stats"] = True
            vars["before_dedup"] = dedup_stats.get("before_dedup", 0)
            vars["after_dedup"] = dedup_stats.get("after_dedup", dataset_stats["num_examples"])
            vars["dedup_reduction"] = dedup_stats.get("dedup_reduction", 0)
            vars["inter_dedup"] = dedup_stats.get("inter_dedup", False)
            vars["dedup_priority"] = dedup_stats.get("dedup_priority", "N/A")
            vars["inter_dedup_removed"] = dedup_stats.get("inter_dedup_removed", 0)
        else:
            vars["dedup_stats"] = False

        # Preprocessing examples
        preset_for_examples = cleaning_preset if cleaning_preset else "standard"
        examples = get_examples(preset_for_examples)

        if len(examples) >= 1:
            vars["example1_title"] = examples[0]["title"]
            vars["example1_before"] = examples[0]["before"]
            vars["example1_after"] = examples[0]["after"]
            vars["example1_changes"] = "\n".join([f"- {change}" for change in examples[0]["changes"]])

        if len(examples) >= 2:
            vars["example2_before"] = examples[1]["before"]
            vars["example2_after"] = examples[1]["after"]
            vars["example2_title"] = examples[1]["title"]
            vars["example2_changes"] = "\n".join([f"- {change}" for change in examples[1]["changes"]])
        else:
            vars["example2_before"] = None

        # VERL schema examples
        vars.update(
            {
                "data_source_example": dataset_stats.get("data_source_example", "openai/gsm8k"),
                "prompt_example": dataset_stats.get(
                    "prompt_example", "Calculate the sum of all odd numbers from 1 to 99."
                ),
                "ground_truth_example": dataset_stats.get("ground_truth_example", "\\boxed{2500}"),
                "hash_example": "sha256:abc123...",
            }
        )

        # Dataset statistics section
        vars["dataset_statistics"] = dataset_stats.get(
            "statistics_section",
            f"""### Sample Distribution

- **Total Samples**: {self._format_number(dataset_stats['num_examples'])}
- **Dataset Size**: {self._format_bytes(dataset_stats.get('dataset_size', 0))}
- **Average Problem Length**: {dataset_stats.get('avg_length', 'N/A')} characters

### Data Sources

Distribution of problems by original data source:

| Source | Count | Percentage |
|--------|-------|------------|
| Mixed Sources | {self._format_number(dataset_stats['num_examples'])} | 100% |

*Note: Detailed source distribution statistics will be added in future updates.*
""",
        )

        # License notes
        license_type = vars["license"]
        vars["license_notes"] = LICENSE_NOTES.get(license_type, "")

        # Acknowledgments
        vars["special_thanks"] = source_info["special_thanks"]

        # Version history
        vars["version_history"] = version_history or (
            f"""### v1.0.0 (Initial Release)
- Processed {self._format_number(dataset_stats['num_examples'])} samples from {source_info['source_repo_name']}
- Converted to VERL format
- {'Applied ' + cleaning_preset + ' cleaning preset' if cleaning_preset else 'Standard processing applied'}
- Ready for reinforcement learning training
"""
        )

        return vars

    @staticmethod
    def _get_size_category(num_examples: int) -> str:
        """Get HuggingFace size category based on number of examples.

        Args:
            num_examples: Number of samples

        Returns:
            Size category string
        """
        if num_examples < 1000:
            return "n<1K"
        elif num_examples < 10000:
            return "1K<n<10K"
        elif num_examples < 100000:
            return "10K<n<100K"
        elif num_examples < 1000000:
            return "100K<n<1M"
        else:
            return "1M<n<10M"

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Format bytes to human-readable string.

        Args:
            num_bytes: Number of bytes

        Returns:
            Formatted string (e.g., "1.5 GB")
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if num_bytes < 1024.0:
                return f"{num_bytes:.1f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.1f} PB"

    @staticmethod
    def _format_cleaning_patterns(preset: str) -> str:
        """Format cleaning patterns for a preset as markdown list.

        Args:
            preset: Cleaning preset name

        Returns:
            Markdown formatted list of cleaning patterns
        """
        patterns_map = {
            "orz-math": [
                "**Problem Numbering**: Prefixes like `Problem 6.`, `8.3`, `147 Let`, `Task B-3.4.`",
                "**Contest Metadata**: References like `(2004 AIME Problem 3)`, `24th Eötvös 1917`",
                "**Point Allocations**: Indicators like `(8 points)`, `[10 marks]`",
                "**Markdown Headers**: Headers like `## Problem Statement`, `## Task`",
                "**Author Attributions**: LaTeX underlined author names",
                "**Trailing Artifacts**: End-of-problem markers like `## Level 3`, `[ Geometry ]`",
                "**Special Artifacts**: Horizontal rules (`---`), translation instructions",
            ],
            "openr1-math": [
                "**Problem Numbering**: All formats (numeric, roman, letter-number combinations)",
                "**Point Allocations**: Score indicators in various formats",
                "**Contest Metadata**: Competition names, years, problem identifiers",
                "**Markdown Headers**: All markdown header styles (`##`, `###`, etc.)",
                "**Special Artifacts**: Translation artifacts, horizontal rules, topic labels",
            ],
            "skywork-or1": [
                "**Problem Numbering**: Simple numeric prefixes like `Question 230,`",
                "**Basic Formatting**: Minimal cleaning to preserve dataset characteristics",
            ],
            "dapo-math": [
                "**Minimal Cleaning**: DAPO-Math is already well-formatted",
                "**URL Filtering**: Only removes samples containing URLs",
            ],
        }

        patterns = patterns_map.get(preset, [])
        return "\n".join([f"{i+1}. {pattern}" for i, pattern in enumerate(patterns)])


def main():
    """Command-line interface for README generation."""
    parser = argparse.ArgumentParser(description="Generate README for math VERL datasets")
    parser.add_argument("--dataset", required=True, help="Dataset ID (e.g., mathx-5m-verl)")
    parser.add_argument("--output", required=True, help="Output README file path")
    parser.add_argument("--template", help="Custom template path (optional)")
    parser.add_argument("--num-examples", type=int, required=True, help="Number of examples in dataset")
    parser.add_argument("--dataset-size", type=int, help="Dataset size in bytes")
    parser.add_argument("--license", help="Override license (default: from source mapping)")

    args = parser.parse_args()

    # Validate dataset ID
    if args.dataset not in DATASET_IDS and args.dataset != "math-verl-unified":
        print(f"Error: Unknown dataset ID '{args.dataset}'")
        print(f"Available datasets: {DATASET_IDS + ['math-verl-unified']}")
        sys.exit(1)

    # Prepare dataset stats
    dataset_stats = {
        "num_examples": args.num_examples,
        "dataset_size": args.dataset_size or (args.num_examples * 1000),  # Rough estimate
        "license": args.license,
    }

    # Generate README
    generator = ReadmeGenerator(template_path=args.template)
    readme_content = generator.generate_readme(dataset_id=args.dataset, dataset_stats=dataset_stats)

    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(readme_content, encoding="utf-8")

    print(f"✓ README generated successfully: {output_path}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Samples: {args.num_examples:,}")
    print(f"  Size: {generator._format_bytes(dataset_stats['dataset_size'])}")


if __name__ == "__main__":
    main()
