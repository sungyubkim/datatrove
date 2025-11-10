"""
Generate README for the unified math-verl collection.

Usage:
    python generate_unified_readme.py --output output/readmes-unified/math-verl-unified/README.md
"""

import argparse
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from source_dataset_mapping import DATASET_IDS, SOURCE_DATASET_MAPPING
from update_all_readmes import DATASET_STATISTICS


class UnifiedReadmeGenerator:
    """Generate README for the unified math-verl collection."""

    def __init__(self):
        """Initialize generator with unified template."""
        template_path = Path(__file__).parent.parent.parent / "templates" / "math_unified_readme_template.md"
        template_dir = template_path.parent

        self.env = Environment(loader=FileSystemLoader(str(template_dir)))
        self.template = self.env.get_template(template_path.name)

    def generate(self) -> str:
        """Generate unified README content.

        Returns:
            Generated README markdown content
        """
        # Calculate totals
        total_samples = sum(stats["num_examples"] for stats in DATASET_STATISTICS.values())
        total_size = sum(stats["dataset_size"] for stats in DATASET_STATISTICS.values())
        total_size_human = self._format_bytes(total_size)

        # After inter-dataset dedup (v3.0)
        final_samples = 2269166  # From research phase

        # Generate tables and sections
        dataset_table = self._generate_dataset_table()
        source_links = self._generate_source_links()
        cleaning_summary = self._generate_cleaning_summary()
        split_statistics = self._generate_split_statistics()
        license_table = self._generate_license_table()
        source_citations = self._generate_source_citations()
        acknowledgments = self._generate_acknowledgments()
        related_links = self._generate_related_links()

        # Calculate average length (estimate)
        avg_length = int(total_size / final_samples * 0.7)  # Rough estimate

        # Render template
        readme_content = self.template.render(
            total_size_human=total_size_human,
            dataset_table=dataset_table,
            source_links=source_links,
            cleaning_summary=cleaning_summary,
            split_statistics=split_statistics,
            license_table=license_table,
            source_citations=source_citations,
            acknowledgments=acknowledgments,
            related_links=related_links,
            avg_length=avg_length,
        )

        return readme_content

    def _generate_dataset_table(self) -> str:
        """Generate table of source datasets."""
        lines = ["| Dataset | Samples | Size | License | Source |"]
        lines.append("|---------|---------|------|---------|--------|")

        for dataset_id in DATASET_IDS:
            stats = DATASET_STATISTICS[dataset_id]
            source_info = SOURCE_DATASET_MAPPING[dataset_id]

            size_human = self._format_bytes(stats["dataset_size"])
            source_name = source_info["source_repo_name"]
            license_val = stats.get("license", source_info["source_license"])

            lines.append(
                f"| `{dataset_id}` | {stats['num_examples']:,} | {size_human} | "
                f"{license_val} | [{source_name}]({source_info['source_repo_url']}) |"
            )

        return "\n".join(lines)

    def _generate_source_links(self) -> str:
        """Generate list of source dataset links."""
        lines = []

        for i, dataset_id in enumerate(DATASET_IDS, 1):
            source_info = SOURCE_DATASET_MAPPING[dataset_id]
            lines.append(
                f"{i}. **[{source_info['source_repo_name']}]({source_info['source_repo_url']})** "
                f"({source_info['source_license']})"
            )
            if source_info["source_paper_url"]:
                lines.append(f"   - Paper: [{source_info['source_paper_title']}]({source_info['source_paper_url']})")

        return "\n".join(lines)

    def _generate_cleaning_summary(self) -> str:
        """Generate cleaning methodology summary."""
        cleaning_presets = {
            "orz-math": ["orz-math-72k-verl"],
            "openr1-math": ["openr1-math-verl"],
            "skywork-or1": ["skywork-or1-math-verl"],
            "dapo-math": ["dapo-math-17k-verl"],
        }

        lines = []
        for preset, datasets in cleaning_presets.items():
            dataset_names = ", ".join([f"`{d}`" for d in datasets])
            lines.append(f"- **{preset} preset**: {dataset_names}")

            if preset == "orz-math":
                lines.append("  - 7 artifact removal patterns (problem numbers, contest metadata, etc.)")
            elif preset == "openr1-math":
                lines.append("  - 7 artifact removal patterns (markdown headers, translations, etc.)")
            elif preset == "skywork-or1":
                lines.append("  - 2 artifact removal patterns (minimal cleaning)")
            elif preset == "dapo-math":
                lines.append("  - 1 artifact removal pattern (already well-formatted)")

        # Add datasets without specific presets
        preset_datasets = set([d for datasets in cleaning_presets.values() for d in datasets])
        other_datasets = [d for d in DATASET_IDS if d not in preset_datasets]

        if other_datasets:
            other_names = ", ".join([f"`{d}`" for d in other_datasets])
            lines.append(f"- **Standard processing**: {other_names}")
            lines.append("  - URL filtering, format normalization, basic validation")

        return "\n".join(lines)

    def _generate_split_statistics(self) -> str:
        """Generate split distribution statistics."""
        total_samples = 2269166  # After inter-dataset dedup v3.0

        lines = ["| Split | Samples | Percentage |"]
        lines.append("|-------|---------|------------|")
        lines.append(f"| train | {total_samples:,} | 100% |")
        lines.append("")
        lines.append("*Note: All samples are in the training split. Original test/validation splits are preserved in `extra_info.split` field.*")

        return "\n".join(lines)

    def _generate_license_table(self) -> str:
        """Generate license distribution table."""
        licenses = {}

        for dataset_id in DATASET_IDS:
            stats = DATASET_STATISTICS[dataset_id]
            source_info = SOURCE_DATASET_MAPPING[dataset_id]
            license_val = stats.get("license", source_info["source_license"])

            if license_val not in licenses:
                licenses[license_val] = []
            licenses[license_val].append(dataset_id)

        lines = ["| License | Datasets | Count |"]
        lines.append("|---------|----------|-------|")

        for license_val, datasets in sorted(licenses.items()):
            dataset_list = ", ".join([f"`{d}`" for d in datasets])
            lines.append(f"| {license_val} | {dataset_list} | {len(datasets)} |")

        return "\n".join(lines)

    def _generate_source_citations(self) -> str:
        """Generate BibTeX citations for all sources."""
        lines = []

        for dataset_id in DATASET_IDS:
            source_info = SOURCE_DATASET_MAPPING[dataset_id]
            lines.append(f"**{source_info['source_repo_name']}**:")
            lines.append("```bibtex")
            lines.append(source_info["source_citation"])
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    def _generate_acknowledgments(self) -> str:
        """Generate acknowledgments section."""
        lines = []

        for dataset_id in DATASET_IDS:
            source_info = SOURCE_DATASET_MAPPING[dataset_id]
            lines.append(f"- **{source_info['source_repo_name']}**: {source_info['source_authors']}")

        return "\n".join(lines)

    def _generate_related_links(self) -> str:
        """Generate links to individual datasets."""
        lines = []

        for dataset_id in DATASET_IDS:
            source_info = SOURCE_DATASET_MAPPING[dataset_id]
            lines.append(
                f"- [{source_info['dataset_name']}](https://huggingface.co/datasets/sungyub/{dataset_id}) "
                f"- {DATASET_STATISTICS[dataset_id]['num_examples']:,} samples"
            )

        return "\n".join(lines)

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Format bytes to human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if num_bytes < 1024.0:
                return f"{num_bytes:.1f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.1f} PB"


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Generate unified collection README")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output README file path",
    )

    args = parser.parse_args()

    # Generate README
    print("Generating unified collection README...")
    generator = UnifiedReadmeGenerator()
    readme_content = generator.generate()

    # Write to file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(readme_content, encoding="utf-8")

    print(f"✓ Unified README generated: {args.output}")
    print(f"  Total datasets: 9")
    print(f"  Total samples: 2,269,166 (after dedup)")
    print(f"  Version: v3.0")


if __name__ == "__main__":
    main()
