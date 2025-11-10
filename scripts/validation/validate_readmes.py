"""
README validation script for math VERL datasets.

Validates README files for:
- YAML frontmatter syntax
- Required sections presence
- Markdown formatting
- Link validity (optional)
- Content completeness

Usage:
    python validate_readmes.py --readme-dir /path/to/readmes
    python validate_readmes.py --readme-file /path/to/README.md
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


class ReadmeValidator:
    """Validate README files for math VERL datasets."""

    REQUIRED_SECTIONS = [
        "Dataset Summary",
        "Source Dataset",
        "Preprocessing Pipeline",
        "Preprocessing Examples",
        "VERL Schema",
        "Dataset Statistics",
        "Usage",
        "Citation",
        "License",
    ]

    YAML_REQUIRED_FIELDS = [
        "license",
        "language",
        "tags",
        "size_categories",
        "task_categories",
        "dataset_info",
    ]

    def __init__(self, check_links: bool = False):
        """Initialize validator.

        Args:
            check_links: Whether to validate external links (requires network)
        """
        self.check_links = check_links
        self.errors = []
        self.warnings = []

    def validate_file(self, readme_path: Path) -> Tuple[bool, List[str], List[str]]:
        """Validate a single README file.

        Args:
            readme_path: Path to README.md file

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        if not readme_path.exists():
            self.errors.append(f"File not found: {readme_path}")
            return False, self.errors, self.warnings

        content = readme_path.read_text(encoding="utf-8")

        # Run all validation checks
        self._validate_yaml_frontmatter(content)
        self._validate_required_sections(content)
        self._validate_markdown_syntax(content)
        self._validate_content_completeness(content)

        if self.check_links:
            self._validate_links(content)

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings

    def _validate_yaml_frontmatter(self, content: str):
        """Validate YAML frontmatter syntax and required fields.

        Args:
            content: README file content
        """
        # Extract YAML frontmatter
        yaml_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)

        if not yaml_match:
            self.errors.append("Missing YAML frontmatter (should start with ---)")
            return

        yaml_content = yaml_match.group(1)

        try:
            yaml_data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML syntax: {e}")
            return

        # Check required fields
        for field in self.YAML_REQUIRED_FIELDS:
            if field not in yaml_data:
                self.errors.append(f"Missing required YAML field: {field}")

        # Validate specific fields
        if "license" in yaml_data:
            license_value = yaml_data["license"]
            valid_licenses = ["mit", "apache-2.0", "cc-by-4.0", "unknown"]
            if isinstance(license_value, str) and license_value.lower() not in valid_licenses:
                self.warnings.append(f"Unusual license value: {license_value}")

        if "language" in yaml_data:
            if "en" not in yaml_data["language"]:
                self.warnings.append("Language should include 'en' for English")

        if "tags" in yaml_data:
            recommended_tags = ["math", "reasoning", "verl"]
            has_recommended = any(tag in yaml_data["tags"] for tag in recommended_tags)
            if not has_recommended:
                self.warnings.append(f"Consider adding recommended tags: {recommended_tags}")

    def _validate_required_sections(self, content: str):
        """Validate presence of required README sections.

        Args:
            content: README file content
        """
        # Remove YAML frontmatter for section checking
        content_without_yaml = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, count=1, flags=re.DOTALL)

        for section in self.REQUIRED_SECTIONS:
            # Look for section headers (## or ###)
            # Simply check if the section name appears in any header line
            # This allows for emoji and other decorations
            found = False
            for line in content_without_yaml.split('\n'):
                if line.startswith('#') and section.lower() in line.lower():
                    found = True
                    break
            if not found:
                self.errors.append(f"Missing required section: {section}")

    def _validate_markdown_syntax(self, content: str):
        """Validate markdown syntax and formatting.

        Args:
            content: README file content
        """
        lines = content.split("\n")

        # Check for common markdown issues
        for i, line in enumerate(lines, 1):
            # Check for unmatched code fences
            if line.strip().startswith("```"):
                # Count code fences - should be even
                code_fence_count = content.count("```")
                if code_fence_count % 2 != 0:
                    self.warnings.append(f"Unmatched code fence near line {i}")
                break  # Only check once

            # Check for malformed links
            if "[" in line and "]" in line:
                # Simple check for [text](url) format
                link_pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
                matches = re.findall(link_pattern, line)
                for text, url in matches:
                    if not url.strip():
                        self.warnings.append(f"Empty URL in link on line {i}: [{text}]()")
                    if url.startswith(" ") or url.endswith(" "):
                        self.warnings.append(f"URL has extra whitespace on line {i}")

            # Check for malformed headers
            if line.startswith("#"):
                if not re.match(r"^#{1,6}\s+.+", line):
                    self.warnings.append(f"Malformed header on line {i}: missing space after #")

    def _validate_content_completeness(self, content: str):
        """Validate content completeness and quality.

        Args:
            content: README file content
        """
        # Check for placeholder text
        placeholders = [
            "TODO",
            "TBD",
            "FIXME",
            "XXX",
            "coming soon",
            "to be added",
            "placeholder",
        ]

        for placeholder in placeholders:
            if placeholder.lower() in content.lower():
                self.warnings.append(f"Found placeholder text: '{placeholder}'")

        # Check minimum length
        if len(content) < 1000:
            self.warnings.append(f"README seems too short ({len(content)} characters)")

        # Check for example code blocks
        if "```python" not in content:
            self.warnings.append("Missing Python code examples")

        # Check for citation information
        if "@" not in content:  # BibTeX entries contain @
            self.warnings.append("Missing BibTeX citation")

    def _validate_links(self, content: str):
        """Validate external links (requires network access).

        Args:
            content: README file content
        """
        # Extract all URLs
        url_pattern = r"https?://[^\s\)\]]+"
        urls = re.findall(url_pattern, content)

        if not urls:
            self.warnings.append("No external links found")
            return

        # Note: Actual link validation would require requests library
        # and network access. For now, just check URL format.
        for url in urls:
            # Basic URL validation
            if " " in url:
                self.warnings.append(f"URL contains spaces: {url}")

            # Check for common issues
            if url.endswith("."):
                self.warnings.append(f"URL ends with period (may be incorrect): {url}")

            # Check for broken markdown URLs
            if "[" in url or "]" in url:
                self.warnings.append(f"URL may be part of malformed markdown: {url}")

    def validate_directory(self, readme_dir: Path) -> Dict[str, Tuple[bool, List[str], List[str]]]:
        """Validate all README files in a directory.

        Args:
            readme_dir: Directory containing dataset subdirectories with READMEs

        Returns:
            Dictionary mapping dataset name to validation results
        """
        results = {}

        # Find all README.md files
        readme_files = list(readme_dir.glob("*/README.md"))

        if not readme_files:
            print(f"Warning: No README.md files found in {readme_dir}")
            return results

        for readme_path in readme_files:
            dataset_name = readme_path.parent.name
            is_valid, errors, warnings = self.validate_file(readme_path)
            results[dataset_name] = (is_valid, errors, warnings)

        return results


def print_validation_results(
    results: Dict[str, Tuple[bool, List[str], List[str]]], verbose: bool = False
):
    """Print validation results in a formatted way.

    Args:
        results: Validation results dictionary
        verbose: Whether to print all warnings
    """
    print("\nVALIDATION RESULTS")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for is_valid, _, _ in results.values() if is_valid)
    failed = total - passed

    for dataset_name, (is_valid, errors, warnings) in results.items():
        status = "✓ PASS" if is_valid else "✗ FAIL"
        status_color = "\033[92m" if is_valid else "\033[91m"  # Green or Red
        reset_color = "\033[0m"

        print(f"\n{status_color}{status}{reset_color} {dataset_name}")

        if errors:
            print(f"  Errors ({len(errors)}):")
            for error in errors:
                print(f"    ✗ {error}")

        if warnings and (verbose or not is_valid):
            print(f"  Warnings ({len(warnings)}):")
            for warning in warnings[:5]:  # Limit to first 5 warnings
                print(f"    ⚠ {warning}")
            if len(warnings) > 5:
                print(f"    ... and {len(warnings) - 5} more warnings")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{total} passed, {failed}/{total} failed")

    if failed == 0:
        print("\n✓ All READMEs are valid!")
    else:
        print(f"\n✗ {failed} README(s) need attention")

    return failed == 0


def main():
    """Command-line interface for README validation."""
    parser = argparse.ArgumentParser(description="Validate README files for math VERL datasets")
    parser.add_argument(
        "--readme-dir",
        type=Path,
        help="Directory containing dataset subdirectories with READMEs",
    )
    parser.add_argument(
        "--readme-file",
        type=Path,
        help="Single README file to validate",
    )
    parser.add_argument(
        "--check-links",
        action="store_true",
        help="Check external links (requires network)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all warnings",
    )

    args = parser.parse_args()

    if not args.readme_dir and not args.readme_file:
        parser.error("Either --readme-dir or --readme-file must be specified")

    validator = ReadmeValidator(check_links=args.check_links)

    if args.readme_file:
        # Validate single file
        is_valid, errors, warnings = validator.validate_file(args.readme_file)
        results = {args.readme_file.stem: (is_valid, errors, warnings)}
        all_valid = print_validation_results(results, verbose=args.verbose)
    else:
        # Validate directory
        results = validator.validate_directory(args.readme_dir)
        if not results:
            print(f"No README files found in {args.readme_dir}")
            sys.exit(1)
        all_valid = print_validation_results(results, verbose=args.verbose)

    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
