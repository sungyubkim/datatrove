"""Tests for HuggingFace Hub compatibility.

This module validates README format, YAML metadata, and
overall HuggingFace Hub compatibility.
"""

import re
from pathlib import Path

import pytest
import yaml


class TestREADMEFormat:
    """Test README.md format and content."""

    @pytest.fixture
    def readme_path(self):
        """Path to README file."""
        return Path("output/ifbench-rlvr-verl/README.md")

    @pytest.fixture
    def readme_content(self, readme_path):
        """Load README content."""
        if not readme_path.exists():
            pytest.skip("README file not found")

        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_readme_exists(self, readme_path):
        """Test that README file exists."""
        assert readme_path.exists(), "README.md not found"

    def test_has_yaml_frontmatter(self, readme_content):
        """Test that README has YAML frontmatter."""
        assert readme_content.startswith("---\n"), "README should start with YAML frontmatter"

        # Check for closing ---
        parts = readme_content.split("---\n", 2)
        assert len(parts) >= 3, "README should have properly closed YAML frontmatter"

    def test_yaml_frontmatter_valid(self, readme_content):
        """Test that YAML frontmatter is valid YAML."""
        # Extract YAML frontmatter
        parts = readme_content.split("---\n", 2)
        yaml_content = parts[1]

        # Should parse without error
        metadata = yaml.safe_load(yaml_content)
        assert isinstance(metadata, dict), "YAML frontmatter should be a dictionary"

    def test_yaml_has_required_fields(self, readme_content):
        """Test that YAML has all required fields."""
        parts = readme_content.split("---\n", 2)
        metadata = yaml.safe_load(parts[1])

        required_fields = ["license", "task_categories", "language", "tags", "dataset_info"]
        for field in required_fields:
            assert field in metadata, f"YAML missing required field: {field}"

    def test_yaml_license(self, readme_content):
        """Test that license is correct."""
        parts = readme_content.split("---\n", 2)
        metadata = yaml.safe_load(parts[1])

        assert metadata["license"] == "odc-by", "License should be odc-by"

    def test_yaml_dataset_info(self, readme_content):
        """Test that dataset_info has correct structure."""
        parts = readme_content.split("---\n", 2)
        metadata = yaml.safe_load(parts[1])

        dataset_info = metadata["dataset_info"]
        assert "features" in dataset_info
        assert "splits" in dataset_info

        # Check features
        features = dataset_info["features"]
        assert len(features) == 6, "Should have 6 features"

        feature_names = [f["name"] for f in features]
        expected_names = ["data_source", "prompt", "ability", "reward_model", "extra_info", "dataset"]
        assert set(feature_names) == set(expected_names), f"Feature names mismatch: {feature_names}"

        # Check splits
        splits = dataset_info["splits"]
        assert len(splits) > 0, "Should have at least one split"
        assert "name" in splits[0]
        assert "num_examples" in splits[0]

    def test_yaml_num_examples(self, readme_content):
        """Test that num_examples matches actual data."""
        parts = readme_content.split("---\n", 2)
        metadata = yaml.safe_load(parts[1])

        splits = metadata["dataset_info"]["splits"]
        train_split = [s for s in splits if s["name"] == "train"][0]

        assert train_split["num_examples"] == 14973, "num_examples should be 14973"

    def test_has_overview_section(self, readme_content):
        """Test that README has overview section."""
        assert "## Overview" in readme_content, "README should have Overview section"

    def test_has_dataset_statistics(self, readme_content):
        """Test that README has dataset statistics."""
        assert "## Dataset Statistics" in readme_content or "14,973" in readme_content

    def test_has_schema_documentation(self, readme_content):
        """Test that README has schema documentation."""
        assert "## Dataset Structure" in readme_content or "### Fields" in readme_content

    def test_has_constraint_types_documentation(self, readme_content):
        """Test that README documents constraint types."""
        assert "## Constraint Types" in readme_content or "constraint" in readme_content.lower()

    def test_has_usage_examples(self, readme_content):
        """Test that README has usage examples."""
        assert "## Usage" in readme_content or "```python" in readme_content

    def test_has_citation(self, readme_content):
        """Test that README has citation information."""
        assert "## Citation" in readme_content or "```bibtex" in readme_content

    def test_mentions_all_constraint_types(self, readme_content):
        """Test that README mentions all 25 constraint types."""
        # Should mention key constraint categories
        categories = [
            "keywords",
            "language",
            "length",
            "content",
            "format",
            "case",
            "punctuation",
        ]

        content_lower = readme_content.lower()
        for category in categories:
            assert category in content_lower, f"README should mention category: {category}"

    def test_code_examples_valid_syntax(self, readme_content):
        """Test that Python code examples have valid syntax."""
        # Extract Python code blocks
        code_blocks = re.findall(r"```python\n(.*?)\n```", readme_content, re.DOTALL)

        assert len(code_blocks) > 0, "README should have Python code examples"

        # Each code block should be valid Python syntax
        for i, code in enumerate(code_blocks):
            try:
                compile(code, f"<code_block_{i}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Code block {i} has syntax error: {e}")

    def test_links_valid_format(self, readme_content):
        """Test that links have valid markdown format."""
        # Find markdown links
        links = re.findall(r"\[([^\]]+)\]\(([^\)]+)\)", readme_content)

        assert len(links) > 0, "README should have links"

        for text, url in links:
            assert url.startswith("http") or url.startswith("#"), f"Invalid link format: {url}"


class TestDatasetCardCompatibility:
    """Test compatibility with HuggingFace dataset card requirements."""

    @pytest.fixture
    def readme_path(self):
        """Path to README file."""
        return Path("output/ifbench-rlvr-verl/README.md")

    def test_has_pretty_name(self, readme_path):
        """Test that dataset has a pretty name."""
        if not readme_path.exists():
            pytest.skip("README file not found")

        with open(readme_path, "r") as f:
            content = f.read()

        parts = content.split("---\n", 2)
        metadata = yaml.safe_load(parts[1])

        assert "pretty_name" in metadata, "Should have pretty_name field"

    def test_has_size_category(self, readme_path):
        """Test that dataset has size category."""
        if not readme_path.exists():
            pytest.skip("README file not found")

        with open(readme_path, "r") as f:
            content = f.read()

        parts = content.split("---\n", 2)
        metadata = yaml.safe_load(parts[1])

        assert "size_categories" in metadata, "Should have size_categories field"
        assert "10K<n<100K" in metadata["size_categories"], "Size category should be 10K<n<100K"

    def test_has_task_categories(self, readme_path):
        """Test that dataset has task categories."""
        if not readme_path.exists():
            pytest.skip("README file not found")

        with open(readme_path, "r") as f:
            content = f.read()

        parts = content.split("---\n", 2)
        metadata = yaml.safe_load(parts[1])

        assert "task_categories" in metadata, "Should have task_categories field"
        assert len(metadata["task_categories"]) > 0, "Should have at least one task category"

    def test_has_language(self, readme_path):
        """Test that dataset specifies language."""
        if not readme_path.exists():
            pytest.skip("README file not found")

        with open(readme_path, "r") as f:
            content = f.read()

        parts = content.split("---\n", 2)
        metadata = yaml.safe_load(parts[1])

        assert "language" in metadata, "Should have language field"
        assert "en" in metadata["language"], "Should include English"

    def test_has_tags(self, readme_path):
        """Test that dataset has relevant tags."""
        if not readme_path.exists():
            pytest.skip("README file not found")

        with open(readme_path, "r") as f:
            content = f.read()

        parts = content.split("---\n", 2)
        metadata = yaml.safe_load(parts[1])

        assert "tags" in metadata, "Should have tags field"
        tags = metadata["tags"]

        # Should have relevant tags
        expected_tags = ["evaluation", "ifeval", "instruction-following"]
        for tag in expected_tags:
            assert tag in tags, f"Should have tag: {tag}"


class TestSchemaConsistency:
    """Test consistency between README schema and actual data."""

    @pytest.fixture
    def readme_path(self):
        """Path to README file."""
        return Path("output/ifbench-rlvr-verl/README.md")

    @pytest.fixture
    def parquet_path(self):
        """Path to Parquet file."""
        return Path("output/ifbench-rlvr-verl/train.parquet")

    def test_schema_matches_actual_data(self, readme_path, parquet_path):
        """Test that README schema matches actual Parquet schema."""
        if not readme_path.exists() or not parquet_path.exists():
            pytest.skip("Required files not found")

        # Load README schema
        with open(readme_path, "r") as f:
            content = f.read()
        parts = content.split("---\n", 2)
        metadata = yaml.safe_load(parts[1])
        readme_features = [f["name"] for f in metadata["dataset_info"]["features"]]

        # Load actual schema
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        actual_features = table.schema.names

        assert set(readme_features) == set(actual_features), f"Schema mismatch: {readme_features} vs {actual_features}"

    def test_num_examples_matches(self, readme_path, parquet_path):
        """Test that README num_examples matches actual data."""
        if not readme_path.exists() or not parquet_path.exists():
            pytest.skip("Required files not found")

        # Load README num_examples
        with open(readme_path, "r") as f:
            content = f.read()
        parts = content.split("---\n", 2)
        metadata = yaml.safe_load(parts[1])
        readme_num_examples = metadata["dataset_info"]["splits"][0]["num_examples"]

        # Load actual count
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        actual_num_examples = table.num_rows

        assert readme_num_examples == actual_num_examples, f"Example count mismatch: {readme_num_examples} vs {actual_num_examples}"
