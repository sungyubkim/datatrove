"""Tests for Parquet format conversion and validation.

This module validates the Parquet conversion from JSONL format,
ensuring data integrity and HuggingFace compatibility.
"""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest


class TestParquetConversion:
    """Test Parquet file conversion and format."""

    @pytest.fixture
    def parquet_path(self):
        """Path to the Parquet file."""
        return Path("output/ifbench-rlvr-verl/train.parquet")

    @pytest.fixture
    def jsonl_path(self):
        """Path to the JSONL file."""
        return Path("output/ifbench-rlvr-verl/train.jsonl")

    def test_parquet_file_exists(self, parquet_path):
        """Test that Parquet file was created."""
        assert parquet_path.exists(), "Parquet file not found"

    def test_parquet_readable(self, parquet_path):
        """Test that Parquet file can be read."""
        if not parquet_path.exists():
            pytest.skip("Parquet file not found")

        table = pq.read_table(parquet_path)
        assert table.num_rows > 0, "Parquet file is empty"

    def test_row_count_matches_jsonl(self, parquet_path, jsonl_path):
        """Test that Parquet has same number of rows as JSONL."""
        if not parquet_path.exists() or not jsonl_path.exists():
            pytest.skip("Required files not found")

        # Count JSONL rows
        with open(jsonl_path, "r") as f:
            jsonl_rows = sum(1 for _ in f)

        # Count Parquet rows
        table = pq.read_table(parquet_path)
        parquet_rows = table.num_rows

        assert parquet_rows == jsonl_rows, f"Row count mismatch: {parquet_rows} vs {jsonl_rows}"

    def test_parquet_schema(self, parquet_path):
        """Test that Parquet has correct schema."""
        if not parquet_path.exists():
            pytest.skip("Parquet file not found")

        table = pq.read_table(parquet_path)
        schema = table.schema

        # Check field names
        expected_fields = ["data_source", "prompt", "ability", "reward_model", "extra_info", "dataset"]
        actual_fields = schema.names

        assert set(actual_fields) == set(expected_fields), f"Schema fields mismatch: {actual_fields}"

    def test_parquet_data_types(self, parquet_path):
        """Test that Parquet has correct data types."""
        if not parquet_path.exists():
            pytest.skip("Parquet file not found")

        import pyarrow as pa

        table = pq.read_table(parquet_path)
        schema = table.schema

        # Check data types
        assert schema.field("data_source").type == pa.string()
        assert schema.field("ability").type == pa.string()
        assert schema.field("dataset").type == pa.string()

        # Check nested types
        assert schema.field("prompt").type == pa.list_(
            pa.struct([("role", pa.string()), ("content", pa.string())])
        )
        assert schema.field("reward_model").type == pa.struct([("style", pa.string()), ("ground_truth", pa.string())])
        assert schema.field("extra_info").type == pa.struct([("index", pa.int64())])

    def test_data_integrity_sample(self, parquet_path, jsonl_path):
        """Test that data content matches between JSONL and Parquet."""
        if not parquet_path.exists() or not jsonl_path.exists():
            pytest.skip("Required files not found")

        # Load first 10 examples from JSONL
        jsonl_examples = []
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                jsonl_examples.append(json.loads(line))

        # Load from Parquet
        table = pq.read_table(parquet_path)
        parquet_df = table.to_pandas()

        # Compare first 10 rows
        for i in range(min(10, len(jsonl_examples))):
            jsonl_ex = jsonl_examples[i]
            parquet_row = parquet_df.iloc[i]

            # Check simple fields
            assert parquet_row["data_source"] == jsonl_ex["data_source"]
            assert parquet_row["ability"] == jsonl_ex["ability"]
            assert parquet_row["dataset"] == jsonl_ex["dataset"]

            # Check nested fields
            assert parquet_row["reward_model"]["style"] == jsonl_ex["reward_model"]["style"]
            assert parquet_row["reward_model"]["ground_truth"] == jsonl_ex["reward_model"]["ground_truth"]
            assert parquet_row["extra_info"]["index"] == jsonl_ex["extra_info"]["index"]

    def test_parquet_compression(self, parquet_path):
        """Test that Parquet uses compression."""
        if not parquet_path.exists():
            pytest.skip("Parquet file not found")

        parquet_file = pq.ParquetFile(parquet_path)
        metadata = parquet_file.metadata

        # Check that compression is used
        assert metadata.num_row_groups > 0
        row_group = metadata.row_group(0)
        column = row_group.column(0)

        # Should use some compression (snappy, gzip, etc.)
        assert column.compression != "UNCOMPRESSED"

    def test_parquet_file_size_reasonable(self, parquet_path, jsonl_path):
        """Test that Parquet file size is smaller than JSONL."""
        if not parquet_path.exists() or not jsonl_path.exists():
            pytest.skip("Required files not found")

        parquet_size = parquet_path.stat().st_size
        jsonl_size = jsonl_path.stat().st_size

        # Parquet should be smaller due to compression
        assert parquet_size < jsonl_size, f"Parquet ({parquet_size}) should be smaller than JSONL ({jsonl_size})"

        # Should achieve at least 1.5x compression
        compression_ratio = jsonl_size / parquet_size
        assert compression_ratio >= 1.5, f"Compression ratio {compression_ratio:.2f}x is too low"


class TestHuggingFaceCompatibility:
    """Test compatibility with HuggingFace datasets library."""

    @pytest.fixture
    def parquet_path(self):
        """Path to the Parquet file."""
        return Path("output/ifbench-rlvr-verl/train.parquet")

    def test_load_with_datasets_library(self, parquet_path):
        """Test loading Parquet with datasets library."""
        if not parquet_path.exists():
            pytest.skip("Parquet file not found")

        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")

        assert len(dataset) > 0, "Dataset is empty"
        assert "data_source" in dataset.column_names
        assert "prompt" in dataset.column_names
        assert "reward_model" in dataset.column_names

    def test_dataset_features(self, parquet_path):
        """Test that dataset has correct feature types."""
        if not parquet_path.exists():
            pytest.skip("Parquet file not found")

        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
        features = dataset.features

        # Check that all expected features are present
        expected_features = ["data_source", "prompt", "ability", "reward_model", "extra_info", "dataset"]
        for feature_name in expected_features:
            assert feature_name in features, f"Missing feature: {feature_name}"

        # Check that prompt is a list/sequence
        assert "prompt" in features
        # Check that nested structures exist
        assert "reward_model" in features
        assert "extra_info" in features

    def test_dataset_iteration(self, parquet_path):
        """Test that dataset can be iterated."""
        if not parquet_path.exists():
            pytest.skip("Parquet file not found")

        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")

        # Iterate first 10 examples
        count = 0
        for example in dataset:
            count += 1
            assert "data_source" in example
            assert "prompt" in example
            assert "reward_model" in example
            if count >= 10:
                break

        assert count == 10, "Failed to iterate examples"

    def test_dataset_indexing(self, parquet_path):
        """Test that dataset supports indexing."""
        if not parquet_path.exists():
            pytest.skip("Parquet file not found")

        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")

        # Test single index
        example = dataset[0]
        assert isinstance(example, dict)
        assert "data_source" in example

        # Test slice
        batch = dataset[:10]
        assert isinstance(batch, dict)
        assert len(batch["data_source"]) == 10

    def test_scorer_compatibility(self, parquet_path):
        """Test that dataset examples work with scorer."""
        if not parquet_path.exists():
            pytest.skip("Parquet file not found")

        from datasets import load_dataset
        from datatrove.utils.reward_score import compute_score

        dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")

        # Get first example
        example = dataset[0]

        # Create test response
        response = "<think>test reasoning</think>\ntest answer"

        # Should not raise error
        score = compute_score(
            data_source="sungyub/ifeval-rlvr-verl",
            solution_str=response,
            ground_truth=example["reward_model"]["ground_truth"],
            format_type="auto",
        )

        assert isinstance(score, dict)
        assert "score" in score
