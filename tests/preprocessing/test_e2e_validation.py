"""End-to-end validation tests for RLVR-to-IFBench transformation.

This module validates the complete transformation pipeline from RLVR-IFeval
to IFBench-VERL format, including batch processing and scorer integration.
"""

import json
import random
from pathlib import Path

import pytest


class TestEndToEndValidation:
    """End-to-end validation tests for the complete pipeline."""

    @pytest.fixture
    def processed_dataset_path(self):
        """Path to the processed dataset."""
        return Path("output/ifbench-rlvr-verl/train.jsonl")

    @pytest.fixture
    def sample_examples(self, processed_dataset_path):
        """Load random sample of processed examples."""
        if not processed_dataset_path.exists():
            pytest.skip("Processed dataset not found. Run process_rlvr_to_ifbench.py first.")

        examples = []
        with open(processed_dataset_path, "r") as f:
            all_lines = f.readlines()
            # Sample 50 random examples
            sample_lines = random.sample(all_lines, min(50, len(all_lines)))
            for line in sample_lines:
                examples.append(json.loads(line))

        return examples

    def test_processed_dataset_exists(self, processed_dataset_path):
        """Test that the processed dataset file exists."""
        assert processed_dataset_path.exists(), "Processed dataset not found"

    def test_all_examples_have_required_fields(self, sample_examples):
        """Test that all examples have required IFBench-VERL fields."""
        required_fields = ["data_source", "prompt", "ability", "reward_model", "extra_info", "dataset"]

        for idx, example in enumerate(sample_examples):
            for field in required_fields:
                assert field in example, f"Example {idx} missing field: {field}"

    def test_all_examples_validate_schema(self, sample_examples):
        """Test that all examples pass schema validation."""
        from datatrove.preprocessing.rlvr_to_ifbench import validate_ifbench_schema

        for idx, example in enumerate(sample_examples):
            try:
                assert validate_ifbench_schema(example)
            except Exception as e:
                pytest.fail(f"Example {idx} failed validation: {e}")

    def test_all_examples_scoreable(self, sample_examples):
        """Test that all examples can be scored."""
        from datatrove.utils.reward_score import compute_score

        for idx, example in enumerate(sample_examples):
            try:
                # Create a dummy response
                response = "<think>Test response</think>\ntest answer"

                # Should not raise an error
                score = compute_score(
                    data_source="sungyub/ifeval-rlvr-verl",
                    solution_str=response,
                    ground_truth=example["reward_model"]["ground_truth"],
                    format_type="auto",
                )

                assert isinstance(score, dict), f"Example {idx}: Score should be a dict"
                assert "score" in score, f"Example {idx}: Score dict should have 'score' key"

            except Exception as e:
                pytest.fail(f"Example {idx} failed scoring: {e}")

    def test_constraint_type_coverage(self, sample_examples):
        """Test that sample covers various constraint types."""
        import ast

        constraint_types = set()
        for example in sample_examples:
            gt = ast.literal_eval(example["reward_model"]["ground_truth"])
            for constraint in gt:
                for instruction_id in constraint["instruction_id"]:
                    constraint_types.add(instruction_id)

        # Should have at least 10 different constraint types in sample
        assert len(constraint_types) >= 10, f"Only {len(constraint_types)} constraint types found in sample"

    def test_score_correct_vs_incorrect_responses(self, sample_examples):
        """Test that correct responses score higher than incorrect ones."""
        from datatrove.utils.reward_score import compute_score
        import ast

        # Find a lowercase constraint example
        lowercase_example = None
        for example in sample_examples:
            gt = ast.literal_eval(example["reward_model"]["ground_truth"])
            if any("english_lowercase" in inst_id for constraint in gt for inst_id in constraint["instruction_id"]):
                lowercase_example = example
                break

        if lowercase_example is None:
            pytest.skip("No lowercase constraint example found in sample")

        # Correct response (all lowercase)
        correct = "<think>creating lowercase response</think>\nthis is all lowercase text"
        correct_score = compute_score(
            data_source="sungyub/ifeval-rlvr-verl",
            solution_str=correct,
            ground_truth=lowercase_example["reward_model"]["ground_truth"],
            format_type="auto",
        )

        # Incorrect response (has uppercase)
        incorrect = "<think>Creating response</think>\nThis Has Uppercase Letters"
        incorrect_score = compute_score(
            data_source="sungyub/ifeval-rlvr-verl",
            solution_str=incorrect,
            ground_truth=lowercase_example["reward_model"]["ground_truth"],
            format_type="auto",
        )

        assert correct_score["score"] > incorrect_score["score"], (
            f"Correct score ({correct_score['score']}) should be higher than "
            f"incorrect score ({incorrect_score['score']})"
        )

    def test_stats_file_accuracy(self):
        """Test that stats file matches actual dataset."""
        stats_path = Path("output/ifbench-rlvr-verl/train_stats.json")
        dataset_path = Path("output/ifbench-rlvr-verl/train.jsonl")

        if not stats_path.exists() or not dataset_path.exists():
            pytest.skip("Required files not found")

        with open(stats_path, "r") as f:
            stats = json.load(f)

        with open(dataset_path, "r") as f:
            actual_lines = sum(1 for _ in f)

        assert stats["transformed_examples"] == actual_lines, (
            f"Stats says {stats['transformed_examples']} examples, "
            f"but file has {actual_lines} lines"
        )

    def test_prompt_content_preserved(self, sample_examples):
        """Test that prompt content is properly preserved."""
        for idx, example in enumerate(sample_examples):
            assert "prompt" in example
            assert isinstance(example["prompt"], list)
            assert len(example["prompt"]) > 0

            first_msg = example["prompt"][0]
            assert "role" in first_msg
            assert "content" in first_msg
            assert first_msg["role"] == "user"
            assert len(first_msg["content"]) > 0

    def test_ground_truth_format(self, sample_examples):
        """Test that ground truth uses Python literal format."""
        import ast

        for idx, example in enumerate(sample_examples):
            gt_str = example["reward_model"]["ground_truth"]

            # Should be a string
            assert isinstance(gt_str, str)

            # Should be parseable as Python literal
            try:
                gt = ast.literal_eval(gt_str)
            except Exception as e:
                pytest.fail(f"Example {idx}: Failed to parse ground truth as Python literal: {e}")

            # Should be a list
            assert isinstance(gt, list)
            assert len(gt) > 0

            # Each element should have instruction_id and kwargs
            for constraint in gt:
                assert "instruction_id" in constraint
                assert "kwargs" in constraint
                assert isinstance(constraint["instruction_id"], list)
                assert isinstance(constraint["kwargs"], list)


class TestBatchProcessingPerformance:
    """Performance and quality tests for batch processing."""

    def test_no_failed_transformations(self):
        """Test that there were no failed transformations."""
        stats_path = Path("output/ifbench-rlvr-verl/train_stats.json")

        if not stats_path.exists():
            pytest.skip("Stats file not found")

        with open(stats_path, "r") as f:
            stats = json.load(f)

        assert stats["failed_examples"] == 0, f"{stats['failed_examples']} examples failed transformation"

    def test_all_examples_transformed(self):
        """Test that all examples were successfully transformed."""
        stats_path = Path("output/ifbench-rlvr-verl/train_stats.json")

        if not stats_path.exists():
            pytest.skip("Stats file not found")

        with open(stats_path, "r") as f:
            stats = json.load(f)

        assert stats["transformed_examples"] == stats["total_examples"], (
            f"Only {stats['transformed_examples']}/{stats['total_examples']} examples transformed"
        )

    def test_dataset_size_matches_rlvr(self):
        """Test that processed dataset size matches RLVR dataset."""
        stats_path = Path("output/ifbench-rlvr-verl/train_stats.json")

        if not stats_path.exists():
            pytest.skip("Stats file not found")

        with open(stats_path, "r") as f:
            stats = json.load(f)

        # RLVR-IFeval train split has 14,973 examples
        expected_size = 14973
        assert stats["total_examples"] == expected_size, (
            f"Expected {expected_size} examples, got {stats['total_examples']}"
        )
