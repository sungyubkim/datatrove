"""Tests for RLVR to IFBench-VERL schema transformation."""

import ast
import json

import pytest


class TestSchemaTransformer:
    """Test transforming RLVR examples to IFBench-VERL format."""

    def test_transform_basic_example(self):
        """Test basic schema transformation"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench

        rlvr_example = {
            "messages": [{"content": "Write in lowercase", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "validate_lowercase", "N": None}),
            "dataset": "ifeval",
        }

        result = transform_to_ifbench(rlvr_example, index=0)

        # Check all required fields
        assert result["data_source"] == "allenai/IF_multi_constraints_upto5"
        assert result["prompt"] == [{"content": "Write in lowercase", "role": "user"}]
        assert result["ability"] == "instruction_following"
        assert result["dataset"] == "ifeval"
        assert result["reward_model"]["style"] == "ifeval"
        assert result["extra_info"]["index"] == 0

    def test_transform_preserves_prompt_content(self):
        """Test prompt content is preserved exactly"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench

        content = "Test with unicode: 한글 中文 🎉\nAnd newlines\tand\ttabs"
        rlvr_example = {
            "messages": [{"content": content, "role": "user"}],
            "ground_truth": json.dumps({"func_name": "validate_lowercase"}),
            "dataset": "ifeval",
        }

        result = transform_to_ifbench(rlvr_example, index=0)

        assert result["prompt"][0]["content"] == content

    def test_transform_ground_truth_is_json_string(self):
        """Test ground_truth is a JSON string"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench

        rlvr_example = {
            "messages": [{"content": "Test", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "validate_lowercase"}),
            "dataset": "ifeval",
        }

        result = transform_to_ifbench(rlvr_example, index=0)

        # Must be a string
        assert isinstance(result["reward_model"]["ground_truth"], str)

        # Must be valid Python literal
        gt = ast.literal_eval(result["reward_model"]["ground_truth"])
        assert isinstance(gt, list)

    def test_transform_ground_truth_structure(self):
        """Test ground_truth has correct IFEval structure"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench

        rlvr_example = {
            "messages": [{"content": "Test", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "verify_paragraph_count", "N": 5}),
            "dataset": "ifeval",
        }

        result = transform_to_ifbench(rlvr_example, index=0)

        gt = ast.literal_eval(result["reward_model"]["ground_truth"])

        # Must be list with single constraint
        assert isinstance(gt, list)
        assert len(gt) == 1

        # Must have instruction_id and kwargs
        assert "instruction_id" in gt[0]
        assert "kwargs" in gt[0]

        # instruction_id must be list
        assert isinstance(gt[0]["instruction_id"], list)
        assert len(gt[0]["instruction_id"]) == 1

        # kwargs must be list
        assert isinstance(gt[0]["kwargs"], list)
        assert len(gt[0]["kwargs"]) == 1

    def test_transform_with_index(self):
        """Test index is correctly set"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench

        rlvr_example = {
            "messages": [{"content": "Test", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "validate_lowercase"}),
            "dataset": "ifeval",
        }

        result = transform_to_ifbench(rlvr_example, index=42)

        assert result["extra_info"]["index"] == 42

    def test_transform_empty_messages_raises_error(self):
        """Test error handling for empty messages"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench

        rlvr_example = {
            "messages": [],
            "ground_truth": json.dumps({"func_name": "validate_lowercase"}),
            "dataset": "ifeval",
        }

        with pytest.raises(ValueError, match="messages"):
            transform_to_ifbench(rlvr_example, index=0)

    def test_transform_missing_messages_raises_error(self):
        """Test error handling for missing messages field"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench

        rlvr_example = {
            "ground_truth": json.dumps({"func_name": "validate_lowercase"}),
            "dataset": "ifeval",
        }

        with pytest.raises(ValueError, match="messages"):
            transform_to_ifbench(rlvr_example, index=0)

    def test_transform_invalid_ground_truth_json(self):
        """Test error handling for invalid ground truth JSON"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench

        rlvr_example = {
            "messages": [{"content": "Test", "role": "user"}],
            "ground_truth": "{invalid json",
            "dataset": "ifeval",
        }

        with pytest.raises(json.JSONDecodeError):
            transform_to_ifbench(rlvr_example, index=0)

    def test_transform_unknown_function(self):
        """Test error handling for unknown function name"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench

        rlvr_example = {
            "messages": [{"content": "Test", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "unknown_function_xyz"}),
            "dataset": "ifeval",
        }

        with pytest.raises(KeyError, match="Unknown RLVR function"):
            transform_to_ifbench(rlvr_example, index=0)

    def test_transform_default_dataset(self):
        """Test default dataset value if not provided"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench

        rlvr_example = {
            "messages": [{"content": "Test", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "validate_lowercase"}),
            # No dataset field
        }

        result = transform_to_ifbench(rlvr_example, index=0)

        assert result["dataset"] == "ifeval"

    def test_transform_multiple_params(self):
        """Test transformation with complex parameters"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench

        rlvr_example = {
            "messages": [{"content": "Test", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "validate_paragraphs", "N": 3, "i": 2, "first_word": "Hello"}),
            "dataset": "ifeval",
        }

        result = transform_to_ifbench(rlvr_example, index=0)

        gt = ast.literal_eval(result["reward_model"]["ground_truth"])
        kwargs = gt[0]["kwargs"][0]

        assert kwargs["num_paragraphs"] == 3
        assert kwargs["nth_paragraph"] == 2
        assert kwargs["first_word"] == "Hello"


class TestSchemaValidation:
    """Test IFBench-VERL schema validation."""

    def test_validate_valid_schema(self):
        """Test validation of valid schema"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench, validate_ifbench_schema

        rlvr_example = {
            "messages": [{"content": "Test", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "validate_lowercase"}),
            "dataset": "ifeval",
        }

        result = transform_to_ifbench(rlvr_example, index=0)

        assert validate_ifbench_schema(result) is True

    def test_validate_missing_field(self):
        """Test validation fails for missing required field"""
        from datatrove.preprocessing.rlvr_to_ifbench import validate_ifbench_schema

        incomplete = {
            "data_source": "allenai/IF_multi_constraints_upto5",
            "prompt": [{"content": "Test", "role": "user"}],
            # Missing other required fields
        }

        with pytest.raises(AssertionError, match="Missing required field"):
            validate_ifbench_schema(incomplete)

    def test_validate_invalid_prompt_structure(self):
        """Test validation fails for invalid prompt structure"""
        from datatrove.preprocessing.rlvr_to_ifbench import validate_ifbench_schema

        invalid = {
            "data_source": "allenai/IF_multi_constraints_upto5",
            "prompt": "not a list",  # Should be a list
            "ability": "instruction_following",
            "reward_model": {"style": "ifeval", "ground_truth": '[{}]'},
            "extra_info": {"index": 0},
            "dataset": "ifeval",
        }

        with pytest.raises(AssertionError, match="prompt must be a list"):
            validate_ifbench_schema(invalid)

    def test_validate_invalid_ground_truth(self):
        """Test validation fails for invalid ground_truth Python literal"""
        from datatrove.preprocessing.rlvr_to_ifbench import validate_ifbench_schema

        invalid = {
            "data_source": "allenai/IF_multi_constraints_upto5",
            "prompt": [{"content": "Test", "role": "user"}],
            "ability": "instruction_following",
            "reward_model": {"style": "ifeval", "ground_truth": "{invalid}"},
            "extra_info": {"index": 0},
            "dataset": "ifeval",
        }

        with pytest.raises((ValueError, SyntaxError)):
            validate_ifbench_schema(invalid)

    def test_validate_all_constraint_types(self):
        """Test validation for all constraint types"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench, validate_ifbench_schema
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import RLVR_TO_IFEVAL_MAP

        for func_name in RLVR_TO_IFEVAL_MAP.keys():
            rlvr_example = {
                "messages": [{"content": f"Test {func_name}", "role": "user"}],
                "ground_truth": json.dumps({"func_name": func_name}),
                "dataset": "ifeval",
            }

            result = transform_to_ifbench(rlvr_example, index=0)
            assert validate_ifbench_schema(result) is True
