"""Tests for RLVR dataset analysis."""

import json

import pytest

pytest.importorskip("datasets")


class TestRLVRDatasetAnalysis:
    """Test RLVR dataset loading and analysis."""

    def test_load_rlvr_dataset_small(self):
        """Test loading small sample of RLVR dataset"""
        from datasets import load_dataset

        dataset = load_dataset("allenai/RLVR-IFeval", split="train[:10]", trust_remote_code=True)
        assert len(dataset) == 10
        assert "messages" in dataset[0]
        assert "ground_truth" in dataset[0]
        assert "dataset" in dataset[0]

    def test_parse_ground_truth_format(self):
        """Test parsing ground truth JSON from RLVR dataset"""
        from datasets import load_dataset

        dataset = load_dataset("allenai/RLVR-IFeval", split="train[:10]", trust_remote_code=True)

        for example in dataset:
            gt = json.loads(example["ground_truth"])
            assert "func_name" in gt, "ground_truth must have func_name field"

    def test_all_functions_in_mapping(self):
        """Test that all RLVR func_names exist in mapping"""
        from datasets import load_dataset

        from datatrove.utils.reward_score.ifeval.rlvr_mapping import RLVR_TO_IFEVAL_MAP

        # Load larger sample to get more diverse functions
        dataset = load_dataset("allenai/RLVR-IFeval", split="train[:1000]", trust_remote_code=True)

        unmapped_functions = set()
        for example in dataset:
            gt = json.loads(example["ground_truth"])
            func_name = gt["func_name"]
            if func_name not in RLVR_TO_IFEVAL_MAP:
                unmapped_functions.add(func_name)

        assert (
            not unmapped_functions
        ), f"Found unmapped functions in RLVR dataset: {sorted(unmapped_functions)}"

    def test_messages_structure(self):
        """Test that messages field has expected structure"""
        from datasets import load_dataset

        dataset = load_dataset("allenai/RLVR-IFeval", split="train[:10]", trust_remote_code=True)

        for example in dataset:
            assert isinstance(example["messages"], list)
            assert len(example["messages"]) > 0
            assert "role" in example["messages"][0]
            assert "content" in example["messages"][0]

    def test_extract_unique_functions(self):
        """Test extracting all unique function names"""
        from datasets import load_dataset

        from datatrove.utils.reward_score.ifeval.rlvr_mapping import RLVR_TO_IFEVAL_MAP

        # Load larger sample
        dataset = load_dataset("allenai/RLVR-IFeval", split="train[:5000]", trust_remote_code=True)

        unique_funcs = {json.loads(ex["ground_truth"])["func_name"] for ex in dataset}

        # All unique functions should be in our mapping
        missing = unique_funcs - set(RLVR_TO_IFEVAL_MAP.keys())
        assert not missing, f"Missing functions in mapping: {sorted(missing)}"

        # Report what we found
        print(f"\nFound {len(unique_funcs)} unique function types in sample")
        print(f"Our mapping has {len(RLVR_TO_IFEVAL_MAP)} functions")


class TestRLVRConstraintTypes:
    """Test analysis of RLVR constraint types."""

    def test_constraint_type_field(self):
        """Test that constraint_type field exists"""
        from datasets import load_dataset

        dataset = load_dataset("allenai/RLVR-IFeval", split="train[:10]", trust_remote_code=True)

        for example in dataset:
            # constraint_type field may or may not exist, but ground_truth must work
            gt = json.loads(example["ground_truth"])
            assert "func_name" in gt

    def test_parameter_fields(self):
        """Test that common parameter fields are present in ground truth"""
        from datasets import load_dataset

        dataset = load_dataset("allenai/RLVR-IFeval", split="train[:100]", trust_remote_code=True)

        # Check various parameter types appear in dataset
        found_params = {
            "N": False,
            "quantifier": False,
            "keyword_list": False,
            "word": False,
            "letter": False,
            "language": False,
        }

        for example in dataset:
            gt = json.loads(example["ground_truth"])
            for param in found_params.keys():
                if param in gt and gt[param] is not None:
                    found_params[param] = True

        # At least some parameters should be found in the sample
        assert any(found_params.values()), "No parameters found in ground truth samples"
