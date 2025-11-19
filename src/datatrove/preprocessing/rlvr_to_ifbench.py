"""Transform RLVR-IFeval dataset to IFBench-VERL format.

This module provides functions to convert RLVR-IFeval dataset format
to the IFBench-VERL format used by datatrove. This enables using RLVR
data with the existing IFEval scorer infrastructure.
"""

import ast
import json
from typing import Any, Dict, List, Optional

from datatrove.utils.reward_score.ifeval.rlvr_mapping import get_ifeval_instruction_id, map_param_names


def parse_rlvr_ground_truth(ground_truth_str: str) -> List[Dict[str, Any]]:
    """Parse RLVR ground truth JSON to IFEval format.

    RLVR ground truth is a JSON string with fields like:
    {
        "func_name": "validate_lowercase",
        "N": null,
        "quantifier": null,
        ...
    }

    IFEval format is a list of constraint dicts:
    [
        {
            "instruction_id": ["change_case:english_lowercase"],
            "kwargs": [None]
        }
    ]

    Args:
        ground_truth_str: JSON string from RLVR dataset

    Returns:
        List with single constraint dict in IFEval format

    Raises:
        json.JSONDecodeError: If ground_truth_str is not valid JSON
        KeyError: If func_name is not recognized

    Examples:
        >>> parse_rlvr_ground_truth('{"func_name": "validate_lowercase", "N": null}')
        [{'instruction_id': ['change_case:english_lowercase'], 'kwargs': [None]}]

        >>> parse_rlvr_ground_truth('{"func_name": "verify_paragraph_count", "N": 5}')
        [{'instruction_id': ['length_constraints:number_paragraphs'], 'kwargs': [{'num_paragraphs': 5}]}]
    """
    # Parse JSON
    gt_dict = json.loads(ground_truth_str)

    # Extract function name
    func_name = gt_dict["func_name"]

    # Map to IFEval instruction ID (raises KeyError if unknown)
    try:
        instruction_id = get_ifeval_instruction_id(func_name)
    except KeyError:
        raise KeyError(f"Unknown RLVR function: {func_name}")

    # Extract and transform parameters
    kwargs = map_param_names(func_name, gt_dict)

    # Return in IFEval format (single-constraint list)
    return [{"instruction_id": [instruction_id], "kwargs": [kwargs]}]


def transform_to_ifbench(rlvr_example: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Transform a single RLVR example to IFBench-VERL format.

    RLVR format:
    {
        "messages": [{"content": "...", "role": "user"}],
        "ground_truth": '{"func_name": "...", ...}',
        "dataset": "ifeval",
        "constraint_type": "...",  # optional
        "constraint": "..."  # optional
    }

    IFBench-VERL format:
    {
        "data_source": "allenai/IF_multi_constraints_upto5",
        "prompt": [{"content": "...", "role": "user"}],
        "ability": "instruction_following",
        "reward_model": {
            "style": "ifeval",
            "ground_truth": '[{"instruction_id": [...], "kwargs": [...]}]'
        },
        "extra_info": {"index": 0},
        "dataset": "ifeval"
    }

    Args:
        rlvr_example: Example from RLVR dataset
        index: Sequential index for this example

    Returns:
        Transformed example in IFBench-VERL format

    Raises:
        ValueError: If messages field is empty or invalid
        KeyError: If func_name is not recognized
        json.JSONDecodeError: If ground_truth is not valid JSON

    Example:
        >>> rlvr_ex = {
        ...     "messages": [{"content": "Write lowercase", "role": "user"}],
        ...     "ground_truth": '{"func_name": "validate_lowercase", "N": null}',
        ...     "dataset": "ifeval"
        ... }
        >>> result = transform_to_ifbench(rlvr_ex, index=0)
        >>> result["data_source"]
        'allenai/IF_multi_constraints_upto5'
        >>> result["ability"]
        'instruction_following'
    """
    # Validate messages
    if "messages" not in rlvr_example or not rlvr_example["messages"]:
        raise ValueError("RLVR example must have non-empty 'messages' field")

    # Parse and transform ground truth
    ground_truth_list = parse_rlvr_ground_truth(rlvr_example["ground_truth"])

    # Build IFBench-VERL format
    ifbench_example = {
        "data_source": "allenai/IF_multi_constraints_upto5",
        "prompt": rlvr_example["messages"],
        "ability": "instruction_following",
        "reward_model": {
            "style": "ifeval",
            # Use repr() to preserve Python syntax (None instead of null)
            # This matches ifbench-verl format and is compatible with ast.literal_eval()
            "ground_truth": repr(ground_truth_list),
        },
        "extra_info": {"index": index},
        "dataset": rlvr_example.get("dataset", "ifeval"),
    }

    return ifbench_example


def validate_ifbench_schema(example: Dict[str, Any]) -> bool:
    """Validate that an example matches the IFBench-VERL schema.

    Args:
        example: Example to validate

    Returns:
        True if valid

    Raises:
        AssertionError: If validation fails

    Example:
        >>> ex = transform_to_ifbench(rlvr_example, index=0)
        >>> validate_ifbench_schema(ex)
        True
    """
    # Check top-level fields
    required_fields = ["data_source", "prompt", "ability", "reward_model", "extra_info", "dataset"]
    for field in required_fields:
        assert field in example, f"Missing required field: {field}"

    # Check prompt structure
    assert isinstance(example["prompt"], list), "prompt must be a list"
    assert len(example["prompt"]) > 0, "prompt must not be empty"
    assert "role" in example["prompt"][0], "prompt[0] must have 'role' field"
    assert "content" in example["prompt"][0], "prompt[0] must have 'content' field"

    # Check reward_model structure
    assert "style" in example["reward_model"], "reward_model must have 'style' field"
    assert "ground_truth" in example["reward_model"], "reward_model must have 'ground_truth' field"
    assert example["reward_model"]["style"] == "ifeval", "reward_model.style must be 'ifeval'"

    # Validate ground_truth is valid Python literal (uses ast.literal_eval, not json.loads)
    gt = ast.literal_eval(example["reward_model"]["ground_truth"])
    assert isinstance(gt, list), "ground_truth must be a list"
    assert len(gt) > 0, "ground_truth must not be empty"
    assert "instruction_id" in gt[0], "ground_truth[0] must have 'instruction_id'"
    assert "kwargs" in gt[0], "ground_truth[0] must have 'kwargs'"

    # Check data_source
    assert example["data_source"] == "allenai/IF_multi_constraints_upto5", "data_source mismatch"

    # Check ability
    assert example["ability"] == "instruction_following", "ability must be 'instruction_following'"

    # Check extra_info
    assert "index" in example["extra_info"], "extra_info must have 'index' field"

    return True
