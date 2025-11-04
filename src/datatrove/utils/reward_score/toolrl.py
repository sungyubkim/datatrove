"""
ToolRL reward scoring for tool learning tasks.

This module implements the reward function from the ToolRL paper:
"ToolRL: Reward is All Tool Learning Needs" (Qian et al., 2025)

The reward function evaluates model outputs on three components:
1. Format: Correct XML-like structure (<think>, <tool_call>, <response>)
2. Correctness: Tool name and parameter matching
3. Length (optional): Reasoning length in <think> tags

Reference: https://github.com/qiancheng0/ToolRL
"""

import re
import json
import os
from collections import Counter


def match_score(list1, list2):
    """
    Compute a similarity score considering element frequency, ignoring order.

    Args:
        list1: First list of elements
        list2: Second list of elements

    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    if list1 == list2:
        return 1.0

    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)
    count2 = Counter(list2)

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection

    return intersection / max_possible if max_possible > 0 else 0.0


def compute_format_reward(response, answer, max_reward=1.0, min_reward=0.0):
    """
    Evaluate format correctness of the response.

    Expected formats based on ground truth:
    - Response only: <think>...</think>\n<response>...</response>
    - Tool call only: <think>...</think>\n<tool_call>\n...\n</tool_call>
    - Both: <think>...</think>\n<tool_call>\n...\n</tool_call>\n<response>...</response>
    - Think only: <think>...</think>

    Args:
        response: Model output string
        answer: Ground truth string
        max_reward: Maximum reward for correct format
        min_reward: Minimum reward for incorrect format

    Returns:
        float: Format reward score
    """
    reward = min_reward

    if "<response>" in answer and "<tool_call>" not in answer:
        # Expect: <think>...</think>\n<response>...</response>
        pattern = r"^<think>.*?</think>\n<response>.*?</response>$"
        if re.search(pattern, response, re.DOTALL) and \
           response.count("<response>") == 1 and response.count("</response>") == 1:
            reward = max_reward

    elif "<response>" not in answer and "<tool_call>" in answer:
        # Expect: <think>...</think>\n<tool_call>\n...\n</tool_call>
        pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$"
        if re.search(pattern, response, re.DOTALL) and \
           response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1:
            reward = max_reward

    elif "<response>" in answer and "<tool_call>" in answer:
        # Expect: <think>...</think>\n<tool_call>\n...\n</tool_call>\n<response>...</response>
        pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$"
        if re.search(pattern, response, re.DOTALL) and \
           response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1 and \
           response.count("<response>") == 1 and response.count("</response>") == 1:
            reward = max_reward

    else:
        # Expect: <think>...</think>
        pattern = r"^<think>.*?</think>$"
        if re.search(pattern, response, re.DOTALL):
            reward = max_reward

    return reward


def compute_tool_call_reward(gt_tools, pd_tools, max_reward=3.0, min_reward=-3.0):
    """
    Compute reward for tool call correctness.

    Evaluates:
    - Tool name matching (frequency-based)
    - Parameter key matching (frequency-based)
    - Parameter value matching (exact match)

    Args:
        gt_tools: List of ground truth tool dicts with "name" and "parameters"
        pd_tools: List of predicted tool dicts with "name" and "parameters"
        max_reward: Maximum possible reward
        min_reward: Minimum possible reward

    Returns:
        float: Correctness reward score
    """
    if gt_tools == pd_tools:
        return max_reward

    # Score tool name matching
    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(list(gt_names), list(pd_names))

    local_max_possible = 1.0
    used_pd_indices = set()

    # Match each gt_tool to best pd_tool
    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]

        local_max_possible += 1.0 + len(gt_params)

        best_match_score = 0.0
        best_match_index = -1

        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue

            pd_params = pd_tool["parameters"]

            # Score parameter keys
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))

            # Score parameter values
            correctness_score = sum(
                1.0 for k, v in gt_params.items()
                if k in pd_params and pd_params[k] == v
            )

            total_score = param_score + correctness_score

            if total_score > best_match_score:
                best_match_score = total_score
                best_match_index = i

        if best_match_index != -1:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    # Normalize to reward range
    normalized_score = score / local_max_possible if local_max_possible > 0 else 0.0
    return (max_reward - min_reward) * normalized_score + min_reward


def compute_correctness_reward(response, answer, max_reward=3.0, min_reward=-3.0):
    """
    Evaluate correctness of tool calls in the response.

    Args:
        response: Model output string
        answer: Ground truth string
        max_reward: Maximum reward for correct tool calls
        min_reward: Minimum reward for incorrect tool calls

    Returns:
        float: Correctness reward score
    """
    # If no tool call expected, return 0
    if "<tool_call>" not in answer:
        return 0.0

    # Parse ground truth tools
    try:
        gt_tool_call = answer.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        gt_tools = gt_tool_call.split("\n")
        gt_tools = [json.loads(tool) for tool in gt_tools]
    except Exception:
        return 0.0

    # Parse predicted tools
    try:
        if "<tool_call>" not in response or "</tool_call>" not in response:
            return min_reward

        pd_tool_call = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        pd_tools = pd_tool_call.split("\n")
        pd_tools = [json.loads(tool) for tool in pd_tools]

        return compute_tool_call_reward(gt_tools, pd_tools, max_reward, min_reward)
    except Exception:
        return min_reward


def compute_length_reward(response, max_reward=1.0, min_reward=0.0, max_words=512):
    """
    Reward longer reasoning in <think> tags.

    Args:
        response: Model output string
        max_reward: Maximum reward for optimal length
        min_reward: Minimum reward for short/missing reasoning
        max_words: Target word count for maximum reward

    Returns:
        float: Length reward score
    """
    if "<think>" not in response or "</think>" not in response:
        return min_reward

    think_content = response.split("<think>")[-1].split("</think>")[0].strip()
    word_count = len(think_content.split())

    # Linear scaling up to max_words
    reward_ratio = min(word_count / max_words, 1.0)

    return reward_ratio * (max_reward - min_reward) + min_reward


def extract_assistant_response(solution_str, model_type="auto"):
    """
    Extract assistant response from chat template.

    Supports:
    - Llama format: <|start_header_id|>assistant<|end_header_id|>...<|eot_id|>
    - Qwen format: <|im_start|>assistant...<|im_end|>
    - Raw format: Direct response without chat template

    Args:
        solution_str: Full model output including chat template
        model_type: Model type ("llama", "qwen", or "auto" for detection)

    Returns:
        str: Extracted assistant response
    """
    # Auto-detect model type
    if model_type == "auto":
        if "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
            model_type = "llama"
        elif "<|im_start|>assistant" in solution_str:
            model_type = "qwen"
        else:
            # Assume raw format
            return solution_str.strip()

    # Extract based on model type
    if model_type == "llama":
        return solution_str.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
    elif model_type == "qwen":
        return solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        return solution_str.strip()


def compute_score(
    model_output: str,
    ground_truth: str,
    step: int = 0,
    model_type: str = "auto",
    enable_length_reward: bool = False,
    **kwargs
) -> dict:
    """
    Compute ToolRL reward score for tool learning tasks.

    This function evaluates model outputs on three components:
    1. Format: XML-like structure validation
    2. Correctness: Tool name and parameter matching
    3. Length (optional): Reasoning length in <think> tags

    Args:
        model_output: Full model output string (may include chat template)
        ground_truth: Expected output from reward_model.ground_truth
        step: Training step number (for dynamic reward scaling, not used here)
        model_type: Model type for response extraction ("llama", "qwen", "auto")
        enable_length_reward: Whether to include length reward component
        **kwargs: Additional arguments (ignored)

    Returns:
        dict: Reward scores with keys:
            - score: Total reward (sum of all components)
            - reward_fmt: Format reward (0 to 1)
            - reward_correct: Correctness reward (-3 to 3)
            - reward_length: Length reward (0 to 1, if enabled)
            - reward_think: Binary indicator if <think> tags present
    """
    # Extract assistant response
    response = extract_assistant_response(model_output, model_type)

    # Compute component scores
    format_score = compute_format_reward(response, ground_truth, max_reward=1.0, min_reward=0.0)
    correctness_score = compute_correctness_reward(response, ground_truth, max_reward=3.0, min_reward=-3.0)

    # Optional length reward
    length_score = 0.0
    if enable_length_reward:
        length_score = compute_length_reward(response, max_reward=1.0, min_reward=0.0, max_words=512)

    # Binary indicator for <think> presence
    think_indicator = 1.0 if "<think>" in response and "</think>" in response else 0.0

    # Total score
    total_score = format_score + correctness_score + length_score

    return {
        "score": total_score,
        "reward_fmt": format_score,
        "reward_correct": correctness_score,
        "reward_length": length_score,
        "reward_think": think_indicator,
    }
