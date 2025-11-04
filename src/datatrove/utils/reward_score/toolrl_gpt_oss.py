# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reward functions for GPT OSS 120B format.

This module provides reward functions for evaluating model outputs in GPT OSS format:
- Analysis channel: <|start|>assistant<|channel|>analysis<|message|>...<|end|>
- Tool calls: <|start|>assistant to=functions.{name}<|channel|>commentary json<|message|>{params}<|call|>
- Final response: <|start|>assistant<|channel|>final<|message|>...<|return|>
"""

import re
import json
import os
from collections import Counter


def match_score(list1, list2):
    """Compute a similarity score considering element frequency, ignoring order."""
    if list1 == list2:
        return 1.0

    if os.getenv("REFINEDREWARD", 0) == "1":
        print("REFINEDREWARD is set to 1, so strict match is used")
        if list1 != list2:
            return 0.0

    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)
    count2 = Counter(list2)

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection

    return intersection / max_possible if max_possible > 0 else 0.0


# Customized reward functions: format (GPT OSS version)
def customize_format_reward_func(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step >= 30:
            max_possible_reward = max_possible_reward / 2
            min_possible_reward = min_possible_reward / 2
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward

    # Schedule reward
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = 2 - (2 - max_possible_reward) * step / 150
        min_possible_reward = -2 + (2 + min_possible_reward) * step / 150
        if max_possible_reward < 1.0:
            max_possible_reward = 1.0
        if min_possible_reward > -1.0:
            min_possible_reward = -1.0

    rewards = []
    responses = [completion[0]['content'] for completion in completions]

    print("\n======= Answer ======= ")
    print(answer[0])
    print("\n======= Responses ======= ")
    for idx, response in enumerate(responses):
        print(f"*** Response {idx+1}***\n{response}")

    for response, ans in zip(responses, answer):
        reward = min_possible_reward

        # Check if answer contains tool calls or final response
        has_tool_call_in_ans = 'to=functions.' in ans and '<|call|>' in ans
        has_final_in_ans = '<|channel|>final' in ans and '<|return|>' in ans
        has_analysis_in_ans = '<|channel|>analysis' in ans

        # Pattern 1: Analysis + Tool call (most common)
        if has_analysis_in_ans and has_tool_call_in_ans and not has_final_in_ans:
            # Should have: analysis channel + tool call(s)
            pattern = r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>.*?<\|start\|>assistant to=functions\.\w+<\|channel\|>commentary json<\|message\|>.*?<\|call\|>'
            if re.search(pattern, response, re.DOTALL):
                # Count tokens to ensure format is correct
                if ('<|start|>' in response and '<|channel|>' in response and
                    '<|message|>' in response and '<|call|>' in response):
                    reward = max_possible_reward

        # Pattern 2: Analysis + Final response (no tool call)
        elif has_analysis_in_ans and has_final_in_ans and not has_tool_call_in_ans:
            # Should have: analysis channel + final channel
            pattern = r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>.*?<\|start\|>assistant<\|channel\|>final<\|message\|>.*?<\|return\|>'
            if re.search(pattern, response, re.DOTALL):
                if ('<|start|>' in response and '<|channel|>' in response and
                    '<|message|>' in response and '<|return|>' in response):
                    reward = max_possible_reward

        # Pattern 3: Analysis + Tool call + Final response (rare)
        elif has_analysis_in_ans and has_tool_call_in_ans and has_final_in_ans:
            # Should have all three
            pattern = r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>.*?<\|start\|>assistant to=functions\.\w+<\|channel\|>commentary json<\|message\|>.*?<\|call\|>.*?<\|start\|>assistant<\|channel\|>final<\|message\|>.*?<\|return\|>'
            if re.search(pattern, response, re.DOTALL):
                if ('<|start|>' in response and '<|channel|>' in response and
                    '<|message|>' in response and '<|call|>' in response and '<|return|>' in response):
                    reward = max_possible_reward

        # Pattern 4: Analysis only (very rare)
        elif has_analysis_in_ans and not has_tool_call_in_ans and not has_final_in_ans:
            pattern = r'^<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>$'
            if re.search(pattern, response, re.DOTALL):
                reward = max_possible_reward

        rewards.append(reward)

    print("\n======= Reward for <format> =======")
    print("Reward function for <format> is called ...")
    print(rewards)
    return rewards


# Customized reward functions: length (GPT OSS version)
def customize_length_reward_func(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    # Schedule length
    if os.getenv("SCHEDULELENGTH", 0) == "1":
        print("SCHEDULELENGTH is set to 1, so schedule max reward for length is used")
        max_reward_len = (640 - 384) * step / 105 + 384
    else:
        max_reward_len = 512

    """Reward function that gives higher scores to longer analysis channel content."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []

    for response, ans in zip(responses, answer):
        # Extract analysis channel content
        analysis_pattern = r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>'
        analysis_match = re.search(analysis_pattern, response, re.DOTALL)

        if not analysis_match:
            rewards.append(min_possible_reward)
            continue

        analysis_content = analysis_match.group(1).strip()
        word_count = len(analysis_content.split())

        reward = round(word_count / max_reward_len, 2)
        if reward > 1.0:
            reward = 1.0

        final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
        rewards.append(final_reward)

    print("\n======= Reward for <length> =======")
    print("Reward function for <length> is called ...")
    print(rewards)
    return rewards


def parse_tool_calls_from_gpt_oss(text):
    """
    Parse tool calls from GPT OSS format.

    Format: <|start|>assistant to=functions.{name}<|channel|>commentary json<|message|>{params}<|call|>

    Returns:
        List of dicts with 'name' and 'parameters' keys
    """
    tools = []

    # Pattern to match tool calls
    # Captures: to=functions.{name} and {params}
    pattern = r'<\|start\|>assistant to=functions\.(\w+)<\|channel\|>commentary json<\|message\|>(.*?)<\|call\|>'

    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        tool_name = match.group(1)
        params_json = match.group(2).strip()

        try:
            parameters = json.loads(params_json)
            tools.append({
                "name": tool_name,
                "parameters": parameters
            })
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse tool call parameters: {params_json}")
            print(f"Error: {e}")
            # Add with raw string if parsing fails
            tools.append({
                "name": tool_name,
                "parameters": {"_raw": params_json}
            })

    return tools


def compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward):
    if gt_tools == pd_tools:
        print("Max possible score:", "Exact Match!")
        print("Score:", max_possible_reward)
        return max_possible_reward

    if os.getenv("COARSEREWARD", 0) == "1":
        print("COARSEREWARD is set to 1, so coarse reward is used")
        if gt_tools != pd_tools:
            return min_possible_reward

    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(list(gt_names), list(pd_names))

    local_max_possible = 1.0
    used_pd_indices = set()

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]

        if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
            print("INTERMEDIATEREWARD is set to 1, so local max possible is changed")
            local_max_possible += 1.0
        else:
            local_max_possible += 1.0 + len(gt_params)

        best_match = None
        best_match_score = 0.0
        best_match_index = -1

        # Find the best matching unused pd_tool
        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue

            if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
                if gt_tool == pd_tool:
                    best_match = pd_tool
                    best_match_index = i
                    best_match_score = 1.0
                    break
                else:
                    continue

            pd_params = pd_tool["parameters"]
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))

            # Calculate correctness score for parameter values
            correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

            total_score = param_score + correctness_score

            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i

        if best_match:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    print()
    print("Max possible score:", local_max_possible)
    print("Score:", score)

    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward


# Customized reward functions: tool call correctness (GPT OSS version)
def customize_correctness_reward_tool(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step < 30:
            max_possible_reward = max_possible_reward / 3
            min_possible_reward = min_possible_reward / 3
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward

    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = (max_possible_reward - 2) * step / 150 + 2
        min_possible_reward = (min_possible_reward + 2) * step / 150 - 2
        if max_possible_reward > 3.0:
            max_possible_reward = 3.0
        if min_possible_reward < -3.0:
            min_possible_reward = -3.0

    responses = [completion[0]['content'] for completion in completions]
    rewards = []

    for response, ans in zip(responses, answer):
        reward = 0.0

        # Check if answer contains tool calls
        if 'to=functions.' not in ans or '<|call|>' not in ans:
            # No tool call expected, so no reward/penalty
            rewards.append(reward)
            continue

        # Parse ground truth tool calls
        try:
            gt_tools = parse_tool_calls_from_gpt_oss(ans)
        except Exception as e:
            print(f"Error parsing ground truth tool calls: {e}")
            rewards.append(min_possible_reward)
            continue

        # Parse predicted tool calls
        try:
            # Check if response has tool call format
            if 'to=functions.' not in response or '<|call|>' not in response:
                reward = min_possible_reward
            else:
                pd_tools = parse_tool_calls_from_gpt_oss(response)
                reward = compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward)
        except Exception as e:
            print(f"Error parsing predicted tool calls: {e}")
            reward = min_possible_reward

        rewards.append(reward)

    print("\n======= Reward for <tool call> =======")
    print("Reward function for <tool call> correctness is called ...")
    print(rewards)
    return rewards


def compute_score(solution_str, ground_truth, step=0):
    """
    The scoring function for GPT OSS format.

    Args:
        solution_str: the solution text (includes prompt + generation)
        ground_truth: the ground truth in GPT OSS format
        step: training step for scheduling rewards
    """
    # Extract assistant response from GPT OSS format
    # Format: <|start|>assistant...<|end|> or <|start|>assistant...<|return|> or <|start|>assistant...<|call|>

    # Find all assistant messages
    assistant_pattern = r'<\|start\|>assistant.*?(?:<\|end\|>|<\|return\|>|<\|call\|>)'
    matches = re.findall(assistant_pattern, solution_str, re.DOTALL)

    if matches:
        # Join all assistant messages
        predict_str = '\n'.join(matches)
    else:
        # Fallback: try to extract everything after last <|start|>assistant
        if '<|start|>assistant' in solution_str:
            predict_str = '<|start|>assistant' + solution_str.split('<|start|>assistant')[-1]
        else:
            print("Warning: Could not find assistant response in GPT OSS format")
            predict_str = solution_str

    # Set reward ranges
    if str(os.getenv("CORRECTMAX1", 0)) == "1":
        print("CORRECTMAX1 is set to 1, so max score is set to 1")
        tool_max_possible = 1.0
        tool_min_possible = -1.0
    else:
        tool_max_possible = 3.0
        tool_min_possible = -3.0

    format_max_possible = 1.0
    format_min_possible = 0.0

    length_max_possible = 1.0
    length_min_possible = 0.0

    completions = [[{"role": "assistant", "content": predict_str}]]
    answer = [ground_truth]

    # Compute individual scores
    format_score = customize_format_reward_func(completions, answer, step, format_max_possible, format_min_possible)[0]
    correctness_score = customize_correctness_reward_tool(completions, answer, step, tool_max_possible, tool_min_possible)[0]

    if str(os.getenv("WITHLENGTH", 0)) == "1":
        print("WITHLENGTH is set to 1, so length score is set!")
        length_score = customize_length_reward_func(completions, answer, step, length_max_possible, length_min_possible)[0]
    else:
        length_score = 0

    score = format_score + correctness_score + length_score

    return score, format_score, correctness_score, length_score
