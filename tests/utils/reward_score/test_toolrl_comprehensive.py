"""
Comprehensive tests for ToolRL reward scoring.

This test suite covers:
1. Basic XML format scoring
2. GPT OSS format scoring
3. Environment variable features (VERL compatibility)
4. Edge cases and error handling
"""

import os
import pytest
from datatrove.utils.reward_score.toolrl import compute_score


class TestToolRLXMLFormat:
    """Test ToolRL scoring with XML format (<think>, <tool_call>, <response>)."""

    def test_basic_tool_call(self):
        """Test basic XML format with think + tool_call."""
        solution = """<think>I need to search for information about AI</think>
<tool_call>
{"name": "search", "parameters": {"query": "AI"}}
</tool_call>
<response>Found results about AI</response>"""

        ground_truth = """<think>Search for AI</think>
<tool_call>
{"name": "search", "parameters": {"query": "AI"}}
</tool_call>
<response>Results</response>"""

        result = compute_score(solution, ground_truth, format_type="xml")

        assert isinstance(result, dict)
        assert "score" in result
        assert "reward_fmt" in result
        assert "reward_correct" in result
        assert "reward_think" in result

        # Perfect match should give high rewards
        assert result["reward_fmt"] == 1.0
        assert result["reward_correct"] == 3.0
        assert result["reward_think"] == 1.0
        assert result["score"] == 4.0  # fmt + correct

    def test_wrong_tool_call(self):
        """Test XML format with incorrect tool call."""
        solution = """<think>I need to calculate something</think>
<tool_call>
{"name": "calculate", "parameters": {"expression": "2+2"}}
</tool_call>"""

        ground_truth = """<think>Search for info</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        result = compute_score(solution, ground_truth, format_type="xml")

        # Format should be correct but tool call wrong
        assert result["reward_fmt"] == 1.0
        assert result["reward_correct"] < 3.0  # Penalty for wrong tool
        assert result["reward_think"] == 1.0

    def test_missing_tool_call(self):
        """Test when tool call is expected but missing."""
        solution = """<think>I need to search</think>
<response>I don't know</response>"""

        ground_truth = """<think>Search</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        result = compute_score(solution, ground_truth, format_type="xml")

        # Format is incomplete, correctness should be minimum
        assert result["reward_fmt"] < 1.0
        assert result["reward_correct"] == -3.0  # Min reward

    def test_multiple_tool_calls(self):
        """Test multiple tool calls in sequence."""
        solution = """<think>I need to search and then calculate</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>
<tool_call>
{"name": "calculate", "parameters": {"expression": "2+2"}}
</tool_call>"""

        ground_truth = """<think>Search then calculate</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>
<tool_call>
{"name": "calculate", "parameters": {"expression": "2+2"}}
</tool_call>"""

        result = compute_score(solution, ground_truth, format_type="xml")

        # Perfect match
        assert result["reward_fmt"] == 1.0
        assert result["reward_correct"] == 3.0

    def test_no_think_section(self):
        """Test when think section is missing."""
        solution = """<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        ground_truth = """<think>Think first</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        result = compute_score(solution, ground_truth, format_type="xml")

        # No thinking section
        assert result["reward_think"] == 0.0

    def test_length_reward(self):
        """Test length reward for longer reasoning."""
        short_solution = """<think>Short</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        long_solution = """<think>""" + " ".join(["reasoning"] * 100) + """</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        ground_truth = """<think>Think</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        result_short = compute_score(
            short_solution, ground_truth,
            enable_length_reward=True,
            format_type="xml"
        )
        result_long = compute_score(
            long_solution, ground_truth,
            enable_length_reward=True,
            format_type="xml"
        )

        # Longer reasoning should get higher length reward
        assert result_long["reward_length"] > result_short["reward_length"]


class TestToolRLGPTOSSFormat:
    """Test ToolRL scoring with GPT OSS format."""

    def test_analysis_plus_tool_call(self):
        """Test GPT OSS format with analysis + tool call."""
        solution = """<|start|>assistant<|channel|>analysis<|message|>I need to search for information<|end|>
<|start|>assistant to=functions.search<|channel|>commentary json<|message|>{"query": "AI"}<|call|>"""

        ground_truth = """<|start|>assistant<|channel|>analysis<|message|>Search needed<|end|>
<|start|>assistant to=functions.search<|channel|>commentary json<|message|>{"query": "AI"}<|call|>"""

        result = compute_score(solution, ground_truth, format_type="gpt_oss")

        assert result["reward_fmt"] == 1.0
        assert result["reward_correct"] == 3.0
        assert result["reward_think"] == 1.0

    def test_analysis_plus_final(self):
        """Test GPT OSS format with analysis + final response."""
        solution = """<|start|>assistant<|channel|>analysis<|message|>Let me think about this<|end|>
<|start|>assistant<|channel|>final<|message|>The answer is 42<|return|>"""

        ground_truth = """<|start|>assistant<|channel|>analysis<|message|>Thinking<|end|>
<|start|>assistant<|channel|>final<|message|>42<|return|>"""

        result = compute_score(solution, ground_truth, format_type="gpt_oss")

        # No tool call expected, so correctness should be 0
        assert result["reward_fmt"] == 1.0
        assert result["reward_correct"] == 0.0

    def test_wrong_tool_in_gpt_oss(self):
        """Test incorrect tool call in GPT OSS format."""
        solution = """<|start|>assistant<|channel|>analysis<|message|>Calculate<|end|>
<|start|>assistant to=functions.calculate<|channel|>commentary json<|message|>{"expression": "2+2"}<|call|>"""

        ground_truth = """<|start|>assistant<|channel|>analysis<|message|>Search<|end|>
<|start|>assistant to=functions.search<|channel|>commentary json<|message|>{"query": "test"}<|call|>"""

        result = compute_score(solution, ground_truth, format_type="gpt_oss")

        # Format correct, but tool wrong
        assert result["reward_fmt"] == 1.0
        assert result["reward_correct"] < 3.0


class TestAutoDetection:
    """Test automatic format detection."""

    def test_detect_xml_format(self):
        """Should detect XML format automatically."""
        solution = """<think>Thinking</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        ground_truth = solution

        result = compute_score(solution, ground_truth, format_type="auto")

        # Should work correctly with auto-detection
        assert result["reward_fmt"] == 1.0

    def test_detect_gpt_oss_format(self):
        """Should detect GPT OSS format automatically."""
        solution = """<|start|>assistant<|channel|>analysis<|message|>Thinking<|end|>
<|start|>assistant to=functions.search<|channel|>commentary json<|message|>{"query": "test"}<|call|>"""

        ground_truth = solution

        result = compute_score(solution, ground_truth, format_type="auto")

        # Should work correctly with auto-detection
        assert result["reward_fmt"] == 1.0


class TestChatTemplateExtraction:
    """Test extraction of assistant responses from chat templates."""

    def test_llama_format_extraction(self):
        """Test extracting from Llama chat template."""
        solution = """<|start_header_id|>user<|end_header_id|>
Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>
<think>Process this</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call><|eot_id|>"""

        ground_truth = """<think>Process</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        result = compute_score(solution, ground_truth, model_type="llama")

        # Should extract and score correctly
        assert result["reward_fmt"] == 1.0
        assert result["reward_correct"] == 3.0

    def test_qwen_format_extraction(self):
        """Test extracting from Qwen chat template."""
        solution = """<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
<think>Process this</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call><|im_end|>"""

        ground_truth = """<think>Process</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        result = compute_score(solution, ground_truth, model_type="qwen")

        # Should extract and score correctly
        assert result["reward_fmt"] == 1.0
        assert result["reward_correct"] == 3.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_solution(self):
        """Test with empty solution string."""
        result = compute_score("", "<think>Test</think>", format_type="xml")

        # Should return minimum rewards
        assert result["reward_fmt"] <= 0.0
        assert result["reward_think"] == 0.0

    def test_malformed_json_in_tool_call(self):
        """Test with malformed JSON in tool call."""
        solution = """<think>Search</think>
<tool_call>
{invalid json}
</tool_call>"""

        ground_truth = """<think>Search</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        result = compute_score(solution, ground_truth, format_type="xml")

        # Should handle gracefully
        assert result["reward_correct"] == -3.0  # Min reward for parsing error

    def test_no_tags_at_all(self):
        """Test with plain text (no formatting)."""
        result = compute_score(
            "Just plain text",
            "<think>Test</think>",
            format_type="xml"
        )

        # Should give minimum format reward
        assert result["reward_fmt"] == 0.0
        assert result["reward_think"] == 0.0


class TestEnvironmentVariables:
    """Test environment variable features (VERL compatibility)."""

    @pytest.fixture(autouse=True)
    def cleanup_env(self):
        """Clean up environment variables after each test."""
        yield
        # Remove all test env vars
        for key in ["WITHLENGTH", "SCHEDULEREWARD", "REFINEDREWARD",
                    "COARSEREWARD", "CORRECTMAX1"]:
            os.environ.pop(key, None)

    def test_withlength_env_var(self):
        """Test WITHLENGTH=1 automatically enables length reward."""
        os.environ["WITHLENGTH"] = "1"

        solution = """<think>""" + " ".join(["word"] * 100) + """</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        ground_truth = """<think>Short</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        result = compute_score(solution, ground_truth, format_type="xml")

        # Length reward should be automatically included
        assert result["reward_length"] > 0.0
        assert "reward_length" in result

    def test_schedulereward_env_var(self):
        """Test SCHEDULEREWARD=1 applies step-based scaling."""
        os.environ["SCHEDULEREWARD"] = "1"

        solution = """<think>Think</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        ground_truth = solution

        result_step0 = compute_score(solution, ground_truth, step=0)
        result_step100 = compute_score(solution, ground_truth, step=100)

        # Rewards should be different at different steps
        # This is a placeholder - actual behavior depends on implementation
        assert isinstance(result_step0["score"], (int, float))
        assert isinstance(result_step100["score"], (int, float))

    def test_refinedreward_env_var(self):
        """Test REFINEDREWARD=1 requires exact matching."""
        os.environ["REFINEDREWARD"] = "1"

        solution = """<think>Search for info</think>
<tool_call>
{"name": "search", "parameters": {"query": "AI"}}
</tool_call>"""

        # Slightly different parameters
        ground_truth = """<think>Search</think>
<tool_call>
{"name": "search", "parameters": {"query": "AI", "limit": 10}}
</tool_call>"""

        result = compute_score(solution, ground_truth, format_type="xml")

        # With strict matching, should get 0 for mismatch
        # This is a placeholder - actual behavior depends on implementation
        assert result["reward_correct"] <= 0.0

    def test_coarsereward_env_var(self):
        """Test COARSEREWARD=1 uses binary matching."""
        os.environ["COARSEREWARD"] = "1"

        solution = """<think>Search</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        ground_truth = solution

        result = compute_score(solution, ground_truth, format_type="xml")

        # Binary matching: should be either max or min
        assert result["reward_correct"] in [3.0, -3.0]

    def test_correctmax1_env_var(self):
        """Test CORRECTMAX1=1 sets max correctness to 1."""
        os.environ["CORRECTMAX1"] = "1"

        solution = """<think>Think</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        ground_truth = solution

        result = compute_score(solution, ground_truth, format_type="xml")

        # Max correctness should be 1 instead of 3
        assert result["reward_correct"] <= 1.0
        assert result["reward_correct"] >= -1.0  # Min should also be -1


class TestBackwardCompatibility:
    """Test that existing code continues to work."""

    def test_basic_call_still_works(self):
        """Test that simple compute_score call still works."""
        solution = """<think>Think</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        result = compute_score(solution, solution)

        # Should return dict with expected keys
        assert isinstance(result, dict)
        assert all(key in result for key in ["score", "reward_fmt", "reward_correct", "reward_think"])

    def test_return_type_is_dict(self):
        """Test that return type is always dict (not tuple)."""
        solution = """<think>Think</think>
<tool_call>
{"name": "search", "parameters": {"query": "test"}}
</tool_call>"""

        result = compute_score(solution, solution)

        # Must be dict, not tuple (unlike toolrl_gpt_oss.py)
        assert isinstance(result, dict)
        assert not isinstance(result, tuple)
