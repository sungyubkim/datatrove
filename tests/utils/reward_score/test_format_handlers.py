"""
Unit tests for format handlers.

Tests the format detection, extraction, and scoring functionality
for both XML and GPT OSS formats.
"""

import pytest
from datatrove.utils.reward_score.format_handlers import (
    detect_format,
    get_format_handler,
    extract_thinking,
    remove_thinking,
    extract_final_response,
    extract_tool_calls,
    check_format,
    extract_assistant_response,
    XMLFormatHandler,
    GPTOSSFormatHandler,
)


class TestFormatDetection:
    """Test format auto-detection."""

    def test_detect_xml_format(self):
        """XML format should be detected from <think> tags."""
        text = "<think>reasoning</think>\n<response>answer</response>"
        assert detect_format(text) == "xml"

    def test_detect_gpt_oss_format(self):
        """GPT OSS format should be detected from special tokens."""
        text = "<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>"
        assert detect_format(text) == "gpt_oss"

    def test_detect_preference_gpt_oss(self):
        """Preference should be respected if format matches."""
        text = "<|start|>assistant<|channel|>final<|message|>answer<|return|>"
        assert detect_format(text, preference="gpt_oss") == "gpt_oss"

    def test_detect_fallback_to_xml(self):
        """Should fall back to XML if no format detected."""
        text = "plain text without format markers"
        assert detect_format(text) == "xml"


class TestXMLFormatHandler:
    """Test XML format handler."""

    @pytest.fixture
    def handler(self):
        return XMLFormatHandler()

    def test_detect_xml_format(self, handler):
        """Should detect XML format from tags."""
        assert handler.detect("<think>test</think>") is True
        assert handler.detect("<tool_call>test</tool_call>") is True
        assert handler.detect("<response>test</response>") is True
        assert handler.detect("plain text") is False

    def test_extract_thinking(self, handler):
        """Should extract thinking content from <think> tags."""
        text = "<think>reasoning here</think>\nFinal answer"
        thinking, success = handler.extract_thinking(text)
        assert thinking == "reasoning here"
        assert success is True

    def test_extract_thinking_no_tags(self, handler):
        """Should return None if no thinking tags."""
        text = "Final answer without thinking"
        thinking, success = handler.extract_thinking(text)
        assert thinking is None
        assert success is True

    def test_remove_thinking(self, handler):
        """Should remove <think> sections."""
        text = "<think>reasoning</think>\nFinal answer"
        result = handler.remove_thinking(text)
        assert result == "Final answer"
        assert "<think>" not in result

    def test_extract_final_response(self, handler):
        """Should extract from <response> tags."""
        text = "<think>reasoning</think>\n<response>final answer</response>"
        response = handler.extract_final_response(text)
        assert response == "final answer"

    def test_extract_tool_calls(self, handler):
        """Should extract tool calls from XML format."""
        text = '<tool_call>\n{"name": "search", "parameters": {"query": "test"}}\n</tool_call>'
        tools = handler.extract_tool_calls(text)
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert tools[0]["parameters"]["query"] == "test"

    def test_check_format_valid(self, handler):
        """Should validate correct XML format."""
        text = "<think>reasoning</think><answer>code</answer>"
        assert handler.check_format(text, ["think", "answer"]) is True

    def test_extract_assistant_response_qwen(self, handler):
        """Should extract from Qwen chat template."""
        text = "system<|im_start|>assistant\nresponse content<|im_end|>user"
        response = handler.extract_assistant_response(text, model_type="qwen")
        assert "response content" in response

    def test_extract_assistant_response_llama(self, handler):
        """Should extract from Llama chat template."""
        text = "<|start_header_id|>assistant<|end_header_id|>response content<|eot_id|>"
        response = handler.extract_assistant_response(text, model_type="llama")
        assert "response content" in response

    def test_compute_format_reward_qwen3_style(self, handler):
        """Should reward Qwen3 format with trailing content."""
        response = "<think>reasoning here</think>\nFinal answer: Paris"
        ground_truth = "<think>gt reasoning</think>\nFinal answer: Something"

        reward = handler.compute_format_reward(response, ground_truth, max_reward=1.0, min_reward=0.0)
        assert reward == 1.0, "Qwen3 format (<think> + plain text) should get full reward"

    def test_compute_format_reward_traditional_response_tag(self, handler):
        """Should reward traditional format with <response> tag."""
        response = "<think>reasoning</think>\n<response>answer</response>"
        ground_truth = "<think>gt</think>\n<response>gt_answer</response>"

        reward = handler.compute_format_reward(response, ground_truth)
        assert reward == 1.0

    def test_compute_format_reward_tool_call_format(self, handler):
        """Should reward tool call format."""
        response = '<think>search needed</think>\n<tool_call>\n{"name": "search", "parameters": {"query": "AI"}}\n</tool_call>'
        ground_truth = '<think>gt</think>\n<tool_call>\n{"name": "search", "parameters": {}}\n</tool_call>'

        reward = handler.compute_format_reward(response, ground_truth)
        assert reward == 1.0

    def test_compute_format_reward_think_only_no_trailing(self, handler):
        """Should reward <think> only without trailing content."""
        response = "<think>just thinking</think>"
        ground_truth = "<think>gt</think>"

        reward = handler.compute_format_reward(response, ground_truth)
        assert reward == 1.0

    def test_compute_format_reward_invalid_format(self, handler):
        """Should not reward invalid format."""
        response = "No tags here"
        ground_truth = "<think>gt</think>"

        reward = handler.compute_format_reward(response, ground_truth, max_reward=1.0, min_reward=0.0)
        assert reward == 0.0


class TestGPTOSSFormatHandler:
    """Test GPT OSS format handler."""

    @pytest.fixture
    def handler(self):
        return GPTOSSFormatHandler()

    def test_detect_gpt_oss_format(self, handler):
        """Should detect GPT OSS format from special tokens."""
        assert handler.detect("<|start|>assistant<|channel|>analysis") is True
        assert handler.detect("plain text") is False

    def test_extract_thinking(self, handler):
        """Should extract from analysis channel."""
        text = "<|start|>assistant<|channel|>analysis<|message|>reasoning here<|end|>"
        thinking, success = handler.extract_thinking(text)
        assert thinking == "reasoning here"
        assert success is True

    def test_extract_thinking_no_analysis(self, handler):
        """Should return None if no analysis channel."""
        text = "<|start|>assistant<|channel|>final<|message|>answer<|return|>"
        thinking, success = handler.extract_thinking(text)
        assert thinking is None
        assert success is True

    def test_remove_thinking(self, handler):
        """Should remove analysis channel."""
        text = "<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>\n<|start|>assistant<|channel|>final<|message|>answer<|return|>"
        result = handler.remove_thinking(text)
        assert "<|channel|>analysis" not in result
        assert "<|channel|>final" in result

    def test_extract_final_response(self, handler):
        """Should extract from final channel."""
        text = "<|start|>assistant<|channel|>final<|message|>final answer<|return|>"
        response = handler.extract_final_response(text)
        assert response == "final answer"

    def test_extract_tool_calls(self, handler):
        """Should extract tool calls from GPT OSS format."""
        text = '<|start|>assistant to=functions.search<|channel|>commentary json<|message|>{"query": "test"}<|call|>'
        tools = handler.extract_tool_calls(text)
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert tools[0]["parameters"]["query"] == "test"

    def test_has_analysis(self, handler):
        """Should detect analysis channel."""
        text = "<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>"
        assert handler.has_analysis(text) is True

    def test_has_final_response(self, handler):
        """Should detect final channel."""
        text = "<|start|>assistant<|channel|>final<|message|>answer<|return|>"
        assert handler.has_final_response(text) is True

    def test_has_tool_call(self, handler):
        """Should detect tool calls."""
        text = '<|start|>assistant to=functions.search<|channel|>commentary json<|message|>{}<|call|>'
        assert handler.has_tool_call(text) is True

    def test_compute_format_reward_analysis_and_final(self, handler):
        """Should reward correct analysis + final pattern."""
        ground_truth = "<|start|>assistant<|channel|>analysis<|message|>gt<|end|>\n<|start|>assistant<|channel|>final<|message|>gt<|return|>"
        response = "<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>\n<|start|>assistant<|channel|>final<|message|>answer<|return|>"
        reward = handler.compute_format_reward(response, ground_truth, max_reward=1.0, min_reward=0.0)
        assert reward == 1.0

    def test_compute_format_reward_analysis_and_tool_call(self, handler):
        """Should reward correct analysis + tool call pattern."""
        ground_truth = '<|start|>assistant<|channel|>analysis<|message|>gt<|end|>\n<|start|>assistant to=functions.search<|channel|>commentary json<|message|>{}<|call|>'
        response = '<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>\n<|start|>assistant to=functions.search<|channel|>commentary json<|message|>{"query": "test"}<|call|>'
        reward = handler.compute_format_reward(response, ground_truth, max_reward=1.0, min_reward=0.0)
        assert reward == 1.0


class TestUnifiedAPIFunctions:
    """Test unified API convenience functions."""

    def test_extract_thinking_xml(self):
        """Should extract thinking from XML format."""
        text = "<think>reasoning</think>\nFinal answer"
        thinking, success = extract_thinking(text, format_type="xml")
        assert thinking == "reasoning"
        assert success is True

    def test_extract_thinking_gpt_oss(self):
        """Should extract thinking from GPT OSS format."""
        text = "<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>"
        thinking, success = extract_thinking(text, format_type="gpt_oss")
        assert thinking == "reasoning"
        assert success is True

    def test_extract_thinking_auto_detect(self):
        """Should auto-detect format and extract thinking."""
        xml_text = "<think>xml reasoning</think>"
        xml_thinking, _ = extract_thinking(xml_text, format_type="auto")
        assert xml_thinking == "xml reasoning"

        gpt_oss_text = "<|start|>assistant<|channel|>analysis<|message|>gpt reasoning<|end|>"
        gpt_thinking, _ = extract_thinking(gpt_oss_text, format_type="auto")
        assert gpt_thinking == "gpt reasoning"

    def test_remove_thinking_auto_detect(self):
        """Should auto-detect format and remove thinking."""
        xml_text = "<think>reasoning</think>\nXML answer"
        xml_result = remove_thinking(xml_text)
        assert "XML answer" in xml_result
        assert "<think>" not in xml_result

        gpt_text = "<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>\n<|start|>assistant<|channel|>final<|message|>GPT answer<|return|>"
        gpt_result = remove_thinking(gpt_text)
        assert "<|channel|>final" in gpt_result
        assert "<|channel|>analysis" not in gpt_result

    def test_extract_tool_calls_auto_detect(self):
        """Should auto-detect format and extract tool calls."""
        xml_text = '<tool_call>\n{"name": "search", "parameters": {"query": "xml"}}\n</tool_call>'
        xml_tools = extract_tool_calls(xml_text)
        assert len(xml_tools) == 1
        assert xml_tools[0]["name"] == "search"

        gpt_text = '<|start|>assistant to=functions.search<|channel|>commentary json<|message|>{"query": "gpt"}<|call|>'
        gpt_tools = extract_tool_calls(gpt_text)
        assert len(gpt_tools) == 1
        assert gpt_tools[0]["name"] == "search"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
