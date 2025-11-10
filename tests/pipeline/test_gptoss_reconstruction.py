"""
Unit tests for GPT OSS format reconstruction from vLLM reasoning parser output.

Tests the reconstruct_gptoss_from_vllm_response function which reconstructs
the original GPT OSS format from vLLM's parsed response structure.
"""

import pytest

from datatrove.pipeline.inference.run_inference import reconstruct_gptoss_from_vllm_response


class TestGPTOSSReconstruction:
    """Test suite for GPT OSS format reconstruction."""

    def test_no_parsing_standard_response(self):
        """Test that standard responses without reasoning parser are unchanged."""
        choice = {
            "message": {
                "role": "assistant",
                "content": "Hello, how can I help you?"
            }
        }
        result = reconstruct_gptoss_from_vllm_response(choice)
        assert result == "Hello, how can I help you?"

    def test_no_parsing_empty_content(self):
        """Test handling of empty content when parser is not active."""
        choice = {
            "message": {
                "role": "assistant",
                "content": ""
            }
        }
        result = reconstruct_gptoss_from_vllm_response(choice)
        assert result == ""

    def test_no_parsing_none_content(self):
        """Test handling of None content when parser is not active."""
        choice = {
            "message": {
                "role": "assistant",
                "content": None
            }
        }
        result = reconstruct_gptoss_from_vllm_response(choice)
        assert result == ""

    def test_analysis_and_final_channels(self):
        """Test reconstruction with both analysis and final channels (typical math problem)."""
        choice = {
            "message": {
                "role": "assistant",
                "content": "The answer is 42",
                "reasoning_content": "Let me think step by step. 6 times 7 equals 42."
            }
        }
        result = reconstruct_gptoss_from_vllm_response(choice)

        expected = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Let me think step by step. 6 times 7 equals 42.<|end|>\n"
            "<|start|>assistant<|channel|>final<|message|>"
            "The answer is 42<|return|>"
        )
        assert result == expected

    def test_analysis_only(self):
        """Test reconstruction with only analysis channel."""
        choice = {
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Thinking about the problem..."
            }
        }
        result = reconstruct_gptoss_from_vllm_response(choice)

        expected = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Thinking about the problem...<|end|>"
        )
        assert result == expected

    def test_single_tool_call(self):
        """Test reconstruction with a single tool call."""
        choice = {
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I need to search for information",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "GPT OSS format"}'
                        }
                    }
                ]
            }
        }
        result = reconstruct_gptoss_from_vllm_response(choice)

        expected = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "I need to search for information<|end|>\n"
            "<|start|>assistant to=functions.search"
            "<|channel|>commentary json<|message|>"
            '{"query": "GPT OSS format"}<|call|>'
        )
        assert result == expected

    def test_multiple_tool_calls(self):
        """Test reconstruction with multiple tool calls."""
        choice = {
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I need multiple tools",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "AI"}'
                        }
                    },
                    {
                        "function": {
                            "name": "calculate",
                            "arguments": '{"expression": "2+2"}'
                        }
                    }
                ]
            }
        }
        result = reconstruct_gptoss_from_vllm_response(choice)

        # Should have analysis + both tool calls
        assert "<|channel|>analysis<|message|>I need multiple tools<|end|>" in result
        assert "to=functions.search" in result
        assert '{"query": "AI"}' in result
        assert "to=functions.calculate" in result
        assert '{"expression": "2+2"}' in result

    def test_tool_call_without_analysis(self):
        """Test reconstruction with tool call but no analysis channel."""
        choice = {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo"}'
                        }
                    }
                ]
            }
        }
        result = reconstruct_gptoss_from_vllm_response(choice)

        expected = (
            "<|start|>assistant to=functions.get_weather"
            "<|channel|>commentary json<|message|>"
            '{"location": "Tokyo"}<|call|>'
        )
        assert result == expected

    def test_tool_call_suppresses_final_channel(self):
        """Test that final channel is not included when tool calls are present."""
        choice = {
            "message": {
                "role": "assistant",
                "content": "This should be ignored",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": "{}"
                        }
                    }
                ]
            }
        }
        result = reconstruct_gptoss_from_vllm_response(choice)

        # Should have tool call but NOT final channel
        assert "to=functions.search" in result
        assert "<|channel|>final" not in result
        assert "This should be ignored" not in result

    def test_malformed_tool_call_missing_function(self):
        """Test handling of malformed tool call with missing function."""
        choice = {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        # Missing "function" key
                    }
                ]
            }
        }
        result = reconstruct_gptoss_from_vllm_response(choice)

        # Should use "unknown" as default tool name and "{}" as default args
        assert "to=functions.unknown" in result
        assert "{}" in result

    def test_malformed_tool_call_missing_name(self):
        """Test handling of malformed tool call with missing name."""
        choice = {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "arguments": '{"key": "value"}'
                        }
                    }
                ]
            }
        }
        result = reconstruct_gptoss_from_vllm_response(choice)

        assert "to=functions.unknown" in result
        assert '{"key": "value"}' in result

    def test_malformed_tool_call_missing_arguments(self):
        """Test handling of malformed tool call with missing arguments."""
        choice = {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "my_function"
                        }
                    }
                ]
            }
        }
        result = reconstruct_gptoss_from_vllm_response(choice)

        assert "to=functions.my_function" in result
        assert "{}" in result

    def test_empty_message_dict(self):
        """Test handling of empty message dict."""
        choice = {
            "message": {}
        }
        result = reconstruct_gptoss_from_vllm_response(choice)
        assert result == ""

    def test_missing_message_key(self):
        """Test handling of missing message key."""
        choice = {}
        result = reconstruct_gptoss_from_vllm_response(choice)
        assert result == ""

    def test_complex_realistic_math_problem(self):
        """Test realistic math problem with detailed reasoning."""
        choice = {
            "message": {
                "role": "assistant",
                "content": "Therefore, the answer is \\boxed{42}",
                "reasoning_content": (
                    "Let me solve this step by step.\n"
                    "First, I'll identify what we're looking for.\n"
                    "Then, I'll apply the formula: 6 × 7 = 42."
                )
            }
        }
        result = reconstruct_gptoss_from_vllm_response(choice)

        assert "<|start|>assistant<|channel|>analysis<|message|>" in result
        assert "Let me solve this step by step." in result
        assert "<|end|>" in result
        assert "<|start|>assistant<|channel|>final<|message|>" in result
        assert "\\boxed{42}" in result
        assert "<|return|>" in result

    def test_complex_realistic_tool_call(self):
        """Test realistic tool call scenario with analysis."""
        choice = {
            "message": {
                "role": "assistant",
                "reasoning_content": "To answer this question, I need to search the database for user information.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "database_query",
                            "arguments": '{"table": "users", "filter": {"age": ">18"}, "limit": 10}'
                        }
                    }
                ]
            }
        }
        result = reconstruct_gptoss_from_vllm_response(choice)

        assert "I need to search the database" in result
        assert "to=functions.database_query" in result
        assert '"table": "users"' in result
        assert "<|call|>" in result
