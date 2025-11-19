#!/usr/bin/env python3
"""
Test GPT-OSS format support across all scorers.

Verifies that all domain scorers correctly handle GPT-OSS format:
- <|start|>assistant<|channel|>analysis<|message|>...<|end|>
- <|start|>assistant<|channel|>final<|message|>...<|return|>
"""

import pytest
from datatrove.utils.reward_score import default_compute_score


class TestMathScorerGPTOSS:
    """Test math.py scorer with GPT-OSS format."""

    def test_correct_answer_gpt_oss_format(self):
        """Test correct math answer in GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Let me solve this step by step. 2 + 2 = 4'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            '\\boxed{4}'
            '<|return|>'
        )
        ground_truth = '\\boxed{4}'
        result = default_compute_score("openai/gsm8k", model_output, ground_truth, format_type="gpt_oss")

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_wrong_answer_gpt_oss_format(self):
        """Test wrong math answer in GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Let me solve this. 2 + 2 = 5'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            '\\boxed{5}'
            '<|return|>'
        )
        ground_truth = '\\boxed{4}'
        result = default_compute_score("openai/gsm8k", model_output, ground_truth, format_type="gpt_oss")

        assert result["score"] == 0.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_missing_analysis_channel(self):
        """Test missing analysis channel in GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>final<|message|>'
            '\\boxed{4}'
            '<|return|>'
        )
        ground_truth = '\\boxed{4}'
        result = default_compute_score("openai/gsm8k", model_output, ground_truth, format_type="gpt_oss")

        # Missing analysis is OK (thinking is optional for math.py)
        assert result["reward_think"] == 1.0


class TestLogicScorerGPTOSS:
    """Test logic.py scorers with GPT-OSS format."""

    def test_ordering_puzzle_gpt_oss(self):
        """Test ordering puzzle with GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Let me analyze the order...'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            '["A", "B", "C"]'
            '<|return|>'
        )
        ground_truth = ["A", "B", "C"]
        result = default_compute_score("ordering_puzzle", model_output, ground_truth, format_type="gpt_oss")

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_zebra_puzzle_gpt_oss(self):
        """Test zebra puzzle with GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Solving the zebra puzzle...'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            '{"header": ["Color", "Pet"], "rows": [["Red", "Dog"], ["Blue", "Cat"]]}'
            '<|return|>'
        )
        ground_truth = {
            "header": ["Color", "Pet"],
            "rows": [["Red", "Dog"], ["Blue", "Cat"]]
        }
        result = default_compute_score("zebra_puzzle", model_output, ground_truth, format_type="gpt_oss")

        assert result["score"] == 1.0

    def test_graph_logical_gpt_oss(self):
        """Test graph logical with GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Analyzing the graph...'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            'Yes'
            '<|return|>'
        )
        ground_truth = 'Yes'
        result = default_compute_score("graph_logical", model_output, ground_truth, format_type="gpt_oss")

        assert result["score"] == 1.0

    def test_arcagi_gpt_oss(self):
        """Test ARC-AGI with GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Solving the pattern...'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            '[[0, 1], [1, 0]]'
            '<|return|>'
        )
        ground_truth = [[0, 1], [1, 0]]
        result = default_compute_score("arcagi1", model_output, ground_truth, format_type="gpt_oss")

        assert result["score"] == 1.0


class TestTableScorerGPTOSS:
    """Test table scorers with GPT-OSS format."""

    def test_table_boxed_gpt_oss(self):
        """Test table_boxed scorer with GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Let me analyze the table...'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            '\\boxed{42}'
            '<|return|>'
        )
        ground_truth = '42'
        result = default_compute_score("hitab", model_output, ground_truth, format_type="gpt_oss")

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_tqa_gpt_oss(self):
        """Test TQA scorer with GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Analyzing the table question...'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            '{"answer": ["Paris", "London"]}'
            '<|return|>'
        )
        ground_truth = ["Paris", "London"]
        result = default_compute_score("WTQ", model_output, ground_truth, format_type="gpt_oss")

        assert result["score"] == 1.0

    def test_tfv_gpt_oss(self):
        """Test TabFact verification with GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Verifying the fact...'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            '{"answer": "entailed"}'
            '<|return|>'
        )
        ground_truth = "entailed"
        result = default_compute_score("TabFact", model_output, ground_truth, format_type="gpt_oss")

        assert result["score"] >= 1.0

    def test_ff_tqa_gpt_oss(self):
        """Test free-form TQA with GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Answering the table question...'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            '{"answer": "Paris"}'
            '<|return|>'
        )
        ground_truth = "Paris"
        result = default_compute_score("FeTaQA", model_output, ground_truth, format_type="gpt_oss")

        assert result["score"] > 0.0  # BLEU/ROUGE scoring


class TestDocQAScorerGPTOSS:
    """Test DocQA-RL-VERL scorers with GPT-OSS format."""

    def test_long_scorer_gpt_oss(self):
        """Test long-context multiple choice with GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Analyzing the long document...'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            'The correct answer is (A)'
            '<|return|>'
        )
        ground_truth = 'A'
        result = default_compute_score("long_toc_choices", model_output, ground_truth, format_type="gpt_oss")

        assert result["score"] == 1.0
        assert result["accurate_score"] == 1.0
        assert result["format_score"] == 1.0

    def test_docmath_scorer_gpt_oss(self):
        """Test document math with GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Calculating from the document...'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            'the answer is 42'
            '<|return|>'
        )
        ground_truth = '42'
        result = default_compute_score("docmath", model_output, ground_truth, format_type="gpt_oss")

        assert result["score"] == 1.0
        assert result["accurate_score"] == 1.0
        assert result["format_score"] == 1.0

    def test_docqa_scorer_gpt_oss(self):
        """Test document QA with GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Reading the document...'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            'the answer is Paris'
            '<|return|>'
        )
        ground_truth = 'Paris'
        result = default_compute_score("multihoprag", model_output, ground_truth, format_type="gpt_oss")

        assert result["score"] == 1.0
        # Note: em is 0.0 because "the answer is Paris" != "Paris", but sub_em is 1.0
        assert result["sub_em"] == 1.0
        assert result["format_score"] == 1.0


class TestAutoDetection:
    """Test automatic format detection."""

    def test_auto_detect_gpt_oss(self):
        """Test auto-detection of GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Let me solve this. 2 + 2 = 4'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            '\\boxed{4}'
            '<|return|>'
        )
        ground_truth = '\\boxed{4}'
        # Don't specify format_type - should auto-detect
        result = default_compute_score("openai/gsm8k", model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0

    def test_auto_detect_xml(self):
        """Test auto-detection of XML format."""
        model_output = '<think>Let me solve this. 2 + 2 = 4</think>\\boxed{4}'
        ground_truth = '\\boxed{4}'
        # Don't specify format_type - should auto-detect
        result = default_compute_score("openai/gsm8k", model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0


class TestMixedFormatScenarios:
    """Test edge cases and mixed scenarios."""

    def test_gpt_oss_with_multiple_channels(self):
        """Test GPT-OSS with multiple analysis messages."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'First thought...'
            '<|end|>\n'
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Second thought...'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            '\\boxed{4}'
            '<|return|>'
        )
        ground_truth = '\\boxed{4}'
        result = default_compute_score("openai/gsm8k", model_output, ground_truth, format_type="gpt_oss")

        assert result["score"] == 1.0

    def test_gpt_oss_without_final_channel(self):
        """Test GPT-OSS format without final channel."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Let me think... The answer is 4'
            '<|end|>'
        )
        ground_truth = '4'
        result = default_compute_score("openai/gsm8k", model_output, ground_truth, format_type="gpt_oss")

        # Should still work - final channel is optional for some scorers
        # But might fail at extraction stage
        assert "score" in result


class TestDocQARequiresThinking:
    """Test that docmath, long, and docqa require thinking section."""

    def test_docmath_requires_thinking(self):
        """Test docmath requires thinking section."""
        model_output = 'the answer is 42'  # No thinking section
        ground_truth = '42'
        result = default_compute_score("docmath", model_output, ground_truth, format_type="xml")

        assert result["score"] == 0.0
        assert result["format_score"] == 0.0

    def test_long_requires_thinking(self):
        """Test long requires thinking section."""
        model_output = 'The correct answer is A'  # No thinking section
        ground_truth = 'A'
        result = default_compute_score("long_toc_choices", model_output, ground_truth, format_type="xml")

        assert result["score"] == 0.0
        assert result["format_score"] == 0.0

    def test_docqa_requires_thinking(self):
        """Test docqa requires thinking section."""
        model_output = 'the answer is Paris'  # No thinking section
        ground_truth = 'Paris'
        result = default_compute_score("multihoprag", model_output, ground_truth, format_type="xml")

        assert result["score"] == 0.0
        assert result["format_score"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
