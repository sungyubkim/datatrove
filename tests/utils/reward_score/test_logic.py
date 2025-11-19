#!/usr/bin/env python3
"""
Test suite for logic domain scorers.

Tests:
1. ordering_puzzle - List ordering with sequence matching
2. zebra_puzzle - Structured grid validation
3. graph_logical - String matching for graph problems
4. arcagi1, arcagi2, barc - 2D array comparison

Verifies Qwen3 compatibility (no <answer> tags required).
"""

import pytest
from datatrove.utils.reward_score import default_compute_score


class TestOrderingPuzzleScorer:
    """Test ordering_puzzle scorer with list answers."""

    def test_correct_answer_with_tags(self):
        """Test correct answer with <answer> tags (traditional format)."""
        model_output = '<think>Let me analyze the order...</think><answer>["A", "B", "C"]</answer>'
        ground_truth = ["A", "B", "C"]
        result = default_compute_score("ordering_puzzle", model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_correct_answer_without_tags(self):
        """Test correct answer without <answer> tags (Qwen3 format)."""
        model_output = '<think>Let me analyze the order...</think>["A", "B", "C"]'
        ground_truth = ["A", "B", "C"]
        result = default_compute_score("ordering_puzzle", model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_wrong_order(self):
        """Test wrong ordering."""
        model_output = '<think>Analysis...</think>["C", "B", "A"]'
        ground_truth = ["A", "B", "C"]
        result = default_compute_score("ordering_puzzle", model_output, ground_truth)

        assert result["score"] == 0.0

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        model_output = '<think>Analysis...</think>["a", "b", "c"]'
        ground_truth = ["A", "B", "C"]
        result = default_compute_score("ordering_puzzle", model_output, ground_truth)

        assert result["score"] == 1.0

    def test_missing_think_tag(self):
        """Test missing </think> tag (thinking is optional)."""
        model_output = '["A", "B", "C"]'
        ground_truth = ["A", "B", "C"]
        result = default_compute_score("ordering_puzzle", model_output, ground_truth)

        # Thinking is optional - no <think> tags is OK
        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0


class TestZebraPuzzleScorer:
    """Test zebra_puzzle scorer with dict answers."""

    def test_correct_answer_with_tags(self):
        """Test correct answer with <answer> tags."""
        model_output = '''<think>Let me solve the zebra puzzle...</think><answer>{
            "header": ["Color", "Pet"],
            "rows": [["Red", "Dog"], ["Blue", "Cat"]]
        }</answer>'''
        ground_truth = {
            "header": ["Color", "Pet"],
            "rows": [["Red", "Dog"], ["Blue", "Cat"]]
        }
        result = default_compute_score("zebra_puzzle", model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_correct_answer_without_tags(self):
        """Test correct answer without <answer> tags (Qwen3 format)."""
        model_output = '''<think>Let me solve the zebra puzzle...</think>{
            "header": ["Color", "Pet"],
            "rows": [["Red", "Dog"], ["Blue", "Cat"]]
        }'''
        ground_truth = {
            "header": ["Color", "Pet"],
            "rows": [["Red", "Dog"], ["Blue", "Cat"]]
        }
        result = default_compute_score("zebra_puzzle", model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_partial_match(self):
        """Test partial cell match."""
        model_output = '''<think>Analysis...</think>{
            "header": ["Color", "Pet"],
            "rows": [["Red", "Cat"], ["Blue", "Cat"]]
        }'''
        ground_truth = {
            "header": ["Color", "Pet"],
            "rows": [["Red", "Dog"], ["Blue", "Cat"]]
        }
        result = default_compute_score("zebra_puzzle", model_output, ground_truth)

        # Should be 0.75 (3 out of 4 cells correct)
        assert result["score"] == 0.75

    def test_wrong_header(self):
        """Test wrong header."""
        model_output = '''<think>Analysis...</think>{
            "header": ["Wrong", "Header"],
            "rows": [["Red", "Dog"], ["Blue", "Cat"]]
        }'''
        ground_truth = {
            "header": ["Color", "Pet"],
            "rows": [["Red", "Dog"], ["Blue", "Cat"]]
        }
        result = default_compute_score("zebra_puzzle", model_output, ground_truth)

        assert result["score"] == 0.0


class TestGraphLogicalScorer:
    """Test graph_logical scorer with string answers."""

    def test_correct_answer_with_tags(self):
        """Test correct answer with <answer> tags."""
        model_output = '<think>Analyzing the graph...</think><answer>Yes</answer>'
        ground_truth = 'Yes'
        result = default_compute_score("graph_logical", model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_correct_answer_without_tags(self):
        """Test correct answer without <answer> tags (Qwen3 format)."""
        model_output = '<think>Analyzing the graph...</think>Yes'
        ground_truth = 'Yes'
        result = default_compute_score("graph_logical", model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        model_output = '<think>Analysis...</think>yes'
        ground_truth = 'Yes'
        result = default_compute_score("graph_logical", model_output, ground_truth)

        assert result["score"] == 1.0

    def test_wrong_answer(self):
        """Test wrong answer."""
        model_output = '<think>Analysis...</think>No'
        ground_truth = 'Yes'
        result = default_compute_score("graph_logical", model_output, ground_truth)

        assert result["score"] == 0.0


class TestARCAGIScorer:
    """Test ARC-AGI scorer with 2D array answers."""

    def test_correct_answer_with_tags(self):
        """Test correct answer with <answer> tags."""
        model_output = '<think>Solving the pattern...</think><answer>[[0, 1], [1, 0]]</answer>'
        ground_truth = [[0, 1], [1, 0]]
        result = default_compute_score("arcagi1", model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_correct_answer_without_tags(self):
        """Test correct answer without <answer> tags (Qwen3 format)."""
        model_output = '<think>Solving the pattern...</think>[[0, 1], [1, 0]]'
        ground_truth = [[0, 1], [1, 0]]
        result = default_compute_score("arcagi1", model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_partial_match(self):
        """Test partial pixel match."""
        model_output = '<think>Analysis...</think>[[0, 1], [0, 0]]'
        ground_truth = [[0, 1], [1, 0]]
        result = default_compute_score("arcagi1", model_output, ground_truth)

        # Should be 0.75 (3 out of 4 pixels correct)
        assert result["score"] == 0.75

    def test_size_mismatch_with_padding(self):
        """Test size mismatch with automatic padding."""
        model_output = '<think>Analysis...</think>[[0, 1]]'
        ground_truth = [[0, 1], [1, 0]]
        result = default_compute_score("arcagi2", model_output, ground_truth)

        # Padded: [[0,1], [0,0]] vs GT: [[0,1], [1,0]]
        # Comparison: (0,0)✓ (1,1)✓ (0,1)✗ (0,0)✓ = 3/4 = 0.75
        assert result["score"] == 0.75

    def test_barc_data_source(self):
        """Test BARC data source routing."""
        model_output = '<think>Analysis...</think>[[1, 2], [3, 4]]'
        ground_truth = [[1, 2], [3, 4]]
        result = default_compute_score("barc", model_output, ground_truth)

        assert result["score"] == 1.0


class TestCascadeRewards:
    """Test cascade reward system."""

    def test_cascade_failure_think(self):
        """Test cascade with malformed think tags."""
        model_output = '<think>incomplete thinking section["A", "B", "C"]'
        ground_truth = ["A", "B", "C"]
        result = default_compute_score("ordering_puzzle", model_output, ground_truth)

        # Cascade stops at reward_think (missing </think>)
        assert result["reward_think"] == 0.0
        assert result["reward_fmt"] == 0.0
        assert result["score"] == 0.0

    def test_cascade_failure_format(self):
        """Test cascade failure at format step."""
        model_output = '<think>Good thinking...</think>'  # No answer content
        ground_truth = ["A", "B", "C"]
        result = default_compute_score("ordering_puzzle", model_output, ground_truth)

        # reward_think passes, but reward_fmt fails
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 0.0
        assert result["score"] == 0.0

    def test_cascade_success_all_stages(self):
        """Test cascade success through all stages."""
        model_output = '<think>Good thinking...</think>["A", "B", "C"]'
        ground_truth = ["A", "B", "C"]
        result = default_compute_score("ordering_puzzle", model_output, ground_truth)

        # All stages pass
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0
        assert result["score"] == 1.0


class TestIntegrationWithDefaultComputeScore:
    """Test integration with default_compute_score routing."""

    def test_ordering_puzzle_routing(self):
        """Test ordering_puzzle data source routing."""
        model_output = '<think>Analysis...</think>["A", "B", "C"]'
        ground_truth = ["A", "B", "C"]
        result = default_compute_score("ordering_puzzle", model_output, ground_truth)

        assert "score" in result
        assert "reward_think" in result
        assert "reward_fmt" in result

    def test_zebra_puzzle_routing(self):
        """Test zebra_puzzle data source routing."""
        model_output = '''<think>Analysis...</think>{
            "header": ["A"],
            "rows": [["B"]]
        }'''
        ground_truth = {"header": ["A"], "rows": [["B"]]}
        result = default_compute_score("zebra_puzzle", model_output, ground_truth)

        assert "score" in result
        assert "reward_think" in result
        assert "reward_fmt" in result

    def test_graph_logical_routing(self):
        """Test graph_logical data source routing."""
        model_output = '<think>Analysis...</think>Yes'
        ground_truth = 'Yes'
        result = default_compute_score("graph_logical", model_output, ground_truth)

        assert "score" in result
        assert "reward_think" in result
        assert "reward_fmt" in result

    def test_arcagi_routing_variants(self):
        """Test all ARC-AGI variant routing."""
        model_output = '<think>Analysis...</think>[[0, 1]]'
        ground_truth = [[0, 1]]

        for data_source in ["arcagi1", "arcagi2", "barc"]:
            result = default_compute_score(data_source, model_output, ground_truth)
            assert "score" in result
            assert "reward_think" in result
            assert "reward_fmt" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
