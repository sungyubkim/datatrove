#!/usr/bin/env python3
"""
Test suite for docqa-rl-verl scorers (long, docmath, docqa).

Tests:
1. long.py - Long-context multiple choice QA (A-D)
2. docmath.py - Document math with numeric answers
3. docqa.py - Document QA with free text (EM/F1)
"""

import pytest
from datatrove.utils.reward_score import default_compute_score


class TestLongScorer:
    """Test long.py scorer for long-context multiple choice."""

    def test_correct_answer_with_parentheses(self):
        """Test correct answer with pattern: (A)"""
        predict_str = '<think>Analysis...</think>The correct answer is (B)'
        ground_truth = 'B'
        result = default_compute_score("long_toc_choices", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0
        assert result["format_score"] == 1.0
        assert result["score"] == 1.0

    def test_correct_answer_without_parentheses(self):
        """Test correct answer with pattern: A"""
        predict_str = '<think>Analysis...</think>The correct answer is C'
        ground_truth = 'C'
        result = default_compute_score("long_toc_choices", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0
        assert result["format_score"] == 1.0
        assert result["score"] == 1.0

    def test_wrong_answer(self):
        """Test wrong answer."""
        predict_str = '<think>Analysis...</think>The correct answer is (A)'
        ground_truth = 'B'
        result = default_compute_score("long_toc_choices", predict_str, ground_truth)

        assert result["accurate_score"] == 0.0

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        predict_str = '<think>Analysis...</think>the correct answer is (d)'
        ground_truth = 'D'
        result = default_compute_score("long_toc_choices", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0

    def test_ground_truth_dict(self):
        """Test ground truth in dict format."""
        predict_str = '<think>Analysis...</think>The correct answer is A'
        ground_truth = {'answer': 'A'}
        result = default_compute_score("long_toc_choices", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0

    def test_missing_think_tag(self):
        """Test missing </think> tag."""
        predict_str = 'The correct answer is (A)'
        ground_truth = 'A'
        result = default_compute_score("long_toc_choices", predict_str, ground_truth)

        assert result["score"] == 0.0
        assert result["format_score"] == 0.0

    def test_missing_answer_pattern(self):
        """Test missing answer pattern."""
        predict_str = '<think>Analysis...</think>The answer might be A or B'
        ground_truth = 'A'
        result = default_compute_score("long_toc_choices", predict_str, ground_truth)

        assert result["score"] == 0.0


class TestDocmathScorer:
    """Test docmath.py scorer for document math."""

    def test_exact_match(self):
        """Test exact numeric match."""
        predict_str = '<think>Calculation...</think>the answer is 42'
        ground_truth = '42'
        result = default_compute_score("docmath", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0
        assert result["format_score"] == 1.0
        assert result["score"] == 1.0

    def test_with_comma(self):
        """Test number with comma."""
        predict_str = '<think>Calculation...</think>the answer is 1,234'
        ground_truth = '1234'
        result = default_compute_score("docmath", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0

    def test_with_decimal(self):
        """Test decimal number."""
        predict_str = '<think>Calculation...</think>the answer is 3.14'
        ground_truth = '3.14'
        result = default_compute_score("docmath", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0

    def test_with_currency(self):
        """Test number with currency symbol."""
        predict_str = '<think>Calculation...</think>the answer is $100'
        ground_truth = '100'
        result = default_compute_score("docmath", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0

    def test_with_percentage(self):
        """Test number with percentage."""
        predict_str = '<think>Calculation...</think>the answer is 25%'
        ground_truth = '25'
        result = default_compute_score("docmath", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0

    def test_tolerance(self):
        """Test tolerance-based comparison."""
        predict_str = '<think>Calculation...</think>the answer is 42.001'
        ground_truth = '42.000'
        result = default_compute_score("docmath", predict_str, ground_truth)

        # Should be correct within tolerance
        assert result["accurate_score"] == 1.0

    def test_negative_number(self):
        """Test negative number."""
        predict_str = '<think>Calculation...</think>the answer is -15'
        ground_truth = '-15'
        result = default_compute_score("docmath", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0

    def test_missing_think_tag(self):
        """Test missing </think> tag."""
        predict_str = 'the answer is 42'
        ground_truth = '42'
        result = default_compute_score("docmath", predict_str, ground_truth)

        assert result["score"] == 0.0
        assert result["format_score"] == 0.0

    def test_missing_answer_pattern(self):
        """Test missing answer pattern."""
        predict_str = '<think>Calculation...</think>Result: 42'
        ground_truth = '42'
        result = default_compute_score("docmath", predict_str, ground_truth)

        assert result["score"] == 0.0

    def test_ground_truth_dict(self):
        """Test ground truth in dict format."""
        predict_str = '<think>Calculation...</think>the answer is 100'
        ground_truth = {'answer': '100'}
        result = default_compute_score("docmath", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0


class TestDocqaScorer:
    """Test docqa.py scorer for document QA."""

    def test_exact_match(self):
        """Test exact match."""
        predict_str = '<think>Analysis...</think>the answer is Paris'
        ground_truth = 'Paris'
        result = default_compute_score("multihoprag", predict_str, ground_truth)

        assert result["em"] == 1.0
        assert result["sub_em"] == 1.0
        assert result["score"] == 1.0
        assert result["format_score"] == 1.0

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        predict_str = '<think>Analysis...</think>the answer is PARIS'
        ground_truth = 'Paris'
        result = default_compute_score("multihoprag", predict_str, ground_truth)

        # After normalization (lowercasing), should match
        assert result["em"] == 1.0

    def test_punctuation_removal(self):
        """Test punctuation removal."""
        predict_str = '<think>Analysis...</think>the answer is Paris.'
        ground_truth = 'Paris'
        result = default_compute_score("multihoprag", predict_str, ground_truth)

        assert result["em"] == 1.0

    def test_article_removal(self):
        """Test article removal."""
        predict_str = '<think>Analysis...</think>the answer is the capital of France'
        ground_truth = 'capital of France'
        result = default_compute_score("multihoprag", predict_str, ground_truth)

        # "the" should be removed during normalization
        assert result["em"] == 1.0

    def test_substring_match(self):
        """Test substring exact match."""
        predict_str = '<think>Analysis...</think>the answer is Paris, France'
        ground_truth = 'Paris'
        result = default_compute_score("multihoprag", predict_str, ground_truth)

        # sub_em should be True (Paris is substring)
        assert result["sub_em"] == 1.0
        assert result["score"] == 1.0

    def test_f1_score(self):
        """Test F1 score computation."""
        predict_str = '<think>Analysis...</think>the answer is Paris is beautiful'
        ground_truth = 'Paris is great'
        result = default_compute_score("multihoprag", predict_str, ground_truth)

        # Should have partial overlap (Paris, is)
        assert result["f1"] > 0.0
        assert result["f1"] < 1.0
        assert result["precision"] > 0.0
        assert result["recall"] > 0.0

    def test_no_match(self):
        """Test completely different answers."""
        predict_str = '<think>Analysis...</think>the answer is London'
        ground_truth = 'Paris'
        result = default_compute_score("multihoprag", predict_str, ground_truth)

        assert result["em"] == 0.0
        assert result["sub_em"] == 0.0
        assert result["f1"] == 0.0

    def test_missing_think_tag(self):
        """Test missing </think> tag."""
        predict_str = 'the answer is Paris'
        ground_truth = 'Paris'
        result = default_compute_score("multihoprag", predict_str, ground_truth)

        assert result["score"] == 0.0
        assert result["format_score"] == 0.0

    def test_missing_answer_pattern(self):
        """Test missing answer pattern."""
        predict_str = '<think>Analysis...</think>Result: Paris'
        ground_truth = 'Paris'
        result = default_compute_score("multihoprag", predict_str, ground_truth)

        assert result["score"] == 0.0

    def test_ground_truth_dict(self):
        """Test ground truth in dict format."""
        predict_str = '<think>Analysis...</think>the answer is Paris'
        ground_truth = {'answer': 'Paris'}
        result = default_compute_score("multihoprag", predict_str, ground_truth)

        assert result["em"] == 1.0

    def test_musique_data_source(self):
        """Test musique data source routing."""
        predict_str = '<think>Analysis...</think>the answer is Paris'
        ground_truth = 'Paris'
        result = default_compute_score("musique", predict_str, ground_truth)

        assert result["em"] == 1.0


class TestIntegrationWithDefaultComputeScore:
    """Test integration with default_compute_score."""

    def test_long_routing(self):
        """Test long_toc_choices data source routing."""
        predict_str = '<think>Analysis...</think>The correct answer is A'
        ground_truth = 'A'
        result = default_compute_score("long_toc_choices", predict_str, ground_truth)

        assert "score" in result
        assert "accurate_score" in result

    def test_docmath_routing(self):
        """Test docmath data source routing."""
        predict_str = '<think>Calculation...</think>the answer is 42'
        ground_truth = '42'
        result = default_compute_score("docmath_test", predict_str, ground_truth)

        assert "score" in result
        assert "accurate_score" in result

    def test_multihoprag_routing(self):
        """Test multihoprag data source routing."""
        predict_str = '<think>Analysis...</think>the answer is Paris'
        ground_truth = 'Paris'
        result = default_compute_score("multihoprag_test", predict_str, ground_truth)

        assert "score" in result
        assert "em" in result
        assert "sub_em" in result
        assert "f1" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
