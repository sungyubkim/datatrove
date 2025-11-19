"""
Unit tests for the _check_multiline_occurrence helper method in MathDatasetCleaner.

This test suite validates the multiline pattern detection logic that reduces false
positives in multi-part problem filtering.
"""

import re

import pytest

from datatrove.pipeline.formatters.math_cleaner import MathDatasetCleaner


class TestMultilineOccurrence:
    """Test the _check_multiline_occurrence helper method"""

    @pytest.fixture
    def cleaner(self):
        """Create a basic cleaner instance"""
        return MathDatasetCleaner.from_preset("orz-math")

    def test_both_patterns_on_same_line_returns_false(self, cleaner):
        """Should NOT filter: patterns on same line"""
        pattern1 = re.compile(r'\$\(1\)\$')
        pattern2 = re.compile(r'\$\(2\)\$')
        text = "From equations $(1)$ and $(2)$, we derive..."

        result = cleaner._check_multiline_occurrence(text, pattern1, pattern2)
        assert result is False, "Same-line patterns should not be filtered"

    def test_patterns_on_different_lines_returns_true(self, cleaner):
        """Should filter: patterns across multiple lines"""
        pattern1 = re.compile(r'\$\(1\)\$')
        pattern2 = re.compile(r'\$\(2\)\$')
        text = """
Problem $(1)$ Calculate the area.

Problem $(2)$ Find the perimeter.
"""
        result = cleaner._check_multiline_occurrence(text, pattern1, pattern2)
        assert result is True, "Multi-line patterns should be filtered"

    def test_only_first_pattern_found_returns_false(self, cleaner):
        """Should NOT filter: only one pattern present"""
        pattern1 = re.compile(r'\(i\)\s')
        pattern2 = re.compile(r'\(ii\)')
        text = "In case (i) we have a solution."

        result = cleaner._check_multiline_occurrence(text, pattern1, pattern2)
        assert result is False, "Missing second pattern should not filter"

    def test_only_second_pattern_found_returns_false(self, cleaner):
        """Should NOT filter: only one pattern present"""
        pattern1 = re.compile(r'\(i\)\s')
        pattern2 = re.compile(r'\(ii\)')
        text = "The result (ii) shows convergence."

        result = cleaner._check_multiline_occurrence(text, pattern1, pattern2)
        assert result is False, "Missing first pattern should not filter"

    def test_neither_pattern_found_returns_false(self, cleaner):
        """Should NOT filter: no patterns present"""
        pattern1 = re.compile(r'\$\(I\)\$')
        pattern2 = re.compile(r'\$\(II\)\$')
        text = "A simple math problem with no parts."

        result = cleaner._check_multiline_occurrence(text, pattern1, pattern2)
        assert result is False, "No patterns should not filter"

    def test_patterns_reversed_order_still_works(self, cleaner):
        """Should filter: pattern2 before pattern1"""
        pattern1 = re.compile(r'\(i\)\s')
        pattern2 = re.compile(r'\(ii\)')
        text = """
Part (ii) Find the derivative.

Part (i) Calculate the integral.
"""
        result = cleaner._check_multiline_occurrence(text, pattern1, pattern2)
        assert result is True, "Reversed order with newline should filter"

    def test_multiple_newlines_between_patterns(self, cleaner):
        """Should filter: many lines between patterns"""
        pattern1 = re.compile(r'\$\(1\)\$')
        pattern2 = re.compile(r'\$\(2\)\$')
        text = """
Problem $(1)$ What is the first answer?

Here is some explanation text.
More explanation here.

Problem $(2)$ What is the second answer?
"""
        result = cleaner._check_multiline_occurrence(text, pattern1, pattern2)
        assert result is True, "Multiple newlines should still filter"

    def test_same_line_with_long_text_between(self, cleaner):
        """Should NOT filter: same line even with lots of text between"""
        pattern1 = re.compile(r'\(i\)\s')
        pattern2 = re.compile(r'\(ii\)')
        text = "In cases (i) where x > 0 and also in cases (ii) where x < 0, we can prove convergence."

        result = cleaner._check_multiline_occurrence(text, pattern1, pattern2)
        assert result is False, "Same line with long text should not filter"

    def test_newline_immediately_between_patterns(self, cleaner):
        """Should filter: patterns separated by just a newline"""
        pattern1 = re.compile(r'\$\(I\)\$')
        pattern2 = re.compile(r'\$\(II\)\$')
        text = "Problem $(I)$\nProblem $(II)$"

        result = cleaner._check_multiline_occurrence(text, pattern1, pattern2)
        assert result is True, "Single newline between patterns should filter"
