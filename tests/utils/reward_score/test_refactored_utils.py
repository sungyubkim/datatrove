#!/usr/bin/env python3
"""
Backward compatibility tests for refactored utility functions.

Tests include:
1. New utility functions work correctly
2. Existing scorers (math, logic) still produce same outputs
3. Both XML and GPT-OSS formats are supported
"""

import pytest
import sys
from pathlib import Path

# Add datatrove to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from datatrove.utils.reward_score.utils import (
    parse_think,
    parse_answer,
    extract_content_from_tags,
    parse_json_with_fallback,
    normalize_text,
    normalize_numeric,
    compare_sets_unordered
)


class TestExtractContentFromTags:
    """Test the new extract_content_from_tags utility function."""

    def test_single_answer_tag(self):
        """Should extract content from single <answer> tag."""
        text = "<answer>42</answer>"
        content, success = extract_content_from_tags(text, "answer")
        assert success is True
        assert content == "42"

    def test_multiple_answer_tags(self):
        """Should extract content from the last <answer> tag."""
        text = "<answer>first</answer>\n<answer>second</answer>"
        content, success = extract_content_from_tags(text, "answer")
        assert success is True
        assert content == "second"

    def test_no_tags(self):
        """Should return empty string and False when no tags found."""
        text = "No tags here"
        content, success = extract_content_from_tags(text, "answer")
        assert success is False
        assert content == ""

    def test_custom_tag_name(self):
        """Should work with custom tag names."""
        text = "<solution>my solution</solution>"
        content, success = extract_content_from_tags(text, "solution")
        assert success is True
        assert content == "my solution"

    def test_multiline_content(self):
        """Should handle multiline content."""
        text = """<answer>
Line 1
Line 2
Line 3
</answer>"""
        content, success = extract_content_from_tags(text, "answer")
        assert success is True
        assert "Line 1" in content
        assert "Line 3" in content


class TestParseJsonWithFallback:
    """Test the new parse_json_with_fallback utility function."""

    def test_parse_list(self):
        """Should parse list correctly."""
        result = parse_json_with_fallback('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_parse_dict(self):
        """Should parse dict correctly."""
        result = parse_json_with_fallback('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_python_literal(self):
        """Should parse Python literal with single quotes."""
        result = parse_json_with_fallback("['a', 'b', 'c']")
        assert result == ["a", "b", "c"]

    def test_parse_string_type(self):
        """Should return plain text for str type."""
        result = parse_json_with_fallback('plain text', expected_type='str')
        assert result == 'plain text'

    def test_empty_content(self):
        """Should return None for empty content."""
        result = parse_json_with_fallback('')
        assert result is None

    def test_invalid_json(self):
        """Should return None for invalid JSON."""
        result = parse_json_with_fallback('{invalid json}')
        assert result is None


class TestNormalizeText:
    """Test the new normalize_text utility function."""

    def test_lowercase_and_strip(self):
        """Should lowercase and strip by default."""
        result = normalize_text("  Hello World  ")
        assert result == "hello world"

    def test_no_lowercase(self):
        """Should preserve case when lowercase=False."""
        result = normalize_text("  Hello World  ", lowercase=False)
        assert result == "Hello World"

    def test_no_strip(self):
        """Should preserve whitespace when strip=False."""
        result = normalize_text("  hello world  ", strip=False)
        assert result == "  hello world  "

    def test_non_string_input(self):
        """Should convert non-string to string."""
        result = normalize_text(42)
        assert result == "42"


class TestNormalizeNumeric:
    """Test the new normalize_numeric utility function."""

    def test_string_number(self):
        """Should parse string numbers."""
        result = normalize_numeric("42.12345", precision=2)
        assert result == 42.12

    def test_integer_input(self):
        """Should handle integer input."""
        result = normalize_numeric(42)
        assert result == 42.0

    def test_float_input(self):
        """Should handle float input."""
        result = normalize_numeric(42.5)
        assert result == 42.5

    def test_remove_comma(self):
        """Should remove commas from numbers."""
        result = normalize_numeric("1,234.56", precision=2)
        assert result == 1234.56

    def test_remove_percent(self):
        """Should remove percent sign."""
        result = normalize_numeric("25%", precision=2)
        assert result == 25.0

    def test_invalid_input(self):
        """Should return None for invalid input."""
        result = normalize_numeric("not a number")
        assert result is None


class TestCompareSetsUnordered:
    """Test the new compare_sets_unordered utility function."""

    def test_exact_match_different_order(self):
        """Should return 1.0 for exact match regardless of order."""
        result = compare_sets_unordered(['a', 'b', 'c'], ['c', 'b', 'a'])
        assert result == 1.0

    def test_partial_overlap(self):
        """Should compute Jaccard similarity for partial overlap."""
        # Intersection: {'b'} (1 item)
        # Union: {'a', 'b', 'c', 'd'} (4 items)
        # Jaccard: 1/4 = 0.25
        result = compare_sets_unordered(['a', 'b'], ['b', 'c', 'd'])
        assert result == 0.25

    def test_no_overlap(self):
        """Should return 0.0 for no overlap."""
        result = compare_sets_unordered(['a', 'b'], ['c', 'd'])
        assert result == 0.0

    def test_empty_prediction(self):
        """Should return 0.0 for empty prediction."""
        result = compare_sets_unordered([], ['a', 'b'])
        assert result == 0.0

    def test_empty_ground_truth(self):
        """Should return 1.0 for empty prediction and empty ground truth."""
        result = compare_sets_unordered([], [])
        assert result == 1.0

    def test_case_insensitive(self):
        """Should be case-insensitive by default."""
        result = compare_sets_unordered(['A', 'B'], ['a', 'b'])
        assert result == 1.0


class TestBackwardCompatibility:
    """Test that existing utilities still work correctly."""

    def test_parse_think_xml_format(self):
        """Existing parse_think should work with XML format."""
        text = "<think>reasoning here</think>\nThe answer is 42"
        result, success = parse_think(text, format_type="xml")
        assert success is True
        assert "<think>" not in result
        assert "The answer is 42" in result

    def test_parse_think_auto_detect(self):
        """Auto-detection should still work."""
        text = "<think>reasoning</think>\nAnswer"
        result, success = parse_think(text, format_type="auto")
        assert success is True

    def test_parse_answer_boxed(self):
        """Existing parse_answer should extract \\boxed{} correctly."""
        text = "The answer is \\boxed{42}"
        result, format_idx = parse_answer(text, format_type="xml")
        assert format_idx == 0  # \\boxed{} is format index 0
        assert result == "42"

    def test_parse_answer_nested_boxed(self):
        """Should handle nested brackets in \\boxed{}."""
        text = "The answer is \\boxed{[1, 2, 3]}"
        result, format_idx = parse_answer(text, format_type="xml")
        assert format_idx == 0
        assert result == "[1, 2, 3]"


class TestIntegrationWithExistingScorers:
    """Integration tests to ensure refactored code doesn't break existing scorers."""

    def test_math_scorer_still_works(self):
        """Math scorer should still work correctly."""
        from datatrove.utils.reward_score import math

        # Test correct answer
        model_output = "<think>Let me calculate</think>\nThe answer is \\boxed{42}"
        ground_truth = "\\boxed{42}"

        result = math.compute_score(
            model_output=model_output,
            ground_truth=ground_truth,
            format_type="xml"
        )

        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0
        assert result["score"] == 1.0

    def test_logic_scorer_still_works(self):
        """Logic scorer should still work correctly."""
        from datatrove.utils.reward_score import logic

        # Test ordering puzzle
        model_output = "<think>Solving...</think>\n<answer>['a', 'b', 'c']</answer>"
        ground_truth = ['a', 'b', 'c']

        result = logic.compute_score(
            model_output=model_output,
            ground_truth=ground_truth,
            data_source="ordering_puzzle",
            format_type="xml"
        )

        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0
        assert result["score"] == 1.0


def main():
    """Run all tests."""
    print("=" * 80)
    print("RUNNING BACKWARD COMPATIBILITY TESTS")
    print("=" * 80)

    # Run pytest
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v"],
        cwd=Path(__file__).parent.parent.parent.parent
    )

    return result.returncode


if __name__ == "__main__":
    exit(main())
