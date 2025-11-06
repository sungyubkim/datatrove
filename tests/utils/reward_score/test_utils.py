import pytest

from datatrove.utils.reward_score.utils import (
    extract_answer_recursive,
    normalize_ground_truth,
    parse_answer,
    parse_think,
)


class TestParseThink:
    """Tests for parse_think function."""

    def test_single_think_tag_normal(self):
        """Test normal case with single think tag."""
        text = "<think>This is my thought process</think>The answer is 42"
        result, passed = parse_think(text)
        assert passed is True
        assert result == "The answer is 42"

    def test_single_think_tag_with_whitespace(self):
        """Test that whitespace after closing tag is stripped."""
        text = "<think>thoughts</think>   \n  answer"
        result, passed = parse_think(text)
        assert passed is True
        assert result == "answer"

    def test_no_think_tag(self):
        """Test case with no think tag returns original text."""
        text = "Just a plain answer"
        result, passed = parse_think(text)
        assert passed is True
        assert result == text

    def test_unbalanced_tags_missing_close(self):
        """Test unbalanced tags - missing closing tag."""
        text = "<think>This has no closing tag"
        result, passed = parse_think(text)
        assert passed is False
        assert result == text

    def test_unbalanced_tags_extra_open(self):
        """Test unbalanced tags - extra opening tag."""
        text = "<think>First<think>Second</think>"
        result, passed = parse_think(text)
        assert passed is False
        assert result == text

    def test_unbalanced_tags_extra_close(self):
        """Test unbalanced tags - extra closing tag."""
        text = "<think>thoughts</think>answer</think>"
        result, passed = parse_think(text)
        assert passed is False
        assert result == text

    def test_only_closing_tag(self):
        """Test with only closing tag (0 opening tags)."""
        text = "Some text</think>more text"
        result, passed = parse_think(text)
        assert passed is True
        assert result == text

    def test_multiple_balanced_tags(self):
        """Test multiple balanced think tags."""
        text = "<think>first</think>middle<think>second</think>end"
        result, passed = parse_think(text)
        assert passed is False
        assert result == text

    def test_empty_think_tag(self):
        """Test empty think tag."""
        text = "<think></think>answer"
        result, passed = parse_think(text)
        assert passed is True
        assert result == "answer"

    def test_multiline_think_content(self):
        """Test think tag with multiline content."""
        text = """<think>
        Step 1: Do something
        Step 2: Do something else
        </think>Final answer is here"""
        result, passed = parse_think(text)
        assert passed is True
        assert result == "Final answer is here"


class TestParseAnswer:
    """Tests for parse_answer function."""

    def test_simple_boxed_answer(self):
        """Test simple boxed answer."""
        text = r"The answer is \boxed{42}"
        result, idx = parse_answer(text)
        assert idx == 0
        assert result == "42"

    def test_boxed_with_expression(self):
        """Test boxed answer with mathematical expression."""
        text = r"Solution: \boxed{x + 5}"
        result, idx = parse_answer(text)
        assert idx == 0
        assert result == "x + 5"

    def test_nested_braces(self):
        """Test boxed answer with nested braces."""
        text = r"Answer: \boxed{{a, b}}"
        result, idx = parse_answer(text)
        assert idx == 0
        assert result == "{a, b}"

    def test_nested_function_braces(self):
        """Test boxed answer with nested function."""
        text = r"\boxed{f(x) = {y}}"
        result, idx = parse_answer(text)
        assert idx == 0
        assert result == "f(x) = {y}"

    def test_multiple_boxed_returns_last(self):
        """Test that multiple boxed answers return the last one."""
        text = r"First \boxed{wrong} and then \boxed{correct}"
        result, idx = parse_answer(text)
        assert idx == 0
        assert result == "correct"

    def test_no_boxed_returns_original(self):
        """Test text without boxed format returns original."""
        text = "Just plain text"
        result, idx = parse_answer(text)
        assert idx == -1
        assert result == text

    def test_incomplete_boxed(self):
        """Test incomplete boxed format."""
        text = r"\boxed{incomplete"
        result, idx = parse_answer(text)
        assert idx == -1
        assert result == text

    def test_non_string_input_int(self):
        """Test non-string input returns unchanged with -1."""
        result, idx = parse_answer(42)
        assert idx == -1
        assert result == 42

    def test_non_string_input_none(self):
        """Test None input."""
        result, idx = parse_answer(None)
        assert idx == -1
        assert result is None

    def test_non_string_input_list(self):
        """Test list input."""
        input_list = [1, 2, 3]
        result, idx = parse_answer(input_list)
        assert idx == -1
        assert result == input_list

    def test_boxed_with_brackets(self):
        """Test boxed with square brackets."""
        text = r"\boxed{[a, b, c]}"
        result, idx = parse_answer(text)
        assert idx == 0
        assert result == "[a, b, c]"

    def test_boxed_with_parentheses(self):
        """Test boxed with parentheses."""
        text = r"\boxed{(x + y)}"
        result, idx = parse_answer(text)
        assert idx == 0
        assert result == "(x + y)"

    def test_complex_latex_expression(self):
        """Test boxed with complex LaTeX expression."""
        text = r"\boxed{\frac{1}{2}}"
        result, idx = parse_answer(text)
        assert idx == 0
        assert result == r"\frac{1}{2}"


class TestExtractAnswerRecursive:
    """Tests for extract_answer_recursive function."""

    def test_simple_match(self):
        """Test simple pattern matching."""
        text = r"\boxed{simple}"
        result = extract_answer_recursive(text, r"\\boxed\{", "}")
        assert result == "simple"

    def test_nested_square_brackets(self):
        """Test nested square brackets."""
        text = r"\boxed{[a, b, c]}"
        result = extract_answer_recursive(text, r"\\boxed\{", "}")
        assert result == "[a, b, c]"

    def test_nested_parentheses(self):
        """Test nested parentheses."""
        text = r"\boxed{(x + y)}"
        result = extract_answer_recursive(text, r"\\boxed\{", "}")
        assert result == "(x + y)"

    def test_nested_braces(self):
        """Test nested braces."""
        text = r"\boxed{{nested}}"
        result = extract_answer_recursive(text, r"\\boxed\{", "}")
        assert result == "{nested}"

    def test_complex_nested_expression(self):
        """Test complex nested expression."""
        text = r"\boxed{\sqrt{x^2 + y^2}}"
        result = extract_answer_recursive(text, r"\\boxed\{", "}")
        assert result == r"\sqrt{x^2 + y^2}"

    def test_multiple_nesting_levels(self):
        """Test multiple nesting levels."""
        text = r"\boxed{{{a}}}"
        result = extract_answer_recursive(text, r"\\boxed\{", "}")
        assert result == "{{a}}"

    def test_no_pattern_match(self):
        """Test when start pattern is not found."""
        text = "No boxed here"
        result = extract_answer_recursive(text, r"\\boxed\{", "}")
        assert result is None

    def test_unbalanced_braces(self):
        """Test unbalanced braces."""
        text = r"\boxed{unbalanced{"
        result = extract_answer_recursive(text, r"\\boxed\{", "}")
        assert result is None

    def test_missing_end_pattern(self):
        """Test missing end pattern after closing brace."""
        text = r"\boxed{content]"
        result = extract_answer_recursive(text, r"\\boxed\{", "}")
        # This should fail because end_pattern expects "}"
        assert result is None

    def test_multiple_matches_returns_last(self):
        """Test multiple matches returns the last one."""
        text = r"\boxed{first} and \boxed{last}"
        result = extract_answer_recursive(text, r"\\boxed\{", "}")
        assert result == "last"

    def test_mixed_bracket_types(self):
        """Test mixed bracket types in content."""
        text = r"\boxed{array[i] + func(x) + {key: value}}"
        result = extract_answer_recursive(text, r"\\boxed\{", "}")
        assert result == "array[i] + func(x) + {key: value}"

    def test_empty_content(self):
        """Test empty boxed content."""
        text = r"\boxed{}"
        result = extract_answer_recursive(text, r"\\boxed\{", "}")
        assert result == ""

    def test_find_matching_brace_with_nested_structures(self):
        """Test complex nested structure with all bracket types."""
        text = r"\boxed{f(x) = [a, {b: c}]}"
        result = extract_answer_recursive(text, r"\\boxed\{", "}")
        assert result == "f(x) = [a, {b: c}]"


class TestNormalizeGroundTruth:
    """Tests for normalize_ground_truth function."""

    def test_dict_format(self):
        """Test dict format with 'answer' key."""
        ground_truth = {"answer": "Paris"}
        result = normalize_ground_truth(ground_truth)
        assert result == "Paris"

    def test_string_format(self):
        """Test plain string format."""
        ground_truth = "Paris"
        result = normalize_ground_truth(ground_truth)
        assert result == "Paris"

    def test_list_with_string(self):
        """Test list with single string element."""
        ground_truth = ["Paris"]
        result = normalize_ground_truth(ground_truth)
        assert result == "Paris"

    def test_list_with_dict(self):
        """Test list with dict containing 'answer' key."""
        ground_truth = [{"answer": "Paris"}]
        result = normalize_ground_truth(ground_truth)
        assert result == "Paris"

    def test_empty_list(self):
        """Test empty list returns empty string."""
        ground_truth = []
        result = normalize_ground_truth(ground_truth)
        assert result == ""

    def test_custom_key(self):
        """Test dict with custom key parameter."""
        ground_truth = {"solution": "42"}
        result = normalize_ground_truth(ground_truth, key="solution")
        assert result == "42"
