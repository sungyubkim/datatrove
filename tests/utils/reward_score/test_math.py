import pytest

from datatrove.utils.reward_score.math import compute_score


class TestComputeScore:
    """Tests for compute_score function."""

    def test_perfect_answer(self):
        """Test completely correct answer with all components."""
        model_output = "<think>Let me solve: 40 + 2 = 42</think>The answer is \\boxed{42}"
        ground_truth = "\\boxed{42}"
        result = compute_score(model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0

    def test_missing_think_tag(self):
        """Test answer without think tag - think tag is optional."""
        model_output = "The answer is \\boxed{42}"
        ground_truth = "\\boxed{42}"
        result = compute_score(model_output, ground_truth)

        # Without think tag, parse_think returns (text, True), so it passes
        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0

    def test_unbalanced_think_tag(self):
        """Test answer with unbalanced think tag."""
        model_output = "<think>Let me solve this problem...\\boxed{42}"
        ground_truth = "\\boxed{42}"
        result = compute_score(model_output, ground_truth)

        assert result["score"] == 0.0
        assert result["reward_think"] == 0.0
        assert result["reward_format"] == 0.0

    def test_format_mismatch_prediction_missing_boxed(self):
        """Test format mismatch - prediction missing boxed format."""
        model_output = "<think>Calculating...</think>The answer is 42"
        ground_truth = "\\boxed{42}"
        result = compute_score(model_output, ground_truth)

        assert result["score"] == 0.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 0.0

    def test_format_mismatch_ground_truth_missing_boxed(self):
        """Test format mismatch - ground truth missing boxed format."""
        model_output = "<think>Calculating...</think>The answer is \\boxed{42}"
        ground_truth = "42"
        result = compute_score(model_output, ground_truth)

        assert result["score"] == 0.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 0.0

    def test_wrong_answer(self):
        """Test correct format but wrong answer."""
        model_output = "<think>Calculating...</think>The answer is \\boxed{43}"
        ground_truth = "\\boxed{42}"
        result = compute_score(model_output, ground_truth)

        assert result["score"] == 0.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0

    def test_mathematically_equivalent_fraction_decimal(self):
        """Test mathematically equivalent answers: fraction vs decimal."""
        model_output = "<think>Half is...</think>\\boxed{0.5}"
        ground_truth = "\\boxed{1/2}"
        result = compute_score(model_output, ground_truth)

        # With strict=False, should recognize equivalence
        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0

    def test_float_rounding(self):
        """Test float rounding with float_rounding=2."""
        model_output = "<think>Dividing...</think>\\boxed{2.333}"
        ground_truth = "\\boxed{2.33}"
        result = compute_score(model_output, ground_truth)

        # With float_rounding=2, should match
        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0

    def test_complex_latex_expression(self):
        """Test complex LaTeX expression."""
        model_output = "<think>Simplifying...</think>\\boxed{\\frac{x^2 + 2x + 1}{x + 1}}"
        ground_truth = "\\boxed{x + 1}"
        result = compute_score(model_output, ground_truth)

        # Math verify should handle algebraic simplification
        # This might be True or False depending on math_verify's capabilities
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0
        # Score depends on whether math_verify can verify algebraic equivalence

    def test_multiple_boxed_in_output(self):
        """Test with multiple boxed expressions in output."""
        model_output = "<think>First I get \\boxed{10}, then \\boxed{42}</think>\\boxed{42}"
        ground_truth = "\\boxed{42}"
        result = compute_score(model_output, ground_truth)

        # Should use last boxed in both prediction and ground truth
        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0

    def test_negative_numbers(self):
        """Test with negative numbers."""
        model_output = "<think>Subtracting...</think>\\boxed{-5}"
        ground_truth = "\\boxed{-5}"
        result = compute_score(model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0

    def test_algebraic_expression(self):
        """Test with algebraic expression."""
        model_output = "<think>Solving for x...</think>\\boxed{x = 3}"
        ground_truth = "\\boxed{x = 3}"
        result = compute_score(model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0

    def test_vector_or_set_notation(self):
        """Test with vector or set notation."""
        model_output = "<think>The solution set is...</think>\\boxed{{1, 2, 3}}"
        ground_truth = "\\boxed{{1, 2, 3}}"
        result = compute_score(model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0

    def test_timeout_score_parameter(self):
        """Test that timeout_score parameter is accepted (interface consistency)."""
        model_output = "<think>Solving...</think>\\boxed{42}"
        ground_truth = "\\boxed{42}"
        result = compute_score(model_output, ground_truth, timeout_score=0.5)

        # timeout_score is not currently used in the implementation
        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0

    def test_empty_think_tag(self):
        """Test with empty think tag."""
        model_output = "<think></think>\\boxed{42}"
        ground_truth = "\\boxed{42}"
        result = compute_score(model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0

    def test_multiline_think_content(self):
        """Test with multiline think content."""
        model_output = """<think>
        Step 1: Understand the problem
        Step 2: Calculate 40 + 2
        Step 3: Get 42
        </think>
        The final answer is \\boxed{42}"""
        ground_truth = "\\boxed{42}"
        result = compute_score(model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0

    def test_string_answer(self):
        """Test with string answer (should work if both match)."""
        model_output = "<think>The country is...</think>\\boxed{France}"
        ground_truth = "\\boxed{France}"
        result = compute_score(model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0

    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        model_output = "<think>Solving...</think>  \\boxed{42}  "
        ground_truth = "\\boxed{42}"
        result = compute_score(model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0

    def test_case_sensitivity(self):
        """Test case sensitivity in answers - math_verify is case-insensitive."""
        model_output = "<think>Answering...</think>\\boxed{ABC}"
        ground_truth = "\\boxed{abc}"
        result = compute_score(model_output, ground_truth)

        # math_verify treats these as equivalent (case-insensitive)
        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_format"] == 1.0
