"""Tests for MathDatasetCleaner formatter using real dataset samples."""

import pytest

from datatrove.data import Document
from datatrove.pipeline.formatters import MathDatasetCleaner


class TestMathDatasetCleanerORZ:
    """Tests using actual problematic samples from ORZ-Math dataset."""

    def test_orz_problem_number_8_3(self):
        """ORZ Sample 10: '8.3 In the tetrahedron...'"""
        doc = Document(
            id="test-orz-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "8.3 In the tetrahedron $ABCD$, edge $AB = 1$. Find the maximum value.",
                    }
                ],
                "ground_truth": "\\frac{1}{2}",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "In the tetrahedron $ABCD$, edge $AB = 1$. Find the maximum value."
        assert result.metadata["ground_truth"] == "\\frac{1}{2}"  # Ground truth unchanged

    def test_orz_problem_number_g1_4(self):
        """ORZ Sample 16: 'G1.4 When 491 is divided by...'"""
        doc = Document(
            id="test-orz-2",
            text="",
            metadata={
                "prompt": [
                    {"role": "user", "content": "G1.4 When 491 is divided by a certain number, the remainder is 15."}
                ],
                "ground_truth": "7",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "When 491 is divided by a certain number, the remainder is 15."

    def test_orz_contest_metadata_eotvos(self):
        """ORZ Sample 21: '24th Eötvös 1917 Problem 2 A square...'"""
        doc = Document(
            id="test-orz-3",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "24th Eötvös 1917 Problem 2 A square is divided into $n^2$ smaller squares.",
                    }
                ],
                "ground_truth": "n+1",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "A square is divided into $n^2$ smaller squares."

    def test_orz_contest_metadata_apmc(self):
        """ORZ Sample 31: '20th APMC 1997 Problem 3 The 97 numbers...'"""
        doc = Document(
            id="test-orz-4",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "20th APMC 1997 Problem 3 The 97 numbers from 1 to 97 are written on a blackboard.",
                    }
                ],
                "ground_truth": "1",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "The 97 numbers from 1 to 97 are written on a blackboard."

    def test_orz_contest_metadata_parentheses(self):
        """ORZ: '(2004 College Entrance Examination...)'"""
        doc = Document(
            id="test-orz-5",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "(2004 College Entrance Examination, Guangdong Province) Find the value of $x$.",
                    }
                ],
                "ground_truth": "5",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Find the value of $x$."

    def test_orz_point_allocation(self):
        """ORZ: '(8 points)' in text"""
        doc = Document(
            id="test-orz-6",
            text="",
            metadata={
                "prompt": [{"role": "user", "content": "Calculate the following (8 points) $\\frac{1}{2} + \\frac{1}{3}$"}],
                "ground_truth": "\\frac{5}{6}",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Calculate the following $\\frac{1}{2} + \\frac{1}{3}$"

    def test_orz_vague_ground_truth_preserved(self):
        """ORZ: Vague ground truth 'B' should be preserved (19% of dataset)"""
        doc = Document(
            id="test-orz-7",
            text="",
            metadata={
                "prompt": [{"role": "user", "content": "What is the answer?"}],
                "ground_truth": "B",  # MCQ answer - should be preserved
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # Ground truth should be unchanged (preserved per user requirement)
        assert result.metadata["ground_truth"] == "B"


class TestMathDatasetCleanerOpenR1:
    """Tests using actual problematic samples from OpenR1-Math dataset."""

    def test_openr1_problem_with_points(self):
        """OpenR1 Sample 8: 'Problem 6. (8 points) In the plane...'"""
        doc = Document(
            id="test-openr1-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Problem 6. (8 points) In the plane, there is a non-closed, non-self-intersecting broken line.",
                    }
                ],
                "ground_truth": "9",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "In the plane, there is a non-closed, non-self-intersecting broken line."

    def test_openr1_markdown_header(self):
        """OpenR1 Sample 15: '## Problem Statement...'"""
        doc = Document(
            id="test-openr1-2",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "## Problem Statement\n\nCalculate the limit of the numerical sequence: $\\lim_{n \\rightarrow \\infty} \\frac{(n+1)^{4}-(n-1)^{4}}{(n+1)^{3}+(n-1)^{3}}$",
                    }
                ],
                "ground_truth": "4",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        expected = "Calculate the limit of the numerical sequence: $\\lim_{n \\rightarrow \\infty} \\frac{(n+1)^{4}-(n-1)^{4}}{(n+1)^{3}+(n-1)^{3}}$"
        assert result.metadata["prompt"][0]["content"] == expected

    def test_openr1_zadatak_header(self):
        """OpenR1: '## Zadatak B-1.2.'"""
        doc = Document(
            id="test-openr1-3",
            text="",
            metadata={
                "prompt": [{"role": "user", "content": "## Zadatak B-1.2.\n\nSolve for $x$: $2x + 3 = 7$"}],
                "ground_truth": "2",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Solve for $x$: $2x + 3 = 7$"

    def test_openr1_contest_metadata(self):
        """OpenR1: Contest metadata removal"""
        doc = Document(
            id="test-openr1-4",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "(1968 Bulgarian Competition Problem) Find all real solutions to the equation.",
                    }
                ],
                "ground_truth": "x=0",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Find all real solutions to the equation."

    def test_openr1_parenthesized_number(self):
        """OpenR1 NEW: '(2) Find the value of...'"""
        doc = Document(
            id="test-openr1-5",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "(2) Find the value of $x$ when $2x + 3 = 7$.",
                    }
                ],
                "ground_truth": "2",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Find the value of $x$ when $2x + 3 = 7$."

    def test_openr1_single_digit_period(self):
        """OpenR1 NEW: '1. Calculate the following...'"""
        doc = Document(
            id="test-openr1-6",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "1. Calculate the following: $\\frac{3}{4} + \\frac{1}{2}$",
                    }
                ],
                "ground_truth": "\\frac{5}{4}",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Calculate the following: $\\frac{3}{4} + \\frac{1}{2}$"

    def test_openr1_letter_number_prefix(self):
        """OpenR1 NEW: 'B1. Determine whether...'"""
        doc = Document(
            id="test-openr1-7",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "B1. Determine whether the following statement is true.",
                    }
                ],
                "ground_truth": "True",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Determine whether the following statement is true."

    def test_openr1_roman_numeral(self):
        """OpenR1 NEW: 'II. Find all solutions...'"""
        doc = Document(
            id="test-openr1-8",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "II. Find all solutions to the equation $x^2 - 5x + 6 = 0$.",
                    }
                ],
                "ground_truth": "x=2,3",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Find all solutions to the equation $x^2 - 5x + 6 = 0$."

    def test_multiple_choice_options_preserved(self):
        """Regression test: Multiple-choice options C. and D. should NOT be removed.

        These are NOT Roman numerals in this context - they're multiple-choice labels.
        The Roman numeral pattern should only match multi-letter sequences (II., III., IV.)
        to avoid removing single-letter options like C. (100), D. (500), M. (1000), I. (1).
        """
        doc = Document(
            id="test-mcq-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            "Which ancient civilization made the greatest contributions to mathematics?\n"
                            "A. Indians\n"
                            "B. China\n"
                            "C. Babylon\n"
                            "D. Arabs"
                        ),
                    }
                ],
                "ground_truth": "A",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]
        content = result.metadata["prompt"][0]["content"]

        # All options should be preserved with their labels
        assert "A. Indians" in content
        assert "B. China" in content
        assert "C. Babylon" in content, "Option C. should NOT be removed (not a Roman numeral section header)"
        assert "D. Arabs" in content, "Option D. should NOT be removed (not a Roman numeral section header)"

    def test_openr1_task_prefix(self):
        """OpenR1 NEW: 'Task 2. Solve for x...'"""
        doc = Document(
            id="test-openr1-9",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Task 2. Solve for $x$ in the equation $3x - 7 = 8$.",
                    }
                ],
                "ground_truth": "5",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Solve for $x$ in the equation $3x - 7 = 8$."

    def test_openr1_markdown_task_header(self):
        """OpenR1 NEW: '## Task\\n\\nCalculate...'"""
        doc = Document(
            id="test-openr1-10",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "## Task\n\nCalculate the area of a circle with radius $r = 5$.",
                    }
                ],
                "ground_truth": "25\\pi",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Calculate the area of a circle with radius $r = 5$."

    def test_openr1_markdown_condition_header(self):
        """OpenR1 NEW: '## Condition\\n\\nGiven that...'"""
        doc = Document(
            id="test-openr1-11",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "## Condition\n\nGiven that $a = 3$ and $b = 4$, find $\\sqrt{a^2 + b^2}$.",
                    }
                ],
                "ground_truth": "5",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Given that $a = 3$ and $b = 4$, find $\\sqrt{a^2 + b^2}$."

    def test_openr1_horizontal_rule(self):
        """OpenR1 NEW: Problem with horizontal rule separator"""
        doc = Document(
            id="test-openr1-12",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Find the value of $x$.\n\n---\n\nGiven: $2x + 3 = 7$",
                    }
                ],
                "ground_truth": "2",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert "---" not in result.metadata["prompt"][0]["content"]
        assert "Find the value of $x$" in result.metadata["prompt"][0]["content"]
        assert "Given: $2x + 3 = 7$" in result.metadata["prompt"][0]["content"]

    def test_openr1_translation_artifact(self):
        """OpenR1 NEW: Translation instruction artifact"""
        doc = Document(
            id="test-openr1-13",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Please retain the original text's line breaks and format, and output the translation result directly. Calculate $2 + 2$.",
                    }
                ],
                "ground_truth": "4",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        assert "Please retain" not in content
        assert "translation" not in content
        assert "Calculate $2 + 2$" in content

    def test_openr1_example_prefix(self):
        """OpenR1 NEW: 'Example 2. Solve the equation...'"""
        doc = Document(
            id="test-openr1-14",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Example 2. Solve the equation $x^2 = 9$.",
                    }
                ],
                "ground_truth": "x=±3",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Solve the equation $x^2 = 9$."

    def test_openr1_example_without_punctuation(self):
        """OpenR1 NEW: 'Example 4 Given that...' (no colon/period)"""
        doc = Document(
            id="test-openr1-15",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Example 4 Given that $\\alpha^{2005}+\\beta^{2005}$ can be expressed as a bivariate polynomial.",
                    }
                ],
                "ground_truth": "42",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Given that $\\alpha^{2005}+\\beta^{2005}$ can be expressed as a bivariate polynomial."

    def test_openr1_topic_label_square_brackets(self):
        """OpenR1 NEW: '[ Invariants ] On the board...'"""
        doc = Document(
            id="test-openr1-16",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "[ Invariants ]\n\nOn the board, the numbers $1,2, \\ldots, 20$ are written.",
                    }
                ],
                "ground_truth": "210",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "On the board, the numbers $1,2, \\ldots, 20$ are written."

    def test_openr1_proof_prefix(self):
        """OpenR1 NEW: 'Proof: Consider the...'"""
        doc = Document(
            id="test-openr1-17",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Proof: Consider the following inequality: $a^2 + b^2 \\geq 2ab$.",
                    }
                ],
                "ground_truth": "True",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Consider the following inequality: $a^2 + b^2 \\geq 2ab$."

    def test_openr1_hint_prefix(self):
        """OpenR1 NEW: 'Hint: Use the quadratic formula...'"""
        doc = Document(
            id="test-openr1-18",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Hint: Use the quadratic formula to solve $x^2 + 5x + 6 = 0$.",
                    }
                ],
                "ground_truth": "x=-2,-3",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Use the quadratic formula to solve $x^2 + 5x + 6 = 0$."

    def test_openr1_note_prefix(self):
        """OpenR1 NEW: 'Note. The triangle is isosceles...'"""
        doc = Document(
            id="test-openr1-19",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Note. The triangle is isosceles with sides $a = b = 5$.",
                    }
                ],
                "ground_truth": "5",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "The triangle is isosceles with sides $a = b = 5$."


class TestMathDatasetCleanerSkywork:
    """Tests using actual problematic samples from Skywork-OR1 dataset."""

    def test_skywork_question_230(self):
        """Skywork Sample 36: 'Question 230, Let $S$ be...'"""
        doc = Document(
            id="test-skywork-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Question 230, Let $S$ be the set of ordered 7-tuples $(a_1, a_2, \\ldots, a_7)$.",
                    }
                ],
                "ground_truth": "343",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("skywork-or1")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Let $S$ be the set of ordered 7-tuples $(a_1, a_2, \\ldots, a_7)$."


class TestMathDatasetCleanerDAPO:
    """Tests using DAPO-Math dataset (already very clean)."""

    def test_dapo_already_clean(self):
        """DAPO: Most samples are already clean, minimal changes expected"""
        doc = Document(
            id="test-dapo-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "For which $n$ is $n^4 + 6n^3 + 11n^2 + 3n + 31$ a perfect square?",
                    }
                ],
                "ground_truth": "10",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("dapo-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # Should be unchanged (already clean)
        assert result.metadata["prompt"][0]["content"] == "For which $n$ is $n^4 + 6n^3 + 11n^2 + 3n + 31$ a perfect square?"


class TestMathDatasetCleanerLaTeX:
    """Tests verifying LaTeX preservation across all datasets."""

    def test_latex_fractions_unchanged(self):
        """Verify \\frac stays as \\frac (NO escaping changes)"""
        doc = Document(
            id="test-latex-1",
            text="",
            metadata={
                "prompt": [{"role": "user", "content": "Calculate $\\frac{3}{4} + \\frac{1}{2}$"}],
                "ground_truth": "\\frac{5}{4}",
            },
        )

        cleaner = MathDatasetCleaner()
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # LaTeX unchanged
        assert "\\frac{3}{4}" in result.metadata["prompt"][0]["content"]
        assert "\\frac{1}{2}" in result.metadata["prompt"][0]["content"]
        assert result.metadata["ground_truth"] == "\\frac{5}{4}"

    def test_latex_complex_expressions_unchanged(self):
        """Verify complex LaTeX expressions are preserved"""
        content = "Solve $\\lim_{n \\rightarrow \\infty} \\frac{\\sin(n)}{n} + \\sqrt{n^2 + 1}$"
        doc = Document(
            id="test-latex-2",
            text="",
            metadata={
                "prompt": [{"role": "user", "content": content}],
                "ground_truth": "0",
            },
        )

        cleaner = MathDatasetCleaner()
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # Complex LaTeX unchanged
        assert result.metadata["prompt"][0]["content"] == content


class TestMathDatasetCleanerSchema:
    """Tests verifying schema preservation and metadata integrity."""

    def test_schema_preservation(self):
        """Verify output schema matches input schema exactly"""
        doc = Document(
            id="test-schema-1",
            text="original text",
            metadata={
                "prompt": [{"role": "user", "content": "Problem 1. Solve for $x$."}],
                "responses": [{"role": "assistant", "content": "x = 5"}],
                "ground_truth": "5",
                "extra_info": {
                    "difficulty": 2,
                    "source": "test",
                    "custom_field": "value",
                },
            },
        )

        cleaner = MathDatasetCleaner()
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # Verify all fields present
        assert result.id == "test-schema-1"
        assert result.text == "original text"  # Text field unchanged
        assert "prompt" in result.metadata
        assert "responses" in result.metadata
        assert "ground_truth" in result.metadata
        assert "extra_info" in result.metadata

    def test_extra_info_unchanged(self):
        """Verify extra_info is NEVER modified (per user requirement)"""
        original_extra_info = {
            "difficulty": 3,
            "source": "AIME",
            "year": 2004,
            "custom_metadata": {"key": "value"},
        }

        doc = Document(
            id="test-extra-1",
            text="",
            metadata={
                "prompt": [{"role": "user", "content": "8.3 Solve this problem."}],
                "ground_truth": "42",
                "extra_info": original_extra_info.copy(),
            },
        )

        cleaner = MathDatasetCleaner()
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # extra_info should be completely unchanged
        assert result.metadata["extra_info"] == original_extra_info
        # Verify no new fields added
        assert set(result.metadata["extra_info"].keys()) == set(original_extra_info.keys())

    def test_ground_truth_never_modified(self):
        """Verify ground_truth field is NEVER modified"""
        doc = Document(
            id="test-gt-1",
            text="",
            metadata={
                "prompt": [{"role": "user", "content": "8.3 What is the answer?"}],
                "ground_truth": "This is the ground truth with Problem 5. and (8 points)",
            },
        )

        cleaner = MathDatasetCleaner()
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # Ground truth should be completely unchanged (no cleaning applied)
        assert result.metadata["ground_truth"] == "This is the ground truth with Problem 5. and (8 points)"

    def test_responses_field_unchanged(self):
        """Verify responses field is never modified"""
        doc = Document(
            id="test-resp-1",
            text="",
            metadata={
                "prompt": [{"role": "user", "content": "Problem 1. Solve for $x$."}],
                "responses": [
                    {"role": "assistant", "content": "Let me solve Problem 1. The answer is x = 5 (10 points)"}
                ],
                "ground_truth": "5",
            },
        )

        cleaner = MathDatasetCleaner()
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # Responses unchanged (only prompt is cleaned)
        assert result.metadata["responses"][0]["content"] == "Let me solve Problem 1. The answer is x = 5 (10 points)"


class TestMathDatasetCleanerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_prompt(self):
        """Document without prompt in metadata should be skipped"""
        doc = Document(id="test-edge-1", text="some text", metadata={})

        cleaner = MathDatasetCleaner()
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # Should return original document unchanged
        assert result.id == "test-edge-1"
        assert result.text == "some text"

    def test_empty_prompt(self):
        """Document with empty prompt should be skipped"""
        doc = Document(id="test-edge-2", text="", metadata={"prompt": []})

        cleaner = MathDatasetCleaner()
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # Should return original document unchanged
        assert result.metadata["prompt"] == []

    def test_empty_content(self):
        """Document with empty content should be skipped"""
        doc = Document(
            id="test-edge-3",
            text="",
            metadata={
                "prompt": [{"role": "user", "content": ""}],
                "ground_truth": "5",
            },
        )

        cleaner = MathDatasetCleaner()
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # Should return original document unchanged
        assert result.metadata["prompt"][0]["content"] == ""

    def test_whitespace_normalization(self):
        """Test whitespace normalization"""
        doc = Document(
            id="test-edge-4",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "  Problem 1.   Solve for    $x$.\n\n\n\nWith extra newlines.  ",
                    }
                ],
                "ground_truth": "5",
            },
        )

        cleaner = MathDatasetCleaner(remove_problem_numbers=True, normalize_whitespace=True)
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # Whitespace normalized and problem number removed
        assert result.metadata["prompt"][0]["content"] == "Solve for $x$.\n\nWith extra newlines."


class TestMathDatasetCleanerConfiguration:
    """Tests for configuration and preset functionality."""

    def test_preset_orz_configuration(self):
        """Test ORZ preset has correct configuration"""
        cleaner = MathDatasetCleaner.from_preset("orz-math")
        assert cleaner.remove_problem_numbers is True
        assert cleaner.remove_point_allocations is True
        assert cleaner.remove_contest_metadata is True
        assert cleaner.remove_markdown_headers is True  # Added based on analysis
        assert cleaner.remove_special_artifacts is True  # NEW: horizontal rules, translation artifacts
        assert cleaner.detect_image_references is True

    def test_preset_openr1_configuration(self):
        """Test OpenR1 preset has correct configuration"""
        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        assert cleaner.remove_problem_numbers is True
        assert cleaner.remove_point_allocations is True
        assert cleaner.remove_contest_metadata is True
        assert cleaner.remove_markdown_headers is True  # OpenR1 has markdown headers
        assert cleaner.remove_special_artifacts is True  # NEW: horizontal rules, translation artifacts
        assert cleaner.detect_image_references is True

    def test_preset_skywork_configuration(self):
        """Test Skywork preset has correct configuration"""
        cleaner = MathDatasetCleaner.from_preset("skywork-or1")
        assert cleaner.remove_problem_numbers is True
        assert cleaner.remove_point_allocations is False  # Minimal cleaning for Skywork
        assert cleaner.remove_contest_metadata is False
        assert cleaner.remove_markdown_headers is False

    def test_preset_dapo_configuration(self):
        """Test DAPO preset has correct configuration"""
        cleaner = MathDatasetCleaner.from_preset("dapo-math")
        assert cleaner.remove_problem_numbers is True
        assert cleaner.remove_point_allocations is False  # Minimal cleaning needed
        assert cleaner.remove_contest_metadata is False
        assert cleaner.remove_markdown_headers is False

    def test_invalid_preset_raises_error(self):
        """Test that invalid preset name raises ValueError"""
        with pytest.raises(ValueError, match="Unknown preset"):
            MathDatasetCleaner.from_preset("invalid-preset")

    def test_custom_configuration(self):
        """Test custom configuration works"""
        cleaner = MathDatasetCleaner(
            remove_problem_numbers=False,
            remove_point_allocations=True,
            remove_contest_metadata=False,
            remove_markdown_headers=False,
            detect_image_references=False,
            normalize_whitespace=False,
        )

        doc = Document(
            id="test-custom-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Problem 1. Calculate (8 points) the value.",
                    }
                ],
                "ground_truth": "5",
            },
        )

        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # Problem number kept (disabled), but points removed (enabled)
        assert result.metadata["prompt"][0]["content"] == "Problem 1. Calculate the value."


class TestMathDatasetCleanerImageDetection:
    """Tests for image reference detection."""

    def test_detect_markdown_image(self):
        """Test detection of markdown images"""
        doc = Document(
            id="test-img-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "As shown in the diagram: ![](https://cdn.mathpix.com/cropped/image.png)",
                    }
                ],
                "ground_truth": "5",
            },
        )

        cleaner = MathDatasetCleaner(detect_image_references=True)
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # Image reference should be detected but NOT removed
        assert "![](https://cdn.mathpix.com/cropped/image.png)" in result.metadata["prompt"][0]["content"]

    def test_detect_asy_diagram(self):
        """Test detection of asymptote diagrams"""
        doc = Document(
            id="test-img-2",
            text="",
            metadata={
                "prompt": [{"role": "user", "content": "In the diagram: [asy] draw((0,0)--(1,1)); [/asy]"}],
                "ground_truth": "5",
            },
        )

        cleaner = MathDatasetCleaner(detect_image_references=True)
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # ASY code should be detected but NOT removed
        assert "[asy]" in result.metadata["prompt"][0]["content"]

    def test_detect_figure_reference(self):
        """Test detection of figure references"""
        doc = Document(
            id="test-img-3",
            text="",
            metadata={
                "prompt": [{"role": "user", "content": "As shown in Figure 1, the triangle has sides..."}],
                "ground_truth": "5",
            },
        )

        cleaner = MathDatasetCleaner(detect_image_references=True)
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # Figure reference should be detected but NOT removed
        assert "Figure 1" in result.metadata["prompt"][0]["content"]


class TestMathDatasetCleanerNewFeatures:
    """Tests for new artifact removal and filtering features."""

    def test_task_label_b34_format(self):
        """Test removal of 'Task B-3.4.' format"""
        doc = Document(
            id="test-new-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Task B-3.4. Solve the equation $x^{\\log _{5} 6}-5 \\cdot 6^{\\log _{5} \\sqrt{x}}=6$.",
                    }
                ],
                "ground_truth": "25",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "Solve the equation $x^{\\log _{5} 6}-5 \\cdot 6^{\\log _{5} \\sqrt{x}}=6$."
        assert result.metadata["ground_truth"] == "25"

    def test_author_attribution_removal(self):
        """Test removal of author attribution in LaTeX underline format"""
        doc = Document(
            id="test-new-2",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "$\\underline{\\text { Khachaturyan A.V. }}$\n\n13 children sat at a round table.",
                    }
                ],
                "ground_truth": "7",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        assert result.metadata["prompt"][0]["content"] == "13 children sat at a round table."
        assert result.metadata["ground_truth"] == "7"

    def test_translation_instruction_removal(self):
        """Test removal of 'Translate the above text into English...' instruction"""
        doc = Document(
            id="test-new-3",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "$7.9 \\quad 3 \\cdot 5^{2x-1}-2 \\cdot 5^{x-1}=0.2$.\n\nTranslate the above text into English, keeping the original text's line breaks and format, and output the translation result directly.",
                    }
                ],
                "ground_truth": "0",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        # Translation instruction should be removed
        assert "Translate the above text" not in result.metadata["prompt"][0]["content"]
        assert "$7.9 \\quad 3 \\cdot 5^{2x-1}-2 \\cdot 5^{x-1}=0.2$." in result.metadata["prompt"][0]["content"]
        assert result.metadata["ground_truth"] == "0"

    def test_url_filtering(self):
        """Test filtering out samples with URLs"""
        doc = Document(
            id="test-new-4",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Let $S$ be a [set](https://artofproblemsolving.com/wiki/index.php/Set) with six elements.",
                    }
                ],
                "ground_truth": "710",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))

        # Sample should be filtered out (empty result)
        assert len(result) == 0

    def test_multipart_filtering(self):
        """Test filtering out multi-part problems with a), b), c)"""
        doc = Document(
            id="test-new-5",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "a) Calculate the total amount of milk in liters.\n\nb) Calculate the smallest possible number of tankers.",
                    }
                ],
                "ground_truth": "4",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))

        # Sample should be filtered out (empty result)
        assert len(result) == 0

    def test_url_filtering_disabled(self):
        """Test that URL filtering can be disabled"""
        doc = Document(
            id="test-new-6",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Visit https://example.com for more information about this problem.",
                    }
                ],
                "ground_truth": "42",
            },
        )

        cleaner = MathDatasetCleaner(filter_url_samples=False)
        result = list(cleaner.run([doc], rank=0, world_size=1))

        # Sample should NOT be filtered out when filtering is disabled
        assert len(result) == 1
        assert result[0].metadata["ground_truth"] == "42"

    def test_multipart_filtering_disabled(self):
        """Test that multipart filtering can be disabled"""
        doc = Document(
            id="test-new-7",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "a) Prove that X. b) Find Y.",
                    }
                ],
                "ground_truth": "5",
            },
        )

        cleaner = MathDatasetCleaner(filter_multipart_samples=False)
        result = list(cleaner.run([doc], rank=0, world_size=1))

        # Sample should NOT be filtered out when filtering is disabled
        assert len(result) == 1
        assert result[0].metadata["ground_truth"] == "5"


def test_multipart_false_positive_function_notation():
    """Test that function notation like f(x), g(y) is NOT filtered as multipart."""
    doc = Document(
        id="test-fp-1",
        text="",
        metadata={
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Given that f(x) is a function defined on R, and satisfies "
                        "f(x+2)[1-f(x)]=1+f(x), f(1)=9997, then the value of f(2009) is ___."
                    ),
                }
            ],
            "ground_truth": "42",
        },
    )

    cleaner = MathDatasetCleaner(filter_multipart_samples=True)
    result = list(cleaner.run([doc], rank=0, world_size=1))

    # Should NOT be filtered (function notation, not multipart problem)
    assert len(result) == 1


def test_multipart_false_positive_coordinates():
    """Test that coordinate notation like (a, b) is NOT filtered as multipart."""
    doc = Document(
        id="test-fp-2",
        text="",
        metadata={
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Find the distance between points (a, b) and (c, d) "
                        "in the coordinate plane."
                    ),
                }
            ],
            "ground_truth": "sqrt((c-a)^2 + (d-b)^2)",
        },
    )

    cleaner = MathDatasetCleaner(filter_multipart_samples=True)
    result = list(cleaner.run([doc], rank=0, world_size=1))

    # Should NOT be filtered (coordinate notation, not multipart problem)
    assert len(result) == 1


def test_multipart_false_positive_math_expression():
    """Test that math expressions like ab=2(a+b) are NOT filtered as multipart."""
    doc = Document(
        id="test-fp-3",
        text="",
        metadata={
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Solve the equation where ab=2(a+b), bc=3(b+c), and cd=4(c+d). "
                        "Find the value of d."
                    ),
                }
            ],
            "ground_truth": "24",
        },
    )

    cleaner = MathDatasetCleaner(filter_multipart_samples=True)
    result = list(cleaner.run([doc], rank=0, world_size=1))

    # Should NOT be filtered (variable expression, not multipart problem)
    assert len(result) == 1


def test_multipart_false_positive_gcd_notation():
    """Test that function calls like gcd(a, b) are NOT filtered as multipart."""
    doc = Document(
        id="test-fp-4",
        text="",
        metadata={
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Let p be a prime and A be a set such that for any subset, "
                        "the product is not a perfect p-th power. What is the largest "
                        "possible number of elements in A?"
                    ),
                }
            ],
            "ground_truth": "p-1",
        },
    )

    cleaner = MathDatasetCleaner(filter_multipart_samples=True)
    result = list(cleaner.run([doc], rank=0, world_size=1))

    # Should NOT be filtered (variable in context, not multipart problem)
    assert len(result) == 1


def test_multipart_true_positive_with_newlines():
    """Test that true multipart problems with a), b) structure ARE filtered."""
    doc = Document(
        id="test-tp-1",
        text="",
        metadata={
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Consider the quadratic equation x^2 - 3px - p = 0.\n\n"
                        "a) Prove that 3px1 + x2^2 - p > 0.\n\n"
                        "b) Find the least possible value of the expression A."
                    ),
                }
            ],
            "ground_truth": "5",
        },
    )

    cleaner = MathDatasetCleaner(filter_multipart_samples=True)
    result = list(cleaner.run([doc], rank=0, world_size=1))

    # Should be filtered (true multipart problem with a), b) structure)
    assert len(result) == 0


def test_multipart_true_positive_sequential_parts():
    """Test that problems with sequential a), b), c) parts ARE filtered."""
    doc = Document(
        id="test-tp-2",
        text="",
        metadata={
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Point D lies on side AC of triangle ABC.\n\n"
                        "a) Find the angle ABC.\n\n"
                        "b) Suppose MP=1, NT=3/2, BD=sqrt(5). Find the area.\n\n"
                        "c) Verify your answer using the Pythagorean theorem."
                    ),
                }
            ],
            "ground_truth": "15",
        },
    )

    cleaner = MathDatasetCleaner(filter_multipart_samples=True)
    result = list(cleaner.run([doc], rank=0, world_size=1))

    # Should be filtered (true multipart problem with a), b), c) structure)
    assert len(result) == 0


def test_multipart_roman_numeral_true_positive():
    """Test that Roman numeral multipart problems like (I), (II) ARE filtered."""
    doc = Document(
        id="test-roman-tp-1",
        text="",
        metadata={
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Let the function $f(x)=\\sin x+\\sqrt{3} \\cos x+1$,\n"
                        "(I) Find the maximum and minimum values of the function $f(x)$ on $\\left[0, \\frac{\\pi}{2}\\right]$;\n"
                        "(II) If real numbers $a, b, c$ satisfy $a f(x)+b f(x-c)=1$ for any $x \\in \\mathbb{R}$, "
                        "find the value of $\\frac{b \\cos c}{a}$."
                    ),
                }
            ],
            "ground_truth": "-1",
        },
    )

    cleaner = MathDatasetCleaner(filter_multipart_samples=True)
    result = list(cleaner.run([doc], rank=0, world_size=1))

    # Should be filtered (true multipart problem with Roman numeral structure)
    assert len(result) == 0


def test_multipart_roman_numeral_false_positive():
    """Test that Roman numeral function notation like f(I) is NOT filtered as multipart."""
    doc = Document(
        id="test-roman-fp-1",
        text="",
        metadata={
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Let f be a function defined on interval I=[0,1] such that "
                        "f(I) is a compact subset of R. Prove that f is uniformly continuous."
                    ),
                }
            ],
            "ground_truth": "proof required",
        },
    )

    cleaner = MathDatasetCleaner(filter_multipart_samples=True)
    result = list(cleaner.run([doc], rank=0, world_size=1))

    # Should NOT be filtered (not a multipart problem, just function notation)
    assert len(result) == 1


def test_multipart_roman_numeral_three_parts():
    """Test that three-part Roman numeral problems (I), (II), (III) ARE filtered."""
    doc = Document(
        id="test-roman-tp-2",
        text="",
        metadata={
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Given the function $f(x)=\\sin \\left(x+\\frac{\\pi}{6}\\right)+\\sin \\left(x-\\frac{\\pi}{6}\\right)+\\cos x+a$\n"
                        "(I) Find the smallest positive period of the function $f(x)$.\n"
                        "(II) If the maximum value of $f(x)$ is 1, find the value of $a$.\n"
                        "(III) If $x \\in\\left[-\\frac{\\pi}{2}, \\frac{\\pi}{2}\\right]$ when the maximum value "
                        "of $f(x)$ is 1, find the value of $a$."
                    ),
                }
            ],
            "ground_truth": "-1",
        },
    )

    cleaner = MathDatasetCleaner(filter_multipart_samples=True)
    result = list(cleaner.run([doc], rank=0, world_size=1))

    # Should be filtered (true multipart problem with 3 parts)
    assert len(result) == 0


class TestMathDatasetCleanerNewArtifacts:
    """Tests for new artifact removal features: trailing artifacts and standalone numbers."""

    def test_trailing_markdown_header_sample_6(self):
        """User Sample 6: Remove '## second grade' at end of problem"""
        doc = Document(
            id="test-trailing-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            "Ana, Biljana, Vesna, and Gordana crossed the river in a canoe as follows: "
                            "There were three trips from the left to the right bank, each time with two "
                            "girls in the canoe, one of whom was rowing. On both trips from the right bank "
                            "to the left, there was only one girl in the canoe. It is known that Ana "
                            "can only row if she is alone in the canoe, and Biljana can row if she is alone "
                            "or with Vesna. It is also known that each girl rowed at least once. "
                            "Which of them rowed twice?\n\n## second grade"
                        ),
                    }
                ],
                "ground_truth": "Vesna",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        assert "## second grade" not in content
        assert "Which of them rowed twice?" in content
        assert result.metadata["ground_truth"] == "Vesna"

    def test_standalone_number_147(self):
        """User Sample 15: Remove '147 ' at start before 'Let'"""
        doc = Document(
            id="test-standalone-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "147 Let $x, y, z > 0$ and $x + y + z = 1$, then the minimum value of $\\frac{1}{x} + \\frac{4}{y} + \\frac{9}{z}$ is",
                    }
                ],
                "ground_truth": "36",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        assert not content.startswith("147 ")
        assert content.startswith("Let $x, y, z > 0$")
        assert result.metadata["ground_truth"] == "36"

    def test_standalone_number_1_find(self):
        """User Sample 13: Remove '1. ' at start"""
        doc = Document(
            id="test-standalone-2",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "1. Find the sum of the squares of two numbers if it is known that their arithmetic mean is 8, and the geometric mean is $2 \\sqrt{5}$.",
                    }
                ],
                "ground_truth": "216",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        assert not content.startswith("1. ")
        assert content.startswith("Find the sum")
        assert result.metadata["ground_truth"] == "216"

    def test_contest_metadata_ico_2021(self):
        """User Sample 17: Contest metadata already handled by existing patterns"""
        doc = Document(
            id="test-contest-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            "(The 2021 ICO P4)\n\n"
                            "The path index of a graph $G$ is the minimum number of paths needed to pass through "
                            "each vertex of $G$ exactly once. Given a connected graph $G$, what is the maximum "
                            "possible value for its path index, knowing that the largest set of vertices in $G$ "
                            "that are pairwise non-adjacent is $n>1$ (independence number)?"
                        ),
                    }
                ],
                "ground_truth": "n-1",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        assert "(The 2021 ICO P4)" not in content
        assert content.startswith("The path index")
        assert result.metadata["ground_truth"] == "n-1"

    def test_trailing_bold_label(self):
        """Test removal of bold labels at end like '**Level 3**'"""
        doc = Document(
            id="test-trailing-2",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Calculate the value of $\\sqrt{144}$.\n\n**Level 3**",
                    }
                ],
                "ground_truth": "12",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        assert "**Level 3**" not in content
        assert "Calculate the value" in content

    def test_trailing_category_label(self):
        """Test removal of category labels at end like '[ Geometry ]'"""
        doc = Document(
            id="test-trailing-3",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Find the area of a circle with radius 5.\n\n[ Geometry ]",
                    }
                ],
                "ground_truth": "25\\pi",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        assert "[ Geometry ]" not in content
        assert "Find the area" in content

    def test_false_positive_year_in_middle(self):
        """False positive: Year in middle of text should NOT be removed"""
        doc = Document(
            id="test-fp-year-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "In 2024, a mathematician discovered that 147 primes exist in a certain sequence.",
                    }
                ],
                "ground_truth": "147",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        # "2024" and "147" should NOT be removed (not at start with capital letter after)
        assert "In 2024" in content
        assert "147 primes" in content

    def test_false_positive_number_in_problem(self):
        """False positive: Numbers that are part of the problem should NOT be removed"""
        doc = Document(
            id="test-fp-number-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Calculate 147 * 2 and then find the square root of the result.",
                    }
                ],
                "ground_truth": "\\sqrt{294}",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        # "147" should NOT be removed (not followed by capitalized word at start)
        assert "Calculate 147 * 2" in content

    def test_false_positive_markdown_in_middle(self):
        """False positive: Markdown headers in middle should NOT be removed"""
        doc = Document(
            id="test-fp-markdown-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Consider the following:\n\n## Part A\n\nFind x such that x^2 = 4.\n\n## Part B\n\nFind y such that y^2 = 9.",
                    }
                ],
                "ground_truth": "x=±2, y=±3",
            },
        )

        # Use cleaner WITHOUT trailing artifacts enabled (default)
        cleaner = MathDatasetCleaner(
            remove_problem_numbers=True,
            remove_markdown_headers=True,  # Only removes at START
            remove_trailing_artifacts=False,  # Don't remove at END
        )
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        # Middle headers should remain (only trailing headers are removed when enabled)
        assert "## Part A" in content
        assert "## Part B" in content

    def test_trailing_artifacts_disabled_by_default(self):
        """Test that trailing artifact removal is disabled by default"""
        doc = Document(
            id="test-disabled-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Solve for x in the equation $2x + 3 = 7$.\n\n## Level 2",
                    }
                ],
                "ground_truth": "2",
            },
        )

        # Default cleaner should NOT remove trailing artifacts
        cleaner = MathDatasetCleaner()
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        # Trailing header should remain (feature disabled by default)
        assert "## Level 2" in content

    def test_standalone_number_only_1_to_3_digits(self):
        """Test that standalone number pattern only matches 1-3 digits"""
        doc = Document(
            id="test-standalone-3",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "2024 In this year, a new theorem was discovered about prime numbers.",
                    }
                ],
                "ground_truth": "N/A",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        # "2024" should NOT be removed (4 digits, pattern only matches 1-3 digits)
        assert content.startswith("2024 In")

    def test_multiple_trailing_artifacts(self):
        """Test removal of multiple trailing artifacts"""
        doc = Document(
            id="test-trailing-4",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Find all prime numbers less than 20.\n\n## Number Theory\n\n**Easy**",
                    }
                ],
                "ground_truth": "2, 3, 5, 7, 11, 13, 17, 19",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        assert "## Number Theory" not in content
        assert "**Easy**" not in content
        assert "Find all prime numbers" in content

    def test_bracket_number_12(self):
        """Test removal of bracket number '[12] ' at start"""
        doc = Document(
            id="test-bracket-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "[12] Find the number of subsets $S$ of $\\{1,2, \\ldots 6\\}$ satisfying the following conditions: S is non-empty.",
                    }
                ],
                "ground_truth": "N/A",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        assert not content.startswith("[12] ")
        assert content.startswith("Find the number of subsets")

    def test_task_dash_format(self):
        """Test removal of 'Task 2 - 200512 ' format"""
        doc = Document(
            id="test-task-dash-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Task 2 - 200512 To transport a certain amount of gravel, a truck with a 5 t loading capacity would have had to make exactly 105 fully loaded trips.",
                    }
                ],
                "ground_truth": "N/A",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        assert not content.startswith("Task 2 - 200512 ")
        assert content.startswith("To transport a certain amount")

    def test_multi_letter_country_code(self):
        """Test removal of 'N8 (IRN) ' format with multi-letter prefix"""
        doc = Document(
            id="test-country-code-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "N8 (IRN) Let $p$ be a prime number and let $A$ be a set of positive integers.",
                    }
                ],
                "ground_truth": "N/A",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        assert not content.startswith("N8 (IRN) ")
        assert content.startswith("Let $p$ be a prime number")

    def test_quoted_contest_name(self):
        """Test removal of '(8th "Hope Cup" ...)' format"""
        doc = Document(
            id="test-quoted-contest-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": '(8th "Hope Cup" Invitational Competition Question) If $a+b+c=1$, what is the maximum value?',
                    }
                ],
                "ground_truth": "N/A",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        assert "(8th" not in content
        assert '"Hope Cup"' not in content
        assert content.startswith("If $a+b+c=1$")

    def test_latex_author_with_brackets(self):
        """Test removal of '$ [ topic ] Author: Name $' format"""
        doc = Document(
            id="test-latex-author-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "$ [ Extreme principle (other) . ] Author: Shapovalov $A . B$. In a $29 \\times 29$ table, the numbers were written.",
                    }
                ],
                "ground_truth": "N/A",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))[0]

        content = result.metadata["prompt"][0]["content"]
        assert "Extreme principle" not in content
        assert "Shapovalov" not in content
        assert content.startswith("In a $29 \\times 29$ table")

    def test_multipart_lowercase_roman_numerals(self):
        """Test filtering of multipart problems with lowercase roman numerals: (i)...(ii)..."""
        doc = Document(
            id="test-multipart-roman-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "(i) Find all integers $n \\geqslant 1$ such that $n$ divides $2^{n}-1$.\n\n(ii) Find all odd integers $n \\geqslant 1$ such that $n$ divides $3^{n}+1$.",
                    }
                ],
                "ground_truth": "N/A",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))

        # Document should be filtered out (empty list)
        assert len(result) == 0

    def test_multipart_false_positive_in_conditions(self):
        """Test that (i), (ii) in problem conditions are NOT filtered (lowercase after pattern)"""
        doc = Document(
            id="test-multipart-false-positive-1",
            text="",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "N8 (IRN) Let $p$ be a prime number satisfying: (i) the set of prime divisors of elements in $A$ consists of $p-1$ elements; (ii) for any nonempty subset of $A$, the product is not a perfect $p$ th power.",
                    }
                ],
                "ground_truth": "N/A",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))

        # Document should NOT be filtered (multipart pattern requires uppercase after)
        assert len(result) == 1
        content = result[0].metadata["prompt"][0]["content"]
        # Should still have the problem (with N8 (IRN) removed)
        assert "Let $p$ be a prime number" in content


class TestMultilineFilteringIntegration:
    """Integration tests for multiline-aware multi-part filtering.

    Tests validate that the refactored filtering logic:
    1. Does NOT filter false positives (equation/case references on same line)
    2. DOES filter true positives (actual multi-part problems across lines)
    """

    # === False Positive Tests (should NOT be filtered) ===

    def test_equation_reference_not_filtered(self):
        """Equation references like 'equations $(1)$ and $(2)$' should NOT be filtered"""
        doc = Document(
            text="",
            id="eq_ref_test",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "From equations $(1)$ and $(2)$, we can derive the final answer is 42.",
                    }
                ],
                "ground_truth": "42",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))

        assert len(result) == 1, "Equation reference should NOT be filtered"
        assert result[0].metadata["prompt"][0]["content"] == "From equations $(1)$ and $(2)$, we can derive the final answer is 42."

    def test_case_description_not_filtered(self):
        """Case descriptions like 'cases (i) and (ii)' should NOT be filtered"""
        doc = Document(
            text="",
            id="case_ref_test",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "In cases (i) and (ii), prove that the limit exists and equals zero.",
                    }
                ],
                "ground_truth": "0",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))

        assert len(result) == 1, "Case reference should NOT be filtered"
        assert result[0].metadata["prompt"][0]["content"] == "In cases (i) and (ii), prove that the limit exists and equals zero."

    def test_set_definition_not_filtered(self):
        """Set definitions like 'Let $(I)$ be...' should NOT be filtered"""
        doc = Document(
            text="",
            id="set_def_test",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Let $(I)$ be set A and $(II)$ be set B, then prove intersection is empty.",
                    }
                ],
                "ground_truth": "empty",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))

        assert len(result) == 1, "Set definition should NOT be filtered"
        assert result[0].metadata["prompt"][0]["content"] == "Let $(I)$ be set A and $(II)$ be set B, then prove intersection is empty."

    def test_inline_roman_reference_not_filtered(self):
        """Inline Roman numeral references should NOT be filtered"""
        doc = Document(
            text="",
            id="inline_roman_test",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": "Compare theorems (i) and (ii) from the previous section and derive a new result.",
                    }
                ],
                "ground_truth": "proven",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))

        assert len(result) == 1, "Inline Roman reference should NOT be filtered"

    # === True Positive Tests (should be filtered) ===

    def test_multiline_dollar_arabic_filtered(self):
        """Multi-line problems with $(1)$ and $(2)$ SHOULD be filtered"""
        doc = Document(
            text="",
            id="multiline_arabic_test",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": """$(1)$ Calculate the area of the triangle.

$(2)$ Find the perimeter of the circle.""",
                    }
                ],
                "ground_truth": "multi-part",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))

        assert len(result) == 0, "Multi-line dollar Arabic should be filtered"

    def test_multiline_lowercase_roman_filtered(self):
        """Multi-line problems with (i) and (ii) SHOULD be filtered"""
        doc = Document(
            text="",
            id="multiline_roman_test",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": """(i) Prove the function is continuous.

(ii) Show that the derivative exists everywhere.""",
                    }
                ],
                "ground_truth": "multi-part",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))

        assert len(result) == 0, "Multi-line lowercase Roman should be filtered"

    def test_multiline_dollar_roman_filtered(self):
        """Multi-line problems with $(I)$ and $(II)$ SHOULD be filtered"""
        doc = Document(
            text="",
            id="multiline_dollar_roman_test",
            metadata={
                "prompt": [
                    {
                        "role": "user",
                        "content": """$(I)$ Find all solutions to the equation.

$(II)$ Verify each solution is valid.""",
                    }
                ],
                "ground_truth": "multi-part",
            },
        )

        cleaner = MathDatasetCleaner.from_preset("orz-math")
        result = list(cleaner.run([doc], rank=0, world_size=1))

        assert len(result) == 0, "Multi-line dollar Roman should be filtered"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
