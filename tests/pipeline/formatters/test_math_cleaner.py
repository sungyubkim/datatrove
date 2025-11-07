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
        assert cleaner.remove_markdown_headers is False  # Not needed for ORZ
        assert cleaner.detect_image_references is True

    def test_preset_openr1_configuration(self):
        """Test OpenR1 preset has correct configuration"""
        cleaner = MathDatasetCleaner.from_preset("openr1-math")
        assert cleaner.remove_problem_numbers is True
        assert cleaner.remove_point_allocations is True
        assert cleaner.remove_contest_metadata is True
        assert cleaner.remove_markdown_headers is True  # OpenR1 has markdown headers
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
