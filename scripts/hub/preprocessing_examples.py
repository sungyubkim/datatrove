"""
Preprocessing examples for README generation.
Contains before/after samples for each cleaning preset.
"""

from typing import Dict, List, TypedDict


class PreprocessingExample(TypedDict):
    """Type definition for a single preprocessing example."""

    title: str
    before: str
    after: str
    changes: List[str]


# Preprocessing examples for each cleaning preset
PREPROCESSING_EXAMPLES: Dict[str, List[PreprocessingExample]] = {
    "orz-math": [
        {
            "title": "Contest Metadata and Problem Number Removal",
            "before": "24th Eötvös 1917 Problem 2 A square is divided into $n^2$ smaller squares. Find the number of squares with sides parallel to the sides of the original square.",
            "after": "A square is divided into $n^2$ smaller squares. Find the number of squares with sides parallel to the sides of the original square.",
            "changes": [
                "✓ Removed contest name ('24th Eötvös 1917')",
                "✓ Removed problem number ('Problem 2')",
                "✓ Preserved mathematical LaTeX notation ($n^2$)",
            ],
        },
        {
            "title": "Problem Number and Point Allocation Removal",
            "before": "Problem 6. (8 points) In the plane, there is a non-closed, non-self-intersecting broken line consisting of $n$ segments. What is the maximum number of self-intersection points?",
            "after": "In the plane, there is a non-closed, non-self-intersecting broken line consisting of $n$ segments. What is the maximum number of self-intersection points?",
            "changes": [
                "✓ Removed problem number prefix ('Problem 6.')",
                "✓ Removed point allocation ('(8 points)')",
                "✓ Preserved problem statement and LaTeX ($n$)",
            ],
        },
    ],
    "openr1-math": [
        {
            "title": "Markdown Header and Task Label Removal",
            "before": "## Problem Statement\n\nCalculate the limit of the numerical sequence: $\\lim_{n \\rightarrow \\infty} \\frac{(n+1)^{4}-(n-1)^{4}}{(n+1)^{3}+(n-1)^{3}}$",
            "after": "Calculate the limit of the numerical sequence: $\\lim_{n \\rightarrow \\infty} \\frac{(n+1)^{4}-(n-1)^{4}}{(n+1)^{3}+(n-1)^{3}}$",
            "changes": [
                "✓ Removed markdown header ('## Problem Statement')",
                "✓ Cleaned up extra newlines",
                "✓ Preserved complex LaTeX expressions",
            ],
        },
        {
            "title": "Translation Artifact Removal",
            "before": "Please retain the original text's line breaks and format, and output the translation result directly. Calculate $2 + 2$.",
            "after": "Calculate $2 + 2$.",
            "changes": [
                "✓ Removed translation instruction artifact",
                "✓ Kept only the actual problem statement",
            ],
        },
    ],
    "skywork-or1": [
        {
            "title": "Question Number Removal",
            "before": "Question 230, Let $S$ be the set of ordered 7-tuples $(a_1, a_2, \\ldots, a_7)$ of positive integers such that the sum equals 8. Find the number of elements in $S$.",
            "after": "Let $S$ be the set of ordered 7-tuples $(a_1, a_2, \\ldots, a_7)$ of positive integers such that the sum equals 8. Find the number of elements in $S$.",
            "changes": [
                "✓ Removed question number prefix ('Question 230,')",
                "✓ Preserved mathematical notation and structure",
            ],
        },
    ],
    "dapo-math": [
        {
            "title": "Minimal Cleaning (Already Clean Dataset)",
            "before": "For which $n$ is $n^4 + 6n^3 + 11n^2 + 3n + 31$ a perfect square?",
            "after": "For which $n$ is $n^4 + 6n^3 + 11n^2 + 3n + 31$ a perfect square?",
            "changes": [
                "✓ No changes needed - DAPO-Math is already well-formatted",
                "✓ Only basic validation and URL filtering applied",
            ],
        },
    ],
    "standard": [
        {
            "title": "Standard Processing (No Cleaning Preset)",
            "before": "Calculate the derivative of $f(x) = x^2 + 3x + 2$ with respect to $x$.",
            "after": "Calculate the derivative of $f(x) = x^2 + 3x + 2$ with respect to $x$.",
            "changes": [
                "✓ Format normalization to VERL schema",
                "✓ URL filtering (samples with URLs removed)",
                "✓ Basic text validation",
                "✓ No artifact removal applied",
            ],
        },
    ],
}


def get_examples(preset: str) -> List[PreprocessingExample]:
    """Get preprocessing examples for a specific preset.

    Args:
        preset: Cleaning preset name ('orz-math', 'openr1-math', 'skywork-or1', 'dapo-math', 'standard')

    Returns:
        List of preprocessing examples

    Raises:
        KeyError: If preset not found
    """
    if preset not in PREPROCESSING_EXAMPLES:
        available = list(PREPROCESSING_EXAMPLES.keys())
        raise KeyError(f"Preset '{preset}' not found. Available presets: {available}")

    return PREPROCESSING_EXAMPLES[preset]


def format_example_for_readme(example: PreprocessingExample) -> str:
    """Format a preprocessing example for README markdown.

    Args:
        example: Preprocessing example to format

    Returns:
        Formatted markdown string
    """
    changes_list = "\n".join([f"- {change}" for change in example["changes"]])

    return f"""### {example['title']}

**Before Cleaning:**
```
{example['before']}
```

**After Cleaning:**
```
{example['after']}
```

**Changes Applied:**
{changes_list}
"""


if __name__ == "__main__":
    # Print examples for all presets
    print("Preprocessing Examples for Math Dataset Cleaning")
    print("=" * 70)

    for preset, examples in PREPROCESSING_EXAMPLES.items():
        print(f"\n{preset.upper()} Preset:")
        print("-" * 70)
        for i, example in enumerate(examples, 1):
            print(f"\nExample {i}: {example['title']}")
            print(f"Before: {example['before'][:80]}...")
            print(f"After: {example['after'][:80]}...")
            print(f"Changes: {len(example['changes'])} modifications")
