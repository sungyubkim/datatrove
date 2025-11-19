"""Math Dataset Cleaner for VERL format datasets.

This module provides a formatter that cleans math problem datasets in VERL format
by removing various artifacts like problem numbering, contest metadata, point allocations,
and markdown headers while preserving the mathematical content.

Conservative approach: Only removes clear artifacts, preserves all content including
vague ground truths, non-English text, and LaTeX formatting.
"""

import re
from typing import TYPE_CHECKING

from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger
from datatrove.utils.typeshelper import StatHints

if TYPE_CHECKING:
    from datatrove.data import Document, DocumentsPipeline


# Problem numbering patterns (from all 4 datasets)
PROBLEM_NUMBER_PATTERNS = [
    r"^Problem\s+\d+[\.:]\s*",  # "Problem 6."
    r"^Question\s+\d+[\.:,]\s*",  # "Question 230,"
    r"^Exercise\s+\d+[\.:]\s*",  # "Exercise 12:"
    r"^Task\s+\d+[\.:]\s*",  # "Task 2."
    r"^Task\s+[A-Z]-?\d+\.\d+\.?\s*",  # "Task B-3.4.", "Task B-2.2."
    r"^Example\s+\d+[\.:,]?\s+",  # "Example 31", "Example 31:", "Example 31."
    r"^\d+\.\d+(?:[\.:]\s*|\s+)",  # "8.3:", "8.3.", "8.3 "
    r"^[A-Z]\d+\.\d+(?:[\.:]\s*|\s+)",  # "G1.4:", "I2.1:", "G1.4 "
    r"^[A-Z]\d+[\.:]\s*",  # "B1.", "G2.", "I4." (letter-number without decimal)
    r"^[A-Z]\d+\s*\([A-Z]+\)\s*",  # "A2 (RUS)"
    r"^\d+[\.:]\s+",  # "1. ", "2. ", "5. " (single/multi digit followed by period and space)
    r"^\(\d+\)\s*",  # "(1)", "(2)", "(3)" at start
    r"^[IVXLCDM]{2,}[\.:]\s+",  # "II.", "III.", "IV." (multi-letter Roman numerals only)
    r"^\[\s*[A-Za-z][^\]]{0,50}\s*\]\s*",  # "[ Invariants ]", "[ Geometry ]"
    r"^(Proof|Hint|Note|Remark|Observation|Claim|Lemma)[\.:]\s+",  # "Proof:", "Hint:", etc.
    r"^\d{1,3}\s+(?=[A-Z][a-z])",  # "147 Let", "1 Find" (1-3 digits, space, capitalized word)
    r"^\[\d+\]\s*",  # "[12] Find...", "[200] Let..." (bracketed numbers)
    r"^Task\s+\d+\s+-\s+\d+\s+",  # "Task 2 - 200512 To transport..." (task with ID number)
    r"^[A-Z]+\d+\s*\([A-Z]{2,}\)\s*",  # "N8 (IRN) Let..." (letter-number with country code)
]

# Contest metadata patterns
CONTEST_METADATA_PATTERNS = [
    # "20th APMC 1997 Problem 3", "24th Eötvös 1917 Problem 2"
    # Use \S+ to match contest names with unicode characters (e.g., Eötvös)
    r"^\d+(?:st|nd|rd|th)\s+\S+\s+\d{4}\s+Problem\s+\d+\s+",
    # "(2004 College Entrance Examination...)" - more conservative
    r"\((?:19|20)\d{2}[,\s]+[^)]{0,80}(?:Competition|Entrance Examination)[^)]*\)\s*",
    # "2004 AIME Problem 3", "1997 IMO 3"
    r"(?:19|20)\d{2},?\s+(?:AIME|IMO|AMC|USAMO|BMO|Eötvös|APMC)\s*(?:Problem)?\s*\d+[\.:,]?\s+",
    # "(The 2021 ICO P4)" format - contest with "The" prefix
    r"^\(The\s+\d{4}\s+[A-Z]{2,}(?:\s+P\d+)?[^)]*\)\s*\n*",
    # "(8th "Hope Cup" Invitational Competition Question)" - quoted contest names
    r"^\(\d+(?:st|nd|rd|th)\s+\"[^\"]+\"[^)]*\)\s*",
]

# Point allocations
POINT_ALLOCATION_PATTERNS = [
    r"\s*\(\d+\s*points?\)\s*",  # " (8 points) ", "(1 point) "
    r"\s*\[\d+\s*points?\]\s*",  # " [15 points] "
    r"Reviews?\s+\([^)]*\d+\s*points?[^)]*\)\s*",  # "Reviews (from 7th grade. 1 point)"
]

# Author attribution patterns
AUTHOR_ATTRIBUTION_PATTERNS = [
    r"^\$\\underline\{\\text\s*\{[^\}]+\}\}\$\.?\s*\n*",  # "$\underline{\text { Author Name }}$"
    r"^\$\s*\[[^\]]+\]\s*Author:.*?\$\.\s*",  # "$ [ Extreme principle ] Author: Shapovalov $A . B$."
]

# Markdown headers and header text (at start of text)
MARKDOWN_HEADER_PATTERNS = [
    # Remove all markdown headers (# to ######) with any content
    r"^#{1,6}\s+[^\n]+\n+",
    # Remove standalone header keywords (in case markdown prefix was already removed)
    r"^(Problem Statement|Problem|Task|Condition|Solution|Answer|Zadatak):?\s*\n+",
]

# Image reference detection (for statistics only)
IMAGE_REFERENCE_PATTERNS = [
    r"!\[.*?\]\(https?://[^\)]+\)",  # Markdown images
    r"\[asy\]",  # Asymptote diagrams (just presence)
    r"\[[\w\-]+\.(gif|png|jpg|jpeg)\]",  # [file.gif]
    r"(?:see|refer to|shown in)\s+(?:Figure|Diagram|figure|diagram)\s+\d+",
]

# Special artifacts (horizontal rules, translation instructions, etc.)
SPECIAL_ARTIFACT_PATTERNS = [
    # Horizontal rules (must have newlines around them)
    r"\n\s*[-=_*]{3,}\s*\n",  # "---", "===", "___", "***" as separators
    # Translation instruction artifacts
    r"[Tt]ranslate\s+the\s+above\s+text\s+into\s+English[^\n]*directly\.?",  # "Translate the above text into English..."
    r"[Pp]lease retain the original text'?s? line breaks and format,?\s*and output the translation result directly\.?",
    r"[Tt]ranslation:?\s*$",  # "Translation:" at end of line
    # Answer/response markers at the end (can interfere with problems)
    r"\n\s*(?:Answer|Solution):\s*$",  # Trailing "Answer:" or "Solution:"
]

# Trailing artifacts (at the END of text)
TRAILING_ARTIFACT_PATTERNS = [
    # Markdown headers at end (e.g., "## second grade", "## Level 3")
    r"\n+#{1,6}\s+[^\n]+$",
    # Bold labels at end (e.g., "**Level 3**", "**Easy**")
    r"\n+\*\*[^\*]+\*\*\s*$",
    # Category/topic labels in various formats
    r"\n+\[\s*[A-Za-z][^\]]{0,50}\s*\]\s*$",  # "[ Geometry ]", "[ Algebra ]" at end
]


# Preset configurations for each dataset
CLEANING_PRESETS = {
    "orz-math": {
        "remove_problem_numbers": True,  # 10-15%
        "remove_point_allocations": True,  # 2-3%
        "remove_contest_metadata": True,  # 5-8%
        "remove_markdown_headers": True,  # 7-8% (added based on analysis)
        "remove_author_attributions": True,  # NEW: author names in LaTeX underline
        "remove_special_artifacts": True,  # NEW: horizontal rules, translation artifacts
        "remove_trailing_artifacts": True,  # NEW: trailing markdown headers, category labels
        "filter_url_samples": True,  # NEW: filter out samples with URLs (0.5%)
        "filter_multipart_samples": True,  # NEW: filter out multi-part problems (1.3%)
        "detect_image_references": True,  # 1-2%
        "normalize_whitespace": True,
    },
    "openr1-math": {
        "remove_problem_numbers": True,  # 11%
        "remove_point_allocations": True,  # 4%
        "remove_contest_metadata": True,  # 5%
        "remove_markdown_headers": True,  # 5%
        "remove_author_attributions": True,  # Author names in LaTeX underline
        "remove_special_artifacts": True,  # Horizontal rules, translation artifacts
        "remove_trailing_artifacts": True,  # Trailing markdown headers, category labels
        "filter_url_samples": True,  # Filter out samples with URLs (0.5%)
        "filter_multipart_samples": True,  # Filter out multi-part problems (1.3%)
        "detect_image_references": True,  # 5%
        "normalize_whitespace": True,
    },
    "skywork-or1": {
        "remove_problem_numbers": True,  # 2%
        "remove_point_allocations": False,
        "remove_contest_metadata": False,
        "remove_markdown_headers": False,
        "detect_image_references": True,  # 3%
        "normalize_whitespace": True,
    },
    "dapo-math": {
        "remove_problem_numbers": True,  # <1%
        "remove_point_allocations": False,
        "remove_contest_metadata": False,
        "remove_markdown_headers": False,
        "detect_image_references": False,
        "normalize_whitespace": True,
    },
}


class MathDatasetCleaner(PipelineStep):
    """Clean math problem datasets in VERL format.

    Removes various artifacts from math problems including:
    - Problem numbering prefixes (e.g., "Problem 6.", "8.3 ", "Question 230,", "(1)", "1. ", "147 Let")
    - Contest metadata (e.g., "2004 AIME Problem 3", "20th APMC 1997")
    - Point allocations (e.g., "(8 points)", "[15 points]")
    - Markdown headers (e.g., "## Problem Statement", "## Task")
    - Special artifacts (e.g., horizontal rules "---", translation instructions)
    - Trailing artifacts (e.g., "## second grade" at end, "**Level 3**" at end)

    Preserves:
    - LaTeX formatting (NO escaping changes)
    - Ground truth values (unchanged)
    - extra_info metadata (unchanged)
    - Vague ground truths (single letters, etc.)
    - Non-English text
    - Answer content

    This formatter is designed for VERL format datasets where the problem text
    is stored in doc.metadata["prompt"][0]["content"].

    Args:
        remove_problem_numbers: Remove problem numbering prefixes (default: True)
        remove_point_allocations: Remove point allocation text (default: True)
        remove_contest_metadata: Remove contest source metadata (default: True)
        remove_markdown_headers: Remove markdown headers at start (default: True)
        remove_trailing_artifacts: Remove artifacts at end of text (default: False)
        remove_special_artifacts: Remove horizontal rules, translation artifacts (default: False)
        detect_image_references: Detect and count image references (default: True)
        normalize_whitespace: Normalize excessive whitespace (default: True)
        log_cleaning_stats: Log cleaning statistics (default: True)

    Example:
        >>> from datatrove.pipeline.readers import HuggingFaceReader
        >>> from datatrove.pipeline.formatters import MathDatasetCleaner
        >>> from datatrove.pipeline.writers import ParquetWriter
        >>>
        >>> # Use preset configuration
        >>> reader = HuggingFaceReader("sungyub/orz-math-72k-verl")
        >>> cleaner = MathDatasetCleaner.from_preset("orz-math")
        >>> writer = ParquetWriter("output/orz-math-cleaned")
        >>>
        >>> pipeline = reader | cleaner | writer

        >>> # Or use custom configuration
        >>> cleaner = MathDatasetCleaner(
        ...     remove_problem_numbers=True,
        ...     remove_contest_metadata=True,
        ...     remove_point_allocations=False,
        ... )
    """

    name = "🧹 MathCleaner"
    type = "✂️ - FORMAT"

    def __init__(
        self,
        remove_problem_numbers: bool = True,
        remove_point_allocations: bool = True,
        remove_contest_metadata: bool = True,
        remove_markdown_headers: bool = True,
        remove_author_attributions: bool = False,
        remove_trailing_artifacts: bool = False,
        remove_special_artifacts: bool = False,
        filter_url_samples: bool = False,
        filter_multipart_samples: bool = False,
        detect_image_references: bool = True,
        normalize_whitespace: bool = True,
        log_cleaning_stats: bool = True,
    ):
        """Initialize the MathDatasetCleaner.

        Args:
            remove_problem_numbers: Remove problem numbering prefixes
            remove_point_allocations: Remove point allocation text
            remove_contest_metadata: Remove contest source metadata
            remove_markdown_headers: Remove markdown headers at start
            remove_author_attributions: Remove author attribution in LaTeX underline format
            remove_trailing_artifacts: Remove artifacts at end of text (markdown headers, labels)
            remove_special_artifacts: Remove horizontal rules, translation artifacts
            filter_url_samples: Filter out samples containing URLs (delete entire sample)
            filter_multipart_samples: Filter out multi-part problems with a), b), c) (delete entire sample)
            detect_image_references: Detect and count image references
            normalize_whitespace: Normalize excessive whitespace
            log_cleaning_stats: Log cleaning statistics
        """
        super().__init__()
        self.remove_problem_numbers = remove_problem_numbers
        self.remove_point_allocations = remove_point_allocations
        self.remove_contest_metadata = remove_contest_metadata
        self.remove_markdown_headers = remove_markdown_headers
        self.remove_author_attributions = remove_author_attributions
        self.remove_trailing_artifacts = remove_trailing_artifacts
        self.remove_special_artifacts = remove_special_artifacts
        self.filter_url_samples = filter_url_samples
        self.filter_multipart_samples = filter_multipart_samples
        self.detect_image_references = detect_image_references
        self.normalize_whitespace = normalize_whitespace
        self.log_cleaning_stats = log_cleaning_stats

        # Compile regex patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns for efficient matching."""
        if self.remove_problem_numbers:
            self.problem_number_regex = re.compile(
                "|".join(f"({p})" for p in PROBLEM_NUMBER_PATTERNS), re.MULTILINE
            )
        if self.remove_contest_metadata:
            self.contest_metadata_regex = re.compile(
                "|".join(f"({p})" for p in CONTEST_METADATA_PATTERNS), re.MULTILINE
            )
        if self.remove_point_allocations:
            self.point_allocation_regex = re.compile(
                "|".join(f"({p})" for p in POINT_ALLOCATION_PATTERNS), re.MULTILINE
            )
        if self.remove_markdown_headers:
            # Don't use MULTILINE - we only want to match at the absolute START of text
            self.markdown_header_regex = re.compile(
                "|".join(f"({p})" for p in MARKDOWN_HEADER_PATTERNS)
            )
        if self.remove_author_attributions:
            self.author_attribution_regex = re.compile(
                "|".join(f"({p})" for p in AUTHOR_ATTRIBUTION_PATTERNS), re.MULTILINE
            )
        if self.remove_special_artifacts:
            self.special_artifact_regex = re.compile(
                "|".join(f"({p})" for p in SPECIAL_ARTIFACT_PATTERNS), re.MULTILINE
            )
        if self.remove_trailing_artifacts:
            self.trailing_artifact_regex = re.compile(
                "|".join(f"({p})" for p in TRAILING_ARTIFACT_PATTERNS), re.MULTILINE
            )
        if self.detect_image_references:
            self.image_reference_regex = re.compile(
                "|".join(f"({p})" for p in IMAGE_REFERENCE_PATTERNS), re.DOTALL | re.MULTILINE
            )

        # Compile filtering patterns
        if self.filter_url_samples:
            # Match http, https, www., or artofproblemsolving
            self.url_filter_regex = re.compile(
                r"https?://|www\.|artofproblemsolving\.com", re.IGNORECASE
            )
        if self.filter_multipart_samples:
            # Match multipart patterns at line start with uppercase sentence:
            # - Lowercase: a), b), c) (closing paren only)
            # - Roman numerals: (I), (II), (III), (i), (ii), (iii) (both parens)
            # - Unicode Roman numerals: (Ⅰ), (Ⅱ), (Ⅲ), (ⅰ), (ⅱ), (ⅲ) (U+2160-217F)
            # - Markdown bullets: "- (I)", "* (I)", "+ (I)"
            # Requires newline + optional markdown bullet + pattern + space + uppercase letter
            # This avoids false positives from math formulas like ab), f(x), gcd(a,b)
            self.multipart_filter_regex = re.compile(
                r"(?:^|\n)\s*(?:[-*+]\s+)?(?:[a-z]\)|\([IVXivx\u2160-\u217F]+\))\s+[A-Z]", re.MULTILINE
            )
            # Additional pattern: Lowercase Roman numerals without uppercase requirement
            # Detects multi-part problems like "(i) $formula$; (ii) $formula$"
            # Only filters if BOTH (i) and (ii) are present to avoid false positives
            self.lowercase_roman_i = re.compile(r'\(i\)\s')
            self.lowercase_roman_ii = re.compile(r'\(ii\)')

            # LaTeX/Dollar-wrapped Roman numerals: $(I)$, $(II)$, etc.
            # Detects multi-part problems like "$(I)$ Find...; $(II)$ Calculate..."
            # Only filters if BOTH first and second markers are present
            self.dollar_roman_i = re.compile(r'\$\([IVX]+\)\$')
            self.dollar_roman_ii = re.compile(r'\$\(II\)\$|\$\(ii\)\$')

            # LaTeX/Dollar-wrapped Arabic numerals: $(1)$, $(2)$, etc.
            # Detects multi-part problems like "$(1)$ Find...; $(2)$ Calculate..."
            # Only filters if BOTH (1) and (2) are present
            self.dollar_arabic_1 = re.compile(r'\$\(1\)\$')
            self.dollar_arabic_2 = re.compile(r'\$\(2\)\$')

    @classmethod
    def from_preset(cls, preset_name: str) -> "MathDatasetCleaner":
        """Create a MathDatasetCleaner with preset configuration.

        Args:
            preset_name: Name of the preset ("orz-math", "openr1-math",
                         "skywork-or1", "dapo-math")

        Returns:
            MathDatasetCleaner instance with preset configuration

        Raises:
            ValueError: If preset_name is not recognized

        Example:
            >>> cleaner = MathDatasetCleaner.from_preset("orz-math")
        """
        if preset_name not in CLEANING_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available presets: {', '.join(CLEANING_PRESETS.keys())}"
            )

        config = CLEANING_PRESETS[preset_name]
        return cls(**config)

    def _check_multiline_occurrence(self, text: str, pattern1: re.Pattern, pattern2: re.Pattern) -> bool:
        """Check if two patterns occur in the text with at least one newline between them.

        This reduces false positives by only filtering when patterns appear across
        multiple lines (indicating true multi-part problems) rather than on the same
        line (often just references like "equations (1) and (2)").

        Args:
            text: The text to search
            pattern1: First regex pattern to find
            pattern2: Second regex pattern to find

        Returns:
            True if both patterns found AND separated by newline, False otherwise
        """
        match1 = pattern1.search(text)
        match2 = pattern2.search(text)

        if not match1 or not match2:
            return False

        # Get positions of both matches
        pos1 = match1.start()
        pos2 = match2.start()

        # Ensure pos1 comes before pos2
        start_pos = min(pos1, pos2)
        end_pos = max(pos1, pos2)

        # Check if there's a newline between them
        between_text = text[start_pos:end_pos]
        return '\n' in between_text

    def clean_text(self, text: str) -> tuple[str, dict]:
        """Clean a single text string.

        Args:
            text: The text to clean

        Returns:
            Tuple of (cleaned_text, stats_dict) where stats_dict contains
            counts of each type of cleaning operation applied
        """
        if not text:
            return text, {}

        original_text = text
        stats = {
            "problem_number_removed": False,
            "contest_metadata_removed": False,
            "point_allocation_removed": False,
            "markdown_header_removed": False,
            "author_attribution_removed": False,
            "special_artifact_removed": False,
            "trailing_artifact_removed": False,
            "image_reference_detected": False,
        }

        # First, strip leading/trailing whitespace so patterns can match from start
        text = text.strip()

        # Remove markdown headers (must be at start)
        if self.remove_markdown_headers and hasattr(self, "markdown_header_regex"):
            new_text = self.markdown_header_regex.sub("", text)
            if new_text != text:
                stats["markdown_header_removed"] = True
                text = new_text

        # Remove author attributions (must be at start)
        if self.remove_author_attributions and hasattr(self, "author_attribution_regex"):
            new_text = self.author_attribution_regex.sub("", text)
            if new_text != text:
                stats["author_attribution_removed"] = True
                text = new_text

        # Remove contest metadata
        if self.remove_contest_metadata and hasattr(self, "contest_metadata_regex"):
            new_text = self.contest_metadata_regex.sub("", text)
            if new_text != text:
                stats["contest_metadata_removed"] = True
                text = new_text

        # Remove problem numbers
        if self.remove_problem_numbers and hasattr(self, "problem_number_regex"):
            new_text = self.problem_number_regex.sub("", text)
            if new_text != text:
                stats["problem_number_removed"] = True
                text = new_text

        # Remove point allocations
        if self.remove_point_allocations and hasattr(self, "point_allocation_regex"):
            new_text = self.point_allocation_regex.sub(" ", text)  # Replace with single space
            if new_text != text:
                stats["point_allocation_removed"] = True
                text = new_text

        # Remove special artifacts (horizontal rules, translation instructions)
        if self.remove_special_artifacts and hasattr(self, "special_artifact_regex"):
            new_text = self.special_artifact_regex.sub(" ", text)  # Replace with single space
            if new_text != text:
                stats["special_artifact_removed"] = True
                text = new_text

        # Remove trailing artifacts (at end of text) - AFTER other cleanings
        if self.remove_trailing_artifacts and hasattr(self, "trailing_artifact_regex"):
            new_text = self.trailing_artifact_regex.sub("", text)
            if new_text != text:
                stats["trailing_artifact_removed"] = True
                text = new_text

        # Detect image references (but don't remove)
        if self.detect_image_references and hasattr(self, "image_reference_regex"):
            if self.image_reference_regex.search(text):
                stats["image_reference_detected"] = True

        # Normalize whitespace (final cleanup)
        if self.normalize_whitespace:
            # Normalize internal whitespace (collapse multiple spaces/newlines)
            # But preserve intentional formatting (e.g., between paragraphs)
            text = re.sub(r"\n\n\n+", "\n\n", text)  # Max 2 consecutive newlines
            text = re.sub(r"[ \t]+", " ", text)  # Collapse spaces/tabs to single space
            text = text.strip()  # Final strip

        return text, stats

    def run(self, data: "DocumentsPipeline", rank: int = 0, world_size: int = 1) -> "DocumentsPipeline":
        """Clean math problems in VERL format documents.

        Args:
            data: Generator of Documents containing VERL format data
            rank: Rank of this task
            world_size: Total number of tasks

        Yields:
            Documents with cleaned problem text
        """
        for doc in data:
            self.stat_update(StatHints.total)

            with self.track_time():
                try:
                    # VERL format: problem text is in prompt[0]["content"]
                    if not doc.metadata or "prompt" not in doc.metadata:
                        logger.warning(f"Document {doc.id} missing 'prompt' in metadata, skipping")
                        self.stat_update("skipped_no_prompt")
                        yield doc
                        continue

                    prompt = doc.metadata["prompt"]
                    if not prompt or not isinstance(prompt, list) or len(prompt) == 0:
                        logger.warning(f"Document {doc.id} has invalid prompt format, skipping")
                        self.stat_update("skipped_invalid_prompt")
                        yield doc
                        continue

                    # Get the user message (first item in prompt)
                    user_message = prompt[0]
                    if not isinstance(user_message, dict) or "content" not in user_message:
                        logger.warning(f"Document {doc.id} has invalid user message format, skipping")
                        self.stat_update("skipped_invalid_message")
                        yield doc
                        continue

                    original_content = user_message["content"]
                    if not original_content:
                        self.stat_update("skipped_empty_content")
                        yield doc
                        continue

                    # Filter out samples with URLs if enabled
                    if self.filter_url_samples and hasattr(self, "url_filter_regex"):
                        if self.url_filter_regex.search(original_content):
                            self.stat_update("filtered_url_sample")
                            continue  # Skip this sample entirely (don't yield)

                    # Filter out multi-part problems if enabled
                    if self.filter_multipart_samples and hasattr(self, "multipart_filter_regex"):
                        if self.multipart_filter_regex.search(original_content):
                            self.stat_update("filtered_multipart_sample")
                            continue  # Skip this sample entirely (don't yield)

                    # Filter lowercase Roman numeral multi-part problems: (i) and (ii) together
                    if self.filter_multipart_samples and hasattr(self, "lowercase_roman_i"):
                        if self._check_multiline_occurrence(
                            original_content,
                            self.lowercase_roman_i,
                            self.lowercase_roman_ii
                        ):
                            self.stat_update("filtered_multipart_sample")
                            continue  # Skip this sample entirely (don't yield)

                    # Filter dollar-wrapped Roman numeral multi-part problems: $(I)$ and $(II)$ together
                    if self.filter_multipart_samples and hasattr(self, "dollar_roman_i"):
                        if self._check_multiline_occurrence(
                            original_content,
                            self.dollar_roman_i,
                            self.dollar_roman_ii
                        ):
                            self.stat_update("filtered_multipart_sample")
                            continue  # Skip this sample entirely (don't yield)

                    # Filter dollar-wrapped Arabic numeral multi-part problems: $(1)$ and $(2)$ together
                    if self.filter_multipart_samples and hasattr(self, "dollar_arabic_1"):
                        if self._check_multiline_occurrence(
                            original_content,
                            self.dollar_arabic_1,
                            self.dollar_arabic_2
                        ):
                            self.stat_update("filtered_multipart_sample")
                            continue  # Skip this sample entirely (don't yield)

                    # Clean the text
                    cleaned_content, cleaning_stats = self.clean_text(original_content)

                    # Update the document if text changed
                    if cleaned_content != original_content:
                        user_message["content"] = cleaned_content
                        self.stat_update("modified")

                        # Update individual cleaning stats
                        if self.log_cleaning_stats:
                            if cleaning_stats.get("problem_number_removed"):
                                self.stat_update("problem_number_removed")
                            if cleaning_stats.get("contest_metadata_removed"):
                                self.stat_update("contest_metadata_removed")
                            if cleaning_stats.get("point_allocation_removed"):
                                self.stat_update("point_allocation_removed")
                            if cleaning_stats.get("markdown_header_removed"):
                                self.stat_update("markdown_header_removed")
                            if cleaning_stats.get("author_attribution_removed"):
                                self.stat_update("author_attribution_removed")
                            if cleaning_stats.get("special_artifact_removed"):
                                self.stat_update("special_artifact_removed")
                            if cleaning_stats.get("trailing_artifact_removed"):
                                self.stat_update("trailing_artifact_removed")
                    else:
                        self.stat_update("unchanged")

                    # Count image references (independent of modification)
                    if self.log_cleaning_stats and cleaning_stats.get("image_reference_detected"):
                        self.stat_update("image_reference_detected")

                    yield doc

                except Exception as e:
                    logger.error(f"Failed to clean document {doc.id}: {e}")
                    self.stat_update("error")
                    yield doc  # Yield original document on error
