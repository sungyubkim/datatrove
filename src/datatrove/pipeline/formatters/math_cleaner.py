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
    r"^\d+\.\d+(?:[\.:]\s*|\s+)",  # "8.3:", "8.3.", "8.3 "
    r"^[A-Z]\d+\.\d+(?:[\.:]\s*|\s+)",  # "G1.4:", "I2.1:", "G1.4 "
    r"^[A-Z]\d+\s*\([A-Z]+\)\s*",  # "A2 (RUS)"
    r"^Example\s+\d+[\.:]\s*",  # "Example 31:"
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
]

# Point allocations
POINT_ALLOCATION_PATTERNS = [
    r"\s*\(\d+\s*points?\)\s*",  # " (8 points) ", "(1 point) "
    r"\s*\[\d+\s*points?\]\s*",  # " [15 points] "
    r"Reviews?\s+\([^)]*\d+\s*points?[^)]*\)\s*",  # "Reviews (from 7th grade. 1 point)"
]

# Markdown headers and header text (at start of text)
MARKDOWN_HEADER_PATTERNS = [
    # Remove all markdown headers (# to ######) with any content
    r"^#{1,6}\s+[^\n]+\n+",
    # Remove standalone header keywords (in case markdown prefix was already removed)
    r"^(Problem Statement|Condition|Task|Solution|Zadatak):?\s*\n+",
]

# Image reference detection (for statistics only)
IMAGE_REFERENCE_PATTERNS = [
    r"!\[.*?\]\(https?://[^\)]+\)",  # Markdown images
    r"\[asy\]",  # Asymptote diagrams (just presence)
    r"\[[\w\-]+\.(gif|png|jpg|jpeg)\]",  # [file.gif]
    r"(?:see|refer to|shown in)\s+(?:Figure|Diagram|figure|diagram)\s+\d+",
]


# Preset configurations for each dataset
CLEANING_PRESETS = {
    "orz-math": {
        "remove_problem_numbers": True,  # 10-15%
        "remove_point_allocations": True,  # 2-3%
        "remove_contest_metadata": True,  # 5-8%
        "remove_markdown_headers": True,  # 7-8% (added based on analysis)
        "detect_image_references": True,  # 1-2%
        "normalize_whitespace": True,
    },
    "openr1-math": {
        "remove_problem_numbers": True,  # 11%
        "remove_point_allocations": True,  # 4%
        "remove_contest_metadata": True,  # 5%
        "remove_markdown_headers": True,  # 5%
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
    - Problem numbering prefixes (e.g., "Problem 6.", "8.3 ", "Question 230,")
    - Contest metadata (e.g., "2004 AIME Problem 3", "20th APMC 1997")
    - Point allocations (e.g., "(8 points)", "[15 points]")
    - Markdown headers (e.g., "## Problem Statement")

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
        remove_markdown_headers: Remove markdown headers (default: True)
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
        detect_image_references: bool = True,
        normalize_whitespace: bool = True,
        log_cleaning_stats: bool = True,
    ):
        """Initialize the MathDatasetCleaner.

        Args:
            remove_problem_numbers: Remove problem numbering prefixes
            remove_point_allocations: Remove point allocation text
            remove_contest_metadata: Remove contest source metadata
            remove_markdown_headers: Remove markdown headers
            detect_image_references: Detect and count image references
            normalize_whitespace: Normalize excessive whitespace
            log_cleaning_stats: Log cleaning statistics
        """
        super().__init__()
        self.remove_problem_numbers = remove_problem_numbers
        self.remove_point_allocations = remove_point_allocations
        self.remove_contest_metadata = remove_contest_metadata
        self.remove_markdown_headers = remove_markdown_headers
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
            self.markdown_header_regex = re.compile(
                "|".join(f"({p})" for p in MARKDOWN_HEADER_PATTERNS), re.MULTILINE
            )
        if self.detect_image_references:
            self.image_reference_regex = re.compile(
                "|".join(f"({p})" for p in IMAGE_REFERENCE_PATTERNS), re.DOTALL | re.MULTILINE
            )

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
