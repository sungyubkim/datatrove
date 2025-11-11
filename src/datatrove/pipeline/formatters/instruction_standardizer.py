"""Instruction Format Standardizer for VERL code datasets.

This module provides a formatter that standardizes instruction formats across
different code generation datasets, converting them to a consistent CodeContests-style
format with explicit sections for problem, constraints, input/output formats, and examples.

Supports multiple domains:
- Python code generation (LeetCode, HumanEval, etc.)
- Verilog HDL (hardware design)
- Competitive programming (CodeContests, TACO, APPS, Codeforces)
"""

import re
from typing import TYPE_CHECKING, Dict, Tuple

from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger
from datatrove.utils.typeshelper import StatHints

if TYPE_CHECKING:
    from datatrove.data import Document, DocumentsPipeline


# Domain detection patterns
VERILOG_INDICATORS = [
    r"module\s+\w+",
    r"input\s+(?:wire\s+)?",
    r"output\s+(?:reg\s+)?",
    r"always\s+@",
    r"assign\s+",
    r"verilog",
    r"hdl",
    r"signal",
    r"port\s+width",
]

PYTHON_INDICATORS = [
    r"def\s+\w+\s*\(",
    r"class\s+\w+",
    r"return\s+",
    r"function\s+that",
    r"implement\s+",
    r"List\[",
    r"python",
]

COMPETITIVE_INDICATORS = [
    r"time\s+limit",
    r"memory\s+limit",
    r"codeforces",
    r"codecontests",
    r"competitive",
    r"stdin",
    r"stdout",
]


# Section extraction patterns
CONSTRAINT_PATTERNS = [
    r"(?:^|\n)\s*(?:Constraints?|Limitations?|Restrictions?):?\s*\n",
    r"(?:^|\n)\s*(?:\d+\s*[≤<=].*?[≤<=]\s*\d+)",  # Math constraints like "1 ≤ n ≤ 10^5"
    r"(?:^|\n)\s*[-*]\s*(?:\d+\s*[≤<=])",  # Bullet point constraints
]

INPUT_FORMAT_PATTERNS = [
    r"(?:^|\n)\s*(?:Input(?:\s+Format)?|Input Description):?\s*\n",
]

OUTPUT_FORMAT_PATTERNS = [
    r"(?:^|\n)\s*(?:Output(?:\s+Format)?|Output Description):?\s*\n",
]

EXAMPLE_PATTERNS = [
    r"(?:^|\n)\s*(?:Example|Sample|Test Case)s?:?\s*\n",
    r"(?:^|\n)\s*(?:Input|Output):?\s*\n",
]


# Preset configurations for different domains
STANDARDIZATION_PRESETS = {
    "python-code": {
        "domain": "python",
        "extract_constraints": True,
        "extract_io_format": True,
        "extract_examples": True,
        "add_function_signature": True,
        "normalize_formatting": True,
        "detect_domain_specific": True,
    },
    "verilog-hdl": {
        "domain": "verilog",
        "extract_constraints": True,
        "extract_io_format": True,
        "extract_examples": True,
        "preserve_signal_tables": True,
        "normalize_formatting": True,
        "detect_domain_specific": True,
    },
    "competitive": {
        "domain": "competitive",
        "extract_constraints": True,
        "extract_io_format": True,
        "extract_examples": True,
        "extract_time_complexity": True,
        "normalize_formatting": True,
        "detect_domain_specific": True,
    },
    "auto": {
        "domain": "auto",  # Auto-detect domain
        "extract_constraints": True,
        "extract_io_format": True,
        "extract_examples": True,
        "normalize_formatting": True,
        "detect_domain_specific": True,
    },
}


class InstructionStandardizer(PipelineStep):
    """Standardize instruction formats for code generation datasets in VERL format.

    Converts various instruction formats to a consistent structure (v1.2):

    ## Problem
    [Clear problem description]

    ## Constraints
    - [Constraint 1: novel info from problem narrative]
    - [Constraint 2: behavioral/memory requirements]
    (Note: Signal width/I/O info NOT duplicated - see Signal Interface)

    ## Example
    **Input:**
    ```
    [Example input]
    ```

    **Output:**
    ```
    [Example output]
    ```

    **Explanation:** [Example explanation]

    ## Implementation Requirements
    [Sandbox-optimized coding guidelines - domain-specific]

    ## [Domain-Specific Section]
    Python: Function Signature
    Verilog: Signal Interface Table (contains ALL I/O specifications)
    Competitive: Additional Test Cases

    Format Version History:
    - v1.0: Basic 6-section format
    - v1.1: Added Implementation Requirements (7-section)
    - v1.2: Removed redundant Input/Output Format sections for Verilog (5-section)

    Args:
        domain: Target domain ("python", "verilog", "competitive", "auto")
        extract_constraints: Extract and format constraints section
        extract_io_format: Extract input/output format sections
        extract_examples: Extract and format examples
        add_function_signature: Add function signature section (Python)
        preserve_signal_tables: Preserve signal tables (Verilog)
        extract_time_complexity: Extract time/space complexity (Competitive)
        normalize_formatting: Normalize whitespace and formatting
        detect_domain_specific: Auto-detect domain from content

    Example:
        >>> from datatrove.pipeline.readers import HuggingFaceReader
        >>> from datatrove.pipeline.formatters import InstructionStandardizer
        >>> from datatrove.pipeline.writers import ParquetWriter
        >>>
        >>> # Use preset configuration
        >>> reader = HuggingFaceReader("sungyub/kodcode-v1-verl")
        >>> standardizer = InstructionStandardizer.from_preset("python-code")
        >>> writer = ParquetWriter("output/kodcode-standardized")
        >>>
        >>> pipeline = reader | standardizer | writer

        >>> # Or use custom configuration
        >>> standardizer = InstructionStandardizer(
        ...     domain="python",
        ...     extract_constraints=True,
        ...     extract_examples=True,
        ... )
    """

    name = "📝 InstructionStandardizer"
    type = "✂️ - FORMAT"

    def __init__(
        self,
        domain: str = "auto",
        extract_constraints: bool = True,
        extract_io_format: bool = True,
        extract_examples: bool = True,
        add_function_signature: bool = False,
        preserve_signal_tables: bool = False,
        extract_time_complexity: bool = False,
        normalize_formatting: bool = True,
        detect_domain_specific: bool = True,
        log_stats: bool = True,
    ):
        """Initialize the InstructionStandardizer.

        Args:
            domain: Target domain ("python", "verilog", "competitive", "auto")
            extract_constraints: Extract and format constraints section
            extract_io_format: Extract input/output format sections
            extract_examples: Extract and format examples
            add_function_signature: Add function signature section (Python)
            preserve_signal_tables: Preserve signal tables (Verilog)
            extract_time_complexity: Extract time/space complexity (Competitive)
            normalize_formatting: Normalize whitespace and formatting
            detect_domain_specific: Auto-detect domain from content
            log_stats: Log processing statistics
        """
        super().__init__()
        self.domain = domain
        self.extract_constraints = extract_constraints
        self.extract_io_format = extract_io_format
        self.extract_examples = extract_examples
        self.add_function_signature = add_function_signature
        self.preserve_signal_tables = preserve_signal_tables
        self.extract_time_complexity = extract_time_complexity
        self.normalize_formatting = normalize_formatting
        self.detect_domain_specific = detect_domain_specific
        self.log_stats = log_stats

        # Compile regex patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns for efficient matching."""
        # Domain detection patterns
        self.verilog_regex = re.compile(
            "|".join(f"({p})" for p in VERILOG_INDICATORS), re.IGNORECASE | re.MULTILINE
        )
        self.python_regex = re.compile(
            "|".join(f"({p})" for p in PYTHON_INDICATORS), re.IGNORECASE | re.MULTILINE
        )
        self.competitive_regex = re.compile(
            "|".join(f"({p})" for p in COMPETITIVE_INDICATORS), re.IGNORECASE | re.MULTILINE
        )

        # Section extraction patterns
        if self.extract_constraints:
            self.constraint_regex = re.compile(
                "|".join(f"({p})" for p in CONSTRAINT_PATTERNS), re.MULTILINE
            )

        if self.extract_io_format:
            self.input_format_regex = re.compile(
                "|".join(f"({p})" for p in INPUT_FORMAT_PATTERNS), re.MULTILINE
            )
            self.output_format_regex = re.compile(
                "|".join(f"({p})" for p in OUTPUT_FORMAT_PATTERNS), re.MULTILINE
            )

        if self.extract_examples:
            self.example_regex = re.compile(
                "|".join(f"({p})" for p in EXAMPLE_PATTERNS), re.MULTILINE
            )

    @classmethod
    def from_preset(cls, preset_name: str) -> "InstructionStandardizer":
        """Create an InstructionStandardizer with preset configuration.

        Args:
            preset_name: Name of the preset ("python-code", "verilog-hdl",
                         "competitive", "auto")

        Returns:
            InstructionStandardizer instance with preset configuration

        Raises:
            ValueError: If preset_name is not recognized

        Example:
            >>> standardizer = InstructionStandardizer.from_preset("python-code")
        """
        if preset_name not in STANDARDIZATION_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available presets: {', '.join(STANDARDIZATION_PRESETS.keys())}"
            )

        config = STANDARDIZATION_PRESETS[preset_name]
        return cls(**config)

    def detect_domain(self, text: str) -> str:
        """Detect the domain of the instruction text.

        Args:
            text: The instruction text to analyze

        Returns:
            Detected domain: "verilog", "python", "competitive", or "unknown"
        """
        if not self.detect_domain_specific:
            return self.domain if self.domain != "auto" else "unknown"

        # Count matches for each domain
        verilog_matches = len(self.verilog_regex.findall(text))
        python_matches = len(self.python_regex.findall(text))
        competitive_matches = len(self.competitive_regex.findall(text))

        # Determine domain by highest match count
        if verilog_matches > max(python_matches, competitive_matches):
            return "verilog"
        elif python_matches > competitive_matches:
            return "python"
        elif competitive_matches > 0:
            return "competitive"
        else:
            return self.domain if self.domain != "auto" else "unknown"

    def parse_sections(self, text: str, domain: str) -> Dict[str, str]:
        """Parse the instruction text into sections using domain-specific parsers.

        Args:
            text: The instruction text to parse
            domain: The detected or configured domain

        Returns:
            Dictionary with section names as keys and content as values
        """
        from datatrove.pipeline.formatters.instruction_parsers import get_parser_for_domain

        # Get domain-specific parser
        parser = get_parser_for_domain(domain)

        # Use parser to extract sections
        sections = parser.parse(text)

        return sections

    def generate_standard_format(self, sections: Dict[str, str], domain: str) -> str:
        """Generate standardized instruction format from parsed sections.

        Args:
            sections: Dictionary of parsed sections
            domain: The domain of the instruction

        Returns:
            Standardized instruction text
        """
        output_parts = []

        # Problem section
        if sections["problem"]:
            output_parts.append("## Problem\n")
            output_parts.append(sections["problem"])
            output_parts.append("\n\n")

        # Constraints section
        if sections["constraints"] or self.extract_constraints:
            output_parts.append("## Constraints\n")
            if sections["constraints"]:
                # Format constraints as bullet points if not already
                constraints_text = sections["constraints"]
                if not re.match(r"^\s*[-*]", constraints_text):
                    # Add bullet points to each line
                    lines = [line.strip() for line in constraints_text.split("\n") if line.strip()]
                    constraints_text = "\n".join(f"- {line}" for line in lines)
                output_parts.append(constraints_text)
            else:
                output_parts.append("- Not specified")
            output_parts.append("\n\n")

        # Example section
        if sections["examples"]:
            output_parts.append("## Example\n")
            output_parts.append(sections["examples"])
            output_parts.append("\n\n")

        # Implementation Requirements section (sandbox optimization)
        from datatrove.pipeline.formatters.instruction_parsers import get_parser_for_domain
        parser = get_parser_for_domain(domain)
        implementation_hints = parser.generate_implementation_hints()
        if implementation_hints:
            output_parts.append("## Implementation Requirements\n")
            output_parts.append(implementation_hints)
            output_parts.append("\n\n")

        # Domain-specific section
        if sections["domain_specific"]:
            if domain == "python":
                output_parts.append("## Function Signature\n")
            elif domain == "verilog":
                output_parts.append("## Signal Interface\n")
            elif domain == "competitive":
                output_parts.append("## Additional Notes\n")
            output_parts.append(sections["domain_specific"])
            output_parts.append("\n")

        result = "".join(output_parts).strip()

        # Normalize formatting if enabled
        if self.normalize_formatting:
            # Collapse excessive newlines (max 2 consecutive)
            result = re.sub(r"\n\n\n+", "\n\n", result)
            # Ensure single space after periods
            result = re.sub(r"\.  +", ". ", result)

        return result

    def standardize_instruction(self, text: str) -> Tuple[str, Dict[str, any]]:
        """Standardize a single instruction text.

        Args:
            text: The instruction text to standardize

        Returns:
            Tuple of (standardized_text, stats_dict) where stats_dict contains
            information about the standardization process
        """
        if not text:
            return text, {}

        stats = {
            "detected_domain": "unknown",
            "sections_extracted": 0,
            "format_changed": False,
        }

        # Detect domain
        detected_domain = self.detect_domain(text)
        stats["detected_domain"] = detected_domain

        # Parse sections
        sections = self.parse_sections(text, detected_domain)
        stats["sections_extracted"] = sum(1 for v in sections.values() if v)

        # Generate standard format
        standardized_text = self.generate_standard_format(sections, detected_domain)
        stats["format_changed"] = standardized_text != text

        return standardized_text, stats

    def run(
        self, data: "DocumentsPipeline", rank: int = 0, world_size: int = 1
    ) -> "DocumentsPipeline":
        """Standardize instruction formats in VERL format documents.

        Args:
            data: Generator of Documents containing VERL format data
            rank: Rank of this task
            world_size: Total number of tasks

        Yields:
            Documents with standardized instruction text
        """
        for doc in data:
            self.stat_update(StatHints.total)

            with self.track_time():
                try:
                    # VERL format: instruction text is in prompt[0]["content"]
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

                    # Get the first user message (skip system messages)
                    user_message = None
                    for msg in prompt:
                        if isinstance(msg, dict) and msg.get('role') == 'user':
                            user_message = msg
                            break

                    # Fallback to first message if no user message found (backward compatibility)
                    if not user_message:
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

                    # Standardize the instruction
                    standardized_content, standardization_stats = self.standardize_instruction(
                        original_content
                    )

                    # Update the document if text changed
                    if standardized_content != original_content:
                        user_message["content"] = standardized_content

                        # Note: Format version tracking removed to preserve exact schema compatibility
                        # Format history: v1.0 → v1.1 (added Implementation Requirements) → v1.2 (removed redundant I/O sections)
                        # Clients can infer version from section structure if needed

                        self.stat_update("standardized")

                        # Update domain-specific stats
                        if self.log_stats:
                            detected_domain = standardization_stats.get("detected_domain", "unknown")
                            self.stat_update(f"domain_{detected_domain}")
                            if standardization_stats.get("format_changed"):
                                self.stat_update("format_changed")
                    else:
                        self.stat_update("unchanged")

                    yield doc

                except Exception as e:
                    logger.error(f"Failed to standardize document {doc.id}: {e}")
                    self.stat_update("error")
                    yield doc  # Yield original document on error
